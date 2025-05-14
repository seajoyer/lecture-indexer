import os
import logging
import uuid
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
import traceback

# Import system components
from youtube_extractor import YouTubeExtractor
from transcript_processor import TranscriptProcessor
from unified_concept_extractor import UnifiedConceptExtractor
from data_access import get_data_access
from concept_repository import get_concept_repository
from concept_candidate_extractor import get_concept_candidate_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Coordinates the end-to-end process of video data acquisition,
    transcript processing, and concept extraction with repository integration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get("output_dir", "data/processed")

        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self._init_components()

        logger.info("DataPipeline initialized with repository-based concept processing and candidate extraction")

    def _init_components(self):
        """Initialize pipeline components."""
        # Get YouTube API key from config
        youtube_api_key = self.config.get("youtube_api_key", "")
        if not youtube_api_key:
            logger.warning("No YouTube API key provided, using test mode")
            youtube_api_key = "test_api_key"

        # Initialize components
        self.youtube_extractor = YouTubeExtractor(youtube_api_key)
        self.transcript_processor = TranscriptProcessor()
        self.concept_extractor = UnifiedConceptExtractor()
        self.data_access = get_data_access()
        self.concept_repository = get_concept_repository()

        # Initialize the concept candidate extractor
        self.candidate_extractor = get_concept_candidate_extractor()

        logger.info("Pipeline components initialized")

    def process_video(self, video_url: str, language_preference: List[str] = ['en', 'ru']) -> Dict[str, Any]:
        """
        Process a YouTube video through the entire pipeline.

        Args:
            video_url: YouTube video URL
            language_preference: List of language codes in order of preference

        Returns:
            Dictionary with processing results
        """
        # Generate a unique job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Starting video processing job {job_id} for URL: {video_url}")

        try:
            # Step 1: Validate URL and extract video ID
            valid, video_id = self.youtube_extractor.validate_video_url(video_url)
            if not valid or not video_id:
                error_msg = f"Invalid YouTube URL: {video_url}"
                logger.error(error_msg)
                return {
                    "job_id": job_id,
                    "status": "error",
                    "error": error_msg,
                    "video_url": video_url
                }

            logger.info(f"Validated YouTube URL, video ID: {video_id}")

            # Step 2: Extract video metadata
            metadata = self.youtube_extractor.extract_video_metadata(video_id)
            logger.info(f"Extracted metadata for video: {video_id}")

            # Step 3: Extract transcript
            raw_transcript = self.youtube_extractor.extract_transcript(video_id, language_preference)
            logger.info(f"Extracted transcript with {len(raw_transcript)} segments")

            # Step 4: Process transcript
            processed_transcript = self.transcript_processor.process_transcript(raw_transcript, metadata)
            processed_transcript["video_id"] = video_id  # Ensure video_id is included
            logger.info(f"Processed transcript with {len(processed_transcript['segments'])} segments")

            # Record detected language
            detected_language = processed_transcript.get("language", "en")

            # Step 5: Extract concepts using repository-based matching
            concept_start_time = time.time()
            domain_features = self.concept_extractor.extract_concepts_from_transcript(processed_transcript)
            concept_time = time.time() - concept_start_time

            # Get concepts
            concepts = domain_features.get('concepts', [])

            # Log concept statistics
            total_concepts = len(concepts)
            educational_concepts = domain_features.get('educational_concepts_count', 0)
            passing_concepts = domain_features.get('passing_concepts_count', 0)

            logger.info(f"Extracted {total_concepts} concepts "
                       f"({educational_concepts} educational, {passing_concepts} passing) "
                       f"in {concept_time:.2f}s")

            # Step 6: Extract concept candidates
            candidate_start_time = time.time()
            candidates = self.candidate_extractor.extract_concept_candidates(
                processed_transcript,
                metadata
            )
            candidate_time = time.time() - candidate_start_time

            logger.info(f"Extracted {len(candidates)} concept candidates in {candidate_time:.2f}s")

            # Step 7: Save results to database
            database_start_time = time.time()

            # Save video metadata
            video_data = {
                "video_id": video_id,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "channel": metadata.get("channel", ""),
                "publication_date": metadata.get("publication_date", ""),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "language": detected_language,
                "indexed_at": datetime.now().isoformat(),
                "processing_status": "completed"
            }
            self.data_access.save_video(video_data)

            # Save segments
            self.data_access.save_segments(video_id, processed_transcript.get("segments", []))

            # Step 8: Make sure all repository concepts referenced by occurrences exist in the database
            # This ensures we meet foreign key constraints
            used_concept_ids = set()
            for concept in concepts:
                concept_id = concept.get("concept_id")
                if concept_id:
                    used_concept_ids.add(concept_id)

            # Save all referenced concepts to the database if they're not already there
            for concept_id in used_concept_ids:
                # Get concept from repository
                concept_data = self.concept_repository.get_concept(concept_id)
                if concept_data:
                    # Save to database
                    self.data_access.save_repository_concept(concept_data)
                    logger.debug(f"Saved concept {concept_id} to database to meet foreign key constraints")

            # Save occurrences
            all_occurrences = []
            for concept in concepts:
                occurrences = concept.get("occurrences", [])
                all_occurrences.extend(occurrences)

            if all_occurrences:
                # Log occurrence details for debugging
                logger.info(f"Attempting to save {len(all_occurrences)} concept occurrences")

                # Group occurrences by concept for clearer logging
                concepts_with_occurrences = {}
                for occ in all_occurrences:
                    concept_id = occ.get("concept_id", "unknown")
                    if concept_id not in concepts_with_occurrences:
                        concepts_with_occurrences[concept_id] = 0
                    concepts_with_occurrences[concept_id] += 1

                # Log summary of occurrences by concept
                for concept_id, count in concepts_with_occurrences.items():
                    logger.info(f"Concept {concept_id}: {count} occurrences")

                # Save the occurrences
                success = self.data_access.save_occurrences(all_occurrences)
                if success:
                    logger.info(f"Successfully saved concept occurrences")
                else:
                    logger.warning(f"Failed to save some concept occurrences")

                # Verify occurrences were actually saved
                for concept_id in concepts_with_occurrences.keys():
                    count_query = "SELECT COUNT(*) as count FROM occurrences WHERE concept_id = ?"
                    count_result = self.data_access.execute_query(count_query, (concept_id,))
                    if count_result and count_result[0]["count"] > 0:
                        logger.info(f"Verified: concept {concept_id} has {count_result[0]['count']} occurrences in database")
                    else:
                        logger.warning(f"Verification failed: concept {concept_id} has no occurrences in database")
            else:
                logger.warning(f"No concept occurrences to save for video {video_id}")

            database_time = time.time() - database_start_time
            logger.info(f"Saved results to database in {database_time:.2f}s")

            # Create a comprehensive result
            result = {
                "job_id": job_id,
                "status": "completed",
                "video_id": video_id,
                "video_url": video_url,
                "metadata": metadata,
                "language": detected_language,
                "transcript_segments": len(processed_transcript.get("segments", [])),
                "concepts": {
                    "total": total_concepts,
                    "educational": educational_concepts,
                    "passing": passing_concepts
                },
                "concept_candidates": {
                    "total": len(candidates),
                    "candidate_ids": [c.get("candidate_id") for c in candidates]
                },
                "processing_time": {
                    "concept_extraction_seconds": concept_time,
                    "candidate_extraction_seconds": candidate_time,
                    "database_seconds": database_time,
                    "total_seconds": concept_time + candidate_time + database_time
                },
                "timestamp": datetime.now().isoformat()
            }

            # Save result to file (for backward compatibility)
            self._save_result(result)

            logger.info(f"Successfully processed video {video_id}")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")

            # Log detailed traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Traceback: {error_traceback}")

            # Create error result
            error_result = {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "error_traceback": error_traceback,
                "video_url": video_url,
                "timestamp": datetime.now().isoformat()
            }

            # If we have a video_id, include it
            if 'video_id' in locals() and video_id:
                error_result["video_id"] = video_id

            # If we have metadata, include it
            if 'metadata' in locals() and metadata:
                error_result["metadata"] = metadata

            # Save error result
            self._save_result(error_result)

            return error_result

    def _save_result(self, result: Dict[str, Any]):
        """
        Save processing result to file.

        Args:
            result: Processing result dictionary
        """
        # Create filename using video_id or job_id
        video_id = result.get("video_id", "unknown")
        job_id = result.get("job_id", "unknown")
        filename = f"{video_id}_{job_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved processing result to {filepath}")
        except Exception as e:
            logger.error(f"Error saving result to file: {e}")

    def batch_process_videos(self, video_urls: List[str], language_preference: List[str] = ['en', 'ru']) -> List[Dict[str, Any]]:
        """
        Process multiple YouTube videos.

        Args:
            video_urls: List of YouTube video URLs
            language_preference: List of language codes in order of preference

        Returns:
            List of processing result dictionaries
        """
        logger.info(f"Starting batch processing for {len(video_urls)} videos")

        results = []
        for i, url in enumerate(video_urls):
            try:
                # Process video
                logger.info(f"Processing video {i+1}/{len(video_urls)}: {url}")
                result = self.process_video(url, language_preference)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing video {url}: {e}")
                results.append({
                    "video_url": url,
                    "status": "error",
                    "error": str(e)
                })

        logger.info(f"Batch processing completed for {len(video_urls)} videos")
        return results

    def process_youtube_playlist(self, playlist_url: str, language_preference: List[str] = ['en', 'ru'], max_videos: int = 10) -> Dict[str, Any]:
        """
        Process an entire YouTube playlist.

        Args:
            playlist_url: YouTube playlist URL
            language_preference: List of language codes in order of preference
            max_videos: Maximum number of videos to process

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Starting playlist processing for {playlist_url}")

        try:
            # Validate and extract playlist ID
            valid, playlist_id = self.youtube_extractor.validate_playlist_url(playlist_url)
            if not valid or not playlist_id:
                error_msg = f"Invalid YouTube playlist URL: {playlist_url}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "playlist_url": playlist_url
                }

            # Get playlist metadata
            playlist_metadata = self.youtube_extractor.extract_playlist_metadata(playlist_id)

            # Get videos in playlist
            playlist_videos = self.youtube_extractor.extract_playlist_videos(playlist_id, max_videos)

            if not playlist_videos:
                return {
                    "status": "error",
                    "error": "No videos found in playlist",
                    "playlist_id": playlist_id,
                    "playlist_url": playlist_url
                }

            logger.info(f"Found {len(playlist_videos)} videos in playlist {playlist_id}")

            # Process each video
            video_results = []
            for i, video in enumerate(playlist_videos):
                video_id = video.get("video_id")
                if not video_id:
                    continue

                video_url = f"https://www.youtube.com/watch?v={video_id}"

                try:
                    logger.info(f"Processing playlist video {i+1}/{len(playlist_videos)}: {video_id}")
                    result = self.process_video(video_url, language_preference)
                    video_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing video {video_id}: {e}")
                    video_results.append({
                        "video_id": video_id,
                        "video_url": video_url,
                        "status": "error",
                        "error": str(e)
                    })

            # Calculate playlist statistics
            successful_videos = sum(1 for r in video_results if r.get("status") == "completed")
            total_concepts = sum(
                r.get("concepts", {}).get("total", 0)
                for r in video_results if r.get("status") == "completed"
            )
            educational_concepts = sum(
                r.get("concepts", {}).get("educational", 0)
                for r in video_results if r.get("status") == "completed"
            )
            total_candidates = sum(
                r.get("concept_candidates", {}).get("total", 0)
                for r in video_results if r.get("status") == "completed"
            )

            # Create playlist result
            playlist_result = {
                "status": "completed",
                "playlist_id": playlist_id,
                "playlist_url": playlist_url,
                "playlist_title": playlist_metadata.get("title", ""),
                "playlist_channel": playlist_metadata.get("channel", ""),
                "video_count": len(playlist_videos),
                "processed_count": successful_videos,
                "concepts": {
                    "total": total_concepts,
                    "educational": educational_concepts
                },
                "concept_candidates": {
                    "total": total_candidates
                },
                "videos": video_results,
                "timestamp": datetime.now().isoformat()
            }

            # Save result
            filename = f"playlist_{playlist_id}.json"
            filepath = os.path.join(self.output_dir, filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(playlist_result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error saving playlist result to file: {e}")

            logger.info(f"Playlist processing completed for {playlist_id} with {successful_videos} videos")
            return playlist_result

        except Exception as e:
            logger.error(f"Error processing playlist {playlist_url}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "playlist_url": playlist_url
            }
