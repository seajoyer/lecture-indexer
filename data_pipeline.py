"""
Enhanced data pipeline for the Lecture Video Content Indexer.
Coordinates the end-to-end process of video extraction, transcript processing,
and concept extraction using a unified approach with video-level theory/practice ratio.
"""

import os
import logging
import uuid
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

# Import project modules
from youtube_extractor import YouTubeExtractor
from transcript_processor import TranscriptProcessor
from unified_concept_extractor import UnifiedConceptExtractor
from concept_dedup import apply_concept_deduplication
from performance_utils import time_function, Timer
from cache_manager import cache_get, cache_set

# Configure logging
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Coordinates the end-to-end process of lecture video data acquisition,
    transcript processing, and concept extraction with video-level theory/practice ratio.
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

        logger.info("DataPipeline initialized with unified concept processing approach")

    def _init_components(self):
        """Initialize pipeline components."""
        # Get YouTube API key from config
        youtube_api_key = self.config.get("youtube_api_key")
        if not youtube_api_key:
            logger.warning("No YouTube API key provided, using test mode")
            youtube_api_key = "test_api_key"

        # Initialize components
        self.youtube_extractor = YouTubeExtractor(youtube_api_key)
        self.transcript_processor = TranscriptProcessor()
        self.concept_extractor = UnifiedConceptExtractor()

        logger.info("Pipeline components initialized")

    @time_function(10000)  # Log warning if takes more than 10 seconds
    def process_video(self, video_url: str, language_preference: List[str] = ['en', 'ru']) -> Dict[str, Any]:
        """
        Process a YouTube video through the entire pipeline.

        Args:
            video_url: YouTube video URL
            language_preference: List of language codes in order of preference

        Returns:
            Dictionary with processing results
        """
        # Create a timer for overall process
        timer = Timer("process_video").start()

        # Generate a unique job ID based on timestamp and UUID
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

            # Check cache for previously processed result
            cache_key = f"processed_video_{video_id}"
            cached_result = cache_get("video", cache_key)
            if cached_result:
                logger.info(f"Using cached processing result for video {video_id}")
                return cached_result

            # Step 2: Extract video metadata
            metadata = self.youtube_extractor.extract_video_metadata(video_id)
            logger.info(f"Extracted metadata for video: {video_id}")

            # Step 3: Extract transcript
            raw_transcript = self.youtube_extractor.extract_transcript(video_id, language_preference)
            logger.info(f"Extracted transcript with {len(raw_transcript)} segments")

            # Step 4: Process transcript
            processed_transcript = self.transcript_processor.process_transcript(raw_transcript, metadata)
            logger.info(f"Processed transcript with {len(processed_transcript['segments'])} segments")

            # Step 5: Calculate theory/practice ratio from processed transcript
            theory_practice_results = processed_transcript.get("global_analysis", {})
            if not theory_practice_results:
                # If global analysis not found, calculate directly
                theory_practice_results = {
                    "theory_practice_ratio": 0.5,  # Default balanced ratio
                    "theoretical_indicators": 0,
                    "practical_indicators": 0
                }

            logger.info(f"Video-level theory/practice ratio: {theory_practice_results.get('theory_practice_ratio', 0.5):.2f}")

            # Record detected language
            detected_language = processed_transcript.get("language", "en")

            # Step 6: Extract concepts using unified concept extractor
            concept_start_time = time.time()
            domain_features = self.concept_extractor.extract_concepts_from_transcript(processed_transcript)
            concept_time = time.time() - concept_start_time

            # Get concepts from the unified list
            concepts = domain_features.get('concepts', [])

            # Log concept statistics
            total_concepts = len(concepts)
            educational_concepts = sum(1 for c in concepts if c.get('is_educational', False))

            logger.info(f"Extracted {total_concepts} concepts ({educational_concepts} educational) in {concept_time:.2f}s")

            # Prepare result
            result = {
                "job_id": job_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "video_id": video_id,
                "video_url": video_url,
                "metadata": metadata,
                "transcript": processed_transcript,
                "domain_features": domain_features,
                "theory_practice_results": theory_practice_results
            }

            # Step 7: Apply concept deduplication
            dedup_start_time = time.time()
            logger.info(f"Applying concept deduplication for language: {detected_language}")
            deduplicated_result = apply_concept_deduplication(result, detected_language)
            dedup_time = time.time() - dedup_start_time

            # Log deduplication statistics
            concepts_before = len(concepts)
            concepts_after = len(deduplicated_result['domain_features'].get('concepts', []))

            logger.info(f"Deduplication complete: {concepts_before} → {concepts_after} concepts in {dedup_time:.2f}s")

            # Calculate processing time
            processing_time = timer.stop() / 1000  # Convert from ms to seconds
            deduplicated_result["processing_time"] = processing_time

            # Add deduplication stats to result
            if "deduplication_stats" not in deduplicated_result:
                deduplicated_result["deduplication_stats"] = {
                    "original_total": concepts_before,
                    "deduplicated_total": concepts_after,
                    "reduction_percentage": round((concepts_before - concepts_after) /
                                            max(concepts_before, 1) * 100, 2),
                    "processing_time": dedup_time
                }

            # Cache the result
            cache_set("video", cache_key, deduplicated_result)

            # Save result to file (for backward compatibility)
            self._save_result(deduplicated_result)

            logger.info(f"Successfully processed video {video_id} in {processing_time:.2f} seconds")
            return deduplicated_result

        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")

            # Create more detailed error information
            import traceback
            error_traceback = traceback.format_exc()

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

    def process_playlist(self, playlist_url: str, language_preference: List[str] = ['en', 'ru'], max_videos: int = 10) -> Dict[str, Any]:
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
                len(r.get("domain_features", {}).get("concepts", []))
                for r in video_results if r.get("status") == "completed"
            )

            # Calculate average theory/practice ratio
            theory_practice_ratios = [
                r.get("theory_practice_results", {}).get("theory_practice_ratio", 0.5)
                for r in video_results if r.get("status") == "completed"
            ]
            avg_theory_practice_ratio = sum(theory_practice_ratios) / len(theory_practice_ratios) if theory_practice_ratios else 0.5

            # Create playlist result
            playlist_result = {
                "status": "completed",
                "playlist_id": playlist_id,
                "playlist_url": playlist_url,
                "playlist_title": playlist_metadata.get("title", ""),
                "playlist_channel": playlist_metadata.get("channel", ""),
                "video_count": len(playlist_videos),
                "processed_count": successful_videos,
                "total_concepts": total_concepts,
                "avg_theory_practice_ratio": avg_theory_practice_ratio,
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
