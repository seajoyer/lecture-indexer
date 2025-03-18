"""
Data Pipeline module for the Lecture Video Content Indexer.
Coordinates the end-to-end process of video extraction, transcript processing,
domain classification, and theory-practice analysis.
Integrated with database, caching, and performance monitoring.
"""

import os
import logging
import uuid
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor
from data_acquisition.transcript_processor.python.transcript_processor import TranscriptProcessor
from concept_analysis.concept_extractor.python.domain_concept_extractor import DomainClassifier
from concept_analysis.relevance_analyzer.python.theory_practice_classifier import TheoryPracticeClassifier
from database.db_init import get_db_context
from common.utils.performance_utils import measure_time, time_function, measure_memory
from common.utils.cache_manager import CacheRegion

# Configure logging
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Coordinates the end-to-end process of video data acquisition and analysis.
    Manages the flow between different components of the system.
    Integrated with database persistence and performance optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Data Pipeline with configuration.

        Args:
            config: Configuration dictionary with API keys and settings
        """
        with measure_time("data_pipeline_init"):
            logger.info("Initializing Data Pipeline")

            self.config = config
            self.output_dir = config.get("output_dir", "data/processed")

            # Create output directory if it doesn't exist (legacy support)
            os.makedirs(self.output_dir, exist_ok=True)

            # Get database context
            self.db_context = get_db_context()
            if not self.db_context:
                logger.warning("Database context not available, using file-based storage")
            else:
                logger.info("Using database for persistence")
                # Get cache regions
                self.cache = self.db_context.get_cache_region("data_pipeline")
                self.video_cache = self.db_context.get_cache_region("videos")
                self.concept_cache = self.db_context.get_cache_region("concepts")

            # Initialize components
            self._init_components()

    def _init_components(self):
        """Initialize all pipeline components."""
        try:
            # Initialize YouTube data extractor
            # Get the API key from the config and log it to verify
            youtube_api_key = self.config.get("youtube_api_key")

            # Log the API key (first few and last few chars for security)
            if youtube_api_key:
                key_prefix = youtube_api_key[:8] + "..." if len(youtube_api_key) > 8 else "[redacted]"
                logger.info(f"Using YouTube API key starting with: {key_prefix}...")
            else:
                logger.warning("No YouTube API key found in configuration, using test key")
                youtube_api_key = "test_api_key"

            self.youtube_extractor = YouTubeDataExtractor(youtube_api_key)
            logger.info("Initialized YouTube Data Extractor")

            # Initialize transcript processor
            self.transcript_processor = TranscriptProcessor()
            logger.info("Initialized Transcript Processor")

            # Initialize domain classifier
            self.domain_classifier = DomainClassifier(self.config)
            logger.info("Initialized Domain Classifier")

            # Initialize theory-practice classifier
            self.theory_practice_classifier = TheoryPracticeClassifier()
            logger.info("Initialized Theory-Practice Classifier")

            logger.info("All pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
            raise

    @time_function(threshold_ms=10000)  # 10 seconds threshold for video processing
    @measure_memory(threshold_mb=500)  # Alert if memory usage exceeds 500MB
    def process_video(self, video_url: str, language_preference: List[str] = ['en', 'ru']) -> Dict[str, Any]:
        """
        Process a YouTube video through the entire pipeline.
        Stores results in database if available.

        Args:
            video_url: YouTube video URL
            language_preference: List of language codes in order of preference

        Returns:
            Dictionary with processing results
        """
        job_id = str(uuid.uuid4())
        start_time = datetime.now()
        logger.info(f"Starting video processing job {job_id} for URL: {video_url}")

        # Check cache first if database is available
        if hasattr(self, 'cache'):
            cache_key = f"process_video_{video_url}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Using cached processing result for video URL: {video_url}")
                return cached_result

        try:
            # Step 1: Validate URL and extract video ID
            # Use mock_extractor if it exists (for testing), otherwise use youtube_extractor
            extractor = getattr(self, 'mock_extractor', self.youtube_extractor)
            with measure_time("validate_video_url"):
                valid, video_id = extractor.validate_video_url(video_url)
                if not valid or not video_id:
                    error_msg = f"Invalid YouTube URL: {video_url}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"Validated YouTube URL, video ID: {video_id}")

            # Step 2: Check if video already exists in database
            if self.db_context and hasattr(self.db_context, 'video_repository'):
                existing_video = self.db_context.video_repository.get_video(video_id)
                if existing_video and existing_video.get("processing_status") == "completed":
                    logger.info(f"Video {video_id} already processed, retrieving existing data")
                    return self._get_processed_result(video_id)

            # Step 3: Extract video metadata
            with measure_time("extract_video_metadata"):
                metadata = extractor.extract_video_metadata(video_id)
                if not metadata:
                    error_msg = f"Failed to extract metadata for video: {video_id}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"Extracted metadata for video: {video_id}")

            # Step 4: Extract transcript
            with measure_time("extract_transcript"):
                raw_transcript = extractor.extract_transcript(video_id, language_preference)
                if not raw_transcript:
                    error_msg = f"Failed to extract transcript for video: {video_id}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"Extracted transcript with {len(raw_transcript)} segments")

            # Step 5: Process transcript
            with measure_time("process_transcript"):
                processor = getattr(self, 'mock_processor', self.transcript_processor)
                processed_transcript = processor.process_transcript(raw_transcript, metadata)
                logger.info(f"Processed transcript with {len(processed_transcript['segments'])} segments")

            # Step 6: Detect domain
            with measure_time("detect_domain"):
                domain_classifier = getattr(self, 'mock_domain', self.domain_classifier)
                if metadata.get("domain") == "unknown" or metadata.get("domain_confidence", 0) < 0.6:
                    domain, confidence = domain_classifier.classify_transcript(processed_transcript)
                    metadata["domain"] = domain
                    metadata["domain_confidence"] = confidence

                logger.info(f"Classified domain as {metadata['domain']} with confidence {metadata['domain_confidence']:.2f}")

            # Step 7: Extract domain-specific features
            with measure_time("extract_domain_features"):
                domain_features = domain_classifier.extract_domain_specific_features(
                    processed_transcript, metadata["domain"])
                logger.info(f"Extracted domain-specific features")

            # Step 8: Classify theory vs practice
            with measure_time("classify_theory_practice"):
                tp_classifier = getattr(self, 'mock_tp', self.theory_practice_classifier)
                theory_practice_results = tp_classifier.classify_transcript(processed_transcript)
                logger.info(f"Classified theory vs practice: {theory_practice_results['classification']}")

            # Step 9: Extract theory-practice patterns
            with measure_time("extract_theory_practice_patterns"):
                theory_practice_patterns = tp_classifier.extract_theory_practice_patterns(
                    processed_transcript)
                logger.info(f"Extracted {len(theory_practice_patterns.get('theory_to_practice_sequences', []))} " +
                        f"theory-to-practice sequences")

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Step 10: Create the result object
            result = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "video_id": video_id,
                "video_url": video_url,
                "metadata": metadata,
                "transcript": processed_transcript,
                "domain_features": domain_features,
                "theory_practice_results": theory_practice_results,
                "theory_practice_patterns": theory_practice_patterns,
                "processing_time": processing_time,
                "status": "completed"
            }

            # Step 11: Save results to database if available
            if self.db_context and hasattr(self.db_context, 'video_repository'):
                self._save_to_database(result)
            else:
                # Legacy file-based storage
                self._save_result(result)

            # Cache the result if database is available
            if hasattr(self, 'cache'):
                self.cache.set(cache_key, result, ttl=3600*24)  # Cache for 24 hours

            logger.info(f"Successfully completed processing for video {video_id} in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")

            # Calculate processing time for error case
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Create error result
            error_result = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "video_url": video_url,
                "status": "error",
                "error": str(e),
                "processing_time": processing_time
            }

            # Extract video ID if possible
            if 'video_id' in locals() and video_id:
                error_result["video_id"] = video_id

            # Add metadata if available
            if 'metadata' in locals() and metadata:
                error_result["metadata"] = metadata

            # Save error result
            if self.db_context and hasattr(self.db_context, 'video_repository'):
                # Update video status in database
                if 'video_id' in locals() and video_id:
                    try:
                        self.db_context.video_repository.save_video({
                            "video_id": video_id,
                            "processing_status": "error",
                            "processing_errors": str(e),
                            "processing_time": processing_time
                        })
                    except Exception as db_error:
                        logger.error(f"Error updating video status in database: {db_error}")
            else:
                # Legacy file-based storage
                self._save_result(error_result)

            # Re-raise specific exceptions for test cases
            if isinstance(e, ValueError):
                raise

            return error_result

    def _save_to_database(self, result: Dict[str, Any]) -> bool:
        """
        Save processing result to database.

        Args:
            result: Processing result dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            video_id = result.get("video_id")
            metadata = result.get("metadata", {})
            transcript = result.get("transcript", {})
            theory_practice_results = result.get("theory_practice_results", {})
            processing_time = result.get("processing_time", 0)

            # Save video metadata
            video_data = {
                "video_id": video_id,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "channel": metadata.get("channel", ""),
                "publication_date": metadata.get("publication_date", ""),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "language": metadata.get("language", ""),
                "domain": metadata.get("domain", "unknown"),
                "domain_confidence": metadata.get("domain_confidence", 0.0),
                "theory_practice_ratio": theory_practice_results.get("theory_practice_ratio", 0.5),
                "theoretical_segments": theory_practice_results.get("theoretical_segments", 0),
                "practical_segments": theory_practice_results.get("practical_segments", 0),
                "processing_status": "completed",
                "processing_time": processing_time,
                "indexed_at": datetime.now().isoformat()
            }

            self.db_context.video_repository.save_video(video_data)
            logger.info(f"Saved video metadata to database for {video_id}")

            # Save transcript segments
            segments = transcript.get("segments", [])
            if segments:
                self.db_context.video_repository.save_segments(video_id, segments)
                logger.info(f"Saved {len(segments)} segments to database for {video_id}")

            # Save theory-practice patterns
            theory_practice_patterns = result.get("theory_practice_patterns", {})
            if theory_practice_patterns:
                self.db_context.video_repository.save_theory_practice_patterns(video_id, theory_practice_patterns)
                logger.info(f"Saved theory-practice patterns to database for {video_id}")

            # Extract and save concepts if available
            domain_features = result.get("domain_features", {})
            key_concepts = domain_features.get("key_concepts", [])

            if key_concepts and hasattr(self.db_context, 'concept_repository'):
                for concept in key_concepts:
                    # Add video_id to concept data
                    concept_data = concept.copy()
                    concept_data["video_id"] = video_id

                    # Save concept
                    self.db_context.concept_repository.save_concept(concept_data)

                logger.info(f"Saved {len(key_concepts)} concepts to database for {video_id}")

            # Index for search if search repository is available
            if hasattr(self.db_context, 'search_repository'):
                self.db_context.search_repository.index_video_metadata(video_data)
                self.db_context.search_repository.index_segments(video_id, segments)
                logger.info(f"Indexed video content for search for {video_id}")

            return True

        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            return False

    def _save_result(self, result: Dict[str, Any]):
        """
        Save processing result to file (legacy method).

        Args:
            result: Processing result dictionary
        """
        job_id = result.get("job_id")
        video_id = result.get("video_id", "unknown")

        # Create filename
        filename = f"{video_id}_{job_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        try:
            # Save to file
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved processing result to {filepath}")

        except Exception as e:
            logger.error(f"Error saving result to file: {e}")

    @time_function(threshold_ms=30000)  # 30 seconds threshold for batch processing
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
                # Process video, but catch ValueError to continue batch processing
                try:
                    result = self.process_video(url, language_preference)
                    results.append(result)
                except ValueError as e:
                    # Create error result and continue with next video
                    error_result = {
                        "job_id": f"error-{i}",  # Ensure job_id is always present
                        "timestamp": datetime.now().isoformat(),
                        "video_url": url,
                        "status": "error",
                        "error": str(e)
                    }

                    # Extract video_id if available from the error message
                    if "video_id" in str(e):
                        video_id = None
                        try:
                            extractor = getattr(self, 'mock_extractor', self.youtube_extractor)
                            valid, video_id = extractor.validate_video_url(url)
                            if valid and video_id:
                                error_result["video_id"] = video_id
                        except:
                            pass

                    results.append(error_result)
                    logger.error(f"Error in batch processing for URL {url}: {e}")
            except Exception as e:
                # Unexpected error - log and continue with other videos
                error_result = {
                    "job_id": f"error-{i}",  # Ensure job_id is always present
                    "video_url": url,
                    "status": "error",
                    "error": str(e)
                }
                results.append(error_result)
                logger.error(f"Error in batch processing for URL {url}: {e}")

        logger.info(f"Completed batch processing for {len(video_urls)} videos")
        return results

    def get_processed_result(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a previously processed result by video ID.
        Checks database first, then falls back to file system.

        Args:
            video_id: YouTube video ID

        Returns:
            Processing result dictionary if found, None otherwise
        """
        return self._get_processed_result(video_id)

    def _get_processed_result(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Internal method to retrieve a previously processed result by video ID.
        Checks database first, then falls back to file system.

        Args:
            video_id: YouTube video ID

        Returns:
            Processing result dictionary if found, None otherwise
        """
        # First check cache if available
        if hasattr(self, 'video_cache'):
            cache_key = f"video_result_{video_id}"
            cached_result = self.video_cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved processed result from cache for video ID: {video_id}")
                return cached_result

        # Then check database if available
        if self.db_context and hasattr(self.db_context, 'video_repository'):
            try:
                # Get video metadata
                video = self.db_context.video_repository.get_video(video_id)
                if not video:
                    logger.warning(f"No video found in database for video ID: {video_id}")
                    # Fall back to file-based storage
                    return self._get_processed_result_from_file(video_id)

                # Get video segments
                segments = self.db_context.video_repository.get_video_segments(video_id)

                # Get theory-practice patterns
                patterns = self.db_context.video_repository.get_video_theory_practice_patterns(video_id)

                # Get concepts if available
                concepts = []
                if hasattr(self.db_context, 'concept_repository'):
                    concepts = self.db_context.concept_repository.get_concepts_for_video(video_id)

                # Reconstruct the result object
                result = {
                    "video_id": video_id,
                    "status": video.get("processing_status", "unknown"),
                    "metadata": {
                        "title": video.get("title", ""),
                        "description": video.get("description", ""),
                        "channel": video.get("channel", ""),
                        "publication_date": video.get("publication_date", ""),
                        "duration_seconds": video.get("duration_seconds", 0),
                        "language": video.get("language", ""),
                        "domain": video.get("domain", "unknown"),
                        "domain_confidence": video.get("domain_confidence", 0.0)
                    },
                    "transcript": {
                        "segments": segments,
                        "language": video.get("language", "en"),
                        "domain": video.get("domain", "unknown"),
                        "video_id": video_id
                    },
                    "theory_practice_results": {
                        "classification": "theoretical" if video.get("theory_practice_ratio", 0.5) > 0.7 else
                                          "practical" if video.get("theory_practice_ratio", 0.5) < 0.3 else
                                          "mixed",
                        "confidence": 0.8,  # Default confidence
                        "theoretical_segments": video.get("theoretical_segments", 0),
                        "practical_segments": video.get("practical_segments", 0),
                        "mixed_segments": len(segments) - video.get("theoretical_segments", 0) - video.get("practical_segments", 0),
                        "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
                    },
                    "theory_practice_patterns": patterns or {},
                    "domain_features": {
                        "domain": video.get("domain", "unknown"),
                        "theoretical_segments": video.get("theoretical_segments", 0),
                        "practical_segments": video.get("practical_segments", 0),
                        "key_concepts": concepts
                    }
                }

                # Cache the result if caching is available
                if hasattr(self, 'video_cache'):
                    self.video_cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

                logger.info(f"Retrieved processed result from database for video ID: {video_id}")
                return result

            except Exception as e:
                logger.error(f"Error retrieving processed result from database for video ID {video_id}: {e}")
                # Fall back to file-based storage
                return self._get_processed_result_from_file(video_id)
        else:
            # Use file-based storage if database is not available
            return self._get_processed_result_from_file(video_id)

    def _get_processed_result_from_file(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Legacy method to retrieve a processed result from file.

        Args:
            video_id: YouTube video ID

        Returns:
            Processing result dictionary if found, None otherwise
        """
        try:
            # Find matching files
            import os
            files = [f for f in os.listdir(self.output_dir) if f.startswith(f"{video_id}_")]

            if not files:
                logger.warning(f"No processed results found for video ID: {video_id}")
                return None

            # Use the most recent file if multiple exist
            files.sort(reverse=True)  # Sort by filename (which contains timestamp)
            latest_file = files[0]

            # Load result from file
            import json
            filepath = os.path.join(self.output_dir, latest_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                result = json.load(f)

            logger.info(f"Retrieved processed result from file for video ID: {video_id}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving processed result from file for video ID {video_id}: {e}")
            return None
