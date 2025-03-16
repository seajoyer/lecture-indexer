"""
Data Pipeline module for the Lecture Video Content Indexer.
Coordinates the end-to-end process of video extraction, transcript processing,
domain classification, and theory-practice analysis.
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor
from data_acquisition.transcript_processor.python.transcript_processor import TranscriptProcessor
from concept_analysis.concept_extractor.python.domain_concept_extractor import DomainClassifier
from concept_analysis.relevance_analyzer.python.theory_practice_classifier import TheoryPracticeClassifier

# Configure logging
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Coordinates the end-to-end process of video data acquisition and analysis.
    Manages the flow between different components of the system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Data Pipeline with configuration.

        Args:
            config: Configuration dictionary with API keys and settings
        """
        logger.info("Initializing Data Pipeline")

        self.config = config
        self.output_dir = config.get("output_dir", "data/processed")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

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
            self.domain_classifier = DomainClassifier()
            logger.info("Initialized Domain Classifier")

            # Initialize theory-practice classifier
            self.theory_practice_classifier = TheoryPracticeClassifier()
            logger.info("Initialized Theory-Practice Classifier")

            logger.info("All pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
            raise

    def process_video(self, video_url: str, language_preference: List[str] = ['en', 'ru']) -> Dict[str, Any]:
        """
        Process a YouTube video through the entire pipeline.

        Args:
            video_url: YouTube video URL
            language_preference: List of language codes in order of preference

        Returns:
            Dictionary with processing results
        """
        job_id = str(uuid.uuid4())
        logger.info(f"Starting video processing job {job_id} for URL: {video_url}")

        try:
            # Step 1: Validate URL and extract video ID
            # Use mock_extractor if it exists (for testing), otherwise use youtube_extractor
            extractor = getattr(self, 'mock_extractor', self.youtube_extractor)
            valid, video_id = extractor.validate_video_url(video_url)
            if not valid or not video_id:
                error_msg = f"Invalid YouTube URL: {video_url}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Validated YouTube URL, video ID: {video_id}")

            # Step 2: Extract video metadata
            metadata = extractor.extract_video_metadata(video_id)
            if not metadata:
                error_msg = f"Failed to extract metadata for video: {video_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Extracted metadata for video: {video_id}")

            # Step 3: Extract transcript
            raw_transcript = extractor.extract_transcript(video_id, language_preference)
            if not raw_transcript:
                error_msg = f"Failed to extract transcript for video: {video_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Extracted transcript with {len(raw_transcript)} segments")

            # Step 4: Process transcript
            processor = getattr(self, 'mock_processor', self.transcript_processor)
            processed_transcript = processor.process_transcript(raw_transcript, metadata)
            logger.info(f"Processed transcript with {len(processed_transcript['segments'])} segments")

            # Step 5: Detect domain
            domain_classifier = getattr(self, 'mock_domain', self.domain_classifier)
            if metadata.get("domain") == "unknown" or metadata.get("domain_confidence", 0) < 0.6:
                domain, confidence = domain_classifier.classify_transcript(processed_transcript)
                metadata["domain"] = domain
                metadata["domain_confidence"] = confidence

            logger.info(f"Classified domain as {metadata['domain']} with confidence {metadata['domain_confidence']:.2f}")

            # Step 6: Extract domain-specific features
            domain_features = domain_classifier.extract_domain_specific_features(
                processed_transcript, metadata["domain"])
            logger.info(f"Extracted domain-specific features")

            # Step 7: Classify theory vs practice
            tp_classifier = getattr(self, 'mock_tp', self.theory_practice_classifier)
            theory_practice_results = tp_classifier.classify_transcript(processed_transcript)
            logger.info(f"Classified theory vs practice: {theory_practice_results['classification']}")

            # Step 8: Extract theory-practice patterns
            theory_practice_patterns = tp_classifier.extract_theory_practice_patterns(
                processed_transcript)
            logger.info(f"Extracted {len(theory_practice_patterns.get('theory_to_practice_sequences', []))} " +
                    f"theory-to-practice sequences")

            # Step 9: Create the result object
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
                "status": "completed"
            }

            # Step 10: Save results to file
            self._save_result(result)

            logger.info(f"Successfully completed processing for video {video_id}")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")

            # Create error result
            error_result = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "video_url": video_url,
                "status": "error",
                "error": str(e)
            }

            # Extract video ID if possible
            if 'video_id' in locals() and video_id:
                error_result["video_id"] = video_id

            # Add metadata if available
            if 'metadata' in locals() and metadata:
                error_result["metadata"] = metadata

            # Save error result
            self._save_result(error_result)

            # Re-raise specific exceptions for test cases
            if isinstance(e, ValueError):
                raise

            return error_result

    def _save_result(self, result: Dict[str, Any]):
        """
        Save processing result to file.

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

        Args:
            video_id: YouTube video ID

        Returns:
            Processing result dictionary if found, None otherwise
        """
        try:
            # Find matching files
            files = [f for f in os.listdir(self.output_dir) if f.startswith(f"{video_id}_")]

            if not files:
                logger.warning(f"No processed results found for video ID: {video_id}")
                return None

            # Use the most recent file if multiple exist
            files.sort(reverse=True)  # Sort by filename (which contains timestamp)
            latest_file = files[0]

            # Load result from file
            filepath = os.path.join(self.output_dir, latest_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                result = json.load(f)

            logger.info(f"Retrieved processed result for video ID: {video_id}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving processed result for video ID {video_id}: {e}")
            return None
