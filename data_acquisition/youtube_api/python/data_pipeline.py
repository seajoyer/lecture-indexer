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
            youtube_api_key = self.config.get("youtube_api_key")
            if not youtube_api_key:
                raise ValueError("YouTube API key not provided in configuration")

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
            valid, video_id = self.youtube_extractor.validate_video_url(video_url)
            if not valid or not video_id:
                raise ValueError(f"Invalid YouTube URL: {video_url}")

            logger.info(f"Validated YouTube URL, video ID: {video_id}")

            # Step 2: Extract video metadata
            metadata = self.youtube_extractor.extract_video_metadata(video_id)
            if not metadata:
                raise ValueError(f"Failed to extract metadata for video: {video_id}")

            logger.info(f"Extracted metadata for video: {video_id}")

            # Step 3: Extract transcript
            raw_transcript = self.youtube_extractor.extract_transcript(video_id, language_preference)
            if not raw_transcript:
                raise ValueError(f"Failed to extract transcript for video: {video_id}")

            logger.info(f"Extracted transcript with {len(raw_transcript)} segments")

            # Step 4: Process transcript
            processed_transcript = self.transcript_processor.process_transcript(raw_transcript, metadata)
            logger.info(f"Processed transcript with {len(processed_transcript['segments'])} segments")

            # Step 5: Detect domain
            if metadata.get("domain") == "unknown" or metadata.get("domain_confidence", 0) < 0.6:
                domain, confidence = self.domain_classifier.classify_transcript(processed_transcript)
                metadata["domain"] = domain
                metadata["domain_confidence"] = confidence

            logger.info(f"Classified domain as {metadata['domain']} with confidence {metadata['domain_confidence']:.2f}")

            # Step 6: Extract domain-specific features
            domain_features = self.domain_classifier.extract_domain_specific_features(
                processed_transcript, metadata["domain"])
            logger.info(f"Extracted domain-specific features")

            # Step 7: Classify theory vs practice
            theory_practice_results = self.theory_practice_classifier.classify_transcript(processed_transcript)
            logger.info(f"Classified theory vs practice: {theory_practice_results['classification']}")

            # Step 8: Extract theory-practice patterns
            theory_practice_patterns = self.theory_practice_classifier.extract_theory_practice_patterns(
                processed_transcript)
            logger.info(f"Extracted {len(theory_practice_patterns['theory_to_practice_sequences'])} " +
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
        for url in video_urls:
            try:
                result = self.process_video(url, language_preference)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch processing for URL {url}: {e}")
                results.append({
                    "video_url": url,
                    "status": "error",
                    "error": str(e)
                })

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
