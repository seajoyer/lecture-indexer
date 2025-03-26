"""
Simplified data pipeline for the Lecture Video Content Indexer.
Coordinates the end-to-end process of video extraction, transcript processing,
domain classification, and theory-practice analysis.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import simplified modules
from youtube_extractor import YouTubeExtractor
from transcript_processor import TranscriptProcessor
from performance_utils import time_function, Timer
from cache_manager import cache_get, cache_set

# Configure logging
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Coordinates the end-to-end process of video data acquisition and analysis.
    Simplified version with reduced complexity and dependencies.
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

        logger.info("DataPipeline initialized")

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

        # Generate a unique job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}"

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

            # Step 5: Calculate theory/practice ratio
            theory_practice_results = self._calculate_theory_practice_ratio(processed_transcript['segments'])
            logger.info(f"Calculated theory/practice ratio: {theory_practice_results['theory_practice_ratio']:.2f}")

            # Step 6: Extract key concepts
            domain_features = self._extract_domain_features(processed_transcript, metadata["domain"])
            logger.info(f"Extracted {len(domain_features['key_concepts'])} key concepts")

            # Prepare result
            processing_time = timer.stop() / 1000  # Convert from ms to seconds

            result = {
                "job_id": job_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "video_id": video_id,
                "video_url": video_url,
                "metadata": metadata,
                "transcript": processed_transcript,
                "domain_features": domain_features,
                "theory_practice_results": theory_practice_results,
                "processing_time": processing_time
            }

            # Cache the result
            cache_set("video", cache_key, result)

            # Save result to file (for backward compatibility)
            self._save_result(result)

            logger.info(f"Successfully processed video {video_id} in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")
            error_result = {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
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

    def _calculate_theory_practice_ratio(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Calculate theory/practice ratio from segments.

        Args:
            segments: Processed transcript segments

        Returns:
            Dictionary with theory/practice analysis
        """
        if not segments:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "theoretical_segments": 0,
                "practical_segments": 0,
                "mixed_segments": 0,
                "theory_practice_ratio": 0.5
            }

        # Count segment types
        theoretical = sum(1 for segment in segments if segment.get("content_type") == "theoretical")
        practical = sum(1 for segment in segments if segment.get("content_type") == "practical")
        mixed = sum(1 for segment in segments if segment.get("content_type") == "mixed")

        total = theoretical + practical + mixed

        # Calculate theory/practice ratio
        if total > 0:
            # Apply a weighted formula where mixed segments count as 0.5 towards both
            theory_weight = theoretical + (mixed * 0.5)
            practice_weight = practical + (mixed * 0.5)

            theory_practice_ratio = theory_weight / total
        else:
            theory_practice_ratio = 0.5

        # Determine overall classification
        if theory_practice_ratio > 0.7:
            classification = "theoretical"
            confidence = 0.8 if theoretical > practical * 3 else 0.6
        elif theory_practice_ratio < 0.3:
            classification = "practical"
            confidence = 0.8 if practical > theoretical * 3 else 0.6
        else:
            classification = "mixed"
            confidence = 0.7

        return {
            "classification": classification,
            "confidence": confidence,
            "theoretical_segments": theoretical,
            "practical_segments": practical,
            "mixed_segments": mixed,
            "theory_practice_ratio": theory_practice_ratio
        }

    def _extract_domain_features(self, processed_transcript: Dict, domain: str) -> Dict[str, Any]:
        """
        Extract domain-specific features from processed transcript.

        Args:
            processed_transcript: Processed transcript dictionary
            domain: Content domain

        Returns:
            Dictionary with domain-specific features
        """
        segments = processed_transcript.get("segments", [])

        # Extract key terms based on domain
        key_terms = self._extract_key_terms(segments, domain)

        # Convert key terms to concepts
        key_concepts = []
        for term, frequency in key_terms.items():
            # Determine if concept is theoretical or practical
            theoretical = self._is_theoretical_concept(term, segments)

            concept = {
                "text": term,
                "frequency": frequency,
                "domain": domain,
                "theoretical": theoretical,
                "concept_class": "theoretical" if theoretical else "practical"
            }

            key_concepts.append(concept)

        return {
            "domain": domain,
            "key_concepts": key_concepts
        }

    def _extract_key_terms(self, segments: List[Dict], domain: str) -> Dict[str, int]:
        """
        Extract key terms from segments based on domain.

        Args:
            segments: Processed transcript segments
            domain: Content domain

        Returns:
            Dictionary mapping terms to frequencies
        """
        # Combined text for analysis
        combined_text = " ".join([segment.get("text", "") for segment in segments]).lower()

        # Domain-specific key phrase patterns
        domain_patterns = {
            "mathematics": [
                r'\b(calculus|algebra|geometry|topology|analysis)\b',
                r'\b(theorem|proof|equation|function|derivative|integral|limit)\b',
                r'\b(differential equation|matrix|vector|scalar|tensor)\b',
                r'\b(convergence|divergence|series|sequence|set theory)\b'
            ],
            "programming": [
                r'\b(algorithm|data structure|function|class|method|object)\b',
                r'\b(variable|loop|recursion|iteration|conditional)\b',
                r'\b(array|list|stack|queue|tree|graph|hash table)\b',
                r'\b(object-oriented|functional|procedural|event-driven)\b'
            ],
            "physics": [
                r'\b(mechanics|dynamics|kinematics|statics|thermodynamics)\b',
                r'\b(electromagnetism|relativity|quantum|nuclear|particle)\b',
                r'\b(force|energy|momentum|mass|velocity|acceleration)\b',
                r'\b(wave|field|potential|charge|current|voltage)\b'
            ]
        }

        # Get patterns for the specified domain
        patterns = domain_patterns.get(domain, [])
        if not patterns:
            patterns = []
            for d_patterns in domain_patterns.values():
                patterns.extend(d_patterns)

        # Extract key terms using patterns
        import re

        terms = {}
        for pattern in patterns:
            matches = re.finditer(pattern, combined_text)
            for match in matches:
                term = match.group(0)
                terms[term] = terms.get(term, 0) + 1

        # Sort by frequency
        return dict(sorted(terms.items(), key=lambda x: x[1], reverse=True))

    def _is_theoretical_concept(self, term: str, segments: List[Dict]) -> bool:
        """
        Determine if a concept is theoretical based on its context.

        Args:
            term: Concept term
            segments: Processed transcript segments

        Returns:
            True if theoretical, False if practical
        """
        # Find segments that contain this term
        term_segments = []
        for segment in segments:
            if term.lower() in segment.get("text", "").lower():
                term_segments.append(segment)

        if not term_segments:
            return True  # Default to theoretical if no context found

        # Count content types
        theoretical_count = sum(1 for segment in term_segments
                               if segment.get("content_type") == "theoretical")
        practical_count = sum(1 for segment in term_segments
                             if segment.get("content_type") == "practical")

        # Determine if theoretical based on majority
        return theoretical_count >= practical_count

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
