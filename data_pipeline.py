"""
Enhanced data pipeline for the Lecture Video Content Indexer.
Coordinates the end-to-end process of video extraction, transcript processing,
and concept extraction with a unified approach.
"""

import os
import logging
import uuid
import re
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import json

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
    Coordinates the end-to-end process of video data acquisition and analysis
    with a unified concept extraction approach.
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

        logger.info("DataPipeline initialized with unified concept extraction")

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

            # Step 5: Calculate theory/practice ratio
            theory_practice_results = self._calculate_theory_practice_ratio(processed_transcript['segments'])
            logger.info(f"Calculated theory/practice ratio: {theory_practice_results['theory_practice_ratio']:.2f}")

            # Step 6: Extract key concepts using unified extractor
            domain_features = self._extract_domain_features(processed_transcript, metadata["domain"])
            logger.info(f"Extracted {len(domain_features['key_concepts'])} key concepts")

            # Step 7: Apply concept deduplication to eliminate bad concepts and duplicates
            detected_language = processed_transcript.get("language", "en")

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

            # Apply the concept deduplication
            deduplicated_result = apply_concept_deduplication(result, detected_language)

            # Calculate processing time
            processing_time = timer.stop() / 1000  # Convert from ms to seconds
            deduplicated_result["processing_time"] = processing_time

            # Cache the result
            cache_set("video", cache_key, deduplicated_result)

            # Save result to file (for backward compatibility)
            self._save_result(deduplicated_result)

            logger.info(f"Successfully processed video {video_id} in {processing_time:.2f} seconds")
            return deduplicated_result

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

        # Count segment types with confidence weighting
        theoretical_count = 0
        practical_count = 0
        mixed_count = 0

        # Track total confidence-weighted counts
        theoretical_weighted = 0
        practical_weighted = 0
        mixed_weighted = 0

        # Track time distribution
        total_duration = 0
        theoretical_duration = 0
        practical_duration = 0
        mixed_duration = 0

        for segment in segments:
            segment_type = segment.get("content_type", "mixed")
            confidence = segment.get("classification_confidence", 0.6)  # Default confidence if not present

            # Calculate segment duration
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time
            total_duration += duration

            if segment_type == "theoretical":
                theoretical_count += 1
                theoretical_weighted += confidence
                theoretical_duration += duration
            elif segment_type == "practical":
                practical_count += 1
                practical_weighted += confidence
                practical_duration += duration
            else:  # mixed
                mixed_count += 1
                mixed_weighted += confidence
                mixed_duration += duration

        total_segments = theoretical_count + practical_count + mixed_count

        # Calculate theory/practice ratio with improved weighting
        if total_segments > 0:
            # Apply a weighted formula with confidence
            total_weighted = theoretical_weighted + practical_weighted + mixed_weighted

            if total_weighted > 0:
                # Apply confidence-weighted formula
                theory_weight = theoretical_weighted + (mixed_weighted * 0.5)
                theory_practice_ratio = theory_weight / total_weighted
            else:
                theory_practice_ratio = 0.5

            # Factor in duration-based ratio
            if total_duration > 0:
                duration_theory_ratio = (theoretical_duration + (mixed_duration * 0.5)) / total_duration

                # Final ratio is an average of count-based and duration-based ratios
                theory_practice_ratio = (theory_practice_ratio + duration_theory_ratio) / 2

        else:
            theory_practice_ratio = 0.5

        # Determine overall classification
        if theory_practice_ratio > 0.7:
            classification = "theoretical"
            confidence = 0.8 if theory_practice_ratio > 0.85 else 0.7
        elif theory_practice_ratio < 0.3:
            classification = "practical"
            confidence = 0.8 if theory_practice_ratio < 0.15 else 0.7
        else:
            classification = "mixed"
            closeness_to_half = 1.0 - abs(theory_practice_ratio - 0.5) * 2
            confidence = 0.6 + (closeness_to_half * 0.3)

        return {
            "classification": classification,
            "confidence": confidence,
            "theoretical_segments": theoretical_count,
            "practical_segments": practical_count,
            "mixed_segments": mixed_count,
            "theory_practice_ratio": theory_practice_ratio,
            "duration_analysis": {
                "total_duration": total_duration,
                "theoretical_duration": theoretical_duration,
                "practical_duration": practical_duration,
                "mixed_duration": mixed_duration
            }
        }

    def _extract_domain_features(self, processed_transcript: Dict, domain: str) -> Dict[str, Any]:
        """
        Extract domain-specific features from processed transcript using the unified concept extractor.

        Args:
            processed_transcript: Processed transcript dictionary
            domain: Content domain

        Returns:
            Dictionary with domain-specific features
        """
        segments = processed_transcript.get("segments", [])
        language = processed_transcript.get("language", "en")

        # Use the unified concept extractor to extract concepts from segments
        key_concepts = self.concept_extractor.extract_concepts_from_segments(segments, domain, language)

        # Organize concepts by theoretical/practical
        theoretical_concepts = [c for c in key_concepts if c.get("concept_class") == "theoretical"]
        practical_concepts = [c for c in key_concepts if c.get("concept_class") == "practical"]

        # Find relationships between concepts (simplified)
        concept_relationships = self._find_concept_relationships(key_concepts)

        return {
            "domain": domain,
            "key_concepts": key_concepts,
            "theoretical_concepts": theoretical_concepts,
            "practical_concepts": practical_concepts,
            "concept_relationships": concept_relationships
        }

    def _find_concept_relationships(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find relationships between concepts (simplified version).

        Args:
            concepts: List of concept dictionaries

        Returns:
            List of concept relationship dictionaries
        """
        if len(concepts) < 2:
            return []

        # Create a mapping of concepts to their occurrences
        concept_occurrences = {}
        for concept in concepts:
            concept_text = concept.get("text", "").lower()
            concept_occurrences[concept_text] = concept.get("occurrences", [])

        # Track concept co-occurrences in the same segments
        co_occurrences = {}

        # For each pair of concepts, check if they co-occur in segments
        concept_texts = list(concept_occurrences.keys())
        for i, concept1 in enumerate(concept_texts):
            for concept2 in concept_texts[i+1:]:
                # Skip if it's the same concept
                if concept1 == concept2:
                    continue

                # Get segments IDs for each concept
                segments1 = set(occ.get("segment_id") for occ in concept_occurrences.get(concept1, []))
                segments2 = set(occ.get("segment_id") for occ in concept_occurrences.get(concept2, []))

                # Check for common segments
                common_segments = segments1.intersection(segments2)

                if common_segments:
                    co_occurrences[(concept1, concept2)] = len(common_segments)

        # Create relationships from co-occurrences
        relationships = []

        for (source, target), count in co_occurrences.items():
            # Find the concept objects
            source_concept = next((c for c in concepts if c.get("text", "").lower() == source), None)
            target_concept = next((c for c in concepts if c.get("text", "").lower() == target), None)

            if source_concept and target_concept:
                relationship = {
                    "source_concept": source,
                    "target_concept": target,
                    "co_occurrence_count": count,
                    "relationship_type": "related",
                    "source_class": source_concept.get("concept_class", "theoretical"),
                    "target_class": target_concept.get("concept_class", "theoretical")
                }
                relationships.append(relationship)

        # Sort by co-occurrence count
        relationships.sort(key=lambda x: x.get("co_occurrence_count", 0), reverse=True)

        # Limit to top relationships
        max_relationships = min(25, len(relationships))
        return relationships[:max_relationships]

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
