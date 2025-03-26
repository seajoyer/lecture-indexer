"""
Simplified transcript processor for the Lecture Video Content Indexer.
Handles processing of raw transcripts into structured text suitable for analysis.
"""

import re
import uuid
import logging
import nltk
from typing import List, Dict
from nltk.tokenize import sent_tokenize

# Import simplified modules
from cache_manager import cache_get, cache_set
from performance_utils import time_function

# Configure logging
logger = logging.getLogger(__name__)

# Download necessary NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class TranscriptProcessor:
    """
    Processes raw transcripts into structured text suitable for analysis.
    Simplified version with reduced complexity and dependencies.
    """

    def __init__(self):
        """Initialize the transcript processor."""
        logger.info("TranscriptProcessor initialized")

    @time_function(5000)  # Log warning if takes more than 5 seconds
    def process_transcript(self, raw_segments: List[Dict], video_metadata: Dict) -> Dict:
        """
        Process raw transcript segments into a structured format.

        Args:
            raw_segments: List of raw transcript segments
            video_metadata: Video metadata dictionary

        Returns:
            Dictionary containing processed transcript data
        """
        video_id = video_metadata.get("video_id", "")

        # Check cache first
        cache_key = f"processed_transcript_{video_id}"
        cached_result = cache_get("transcript", cache_key)
        if cached_result:
            logger.info(f"Using cached processed transcript for video {video_id}")
            return cached_result

        if not raw_segments:
            logger.warning("Empty transcript provided")
            result = {
                "segments": [],
                "language": "en",
                "domain": "unknown",
                "video_id": video_id
            }
            return result

        # Determine language from first segment or metadata
        language = raw_segments[0].get("language", video_metadata.get("language", "en"))
        if not language:
            language = "en"  # Default to English

        # Domain from metadata
        domain = video_metadata.get("domain", "unknown")

        # Normalize transcript segments
        normalized_segments = self._normalize_transcript(raw_segments, language)

        # Segment into sentences (when possible)
        try:
            sentence_segments = self._segment_into_sentences(normalized_segments, language)
        except Exception as e:
            logger.warning(f"Error segmenting into sentences: {e}, using original segments")
            sentence_segments = normalized_segments

        # Classify segments as theoretical or practical
        classified_segments = self._classify_segments(sentence_segments, domain)

        # Combine results
        result = {
            "segments": classified_segments,
            "language": language,
            "domain": domain,
            "video_id": video_id
        }

        # Cache the result
        cache_set("transcript", cache_key, result)

        logger.info(f"Processed transcript with {len(classified_segments)} segments")
        return result

    def _normalize_transcript(self, raw_segments: List[Dict], language: str) -> List[Dict]:
        """
        Normalize raw transcript segments.

        Args:
            raw_segments: List of raw transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            List of normalized transcript segments
        """
        normalized_segments = []

        for segment in raw_segments:
            text = segment.get("text", "")

            # Skip empty segments
            if not text.strip():
                continue

            # Basic text normalization
            if language == "ru":
                normalized_text = self._normalize_russian_text(text)
            else:
                normalized_text = self._normalize_english_text(text)

            # Create normalized segment
            normalized_segment = {
                "id": str(uuid.uuid4()),
                "start_time": segment.get("start", 0),
                "end_time": segment.get("start", 0) + segment.get("duration", 0),
                "text": normalized_text,
                "language": language
            }

            normalized_segments.append(normalized_segment)

        return normalized_segments

    def _segment_into_sentences(self, normalized_segments: List[Dict], language: str) -> List[Dict]:
        """
        Segment normalized transcript into sentences.

        Args:
            normalized_segments: List of normalized transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            List of sentence segments
        """
        sentence_segments = []

        for segment in normalized_segments:
            text = segment.get("text", "")

            # Use language-specific sentence tokenization when possible
            try:
                sentences = sent_tokenize(text, language='russian' if language == 'ru' else 'english')
            except:
                # Fallback to simple splitting on sentence terminators
                sentences = re.split(r'(?<=[.!?])\s+', text)

            # If no sentences were detected, use the whole segment as one sentence
            if not sentences:
                sentences = [text]

            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time

            # Create sentence segments with interpolated timestamps
            for i, sentence in enumerate(sentences):
                # Skip empty sentences
                if not sentence.strip():
                    continue

                # Estimate time position proportionally to text length
                sentence_length = len(sentence)
                total_length = sum(len(s) for s in sentences)

                if total_length == 0:
                    # Avoid division by zero
                    sentence_start = start_time
                    sentence_end = end_time
                else:
                    prev_length = sum(len(s) for s in sentences[:i])

                    # Calculate start and end times
                    sentence_start = start_time + (duration * prev_length / total_length)
                    sentence_end = sentence_start + (duration * sentence_length / total_length)

                    # Ensure the last sentence ends at the segment end time
                    if i == len(sentences) - 1:
                        sentence_end = end_time

                # Create sentence segment
                sentence_segment = {
                    "id": str(uuid.uuid4()),
                    "start_time": sentence_start,
                    "end_time": sentence_end,
                    "text": sentence.strip(),
                    "language": language,
                    "original_segment_id": segment.get("id")
                }

                sentence_segments.append(sentence_segment)

        return sentence_segments

    def _classify_segments(self, segments: List[Dict], domain: str) -> List[Dict]:
        """
        Classify segments as theoretical or practical.

        Args:
            segments: List of transcript segments
            domain: Content domain

        Returns:
            List of classified segments
        """
        classified_segments = []

        for segment in segments:
            text = segment.get("text", "")

            # Determine if segment is theoretical or practical
            content_type = self._classify_content_type(text, domain)

            # Create classified segment
            classified_segment = segment.copy()
            classified_segment["content_type"] = content_type

            classified_segments.append(classified_segment)

        return classified_segments

    def _classify_content_type(self, text: str, domain: str) -> str:
        """
        Determine if text is theoretical or practical.

        Args:
            text: Text to classify
            domain: Content domain

        Returns:
            Classification ("theoretical", "practical", or "mixed")
        """
        text_lower = text.lower()

        # Check for theoretical indicators
        theoretical_indicators = [
            "definition", "theorem", "proof", "theory", "concept",
            "defined as", "is called", "refers to", "represents",
            "introduced", "developed", "formulated", "demonstrated"
        ]

        # Check for practical indicators
        practical_indicators = [
            "example", "practice", "exercise", "implement", "case study",
            "problem", "application", "solve", "calculate", "compute",
            "let's try", "demonstrate", "step by step", "code", "algorithm"
        ]

        # Count indicators
        theoretical_count = sum(1 for indicator in theoretical_indicators if indicator in text_lower)
        practical_count = sum(1 for indicator in practical_indicators if indicator in text_lower)

        # Classify based on indicator counts
        if theoretical_count > practical_count:
            return "theoretical"
        elif practical_count > theoretical_count:
            return "practical"
        else:
            # Add domain-specific classification when counts are equal
            if domain == "mathematics":
                # Check for math symbols commonly used in theory
                if any(symbol in text for symbol in ["∫", "∑", "∏", "∀", "∃", "→", "∴", "∵"]):
                    return "theoretical"
                # Check for calculation indicators
                elif any(indicator in text_lower for indicator in ["calculate", "compute", "find", "solve"]):
                    return "practical"
            elif domain == "programming":
                # Check for code or programming constructs
                if any(construct in text_lower for construct in ["function", "class", "method", "variable", "object"]):
                    return "theoretical"
                elif any(indicator in text_lower for indicator in ["code", "implement", "write", "program"]):
                    return "practical"
            elif domain == "physics":
                # Check for physics theory
                if any(indicator in text_lower for indicator in ["law", "principle", "hypothesis", "model"]):
                    return "theoretical"
                # Check for physics experiments
                elif any(indicator in text_lower for indicator in ["experiment", "measurement", "observation", "data"]):
                    return "practical"

            # Default to mixed when no clear distinction
            return "mixed"

    def _normalize_english_text(self, text: str) -> str:
        """Normalize English text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix common caption errors
        text = text.replace(" i ", " I ")
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Add space after period

        # Remove speaker identifiers like "[Professor]:"
        text = re.sub(r'\[\w+\]:', '', text)

        # Fix ellipses
        text = re.sub(r'\.\.\.+', '...', text)

        # Remove musical notes, applause indicators, etc.
        text = re.sub(r'\[.*?\]', '', text)

        return text.strip()

    def _normalize_russian_text(self, text: str) -> str:
        """Normalize Russian text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix punctuation issues
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Add space after period

        # Remove speaker identifiers like "[Профессор]:"
        text = re.sub(r'\[\w+\]:', '', text)

        # Fix ellipses
        text = re.sub(r'\.\.\.+', '...', text)

        # Remove musical notes, applause indicators, etc.
        text = re.sub(r'\[.*?\]', '', text)

        return text.strip()
