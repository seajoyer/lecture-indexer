"""
Enhanced Unified Concept Extractor for the Lecture Video Content Indexer.

Identifies concept occurrences in video transcripts by matching against the concept repository.
Calculates educational significance to distinguish between passing mentions and comprehensive explanations.
"""

import re
import uuid
import logging
import time
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter, defaultdict
import string
import json

# Import concept repository for concept matching
try:
    from concept_repository import get_concept_repository
except ImportError:
    logging.warning("Could not import concept_repository - running in limited mode")
    get_concept_repository = lambda: None

# Configure logging
logger = logging.getLogger(__name__)

class UnifiedConceptExtractor:
    """
    Enhanced concept extractor with robust concept matching and educational significance detection.
    Analyzes video transcripts to identify concept occurrences and evaluate their significance.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the concept extractor.

        Args:
            language: Default language code ('en' or 'ru')
        """
        self.language = language

        # Get concept repository
        self.concept_repository = get_concept_repository()
        if not self.concept_repository:
            logger.warning("Concept repository not available - limited functionality")

        # Load enhanced NLP resources
        self._load_nlp_resources()

        logger.info(f"UnifiedConceptExtractor initialized for language: {language}")

    def _load_nlp_resources(self):
        """Load comprehensive NLP resources including educational markers."""
        # Educational content markers for substantive explanations
        self.educational_markers = {
            "en": [
                r'important concept',
                r'key principle',
                r'fundamental idea',
                r'essential to understand',
                r'core concept',
                r'critical to',
                r'central idea',
                r'primarily concerned with',
                r'focuses on',
                r'the main',
                r'in depth',
                r'thoroughly',
                r'explain in detail',
                r'explore the',
                r'analyze',
                r'examine',
                r'investigate',
                r'detailed',
                r'significant',
                r'important',
                r'crucial',
                r'vital',
                r'key',
                r'central',
                r'underlying',
                r'foundation',
                r'basis',
                r'fundamental',
                r'primary',
                r'comprehensive',
                r'thorough',
                r'elaborate',
                r'rigorous',
                r'systematic',
                r'precise',
                r'specific',
                r'in-depth',
                r'detailed analysis',
                r'extensive discussion',
                r'is defined as',
                r'refers to',
                r'means',
                r'is a type of',
                r'is a form of',
                r'is characterized by',
                r'consists of',
                r'comprises',
                r'is composed of'
            ],
            "ru": [
                r'важная концепция',
                r'ключевой принцип',
                r'фундаментальная идея',
                r'необходимо понять',
                r'основная концепция',
                r'критически важно',
                r'центральная идея',
                r'в первую очередь',
                r'фокусируется на',
                r'главный',
                r'подробно',
                r'тщательно',
                r'объяснить детально',
                r'исследовать',
                r'анализировать',
                r'изучить',
                r'исследовать',
                r'детальный',
                r'значительный',
                r'важный',
                r'существенный',
                r'жизненно важный',
                r'ключевой',
                r'центральный',
                r'лежащий в основе',
                r'фундамент',
                r'основа',
                r'фундаментальный',
                r'главный',
                r'всесторонний',
                r'тщательный',
                r'подробный',
                r'строгий',
                r'систематический',
                r'точный',
                r'специфический',
                r'углубленный',
                r'детальный анализ',
                r'обширное обсуждение',
                r'определяется как',
                r'относится к',
                r'означает',
                r'является типом',
                r'является формой',
                r'характеризуется',
                r'состоит из',
                r'включает',
                r'состоит из'
            ]
        }

        # Compile educational markers patterns
        self.educational_markers_regex = {
            lang: re.compile('|'.join(patterns), re.IGNORECASE)
            for lang, patterns in self.educational_markers.items()
        }

    def extract_concepts_from_transcript(
        self,
        processed_transcript: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from a processed transcript by matching against the concept repository.

        Args:
            processed_transcript: Processed transcript from TranscriptProcessor

        Returns:
            Dictionary containing concepts and their occurrences
        """
        segments = processed_transcript.get("segments", [])
        language = processed_transcript.get("language", "en")
        video_id = processed_transcript.get("video_id", "unknown")
        global_analysis = processed_transcript.get("global_analysis", {})

        # Set language for processing
        self.language = language

        # Log input information
        logger.info(f"Extracting concepts from transcript: video_id={video_id}, language={language}, segments={len(segments)}")

        # Extract concepts using repository matching
        result = self.extract_concepts_from_segments(
            segments,
            video_id,
            language,
            global_analysis
        )

        # Add detailed debugging to check concepts and occurrences
        concepts = result.get("concepts", [])
        educational_concepts = sum(1 for c in concepts if c.get("is_educational", False))
        passing_concepts = len(concepts) - educational_concepts

        total_occurrences = sum(len(c.get("occurrences", [])) for c in concepts)

        logger.info(f"Extraction complete: {len(concepts)} concepts found ({educational_concepts} educational, {passing_concepts} passing)")
        logger.info(f"Total occurrences: {total_occurrences}")

        # Log the first 5 concepts with their occurrences for debugging
        for i, concept in enumerate(concepts[:5]):
            concept_id = concept.get("concept_id", "unknown")
            occurrences = concept.get("occurrences", [])
            logger.info(f"Concept {i+1}: {concept_id} - {len(occurrences)} occurrences")

            # Log the first 2 occurrences for each concept
            for j, occ in enumerate(occurrences[:2]):
                segment_id = occ.get("segment_id", "unknown")
                start_time = occ.get("start_time", 0)
                edu_sig = occ.get("educational_significance", 0)

                logger.info(f"  Occurrence {j+1}: segment_id={segment_id}, start_time={start_time}, significance={edu_sig}")

        return result

    def extract_concepts_from_segments(
        self,
        segments: List[Dict[str, Any]],
        video_id: str,
        language: Optional[str] = None,
        global_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from transcript segments by matching against the concept repository.

        Args:
            segments: List of transcript segments
            video_id: Video ID
            language: Optional language filter
            global_analysis: Optional global text analysis data

        Returns:
            Dictionary containing concepts with occurrences
        """
        # Use specified language or default
        lang = language or self.language

        # Check if concept repository is available
        if not self.concept_repository:
            logger.warning("Concept repository not available for matching")
            return {"concepts": []}

        # Track time
        start_time = time.time()

        # First, build a text representation of the full transcript
        combined_text = " ".join([segment.get("text", "") for segment in segments])

        # Use segment map for quick lookups
        segment_map = {segment.get("id", str(uuid.uuid4())): segment for segment in segments}

        # Find matching concepts
        logger.info(f"Finding matching concepts for video {video_id} in language: {lang}")

        # Track matched concepts with their occurrences
        matched_concepts = {}

        # Process each segment
        for segment in segments:
            segment_id = segment.get("id", str(uuid.uuid4()))
            segment_text = segment.get("text", "")
            start_time_sec = segment.get("start_time", 0.0)

            if not segment_text.strip():
                continue

            # Find matching concepts in this segment
            segment_matches = self._find_matching_concepts_in_text(
                segment_text,
                language=lang,
                threshold=0.7  # Adjust threshold as needed
            )

            # Process each matched concept
            for match in segment_matches:
                concept_id = match.get("concept_id")
                concept = match.get("concept")
                similarity = match.get("similarity", 0.0)

                if not concept_id or not concept:
                    continue

                # Calculate educational significance
                educational_significance = self._calculate_educational_significance(
                    segment,
                    concept,
                    match.get("matched_representation", ""),
                    global_analysis
                )

                # Determine occurrence type based on significance
                occurrence_type = "comprehensive" if educational_significance >= 2.5 else "passing"

                # Create occurrence record
                occurrence = {
                    "occurrence_id": str(uuid.uuid4()),
                    "concept_id": concept_id,
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "start_time": start_time_sec,
                    "educational_significance": educational_significance,
                    "occurrence_type": occurrence_type,
                    "similarity": similarity,
                    "context_text": segment_text
                }

                # Add to matched concepts
                if concept_id not in matched_concepts:
                    # Get representations for display
                    representations = concept.get("representations", {})

                    # Create concept entry
                    matched_concepts[concept_id] = {
                        "concept_id": concept_id,
                        "text": match.get("matched_representation", ""),
                        "representations": representations,
                        "language": lang,
                        "occurrences": [occurrence],
                        "educational_significance": educational_significance,
                        "is_educational": educational_significance >= 2.5
                    }
                else:
                    # Update existing concept
                    matched_concepts[concept_id]["occurrences"].append(occurrence)

                    # Update educational significance if this occurrence is more significant
                    if educational_significance > matched_concepts[concept_id]["educational_significance"]:
                        matched_concepts[concept_id]["educational_significance"] = educational_significance
                        matched_concepts[concept_id]["is_educational"] = educational_significance >= 2.5

        # Convert matched concepts to list
        result_concepts = list(matched_concepts.values())

        # Sort by educational significance and occurrence count
        result_concepts.sort(
            key=lambda c: (c.get("educational_significance", 0.0), len(c.get("occurrences", []))),
            reverse=True
        )

        # Calculate stats
        processing_time = time.time() - start_time
        total_concepts = len(result_concepts)
        educational_concepts = sum(1 for c in result_concepts if c.get("is_educational", False))
        passing_concepts = total_concepts - educational_concepts

        logger.info(f"Found {total_concepts} concepts ({educational_concepts} educational, {passing_concepts} passing) in {processing_time:.2f}s")

        # Return the concepts
        return {
            "concepts": result_concepts,
            "educational_concepts_count": educational_concepts,
            "passing_concepts_count": passing_concepts
        }

    def _find_matching_concepts_in_text(self, text: str, language: str, threshold: float = 0.7) -> List[Dict]:
        """
        Find matching concepts in text using the concept repository.

        Args:
            text: Text to search in
            language: Language code
            threshold: Similarity threshold

        Returns:
            List of matching concept dictionaries
        """
        # Use concept repository to find matches
        matches = self.concept_repository.find_concepts_by_text(
            text,
            language=language,
            threshold=threshold,
            max_results=10  # Limit to avoid excessive processing
        )

        # Check if each matched concept exists in the database for valid foreign key relationships
        valid_matches = []
        for match in matches:
            concept_id = match.get("concept_id")
            if not concept_id:
                continue

            # Add to valid matches
            valid_matches.append(match)

        return valid_matches

    def _calculate_educational_significance(
        self,
        segment: Dict[str, Any],
        concept: Dict[str, Any],
        concept_text: str,
        global_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate educational significance score to distinguish between passing mentions and comprehensive explanations.

        Args:
            segment: Segment data
            concept: Concept data
            concept_text: The matched concept text
            global_analysis: Optional global text analysis

        Returns:
            Educational significance score (0.0-5.0)
        """
        significance_score = 0.0

        # Factor 1: Segment's educational value
        segment_edu_value = segment.get("educational_value", 0.0)
        significance_score += segment_edu_value * 0.8  # Weight: 0.8

        # Factor 2: Educational markers in the context
        text = segment.get("text", "")
        language = segment.get("language", "en")

        # Check for educational markers in the text
        lang = language if language in self.educational_markers_regex else 'en'
        has_edu_markers = bool(self.educational_markers_regex[lang].search(text))

        if has_edu_markers:
            significance_score += 1.2  # Weight: 1.2

        # Factor 3: Position of concept in text (central vs. peripheral)
        if concept_text and text:
            concept_pos = text.lower().find(concept_text.lower())
            if concept_pos >= 0:
                # Calculate relative position (0.0 = start, 1.0 = end)
                relative_pos = concept_pos / max(1, len(text) - len(concept_text))

                # Score is highest for concepts in the middle of text (likely the focus)
                # and lower for concepts at the very beginning or end
                centrality = 1.0 - 2.0 * abs(0.5 - relative_pos)
                significance_score += centrality * 0.5  # Weight: 0.5

        # Factor 4: Text surrounding the concept
        if concept_text and text:
            # Look for explanatory phrases near the concept
            concept_pos = text.lower().find(concept_text.lower())
            if concept_pos >= 0:
                # Get surrounding text (50 chars before and after)
                start = max(0, concept_pos - 50)
                end = min(len(text), concept_pos + len(concept_text) + 50)
                surrounding = text[start:end].lower()

                # Check for explanatory phrases
                explanatory_phrases = [
                    "is defined as", "refers to", "means", "is a type of", "is a form of",
                    "is characterized by", "consists of", "comprises", "is composed of"
                ]

                # Translate phrases for Russian
                if language == "ru":
                    explanatory_phrases = [
                        "определяется как", "относится к", "означает", "является типом", "является формой",
                        "характеризуется", "состоит из", "включает", "состоит из"
                    ]

                for phrase in explanatory_phrases:
                    if phrase in surrounding:
                        significance_score += 1.0  # Weight: 1.0
                        break

        # Factor 5: Context length (longer contexts typically have more explanation)
        context_length = len(text.split())
        if context_length > 30:  # Long context
            significance_score += 0.8
        elif context_length > 15:  # Medium context
            significance_score += 0.4

        # Factor 6: Global analysis factors
        if global_analysis:
            # Check if concept is in key terms
            key_terms = global_analysis.get("key_terms", [])
            if concept_text.lower() in [term.lower() for term in key_terms]:
                significance_score += 0.5  # Weight: 0.5

        # Cap the score at 5.0
        return min(significance_score, 5.0)
