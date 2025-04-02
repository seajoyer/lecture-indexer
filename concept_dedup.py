"""
Simplified concept deduplication module for the Lecture Video Content Indexer.
Handles normalization, similarity detection, and merging of duplicate concepts.
"""

import re
import logging
import difflib
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class ConceptDedupExtension:
    """
    Handles concept deduplication by detecting and merging similar concepts.
    """

    def __init__(self, data_access=None, language="en"):
        """
        Initialize the concept deduplication extension.

        Args:
            data_access: Optional data access layer instance
            language: Default language for normalization
        """
        self.data_access = data_access
        self.language = language

        # Import UnifiedConceptExtractor for consistent normalization
        try:
            from unified_concept_extractor import UnifiedConceptExtractor
            self.concept_extractor = UnifiedConceptExtractor(language)
            logger.info("Using UnifiedConceptExtractor for normalization")
        except ImportError:
            self.concept_extractor = None
            logger.warning("UnifiedConceptExtractor not available - using built-in normalization")

        # Set similarity thresholds
        self.similarity_thresholds = {
            "exact_match": 1.0,    # Exact match (after normalization)
            "high_match": 0.92,    # High confidence match
            "medium_match": 0.85,  # Medium confidence match
            "low_match": 0.75      # Low confidence match
        }

        logger.info(f"ConceptDedupExtension initialized with language: {language}")

    def normalize_concept_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Normalize concept text using UnifiedConceptExtractor if available,
        otherwise use built-in normalization.

        Args:
            text: Concept text
            language: Language code (defaults to instance language)

        Returns:
            Normalized text
        """
        if self.concept_extractor:
            # Use the UnifiedConceptExtractor for consistent normalization
            return self.concept_extractor.normalize_concept_text(text, language or self.language)

        # Fallback to simple normalization
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Remove common filler phrases at beginning
        if language == "ru" or self.language == "ru":
            # Russian filler phrases
            normalized = re.sub(r'^это\s+', '', normalized)   # "это " (this is)
            normalized = re.sub(r'^вот\s+', '', normalized)   # "вот " (here)
            normalized = re.sub(r'^да\s+', '', normalized)    # "да " (yes)
        else:
            # English filler phrases
            normalized = re.sub(r'^the\s+', '', normalized)    # "the " at beginning
            normalized = re.sub(r'^a\s+', '', normalized)      # "a " at beginning
            normalized = re.sub(r'^an\s+', '', normalized)     # "an " at beginning
            normalized = re.sub(r'^this\s+', '', normalized)   # "this " at beginning

        return normalized

    def is_valid_concept(self, text: str, language: Optional[str] = None) -> bool:
        """
        Check if text represents a valid concept using UnifiedConceptExtractor if available.

        Args:
            text: Concept text
            language: Language code

        Returns:
            True if valid concept, False otherwise
        """
        if self.concept_extractor:
            # Use the UnifiedConceptExtractor for consistent validation
            return self.concept_extractor.is_valid_concept(text, language or self.language)

        # Fallback to basic validation
        normalized = self.normalize_concept_text(text, language)

        if not normalized:
            return False

        # Check minimum length
        if len(normalized) < 3:
            return False

        # Check word count
        word_count = len(normalized.split())

        # Valid concept has 1-5 words
        if word_count < 1 or word_count > 5:
            return False

        # Check if it's mostly numbers
        if sum(c.isdigit() for c in normalized) / len(normalized) > 0.3:
            return False

        return True

    def calculate_concept_similarity(self, concept1: str, concept2: str, language: Optional[str] = None) -> float:
        """
        Calculate similarity between two concept texts.

        Args:
            concept1: First concept text
            concept2: Second concept text
            language: Language code

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize both concepts
        norm1 = self.normalize_concept_text(concept1, language)
        norm2 = self.normalize_concept_text(concept2, language)

        # If either normalization resulted in an empty string, they're not valid concepts
        if not norm1 or not norm2:
            return 0.0

        # Check for exact match after normalization
        if norm1 == norm2:
            return 1.0

        # Check for substring match
        if norm1 in norm2 or norm2 in norm1:
            # Calculate containment score based on length ratio
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            return shorter / longer * 0.95  # Slightly penalize to prioritize exact matches

        # Calculate string similarity
        string_similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()

        # Calculate word-level similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # Skip word-level similarity for very short concepts
        if len(words1) <= 1 or len(words2) <= 1:
            return string_similarity

        # Calculate Jaccard similarity for words
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        word_similarity = intersection / union if union > 0 else 0.0

        # Weight the two similarity measures based on concept length
        if len(words1) > 2 or len(words2) > 2:
            # For longer concepts, give more weight to word similarity
            combined_similarity = (string_similarity * 0.6) + (word_similarity * 0.4)
        else:
            # For shorter concepts, rely more on string similarity
            combined_similarity = (string_similarity * 0.8) + (word_similarity * 0.2)

        return combined_similarity

    def find_similar_concepts(self, concept: Dict[str, Any], concept_list: List[Dict[str, Any]],
                             threshold: float = 0.8, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find similar concepts to a given concept.

        Args:
            concept: Source concept dictionary
            concept_list: List of concept dictionaries to compare against
            threshold: Minimum similarity threshold
            language: Language code

        Returns:
            List of similar concepts with similarity scores
        """
        if not concept or not concept_list:
            return []

        concept_text = concept.get("text", "")

        # Skip invalid concepts
        if not self.is_valid_concept(concept_text, language):
            return []

        lang = language or concept.get("language", self.language)

        # Find similar concepts
        similar_concepts = []

        for other_concept in concept_list:
            # Skip identical concepts
            if concept == other_concept:
                continue

            other_text = other_concept.get("text", "")

            # Skip invalid comparison concepts
            if not self.is_valid_concept(other_text, lang):
                continue

            # Calculate similarity
            similarity = self.calculate_concept_similarity(concept_text, other_text, lang)

            # Add to results if above threshold
            if similarity >= threshold:
                result = other_concept.copy()
                result["similarity"] = similarity
                similar_concepts.append(result)

        # Sort by similarity (highest first)
        similar_concepts.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return similar_concepts

    def deduplicate_concepts(self, concepts: List[Dict[str, Any]], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Deduplicate and merge similar concepts.

        Args:
            concepts: List of concept dictionaries
            language: Language code

        Returns:
            Deduplicated list of concepts
        """
        if not concepts:
            return []

        # First, filter out invalid concepts
        lang = language or next((c.get("language") for c in concepts if "language" in c), self.language)

        valid_concepts = [c for c in concepts if self.is_valid_concept(c.get("text", ""), lang)]

        # If all concepts were filtered out, return the original concepts as valid
        if not valid_concepts and concepts:
            logger.warning(f"All concepts were filtered out as invalid. Returning original concepts.")
            valid_concepts = concepts.copy()

        if not valid_concepts:
            return []

        # Group by normalized text to identify duplicates
        normalized_groups = defaultdict(list)

        for concept in valid_concepts:
            normalized = self.normalize_concept_text(concept.get("text", ""), lang)
            if normalized:  # Skip empty normalization results
                normalized_groups[normalized].append(concept)

        # Select the best concept from each group
        deduplicated = []

        for normalized_text, group in normalized_groups.items():
            if len(group) == 1:
                # Only one concept with this normalized text
                deduplicated.append(group[0])
            else:
                # Multiple concepts with same normalized text, select the best one
                # Sort by score or frequency
                group.sort(key=lambda x: (
                    x.get("score", 0),
                    x.get("frequency", 0)
                ), reverse=True)

                # Use the highest scoring concept
                best_concept = group[0]

                # Combine frequencies
                total_frequency = sum(c.get("frequency", 1) for c in group)
                best_concept["frequency"] = total_frequency

                # Take definition from any concept if available
                for concept in group:
                    if concept.get("definition") and not best_concept.get("definition"):
                        best_concept["definition"] = concept["definition"]

                deduplicated.append(best_concept)

        # Sort by score and frequency
        deduplicated.sort(key=lambda x: (x.get("score", 0), x.get("frequency", 0)), reverse=True)

        return deduplicated

def apply_concept_deduplication(processed_result: Dict[str, Any], language: str = None) -> Dict[str, Any]:
    """
    Apply concept deduplication to a processed video result.

    Args:
        processed_result: Video processing result dictionary
        language: Language code

    Returns:
        Updated processing result with deduplicated concepts
    """
    if not processed_result:
        return processed_result

    # Extract domain features and language
    domain_features = processed_result.get("domain_features", {})
    video_language = processed_result.get("transcript", {}).get("language", "en")
    language = language or video_language

    # Extract concepts
    key_concepts = domain_features.get("key_concepts", [])
    theoretical_concepts = domain_features.get("theoretical_concepts", [])
    practical_concepts = domain_features.get("practical_concepts", [])

    if not key_concepts:
        logger.info("No concepts to deduplicate")
        return processed_result  # Nothing to deduplicate

    # Initialize deduplicator
    deduplicator = ConceptDedupExtension(language=language)

    # Deduplicate concepts
    deduplicated_key_concepts = deduplicator.deduplicate_concepts(key_concepts, language)
    deduplicated_theoretical = deduplicator.deduplicate_concepts(theoretical_concepts, language)
    deduplicated_practical = deduplicator.deduplicate_concepts(practical_concepts, language)

    # Update domain features
    domain_features["key_concepts"] = deduplicated_key_concepts
    domain_features["theoretical_concepts"] = deduplicated_theoretical
    domain_features["practical_concepts"] = deduplicated_practical

    # Update processed result
    processed_result["domain_features"] = domain_features

    # Log deduplication results
    logger.info(f"Deduplicated from {len(key_concepts)} to {len(deduplicated_key_concepts)} concepts")

    return processed_result
