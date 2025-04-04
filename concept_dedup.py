"""
Enhanced concept deduplication module for the Lecture Video Content Indexer.
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
            "medium_match": 0.85,  # Medium confidence match - REDUCED from 0.85 to avoid over-aggressive merging
            "low_match": 0.70      # Low confidence match
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
        lang = language or self.language
        if lang == "ru":
            # Russian filler phrases
            normalized = re.sub(r'^это\s+', '', normalized)   # "это " (this is)
            normalized = re.sub(r'^вот\s+', '', normalized)   # "вот " (here)
            normalized = re.sub(r'^да\s+', '', normalized)    # "да " (yes)
            normalized = re.sub(r'^ну\s+', '', normalized)    # "ну " (well)
            normalized = re.sub(r'^то\s+', '', normalized)    # "то " (that)
            normalized = re.sub(r'^такое\s+', '', normalized) # "такое " (such)
            normalized = re.sub(r'^оно\s+', '', normalized)   # "оно " (it)
            normalized = re.sub(r'^у\s+нас\s+', '', normalized) # "у нас " (we have)
            normalized = re.sub(r'^мы\s+', '', normalized)    # "мы " (we)
            normalized = re.sub(r'^я\s+', '', normalized)     # "я " (I)
            normalized = re.sub(r'^вы\s+', '', normalized)    # "вы " (you)
            normalized = re.sub(r'^они\s+', '', normalized)   # "они " (they)
            normalized = re.sub(r'^давайте\s+', '', normalized) # "давайте " (let's)

            # Remove problematic phrases completely
            normalized = normalized.replace("то обсуждений давайте", "")
            normalized = normalized.replace("то состояние второго определённо такое", "")
            normalized = normalized.replace("вакуумное состояние оно", "вакуумное состояние")
            normalized = normalized.replace("эрмитово оператора", "эрмитов оператор")
            normalized = normalized.replace("любое собственное состояние", "собственное состояние")

            # Fix ending filler words
            normalized = re.sub(r'\s+оно$', '', normalized)
            normalized = re.sub(r'\s+это$', '', normalized)
            normalized = re.sub(r'\s+такое$', '', normalized)
            normalized = re.sub(r'\s+такое\s+это$', '', normalized)
            normalized = re.sub(r'\s+второго\s+определённо\s+такое$', '', normalized)
            normalized = re.sub(r'\s+второго\s+определённо\s+такое\s+это$', '', normalized)
            normalized = re.sub(r'\s+определённо\s+такое$', '', normalized)
            normalized = re.sub(r'\s+определённо\s+такое\s+это$', '', normalized)
        else:
            # English filler phrases
            normalized = re.sub(r'^the\s+', '', normalized)    # "the " at beginning
            normalized = re.sub(r'^a\s+', '', normalized)      # "a " at beginning
            normalized = re.sub(r'^an\s+', '', normalized)     # "an " at beginning
            normalized = re.sub(r'^this\s+', '', normalized)   # "this " at beginning
            normalized = re.sub(r'^that\s+', '', normalized)   # "that " at beginning
            normalized = re.sub(r'^we\s+', '', normalized)     # "we " at beginning
            normalized = re.sub(r'^I\s+', '', normalized)      # "I " at beginning
            normalized = re.sub(r'^you\s+', '', normalized)    # "you " at beginning
            normalized = re.sub(r'^they\s+', '', normalized)   # "they " at beginning
            normalized = re.sub(r'^it\s+', '', normalized)     # "it " at beginning

        # Remove common filler phrases at end
        if lang == "ru":
            normalized = re.sub(r'\s+это$', '', normalized)     # " это" at end
            normalized = re.sub(r'\s+оно$', '', normalized)     # " оно" at end
            normalized = re.sub(r'\s+такое$', '', normalized)   # " такое" at end
        else:
            normalized = re.sub(r'\s+is$', '', normalized)      # " is" at end
            normalized = re.sub(r'\s+are$', '', normalized)     # " are" at end
            normalized = re.sub(r'\s+be$', '', normalized)      # " be" at end
            normalized = re.sub(r'\s+been$', '', normalized)    # " been" at end
            normalized = re.sub(r'\s+have$', '', normalized)    # " have" at end
            normalized = re.sub(r'\s+has$', '', normalized)     # " has" at end
            normalized = re.sub(r'\s+had$', '', normalized)     # " had" at end

        # Check for complete invalid phrases
        invalid_phrases = [
            "то обсуждений давайте", "то обсуждений", "обсуждений давайте",
            "то состояние второго", "то состояние", "состояние второго",
            "определённо такое", "второго определённо", "давайте это"
        ]
        if lang == "ru" and normalized in invalid_phrases:
            return ""

        # Remove extra whitespace again after all replacements
        normalized = re.sub(r'\s+', ' ', normalized).strip()

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

        # Additional Russian validation
        lang = language or self.language
        if lang == "ru":
            # Check for common Russian filler words that shouldn't be concepts
            filler_words = {
                "это", "вот", "то", "такое", "так", "оно", "она", "они", "мы", "вы", "я",
                "давайте", "обсуждений", "рассмотрим", "посмотрим", "второго", "определённо",
                "да", "нет", "просто", "только", "всегда", "сейчас", "здесь", "там",
                "эрмитово", "оператора"  # Specific problematic terms from output
            }

            # If concept is a single word and it's in the filler words list
            if word_count == 1 and normalized in filler_words:
                return False

            # If concept consists entirely of filler words
            words = normalized.split()
            if all(word in filler_words for word in words):
                return False

            # Common invalid patterns for Russian
            invalid_patterns = [
                r'то\s+\w+\s+это',  # "то [word] это" pattern
                r'это\s+\w+\s+такое',  # "это [word] такое" pattern
                r'вот\s+\w+\s+оно',  # "вот [word] оно" pattern
                r'давайте\s+\w+',  # "давайте [word]" pattern
                r'то\s+обсуждений',  # "то обсуждений" pattern
                r'такое\s+это',  # "такое это" pattern
                r'оно\s+это',  # "оно это" pattern
                r'то\s+состояние',  # "то состояние" pattern
                r'второго\s+определённо',  # "второго определённо" pattern
                r'эрмитово\s+оператора' # "эрмитово оператора" pattern
            ]

            for pattern in invalid_patterns:
                if re.search(pattern, normalized):
                    return False

            # Check specific problematic phrases from output
            problematic_phrases = [
                "то обсуждений давайте это",
                "эрмитово оператора",
                "то состояние второго определённо такое",
                "то состояние второго определённо такое это",
                "вакуумное состояние оно",
                "любое собственное состояние"
            ]

            if normalized in problematic_phrases:
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

        # Add a penalty for Russian concepts with different scientific terms
        # This prevents merging distinct quantum mechanics concepts
        lang = language or self.language
        if lang == "ru":
            # Check if the concepts contain different important terms
            important_terms = {
                "квантовый", "квантовая", "квантовое", "собственное", "эрмитов", "эрмитово",
                "гамильтониан", "волновая", "вакуумное", "матрица", "оператор", "значение",
                "состояние", "функция", "плотности", "вектор", "пространство"
            }

            words1_important = words1.intersection(important_terms)
            words2_important = words2.intersection(important_terms)

            # If both have important terms but they're different, reduce similarity
            if words1_important and words2_important and not words1_important.intersection(words2_important):
                combined_similarity *= 0.7  # Apply penalty

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
            logger.warning(f"All concepts were filtered out as invalid. Returning top original concepts.")
            # Sort by score or frequency
            sorted_concepts = sorted(concepts, key=lambda x: (
                x.get("score", 0),
                x.get("frequency", 0)
            ), reverse=True)

            # Return only the top 20 concepts to filter out the worst ones
            return sorted_concepts[:20]

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

    # Filter out invalid concepts first (directly addresses the issue with the problematic phrases)
    filtered_key_concepts = [c for c in key_concepts if deduplicator.is_valid_concept(c.get("text", ""), language)]
    filtered_theoretical = [c for c in theoretical_concepts if deduplicator.is_valid_concept(c.get("text", ""), language)]
    filtered_practical = [c for c in practical_concepts if deduplicator.is_valid_concept(c.get("text", ""), language)]

    # Deduplicate concepts
    deduplicated_key_concepts = deduplicator.deduplicate_concepts(filtered_key_concepts, language)
    deduplicated_theoretical = deduplicator.deduplicate_concepts(filtered_theoretical, language)
    deduplicated_practical = deduplicator.deduplicate_concepts(filtered_practical, language)

    # Update domain features
    domain_features["key_concepts"] = deduplicated_key_concepts
    domain_features["theoretical_concepts"] = deduplicated_theoretical
    domain_features["practical_concepts"] = deduplicated_practical

    # Update processed result
    processed_result["domain_features"] = domain_features

    # Log deduplication results
    logger.info(f"Filtered from {len(key_concepts)} to {len(filtered_key_concepts)} valid concepts")
    logger.info(f"Deduplicated from {len(filtered_key_concepts)} to {len(deduplicated_key_concepts)} concepts")

    return processed_result
