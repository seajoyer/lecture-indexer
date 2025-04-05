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
            "medium_match": 0.80,  # Medium confidence match - REDUCED from 0.85 to avoid over-aggressive merging
            "low_match": 0.65      # Low confidence match - Reduced for better precision
        }

        # Load language-specific resources
        self._load_language_resources()

        logger.info(f"ConceptDedupExtension initialized with language: {language}")

    def _load_language_resources(self):
        """Load language-specific resources for better deduplication."""

        # Filler phrases to be ignored for similarity calculations
        self.filler_phrases = {
            "en": ["the", "a", "an", "is", "are", "of", "and", "or", "in", "on", "at", "to", "for"],
            "ru": ["это", "вот", "такое", "такой", "такая", "такие", "и", "или", "в", "на", "с", "из", "для", "к"]
        }

        # Domain-specific terms that should not be removed during normalization
        # These are important academic terms that help distinguish concepts
        self.domain_terms = {
            "physics": {
                "en": ["quantum", "wave", "function", "state", "operator", "eigenvalue", "eigenstate",
                       "hamiltonian", "hermitian", "hilbert", "space", "momentum", "energy", "position"],
                "ru": ["квантовый", "квантовая", "волновая", "функция", "состояние", "оператор",
                       "собственное", "значение", "собственный", "гамильтониан", "эрмитов",
                       "гильбертово", "пространство", "импульс", "энергия", "положение"]
            },
            "mathematics": {
                "en": ["function", "derivative", "integral", "vector", "matrix", "theorem", "lemma", "proof"],
                "ru": ["функция", "производная", "интеграл", "вектор", "матрица", "теорема", "лемма", "доказательство"]
            }
        }

        # Words that should be treated as equivalent for similarity comparison
        self.equivalent_terms = {
            "physics": {
                "en": {
                    "wave function": ["wavefunction", "wave-function"],
                    "quantum mechanics": ["quantum theory", "quantum physics"],
                    "eigenvalue": ["eigen-value", "eigen value", "characteristic value"],
                    "eigenstate": ["eigen-state", "eigen state", "characteristic state"],
                    "hamiltonian": ["hamilton operator", "energy operator"],
                    "hermitian operator": ["self-adjoint operator"],
                    "hilbert space": ["state space", "vector space"]
                },
                "ru": {
                    "волновая функция": ["волновой функции", "волновую функцию"],
                    "квантовая механика": ["квантовой механики", "квантовую механику", "квантовая теория"],
                    "собственное значение": ["собственным значением", "собственного значения", "характеристическое значение"],
                    "собственное состояние": ["собственным состоянием", "собственного состояния"],
                    "гамильтониан": ["оператор гамильтона", "гамильтонов оператор", "оператор энергии"],
                    "эрмитов оператор": ["эрмитовый оператор", "эрмитова оператор", "самосопряженный оператор"],
                    "гильбертово пространство": ["пространство состояний", "векторное пространство"]
                }
            }
        }

        # Common problematic phrases in Russian that should be normalized
        self.problematic_phrases = {
            "ru": {
                "то обсуждений давайте": "",
                "то состояние второго определённо такое": "",
                "вакуумное состояние оно": "вакуумное состояние",
                "эрмитово оператора": "эрмитов оператор",
                "любое собственное состояние": "собственное состояние",
                "эрмитово операторов": "эрмитов оператор",
                "операторы рождения определенные": "операторы рождения",
                "операторы уничтожения определенные": "операторы уничтожения"
            }
        }

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

        # Language-specific normalization
        lang = language or self.language

        if lang == "ru":
            # Replace problematic phrases
            for phrase, replacement in self.problematic_phrases.get("ru", {}).items():
                normalized = normalized.replace(phrase, replacement)

            # Remove common filler phrases at beginning
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
                "буду", "будем", "могу", "можем", "можно", "нужно", "должны", "хочу"
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
                "то обсуждений давайте",
                "эрмитово оператора",
                "то состояние второго определённо такое",
                "то состояние второго определённо такое это",
                "вакуумное состояние оно",
                "любое собственное состояние",
                "сейчас скажу",
                "потом обсужу",
                "некоторого некоторой",
                "состоянии вверх",
                "состояние едини на2",
                "приравняют формуле",
                "будем дальше",
                "буду получать",
                "давайте тогда",
                "эта процедура"
            ]

            if normalized in problematic_phrases:
                return False

        return True

    def calculate_concept_similarity(self, concept1: str, concept2: str, language: Optional[str] = None) -> float:
        """
        Calculate similarity between two concept texts with improved algorithm.

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

            # Adjust substring match score based on the ratio of lengths
            # Closer lengths = higher similarity
            length_ratio = shorter / longer

            # If one is a substantial substring of the other and the length difference isn't too large
            if length_ratio > 0.7:
                return 0.9 * length_ratio  # High similarity, but not quite 1.0 (exact match)
            else:
                return 0.8 * length_ratio  # Lower similarity for greater length difference

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

        # Check if key domain terms are shared
        lang = language or self.language

        # Get domain terms for this language
        domain_terms = set()
        for domain, terms_dict in self.domain_terms.items():
            if lang in terms_dict:
                domain_terms.update(terms_dict[lang])
            elif 'en' in terms_dict:  # Fallback to English
                domain_terms.update(terms_dict['en'])

        # Check if any important domain terms are shared between concepts
        words1_domain = words1.intersection(domain_terms)
        words2_domain = words2.intersection(domain_terms)
        shared_domain_terms = words1_domain.intersection(words2_domain)

        # Boost similarity if they share domain terms
        domain_term_bonus = 0.0
        if shared_domain_terms:
            # The more shared domain terms, the higher the bonus
            domain_term_bonus = min(0.2, len(shared_domain_terms) * 0.1)  # Cap at 0.2 boost

        # Weight the two similarity measures based on concept length
        if len(words1) > 2 or len(words2) > 2:
            # For longer concepts, give more weight to word similarity
            combined_similarity = (string_similarity * 0.4) + (word_similarity * 0.6) + domain_term_bonus
        else:
            # For shorter concepts, rely more on string similarity
            combined_similarity = (string_similarity * 0.7) + (word_similarity * 0.3) + domain_term_bonus

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
                # Apply stronger penalty to prevent incorrect merging
                combined_similarity *= 0.6  # Apply penalty

        # Cap at 1.0
        return min(combined_similarity, 1.0)

    def find_similar_concepts(self,
                             concept: Dict[str, Any],
                             concept_list: List[Dict[str, Any]],
                             threshold: float = 0.8,
                             language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find similar concepts to a given concept with improved matching.

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
        Deduplicate and merge similar concepts with improved algorithm.

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

        # Use normalized text for grouping
        normalized_to_concept = {}
        for concept in valid_concepts:
            normalized = self.normalize_concept_text(concept.get("text", ""), lang)
            if normalized:
                # If we've seen this normalized text before, keep the one with higher score/frequency
                if normalized in normalized_to_concept:
                    existing = normalized_to_concept[normalized]
                    existing_score = existing.get("score", 0) + existing.get("frequency", 0) * 0.5
                    new_score = concept.get("score", 0) + concept.get("frequency", 0) * 0.5

                    if new_score > existing_score:
                        normalized_to_concept[normalized] = concept
                else:
                    normalized_to_concept[normalized] = concept

        # Initial deduplication by exact normalized text
        deduplicated = list(normalized_to_concept.values())

        # Now find similar concepts using our improved similarity measure
        similar_groups = []
        remaining = deduplicated.copy()

        while remaining:
            # Take the first concept as a seed
            seed = remaining.pop(0)

            # Find all concepts similar to this seed
            similar_to_seed = self.find_similar_concepts(
                seed, remaining, threshold=self.similarity_thresholds["medium_match"], language=lang
            )

            # Create a group with the seed and all similar concepts
            group = [seed] + similar_to_seed

            # Remove all similar concepts from the remaining list
            remaining = [c for c in remaining if c not in similar_to_seed]

            # Add this group to our groups
            similar_groups.append(group)

        # Merge each group into a single concept
        merged_concepts = []

        for group in similar_groups:
            if len(group) == 1:
                # If only one concept in the group, just add it
                merged_concepts.append(group[0])
            else:
                # Sort by score, frequency, and word count to find the best representative
                group.sort(key=lambda c: (
                    c.get("score", 0) * 2 +
                    c.get("frequency", 0) * 3 +
                    (0.5 * len(c.get("text", "").split()))  # Slightly favor multi-word concepts
                ), reverse=True)

                # Use the highest scoring concept as the base
                best_concept = group[0].copy()

                # Track additional information from variants
                total_frequency = best_concept.get("frequency", 1)
                variant_texts = []

                # Merge information from other concepts in the group
                for variant in group[1:]:
                    # Accumulate frequency
                    total_frequency += variant.get("frequency", 1)

                    # Collect variant texts
                    variant_texts.append(variant.get("text", ""))

                    # Use definition from variant if primary doesn't have one
                    if variant.get("definition") and not best_concept.get("definition"):
                        best_concept["definition"] = variant["definition"]

                # Update the merged concept
                best_concept["frequency"] = total_frequency
                best_concept["variant_texts"] = variant_texts
                best_concept["variants_count"] = len(variant_texts)

                merged_concepts.append(best_concept)

        # Final sort by frequency and score
        merged_concepts.sort(key=lambda x: (
            x.get("frequency", 1) * 2 +
            x.get("score", 0) +
            (x.get("domain_match", False) * 3)  # Boost domain-matched concepts
        ), reverse=True)

        return merged_concepts

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
