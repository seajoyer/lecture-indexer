"""
Concept deduplication module for the Lecture Video Content Indexer.
Handles normalization, similarity detection, and merging of duplicate concepts.

This module provides tools to:
1. Normalize concept text for consistent matching
2. Detect and deduplicate similar concepts
3. Establish canonical concept relationships
4. Improve search quality by reducing duplicates
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

        # Filler phrases to remove by language - enhanced for Russian
        self.filler_phrases = {
            "en": [
                r'^the\s+', r'^a\s+', r'^an\s+', r'^this\s+', r'^that\s+',
                r'^just\s+', r'^so\s+', r'^only\s+', r'^about\s+', r'^there\s+',
                r'^here\s+', r'^these\s+', r'^those\s+', r'^such\s+', r'^like\s+',
                r'^what\s+', r'^which\s+', r'^where\s+', r'^when\s+', r'^why\s+',
                r'^how\s+', r'^who\s+', r'^I\s+', r'^we\s+', r'^you\s+', r'^it\s+',
                r'\s+is$', r'\s+are$', r'\s+was$', r'\s+were$', r'\s+been$',
                r'\s+can$', r'\s+will$', r'\s+should$', r'\s+could$', r'\s+would$',
                r'\s+have$', r'\s+has$', r'\s+had$'
            ],
            "ru": [
                # Starting phrases (significantly enhanced)
                r'^это\s+', r'^вот\s+', r'^та\s+', r'^тот\s+', r'^те\s+', r'^та\s+',
                r'^такая\s+', r'^такой\s+', r'^такое\s+', r'^такие\s+', r'^просто\s+',
                r'^только\s+', r'^лишь\s+', r'^да\s+', r'^ну\s+', r'^и\s+',
                r'^в\s+', r'^но\s+', r'^на\s+', r'^по\s+', r'^то\s+', r'^у\s+нас\s+',
                r'^мы\s+', r'^я\s+', r'^вы\s+', r'^они\s+', r'^он\s+', r'^она\s+',
                r'^оно\s+', r'^как\s+', r'^что\s+', r'^когда\s+', r'^где\s+',
                r'^давайте\s+', r'^потому\s+', r'^причин\s+', r'^здесь\s+', r'^тут\s+',
                r'^значит\s+', r'^теперь\s+', r'^итак\s+', r'^тогда\s+', r'^дальше\s+',
                r'^там\s+', r'^вообще\s+', r'^кстати\s+', r'^собственно\s+', r'^фактически\s+',
                r'^почему\s+', r'^зачем\s+', r'^чтобы\s+', r'^если\s+', r'^поскольку\s+',
                r'^наверное\s+', r'^наверно\s+', r'^может\s+быть\s+', r'^возможно\s+',
                r'^пожалуй\s+', r'^кажется\s+', r'^действительно\s+',

                # Ending phrases (significantly enhanced)
                r'\s+должна$', r'\s+должен$', r'\s+должно$', r'\s+должны$',
                r'\s+может$', r'\s+могут$', r'\s+будет$', r'\s+будут$', r'\s+было$',
                r'\s+были$', r'\s+есть$', r'\s+имеет$', r'\s+имеют$', r'\s+нужно$',
                r'\s+нужна$', r'\s+надо$', r'\s+необходимо$', r'\s+требуется$',
                r'\s+следует$', r'\s+стоит$', r'\s+хочет$', r'\s+хотят$',
                r'\s+являются$', r'\s+является$', r'\s+представляет$', r'\s+представляют$',
                r'\s+собой$', r'\s+так$', r'\s+вот$', r'\s+просто$', r'\s+только$',
                r'\s+еще$', r'\s+ещё$', r'\s+уже$', r'\s+тоже$', r'\s+также$',
                r'\s+так\s+далее$', r'\s+так\s+далее\s+тому\s+подобное$',
                r'\s+да$', r'\s+нет$', r'\s+конечно$', r'\s+точно$', r'\s+именно$'
            ]
        }

        # Complete phrases to remove - entire matches
        self.complete_phrases = {
            "en": [
                "we have", "we can see", "we can say", "this is", "that is",
                "it is", "it's", "there is", "there are", "we know", "let's",
                "we will", "as we know", "you can see", "you can find", "you know"
            ],
            "ru": [
                "мы имеем", "мы видим", "мы можем", "мы можем видеть", "мы можем сказать",
                "мы знаем", "как мы знаем", "мы будем", "давайте",
                "у нас есть", "у нас будет", "это есть", "это будет", "это значит",
                "это означает", "то есть", "то означает", "то значит", "вот это",
                "да это", "да вот", "ну вот", "ну это", "я думаю", "я считаю",
                "мне кажется", "нам кажется", "нам надо", "нам нужно", "вот так",
                "вот здесь", "вот тут", "вот этот", "вот эта", "вот это", "вот эти",
                "просто так", "просто потому что", "просто надо", "просто нужно",
                "можно видеть", "можно сказать", "можно утверждать", "можно заметить",
                "можно отметить", "можно тогда", "можно так", "можно здесь",
                "вы видите", "вы знаете", "вы можете видеть", "вы можете найти"
            ]
        }

        # Simple conjunctions and prepositions to remove when they're standalone
        self.simple_terms = {
            "en": {"the", "a", "an", "and", "or", "but", "if", "of", "in", "on", "at", "by", "for", "with", "about"},
            "ru": {"и", "или", "но", "если", "в", "на", "под", "над", "при", "у", "для", "о", "об", "к", "от", "из", "до", "с", "со"}
        }

        # Updated similarity thresholds - more strict for better deduplication
        self.similarity_thresholds = {
            "exact_match": 1.0,    # Exact match (after normalization)
            "high_match": 0.92,    # High confidence match (increased from 0.90)
            "medium_match": 0.85,  # Medium confidence match (increased from 0.80)
            "low_match": 0.75      # Low confidence match (increased from 0.70)
        }

        # Updated concept length thresholds
        self.min_concept_length = 3  # Minimum number of characters for a valid concept
        self.min_words = 1           # Minimum number of words for a valid concept
        self.max_words = 6           # Maximum of 6 words (reduced from 7)

        # Increased word overlap threshold for better precision
        self.word_overlap_threshold = 0.6  # Increased from 0.5

        logger.info(f"Concept deduplication extension initialized with language: {language}")

    def normalize_concept_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Enhanced normalization of concept text for better matching and deduplication.

        Args:
            text: Concept text
            language: Language code (defaults to instance language)

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Use specified language or instance language
        lang = language or self.language

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # First, remove complete phrases (whole-phrase matches)
        lang_key = lang if lang in self.complete_phrases else 'en'
        for phrase in self.complete_phrases.get(lang_key, []):
            if normalized == phrase:
                return ""  # Complete match with a filler phrase - invalid concept
            normalized = normalized.replace(phrase, " ")

        # Remove filler phrases based on language
        lang_key = lang if lang in self.filler_phrases else 'en'
        patterns = self.filler_phrases.get(lang_key, [])

        for pattern in patterns:
            normalized = re.sub(pattern, '', normalized)

        # Remove specific filler patterns for multi-word phrases
        if ' ' in normalized:
            # Remove phrases starting with filler verbs/phrases
            verb_prefixes = ['is ', 'are ', 'can ', 'will ', 'has ', 'have ', 'need ', 'should '] if lang == 'en' else \
                            ['является ', 'будет ', 'имеет ', 'нужно ', 'должна ', 'может ', 'хочет ', 'надо ']

            for prefix in verb_prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break

            # Remove filler endings
            verb_suffixes = [' is', ' are', ' be', ' can', ' will'] if lang == 'en' else \
                            [' есть', ' будет', ' имеет', ' должна', ' может', ' надо', ' нужно']

            for suffix in verb_suffixes:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
                    break

        # Remove any remaining leading/trailing whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Final check: if normalized text is just a simple conjunction or preposition, invalidate it
        simple_terms = self.simple_terms.get(lang if lang in self.simple_terms else 'en', set())
        if normalized in simple_terms:
            return ""

        return normalized

    def is_valid_concept(self, text: str, language: Optional[str] = None) -> bool:
        """
        Enhanced check if text represents a valid concept based on length and content.

        Args:
            text: Concept text
            language: Language code

        Returns:
            True if valid concept, False otherwise
        """
        # Normalize and check length
        normalized = self.normalize_concept_text(text, language)

        if not normalized:
            return False

        # Check minimum character length
        if len(normalized) < self.min_concept_length:
            return False

        # Check word count
        word_count = len(normalized.split())

        if word_count < self.min_words or word_count > self.max_words:
            return False

        # Check if it's mostly numbers
        if sum(c.isdigit() for c in normalized) / len(normalized) > 0.3:  # Reduced threshold to 30%
            return False

        # Check if it's a common stopword or filler phrase (too generic)
        lang = language or self.language
        common_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'and', 'or', 'but'} if lang != 'ru' else \
                       {'и', 'или', 'но', 'это', 'этот', 'эти', 'тот', 'те', 'что', 'как', 'да', 'нет'}

        if word_count == 1 and normalized in common_words:
            return False

        # Additional check for invalid Russian concepts
        if lang == 'ru':
            # Single words ending with common verb endings are often not valid concepts
            if word_count == 1 and any(normalized.endswith(suffix) for suffix in
                ['ет', 'ут', 'ют', 'ит', 'ат', 'ят', 'ем', 'им']):
                return False

            # Check for common phrases that aren't valid concepts
            invalid_phrases = [
                'вот так', 'вот это', 'вот тут', 'вот здесь', 'просто так',
                'да вот', 'ну вот', 'ну да', 'ну нет', 'ну ладно',
                'может быть', 'да м'
            ]

            if normalized in invalid_phrases:
                return False

        return True

    def calculate_concept_similarity(self, concept1: str, concept2: str, language: Optional[str] = None) -> float:
        """
        Enhanced calculation of similarity between two concept texts.

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
        Enhanced similar concept detection with more precise matching.

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

    def get_canonical_concept(self, concept: Dict[str, Any], concept_list: List[Dict[str, Any]],
                            language: Optional[str] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Enhanced determination of canonical concept with more sophisticated selection criteria.

        Args:
            concept: Source concept dictionary
            concept_list: List of concept dictionaries to compare against
            language: Language code

        Returns:
            Tuple of (canonical_concept, is_new_canonical)
        """
        # Find similar concepts
        similar_concepts = self.find_similar_concepts(
            concept,
            concept_list,
            threshold=self.similarity_thresholds["low_match"],
            language=language
        )

        if not similar_concepts:
            # No similar concepts found, this is a new canonical concept
            return concept, True

        # Check for high confidence matches
        high_confidence_matches = [c for c in similar_concepts
                                if c.get("similarity", 0) >= self.similarity_thresholds["high_match"]]

        if high_confidence_matches:
            # For high confidence, select best match as canonical
            best_match = high_confidence_matches[0]

            # Enhanced selection logic
            # Calculate quality scores for both concepts
            concept_quality = self._calculate_concept_quality(concept, language)
            match_quality = self._calculate_concept_quality(best_match, language)

            # Compare quality scores with a preference for the current concept
            # (to avoid too many redirects to existing concepts)
            if concept_quality > match_quality * 1.2:
                # This concept is significantly better, use it as canonical
                return concept, True

            # Use existing as canonical
            return best_match, False

        # For medium confidence, apply more sophisticated selection
        medium_matches = [c for c in similar_concepts
                        if c.get("similarity", 0) >= self.similarity_thresholds["medium_match"]]

        if medium_matches:
            # Select canonical based on multiple factors
            candidates = [concept] + medium_matches

            # Calculate quality score for each candidate
            candidates_with_scores = [(c, self._calculate_concept_quality(c, language)) for c in candidates]

            # Sort by quality score
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Return best candidate and whether it's the original concept
            best_candidate = candidates_with_scores[0][0]
            return best_candidate, best_candidate == concept

        # No strong matches, use this concept as canonical
        return concept, True

    def _calculate_concept_quality(self, concept: Dict[str, Any], language: Optional[str] = None) -> float:
        """
        Calculate a quality score for a concept based on multiple factors.

        Args:
            concept: Concept dictionary
            language: Language code

        Returns:
            Quality score (higher is better)
        """
        concept_text = concept.get("text", "")
        normalized_text = self.normalize_concept_text(concept_text, language)

        if not normalized_text:
            return 0.0

        # Word count score (concepts with 2-3 words tend to be best)
        word_count = len(normalized_text.split())
        if word_count == 2 or word_count == 3:
            word_count_score = 1.0
        elif word_count == 1:
            word_count_score = 0.7
        elif word_count == 4:
            word_count_score = 0.8
        else:
            word_count_score = 0.5

        # Frequency score (more occurrences is better)
        freq = concept.get("frequency", concept.get("total_occurrences", concept.get("occurrence_count", 1)))
        freq_score = min(freq / 5, 1.0)  # Normalize frequency (max 5)

        # Confidence/score from concept extraction
        conf_score = min(concept.get("score", concept.get("confidence", 0)) / 10, 1.0)

        # Canonical preference (prefer concepts that are already canonical)
        canonical_score = 0.3 if not concept.get("canonical_concept_id") else 0.0

        # Video count score (concepts appearing in more videos are better)
        video_count = concept.get("video_count", 1)
        video_score = min(video_count / 3, 1.0)  # Normalize video count (max 3)

        # Combined quality score with weightings
        return (
            word_count_score * 0.3 +
            freq_score * 0.2 +
            conf_score * 0.15 +
            canonical_score * 0.15 +
            video_score * 0.2
        )

    def deduplicate_concepts(self, concepts: List[Dict[str, Any]], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced concept deduplication with better handling of near-duplicate concepts.
        Improves text-based deduplication to prevent duplicates with same text but different IDs.

        Args:
            concepts: List of concept dictionaries
            language: Language code

        Returns:
            Deduplicated list of concepts with canonical relationships
        """
        if not concepts:
            return []

        # First, filter out invalid concepts
        lang = language or next((c.get("language") for c in concepts if "language" in c), self.language)

        valid_concepts = [c for c in concepts if self.is_valid_concept(c.get("text", ""), lang)]

        # Important fix: If all concepts were filtered out, return the original concepts as valid
        # This prevents excessive filtering in some languages
        if not valid_concepts and concepts:
            logger.warning(f"All concepts were filtered out as invalid. Returning original concepts.")
            valid_concepts = concepts.copy()

        if not valid_concepts:
            return []

        # Initialize result containers
        canonical_concepts = []  # List of canonical concepts
        canonical_map = {}       # Map of concept_id -> canonical_concept_id
        processed_texts = set()  # Set of normalized texts already processed

        # Keep track of concept texts and their canonical representations
        text_to_canonical = {}   # Map of normalized_text -> canonical concept

        # Process concepts in order of quality score (higher first)
        # This ensures better concepts become canonical
        sorted_concepts = sorted(
            valid_concepts,
            key=lambda x: self._calculate_concept_quality(x, lang),
            reverse=True
        )

        for concept in sorted_concepts:
            concept_text = concept.get("text", "")
            concept_id = concept.get("concept_id", concept.get("id", ""))

            # Skip if no text or ID
            if not concept_text or not concept_id:
                continue

            # Normalize text for better matching
            normalized_text = self.normalize_concept_text(concept_text, lang)

            # Skip invalid concepts after normalization - BUT only if normalization returned empty string
            if normalized_text == "":
                continue

            # Check if we've already processed an exact text match
            if normalized_text in processed_texts:
                # This is a duplicate - find which canonical concept it belongs to
                if normalized_text in text_to_canonical:
                    canonical_id = text_to_canonical[normalized_text].get("concept_id")
                    canonical_map[concept_id] = canonical_id
                    concept["canonical_concept_id"] = canonical_id
                    logger.debug(f"Text-based duplicate: '{concept_text}' -> canonical '{text_to_canonical[normalized_text].get('text')}'")
                continue

            # STEP 1: Check for similar existing concepts before creating a new one
            similar_concepts = self.find_similar_concepts(concept, canonical_concepts, lang)

            canonical_concept_id = None

            if similar_concepts:
                # Get the best matching concept
                best_match = similar_concepts[0]

                # If we have a very similar match, use its ID as canonical
                match_score = best_match.get('similarity', 0)

                if match_score > 0.85:
                    # This is essentially the same concept, use the existing one as canonical
                    canonical_concept_id = best_match.get('concept_id')
                    logger.info(f"Using canonical concept {canonical_concept_id} for similar concept: '{concept_text}'")

                    # Mark as processed
                    processed_texts.add(normalized_text)

                    # Map this concept to its canonical
                    canonical_map[concept_id] = canonical_concept_id
                    concept["canonical_concept_id"] = canonical_concept_id

                    continue

            # This is a new canonical concept
            # Clone and add metadata
            canonical_concept = concept.copy()

            # Ensure it has normalized_text field
            if "normalized_text" not in canonical_concept:
                canonical_concept["normalized_text"] = normalized_text

            # Add to canonical concepts list
            canonical_concepts.append(canonical_concept)

            # Record normalized text as processed
            processed_texts.add(normalized_text)

            # Map this concept to itself (it is canonical)
            if concept_id:
                canonical_map[concept_id] = concept_id

            # Store mapping from text to canonical concept
            text_to_canonical[normalized_text] = canonical_concept

        # Return canonical concepts
        return canonical_concepts

    def apply_concept_deduplication(self, processed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply enhanced concept deduplication to a processed video result.

        Args:
            processed_result: Video processing result dictionary

        Returns:
            Updated processing result
        """
        if not processed_result:
            return processed_result

        # Extract domain features and language
        domain_features = processed_result.get("domain_features", {})
        language = processed_result.get("transcript", {}).get("language", self.language)

        # Extract concepts
        key_concepts = domain_features.get("key_concepts", [])
        theoretical_concepts = domain_features.get("theoretical_concepts", [])
        practical_concepts = domain_features.get("practical_concepts", [])

        if not key_concepts:
            logger.info("No concepts to deduplicate")
            return processed_result  # Nothing to deduplicate

        # First pass: filter out invalid concepts
        filtered_key_concepts = [c for c in key_concepts if self.is_valid_concept(c.get("text", ""), language)]

        # Important: If all were filtered, keep original concepts
        if not filtered_key_concepts and key_concepts:
            logger.warning(f"All {len(key_concepts)} concepts would be filtered out. Keeping original concepts.")
            filtered_key_concepts = key_concepts

        # Log number of filtered concepts
        num_filtered = len(key_concepts) - len(filtered_key_concepts)
        if num_filtered > 0:
            logger.info(f"Filtered out {num_filtered} invalid concepts")

        # Second pass: deduplicate the remaining concepts
        canonical_concepts = self.deduplicate_concepts(filtered_key_concepts, language)

        # If deduplication resulted in 0 concepts, keep the original filtered concepts
        if not canonical_concepts and filtered_key_concepts:
            logger.warning("Deduplication resulted in 0 concepts. Keeping original filtered concepts.")
            canonical_concepts = filtered_key_concepts

        # Create mapping from original text to canonical concept
        text_to_canonical = {}
        canonical_ids = set()

        for concept in filtered_key_concepts:
            concept_text = concept.get("text", "").lower()
            concept_id = concept.get("concept_id", concept.get("id", ""))
            canonical_id = concept.get("canonical_concept_id")

            # Add all canonical IDs to a set for quick lookup
            if canonical_id:
                canonical_ids.add(canonical_id)

                # Find the canonical concept
                canonical = next((c for c in canonical_concepts
                            if c.get("concept_id", c.get("id", "")) == canonical_id), None)

                if canonical:
                    text_to_canonical[concept_text] = canonical

        # Filter theoretical and practical concepts based on canonical relationships
        deduplicated_theoretical = self.deduplicate_concepts(theoretical_concepts, language)
        deduplicated_practical = self.deduplicate_concepts(practical_concepts, language)

        # If deduplication resulted in empty lists, use original lists
        if not deduplicated_theoretical and theoretical_concepts:
            deduplicated_theoretical = theoretical_concepts

        if not deduplicated_practical and practical_concepts:
            deduplicated_practical = practical_concepts

        # Update domain features
        domain_features["key_concepts"] = canonical_concepts
        domain_features["theoretical_concepts"] = deduplicated_theoretical
        domain_features["practical_concepts"] = deduplicated_practical

        # Update processed result
        processed_result["domain_features"] = domain_features

        # Log deduplication results
        logger.info(f"Deduplicated from {len(filtered_key_concepts)} to {len(canonical_concepts)} canonical concepts")
        logger.info(f"Established {len(canonical_ids)} canonical relationships")

        return processed_result
