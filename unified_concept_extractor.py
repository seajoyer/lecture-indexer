"""
Enhanced concept extractor for the Lecture Video Content Indexer.
Implements improved concept matching with morphological variant support for better recognition,
especially for Russian language terms with different grammatical forms.
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

class MorphologicalVariantMatcher:
    """
    Helper class for matching concept variants with morphological differences.
    Particularly useful for languages with rich morphology like Russian.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the variant matcher.

        Args:
            language: Language code ('en' or 'ru')
        """
        self.language = language
        self._init_language_resources()

        # Cache for variant generation to avoid redundant processing
        self._variant_cache = {}  # {(text, language): set of variants}

        # Hard limit to prevent variant explosion for Russian
        self.ru_max_variants = 5
        self.ru_max_phrase_length = 3  # Max words for Russian variant generation

    def _init_language_resources(self):
        """Initialize language-specific resources for variant matching."""
        # OPTIMIZATION: Reduce the number of Russian endings to the most common ones only
        # This dramatically reduces the number of variants generated
        self.russian_endings = {
            # Most common noun endings (simplified)
            'ие': ['ия', 'ий'],  # Focus on most frequent cases only
            'ия': ['ие', 'ий'],
            'ть': ['ти'],
            'ость': ['ости'],
            'а': ['ы'],  # Reduce variants
            'я': ['и'],  # Reduce variants
        }

        # English plurals and verb forms
        self.english_endings = {
            's': [''],  # plural -> singular
            '': ['s'],  # singular -> plural
            'ing': ['', 'e'],  # running -> run, run
            'ed': ['', 'e']    # played -> play, play
        }

    def generate_variants(self, text: str) -> Set[str]:
        """
        Generate possible morphological variants of a term.
        Now with caching for efficiency.

        Args:
            text: The original text

        Returns:
            Set of possible variants
        """
        # Check cache first
        cache_key = (text, self.language)
        if cache_key in self._variant_cache:
            return self._variant_cache[cache_key]

        variants = {text}  # Always include the original

        # OPTIMIZATION: Fast early returns for Russian
        if self.language == 'ru':
            # Skip variant generation for long Russian phrases completely
            word_count = len(text.split())
            if word_count > self.ru_max_phrase_length:
                self._variant_cache[cache_key] = variants
                return variants

            # Skip very short words or common words
            if len(text) <= 3 or text in {"от", "до", "на", "по", "за", "из", "под", "над", "при", "для",
                                        "и", "а", "но", "или", "что", "как", "так", "где", "кто"}:
                self._variant_cache[cache_key] = variants
                return variants

        # Quick return for non-target languages or very short text
        if self.language not in ['ru', 'en'] or len(text) < 3:
            self._variant_cache[cache_key] = variants
            return variants

        if self.language == 'ru':
            variants.update(self._generate_russian_variants_optimized(text))
        else:  # Default to English
            variants.update(self._generate_english_variants(text))

        # Store in cache
        self._variant_cache[cache_key] = variants
        return variants

    def _generate_russian_variants_optimized(self, text: str) -> Set[str]:
        """
        Generate Russian morphological variants with optimizations.

        Args:
            text: Text to generate variants for

        Returns:
            Set of variants
        """
        variants = set()
        words = text.split()

        # OPTIMIZATION: Hard limit for Russian - skip variant generation for longer phrases completely
        if len(words) > self.ru_max_phrase_length:
            return {text}  # Just return original text

        # OPTIMIZATION: For Russian, only process the last word in multi-word phrases
        # This dramatically reduces the combinatorial explosion
        if len(words) > 1:
            # Process just the last word
            last_word = words[-1]
            last_variants = self._generate_russian_word_variants_optimized(last_word)

            # Replace last word with each variant
            prefix = ' '.join(words[:-1])
            for variant in last_variants:
                if variant != last_word:  # Skip original word
                    variants.add(f"{prefix} {variant}")

            return variants

        # For single words, generate variants directly (with hard limit)
        if len(words) == 1:
            return self._generate_russian_word_variants_optimized(words[0])

        return variants

    def _generate_russian_word_variants_optimized(self, word: str) -> Set[str]:
        """
        Generate common Russian morphological variants for a single word.
        Optimized to focus on most productive endings.

        Args:
            word: Russian word

        Returns:
            Set of possible variants
        """
        variants = {word}

        # OPTIMIZATION: Skip very short words and common prepositions/conjunctions
        if len(word) <= 3 or word in {"от", "до", "на", "по", "за", "из", "под", "над", "при", "для",
                                     "и", "а", "но", "или", "что", "как", "так", "где", "кто"}:
            return variants

        # OPTIMIZATION: Hard limit on number of variants for Russian
        # Stop when we reach the maximum number of variants
        variant_count = 0

        # Optimize by focusing on the most productive endings
        for ending, replacements in self.russian_endings.items():
            if word.endswith(ending) and len(word) > len(ending) + 2:
                # Limit the number of replacements
                for replacement in replacements:
                    if variant_count >= self.ru_max_variants:
                        return variants

                    variant = word[:-len(ending)] + replacement
                    variants.add(variant)
                    variant_count += 1

                # Once we've found a matching ending, don't check others
                break

        return variants

    def _generate_english_variants(self, text: str) -> Set[str]:
        """
        Generate English morphological variants.

        Args:
            text: English text

        Returns:
            Set of possible variants
        """
        variants = set()
        words = text.split()

        # Simple handling for multi-word phrases - just add variants with/without "the", "a", "an"
        if len(words) > 1:
            # Remove leading articles if present
            if words[0].lower() in {'the', 'a', 'an'}:
                variants.add(' '.join(words[1:]))
            # Add versions with articles if not present
            elif words[0].lower() not in {'the', 'a', 'an'}:
                variants.add('the ' + text)
                if words[0][0].lower() in 'aeiou':
                    variants.add('an ' + text)
                else:
                    variants.add('a ' + text)

        # For single words or multi-word phrases, handle common morphological changes
        for word in words:
            if len(word) <= 3:
                continue  # Skip very short words

            # Common English endings - handle plurals and word forms
            if word.endswith('s') and len(word) > 4:
                variants.add(text.replace(word, word[:-1]))  # Remove trailing 's'
            elif not word.endswith('s') and len(word) > 3:
                variants.add(text.replace(word, word + 's'))  # Add trailing 's'

            # Handle common verb forms
            if word.endswith('ing') and len(word) > 5:
                variants.add(text.replace(word, word[:-3]))  # Remove 'ing'
                variants.add(text.replace(word, word[:-3] + 'e'))  # Remove 'ing', add 'e'
            elif word.endswith('ed') and len(word) > 4:
                variants.add(text.replace(word, word[:-2]))  # Remove 'ed'
                variants.add(text.replace(word, word[:-1]))  # Remove 'd'
                variants.add(text.replace(word, word[:-2] + 'e'))  # Remove 'ed', add 'e'

        return variants

    def match_variants(self, text: str, target: str) -> float:
        """
        Check if text matches any variant of the target.
        Optimized for performance.

        Args:
            text: Text to check
            target: Target concept

        Returns:
            Similarity score (0.0-1.0), 1.0 if exact match
        """
        # If exact match, return perfect score
        if text.lower() == target.lower():
            return 1.0

        # OPTIMIZATION: Skip expensive processing for Russian text completely
        # For Russian, just check direct substring match instead of variant matching
        if self.language == 'ru':
            text_lower = text.lower()
            target_lower = target.lower()

            # Simple substring check - much faster than variant generation
            if text_lower in target_lower or target_lower in text_lower:
                return 0.8
            return 0.0

        # Skip expensive processing for long texts or non-target languages
        if self.language not in ['ru', 'en'] or len(text) > 30 or len(target) > 30:
            return 0.0

        # Generate variants of both texts (uses cache internally)
        text_variants = self.generate_variants(text.lower())
        target_variants = self.generate_variants(target.lower())

        # Check for matches
        for text_var in text_variants:
            for target_var in target_variants:
                if text_var == target_var:
                    return 0.95  # Very high but not perfect score for variant matches

        # If words have common stems but different endings, return medium score
        text_words = text.lower().split()
        target_words = target.lower().split()

        if len(text_words) == len(target_words):
            # Check if all words except the last one match
            if len(text_words) > 1 and text_words[:-1] == target_words[:-1]:
                # Check if last words are morphological variants
                last_text = text_words[-1]
                last_target = target_words[-1]

                # Check if they share a common stem (3+ characters)
                min_len = min(len(last_text), len(last_target))
                stem_length = min(min_len - 2, 5)  # Use up to 5 chars but leave at least 2 for ending

                if stem_length > 2 and last_text[:stem_length] == last_target[:stem_length]:
                    return 0.85  # Good match score for different forms of the same word

        # No variant match found
        return 0.0

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

        # Initialize variant matcher for morphological matching
        self.variant_matcher = MorphologicalVariantMatcher(language)

        # Cache for processed results to avoid redundant computations
        self._text_match_cache = {}

        # Load enhanced NLP resources
        self._load_nlp_resources()

        # Domain classification patterns - for verifying domain context matches
        self._init_domain_patterns()

        # OPTIMIZATION: Add language-specific thresholds to reduce processing for Russian
        self.matching_thresholds = {
            'en': 0.80,  # Default English threshold
            'ru': 0.95   # Higher threshold for Russian to reduce false positives
        }

        # OPTIMIZATION: Maximum concept candidates to process
        self.max_candidates = {
            'en': 1000,  # Default limit for English
            'ru': 100    # Much lower limit for Russian
        }

        logger.info(f"UnifiedConceptExtractor initialized for language: {language}")

    def _init_domain_patterns(self):
        """Initialize domain classification patterns for context validation."""
        # Domain-specific terms and phrases for verification
        self.domain_terms = {
            "physics": {
                "en": [
                    "quantum", "mechanics", "energy", "particle", "wave", "field",
                    "relativity", "force", "matter", "momentum", "electron", "photon",
                    "nucleus", "atom", "molecule", "quark", "boson", "fermion",
                    "hamiltonian", "lagrangian", "symmetry", "conservation", "entropy",
                    "thermodynamics", "electromagnetism", "nuclear", "radiation"
                ],
                "ru": [
                    "квантовый", "механика", "энергия", "частица", "волна", "поле",
                    "относительность", "сила", "материя", "импульс", "электрон", "фотон",
                    "ядро", "атом", "молекула", "кварк", "бозон", "фермион",
                    "гамильтониан", "лагранжиан", "симметрия", "сохранение", "энтропия",
                    "термодинамика", "электромагнетизм", "ядерный", "излучение"
                ]
            },
            "mathematics": {
                "en": [
                    "algebra", "calculus", "geometry", "topology", "function", "theorem",
                    "proof", "equation", "matrix", "vector", "integral", "derivative",
                    "differential", "series", "convergence", "set", "group", "ring",
                    "field", "space", "manifold", "probability", "statistics"
                ],
                "ru": [
                    "алгебра", "анализ", "геометрия", "топология", "функция", "теорема",
                    "доказательство", "уравнение", "матрица", "вектор", "интеграл", "производная",
                    "дифференциал", "ряд", "сходимость", "множество", "группа", "кольцо",
                    "поле", "пространство", "многообразие", "вероятность", "статистика"
                ]
            },
            "programming": {
                "en": [
                    "algorithm", "programming", "code", "function", "variable", "class",
                    "object", "method", "interface", "inheritance", "polymorphism",
                    "compiler", "interpreter", "syntax", "semantics", "library",
                    "framework", "api", "database", "query", "server", "client"
                ],
                "ru": [
                    "алгоритм", "программирование", "код", "функция", "переменная", "класс",
                    "объект", "метод", "интерфейс", "наследование", "полиморфизм",
                    "компилятор", "интерпретатор", "синтаксис", "семантика", "библиотека",
                    "фреймворк", "апи", "база данных", "запрос", "сервер", "клиент"
                ]
            }
        }

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
        # Update variant matcher language
        self.variant_matcher.language = language

        # Clear the text match cache for a new transcript
        self._text_match_cache = {}

        # Get the detected domain
        domain = processed_transcript.get("domain", global_analysis.get("domain", "unknown"))

        # Log input information
        logger.info(f"Extracting concepts from transcript: video_id={video_id}, language={language}, domain={domain}, segments={len(segments)}")

        # OPTIMIZATION: For Russian, use a faster, simplified approach
        if language == 'ru':
            result = self._extract_concepts_from_segments_russian(
                segments,
                video_id,
                language,
                domain,
                global_analysis
            )
        else:
            # Standard approach for English and other languages
            start_time = time.time()
            result = self.extract_concepts_from_segments(
                segments,
                video_id,
                language,
                domain,
                global_analysis
            )
            processing_time = time.time() - start_time

            # Add detailed debugging to check concepts and occurrences
            concepts = result.get("concepts", [])
            educational_concepts = sum(1 for c in concepts if c.get("is_educational", False))
            passing_concepts = len(concepts) - educational_concepts

            total_occurrences = sum(len(c.get("occurrences", [])) for c in concepts)

            logger.info(f"Extraction complete: {len(concepts)} concepts found ({educational_concepts} educational, {passing_concepts} passing) in {processing_time:.2f}s")
            logger.info(f"Total occurrences: {total_occurrences}")

        return result

    def _extract_concepts_from_segments_russian(
        self,
        segments: List[Dict[str, Any]],
        video_id: str,
        language: str,
        domain: Optional[str] = None,
        global_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Optimized function specifically for Russian language processing.
        Uses simplified matching and analysis to improve performance.

        Args:
            segments: List of transcript segments
            video_id: Video ID
            language: Language code (should be 'ru')
            domain: Optional domain filter
            global_analysis: Optional global text analysis data

        Returns:
            Dictionary containing concepts with occurrences
        """
        # Check if concept repository is available
        if not self.concept_repository:
            logger.warning("Concept repository not available for matching")
            return {"concepts": []}

        start_time = time.time()
        logger.info(f"Using optimized Russian concept extraction for video {video_id}")

        # Track matched concepts with their occurrences
        matched_concepts = {}

        # OPTIMIZATION: Only get concepts for Russian language to limit search space
        ru_concepts = self.concept_repository.list_concepts(language='ru', limit=100)

        # OPTIMIZATION: Extract concept texts for direct matching (avoid variant generation completely)
        concept_texts = {}  # concept_id -> list of representations
        for concept in ru_concepts:
            concept_id = concept.get('concept_id')
            if not concept_id:
                continue

            representations = concept.get('representations', {}).get('ru', [])
            if representations:
                concept_texts[concept_id] = [r.lower() for r in representations]

        # Fast search function using simple text matching instead of variant generation
        def fast_match_concept(text, concepts_dict):
            matches = []
            text_lower = text.lower()

            for concept_id, texts in concepts_dict.items():
                for concept_text in texts:
                    # Direct string matching - much faster than variant generation
                    if concept_text in text_lower:
                        concept = self.concept_repository.get_concept(concept_id)
                        if concept:
                            matches.append({
                                'concept_id': concept_id,
                                'concept': concept,
                                'similarity': 0.9,
                                'match_type': 'direct',
                                'matched_representation': concept_text
                            })
                            break  # Only need one match per concept

            return matches

        # Process segments in batches for better performance
        batch_size = 20
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]

            for segment in batch:
                segment_id = segment.get("id", str(uuid.uuid4()))
                segment_text = segment.get("text", "")
                start_time_sec = segment.get("start_time", 0.0)

                if not segment_text.strip():
                    continue

                # Fast concept matching
                matches = fast_match_concept(segment_text, concept_texts)

                # Process each matched concept
                for match in matches:
                    concept_id = match.get("concept_id")
                    concept = match.get("concept")
                    matched_text = match.get("matched_representation", "")

                    # Skip invalid matches
                    if not concept_id or not concept:
                        continue

                    # Calculate simplified educational significance - faster than full calculation
                    educational_value = min(2.0 + (len(segment_text.split()) / 30), 4.0)

                    # Determine occurrence type based on significance
                    occurrence_type = "comprehensive" if educational_value >= 2.5 else "passing"

                    # Create occurrence record
                    occurrence = {
                        "occurrence_id": str(uuid.uuid4()),
                        "concept_id": concept_id,
                        "video_id": video_id,
                        "segment_id": segment_id,
                        "start_time": start_time_sec,
                        "educational_significance": educational_value,
                        "occurrence_type": occurrence_type,
                        "similarity": 0.9,  # Fixed similarity for fast processing
                        "context_text": segment_text,
                        "matched_variant": matched_text
                    }

                    # Add to matched concepts
                    if concept_id not in matched_concepts:
                        # Create concept entry
                        matched_concepts[concept_id] = {
                            "concept_id": concept_id,
                            "text": matched_text,
                            "representations": concept.get("representations", {}),
                            "language": language,
                            "domain": concept.get("metadata", {}).get("domain", "unknown"),
                            "occurrences": [occurrence],
                            "educational_significance": educational_value,
                            "is_educational": educational_value >= 2.5
                        }
                    else:
                        # Update existing concept
                        matched_concepts[concept_id]["occurrences"].append(occurrence)

                        # Update significance if higher
                        if educational_value > matched_concepts[concept_id]["educational_significance"]:
                            matched_concepts[concept_id]["educational_significance"] = educational_value
                            matched_concepts[concept_id]["is_educational"] = educational_value >= 2.5

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

        logger.info(f"Optimized Russian extraction: found {total_concepts} concepts ({educational_concepts} educational, {passing_concepts} passing) in {processing_time:.2f}s")

        # Return the concepts
        return {
            "concepts": result_concepts,
            "educational_concepts_count": educational_concepts,
            "passing_concepts_count": passing_concepts
        }

    def extract_concepts_from_segments(
        self,
        segments: List[Dict[str, Any]],
        video_id: str,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        global_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from transcript segments by matching against the concept repository.
        Optimized version with early filtering for Russian.

        Args:
            segments: List of transcript segments
            video_id: Video ID
            language: Optional language filter
            domain: Optional domain filter
            global_analysis: Optional global text analysis data

        Returns:
            Dictionary containing concepts with occurrences
        """
        # Use specified language or default
        lang = language or self.language
        # Update variant matcher language
        self.variant_matcher.language = lang

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
        logger.info(f"Finding matching concepts for video {video_id} in language: {lang}, domain: {domain}")

        # Track matched concepts with their occurrences
        matched_concepts = {}

        # OPTIMIZATION: For Russian, pre-categorize segments by length
        # This allows us to apply different processing strategies based on segment complexity
        if lang == 'ru':
            # Group segments by length to process shorter ones first and more thoroughly
            short_segments = []  # 1-10 words
            medium_segments = []  # 11-30 words
            long_segments = []   # 31+ words

            for segment in segments:
                segment_text = segment.get("text", "")
                word_count = len(segment_text.split())

                if word_count <= 10:
                    short_segments.append(segment)
                elif word_count <= 30:
                    medium_segments.append(segment)
                else:
                    long_segments.append(segment)

            # Process short segments first and most thoroughly
            self._process_segment_batch(short_segments, matched_concepts, lang, domain, video_id, 0.85)

            # Process medium segments with slightly higher threshold
            self._process_segment_batch(medium_segments, matched_concepts, lang, domain, video_id, 0.9)

            # Process long segments with simplified matching (higher threshold)
            # For very long segments, we'll be more strict to avoid false positives
            self._process_segment_batch(long_segments, matched_concepts, lang, domain, video_id, 0.95)
        else:
            # For non-Russian languages, process all segments normally
            self._process_segment_batch(segments, matched_concepts, lang, domain, video_id, 0.85)

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

    def _process_segment_batch(
        self,
        segments: List[Dict[str, Any]],
        matched_concepts: Dict[str, Dict[str, Any]],
        language: str,
        domain: Optional[str],
        video_id: str,
        threshold: float
    ) -> None:
        """
        Process a batch of segments to extract concepts.

        Args:
            segments: List of segments to process
            matched_concepts: Dictionary to store matched concepts (modified in place)
            language: Language code
            domain: Optional domain filter
            video_id: Video ID
            threshold: Similarity threshold for concept matching
        """
        # OPTIMIZATION: For Russian, use a higher threshold
        if language == 'ru':
            threshold = max(threshold, self.matching_thresholds['ru'])  # Always use at least the minimum Russian threshold

        for segment in segments:
            segment_id = segment.get("id", str(uuid.uuid4()))
            segment_text = segment.get("text", "")
            start_time_sec = segment.get("start_time", 0.0)

            if not segment_text.strip():
                continue

            # Find matching concepts in this segment using enhanced morphological matching
            segment_matches = self._find_matching_concepts_in_text(
                segment_text,
                language=language,
                domain=domain,
                threshold=threshold
            )

            # Process each matched concept
            for match in segment_matches:
                concept_id = match.get("concept_id")
                concept = match.get("concept")
                similarity = match.get("similarity", 0.0)
                concept_domain = match.get("domain", "unknown")

                # Record the actual matched variant from the text
                matched_variant = match.get("matched_variant", match.get("matched_representation", ""))

                if not concept_id or not concept:
                    continue

                # Skip if domains don't match and we have domain information
                if domain and concept_domain and domain != concept_domain and domain != "unknown" and concept_domain != "unknown":
                    # Only combine domains within certain subject groups
                    science_domains = {"physics", "mathematics"}
                    computer_domains = {"programming", "computer_science"}

                    # Allow matching between related domains
                    domain_match = (
                        (domain in science_domains and concept_domain in science_domains) or
                        (domain in computer_domains and concept_domain in computer_domains)
                    )

                    if not domain_match:
                        logger.debug(f"Skipping concept {concept_id} due to domain mismatch: {domain} vs {concept_domain}")
                        continue

                # Verify the match with domain context validation
                if not self._validate_concept_in_context(segment_text, concept, domain):
                    logger.debug(f"Skipping concept {concept_id} - failed context validation")
                    continue

                # Calculate educational significance
                educational_significance = self._calculate_educational_significance(
                    segment,
                    concept,
                    matched_variant,
                    None  # Omit global_analysis to improve performance
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
                    "context_text": segment_text,
                    "matched_variant": matched_variant  # Store the actual variant that was matched
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
                        "language": language,
                        "domain": concept_domain,
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

    def _find_matching_concepts_in_text(self, text: str, language: str, domain: Optional[str] = None, threshold: float = 0.85) -> List[Dict]:
        """
        Find matching concepts in text using the concept repository with enhanced morphological matching.
        Optimized for performance with early filtering.

        Args:
            text: Text to search in
            language: Language code
            domain: Optional domain filter
            threshold: Similarity threshold

        Returns:
            List of matching concept dictionaries
        """
        # Use concept repository to find matches

        # OPTIMIZATION: For Russian, limit processing of very long texts
        if language == 'ru':
            # For very long Russian texts, increase threshold to reduce false positives
            threshold = max(threshold, self.matching_thresholds['ru'])

            # Limit the number of results to avoid excessive processing
            max_results = 3
        else:
            max_results = 5

        # Check cache first (to avoid redundant repository searches)
        cache_key = (text, language, domain, threshold)
        if cache_key in self._text_match_cache:
            return self._text_match_cache[cache_key]

        # OPTIMIZATION: For Russian, use a much simpler word-boundary based exact matching
        # instead of complex fuzzy matching to improve performance
        if language == 'ru':
            matches = []

            # Get Russian concepts (limited number)
            ru_concepts = self.concept_repository.list_concepts(
                language='ru',
                limit=self.max_candidates['ru']  # Limited number of concepts for Russian
            )

            # Do direct text matching without variant generation
            text_lower = text.lower()

            for concept in ru_concepts:
                concept_id = concept.get('concept_id')
                representations = concept.get('representations', {}).get('ru', [])

                for rep in representations:
                    # Try direct word boundary matching
                    rep_lower = rep.lower()
                    pattern = r'\b' + re.escape(rep_lower) + r'\b'

                    if re.search(pattern, text_lower):
                        matches.append({
                            "concept_id": concept_id,
                            "concept": concept,
                            "similarity": 0.95,  # High similarity for direct matches
                            "match_type": "direct",
                            "matched_representation": rep,
                            "matched_variant": rep,
                            "domain": concept.get("metadata", {}).get("domain", "unknown")
                        })
                        break  # One match per concept is enough

                    # If no word boundary match, try substring match with minimum length check
                    elif len(rep_lower) > 5 and rep_lower in text_lower:
                        matches.append({
                            "concept_id": concept_id,
                            "concept": concept,
                            "similarity": 0.85,  # Lower similarity for substring matches
                            "match_type": "substring",
                            "matched_representation": rep,
                            "matched_variant": rep,
                            "domain": concept.get("metadata", {}).get("domain", "unknown")
                        })
                        break  # One match per concept is enough

            # Store results in cache
            self._text_match_cache[cache_key] = matches

            return matches
        else:
            # Standard repository search for non-Russian languages
            matches = self.concept_repository.find_concepts_by_text(
                text,
                language=language,
                threshold=threshold,
                max_results=max_results
            )

        # Filter matches by domain if specified
        if domain and domain != "unknown":
            filtered_matches = []
            for match in matches:
                concept = match.get("concept", {})
                concept_domain = concept.get("metadata", {}).get("domain", "unknown")

                # Allow matching within subject groups
                science_domains = {"physics", "mathematics"}
                computer_domains = {"programming", "computer_science"}

                # Check if domains match or are in the same subject group
                domain_match = (
                    domain == concept_domain or
                    concept_domain == "unknown" or
                    (domain in science_domains and concept_domain in science_domains) or
                    (domain in computer_domains and concept_domain in computer_domains)
                )

                if domain_match:
                    # Add domain info to the match for later use
                    match["domain"] = concept_domain
                    filtered_matches.append(match)
                else:
                    logger.debug(f"Filtered out concept with domain {concept_domain} (video domain: {domain})")

            matches = filtered_matches

        # If no matches from standard repository search, try morphological variant matching
        # OPTIMIZATION: Skip variant matching for Russian
        if not matches and language == 'en':
            matches = self._find_morphological_variants_in_text(text, language, domain, threshold)

        # Store in cache
        self._text_match_cache[cache_key] = matches

        return matches

    def _validate_concept_in_context(self, text: str, concept: Dict[str, Any], domain: Optional[str] = None) -> bool:
        """
        Validate that the concept actually belongs in the text context by checking
        for domain-specific context terms.

        Args:
            text: Text segment to validate
            concept: Concept dictionary
            domain: Optional domain filter

        Returns:
            True if the concept is validated in context, False otherwise
        """
        # OPTIMIZATION: Skip validation for Russian to improve performance
        if self.language == 'ru':
            return True

        # If we don't have domain information, we can't validate
        if not domain or domain == "unknown":
            # Try to get domain from concept
            concept_domain = concept.get("metadata", {}).get("domain", "unknown")
            if concept_domain == "unknown":
                # Without domain info, we rely on other checks
                return True
            domain = concept_domain

        # Get domain terms for this domain
        domain_terms = self.domain_terms.get(domain, {}).get(self.language, [])

        # If we don't have terms for this domain, assume it's valid
        if not domain_terms:
            return True

        # Check if at least one domain term is present in the text
        text_lower = text.lower()

        # Count domain terms in the text
        term_count = sum(1 for term in domain_terms if term in text_lower)

        # For physics concepts, require at least 2 domain terms to reduce false positives
        if domain == "physics":
            # If the concept is a common English word, require more domain context
            concept_text = concept.get("representations", {}).get("en", [""])[0]
            if concept_text and len(concept_text) <= 5:  # Short common words need more verification
                # Require at least 2 domain terms for short physics concept words
                return term_count >= 2

            # For longer physics concepts, still require at least 1 domain term
            return term_count >= 1

        # For other domains, require at least 1 domain term
        return term_count >= 1

    def _find_morphological_variants_in_text(self, text: str, language: str, domain: Optional[str] = None, threshold: float = 0.85) -> List[Dict]:
        """
        Find concepts by checking for morphological variants in the text.
        This helps match concepts even when they appear in different grammatical forms.
        Optimized for performance with early exits for complex text.

        Args:
            text: Text to search in
            language: Language code
            domain: Optional domain filter
            threshold: Similarity threshold

        Returns:
            List of matching concept dictionaries
        """
        matches = []

        # Skip if no concept repository available
        if not self.concept_repository:
            return matches

        # OPTIMIZATION: Skip completely for Russian
        if language == 'ru':
            return []

        # Get all concepts for the given language
        concepts = self.concept_repository.list_concepts(language=language, limit=self.max_candidates[language])

        # For each concept, check if any of its variants appears in the text
        for concept in concepts:
            concept_id = concept.get("concept_id")
            concept_domain = concept.get("metadata", {}).get("domain", "unknown")

            # Skip concepts from different domains
            if domain and domain != "unknown" and concept_domain != "unknown" and domain != concept_domain:
                # Allow matching between related domains
                science_domains = {"physics", "mathematics"}
                computer_domains = {"programming", "computer_science"}

                domain_match = (
                    (domain in science_domains and concept_domain in science_domains) or
                    (domain in computer_domains and concept_domain in computer_domains)
                )

                if not domain_match:
                    continue

            if not concept_id:
                continue

            # Get all representations for this concept
            representations = concept.get("representations", {}).get(language, [])
            if not representations:
                continue

            # Check each representation
            for representation in representations:
                # Skip very short terms which might cause false positives
                if len(representation) < 4:
                    continue

                # Generate variants for this representation
                variants = self.variant_matcher.generate_variants(representation)

                # Check if any variant appears in the text
                for variant in variants:
                    variant_lower = variant.lower()
                    text_lower = text.lower()

                    # Ensure this is a word boundary match, not a substring
                    pattern = r'\b' + re.escape(variant_lower) + r'\b'
                    if re.search(pattern, text_lower):
                        # Found a variant match
                        matches.append({
                            "concept_id": concept_id,
                            "concept": concept,
                            "similarity": 0.9,  # High similarity for variant matches
                            "match_type": "variant",
                            "matched_representation": representation,
                            "matched_variant": variant,
                            "domain": concept_domain
                        })
                        break  # Found a match for this representation

                # If a match was found for this representation, skip checking others
                if matches and matches[-1]["concept_id"] == concept_id:
                    break

        return matches

    def _calculate_educational_significance(
        self,
        segment: Dict[str, Any],
        concept: Dict[str, Any],
        concept_text: str,
        global_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate educational significance score to distinguish between passing mentions and comprehensive explanations.
        Optimized for performance by focusing on most important factors.

        Args:
            segment: Segment data
            concept: Concept data
            concept_text: The matched concept text
            global_analysis: Optional global text analysis

        Returns:
            Educational significance score (0.0-5.0)
        """
        # OPTIMIZATION: For Russian, use a simplified scoring formula
        if self.language == 'ru':
            # Simplified scoring based primarily on segment length
            text = segment.get("text", "")
            context_length = len(text.split())

            # Longer contexts typically have more explanation
            base_score = 2.0  # Start with a moderate score

            if context_length > 30:  # Long context
                base_score += 1.0
            elif context_length > 15:  # Medium context
                base_score += 0.5

            # Cap the score at 4.0 for Russian to be conservative
            return min(base_score, 4.0)

        # Standard scoring for other languages
        significance_score = 0.0

        # Factor 1: Segment's educational value (most important indicator)
        segment_edu_value = segment.get("educational_value", 0.0)
        significance_score += segment_edu_value * 0.8  # Weight: 0.8

        # Factor 2: Educational markers in the context (strong indicator)
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

        # Factor 4: Context length (longer contexts typically have more explanation)
        context_length = len(text.split())
        if context_length > 30:  # Long context
            significance_score += 0.8
        elif context_length > 15:  # Medium context
            significance_score += 0.4

        # Cap the score at 5.0
        return min(significance_score, 5.0)
