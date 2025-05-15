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

    def _init_language_resources(self):
        """Initialize language-specific resources for variant matching."""
        # Common word endings to normalize for Russian
        self.russian_endings = {
            # Noun endings (singular -> plural, different cases)
            'ие': ['ия', 'ий', 'ием'],
            'ия': ['ие', 'ий', 'ию', 'ией'],
            'ть': ['ти'],
            'ость': ['ости', 'остей', 'остью'],
            'ство': ['ства', 'ствам'],
            'а': ['ы', 'у', 'е'],
            'я': ['и', 'ю', 'е'],
            'й': ['я', 'ю', 'и', 'ем'],
            'ь': ['и', 'ей', 'ью']
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

        Args:
            text: The original text

        Returns:
            Set of possible variants
        """
        variants = {text}  # Always include the original

        if self.language == 'ru':
            variants.update(self._generate_russian_variants(text))
        else:  # Default to English
            variants.update(self._generate_english_variants(text))

        return variants

    def _generate_russian_variants(self, text: str) -> Set[str]:
        """Generate Russian morphological variants."""
        variants = set()
        words = text.split()

        # For single words, apply ending transformations
        if len(words) == 1:
            word = words[0]
            for ending, replacements in self.russian_endings.items():
                if word.endswith(ending) and len(word) > len(ending) + 2:  # Ensure word is long enough
                    for replacement in replacements:
                        variant = word[:-len(ending)] + replacement
                        variants.add(variant)

        # For multi-word terms like "соотношение неопределенности"
        elif len(words) > 1:
            # Often only the last word changes in Russian phrases
            for ending, replacements in self.russian_endings.items():
                if words[-1].endswith(ending) and len(words[-1]) > len(ending) + 2:
                    base_words = words[:-1]  # All words except the last
                    for replacement in replacements:
                        new_last_word = words[-1][:-len(ending)] + replacement
                        variant = ' '.join(base_words + [new_last_word])
                        variants.add(variant)

            # Generate variants where the first word changes too
            for ending, replacements in self.russian_endings.items():
                if words[0].endswith(ending) and len(words[0]) > len(ending) + 2:
                    for replacement in replacements:
                        new_first_word = words[0][:-len(ending)] + replacement
                        variant = ' '.join([new_first_word] + words[1:])
                        variants.add(variant)

        return variants

    def _generate_english_variants(self, text: str) -> Set[str]:
        """Generate English morphological variants."""
        variants = set()
        words = text.split()

        # For single words
        if len(words) == 1:
            word = words[0]
            for ending, replacements in self.english_endings.items():
                if word.endswith(ending) and len(word) > len(ending) + 2:
                    for replacement in replacements:
                        variant = word[:-len(ending)] + replacement
                        variants.add(variant)

        # For multi-word phrases, try changing one word at a time
        elif len(words) > 1:
            # Try variants of the last word
            for ending, replacements in self.english_endings.items():
                if words[-1].endswith(ending) and len(words[-1]) > len(ending) + 2:
                    for replacement in replacements:
                        new_last_word = words[-1][:-len(ending)] + replacement
                        variant = ' '.join(words[:-1] + [new_last_word])
                        variants.add(variant)

        return variants

    def match_variants(self, text: str, target: str) -> float:
        """
        Check if text matches any variant of the target.

        Args:
            text: Text to check
            target: Target concept

        Returns:
            Similarity score (0.0-1.0), 1.0 if exact match
        """
        # If exact match, return perfect score
        if text.lower() == target.lower():
            return 1.0

        # Generate variants of both texts
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

        # Load enhanced NLP resources
        self._load_nlp_resources()

        # Domain classification patterns - for verifying domain context matches
        self._init_domain_patterns()

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

        # Get the detected domain
        domain = processed_transcript.get("domain", global_analysis.get("domain", "unknown"))

        # Log input information
        logger.info(f"Extracting concepts from transcript: video_id={video_id}, language={language}, domain={domain}, segments={len(segments)}")

        # Extract concepts using repository matching
        result = self.extract_concepts_from_segments(
            segments,
            video_id,
            language,
            domain,
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
        domain: Optional[str] = None,
        global_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from transcript segments by matching against the concept repository.

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

        # Process each segment
        for segment in segments:
            segment_id = segment.get("id", str(uuid.uuid4()))
            segment_text = segment.get("text", "")
            start_time_sec = segment.get("start_time", 0.0)

            if not segment_text.strip():
                continue

            # Find matching concepts in this segment using enhanced morphological matching
            segment_matches = self._find_matching_concepts_in_text(
                segment_text,
                language=lang,
                domain=domain,
                threshold=0.85  # Increased threshold to reduce false positives
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
                        "language": lang,
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

    def _find_matching_concepts_in_text(self, text: str, language: str, domain: Optional[str] = None, threshold: float = 0.85) -> List[Dict]:
        """
        Find matching concepts in text using the concept repository with enhanced morphological matching.

        Args:
            text: Text to search in
            language: Language code
            domain: Optional domain filter
            threshold: Similarity threshold

        Returns:
            List of matching concept dictionaries
        """
        # Use concept repository to find matches
        matches = self.concept_repository.find_concepts_by_text(
            text,
            language=language,
            threshold=threshold,
            max_results=5  # Limit to avoid excessive processing
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
        if not matches and language in ['ru', 'en']:
            matches = self._find_morphological_variants_in_text(text, language, domain, threshold)

        # Check if each matched concept exists in the database for valid foreign key relationships
        valid_matches = []
        for match in matches:
            concept_id = match.get("concept_id")
            if not concept_id:
                continue

            # Add to valid matches
            valid_matches.append(match)

        return valid_matches

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

        # Get all concepts for the given language
        concepts = self.concept_repository.list_concepts(language=language, limit=1000)

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
