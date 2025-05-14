"""
Concept Repository for the Video Lecture Content Indexer.

Manages the concept data model, storage, and relationships for the system.
Implements efficient loading, indexing, and fuzzy matching of concepts across languages.
"""

import os
import json
import glob
import uuid
import logging
import time
import re
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from pathlib import Path
from collections import defaultdict

# Try to import optional dependencies with fallbacks
try:
    from rapidfuzz import fuzz, process
    FUZZY_MATCH_AVAILABLE = True
except ImportError:
    import difflib
    FUZZY_MATCH_AVAILABLE = False
    logging.warning("rapidfuzz not available - using difflib for fuzzy matching")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptRepository:
    """
    Core repository for managing concept data.

    This class handles:
    - Loading/saving concepts from JSONL files
    - Indexing concepts for efficient lookup
    - Fuzzy matching for concept identification
    - Relationship management between concepts
    """

    def __init__(self, concepts_dir: str = "concepts"):
        """
        Initialize the concept repository.

        Args:
            concepts_dir: Directory containing concept JSONL files
        """
        self.concepts_dir = concepts_dir
        self.concepts = {}  # concept_id -> concept data
        self.representation_index = {}  # normalized text -> set of concept_ids
        self.language_index = defaultdict(set)  # language -> set of concept_ids
        self.variant_index = defaultdict(set)  # variant form -> set of concept_ids

        # Ensure concepts directory exists
        os.makedirs(self.concepts_dir, exist_ok=True)

        # Create standard concept files if they don't exist
        self._ensure_concept_files()

        # Load all concepts
        self.load_concepts()

        logger.info(f"Concept repository initialized with {len(self.concepts)} concepts")

    def _ensure_concept_files(self):
        """Create standard concept files if they don't exist."""
        standard_files = [
            "mathematics.jsonl",
            "physics.jsonl",
            "computer_science.jsonl",
            "interdisciplinary.jsonl"
        ]

        for filename in standard_files:
            filepath = os.path.join(self.concepts_dir, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Create empty file
                    pass
                logger.info(f"Created empty concept file: {filepath}")

    def load_concepts(self) -> int:
        """
        Load all concepts from JSONL files in the concepts directory.

        Returns:
            Number of concepts loaded
        """
        start_time = time.time()

        # Clear existing data
        self.concepts = {}
        self.representation_index = {}
        self.language_index = defaultdict(set)
        self.variant_index = defaultdict(set)  # Clear variant index too

        # Find all JSONL files
        file_pattern = os.path.join(self.concepts_dir, "*.jsonl")
        concept_files = glob.glob(file_pattern)

        if not concept_files:
            logger.warning(f"No concept files found in {self.concepts_dir}")
            return 0

        # Load concepts from each file
        concept_count = 0
        for file_path in concept_files:
            file_concept_count = self._load_concepts_from_file(file_path)
            concept_count += file_concept_count
            logger.debug(f"Loaded {file_concept_count} concepts from {file_path}")

        # Build the concept relationship graph
        self._build_relationship_graph()

        # Build the variant index for all concepts
        self._build_variant_index()

        load_time = time.time() - start_time
        logger.info(f"Loaded {concept_count} concepts in {load_time:.2f} seconds")

        return concept_count

    def _load_concepts_from_file(self, file_path: str) -> int:
        """
        Load concepts from a single JSONL file.

        Args:
            file_path: Path to the JSONL file

        Returns:
            Number of concepts loaded from the file
        """
        concept_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # Skip empty lines and comments

                    try:
                        concept = json.loads(line)
                        concept_id = concept.get('concept_id')

                        if not concept_id:
                            logger.warning(f"Skipping concept without ID at {file_path}:{line_num}")
                            continue

                        # Validate and normalize concept
                        if self._validate_concept(concept):
                            # Store the concept
                            self.concepts[concept_id] = concept

                            # Index by representations
                            self._index_concept_representations(concept_id, concept)

                            concept_count += 1
                        else:
                            logger.warning(f"Invalid concept format at {file_path}:{line_num}")

                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON at {file_path}:{line_num}")

        except Exception as e:
            logger.error(f"Error loading concepts from {file_path}: {e}")

        return concept_count

    def _validate_concept(self, concept: Dict) -> bool:
        """
        Validate concept structure and normalize if needed.

        Args:
            concept: Concept dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(concept, dict):
            return False

        # Ensure required fields exist
        if 'concept_id' not in concept:
            return False

        # Ensure representations exist and are properly structured
        if 'representations' not in concept:
            concept['representations'] = {}
        else:
            # Ensure all representations are lowercase
            for lang, texts in concept['representations'].items():
                concept['representations'][lang] = [text.lower() for text in texts]

        # Ensure prerequisites and related lists exist
        if 'prerequisites' not in concept:
            concept['prerequisites'] = []

        if 'related' not in concept:
            concept['related'] = []

        # Ensure metadata exists
        if 'metadata' not in concept:
            concept['metadata'] = {}

        if 'created_at' not in concept['metadata']:
            concept['metadata']['created_at'] = datetime.now().isoformat()

        return True

    def _index_concept_representations(self, concept_id: str, concept: Dict):
        """
        Index a concept by its representations for efficient lookup.

        Args:
            concept_id: Concept ID
            concept: Concept dictionary
        """
        representations = concept.get('representations', {})

        for language, texts in representations.items():
            # Add to language index
            self.language_index[language].add(concept_id)

            # Index each representation
            for text in texts:
                if not text:
                    continue

                # Normalize text for indexing
                normalized_text = self._normalize_text(text, language)

                # Add to representation index
                if normalized_text not in self.representation_index:
                    self.representation_index[normalized_text] = set()

                self.representation_index[normalized_text].add(concept_id)

    def _build_variant_index(self):
        """
        Build an index of morphological variants for all concepts.
        This helps in matching different forms of the same concept.
        """
        logger.info("Building morphological variant index...")

        # Process each concept and its representations
        for concept_id, concept in self.concepts.items():
            representations = concept.get('representations', {})

            for language, texts in representations.items():
                for text in texts:
                    if not text:
                        continue

                    # Generate morphological variants based on language
                    variants = self._generate_variants(text, language)

                    # Add each variant to the index, pointing to this concept
                    for variant in variants:
                        variant_normalized = self._normalize_text(variant, language)
                        if variant_normalized:
                            self.variant_index[variant_normalized].add(concept_id)

        logger.info(f"Variant index built with {len(self.variant_index)} entries")

    def _generate_variants(self, text: str, language: str) -> Set[str]:
        """
        Generate common morphological variants of a concept text.

        Args:
            text: Original concept text
            language: Language code

        Returns:
            Set of possible variants
        """
        variants = {text}  # Always include the original text

        # Apply language-specific variant generation
        if language == 'ru':
            # Russian has complex morphology, generate common variants

            # Split into words to handle multi-word concepts
            words = text.split()

            if len(words) > 1:
                # For multi-word concepts like "соотношение неопределенности"

                # Common singular/plural and case variations for last word
                last_word = words[-1]
                word_variants = self._generate_russian_word_variants(last_word)

                # Common case variations for all but last word
                prefix_words = words[:-1]
                prefix_variants = []

                for i, word in enumerate(prefix_words):
                    # Generate variants for each word in the prefix
                    word_vars = self._generate_russian_word_variants(word)

                    # For the first iteration, just add each variant
                    if i == 0:
                        prefix_variants = [[v] for v in word_vars]
                    else:
                        # Combine with existing variants
                        new_variants = []
                        for existing in prefix_variants:
                            for var in word_vars:
                                new_variants.append(existing + [var])
                        prefix_variants = new_variants

                # Combine prefix variants with last word variants
                if prefix_variants:
                    for prefix in prefix_variants:
                        for last_var in word_variants:
                            variant = ' '.join(prefix + [last_var])
                            variants.add(variant)
            else:
                # For single-word concepts
                variants.update(self._generate_russian_word_variants(text))

        elif language == 'en':
            # English has simpler morphology, but still handle common variants
            variants.update(self._generate_english_word_variants(text))

        # Add lowercase variant for all languages
        variants.add(text.lower())

        return variants

    def _generate_russian_word_variants(self, word: str) -> Set[str]:
        """
        Generate common Russian morphological variants for a single word.

        Args:
            word: Russian word

        Returns:
            Set of possible variants
        """
        variants = {word}

        # Common singular/plural and case ending variations
        if len(word) > 3:  # Only process words of meaningful length
            # Handle common singular/plural alterations
            if word.endswith('ие'):
                variants.add(word[:-2] + 'ия')  # ие -> ия (case change)
                variants.add(word[:-2] + 'ий')  # ие -> ий (case change)
                variants.add(word[:-2] + 'ием')  # ие -> ием (instrumental case)
                variants.add(word + 'м')  # +м (instrumental case)
            elif word.endswith('ия'):
                variants.add(word[:-2] + 'ие')  # ия -> ие
                variants.add(word[:-2] + 'ий')  # ия -> ий
                variants.add(word[:-2] + 'ию')  # ия -> ию (accusative)
                variants.add(word[:-2] + 'ией')  # ия -> ией (instrumental)
            elif word.endswith('ть'):
                variants.add(word[:-2] + 'ти')  # ть -> ти (infinitive variant)

            # Handle common plural forms
            if word.endswith('ость'):
                variants.add(word[:-4] + 'ости')  # ость -> ости (genitive or plural)
                variants.add(word[:-4] + 'остей')  # ость -> остей (plural genitive)
                variants.add(word[:-4] + 'остью')  # ость -> остью (instrumental)
            elif word.endswith('ство'):
                variants.add(word[:-4] + 'ства')  # ство -> ства (genitive or plural)
                variants.add(word[:-4] + 'ствам')  # ство -> ствам (plural dative)
            elif word.endswith('ние'):
                variants.add(word[:-2] + 'ия')  # ние -> ния (genitive)
                variants.add(word[:-2] + 'ий')  # ние -> ний (plural genitive)
                variants.add(word[:-3] + 'й')  # ние -> ний (shorter form)
                variants.add(word + 'м')  # +м (instrumental case)
            elif word.endswith('ика'):
                variants.add(word[:-2] + 'ике')  # ика -> ике (locative)
                variants.add(word[:-2] + 'ику')  # ика -> ику (accusative)
                variants.add(word[:-2] + 'ики')  # ика -> ики (plural)
            elif word.endswith('а'):
                variants.add(word[:-1] + 'ы')  # а -> ы (plural)
                variants.add(word[:-1] + 'у')  # а -> у (accusative)
                variants.add(word[:-1] + 'е')  # а -> е (locative)
            elif word.endswith('я'):
                variants.add(word[:-1] + 'и')  # я -> и (plural or genitive)
                variants.add(word[:-1] + 'ю')  # я -> ю (accusative)
                variants.add(word[:-1] + 'е')  # я -> е (locative)
            elif word.endswith('й'):
                variants.add(word[:-1] + 'я')  # й -> я (genitive)
                variants.add(word[:-1] + 'ю')  # й -> ю (accusative)
                variants.add(word[:-1] + 'и')  # й -> и (plural)
                variants.add(word[:-1] + 'ем')  # й -> ем (instrumental)

            # Special case for words like "неопределенность" -> "неопределенностей"
            if word.endswith('ь'):
                variants.add(word[:-1] + 'и')  # ь -> и (plural)
                variants.add(word[:-1] + 'ей')  # ь -> ей (plural genitive)
                variants.add(word[:-1] + 'ью')  # ь -> ью (instrumental)

            # Specific to the example "неопределенности" -> "неопределенностей"
            if word.endswith('ти'):
                variants.add(word[:-2] + 'тей')  # ти -> тей
                variants.add(word[:-2] + 'ть')  # ти -> ть (nominative singular)

            # Handle words like "неопределенностей"
            if word.endswith('тей'):
                variants.add(word[:-3] + 'ть')  # тей -> ть (singular)
                variants.add(word[:-3] + 'ти')  # тей -> ти (genitive singular)

        # Handle simple 2-3 letter prepositions and connectors
        if word in {"от", "до", "на", "по", "за", "из", "под", "над", "при", "для",
                   "и", "а", "но", "или", "что", "как", "так", "где", "кто"}:
            # These words are invariant, no need to generate variants
            pass

        return variants

    def _generate_english_word_variants(self, text: str) -> Set[str]:
        """
        Generate common English morphological variants.

        Args:
            text: English text

        Returns:
            Set of possible variants
        """
        variants = {text}

        # Process multi-word phrases
        words = text.split()

        # Simple handling for multi-word phrases - for now, just add variants with/without "the", "a", "an"
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

    def _normalize_text(self, text: str, language: str = 'en') -> str:
        """
        Normalize text for concept matching.

        Args:
            text: Text to normalize
            language: Language code

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Language-specific normalization
        if language == 'ru':
            # Russian normalization (example: replace 'ё' with 'е')
            normalized = normalized.replace('ё', 'е')

            # Remove common noise words for more flexible matching
            noise_words = {'это', 'вот', 'так', 'так называемое', 'так называемая',
                         'который', 'которая', 'которые', 'которое'}

            for word in noise_words:
                normalized = re.sub(fr'\b{word}\b', '', normalized)

            # Remove common punctuation
            normalized = normalized.replace('-', ' ').replace('"', '').replace("'", '')

            # Handle specific types of concepts
            if 'соотношение' in normalized or 'принцип' in normalized:
                # Special handling for uncertainty principle variations
                normalized = re.sub(r'соотношени[яе]ми?', 'соотношение', normalized)
                normalized = re.sub(r'неопределенносте[йи]', 'неопределенности', normalized)

        elif language == 'en':
            # Enhanced English normalization
            # Remove common articles and determiners
            for prefix in ['the ', 'a ', 'an ', 'this ', 'that ']:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]

            # Remove common noise words for more flexible matching
            noise_words = {'so-called', 'so called', 'known as', 'called', 'which is', 'that is'}

            for word in noise_words:
                normalized = normalized.replace(word, '')

            # Remove hyphens
            normalized = normalized.replace('-', ' ')

        # Remove extra whitespace again after all replacements
        normalized = ' '.join(normalized.split())

        return normalized

    def _build_relationship_graph(self):
        """Build the concept relationship graph for efficient traversal."""
        # This could be expanded with more sophisticated graph algorithms
        # as needed for relationship traversal and learning path generation
        pass

    def find_concepts_by_text(
            self,
            text: str,
            language: Optional[str] = None,
            threshold: float = 0.8,
            max_results: int = 10
        ) -> List[Dict]:
            """
            Find concepts by text using exact and fuzzy matching.

            Args:
                text: Text to search for
                language: Optional language filter
                threshold: Minimum similarity threshold (0.0-1.0)
                max_results: Maximum number of results to return

            Returns:
                List of matching concepts with similarity scores
            """
            if not text:
                return []

            # Normalize query text
            normalized_text = self._normalize_text(text, language or 'en')

            # Initialize results
            matches = []

            # First try exact match
            exact_matches = self._find_exact_matches(normalized_text, language)

            # Then try variant matches (which are still considered exact but use the variant index)
            variant_matches = self._find_variant_matches(normalized_text, language)

            # Deduplicate exact and variant matches
            seen_concept_ids = set(match['concept_id'] for match in exact_matches)
            deduplicated_variant_matches = []

            for match in variant_matches:
                if match['concept_id'] not in seen_concept_ids:
                    seen_concept_ids.add(match['concept_id'])
                    deduplicated_variant_matches.append(match)

            # Combine deduplicated matches
            combined_exact_matches = exact_matches + deduplicated_variant_matches

            # If we found exact or variant matches, return them
            if combined_exact_matches:
                return combined_exact_matches[:max_results]

            # If no exact or variant matches, try fuzzy matching
            fuzzy_matches = self._find_fuzzy_matches(normalized_text, language, threshold)

            # Combine and sort results
            matches = combined_exact_matches + fuzzy_matches
            matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)

            return matches[:max_results]

    def _find_exact_matches(self, normalized_text: str, language: Optional[str] = None) -> List[Dict]:
        """
        Find concepts that exactly match the text.

        Args:
            normalized_text: Normalized text to match
            language: Optional language filter

        Returns:
            List of matching concepts
        """
        results = []

        # Check if there's an exact match in the representation index
        if normalized_text in self.representation_index:
            concept_ids = self.representation_index[normalized_text]

            # Filter by language if specified
            if language:
                concept_ids = concept_ids.intersection(self.language_index.get(language, set()))

            for concept_id in concept_ids:
                concept = self.concepts.get(concept_id)
                if concept:
                    results.append({
                        'concept_id': concept_id,
                        'concept': concept,
                        'similarity': 1.0,
                        'match_type': 'exact'
                    })

        return results

    def _find_variant_matches(self, normalized_text: str, language: Optional[str] = None) -> List[Dict]:
        """
        Find concepts that match morphological variants of the text.

        Args:
            normalized_text: Normalized text to match
            language: Optional language filter

        Returns:
            List of matching concepts via their variants
        """
        results = []

        # Check if there's a match in the variant index
        if normalized_text in self.variant_index:
            concept_ids = self.variant_index[normalized_text]

            # Filter by language if specified
            if language:
                concept_ids = concept_ids.intersection(self.language_index.get(language, set()))

            for concept_id in concept_ids:
                concept = self.concepts.get(concept_id)
                if concept:
                    results.append({
                        'concept_id': concept_id,
                        'concept': concept,
                        'similarity': 0.95,  # Slightly lower than exact match
                        'match_type': 'variant'
                    })

        return results

    def _find_fuzzy_matches(
        self,
        normalized_text: str,
        language: Optional[str] = None,
        threshold: float = 0.8
    ) -> List[Dict]:
        """
        Find concepts that fuzzy match the text.

        Args:
            normalized_text: Normalized text to match
            language: Optional language filter
            threshold: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of matching concepts
        """
        results = []

        # Get candidate concepts
        candidate_concept_ids = set()

        # If language is specified, only look at concepts in that language
        if language:
            candidate_concept_ids = self.language_index.get(language, set())
        else:
            # Otherwise, consider all concepts
            candidate_concept_ids = set(self.concepts.keys())

        # If there are no candidates, return empty results
        if not candidate_concept_ids:
            return []

        # For each candidate, check for fuzzy matches
        for concept_id in candidate_concept_ids:
            concept = self.concepts.get(concept_id)
            if not concept:
                continue

            # Find best match across all representations
            best_similarity = 0.0
            best_representation = None
            best_language = None

            for lang, representations in concept.get('representations', {}).items():
                # Skip languages other than the requested one if specified
                if language and lang != language:
                    continue

                for representation in representations:
                    normalized_repr = self._normalize_text(representation, lang)
                    similarity = self._calculate_similarity(normalized_text, normalized_repr)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_representation = representation
                        best_language = lang

            # If similarity is above threshold, add to results
            if best_similarity >= threshold:
                results.append({
                    'concept_id': concept_id,
                    'concept': concept,
                    'similarity': best_similarity,
                    'match_type': 'fuzzy',
                    'matched_representation': best_representation,
                    'matched_language': best_language
                })

        # Sort by similarity
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return results

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using available algorithms.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0

        # If texts are identical, return perfect match
        if text1 == text2:
            return 1.0

        # Use rapidfuzz if available for better performance
        if FUZZY_MATCH_AVAILABLE:
            return fuzz.ratio(text1, text2) / 100.0
        else:
            # Fallback to difflib
            return difflib.SequenceMatcher(None, text1, text2).ratio()

    def get_concept(self, concept_id: str) -> Optional[Dict]:
        """
        Get a concept by ID.

        Args:
            concept_id: Concept ID

        Returns:
            Concept dictionary or None if not found
        """
        return self.concepts.get(concept_id)

    def add_concept(
        self,
        concept_id: Optional[str] = None,
        representations: Optional[Dict[str, List[str]]] = None,
        prerequisites: Optional[List[str]] = None,
        related: Optional[List[str]] = None,
        file_category: str = "interdisciplinary"
    ) -> Optional[str]:
        """
        Add a new concept.

        Args:
            concept_id: Optional concept ID (generated if not provided)
            representations: Dictionary of language -> list of representations
            prerequisites: Optional list of prerequisite concept IDs
            related: Optional list of related concept IDs
            file_category: Category file to save to

        Returns:
            New concept ID or None if creation failed
        """
        # Generate concept ID if not provided
        if not concept_id:
            concept_id = str(uuid.uuid4())

        # Check if concept ID already exists
        if concept_id in self.concepts:
            logger.warning(f"Concept ID {concept_id} already exists")
            return None

        # Lowercase all representations
        lowercased_representations = {}
        if representations:
            for lang, texts in representations.items():
                lowercased_representations[lang] = [text.lower() for text in texts]

        # Create concept structure
        concept = {
            'concept_id': concept_id,
            'representations': lowercased_representations or {},
            'prerequisites': prerequisites or [],
            'related': related or [],
            'metadata': {
                'created_at': datetime.now().isoformat()
            }
        }

        # Validate concept
        if not self._validate_concept(concept):
            logger.error(f"Invalid concept structure for {concept_id}")
            return None

        # Add to in-memory repository
        self.concepts[concept_id] = concept
        self._index_concept_representations(concept_id, concept)

        # Update variant index for this concept
        self._update_variant_index_for_concept(concept_id, concept)

        # Save to file
        self._save_concept(concept, file_category)

        logger.info(f"Added new concept: {concept_id}")
        return concept_id

    def _update_variant_index_for_concept(self, concept_id: str, concept: Dict):
        """
        Update the variant index for a single concept.

        Args:
            concept_id: Concept ID
            concept: Concept dictionary
        """
        representations = concept.get('representations', {})

        for language, texts in representations.items():
            for text in texts:
                if not text:
                    continue

                # Generate morphological variants based on language
                variants = self._generate_variants(text, language)

                # Add each variant to the index, pointing to this concept
                for variant in variants:
                    variant_normalized = self._normalize_text(variant, language)
                    if variant_normalized:
                        self.variant_index[variant_normalized].add(concept_id)

    def add_representation(
        self,
        concept_id: str,
        text: str,
        language: str = 'en'
    ) -> bool:
        """
        Add a new representation to an existing concept.

        Args:
            concept_id: Concept ID
            text: Representation text
            language: Language code

        Returns:
            True if successful, False otherwise
        """
        concept = self.get_concept(concept_id)
        if not concept:
            logger.warning(f"Concept {concept_id} not found")
            return False

        # Ensure representations structure exists
        if 'representations' not in concept:
            concept['representations'] = {}

        # Ensure language list exists
        if language not in concept['representations']:
            concept['representations'][language] = []

        # Convert text to lowercase
        text = text.lower()

        # Check if representation already exists
        if text in concept['representations'][language]:
            logger.warning(f"Representation '{text}' already exists for concept {concept_id}")
            return False

        # Add new representation
        concept['representations'][language].append(text)

        # Update index
        normalized_text = self._normalize_text(text, language)
        if normalized_text not in self.representation_index:
            self.representation_index[normalized_text] = set()
        self.representation_index[normalized_text].add(concept_id)

        # Add to language index if needed
        self.language_index[language].add(concept_id)

        # Update variant index
        variants = self._generate_variants(text, language)
        for variant in variants:
            variant_normalized = self._normalize_text(variant, language)
            if variant_normalized:
                self.variant_index[variant_normalized].add(concept_id)

        # Update concept's last_updated timestamp
        concept['metadata']['last_updated'] = datetime.now().isoformat()

        # Save changes
        file_category = self._get_concept_file_category(concept_id)
        self._save_concept(concept, file_category)

        logger.info(f"Added representation '{text}' to concept {concept_id}")
        return True

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = 'prerequisite'
    ) -> bool:
        """
        Add a relationship between concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship ('prerequisite' or 'related')

        Returns:
            True if successful, False otherwise
        """
        # Verify both concepts exist
        source = self.get_concept(source_id)
        target = self.get_concept(target_id)

        if not source:
            logger.warning(f"Source concept {source_id} not found")
            return False

        if not target:
            logger.warning(f"Target concept {target_id} not found")
            return False

        # Check relationship type
        if relationship_type not in ['prerequisite', 'related']:
            logger.warning(f"Invalid relationship type: {relationship_type}")
            return False

        # Check if relationship already exists
        if relationship_type == 'prerequisite':
            if target_id in source.get('prerequisites', []):
                logger.warning(f"Prerequisite relationship from {source_id} to {target_id} already exists")
                return False

            # Add prerequisite relationship
            if 'prerequisites' not in source:
                source['prerequisites'] = []
            source['prerequisites'].append(target_id)

        elif relationship_type == 'related':
            if target_id in source.get('related', []):
                logger.warning(f"Related relationship from {source_id} to {target_id} already exists")
                return False

            # Add related relationship
            if 'related' not in source:
                source['related'] = []
            source['related'].append(target_id)

        # Update last_updated timestamp
        source['metadata']['last_updated'] = datetime.now().isoformat()

        # Save changes
        source_category = self._get_concept_file_category(source_id)
        self._save_concept(source, source_category)

        logger.info(f"Added {relationship_type} relationship from {source_id} to {target_id}")
        return True

    def remove_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = 'prerequisite'
    ) -> bool:
        """
        Remove a relationship between concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship ('prerequisite' or 'related')

        Returns:
            True if successful, False otherwise
        """
        # Verify source concept exists
        source = self.get_concept(source_id)

        if not source:
            logger.warning(f"Source concept {source_id} not found")
            return False

        # Check relationship type
        if relationship_type not in ['prerequisite', 'related']:
            logger.warning(f"Invalid relationship type: {relationship_type}")
            return False

        # Check if relationship exists
        if relationship_type == 'prerequisite':
            if target_id not in source.get('prerequisites', []):
                logger.warning(f"Prerequisite relationship from {source_id} to {target_id} not found")
                return False

            # Remove prerequisite relationship
            source['prerequisites'].remove(target_id)

        elif relationship_type == 'related':
            if target_id not in source.get('related', []):
                logger.warning(f"Related relationship from {source_id} to {target_id} not found")
                return False

            # Remove related relationship
            source['related'].remove(target_id)

        # Update last_updated timestamp
        source['metadata']['last_updated'] = datetime.now().isoformat()

        # Save changes
        source_category = self._get_concept_file_category(source_id)
        self._save_concept(source, source_category)

        logger.info(f"Removed {relationship_type} relationship from {source_id} to {target_id}")
        return True

    def list_concepts(
        self,
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        List concepts with pagination.

        Args:
            language: Optional language filter
            limit: Maximum number of concepts to return
            offset: Pagination offset

        Returns:
            List of concepts
        """
        # Filter concepts by language if specified
        if language:
            concept_ids = self.language_index.get(language, set())
            filtered_concepts = [self.concepts[cid] for cid in concept_ids if cid in self.concepts]
        else:
            filtered_concepts = list(self.concepts.values())

        # Sort by created_at timestamp if available
        filtered_concepts.sort(
            key=lambda c: c.get('metadata', {}).get('created_at', ''),
            reverse=True
        )

        # Apply pagination
        paginated = filtered_concepts[offset:offset+limit]

        # Return list of concepts
        return paginated

    def edit_concept(
        self,
        concept_id: str,
        new_concept_id: Optional[str] = None,
        add_representations: Optional[Dict[str, List[str]]] = None,
        remove_representations: Optional[Dict[str, List[str]]] = None,
        add_prerequisites: Optional[List[str]] = None,
        remove_prerequisites: Optional[List[str]] = None,
        add_related: Optional[List[str]] = None,
        remove_related: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_category: Optional[str] = None
    ) -> bool:
        """
        Edit a concept with comprehensive modifications.

        Args:
            concept_id: Current concept ID
            new_concept_id: Optional new concept ID to rename
            add_representations: Dict of language -> list of representations to add
            remove_representations: Dict of language -> list of representations to remove
            add_prerequisites: List of prerequisite concept IDs to add
            remove_prerequisites: List of prerequisite concept IDs to remove
            add_related: List of related concept IDs to add
            remove_related: List of related concept IDs to remove
            metadata: Dict of metadata to update/add
            file_category: Optional new file category

        Returns:
            True if successful, False otherwise
        """
        # Check if concept exists
        concept = self.get_concept(concept_id)
        if not concept:
            logger.warning(f"Concept {concept_id} not found")
            return False

        try:
            # Handle ID change if requested
            current_concept_id = concept_id
            if new_concept_id and new_concept_id != concept_id:
                # Check if new ID already exists
                if new_concept_id in self.concepts:
                    logger.warning(f"Cannot rename to {new_concept_id} - ID already exists")
                    return False

                # Update concept ID
                concept['concept_id'] = new_concept_id

                # Remove old concept from repository
                del self.concepts[concept_id]

                # Add with new ID
                self.concepts[new_concept_id] = concept

                # Update current ID for saving later
                current_concept_id = new_concept_id

                logger.info(f"Renamed concept from {concept_id} to {new_concept_id}")

            # Handle adding representations
            if add_representations:
                for language, texts in add_representations.items():
                    # Ensure language entry exists
                    if language not in concept['representations']:
                        concept['representations'][language] = []

                    # Add each representation (ensuring lowercase)
                    for text in texts:
                        # Skip empty texts
                        if not text:
                            continue

                        text = text.lower()  # Ensure lowercase

                        # Skip if already exists
                        if text in concept['representations'][language]:
                            logger.info(f"Representation '{text}' already exists - skipping")
                            continue

                        # Add representation
                        concept['representations'][language].append(text)

                        # Update indices
                        self.language_index[language].add(current_concept_id)

                        normalized_text = self._normalize_text(text, language)
                        if normalized_text not in self.representation_index:
                            self.representation_index[normalized_text] = set()
                        self.representation_index[normalized_text].add(current_concept_id)

                        # Update variant index
                        variants = self._generate_variants(text, language)
                        for variant in variants:
                            variant_normalized = self._normalize_text(variant, language)
                            if variant_normalized:
                                self.variant_index[variant_normalized].add(current_concept_id)

                        logger.info(f"Added representation '{text}' in language '{language}'")

            # Handle removing representations
            if remove_representations:
                for language, texts in remove_representations.items():
                    # Skip if language doesn't exist
                    if language not in concept['representations']:
                        continue

                    # Convert all texts to lowercase for comparison
                    texts_lower = [t.lower() for t in texts]

                    # Remove each representation
                    for text_lower in texts_lower:
                        if text_lower in concept['representations'][language]:
                            # Remove from concept
                            concept['representations'][language].remove(text_lower)

                            # If no more representations in this language, remove from language index
                            if not concept['representations'][language]:
                                if current_concept_id in self.language_index.get(language, set()):
                                    self.language_index[language].remove(current_concept_id)

                            logger.info(f"Removed representation '{text_lower}' in language '{language}'")

            # Handle adding prerequisites
            if add_prerequisites:
                for prereq_id in add_prerequisites:
                    # Skip if already exists
                    if prereq_id in concept.get('prerequisites', []):
                        continue

                    # Check if target concept exists
                    if prereq_id not in self.concepts:
                        logger.warning(f"Prerequisite concept {prereq_id} not found - skipping")
                        continue

                    # Add prerequisite
                    if 'prerequisites' not in concept:
                        concept['prerequisites'] = []
                    concept['prerequisites'].append(prereq_id)

                    logger.info(f"Added prerequisite {prereq_id}")

            # Handle removing prerequisites
            if remove_prerequisites:
                for prereq_id in remove_prerequisites:
                    # Skip if not in prerequisites
                    if prereq_id not in concept.get('prerequisites', []):
                        continue

                    # Remove prerequisite
                    concept['prerequisites'].remove(prereq_id)

                    logger.info(f"Removed prerequisite {prereq_id}")

            # Handle adding related concepts
            if add_related:
                for related_id in add_related:
                    # Skip if already exists
                    if related_id in concept.get('related', []):
                        continue

                    # Check if target concept exists
                    if related_id not in self.concepts:
                        logger.warning(f"Related concept {related_id} not found - skipping")
                        continue

                    # Add related concept
                    if 'related' not in concept:
                        concept['related'] = []
                    concept['related'].append(related_id)

                    logger.info(f"Added related concept {related_id}")

            # Handle removing related concepts
            if remove_related:
                for related_id in remove_related:
                    # Skip if not in related
                    if related_id not in concept.get('related', []):
                        continue

                    # Remove related concept
                    concept['related'].remove(related_id)

                    logger.info(f"Removed related concept {related_id}")

            # Update metadata
            if metadata:
                if 'metadata' not in concept:
                    concept['metadata'] = {}

                # Update each metadata field
                for key, value in metadata.items():
                    concept['metadata'][key] = value

                logger.info(f"Updated metadata: {', '.join(metadata.keys())}")

            # Always update last_updated timestamp
            if 'metadata' not in concept:
                concept['metadata'] = {}
            concept['metadata']['last_updated'] = datetime.now().isoformat()

            # Save changes
            save_file_category = file_category or self._get_concept_file_category(current_concept_id)

            # If ID changed, delete old concept file
            if new_concept_id and new_concept_id != concept_id:
                self._delete_concept_from_file(concept_id, self._get_concept_file_category(concept_id))

            # Save updated concept
            self._save_concept(concept, save_file_category)

            # If ID changed or representations changed, rebuild indices
            if new_concept_id or add_representations or remove_representations:
                # Clear and rebuild indices for this concept
                self._rebuild_indices_for_concept(current_concept_id, concept)

            logger.info(f"Successfully edited concept {current_concept_id}")
            return True

        except Exception as e:
            logger.error(f"Error editing concept {concept_id}: {e}")
            return False

    def _rebuild_indices_for_concept(self, concept_id: str, concept: Dict):
        """
        Rebuild indices for a specific concept.

        Args:
            concept_id: Concept ID
            concept: Concept dictionary
        """
        # Clear existing entries for this concept
        for lang_set in self.language_index.values():
            if concept_id in lang_set:
                lang_set.remove(concept_id)

        for rep_set in self.representation_index.values():
            if concept_id in rep_set:
                rep_set.remove(concept_id)

        for var_set in self.variant_index.values():
            if concept_id in var_set:
                var_set.remove(concept_id)

        # Rebuild indices
        self._index_concept_representations(concept_id, concept)

        # Rebuild variant index for this concept
        self._update_variant_index_for_concept(concept_id, concept)

    def _delete_concept_from_file(self, concept_id: str, file_category: str) -> bool:
        """
        Delete a concept from its file.

        Args:
            concept_id: Concept ID
            file_category: File category

        Returns:
            True if successful, False otherwise
        """
        # Ensure file category is valid
        if not file_category or not file_category.endswith('.jsonl'):
            file_category = f"{file_category}.jsonl"

        file_path = os.path.join(self.concepts_dir, file_category)

        try:
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                return True

            # Read all concepts from the file
            concepts = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        concepts.append(line)  # Preserve comments and empty lines
                        continue

                    try:
                        existing = json.loads(line)
                        existing_id = existing.get('concept_id')

                        if existing_id and existing_id != concept_id:
                            # Keep all concepts except the one to delete
                            concepts.append(line)
                    except json.JSONDecodeError:
                        # Keep invalid lines as-is
                        concepts.append(line)

            # Write the updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in concepts:
                    f.write(line + '\n')

            return True

        except Exception as e:
            logger.error(f"Error deleting concept {concept_id} from file: {e}")
            return False

    def find_concept_candidates(self, threshold: float = 0.7) -> List[Dict]:
        """
        Find potential new concept candidates based on patterns in existing concepts.

        Args:
            threshold: Similarity threshold for clustering

        Returns:
            List of potential new concepts
        """
        # This would be implemented based on additional data sources,
        # such as analyzing search queries, video transcripts, etc.
        # Not implemented in this version
        return []

    def _save_concept(self, concept: Dict, file_category: str) -> bool:
        """
        Save a concept to the appropriate JSONL file.

        Args:
            concept: Concept dictionary
            file_category: Category file to save to

        Returns:
            True if successful, False otherwise
        """
        # Ensure file category is valid
        if not file_category or not file_category.endswith('.jsonl'):
            file_category = f"{file_category}.jsonl"

        file_path = os.path.join(self.concepts_dir, file_category)

        try:
            # Read all concepts from the file
            concepts = []
            concept_ids = set()

            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            concepts.append(line)  # Preserve comments and empty lines
                            continue

                        try:
                            existing = json.loads(line)
                            existing_id = existing.get('concept_id')

                            if existing_id:
                                if existing_id == concept['concept_id']:
                                    # Skip this concept as we'll add the updated version later
                                    continue

                                # Add to tracking set
                                concept_ids.add(existing_id)

                            # Add to concepts list
                            concepts.append(line)
                        except json.JSONDecodeError:
                            # Keep invalid lines as-is
                            concepts.append(line)

            # Prepare the new concept line
            new_concept_line = json.dumps(concept, ensure_ascii=False)

            # Write the updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write existing concepts
                for line in concepts:
                    f.write(line + '\n')

                # Write the new/updated concept if not already in the file
                if concept['concept_id'] not in concept_ids:
                    f.write(new_concept_line + '\n')

            return True

        except Exception as e:
            logger.error(f"Error saving concept {concept['concept_id']} to {file_path}: {e}")
            return False

    def _get_concept_file_category(self, concept_id: str) -> str:
        """
        Determine which file a concept should be saved to.

        Args:
            concept_id: Concept ID

        Returns:
            File category name
        """
        # This implementation just returns the file a concept was loaded from
        # or defaults to interdisciplinary.jsonl

        # Check if the concept exists and where it was loaded from
        for file_path in glob.glob(os.path.join(self.concepts_dir, "*.jsonl")):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        try:
                            existing = json.loads(line)
                            if existing.get('concept_id') == concept_id:
                                return os.path.basename(file_path)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

        # Default to interdisciplinary if not found
        return "interdisciplinary.jsonl"

    def get_concept_statistics(self) -> Dict:
        """
        Get statistics about the concepts in the repository.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_concepts': len(self.concepts),
            'languages': {},
            'files': {},
            'relationships': {
                'prerequisites_count': 0,
                'related_count': 0
            },
            'variants': len(self.variant_index)
        }

        # Count concepts by language
        for language, concept_ids in self.language_index.items():
            stats['languages'][language] = len(concept_ids)

        # Count concepts by file
        for file_path in glob.glob(os.path.join(self.concepts_dir, "*.jsonl")):
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
                    stats['files'][file_name] = count
            except Exception:
                stats['files'][file_name] = 0

        # Count relationships
        for concept in self.concepts.values():
            stats['relationships']['prerequisites_count'] += len(concept.get('prerequisites', []))
            stats['relationships']['related_count'] += len(concept.get('related', []))

        # Calculate average variants per concept
        if len(self.concepts) > 0:
            stats['avg_variants_per_concept'] = len(self.variant_index) / len(self.concepts)

        return stats


# Singleton instance for global access
_instance = None

def get_concept_repository(concepts_dir: str = "concepts") -> ConceptRepository:
    """
    Get or create the ConceptRepository singleton instance.

    Args:
        concepts_dir: Directory containing concept JSONL files

    Returns:
        ConceptRepository instance
    """
    global _instance

    if _instance is None:
        _instance = ConceptRepository(concepts_dir)

    return _instance
