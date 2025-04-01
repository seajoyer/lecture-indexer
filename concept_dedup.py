"""
Enhanced concept deduplication for the Lecture Video Content Indexer.
This extension improves concept similarity detection and prevents duplicate concepts
when processing new videos.
"""

import re
import unicodedata
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict

class ConceptDedupExtension:
    """Extension for concept deduplication in the Lecture Video Content Indexer."""

    def __init__(self, data_access, config=None):
        """
        Initialize concept deduplication extension.

        Args:
            data_access: Data access instance
            config: Optional configuration dictionary
        """
        self.data_access = data_access
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.concept_cache = {}  # Cache for quick lookups

        # Pre-built stop words for concept normalization
        self.stop_words = {
            'ru': {
                'это', 'такая', 'такое', 'просто', 'самое', 'наша', 'наше', 'вот', 'эта', 'эти',
                'тогда', 'когда', 'у', 'нас', 'вас', 'нашего', 'вашего', 'да', 'нет'
            },
            'en': {
                'the', 'a', 'an', 'this', 'that', 'these', 'those', 'our', 'your', 'my',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'just', 'very', 'then', 'when'
            }
        }

        # Word replacements for normalization (e.g., synonyms)
        self.word_replacements = {
            'ru': {
                'волновая': ['волна', 'волн'],
                'шаровая': ['сферическая', 'шар'],
                'функция': ['функц'],
                'координатном': ['координат', 'координатах'],
                'представлении': ['представление', 'представ'],
                'основного': ['основное', 'основн'],
                'состояния': ['состояние', 'состоян']
            },
            'en': {
                'wave': ['wavefunction', 'wavefunct'],
                'function': ['functional', 'funct'],
                'spherical': ['sphere', 'spher'],
                'coordinate': ['coordinates', 'coord'],
                'representation': ['represents', 'represent'],
                'ground': ['grounded', 'base'],
                'state': ['status', 'condition']
            }
        }

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Language code ('ru' or 'en')
        """
        # Simple language detection based on character set
        cyrillic_count = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c.lower() in 'ёэіїєґў')
        latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')

        return 'ru' if cyrillic_count > latin_count else 'en'

    def normalize_concept_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Normalize concept text for better matching.

        Args:
            text: Input concept text
            language: Optional language code

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Detect language if not provided
        if not language:
            language = self.detect_language(text)

        # Convert to lowercase and strip whitespace
        text = text.lower().strip()

        # Remove accents and normalize Unicode
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove stop words
        stop_words = self.stop_words.get(language, self.stop_words['en'])
        words = text.split()
        words = [w for w in words if w not in stop_words]

        # Apply word replacements (stemming/normalization)
        replacements = self.word_replacements.get(language, {})
        normalized_words = []

        for word in words:
            replaced = False
            for target, alternatives in replacements.items():
                if word == target or any(alt in word for alt in alternatives):
                    normalized_words.append(target)
                    replaced = True
                    break
            if not replaced:
                normalized_words.append(word)

        return ' '.join(normalized_words)

    def calculate_similarity(self, text1: str, text2: str, language: Optional[str] = None) -> float:
        """
        Calculate similarity between two concept texts.

        Args:
            text1: First concept text
            text2: Second concept text
            language: Optional language code

        Returns:
            Similarity score between 0 and 1
        """
        # Normalize texts
        norm_text1 = self.normalize_concept_text(text1, language)
        norm_text2 = self.normalize_concept_text(text2, language)

        # If either text is empty after normalization, return 0
        if not norm_text1 or not norm_text2:
            return 0.0

        # Get words for each text
        words1 = set(norm_text1.split())
        words2 = set(norm_text2.split())

        # Calculate Jaccard similarity (intersection over union)
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Exact match bonus
        if norm_text1 == norm_text2:
            jaccard = min(jaccard + 0.3, 1.0)

        # Substring match bonus
        elif norm_text1 in norm_text2 or norm_text2 in norm_text1:
            jaccard = min(jaccard + 0.2, 1.0)

        # Length similarity bonus (for short concepts)
        if len(words1) <= 3 and len(words2) <= 3 and abs(len(words1) - len(words2)) <= 1:
            jaccard = min(jaccard + 0.1, 1.0)

        return jaccard

    def find_similar_concept(self, concept_text: str, domain: str, language: str = None) -> Optional[Dict[str, Any]]:
        """
        Find similar existing concept in database.

        Args:
            concept_text: Concept text to search for
            domain: Concept domain
            language: Optional language code

        Returns:
            Similar concept or None if not found
        """
        # Detect language if not provided
        if not language:
            language = self.detect_language(concept_text)

        # Normalize input concept
        normalized_text = self.normalize_concept_text(concept_text, language)

        # Try exact match first
        query = """
        SELECT c.* FROM concepts c
        WHERE c.domain = ? AND c.language = ?
        """
        params = [domain, language]

        # Execute query
        concepts = self.data_access.execute_query(query, tuple(params))

        # Filter concepts by similarity
        similar_concepts = []

        for concept in concepts:
            similarity = self.calculate_similarity(
                concept_text, concept.get("text", ""), language
            )

            if similarity >= self.similarity_threshold:
                similar_concepts.append({
                    "concept_id": concept.get("concept_id"),
                    "text": concept.get("text"),
                    "similarity": similarity
                })

        # Sort by similarity (highest first)
        similar_concepts.sort(key=lambda x: x["similarity"], reverse=True)

        # Return most similar concept if any found
        if similar_concepts:
            # Get full concept details
            return self.data_access.get_concept(similar_concepts[0]["concept_id"])

        return None

    def add_to_data_access(self, data_access):
        """
        Add concept deduplication method to data access class.

        Args:
            data_access: DataAccess instance
        """
        # Store original method
        original_save_concept = data_access.save_concept

        # Define enhanced method
        def enhanced_save_concept(concept_data):
            """
            Enhanced concept saving with deduplication.

            Args:
                concept_data: Concept data dictionary

            Returns:
                Concept ID if successful, None otherwise
            """
            # Check if we need to deduplicate
            if not concept_data.get("skip_deduplication", False):
                # Look for similar concepts
                similar_concept = self.find_similar_concept(
                    concept_data.get("text", ""),
                    concept_data.get("domain", "unknown"),
                    concept_data.get("language", "en")
                )

                # If similar concept found, use it instead
                if similar_concept:
                    # Get concept ID
                    concept_id = similar_concept.get("concept_id")

                    # Save occurrences if provided
                    video_id = concept_data.get("video_id")
                    occurrences = concept_data.get("occurrences", [])

                    if video_id and not occurrences:
                        # Find occurrences in the provided video
                        segments = data_access.get_video_segments(video_id)
                        if segments:
                            concept_extractor = getattr(data_access, "_find_concept_occurrences", None)
                            if concept_extractor:
                                occurrences = concept_extractor(
                                    concept_id,
                                    concept_data.get("text", ""),
                                    segments,
                                    video_id
                                )

                    # Save occurrences if found
                    if occurrences:
                        data_access.save_occurrences(concept_id, occurrences)

                    # Return existing concept ID
                    return concept_id

            # Otherwise, save as new concept
            return original_save_concept(concept_data)

        # Replace the method
        data_access.save_concept = enhanced_save_concept
        data_access.find_similar_concept = self.find_similar_concept

        return data_access

    def enhance_search_results(self, search_engine):
        """
        Enhance search engine to deduplicate search results.

        Args:
            search_engine: SearchEngine instance
        """
        # Store original method
        original_search = search_engine.search

        # Define enhanced method
        def enhanced_search(query):
            """
            Enhanced search with result deduplication.

            Args:
                query: Search query dictionary

            Returns:
                Search results with deduplicated concepts
            """
            # Get original results
            results = original_search(query)

            # Deduplicate concept results
            if "results" in results:
                # Group results by normalized text
                concept_groups = defaultdict(list)

                # Track concept results for deduplication
                concept_results = []
                other_results = []

                for result in results["results"]:
                    # Separate concept results from other results
                    if result.get("result_type") == "concept":
                        concept_results.append(result)
                    else:
                        other_results.append(result)

                # Get language from query
                language = query.get("language")

                # Group similar concepts
                for result in concept_results:
                    normalized_text = self.normalize_concept_text(
                        result.get("text", ""), language
                    )
                    concept_groups[normalized_text].append(result)

                # Take only the best result from each group
                deduplicated_concepts = []

                for normalized_text, group in concept_groups.items():
                    # Sort by relevance score
                    group.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                    # Take the best one
                    deduplicated_concepts.append(group[0])

                # Combine deduplicated concepts with other results
                deduplicated_results = other_results + deduplicated_concepts

                # Re-sort all results by relevance score
                deduplicated_results.sort(
                    key=lambda x: x.get("relevance_score", 0), reverse=True
                )

                # Update results
                results["results"] = deduplicated_results

                # Update counts
                results["deduplicated_count"] = len(concept_results) - len(deduplicated_concepts)

            return results

        # Replace the method
        search_engine.search = enhanced_search

        return search_engine

def apply_concept_deduplication(data_pipeline, search_engine, data_access):
    """
    Apply concept deduplication to all components.

    Args:
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        data_access: DataAccess instance
    """
    # Create deduplication extension
    dedup = ConceptDedupExtension(data_access)

    # Apply to data access
    data_access = dedup.add_to_data_access(data_access)

    # Apply to search engine
    search_engine = dedup.enhance_search_results(search_engine)

    return data_pipeline, search_engine, data_access
