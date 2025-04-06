"""
Enhanced concept deduplication module for the Lecture Video Content Indexer.
Implements unified character-level MLCS algorithm for robust concept similarity detection,
handling variations and duplicates across all concept categories.
"""

import re
import logging
import time
from typing import Dict, List, Any, Optional

# Import MLCS algorithm
from mlcs_algorithm import MLCSAlgorithm

# Configure logging
logger = logging.getLogger(__name__)

class ConceptDedupExtension:
    """
    Enhanced concept deduplication that uses character-level MLCS algorithm
    to identify and merge similar concepts across academic lectures.
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

        # Initialize MLCS algorithm
        self.mlcs_processor = MLCSAlgorithm(language)

        # Set thresholds for similarity detection
        self.similarity_thresholds = {
            "exact_match": 1.0,    # Exact match (after normalization)
            "high_match": 0.90,    # High confidence match
            "medium_match": 0.75,  # Medium confidence match
            "low_match": 0.65      # Low confidence match
        }

        # Load basic stopwords for pre-filtering
        self._load_basic_resources()

        logger.info(f"ConceptDedupExtension initialized with MLCS-based similarity detection")

    def _load_basic_resources(self):
        """Load minimal language resources needed for basic filtering."""
        # Simple stopwords for basic filtering (minimal set)
        self.basic_stopwords = {
            'en': {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'is', 'are', 'was', 'were'},
            'ru': {'и', 'или', 'в', 'на', 'с', 'это', 'эти', 'этот', 'та', 'то', 'те'}
        }

        # Common filler phrases to remove
        self.filler_phrases = {
            "en": [r'^the\s+', r'^a\s+', r'^an\s+'],
            "ru": [r'^это\s+', r'^вот\s+', r'^то\s+']
        }

    def normalize_concept_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Simple normalization of concept text for comparison.

        Args:
            text: Concept text
            language: Language code (defaults to instance language)

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Language-specific minimal preprocessing
        lang = language or self.language

        # Apply minimal language-specific normalizations
        if lang in self.filler_phrases:
            for pattern in self.filler_phrases[lang]:
                normalized = re.sub(pattern, '', normalized)

        return normalized.strip()

    def calculate_concept_similarity(self, concept1: str, concept2: str, language: Optional[str] = None) -> float:
        """
        Calculate similarity between concepts using character-level MLCS.

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

        # Convert to character lists for MLCS
        chars1 = list(norm1)
        chars2 = list(norm2)

        # Apply MLCS algorithm to find the longest common subsequence
        mlcs = self.mlcs_processor.find_mlcs([chars1, chars2])

        # Calculate similarity based on MLCS length relative to the shorter string
        # This approach makes the algorithm more tolerant to affixes and variations
        mlcs_length = len(mlcs)
        min_length = min(len(norm1), len(norm2))

        if min_length == 0:
            return 0.0

        # Normalize by the shorter string length for better tolerance to variations
        similarity = mlcs_length / min_length

        return similarity

    def is_valid_concept(self, text: str, language: Optional[str] = None) -> bool:
        """
        Check if text represents a valid concept.

        Args:
            text: Concept text
            language: Language code

        Returns:
            True if valid concept, False otherwise
        """
        normalized = self.normalize_concept_text(text, language)

        if not normalized:
            return False

        # Check minimum length
        if len(normalized) < 3:
            return False

        # Check word count
        words = normalized.split()
        word_count = len(words)

        # Valid concept typically has 1-5 words
        if word_count < 1 or word_count > 5:
            return False

        # Check if it's mostly numbers
        if sum(c.isdigit() for c in normalized) / len(normalized) > 0.3:
            return False

        return True

    def find_similar_concepts(
        self,
        concept: Dict[str, Any],
        concept_list: List[Dict[str, Any]],
        threshold: float = 0.80,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar concepts using character-level MLCS comparison.

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

            # Calculate similarity using character-level MLCS
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
        Deduplicate and merge similar concepts using clustering.

        Args:
            concepts: List of concept dictionaries
            language: Language code

        Returns:
            Deduplicated list of concepts
        """
        if not concepts:
            return []

        # Filter out invalid concepts
        lang = language or next((c.get("language") for c in concepts if "language" in c), self.language)
        valid_concepts = [c for c in concepts if self.is_valid_concept(c.get("text", ""), lang)]

        # If all concepts were filtered out, return original concepts
        if not valid_concepts and concepts:
            logger.warning(f"All concepts were filtered out as invalid. Returning top original concepts.")
            sorted_concepts = sorted(concepts, key=lambda x: (x.get("score", 0), x.get("frequency", 0)), reverse=True)
            return sorted_concepts[:20]  # Return only top 20

        if not valid_concepts:
            return []

        # Use clustering to group similar concepts efficiently
        clusters = self._cluster_similar_concepts(valid_concepts, lang)

        # Select best concept from each cluster
        merged_concepts = self._select_canonical_concepts(clusters)

        # Sort by frequency and score
        merged_concepts.sort(key=lambda x: (x.get("frequency", 1) * 2 + x.get("score", 0)), reverse=True)

        return merged_concepts

    def _cluster_similar_concepts(self, concepts: List[Dict[str, Any]], language: str) -> List[List[Dict[str, Any]]]:
        """
        Group similar concepts into clusters using efficient algorithm.

        Args:
            concepts: List of concept dictionaries
            language: Language code

        Returns:
            List of concept clusters
        """
        # Similarity threshold for clustering
        threshold = self.similarity_thresholds["medium_match"]

        # Initialize clusters
        clusters = []
        processed_indices = set()

        # Process each concept
        for i, concept in enumerate(concepts):
            if i in processed_indices:
                continue

            # Create a new cluster with this concept as seed
            current_cluster = [concept]
            processed_indices.add(i)

            concept_text = concept.get("text", "")

            # Find other concepts that belong to this cluster
            for j, other in enumerate(concepts):
                if j in processed_indices or j == i:
                    continue

                other_text = other.get("text", "")

                # Calculate similarity with cluster seed
                similarity = self.calculate_concept_similarity(concept_text, other_text, language)

                if similarity >= threshold:
                    current_cluster.append(other)
                    processed_indices.add(j)

            # Add cluster if not empty
            if current_cluster:
                clusters.append(current_cluster)

        return clusters

    def _select_canonical_concepts(self, clusters: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Select the best concept from each cluster as the canonical concept.

        Args:
            clusters: List of concept clusters

        Returns:
            List of canonical concepts
        """
        canonical_concepts = []

        for cluster in clusters:
            if not cluster:
                continue

            # Sort by quality metrics - balancing frequency, score and word count
            cluster.sort(key=lambda c: (
                c.get("frequency", 1) * 0.5 +  # Frequency is most important
                c.get("score", 0) * 0.3 +      # Score is next
                len(c.get("text", "").split()) * 0.2  # Slightly favor multi-word concepts
            ), reverse=True)

            # Use the highest scoring concept as canonical
            best_concept = cluster[0].copy()

            # Accumulate information from variants
            variant_texts = []
            total_frequency = best_concept.get("frequency", 1)

            # Track original categories
            original_categories = set()
            if "category" in best_concept:
                original_categories.add(best_concept["category"])

            for variant in cluster[1:]:
                # Add variant text
                variant_texts.append(variant.get("text", ""))

                # Accumulate frequency
                total_frequency += variant.get("frequency", 1)

                # Track original categories
                if "category" in variant:
                    original_categories.add(variant["category"])

                # Use definition from variant if primary doesn't have one
                if variant.get("definition") and not best_concept.get("definition"):
                    best_concept["definition"] = variant["definition"]

            # Update the canonical concept
            best_concept["frequency"] = total_frequency

            if variant_texts:
                best_concept["variant_texts"] = variant_texts
                best_concept["variants_count"] = len(variant_texts)

            # Store original categories information
            if original_categories:
                best_concept["original_categories"] = list(original_categories)

            canonical_concepts.append(best_concept)

        return canonical_concepts

    def batch_process_concepts(self, concepts: List[Dict[str, Any]], batch_size: int = 500,
                              language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process concepts in batches for better performance with large datasets.

        Args:
            concepts: List of concept dictionaries
            batch_size: Maximum concepts per batch
            language: Language code

        Returns:
            Deduplicated list of concepts
        """
        if len(concepts) <= batch_size:
            return self.deduplicate_concepts(concepts, language)

        # Process in batches
        all_deduplicated = []

        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:min(i+batch_size, len(concepts))]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} concepts)")

            batch_results = self.deduplicate_concepts(batch, language)
            all_deduplicated.extend(batch_results)

        # Final deduplication across batches to consolidate results
        logger.info(f"Performing final deduplication across {len(all_deduplicated)} concepts")
        return self.deduplicate_concepts(all_deduplicated, language)


def apply_concept_deduplication(processed_result: Dict[str, Any], language: str = None) -> Dict[str, Any]:
    """
    Apply unified concept deduplication across theoretical and practical concepts.

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

    # Extract concepts by category
    theoretical_concepts = domain_features.get("theoretical_concepts", [])
    practical_concepts = domain_features.get("practical_concepts", [])

    total_concepts = len(theoretical_concepts) + len(practical_concepts)
    if total_concepts == 0:
        logger.info("No concepts to deduplicate")
        return processed_result  # Nothing to deduplicate

    # Initialize deduplicator
    deduplicator = ConceptDedupExtension(language=language)

    # Track original category for each concept
    all_concepts = []
    concept_origins = {}  # concept_id -> original category

    # Add theoretical concepts with category tracking
    for concept in theoretical_concepts:
        concept_copy = concept.copy()
        concept_copy["category"] = "theoretical_concepts"
        concept_id = concept_copy.get("concept_id")
        if concept_id:
            concept_origins[concept_id] = "theoretical_concepts"
        all_concepts.append(concept_copy)

    # Add practical concepts with category tracking
    for concept in practical_concepts:
        concept_copy = concept.copy()
        concept_copy["category"] = "practical_concepts"
        concept_id = concept_copy.get("concept_id")
        if concept_id:
            concept_origins[concept_id] = "practical_concepts"
        all_concepts.append(concept_copy)

    logger.info(f"Deduplicating {len(all_concepts)} concepts across all categories")

    # Deduplicate all concepts together
    start_time = time.time()
    deduplicated_concepts = deduplicator.batch_process_concepts(all_concepts, language=language)
    dedup_time = time.time() - start_time

    logger.info(f"Deduplicated from {len(all_concepts)} to {len(deduplicated_concepts)} concepts in {dedup_time:.2f}s")

    # Now we need to reconstruct the category-specific lists
    new_theoretical_concepts = []
    new_practical_concepts = []

    # Helper function to check if a concept belongs to a category
    def belongs_to_category(concept, category):
        # Check original category
        if "category" in concept and concept["category"] == category:
            return True

        # Check original_categories list
        if "original_categories" in concept and category in concept["original_categories"]:
            return True

        # Check concept_id in origins
        concept_id = concept.get("concept_id")
        if concept_id and concept_id in concept_origins and concept_origins[concept_id] == category:
            return True

        return False

    # Re-categorize concepts based on original category
    for concept in deduplicated_concepts:
        # Create a clean copy without category tracking
        clean_concept = concept.copy()
        if "category" in clean_concept:
            del clean_concept["category"]
        if "original_categories" in clean_concept:
            del clean_concept["original_categories"]

        # Check if it belongs in theoretical list
        if belongs_to_category(concept, "theoretical_concepts") or concept.get("concept_class") == "theoretical":
            new_theoretical_concepts.append(clean_concept)

        # Check if it belongs in practical list
        if belongs_to_category(concept, "practical_concepts") or concept.get("concept_class") == "practical":
            new_practical_concepts.append(clean_concept)

    # Update domain features with deduplicated concept lists
    domain_features["theoretical_concepts"] = new_theoretical_concepts
    domain_features["practical_concepts"] = new_practical_concepts

    # Log deduplication results for each category
    logger.info(f"Theoretical concepts: {len(theoretical_concepts)} → {len(new_theoretical_concepts)}")
    logger.info(f"Practical concepts: {len(practical_concepts)} → {len(new_practical_concepts)}")

    # Update processed result
    processed_result["domain_features"] = domain_features

    # Add deduplication stats
    processed_result["deduplication_stats"] = {
        "original_total": len(all_concepts),
        "deduplicated_total": len(deduplicated_concepts),
        "reduction_percentage": round((len(all_concepts) - len(deduplicated_concepts)) / len(all_concepts) * 100, 2) if all_concepts else 0,
        "processing_time_seconds": dedup_time,
        "theoretical_concepts_original": len(theoretical_concepts),
        "theoretical_concepts_deduplicated": len(new_theoretical_concepts),
        "practical_concepts_original": len(practical_concepts),
        "practical_concepts_deduplicated": len(new_practical_concepts)
    }

    return processed_result
