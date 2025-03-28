"""
Enhanced concept signature generator for the Lecture Video Content Indexer.
Implements Linear Multiple Longest Common Subsequence (MLCS) algorithms to identify
common patterns across video transcripts and model concept relationships.

This module provides advanced analysis of educational content to:
1. Extract core concept signatures that represent the essence of educational ideas
2. Build relationship graphs between concepts to model dependencies and hierarchies
3. Enhance learning path generation by understanding concept prerequisites
"""

import os
import re
import uuid
import logging
import json
import time
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Counter as CounterType
from collections import defaultdict, Counter, deque
import numpy as np
from datetime import datetime

# Import project modules with error handling
try:
    from data_access import get_data_access
    from cache_manager import cache_get, cache_set, cached
    from performance_utils import time_function, Timer
except ImportError:
    # Handle import errors gracefully for testing
    logging.warning("Could not import one or more project modules - running in limited mode")

# Configure logging
logger = logging.getLogger(__name__)

class ConceptSignature:
    """
    Represents the signature pattern of an educational concept extracted from lectures.

    A concept signature contains the core pattern that characterizes the concept,
    along with metadata about its occurrences and relationships to other concepts.
    """

    def __init__(
        self,
        concept_id: str,
        text: str,
        signature_pattern: Optional[List[str]] = None,
        domain: str = "unknown",
        concept_class: str = "theoretical",
        language: str = "en",
        confidence: float = 0.0
    ):
        """
        Initialize a concept signature.

        Args:
            concept_id: Unique identifier for the concept
            text: The concept text
            signature_pattern: List of terms that form the concept signature pattern
            domain: Domain the concept belongs to
            concept_class: Classification (theoretical or practical)
            language: Language code
            confidence: Confidence score for this signature (0.0-1.0)
        """
        self.concept_id = concept_id
        self.text = text
        self.signature_pattern = signature_pattern or self._extract_pattern(text)
        self.domain = domain
        self.concept_class = concept_class
        self.language = language
        self.confidence = confidence
        self.occurrences = []  # List of occurrences in videos
        self.related_concepts = {}  # Mapping of related_concept_id -> relationship_strength
        self.hierarchy_score = 0.0  # Higher values indicate more fundamental concepts
        self.generality_score = 0.0  # Higher values indicate more general concepts
        self.specificity_score = 0.0  # Higher values indicate more specific concepts
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def _extract_pattern(self, text: str) -> List[str]:
        """
        Extract a preliminary pattern from concept text.

        Args:
            text: The concept text

        Returns:
            List of terms in the pattern
        """
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) > 2]

    def add_occurrence(self, video_id: str, segment_id: str, start_time: float,
                      end_time: float, context_type: str, context_text: str) -> None:
        """
        Add an occurrence of this concept in a video.

        Args:
            video_id: Video identifier
            segment_id: Segment identifier
            start_time: Start time in the video
            end_time: End time in the video
            context_type: Context type (theoretical, practical, mixed)
            context_text: The surrounding text
        """
        occurrence = {
            "occurrence_id": str(uuid.uuid4()),
            "video_id": video_id,
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "context_type": context_type,
            "context_text": context_text,
            "added_at": datetime.now().isoformat()
        }
        self.occurrences.append(occurrence)
        self.updated_at = datetime.now().isoformat()

    def add_relationship(self, related_concept_id: str, strength: float,
                        relationship_type: str = "related") -> None:
        """
        Add a relationship to another concept.

        Args:
            related_concept_id: ID of the related concept
            strength: Relationship strength (0.0-1.0)
            relationship_type: Type of relationship
        """
        self.related_concepts[related_concept_id] = {
            "strength": strength,
            "type": relationship_type,
            "added_at": datetime.now().isoformat()
        }
        self.updated_at = datetime.now().isoformat()

    def calculate_hierarchy_score(self, all_concepts: Dict[str, 'ConceptSignature']) -> float:
        """
        Calculate a hierarchy score to determine concept's place in a concept hierarchy.
        Higher scores indicate more fundamental concepts.

        Args:
            all_concepts: Dictionary of all relevant concepts

        Returns:
            Hierarchy score
        """
        # Base score from signature pattern length (shorter = more fundamental)
        if not self.signature_pattern:
            return 0.0

        # More general concepts tend to have shorter patterns
        pattern_score = 1.0 / max(len(self.signature_pattern), 1)

        # Concepts related to many others are more fundamental
        relationship_count = len(self.related_concepts)
        relationship_score = min(relationship_count / 10.0, 1.0)

        # More occurrences may indicate more fundamental concept
        occurrence_score = min(len(self.occurrences) / 20.0, 1.0)

        # Concepts that are referenced by many others are more fundamental
        reference_count = sum(1 for c in all_concepts.values()
                            if self.concept_id in c.related_concepts)
        reference_score = min(reference_count / 5.0, 1.0)

        # Combine scores with weights
        self.hierarchy_score = (
            pattern_score * 0.2 +
            relationship_score * 0.3 +
            occurrence_score * 0.2 +
            reference_score * 0.3
        )

        return self.hierarchy_score

    def match_text(self, text: str) -> float:
        """
        Calculate how well this concept signature matches a text.

        Args:
            text: The text to match against

        Returns:
            Match score (0.0-1.0)
        """
        if not self.signature_pattern:
            return 0.0

        text_lower = text.lower()

        # Count pattern terms that appear in the text
        matches = sum(1 for term in self.signature_pattern if term in text_lower)

        # Calculate match percentage
        if not self.signature_pattern:
            return 0.0

        return matches / len(self.signature_pattern)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "concept_id": self.concept_id,
            "text": self.text,
            "signature_pattern": self.signature_pattern,
            "domain": self.domain,
            "concept_class": self.concept_class,
            "language": self.language,
            "confidence": self.confidence,
            "hierarchy_score": self.hierarchy_score,
            "generality_score": self.generality_score,
            "specificity_score": self.specificity_score,
            "related_concepts": self.related_concepts,
            "occurrences_count": len(self.occurrences),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptSignature':
        """
        Create a ConceptSignature from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ConceptSignature instance
        """
        signature = cls(
            concept_id=data["concept_id"],
            text=data["text"],
            signature_pattern=data.get("signature_pattern"),
            domain=data.get("domain", "unknown"),
            concept_class=data.get("concept_class", "theoretical"),
            language=data.get("language", "en"),
            confidence=data.get("confidence", 0.0)
        )

        signature.hierarchy_score = data.get("hierarchy_score", 0.0)
        signature.generality_score = data.get("generality_score", 0.0)
        signature.specificity_score = data.get("specificity_score", 0.0)
        signature.related_concepts = data.get("related_concepts", {})
        signature.created_at = data.get("created_at", signature.created_at)
        signature.updated_at = data.get("updated_at", signature.updated_at)

        return signature


class RelationshipGraph:
    """
    Models relationships between concepts as a directed graph.

    The relationship graph tracks dependencies, prerequisites, and semantic
    relationships between concepts to support learning path generation and
    concept exploration.
    """

    def __init__(self):
        """Initialize the relationship graph."""
        self.concepts = {}  # concept_id -> ConceptSignature
        self.adjacency_list = defaultdict(set)  # concept_id -> set of related concept_ids
        self.edge_attributes = {}  # (source_id, target_id) -> edge attributes
        self.domain_clusters = defaultdict(set)  # domain -> set of concept_ids

    def add_concept(self, concept: ConceptSignature) -> None:
        """
        Add a concept to the graph.

        Args:
            concept: ConceptSignature to add
        """
        self.concepts[concept.concept_id] = concept
        self.domain_clusters[concept.domain].add(concept.concept_id)

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = "related",
        strength: float = 0.5
    ) -> None:
        """
        Add a directed relationship between concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship
            strength: Relationship strength (0.0-1.0)
        """
        # Ensure both concepts exist
        if source_id not in self.concepts or target_id not in self.concepts:
            logger.warning(f"Cannot add relationship: one or both concepts don't exist: {source_id} -> {target_id}")
            return

        # Add to adjacency list
        self.adjacency_list[source_id].add(target_id)

        # Set edge attributes
        self.edge_attributes[(source_id, target_id)] = {
            "type": relationship_type,
            "strength": strength,
            "created_at": datetime.now().isoformat()
        }

        # Update the concept's related_concepts
        self.concepts[source_id].add_relationship(target_id, strength, relationship_type)

    def get_related_concepts(self, concept_id: str, min_strength: float = 0.0) -> List[Tuple[str, Dict]]:
        """
        Get all concepts related to the given concept.

        Args:
            concept_id: Concept ID
            min_strength: Minimum relationship strength

        Returns:
            List of (related_concept_id, edge_attributes) tuples
        """
        if concept_id not in self.concepts:
            return []

        return [
            (target_id, self.edge_attributes.get((concept_id, target_id), {}))
            for target_id in self.adjacency_list[concept_id]
            if self.edge_attributes.get((concept_id, target_id), {}).get("strength", 0.0) >= min_strength
        ]

    def find_prerequisites(self, concept_id: str) -> List[str]:
        """
        Find prerequisites for a concept - concepts that should be learned before this one.

        Args:
            concept_id: Concept ID

        Returns:
            List of prerequisite concept IDs
        """
        if concept_id not in self.concepts:
            return []

        # Find concepts that this concept depends on
        prerequisites = []

        for source_id, targets in self.adjacency_list.items():
            if concept_id in targets:
                edge_attr = self.edge_attributes.get((source_id, concept_id), {})
                if edge_attr.get("type") == "prerequisite":
                    prerequisites.append(source_id)

        # Sort by hierarchy score (most fundamental first)
        sorted_prereqs = sorted(
            prerequisites,
            key=lambda cid: self.concepts[cid].hierarchy_score if cid in self.concepts else 0,
            reverse=True
        )

        return sorted_prereqs

    def generate_learning_path(
        self,
        concept_ids: List[str],
        theory_practice_ratio: float = 0.5,
        max_concepts: int = 20
    ) -> List[str]:
        """
        Generate an optimized learning path through the given concepts.

        Args:
            concept_ids: List of target concept IDs
            theory_practice_ratio: Ratio of theoretical to practical concepts (0.0-1.0)
            max_concepts: Maximum number of concepts in the path

        Returns:
            Ordered list of concept IDs representing the learning path
        """
        if not concept_ids:
            return []

        # Filter to existing concepts
        target_concepts = [cid for cid in concept_ids if cid in self.concepts]

        if not target_concepts:
            return []

        # Collect all required prerequisites
        all_prerequisites = set()
        for concept_id in target_concepts:
            if concept_id in self.concepts:
                # Recursively get prerequisites
                prereqs = self._get_all_prerequisites(concept_id)
                all_prerequisites.update(prereqs)

        # Combine target concepts and prerequisites
        all_concepts = target_concepts + list(all_prerequisites - set(target_concepts))

        # Limit concepts based on hierarchy score and theory/practice ratio
        theoretical = []
        practical = []

        for concept_id in all_concepts:
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                if concept.concept_class == "theoretical":
                    theoretical.append(concept_id)
                else:
                    practical.append(concept_id)

        # Sort by hierarchy score
        theoretical.sort(key=lambda cid: self.concepts[cid].hierarchy_score if cid in self.concepts else 0, reverse=True)
        practical.sort(key=lambda cid: self.concepts[cid].hierarchy_score if cid in self.concepts else 0, reverse=True)

        # Determine count of each type based on ratio
        total_concept_count = min(len(all_concepts), max_concepts)
        theoretical_count = int(total_concept_count * theory_practice_ratio)
        practical_count = total_concept_count - theoretical_count

        # Cap to available concepts
        theoretical_count = min(theoretical_count, len(theoretical))
        practical_count = min(practical_count, len(practical))

        # Select concepts
        selected_theoretical = theoretical[:theoretical_count]
        selected_practical = practical[:practical_count]

        # Ensure target concepts are included
        for concept_id in target_concepts:
            if concept_id not in selected_theoretical and concept_id not in selected_practical:
                # Determine concept class
                if concept_id in self.concepts:
                    if self.concepts[concept_id].concept_class == "theoretical":
                        selected_theoretical.append(concept_id)
                    else:
                        selected_practical.append(concept_id)

        # Combine concepts and sort them in topological order
        learning_path = self._topological_sort(selected_theoretical + selected_practical)

        return learning_path

    def _get_all_prerequisites(self, concept_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """
        Recursively get all prerequisites for a concept.

        Args:
            concept_id: Concept ID
            visited: Set of already visited concept IDs

        Returns:
            Set of prerequisite concept IDs
        """
        if visited is None:
            visited = set()

        if concept_id in visited:
            return set()

        visited.add(concept_id)
        prerequisites = set(self.find_prerequisites(concept_id))

        for prereq_id in list(prerequisites):
            prerequisites.update(self._get_all_prerequisites(prereq_id, visited))

        return prerequisites

    def _topological_sort(self, concept_ids: List[str]) -> List[str]:
        """
        Perform a topological sort of concepts based on their dependencies.

        Args:
            concept_ids: List of concept IDs to sort

        Returns:
            Topologically sorted list of concept IDs
        """
        # Extract relevant subgraph
        subgraph = defaultdict(set)
        for source_id in concept_ids:
            for target_id in self.adjacency_list[source_id]:
                if target_id in concept_ids:
                    subgraph[source_id].add(target_id)

        # Count incoming edges
        incoming_count = {concept_id: 0 for concept_id in concept_ids}
        for source_id, targets in subgraph.items():
            for target_id in targets:
                incoming_count[target_id] = incoming_count.get(target_id, 0) + 1

        # Queue concepts with no incoming edges
        queue = deque([cid for cid in concept_ids if incoming_count.get(cid, 0) == 0])
        result = []

        # Process the queue
        while queue:
            concept_id = queue.popleft()
            result.append(concept_id)

            # Remove edges from this node
            for target_id in subgraph.get(concept_id, set()):
                incoming_count[target_id] -= 1
                if incoming_count[target_id] == 0:
                    queue.append(target_id)

        # If result doesn't include all concepts, there was a cycle
        # In this case, add remaining concepts in order of hierarchy score
        if len(result) < len(concept_ids):
            remaining = set(concept_ids) - set(result)
            remaining_sorted = sorted(
                remaining,
                key=lambda cid: self.concepts[cid].hierarchy_score if cid in self.concepts else 0,
                reverse=True
            )
            result.extend(remaining_sorted)

        return result

    def calculate_all_hierarchy_scores(self) -> None:
        """Calculate hierarchy scores for all concepts in the graph."""
        for concept in self.concepts.values():
            concept.calculate_hierarchy_score(self.concepts)

    def save_to_json(self, filepath: str) -> None:
        """
        Save the relationship graph to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        data = {
            "concepts": {
                concept_id: concept.to_dict()
                for concept_id, concept in self.concepts.items()
            },
            "edges": [
                {
                    "source": source_id,
                    "target": target_id,
                    "attributes": attributes
                }
                for (source_id, target_id), attributes in self.edge_attributes.items()
            ],
            "metadata": {
                "concept_count": len(self.concepts),
                "relationship_count": len(self.edge_attributes),
                "created_at": datetime.now().isoformat()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'RelationshipGraph':
        """
        Load a relationship graph from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            RelationshipGraph instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = cls()

        # Load concepts
        for concept_id, concept_data in data.get("concepts", {}).items():
            concept = ConceptSignature.from_dict(concept_data)
            graph.add_concept(concept)

        # Load edges
        for edge_data in data.get("edges", []):
            source_id = edge_data.get("source")
            target_id = edge_data.get("target")
            attributes = edge_data.get("attributes", {})

            if source_id and target_id:
                graph.adjacency_list[source_id].add(target_id)
                graph.edge_attributes[(source_id, target_id)] = attributes

        return graph

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the relationship graph to a dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "concepts": {
                concept_id: concept.to_dict()
                for concept_id, concept in self.concepts.items()
            },
            "edges": [
                {
                    "source": source_id,
                    "target": target_id,
                    "attributes": attributes
                }
                for (source_id, target_id), attributes in self.edge_attributes.items()
            ],
            "metadata": {
                "concept_count": len(self.concepts),
                "relationship_count": len(self.edge_attributes),
                "domains": list(self.domain_clusters.keys()),
                "timestamp": datetime.now().isoformat()
            }
        }


class MLCSProcessor:
    """
    Implements Linear Multiple Longest Common Subsequence algorithms to identify
    common patterns across educational transcripts.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the MLCS processor.

        Args:
            language: Language code
        """
        self.language = language
        self.stopwords = self._load_stopwords(language)

    def _load_stopwords(self, language: str) -> Set[str]:
        """
        Load stopwords for the specified language.

        Args:
            language: Language code

        Returns:
            Set of stopwords
        """
        try:
            import nltk
            from nltk.corpus import stopwords

            # Ensure stopwords are downloaded
            try:
                stopwords.words(self._map_language_to_nltk(language))
            except LookupError:
                nltk.download('stopwords', quiet=True)

            # Get stopwords for language
            return set(stopwords.words(self._map_language_to_nltk(language)))
        except (ImportError, LookupError):
            # Fallback to basic English stopwords
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should',
                'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
            }

    def _map_language_to_nltk(self, language: str) -> str:
        """
        Map language code to NLTK language name.

        Args:
            language: Language code

        Returns:
            NLTK language name
        """
        language_map = {
            "en": "english",
            "ru": "russian",
            # Add more mappings as needed
        }
        return language_map.get(language, "english")

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization and normalization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]

    def extract_significant_sequences(
        self,
        texts: List[str],
        min_length: int = 3,
        min_frequency: int = 2
    ) -> List[Tuple[List[str], float]]:
        """
        Extract significant common subsequences from multiple texts.

        Args:
            texts: List of text strings
            min_length: Minimum subsequence length
            min_frequency: Minimum frequency to consider significant

        Returns:
            List of (subsequence, score) tuples
        """
        # Tokenize texts
        tokenized_texts = [self._tokenize_text(text) for text in texts]

        # Find n-grams in each text
        all_ngrams = []

        for tokens in tokenized_texts:
            text_ngrams = set()

            for n in range(min_length, min(len(tokens), 10) + 1):  # Limit n-gram size
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    text_ngrams.add(ngram)

            all_ngrams.append(text_ngrams)

        # Count frequencies across texts
        ngram_counts = Counter()

        for text_ngrams in all_ngrams:
            # Count each n-gram only once per text
            for ngram in text_ngrams:
                ngram_counts[ngram] += 1

        # Filter by frequency and sort by score
        significant_ngrams = []

        for ngram, count in ngram_counts.items():
            if count >= min_frequency:
                # Score based on length and frequency
                score = len(ngram) * (count / len(texts))
                significant_ngrams.append((list(ngram), score))

        # Sort by score
        significant_ngrams.sort(key=lambda x: x[1], reverse=True)

        return significant_ngrams

    def find_mlcs(self, sequences: List[List[str]]) -> List[str]:
        """
        Find the Multiple Longest Common Subsequence across sequences.

        Args:
            sequences: List of token sequences

        Returns:
            MLCS as a list of tokens
        """
        if not sequences:
            return []

        if len(sequences) == 1:
            return sequences[0]

        # For simplicity, use a pairwise approach for MLCS
        result = sequences[0]

        for i in range(1, len(sequences)):
            result = self._lcs(result, sequences[i])

            # If LCS becomes empty, return
            if not result:
                return []

        return result

    def _lcs(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """
        Find the Longest Common Subsequence between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            LCS as a list of tokens
        """
        # Dynamic programming approach
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Backtrack to find LCS
        i, j = m, n
        lcs = []

        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                lcs.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        return list(reversed(lcs))

    def generate_concept_signatures(
        self,
        concepts: List[Dict[str, Any]],
        segments: List[Dict[str, Any]]
    ) -> List[ConceptSignature]:
        """
        Generate concept signatures using MLCS from concept occurrences.

        Args:
            concepts: List of concept dictionaries
            segments: List of transcript segments

        Returns:
            List of ConceptSignature instances
        """
        signatures = []

        for concept in concepts:
            # Create basic signature
            signature = ConceptSignature(
                concept_id=concept.get("concept_id", str(uuid.uuid4())),
                text=concept.get("text", ""),
                domain=concept.get("domain", "unknown"),
                concept_class=concept.get("concept_class", "theoretical"),
                language=concept.get("language", self.language)
            )

            # Find concept occurrences in segments
            concept_text = concept.get("text", "").lower()

            # Collect context texts for this concept
            contexts = []

            for segment in segments:
                segment_text = segment.get("text", "").lower()
                if concept_text in segment_text:
                    context_text = segment.get("text", "")
                    contexts.append(context_text)

                    # Add occurrence to signature
                    signature.add_occurrence(
                        video_id=segment.get("video_id", ""),
                        segment_id=segment.get("id", ""),
                        start_time=segment.get("start_time", 0.0),
                        end_time=segment.get("end_time", 0.0),
                        context_type=segment.get("content_type", "mixed"),
                        context_text=context_text
                    )

            # Extract significant sequences from contexts
            if contexts:
                significant_sequences = self.extract_significant_sequences(
                    contexts, min_length=2, min_frequency=max(2, len(contexts) // 3)
                )

                if significant_sequences:
                    # Use the highest scoring sequence as the signature pattern
                    signature.signature_pattern, score = significant_sequences[0]
                    signature.confidence = min(score / 10.0, 1.0)  # Normalize confidence

            signatures.append(signature)

        return signatures


class ConceptSignatureGenerator:
    """
    Generates concept signatures and relationship graphs from processed video data.

    This main class integrates with the data pipeline and search engine to enhance
    concept extraction and learning path generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the concept signature generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "data/index")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize processor
        self.mlcs_processor = MLCSProcessor()

        # Initialize graph
        self.relationship_graph = RelationshipGraph()

        # Try to initialize data access
        try:
            self.data_access = get_data_access()
            logger.info("Data access initialized for ConceptSignatureGenerator")
        except (NameError, ImportError):
            self.data_access = None
            logger.warning("Data access not available - running in limited mode")

        # Load existing graph if available
        self._load_relationship_graph()

    def _load_relationship_graph(self) -> None:
        """Load existing relationship graph if available."""
        graph_path = os.path.join(self.output_dir, "relationship_graph.json")

        if os.path.exists(graph_path):
            try:
                self.relationship_graph = RelationshipGraph.load_from_json(graph_path)
                logger.info(f"Loaded relationship graph with {len(self.relationship_graph.concepts)} concepts")
            except Exception as e:
                logger.error(f"Error loading relationship graph: {e}")
                self.relationship_graph = RelationshipGraph()

    def _save_relationship_graph(self) -> None:
        """Save relationship graph to file."""
        graph_path = os.path.join(self.output_dir, "relationship_graph.json")

        try:
            self.relationship_graph.save_to_json(graph_path)
            logger.info(f"Saved relationship graph with {len(self.relationship_graph.concepts)} concepts")
        except Exception as e:
            logger.error(f"Error saving relationship graph: {e}")

    @time_function(10000)  # Log warning if takes more than 10 seconds
    def process_video_concepts(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process concepts extracted from a video to identify signatures and relationships.

        Args:
            processing_result: Video processing result from DataPipeline

        Returns:
            Enhanced processing result
        """
        video_id = processing_result.get("video_id")
        if not video_id:
            logger.warning("Missing video_id in processing result")
            return processing_result

        # Extract domain features and transcript segments
        domain_features = processing_result.get("domain_features", {})
        segments = processing_result.get("transcript", {}).get("segments", [])

        # Process concepts
        key_concepts = domain_features.get("key_concepts", [])

        if not key_concepts or not segments:
            logger.warning(f"No concepts or segments found for video {video_id}")
            return processing_result

        # Set language based on transcript
        language = processing_result.get("transcript", {}).get("language", "en")
        self.mlcs_processor.language = language

        # Generate concept signatures
        signatures = self.mlcs_processor.generate_concept_signatures(key_concepts, segments)

        # Add to relationship graph
        for signature in signatures:
            # Add video_id to the concept ID to avoid duplicates
            signature.concept_id = f"{signature.concept_id}_{video_id}"

            # Add to graph
            self.relationship_graph.add_concept(signature)

        # Identify relationships between concepts
        self._identify_concept_relationships(signatures, segments)

        # Update hierarchy scores
        self.relationship_graph.calculate_all_hierarchy_scores()

        # Enhance domain features with signature information
        enhanced_key_concepts = []

        for i, concept in enumerate(key_concepts):
            if i < len(signatures):
                # Add signature information to concept
                enhanced_concept = concept.copy()
                enhanced_concept["signature_pattern"] = signatures[i].signature_pattern
                enhanced_concept["hierarchy_score"] = signatures[i].hierarchy_score
                enhanced_concept["confidence"] = signatures[i].confidence
                enhanced_concept["related_concepts"] = [
                    {"id": rel_id, "strength": rel_data["strength"], "type": rel_data["type"]}
                    for rel_id, rel_data in signatures[i].related_concepts.items()
                ]

                enhanced_key_concepts.append(enhanced_concept)
            else:
                enhanced_key_concepts.append(concept)

        # Update domain features
        domain_features["key_concepts"] = enhanced_key_concepts
        domain_features["concept_signatures"] = [signature.to_dict() for signature in signatures]

        # Save relationship graph
        self._save_relationship_graph()

        # Return enhanced result
        processing_result["domain_features"] = domain_features
        return processing_result

    def _identify_concept_relationships(
        self,
        signatures: List[ConceptSignature],
        segments: List[Dict[str, Any]]
    ) -> None:
        """
        Identify relationships between concepts within a video.

        Args:
            signatures: List of concept signatures
            segments: List of transcript segments
        """
        # Extract segment information
        segment_texts = [segment.get("text", "") for segment in segments]
        segment_types = [segment.get("content_type", "mixed") for segment in segments]
        segment_ids = [segment.get("id", "") for segment in segments]

        # Build segment index
        segment_order = {segment_id: i for i, segment_id in enumerate(segment_ids) if segment_id}

        # Track co-occurrences of concepts in segments
        co_occurrences = defaultdict(int)
        concept_segment_map = defaultdict(set)

        # Map concepts to their segments
        for signature in signatures:
            for occurrence in signature.occurrences:
                segment_id = occurrence.get("segment_id")
                if segment_id and segment_id in segment_order:
                    concept_segment_map[signature.concept_id].add(segment_id)

        # Count co-occurrences
        for i, concept1 in enumerate(signatures):
            for j, concept2 in enumerate(signatures):
                if i != j:
                    # Get shared segments
                    shared_segments = concept_segment_map[concept1.concept_id].intersection(
                        concept_segment_map[concept2.concept_id]
                    )

                    co_occurrences[(concept1.concept_id, concept2.concept_id)] = len(shared_segments)

        # Analyze proximity and ordering
        for i, concept1 in enumerate(signatures):
            for j, concept2 in enumerate(signatures):
                if i != j:
                    # Skip if no co-occurrences
                    if co_occurrences[(concept1.concept_id, concept2.concept_id)] == 0:
                        continue

                    # Determine concept hierarchy
                    strength = min(co_occurrences[(concept1.concept_id, concept2.concept_id)] / 5.0, 1.0)

                    # Check segment order to determine prerequisites
                    concept1_segments = list(concept_segment_map[concept1.concept_id])
                    concept2_segments = list(concept_segment_map[concept2.concept_id])

                    # Get earliest segment for each concept
                    concept1_earliest = min(segment_order[sid] for sid in concept1_segments if sid in segment_order)
                    concept2_earliest = min(segment_order[sid] for sid in concept2_segments if sid in segment_order)

                    # If concept1 appears significantly before concept2, it might be a prerequisite
                    if concept1_earliest < concept2_earliest - 3:  # At least 3 segments earlier
                        self.relationship_graph.add_relationship(
                            concept1.concept_id, concept2.concept_id, "prerequisite", strength
                        )
                    else:
                        # Otherwise just a related concept
                        self.relationship_graph.add_relationship(
                            concept1.concept_id, concept2.concept_id, "related", strength
                        )

    def generate_enhanced_learning_path(
        self,
        concept_ids: List[str],
        theory_practice_ratio: float = 0.5,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an enhanced learning path with concept signatures and relationships.

        Args:
            concept_ids: List of concept IDs
            theory_practice_ratio: Theory/practice ratio preference
            domain: Optional domain filter

        Returns:
            Enhanced learning path dictionary
        """
        # Ensure relationship graph is loaded
        if not self.relationship_graph.concepts:
            self._load_relationship_graph()

        # Filter concepts by domain if specified
        if domain:
            domain_concepts = self.relationship_graph.domain_clusters.get(domain, set())
            filtered_concept_ids = [cid for cid in concept_ids if cid in domain_concepts]
        else:
            filtered_concept_ids = concept_ids

        # Generate learning path
        path_concept_ids = self.relationship_graph.generate_learning_path(
            filtered_concept_ids, theory_practice_ratio
        )

        # Create learning path result
        path_concepts = []
        for concept_id in path_concept_ids:
            if concept_id in self.relationship_graph.concepts:
                concept = self.relationship_graph.concepts[concept_id]

                # Find prerequisites
                prerequisites = self.relationship_graph.find_prerequisites(concept_id)

                # Add to path
                path_concepts.append({
                    "concept_id": concept_id,
                    "text": concept.text,
                    "concept_class": concept.concept_class,
                    "domain": concept.domain,
                    "hierarchy_score": concept.hierarchy_score,
                    "confidence": concept.confidence,
                    "signature_pattern": concept.signature_pattern,
                    "prerequisites": prerequisites,
                    "related_concepts": [
                        rel_id for rel_id, _ in
                        self.relationship_graph.get_related_concepts(concept_id, min_strength=0.3)
                    ]
                })

        # Create learning path structure
        learning_path = {
            "concepts": path_concepts,
            "concept_count": len(path_concepts),
            "theoretical_concepts": sum(1 for c in path_concepts if c["concept_class"] == "theoretical"),
            "practical_concepts": sum(1 for c in path_concepts if c["concept_class"] == "practical"),
            "theory_practice_ratio": theory_practice_ratio,
            "domain": domain,
            "generated_at": datetime.now().isoformat()
        }

        return learning_path

    def store_concept_signatures(self) -> bool:
        """
        Store concept signatures and relationships in the database.

        Returns:
            True if successful, False otherwise
        """
        if not self.data_access:
            logger.warning("Data access not available - cannot store concept signatures")
            return False

        try:
            # Get all concept signatures
            signatures = list(self.relationship_graph.concepts.values())

            # Store each signature
            for signature in signatures:
                # Convert to database format
                concept_data = {
                    "concept_id": signature.concept_id,
                    "text": signature.text,
                    "domain": signature.domain,
                    "concept_class": signature.concept_class,
                    "language": signature.language,
                    "metadata": {
                        "signature_pattern": signature.signature_pattern,
                        "hierarchy_score": signature.hierarchy_score,
                        "confidence": signature.confidence,
                        "related_concepts": signature.related_concepts
                    }
                }

                # Store concept
                self.data_access.save_concept(concept_data)

                # Store occurrences
                occurrences = [
                    {
                        "occurrence_id": occurrence["occurrence_id"],
                        "video_id": occurrence["video_id"],
                        "segment_id": occurrence["segment_id"],
                        "start_time": occurrence["start_time"],
                        "end_time": occurrence["end_time"],
                        "context_type": occurrence["context_type"],
                        "context_text": occurrence["context_text"]
                    }
                    for occurrence in signature.occurrences
                ]

                if occurrences:
                    self.data_access.save_occurrences(signature.concept_id, occurrences)

            # Store relationship graph structure
            graph_data = self.relationship_graph.to_dict()
            graph_path = os.path.join(self.output_dir, "relationship_graph.json")

            with open(graph_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Stored {len(signatures)} concept signatures and relationships")
            return True

        except Exception as e:
            logger.error(f"Error storing concept signatures: {e}")
            return False

    def enhance_search_engine_path(self, search_engine, path_generator_func):
        """
        Enhance the search engine's learning path generation.

        Args:
            search_engine: SearchEngine instance
            path_generator_func: Original path generator function

        Returns:
            Enhanced path generator function
        """
        def enhanced_path_generator(concept_ids, theory_practice_ratio=0.5, domain=None):
            """Enhanced learning path generator function."""
            # First, generate the basic learning path
            base_path = path_generator_func(concept_ids, theory_practice_ratio, domain)

            # Then enhance it with concept signatures and relationships
            enhanced_path = self.generate_enhanced_learning_path(
                concept_ids, theory_practice_ratio, domain
            )

            # Combine the results
            if base_path and enhanced_path:
                # Combine concepts
                base_concepts = {c.get("concept_id"): c for c in base_path.get("concepts", [])}
                enhanced_concepts = {c.get("concept_id"): c for c in enhanced_path.get("concepts", [])}

                # Merge concepts
                for concept_id, concept in enhanced_concepts.items():
                    if concept_id in base_concepts:
                        # Update base concept with enhanced information
                        base_concepts[concept_id].update({
                            "hierarchy_score": concept.get("hierarchy_score", 0.0),
                            "signature_pattern": concept.get("signature_pattern", []),
                            "prerequisites": concept.get("prerequisites", []),
                            "related_concepts": concept.get("related_concepts", [])
                        })
                    else:
                        # Add enhanced concept to base
                        base_path["concepts"].append(concept)

                # Re-order concepts based on prerequisites
                if enhanced_path.get("concepts"):
                    base_path["concepts"] = enhanced_path["concepts"]

                # Update metadata
                base_path["enhanced"] = True
                base_path["concept_signatures_used"] = True

                return base_path

            # Fallback to original path if enhancement failed
            return base_path or enhanced_path

        # Replace the function
        search_engine.generate_learning_path = enhanced_path_generator

        return enhanced_path_generator


# Helper Functions

def get_concept_signature_generator(config: Optional[Dict[str, Any]] = None) -> ConceptSignatureGenerator:
    """
    Get or create the ConceptSignatureGenerator instance.

    Args:
        config: Configuration dictionary

    Returns:
        ConceptSignatureGenerator instance
    """
    global _concept_signature_generator

    if '_concept_signature_generator' not in globals():
        _concept_signature_generator = ConceptSignatureGenerator(config)

    return _concept_signature_generator

def enhance_data_pipeline(data_pipeline):
    """
    Enhance the DataPipeline with concept signature generation.

    Args:
        data_pipeline: DataPipeline instance

    Returns:
        Enhanced DataPipeline
    """
    # Get or create generator
    generator = get_concept_signature_generator(data_pipeline.config)

    # Save original method
    original_process_video = data_pipeline.process_video

    # Define enhanced method
    def enhanced_process_video(url, language_preference=None):
        """Enhanced video processing with concept signatures."""
        # Run original processing
        result = original_process_video(url, language_preference)

        # Enhance with concept signatures
        if result.get("status") == "completed":
            try:
                enhanced_result = generator.process_video_concepts(result)
                logger.info(f"Enhanced video processing with concept signatures for {url}")
                return enhanced_result
            except Exception as e:
                logger.error(f"Error enhancing video with concept signatures: {e}")

        return result

    # Replace method
    data_pipeline.process_video = enhanced_process_video

    return data_pipeline

def enhance_search_engine(search_engine):
    """
    Enhance the SearchEngine with improved learning path generation.

    Args:
        search_engine: SearchEngine instance

    Returns:
        Enhanced SearchEngine
    """
    # Get or create generator
    generator = get_concept_signature_generator(search_engine.config)

    # Enhance learning path generation
    original_generate_learning_path = search_engine.generate_learning_path
    enhanced_function = generator.enhance_search_engine_path(
        search_engine, original_generate_learning_path
    )

    logger.info("Enhanced search engine with improved learning path generation")

    return search_engine
