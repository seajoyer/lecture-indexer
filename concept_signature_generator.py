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
import difflib
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Counter as CounterType
from collections import defaultdict, Counter, deque
import math
from datetime import datetime

# Import MLCS algorithm
from mlcs_algorithm import MLCSAlgorithm

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
        self.educational_weight = 0.0  # Measure of educational significance (vs passing mention)

        # New fields for improved concept matching
        self.normalized_text = self._normalize_text(text, language)
        self.canonical_concept_id = None  # Reference to canonical concept if this is a variant

    def _normalize_text(self, text: str, language: str) -> str:
        """
        Normalize concept text for better matching and deduplication.

        Args:
            text: Original concept text
            language: Language code

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Remove common filler phrases based on language
        if language == "ru":
            # Russian filler phrases to remove
            normalized = re.sub(r'\bэто\s+', '', normalized)   # "это " (this is)
            normalized = re.sub(r'\bвот\s+', '', normalized)   # "вот " (here)
            normalized = re.sub(r'\bда\s+', '', normalized)    # "да " (yes)
            normalized = re.sub(r'\bну\s+', '', normalized)    # "ну " (well)
            normalized = re.sub(r'^то\s+', '', normalized)     # "то " at beginning (then/that)
            normalized = re.sub(r'^у\s+нас\s+', '', normalized)  # "у нас " (we have)
            normalized = re.sub(r'^просто\s+', '', normalized) # "просто " (just)
            normalized = re.sub(r'^давайте\s+', '', normalized) # "давайте " (let's)
            normalized = re.sub(r'^это\s+', '', normalized)    # "это " (this)
            normalized = re.sub(r'^такое\s+', '', normalized)  # "такое " (such)
            normalized = re.sub(r'^такой\s+', '', normalized)  # "такой " (such)
            normalized = re.sub(r'^такая\s+', '', normalized)  # "такая " (such)
            normalized = re.sub(r'^такие\s+', '', normalized)  # "такие " (such)

            # Remove problematic phrases
            normalized = normalized.replace("то обсуждений давайте", "")
            normalized = normalized.replace("то состояние второго определённо такое", "")
            normalized = normalized.replace("некоторого некоторой", "")
            normalized = normalized.replace("состояние едини на2", "")
            normalized = normalized.replace("сейчас скажу", "")
            normalized = normalized.replace("потом обсужу", "")
            normalized = normalized.replace("можно убедиться", "")
            normalized = normalized.replace("второго определённо", "")
        else:
            # English filler phrases to remove
            normalized = re.sub(r'^the\s+', '', normalized)    # "the " at beginning
            normalized = re.sub(r'^a\s+', '', normalized)      # "a " at beginning
            normalized = re.sub(r'^an\s+', '', normalized)     # "an " at beginning
            normalized = re.sub(r'^this\s+', '', normalized)   # "this " at beginning
            normalized = re.sub(r'^that\s+', '', normalized)   # "that " at beginning
            normalized = re.sub(r'^just\s+', '', normalized)   # "just " at beginning
            normalized = re.sub(r'^so\s+', '', normalized)     # "so " at beginning

        return normalized

    def _extract_pattern(self, text: str) -> List[str]:
        """
        Extract a preliminary pattern from concept text.

        Args:
            text: The concept text

        Returns:
            List of terms in the pattern
        """
        # Simple tokenization with improved handling of terms
        words = re.findall(r'\b[\w\-]+\b', text.lower())
        # Filter out very short words
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
        occurrence_count = len(self.occurrences)
        occurrence_score = min(occurrence_count / 20.0, 1.0)

        # Concepts that are referenced by many others are more fundamental
        reference_count = sum(1 for c in all_concepts.values()
                            if self.concept_id in c.related_concepts)
        reference_score = min(reference_count / 5.0, 1.0)

        # Concepts with higher educational weight are likely more fundamental
        educational_score = min(self.educational_weight / 5.0, 0.4)

        # Combine scores with weights
        self.hierarchy_score = (
            pattern_score * 0.2 +
            relationship_score * 0.25 +
            occurrence_score * 0.15 +
            reference_score * 0.25 +
            educational_score * 0.15
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
            "normalized_text": self.normalized_text,
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
            "educational_weight": self.educational_weight,
            "canonical_concept_id": self.canonical_concept_id,
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
        signature.educational_weight = data.get("educational_weight", 0.0)
        signature.created_at = data.get("created_at", signature.created_at)
        signature.updated_at = data.get("updated_at", signature.updated_at)
        signature.canonical_concept_id = data.get("canonical_concept_id")

        # Set normalized text if available or recalculate it
        if "normalized_text" in data:
            signature.normalized_text = data["normalized_text"]
        else:
            signature.normalized_text = signature._normalize_text(data["text"], data.get("language", "en"))

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
        self.language_clusters = defaultdict(set)  # language -> set of concept_ids

    def add_concept(self, concept: ConceptSignature) -> None:
        """
        Add a concept to the graph.

        Args:
            concept: ConceptSignature to add
        """
        self.concepts[concept.concept_id] = concept
        self.domain_clusters[concept.domain].add(concept.concept_id)
        self.language_clusters[concept.language].add(concept.concept_id)

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
                "languages": list(self.language_clusters.keys()),
                "timestamp": datetime.now().isoformat()
            }
        }

    def find_similar_concepts(self, text: str, language: str = None, domain: str = None,
                            max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find concepts similar to the given text.

        Args:
            text: Text to match against
            language: Optional language filter
            domain: Optional domain filter
            max_results: Maximum number of results to return

        Returns:
            List of similar concepts with match scores
        """
        # Filter concepts by language and domain if specified
        candidate_concepts = {}
        for concept_id, concept in self.concepts.items():
            if language and concept.language != language:
                continue
            if domain and concept.domain != domain:
                continue

            # Skip concepts that are variants (have a canonical_concept_id)
            if concept.canonical_concept_id:
                continue

            candidate_concepts[concept_id] = concept

        # Calculate match scores
        matches = []

        # Normalize input text for better matching
        normalized_text = self._normalize_text(text, language) if language else text.lower()

        for concept_id, concept in candidate_concepts.items():
            # Use string similarity for more accurate matching
            similarity = difflib.SequenceMatcher(None,
                                              normalized_text,
                                              concept.normalized_text).ratio()

            # Also check signature pattern matching
            pattern_score = concept.match_text(text)

            # Combine scores, giving more weight to string similarity
            score = (similarity * 0.7) + (pattern_score * 0.3)

            if score > 0.5:  # Set a minimum threshold for matches
                matches.append({
                    "concept_id": concept_id,
                    "text": concept.text,
                    "match_score": score,
                    "similarity": similarity,
                    "pattern_score": pattern_score,
                    "domain": concept.domain,
                    "language": concept.language,
                    "concept_class": concept.concept_class
                })

        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)

        return matches[:max_results]

    def _normalize_text(self, text: str, language: str = "en") -> str:
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

        # Create a temporary concept signature to use its normalization method
        temp = ConceptSignature("temp", text, language=language)
        return temp.normalized_text


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

        # Initialize MLCS algorithm
        self.mlcs_processor = MLCSAlgorithm()

        # Initialize relationship graph
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
            Enhanced processing result with signatures and relationships
        """
        video_id = processing_result.get("video_id")
        if not video_id:
            logger.warning("Missing video_id in processing result")
            return processing_result

        # Extract domain features and transcript segments
        domain_features = processing_result.get("domain_features", {})
        segments = processing_result.get("transcript", {}).get("segments", [])
        language = processing_result.get("transcript", {}).get("language", "en")
        domain = processing_result.get("metadata", {}).get("domain", "unknown")

        # Process theoretical and practical concepts
        theoretical_concepts = domain_features.get("theoretical_concepts", [])
        practical_concepts = domain_features.get("practical_concepts", [])

        all_concepts = theoretical_concepts + practical_concepts

        if not all_concepts or not segments:
            logger.warning(f"No concepts ({len(all_concepts)}) or segments ({len(segments)}) found for video {video_id}")
            return processing_result

        # Set language for processors
        self.mlcs_processor.language = language

        # Ensure concepts have proper IDs
        for concept in all_concepts:
            if "concept_id" not in concept:
                import hashlib
                # Create deterministic ID based on text, domain and language
                text_for_hash = concept.get("text", "").lower().strip()
                concept_hash = hashlib.md5(f"{text_for_hash}:{domain}:{language}".encode()).hexdigest()
                concept["concept_id"] = concept_hash
                logger.debug(f"Generated concept_id {concept_hash} for '{text_for_hash}'")

        # Generate concept signatures
        signatures = self.generate_concept_signatures(all_concepts, segments, domain, language)

        # Add to relationship graph
        for signature in signatures:
            # Make sure concept has a proper ID
            if not signature.concept_id or signature.concept_id == "":
                import hashlib
                text_for_hash = signature.text.lower().strip()
                signature.concept_id = hashlib.md5(f"{text_for_hash}:{domain}:{language}".encode()).hexdigest()
                logger.debug(f"Fixed missing concept_id for '{text_for_hash}'")

            # Add to graph - make sure to store in the database also
            self.relationship_graph.add_concept(signature)

            # If we have data access, ensure concept is saved to database
            if self.data_access:
                concept_data = {
                    "concept_id": signature.concept_id,
                    "text": signature.text,
                    "normalized_text": signature.normalized_text,
                    "domain": signature.domain,
                    "concept_class": signature.concept_class,
                    "language": signature.language,
                    "total_occurrences": len(signature.occurrences),
                    "canonical_concept_id": signature.canonical_concept_id,
                    "video_id": video_id,
                    # Include signature data
                    "signature_pattern": signature.signature_pattern,
                    "hierarchy_score": signature.hierarchy_score,
                    "confidence": signature.confidence,
                    "educational_weight": signature.educational_weight
                }

                self.data_access.save_concept(concept_data)

        # Identify relationships between concepts
        self._identify_concept_relationships(signatures, segments)

        # Update hierarchy scores
        self.relationship_graph.calculate_all_hierarchy_scores()

        # Enhance domain features with signature information - separate by concept class
        enhanced_theoretical_concepts = []
        enhanced_practical_concepts = []

        # Process theoretical concepts
        for concept in theoretical_concepts:
            concept_id = concept.get("concept_id")
            matching_signature = next((s for s in signatures if s.concept_id == concept_id), None)

            if matching_signature:
                enhanced_concept = concept.copy()
                enhanced_concept["signature_pattern"] = matching_signature.signature_pattern
                enhanced_concept["hierarchy_score"] = matching_signature.hierarchy_score
                enhanced_concept["confidence"] = matching_signature.confidence
                enhanced_concept["educational_weight"] = matching_signature.educational_weight
                enhanced_concept["canonical_concept_id"] = matching_signature.canonical_concept_id
                enhanced_concept["related_concepts"] = [
                    {"id": rel_id, "strength": rel_data["strength"], "type": rel_data["type"]}
                    for rel_id, rel_data in matching_signature.related_concepts.items()
                ]
                enhanced_theoretical_concepts.append(enhanced_concept)
            else:
                enhanced_theoretical_concepts.append(concept)

        # Process practical concepts
        for concept in practical_concepts:
            concept_id = concept.get("concept_id")
            matching_signature = next((s for s in signatures if s.concept_id == concept_id), None)

            if matching_signature:
                enhanced_concept = concept.copy()
                enhanced_concept["signature_pattern"] = matching_signature.signature_pattern
                enhanced_concept["hierarchy_score"] = matching_signature.hierarchy_score
                enhanced_concept["confidence"] = matching_signature.confidence
                enhanced_concept["educational_weight"] = matching_signature.educational_weight
                enhanced_concept["canonical_concept_id"] = matching_signature.canonical_concept_id
                enhanced_concept["related_concepts"] = [
                    {"id": rel_id, "strength": rel_data["strength"], "type": rel_data["type"]}
                    for rel_id, rel_data in matching_signature.related_concepts.items()
                ]
                enhanced_practical_concepts.append(enhanced_concept)
            else:
                enhanced_practical_concepts.append(concept)

        # Update domain features
        domain_features["theoretical_concepts"] = enhanced_theoretical_concepts
        domain_features["practical_concepts"] = enhanced_practical_concepts
        domain_features["concept_signatures"] = [signature.to_dict() for signature in signatures]

        # Save relationship graph
        self._save_relationship_graph()

        # Return enhanced result
        processing_result["domain_features"] = domain_features

        # Make sure concept IDs get saved to database
        if self.data_access:
            for concept in enhanced_theoretical_concepts + enhanced_practical_concepts:
                if "concept_id" in concept and concept.get("text"):
                    # Set basic concept data
                    concept_data = {
                        "concept_id": concept["concept_id"],
                        "text": concept["text"],
                        "domain": domain,
                        "concept_class": concept.get("concept_class", "theoretical"),
                        "language": language,
                        "video_id": video_id
                    }
                    self.data_access.save_concept(concept_data)

        return processing_result

    def generate_concept_signatures(
        self,
        concepts: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        domain: str = "physics",
        language: str = "en"
    ) -> List[ConceptSignature]:
        """
        Generate concept signatures using MLCS from concept occurrences.

        Args:
            concepts: List of concept dictionaries
            segments: List of transcript segments
            domain: Content domain
            language: Language code

        Returns:
            List of ConceptSignature instances
        """
        signatures = []

        for concept in concepts:
            # Basic validation - skip invalid concepts
            concept_text = concept.get("text", "")
            if not concept_text or len(concept_text) < 3:
                continue

            # Create basic signature
            signature = ConceptSignature(
                concept_id=concept.get("concept_id", str(uuid.uuid4())),
                text=concept_text,
                domain=domain,
                concept_class=concept.get("concept_class", "theoretical"),
                language=language
            )

            # Find concept occurrences in segments
            concept_text_lower = concept_text.lower()

            # Collect context texts for this concept
            contexts = []

            for segment in segments:
                segment_text = segment.get("text", "").lower()
                if segment_text and concept_text_lower in segment_text:
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

            # Extract signature pattern from contexts
            if contexts:
                # Use the MLCSAlgorithm to extract the signature pattern
                signature_pattern, confidence = self.mlcs_processor.extract_concept_signature(
                    concept_text, contexts, language
                )

                if signature_pattern:
                    signature.signature_pattern = signature_pattern
                    signature.confidence = confidence

                # Calculate educational significance
                self._calculate_educational_significance(signature, concept, contexts)

            # Check for similar existing concepts before adding
            similar_concepts = []
            if self.data_access:
                # Use data access to find similar concepts
                similar_concepts = self.data_access.find_similar_concepts(
                    concept_text,
                    domain,
                    language
                )

            # If we found a very similar concept, set it as canonical
            if similar_concepts:
                best_match = similar_concepts[0]
                match_score = best_match.get('match_score', 0)
                similarity = best_match.get('similarity', 0)

                if match_score >= 3 or similarity > 0.85:
                    # This is essentially the same concept, use the existing one as canonical
                    canonical_id = best_match.get('concept_id')
                    signature.canonical_concept_id = canonical_id
                    logger.info(f"Using canonical concept {canonical_id} for similar concept: '{concept_text}'")

            signatures.append(signature)

        return signatures

    def _calculate_educational_significance(
        self,
        signature: ConceptSignature,
        concept: Dict[str, Any],
        contexts: List[str]
    ) -> None:
        """
        Calculate educational significance of a concept (vs. passing mention)

        Args:
            signature: Concept signature object to update
            concept: Original concept dictionary
            contexts: List of context texts where concept appears
        """
        # Base educational weight
        educational_weight = 0.0

        # Factor 1: Frequency of occurrences
        occurrences_count = len(signature.occurrences)
        frequency_factor = min(occurrences_count, 5) * 0.5
        educational_weight += frequency_factor

        # Factor 2: Context diversity - appears in multiple segments
        segments = set(occ.get("segment_id") for occ in signature.occurrences if occ.get("segment_id"))
        segment_factor = min(len(segments), 3) * 0.7
        educational_weight += segment_factor

        # Factor 3: Duration of coverage
        total_duration = sum(
            (occ.get("end_time", 0) - occ.get("start_time", 0))
            for occ in signature.occurrences
        )
        duration_factor = min(total_duration / 10.0, 3.0)
        educational_weight += duration_factor

        # Factor 4: Context analysis - check for educational markers
        edu_markers = {
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
                r'important'
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
                r'важный'
            ]
        }

        lang = signature.language if signature.language in edu_markers else 'en'
        markers = edu_markers[lang]

        marker_count = 0
        for context in contexts:
            context_lower = context.lower()
            for marker in markers:
                if re.search(r'\b' + marker + r'\b', context_lower):
                    marker_count += 1
                    break  # Count at most one marker per context

        marker_factor = min(marker_count, 3) * 0.8
        educational_weight += marker_factor

        # Set educational weight on signature
        signature.educational_weight = educational_weight

        # Also indicate is_educational if the weight is high enough
        if educational_weight > 2.5:
            # Add as a property on the concept object if it comes from concept extraction
            if "educational_weight" in concept:
                concept["educational_weight"] = educational_weight
                concept["is_educational"] = True

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

        # Analyze proximity, ordering, and semantic relationships
        for i, concept1 in enumerate(signatures):
            for j, concept2 in enumerate(signatures):
                if i == j:
                    continue

                # Skip if no co-occurrences
                if co_occurrences[(concept1.concept_id, concept2.concept_id)] == 0:
                    continue

                # Calculate relationship strength based on co-occurrences
                co_occurrence_count = co_occurrences[(concept1.concept_id, concept2.concept_id)]
                strength = min(co_occurrence_count / 5.0, 1.0)

                # Skip weak relationships
                if strength < 0.2:
                    continue

                # Check if one concept might be a prerequisite of the other
                concept1_segments = list(concept_segment_map[concept1.concept_id])
                concept2_segments = list(concept_segment_map[concept2.concept_id])

                # Get earliest segment for each concept
                concept1_earliest = min(segment_order[sid] for sid in concept1_segments if sid in segment_order)
                concept2_earliest = min(segment_order[sid] for sid in concept2_segments if sid in segment_order)

                # Determine relationship type
                relationship_type = "related"

                # If concept1 consistently appears before concept2, it might be a prerequisite
                if concept1_earliest < concept2_earliest - 3:  # At least 3 segments earlier
                    # Check if concept1 is mentioned in context of concept2
                    text_relationship = False
                    for occurrence in concept2.occurrences:
                        if concept1.text.lower() in occurrence.get("context_text", "").lower():
                            text_relationship = True
                            break

                    if text_relationship:
                        relationship_type = "prerequisite"

                # Check for "is a" relationships
                if concept2.text.lower() in concept1.text.lower() and len(concept1.text) > len(concept2.text):
                    # concept2 might be a more general version of concept1
                    relationship_type = "is_a"
                    strength = max(strength, 0.7)  # Boost strength for "is a" relationships

                # Add relationship to graph
                self.relationship_graph.add_relationship(
                    concept1.concept_id, concept2.concept_id, relationship_type, strength
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
            theory_practice_ratio: Desired ratio of theoretical to practical content
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
                    ],
                    "educational_weight": concept.educational_weight,
                    "canonical_concept_id": concept.canonical_concept_id
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

    def enhanced_path_generator(concept_ids, theory_practice_ratio=0.5, domain=None):
        """Enhanced learning path generator function."""
        # First, generate the basic learning path
        base_path = original_generate_learning_path(concept_ids, theory_practice_ratio, domain)

        # Then enhance it with concept signatures and relationships
        enhanced_path = generator.generate_enhanced_learning_path(
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
                        "related_concepts": concept.get("related_concepts", []),
                        "educational_weight": concept.get("educational_weight", 0.0),
                        "canonical_concept_id": concept.get("canonical_concept_id")
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

    logger.info("Enhanced search engine with improved learning path generation")

    return search_engine
