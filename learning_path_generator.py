"""
Learning Path Generator for the Video Lecture Content Indexer.

Generates optimized learning paths through concept prerequisites using graph algorithms.
Handles concepts with multiple prerequisites and resolves cycles in relationships.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any, Deque
from collections import defaultdict, deque
import time

from concept_repository import get_concept_repository

# Configure logging
logger = logging.getLogger(__name__)

class LearningPathGenerator:
    """
    Generates optimized learning paths through concept relationships.

    This class builds a directed graph from concept prerequisites and creates
    optimized paths that ensure prerequisites are learned before dependent concepts.
    """

    def __init__(self):
        """Initialize the learning path generator."""
        self.repository = get_concept_repository()
        logger.info("LearningPathGenerator initialized")

    def generate_path(
        self,
        target_concept_ids: List[str],
        max_concepts: int = 15,
        theory_practice_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate a learning path for the specified target concepts.

        Args:
            target_concept_ids: List of target concept IDs
            max_concepts: Maximum number of concepts to include
            theory_practice_ratio: Ratio of theoretical to practical content (unused in current version)

        Returns:
            Dictionary with learning path information
        """
        start_time = time.time()

        # Filter to valid concept IDs
        valid_target_ids = [
            cid for cid in target_concept_ids
            if self.repository.get_concept(cid) is not None
        ]

        if not valid_target_ids:
            logger.warning("No valid target concepts provided")
            return {
                "status": "error",
                "message": "No valid target concepts provided",
                "path": []
            }

        # Find all required prerequisites
        start_time_prereqs = time.time()
        all_prerequisites = self._find_all_prerequisites(valid_target_ids)
        prereq_time = time.time() - start_time_prereqs

        # Build the concept graph
        start_time_graph = time.time()
        graph = self._build_concept_graph(all_prerequisites + valid_target_ids)
        graph_time = time.time() - start_time_graph

        # Generate the path with topological sort
        start_time_topo = time.time()
        path = self._topological_sort(all_prerequisites + valid_target_ids, graph)
        topo_time = time.time() - start_time_topo

        # If we have too many concepts, prioritize and limit
        if len(path) > max_concepts:
            path = self._prioritize_concepts(path, valid_target_ids, max_concepts)

        # Create a detailed path with concept information
        detailed_path = self._create_detailed_path(path, valid_target_ids)

        # Generate the final result
        total_time = time.time() - start_time
        result = {
            "status": "success",
            "target_concepts": valid_target_ids,
            "path_length": len(detailed_path),
            "path": detailed_path,
            "metadata": {
                "generation_time_ms": round(total_time * 1000),
                "prerequisite_finding_time_ms": round(prereq_time * 1000),
                "graph_building_time_ms": round(graph_time * 1000),
                "topological_sort_time_ms": round(topo_time * 1000)
            }
        }

        logger.info(f"Generated learning path with {len(detailed_path)} concepts in {total_time:.2f}s")
        return result

    def _find_all_prerequisites(self, concept_ids: List[str]) -> List[str]:
        """
        Find all prerequisites for the given concepts recursively.

        Args:
            concept_ids: List of concept IDs

        Returns:
            List of prerequisite concept IDs
        """
        # Track visited concepts to avoid cycles
        visited = set()
        prerequisites = set()

        def visit(cid):
            """Recursively visit prerequisites."""
            if cid in visited:
                return

            visited.add(cid)

            concept = self.repository.get_concept(cid)
            if not concept:
                return

            # Process prerequisites
            for prereq_id in concept.get('prerequisites', []):
                prerequisites.add(prereq_id)
                visit(prereq_id)

        # Visit each target concept
        for concept_id in concept_ids:
            visit(concept_id)

        # Return list of prerequisites (excluding the targets themselves)
        return list(prerequisites - set(concept_ids))

    def _build_concept_graph(self, concept_ids: List[str]) -> Dict[str, List[str]]:
        """
        Build a directed graph from concepts and their prerequisites.

        Args:
            concept_ids: List of concept IDs to include

        Returns:
            Dictionary mapping concept ID to list of prerequisite concept IDs
        """
        graph = defaultdict(list)

        # Create a set for faster lookups
        concept_set = set(concept_ids)

        # Build the graph
        for concept_id in concept_ids:
            concept = self.repository.get_concept(concept_id)
            if not concept:
                continue

            # Add prerequisites as edges (child -> prerequisite)
            # The direction is from dependent to prerequisite
            for prereq_id in concept.get('prerequisites', []):
                # Only include prerequisites that are in our concept set
                if prereq_id in concept_set:
                    graph[concept_id].append(prereq_id)

        return graph

    def _topological_sort(self, concept_ids: List[str], graph: Dict[str, List[str]]) -> List[str]:
        """
        Perform a topological sort to order concepts by their dependencies.

        Args:
            concept_ids: List of concept IDs to sort
            graph: Dependency graph (concept_id -> list of prerequisite IDs)

        Returns:
            Ordered list of concept IDs
        """
        # Create a set of all concepts
        all_concepts = set(concept_ids)

        # Calculate in-degree for each concept
        # In-degree = number of concepts that depend on this concept
        in_degree = {concept_id: 0 for concept_id in all_concepts}
        for concept_id, prereqs in graph.items():
            for prereq_id in prereqs:
                if prereq_id in all_concepts:
                    in_degree[prereq_id] = in_degree.get(prereq_id, 0) + 1

        # Start with concepts that have no dependencies
        # These are concepts that are not prerequisites for any other concept
        queue = deque([cid for cid in all_concepts if in_degree.get(cid, 0) == 0])

        # Process the queue
        result = []
        while queue:
            concept_id = queue.popleft()
            result.append(concept_id)

            # For each prerequisite of this concept, reduce its in-degree
            for prereq_id in graph.get(concept_id, []):
                if prereq_id in all_concepts:
                    in_degree[prereq_id] -= 1

                    # If in-degree reaches 0, add to queue
                    if in_degree[prereq_id] == 0:
                        queue.append(prereq_id)

        # If not all concepts were included, we have a cycle
        if len(result) < len(all_concepts):
            remaining = all_concepts - set(result)
            logger.warning(f"Detected cycle in concept graph. Remaining concepts: {remaining}")

            # Handle remaining concepts by sorting them by some heuristic
            # Here we sort by their ID for deterministic output
            remaining_sorted = sorted(list(remaining))
            result.extend(remaining_sorted)

        # Reverse the result since we want prerequisites first, then dependents
        return result[::-1]

    def _prioritize_concepts(
        self,
        path: List[str],
        target_concept_ids: List[str],
        max_concepts: int
    ) -> List[str]:
        """
        Prioritize concepts to fit within maximum limit.

        Args:
            path: Full learning path
            target_concept_ids: Target concept IDs
            max_concepts: Maximum number of concepts

        Returns:
            Prioritized path limited to max_concepts
        """
        # Always include target concepts
        must_include = set(target_concept_ids)

        # If we can include all targets plus some prerequisites, do that
        if len(must_include) <= max_concepts:
            # Find out how many prerequisites we can include
            num_prereqs = max_concepts - len(must_include)

            # Get all concepts in path that are not targets
            prereqs = [cid for cid in path if cid not in must_include]

            # Take the first num_prereqs prerequisites
            selected_prereqs = prereqs[:num_prereqs]

            # Combine targets and selected prerequisites
            prioritized = selected_prereqs + list(must_include)

            # Re-sort to maintain correct order
            return [cid for cid in path if cid in set(prioritized)]
        else:
            # If we have more targets than max_concepts, just include what fits
            logger.warning(f"Too many target concepts ({len(must_include)}) for max_concepts ({max_concepts})")

            # Take the first max_concepts targets in the path
            targets_in_path = [cid for cid in path if cid in must_include]
            return targets_in_path[:max_concepts]

    def _create_detailed_path(self, path: List[str], target_concept_ids: List[str]) -> List[Dict]:
        """
        Create a detailed path with concept information.

        Args:
            path: Ordered list of concept IDs
            target_concept_ids: Target concept IDs

        Returns:
            List of dictionaries with detailed concept information
        """
        detailed_path = []

        for concept_id in path:
            concept = self.repository.get_concept(concept_id)
            if not concept:
                continue

            # Get concept representations
            representations = concept.get('representations', {})

            # Get prereqs and related concepts
            prereqs = concept.get('prerequisites', [])
            related = concept.get('related', [])

            # Create detailed entry
            detailed_entry = {
                'concept_id': concept_id,
                'representations': representations,
                'is_target': concept_id in target_concept_ids,
                'prerequisites': prereqs,
                'related': related,
                'metadata': concept.get('metadata', {})
            }

            detailed_path.append(detailed_entry)

        return detailed_path

    def find_concept_path(self, source_id: str, target_id: str) -> List[str]:
        """
        Find a path from source concept to target concept.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID

        Returns:
            List of concept IDs forming a path, or empty list if no path exists
        """
        # Verify both concepts exist
        source = self.repository.get_concept(source_id)
        target = self.repository.get_concept(target_id)

        if not source or not target:
            return []

        # Use breadth-first search to find a path
        visited = set()
        queue = deque([(source_id, [source_id])])

        while queue:
            current_id, path = queue.popleft()

            # If we've reached the target, return the path
            if current_id == target_id:
                return path

            # Avoid cycles
            if current_id in visited:
                continue

            visited.add(current_id)

            # Explore related concepts and prerequisites
            current = self.repository.get_concept(current_id)
            if not current:
                continue

            # Check related concepts
            for related_id in current.get('related', []):
                if related_id not in visited:
                    queue.append((related_id, path + [related_id]))

            # Check prerequisites
            for prereq_id in current.get('prerequisites', []):
                if prereq_id not in visited:
                    queue.append((prereq_id, path + [prereq_id]))

        # No path found
        return []

    def get_recommended_videos(self, concept_id: str, limit: int = 3) -> List[Dict]:
        """
        Get recommended videos for a concept.

        Args:
            concept_id: Concept ID
            limit: Maximum number of videos to return

        Returns:
            List of recommended video dictionaries
        """
        # This would need to be implemented based on additional data sources
        # and video information - placeholder for now
        return []


# Singleton instance for global access
_instance = None

def get_learning_path_generator() -> LearningPathGenerator:
    """
    Get or create the LearningPathGenerator singleton instance.

    Returns:
        LearningPathGenerator instance
    """
    global _instance

    if _instance is None:
        _instance = LearningPathGenerator()

    return _instance
