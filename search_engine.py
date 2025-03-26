"""
Simplified search engine for the Lecture Video Content Indexer.
Provides basic search functionality using SQLite FTS5.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple

# Import simplified modules - Fix the import for cached decorator
from data_access import get_data_access
from cache_manager import cache_get, cache_set, cached
from performance_utils import time_function

# Configure logging
logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Search engine for educational video content.
    Simplified version with reduced complexity and straightforward SQLite FTS5 integration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the search engine with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.index_dir = config.get("index_dir", "data/index")

        # Create index directory if needed
        os.makedirs(self.index_dir, exist_ok=True)

        # Get data access layer
        db_path = os.path.join(self.index_dir, "indexer.db")
        self.data_access = get_data_access(db_path)

        logger.info("SearchEngine initialized")

    @time_function(2000)  # Log warning if takes more than 2 seconds
    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for content matching the query.

        Args:
            query: Structured query dictionary

        Returns:
            Search results dictionary
        """
        # Use the data access layer for search
        return self.data_access.search(query)

    @time_function(5000)  # Log warning if takes more than 5 seconds
    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content for search.

        Args:
            processed_result: Processing result dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract key information for logging
            video_id = processed_result.get("video_id")
            if not video_id:
                logger.error("Missing video_id in processed result")
                return False

            # Index content using data access layer
            success = self.data_access.index_content(processed_result)

            if success:
                logger.info(f"Successfully indexed content for video {video_id}")
            else:
                logger.error(f"Failed to index content for video {video_id}")

            return success

        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return False

    @cached("concept")
    def get_concept_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept.

        Args:
            concept_id: Concept ID

        Returns:
            Concept details dictionary or None if not found
        """
        try:
            # Get concept from data access layer
            concept = self.data_access.get_concept(concept_id)
            if not concept:
                return None

            # Get occurrences
            occurrences_query = """
            SELECT o.*, v.title as video_title
            FROM occurrences o
            JOIN videos v ON o.video_id = v.video_id
            WHERE o.concept_id = ?
            ORDER BY o.video_id, o.start_time
            """
            occurrences = self.data_access.execute_query(occurrences_query, (concept_id,))

            # Group occurrences by video
            videos = {}
            for occurrence in occurrences:
                video_id = occurrence["video_id"]
                if video_id not in videos:
                    videos[video_id] = {
                        "video_id": video_id,
                        "title": occurrence["video_title"],
                        "occurrences": []
                    }
                videos[video_id]["occurrences"].append({
                    "occurrence_id": occurrence["occurrence_id"],
                    "segment_id": occurrence["segment_id"],
                    "start_time": occurrence["start_time"],
                    "end_time": occurrence["end_time"],
                    "context_type": occurrence["context_type"],
                    "context_text": occurrence["context_text"]
                })

            # Combine into result
            result = {
                "concept_id": concept_id,
                "text": concept["text"],
                "domain": concept["domain"],
                "concept_class": concept["concept_class"],
                "total_occurrences": concept["total_occurrences"],
                "videos": list(videos.values())
            }

            return result

        except Exception as e:
            logger.error(f"Error getting concept details for {concept_id}: {e}")
            return None

    @cached("video")
    def get_video_concepts(self, video_id: str, context_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get concepts extracted from a video.

        Args:
            video_id: YouTube video ID
            context_type: Optional context type filter

        Returns:
            Dictionary with video concepts or None if not found
        """
        try:
            # Use data access layer to get video concepts
            return self.data_access.get_video_concepts(video_id)
        except Exception as e:
            logger.error(f"Error getting video concepts for {video_id}: {e}")
            return None

    def generate_learning_path(
        self,
        concept_ids: List[str],
        theory_practice_ratio: float = 0.5,
        domain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a learning path for a set of concepts.
        This is a simplified implementation that just orders concepts by theoretical/practical.

        Args:
            concept_ids: List of concept IDs
            theory_practice_ratio: Desired ratio of theoretical to practical content
            domain: Optional domain filter

        Returns:
            Learning path dictionary or None if generation fails
        """
        try:
            if not concept_ids:
                return None

            # Get concept details for all concepts
            concepts = []
            for concept_id in concept_ids:
                concept = self.get_concept_details(concept_id)
                if concept:
                    concepts.append(concept)

            if not concepts:
                return None

            # Filter by domain if specified
            if domain:
                concepts = [c for c in concepts if c["domain"] == domain]

            # Sort concepts by theoretical/practical based on ratio
            if theory_practice_ratio > 0.7:
                # Theory-heavy: theoretical concepts first
                concepts.sort(key=lambda c: 0 if c["concept_class"] == "theoretical" else 1)
            elif theory_practice_ratio < 0.3:
                # Practice-heavy: practical concepts first
                concepts.sort(key=lambda c: 0 if c["concept_class"] == "practical" else 1)
            else:
                # Balanced: alternate theoretical and practical
                theoretical = [c for c in concepts if c["concept_class"] == "theoretical"]
                practical = [c for c in concepts if c["concept_class"] == "practical"]

                # Create balanced list
                balanced = []
                i_theo = 0
                i_prac = 0

                while i_theo < len(theoretical) or i_prac < len(practical):
                    # Add theoretical if available
                    if i_theo < len(theoretical):
                        balanced.append(theoretical[i_theo])
                        i_theo += 1

                    # Add practical if available
                    if i_prac < len(practical):
                        balanced.append(practical[i_prac])
                        i_prac += 1

                concepts = balanced

            # Create learning path result
            result = {
                "concepts": concepts,
                "theory_practice_ratio": theory_practice_ratio,
                "domain": domain,
                "theoretical_concepts": sum(1 for c in concepts if c["concept_class"] == "theoretical"),
                "practical_concepts": sum(1 for c in concepts if c["concept_class"] == "practical")
            }

            return result

        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return None

    def optimize_database(self) -> bool:
        """
        Optimize the search database.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Run VACUUM on SQLite database
            self.data_access.execute_update("VACUUM")

            # Run ANALYZE on tables
            self.data_access.execute_update("ANALYZE")

            logger.info("Database optimized successfully")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
