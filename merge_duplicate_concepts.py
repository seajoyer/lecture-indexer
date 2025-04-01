#!/usr/bin/env python3
"""
Script to find and merge duplicate concepts in the database.

This script scans the database for similar concepts and establishes canonical relationships
between duplicated concepts. It helps clean up the knowledge base by identifying
and linking similar concepts together.
"""

import os
import logging
import sys
import argparse
import difflib
import time
import re
import sqlite3
from typing import Dict, List, Set, Any, Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedDataAccess:
    """
    Simplified data access class specifically for the concept merging task.
    This avoids dependency on the full data_access module and potential schema conflicts.
    """

    def __init__(self, db_path: str = "data/index/indexer.db"):
        """
        Initialize with database path.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

        # Create directory if needed
        if db_path:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._ensure_schema()

    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        """Make sure the necessary columns exist in the database."""
        try:
            with self._get_connection() as conn:
                # Check if concepts table exists
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'")
                table_exists = cursor.fetchone() is not None

                if not table_exists:
                    logger.warning("Concepts table doesn't exist. Database may not be initialized.")
                    return

                # Check for normalized_text column
                cursor.execute("PRAGMA table_info(concepts)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]

                # Add normalized_text column if it doesn't exist
                if "normalized_text" not in column_names:
                    logger.info("Adding normalized_text column to concepts table")
                    conn.execute("ALTER TABLE concepts ADD COLUMN normalized_text TEXT")

                # Add canonical_concept_id column if it doesn't exist
                if "canonical_concept_id" not in column_names:
                    logger.info("Adding canonical_concept_id column to concepts table")
                    conn.execute("ALTER TABLE concepts ADD COLUMN canonical_concept_id TEXT")

                # Commit changes
                conn.commit()

                logger.info("Database schema updated successfully")
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise RuntimeError(f"Failed to update database schema: {e}")

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """
        Execute a query and return results as dictionaries.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                return []
        except sqlite3.Error as e:
            logger.error(f"Query error: {e}, Query: {query}")
            return []

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an update query.

        Args:
            query: SQL update query
            params: Query parameters

        Returns:
            Number of affected rows
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Update error: {e}, Query: {query}")
            return 0

class ConceptMerger:
    """Finds and merges similar concepts in the database."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the concept merger.

        Args:
            db_path: Optional path to the database
        """
        # Use default path if none provided
        if db_path is None:
            db_path = "data/index/indexer.db"
            logger.info(f"No database path provided, using default: {db_path}")

        # Initialize the simplified data access
        self.data_access = SimplifiedDataAccess(db_path)

        self.similarity_threshold = 0.85  # Similarity threshold for considering concepts duplicates
        self.debug_mode = False
        self.dry_run = False
        self.stats = {
            "total_concepts": 0,
            "duplicate_sets": 0,
            "merged_concepts": 0,
            "canonical_concepts": 0
        }

    def _normalize_concept_text(self, text: str, language: str = "en") -> str:
        """
        Normalize concept text for better matching and deduplication.

        Args:
            text: Concept text
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

    def find_similar_concepts(self, batch_size: int = 1000) -> List[List[Dict[str, Any]]]:
        """
        Find sets of similar concepts in the database.
        Uses efficient batching to handle large databases.

        Args:
            batch_size: Size of concept batches to process

        Returns:
            List of sets of similar concepts
        """
        logger.info("Finding similar concepts...")

        # First, let's get all concepts with their normalized text
        query = """
        SELECT concept_id, text, domain, language, concept_class, normalized_text,
               canonical_concept_id, total_occurrences
        FROM concepts
        WHERE canonical_concept_id IS NULL OR canonical_concept_id = ''
        """

        # Try to include total_occurrences, but handle if it doesn't exist
        try:
            all_concepts = self.data_access.execute_query(query)
        except:
            # If the query fails, try without total_occurrences
            query = """
            SELECT concept_id, text, domain, language, concept_class, normalized_text,
                   canonical_concept_id
            FROM concepts
            WHERE canonical_concept_id IS NULL OR canonical_concept_id = ''
            """
            all_concepts = self.data_access.execute_query(query)

            # Add a default total_occurrences
            for concept in all_concepts:
                concept["total_occurrences"] = 0

        # This will hold our grouped similar concepts
        similar_groups = []

        total_concepts = len(all_concepts)
        self.stats["total_concepts"] = total_concepts

        logger.info(f"Found {total_concepts} concepts to analyze")

        # Process in batches to avoid memory issues
        for i in range(0, total_concepts, batch_size):
            batch = all_concepts[i:min(i+batch_size, total_concepts)]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_concepts + batch_size - 1)//batch_size}")

            # Create normalized texts for concepts that don't have them
            for concept in batch:
                if not concept.get("normalized_text"):
                    concept["normalized_text"] = self._normalize_concept_text(
                        concept["text"],
                        concept.get("language", "en")
                    )

            # Compare each concept with all others in the batch
            for j, concept1 in enumerate(batch):
                # Skip concepts that are already in a group
                if any(concept1 in group for group in similar_groups):
                    continue

                similar = [concept1]

                for concept2 in batch[j+1:]:
                    # Skip concepts that are already in a group
                    if any(concept2 in group for group in similar_groups):
                        continue

                    # Skip if different languages or domains
                    if concept1.get("language") != concept2.get("language") or concept1.get("domain") != concept2.get("domain"):
                        continue

                    # Calculate string similarity between normalized texts
                    similarity = difflib.SequenceMatcher(
                        None,
                        concept1["normalized_text"],
                        concept2["normalized_text"]
                    ).ratio()

                    # Additional specific check for very similar concepts
                    if similarity >= self.similarity_threshold:
                        similar.append(concept2)
                        if self.debug_mode:
                            logger.info(f"Found similar concepts: '{concept1['text']}' and '{concept2['text']}' (similarity: {similarity:.3f})")

                # If we found similar concepts, add them as a group
                if len(similar) > 1:
                    # Sort by total_occurrences (descending) to find best canonical candidate
                    similar.sort(key=lambda c: c.get("total_occurrences", 0), reverse=True)
                    similar_groups.append(similar)

        # Cross-check across batches (simplified approach)
        # This isn't a complete solution but helps catch some cross-batch duplicates
        if len(similar_groups) > 1:
            i = 0
            while i < len(similar_groups):
                j = i + 1
                while j < len(similar_groups):
                    # Check if any concept in group i is similar to any concept in group j
                    group_i = similar_groups[i]
                    group_j = similar_groups[j]

                    # Skip groups with different languages or domains
                    if group_i[0].get("language") != group_j[0].get("language") or group_i[0].get("domain") != group_j[0].get("domain"):
                        j += 1
                        continue

                    # Check for similarity between the first elements of each group
                    similarity = difflib.SequenceMatcher(
                        None,
                        group_i[0]["normalized_text"],
                        group_j[0]["normalized_text"]
                    ).ratio()

                    if similarity >= self.similarity_threshold:
                        # Merge groups
                        combined = group_i + group_j
                        # Sort by occurrences
                        combined.sort(key=lambda c: c.get("total_occurrences", 0), reverse=True)
                        similar_groups[i] = combined
                        similar_groups.pop(j)
                    else:
                        j += 1
                i += 1

        # Update stats
        self.stats["duplicate_sets"] = len(similar_groups)

        return similar_groups

    def merge_similar_concepts(self, similar_groups: List[List[Dict[str, Any]]]) -> int:
        """
        Merge sets of similar concepts by establishing canonical relationships.

        Args:
            similar_groups: List of groups of similar concepts

        Returns:
            Number of updated concepts
        """
        if not similar_groups:
            logger.info("No similar concept groups found")
            return 0

        logger.info(f"Merging {len(similar_groups)} similar concept groups...")

        updated_count = 0

        for group in similar_groups:
            if not group or len(group) < 2:
                continue

            # The first concept in each group becomes the canonical one
            canonical = group[0]
            variants = group[1:]

            if self.debug_mode:
                logger.info(f"Canonical concept: {canonical['text']} ({canonical['concept_id']})")
                logger.info(f"Variants ({len(variants)}): {', '.join(c['text'] for c in variants)}")

            # Skip if in dry run mode
            if self.dry_run:
                updated_count += len(variants)
                continue

            # Update each variant to point to the canonical concept
            for variant in variants:
                try:
                    # Update the canonical_concept_id field
                    self.data_access.execute_update(
                        "UPDATE concepts SET canonical_concept_id = ? WHERE concept_id = ?",
                        (canonical["concept_id"], variant["concept_id"])
                    )

                    # Redirect occurrences from variant to canonical
                    # This step is optional and may be skipped if you want to keep occurrences separate
                    # Uncomment to enable occurrence merging
                    """
                    self.data_access.execute_update(
                        "UPDATE occurrences SET concept_id = ? WHERE concept_id = ?",
                        (canonical["concept_id"], variant["concept_id"])
                    )
                    """

                    updated_count += 1

                except Exception as e:
                    logger.error(f"Error updating concept {variant['concept_id']}: {e}")

        # Update stats
        self.stats["merged_concepts"] = updated_count
        self.stats["canonical_concepts"] = len(similar_groups)

        return updated_count

    def run(self, batch_size: int = 1000, debug: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run the complete concept merging process.

        Args:
            batch_size: Size of concept batches to process
            debug: Enable debug logging
            dry_run: Run without making changes

        Returns:
            Statistics dictionary
        """
        self.debug_mode = debug
        self.dry_run = dry_run

        # Log dry run mode
        if self.dry_run:
            logger.info("Running in DRY RUN mode - no changes will be made")

        start_time = time.time()

        # Find similar concepts
        similar_groups = self.find_similar_concepts(batch_size)

        # Merge them
        updated_count = self.merge_similar_concepts(similar_groups)

        # Optimize the database unless in dry run mode
        if not self.dry_run:
            try:
                logger.info("Optimizing database...")
                self.data_access.execute_update("PRAGMA optimize")
            except Exception as e:
                logger.warning(f"Could not optimize database: {e}")

        # Calculate stats
        execution_time = time.time() - start_time
        self.stats["execution_time"] = f"{execution_time:.2f} seconds"
        self.stats["mode"] = "dry_run" if self.dry_run else "live"

        # Log results
        logger.info(f"Concept merging completed in {execution_time:.2f} seconds")
        logger.info(f"Total concepts: {self.stats['total_concepts']}")
        logger.info(f"Found {self.stats['duplicate_sets']} sets of similar concepts")
        logger.info(f"Merged {self.stats['merged_concepts']} concept variants into {self.stats['canonical_concepts']} canonical concepts")

        return self.stats

    def update_normalized_text(self) -> int:
        """
        Update normalized_text for all concepts in the database.
        This is useful for initial setup or after changing normalization rules.

        Returns:
            Number of updated concepts
        """
        logger.info("Updating normalized_text for all concepts...")

        # Get all concepts
        query = "SELECT concept_id, text, language FROM concepts"
        concepts = self.data_access.execute_query(query)

        if not concepts:
            logger.info("No concepts found")
            return 0

        updated_count = 0

        for concept in concepts:
            concept_id = concept.get("concept_id")
            text = concept.get("text", "")
            language = concept.get("language", "en")

            # Normalize the text
            normalized_text = self._normalize_concept_text(text, language)

            # Update the concept
            try:
                self.data_access.execute_update(
                    "UPDATE concepts SET normalized_text = ? WHERE concept_id = ?",
                    (normalized_text, concept_id)
                )
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating concept {concept_id}: {e}")

        logger.info(f"Updated normalized_text for {updated_count} concepts")
        return updated_count

    def export_similarity_graph(self, output_path: str) -> bool:
        """
        Export a similarity graph for visualization.

        Args:
            output_path: Path to save the graph data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Find similar groups
            similar_groups = self.find_similar_concepts()

            # Create graph data
            nodes = []
            edges = []
            node_ids = set()

            for group in similar_groups:
                if not group or len(group) < 2:
                    continue

                canonical = group[0]
                variants = group[1:]

                # Add canonical node
                if canonical["concept_id"] not in node_ids:
                    nodes.append({
                        "id": canonical["concept_id"],
                        "label": canonical["text"],
                        "type": "canonical",
                        "domain": canonical.get("domain", "unknown"),
                        "language": canonical.get("language", "en"),
                        "occurrences": canonical.get("total_occurrences", 0)
                    })
                    node_ids.add(canonical["concept_id"])

                # Add variant nodes and edges to canonical
                for variant in variants:
                    if variant["concept_id"] not in node_ids:
                        nodes.append({
                            "id": variant["concept_id"],
                            "label": variant["text"],
                            "type": "variant",
                            "domain": variant.get("domain", "unknown"),
                            "language": variant.get("language", "en"),
                            "occurrences": variant.get("total_occurrences", 0)
                        })
                        node_ids.add(variant["concept_id"])

                    # Add edge
                    similarity = difflib.SequenceMatcher(
                        None,
                        canonical["normalized_text"],
                        variant["normalized_text"]
                    ).ratio()

                    edges.append({
                        "source": variant["concept_id"],
                        "target": canonical["concept_id"],
                        "type": "variant",
                        "similarity": similarity
                    })

            # Create graph object
            graph = {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_concepts": self.stats["total_concepts"],
                    "duplicate_sets": self.stats["duplicate_sets"]
                }
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported similarity graph to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting similarity graph: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Find and merge duplicate concepts in the database")

    parser.add_argument(
        "--db-path",
        help="Path to the database file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Size of concept batches to process (default: 1000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making changes"
    )
    parser.add_argument(
        "--update-normalized",
        action="store_true",
        help="Update normalized_text for all concepts"
    )
    parser.add_argument(
        "--export-graph",
        help="Export similarity graph to the specified file"
    )
    parser.add_argument(
        "--stats-output",
        help="Export statistics to the specified JSON file"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create merger
    merger = ConceptMerger(args.db_path)

    # Update normalized_text if requested
    if args.update_normalized:
        updated = merger.update_normalized_text()
        logger.info(f"Updated normalized_text for {updated} concepts")

    # Export graph if requested
    if args.export_graph:
        merger.export_similarity_graph(args.export_graph)

    # Run merger
    stats = merger.run(args.batch_size, args.debug, args.dry_run)

    # Export stats if requested
    if args.stats_output:
        # Create directory if needed
        os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)

        with open(args.stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            logger.info(f"Exported statistics to {args.stats_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
