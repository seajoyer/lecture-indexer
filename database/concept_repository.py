"""
Concept Repository module for the Lecture Video Content Indexer.
Handles persistence operations for concept data with optimized storage and retrieval.
"""

import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from database.db_manager import DBManager

# Configure logging
logger = logging.getLogger(__name__)

class ConceptRepository:
    """
    Repository for concept data with optimized persistence operations.
    Provides methods to save, retrieve, and query concept information.
    """

    def __init__(self, db_manager: DBManager):
        """
        Initialize the concept repository.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._ensure_schema()
        logger.info("ConceptRepository initialized")

    def _ensure_schema(self):
        """Ensure concept-related tables exist in the database."""
        schema_script = """
        -- Concepts table for storing concept information
        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            normalized_text TEXT,
            stemmed_text TEXT,
            domain TEXT,
            concept_class TEXT,
            total_occurrences INTEGER DEFAULT 0,
            theoretical_occurrences INTEGER DEFAULT 0,
            practical_occurrences INTEGER DEFAULT 0,
            indexed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for concept text search
        CREATE INDEX IF NOT EXISTS idx_concepts_text ON concepts(text);
        CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain);
        CREATE INDEX IF NOT EXISTS idx_concepts_class ON concepts(concept_class);

        -- Occurrences table for concept-segment associations
        CREATE TABLE IF NOT EXISTS occurrences (
            occurrence_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            start_time REAL,
            end_time REAL,
            context_type TEXT,
            context_text TEXT,
            relevance_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE
        );

        -- Create indexes for occurrence queries
        CREATE INDEX IF NOT EXISTS idx_occurrences_concept_id ON occurrences(concept_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_video_id ON occurrences(video_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_segment_id ON occurrences(segment_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_context_type ON occurrences(context_type);

        -- Concept relationships table
        CREATE TABLE IF NOT EXISTS concept_relationships (
            relationship_id TEXT PRIMARY KEY,
            source_concept_id TEXT NOT NULL,
            target_concept_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            relationship_strength REAL,
            co_occurrence_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
            FOREIGN KEY (target_concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE
        );

        -- Create indexes for relationship queries
        CREATE INDEX IF NOT EXISTS idx_concept_relationships_source ON concept_relationships(source_concept_id);
        CREATE INDEX IF NOT EXISTS idx_concept_relationships_target ON concept_relationships(target_concept_id);
        CREATE INDEX IF NOT EXISTS idx_concept_relationships_type ON concept_relationships(relationship_type);

        -- Create FTS table for concept search
        CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
            concept_id,
            text,
            normalized_text,
            stemmed_text,
            domain,
            content='concepts',
            content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2'
        );

        -- Creates a trigger to keep the FTS table in sync with the concepts table
        CREATE TRIGGER IF NOT EXISTS concepts_ai AFTER INSERT ON concepts BEGIN
            INSERT INTO concepts_fts(concept_id, text, normalized_text, stemmed_text, domain)
            VALUES (new.concept_id, new.text, new.normalized_text, new.stemmed_text, new.domain);
        END;

        CREATE TRIGGER IF NOT EXISTS concepts_ad AFTER DELETE ON concepts BEGIN
            DELETE FROM concepts_fts WHERE concept_id = old.concept_id;
        END;

        CREATE TRIGGER IF NOT EXISTS concepts_au AFTER UPDATE ON concepts BEGIN
            DELETE FROM concepts_fts WHERE concept_id = old.concept_id;
            INSERT INTO concepts_fts(concept_id, text, normalized_text, stemmed_text, domain)
            VALUES (new.concept_id, new.text, new.normalized_text, new.stemmed_text, new.domain);
        END;
        """

        try:
            self.db_manager.execute_script(schema_script)
            logger.info("Concept repository schema initialized")
        except Exception as e:
            logger.error(f"Error initializing concept repository schema: {e}")
            raise

    def save_concept(self, concept_data: Dict[str, Any]) -> str:
        """
        Save or update a concept.

        Args:
            concept_data: Concept data dictionary

        Returns:
            Concept ID
        """
        # Extract concept information
        concept_text = concept_data.get("text", "")
        normalized_text = concept_data.get("normalized_text", concept_text.lower())
        stemmed_text = concept_data.get("stemmed_text", normalized_text)
        domain = concept_data.get("domain", "unknown")

        # Generate concept ID if not provided
        concept_id = concept_data.get("concept_id")
        if not concept_id:
            # Create deterministic ID based on text and domain
            concept_id = hashlib.md5(f"{normalized_text}:{domain}".encode()).hexdigest()

        # Determine concept class
        concept_class = concept_data.get("concept_class", "")
        if not concept_class:
            theoretical = concept_data.get("theoretical", False)
            concept_class = "theoretical" if theoretical else "practical"

        # Get occurrence counts
        total_occurrences = concept_data.get("total_occurrences", 0)
        theoretical_occurrences = concept_data.get("theoretical_occurrences", 0)
        practical_occurrences = concept_data.get("practical_occurrences", 0)

        # Set timestamps
        current_time = datetime.now().isoformat()
        indexed_at = concept_data.get("indexed_at", current_time)

        try:
            # Check if concept exists
            existing = self.get_concept(concept_id)

            with self.db_manager.transaction() as cursor:
                if existing:
                    # Update existing concept
                    cursor.execute("""
                    UPDATE concepts SET
                        text = ?,
                        normalized_text = ?,
                        stemmed_text = ?,
                        domain = ?,
                        concept_class = ?,
                        total_occurrences = ?,
                        theoretical_occurrences = ?,
                        practical_occurrences = ?,
                        indexed_at = ?,
                        updated_at = ?
                    WHERE concept_id = ?
                    """, (
                        concept_text,
                        normalized_text,
                        stemmed_text,
                        domain,
                        concept_class,
                        total_occurrences,
                        theoretical_occurrences,
                        practical_occurrences,
                        indexed_at,
                        current_time,
                        concept_id
                    ))
                else:
                    # Insert new concept
                    cursor.execute("""
                    INSERT INTO concepts (
                        concept_id, text, normalized_text, stemmed_text,
                        domain, concept_class, total_occurrences,
                        theoretical_occurrences, practical_occurrences,
                        indexed_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        concept_id,
                        concept_text,
                        normalized_text,
                        stemmed_text,
                        domain,
                        concept_class,
                        total_occurrences,
                        theoretical_occurrences,
                        practical_occurrences,
                        indexed_at,
                        current_time,
                        current_time
                    ))

            logger.info(f"Saved concept {concept_id}: {concept_text}")
            return concept_id

        except Exception as e:
            logger.error(f"Error saving concept {concept_text}: {e}")
            raise

    def save_occurrences(self, concept_id: str, occurrences: List[Dict[str, Any]]) -> bool:
        """
        Save concept occurrences.

        Args:
            concept_id: Concept ID
            occurrences: List of occurrence dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not concept_id or not occurrences:
            return False

        try:
            # Prepare batch insert
            query = """
            INSERT OR REPLACE INTO occurrences (
                occurrence_id, concept_id, video_id, segment_id,
                start_time, end_time, context_type, context_text,
                relevance_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            current_time = datetime.now().isoformat()
            params_list = []

            for occurrence in occurrences:
                video_id = occurrence.get("video_id")
                segment_id = occurrence.get("segment_id")

                if not video_id or not segment_id:
                    continue

                # Generate occurrence ID
                occurrence_id = occurrence.get("occurrence_id")
                if not occurrence_id:
                    occurrence_id = hashlib.md5(f"{concept_id}:{segment_id}".encode()).hexdigest()

                params_list.append((
                    occurrence_id,
                    concept_id,
                    video_id,
                    segment_id,
                    occurrence.get("start_time", 0.0),
                    occurrence.get("end_time", 0.0),
                    occurrence.get("context_type", "mixed"),
                    occurrence.get("context_text", ""),
                    occurrence.get("relevance_score", 0.5),
                    current_time
                ))

            if params_list:
                self.db_manager.execute_many(query, params_list)

                # Update concept occurrence counts
                self._update_concept_occurrence_counts(concept_id)

            logger.info(f"Saved {len(params_list)} occurrences for concept {concept_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving occurrences for concept {concept_id}: {e}")
            return False

    def _update_concept_occurrence_counts(self, concept_id: str):
        """
        Update the occurrence counts for a concept.

        Args:
            concept_id: Concept ID
        """
        try:
            with self.db_manager.transaction() as cursor:
                # Get total occurrences
                cursor.execute("""
                SELECT COUNT(*) as total FROM occurrences WHERE concept_id = ?
                """, (concept_id,))
                result = cursor.fetchone()
                total_occurrences = result[0] if result else 0

                # Get theoretical occurrences
                cursor.execute("""
                SELECT COUNT(*) as theoretical FROM occurrences
                WHERE concept_id = ? AND context_type = 'theoretical'
                """, (concept_id,))
                result = cursor.fetchone()
                theoretical_occurrences = result[0] if result else 0

                # Get practical occurrences
                cursor.execute("""
                SELECT COUNT(*) as practical FROM occurrences
                WHERE concept_id = ? AND context_type = 'practical'
                """, (concept_id,))
                result = cursor.fetchone()
                practical_occurrences = result[0] if result else 0

                # Update concept
                cursor.execute("""
                UPDATE concepts SET
                    total_occurrences = ?,
                    theoretical_occurrences = ?,
                    practical_occurrences = ?,
                    updated_at = ?
                WHERE concept_id = ?
                """, (
                    total_occurrences,
                    theoretical_occurrences,
                    practical_occurrences,
                    datetime.now().isoformat(),
                    concept_id
                ))

        except Exception as e:
            logger.error(f"Error updating occurrence counts for concept {concept_id}: {e}")

    def save_concept_relationship(
        self,
        source_concept_id: str,
        target_concept_id: str,
        relationship_type: str,
        relationship_strength: float = 0.5,
        co_occurrence_count: int = 0
    ) -> bool:
        """
        Save a relationship between two concepts.

        Args:
            source_concept_id: Source concept ID
            target_concept_id: Target concept ID
            relationship_type: Type of relationship ('broader', 'narrower', 'related')
            relationship_strength: Strength of relationship (0.0-1.0)
            co_occurrence_count: Number of co-occurrences

        Returns:
            True if successful, False otherwise
        """
        if not source_concept_id or not target_concept_id:
            return False

        # Generate relationship ID
        relationship_id = hashlib.md5(
            f"{source_concept_id}:{target_concept_id}:{relationship_type}".encode()
        ).hexdigest()

        current_time = datetime.now().isoformat()

        try:
            query = """
            INSERT OR REPLACE INTO concept_relationships (
                relationship_id, source_concept_id, target_concept_id,
                relationship_type, relationship_strength, co_occurrence_count,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            self.db_manager.execute_update(query, (
                relationship_id,
                source_concept_id,
                target_concept_id,
                relationship_type,
                relationship_strength,
                co_occurrence_count,
                current_time,
                current_time
            ))

            logger.info(f"Saved relationship between concepts {source_concept_id} and {target_concept_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving concept relationship: {e}")
            return False

    def delete_concept_occurrences_for_video(self, video_id: str) -> bool:
        """
        Delete all concept occurrences for a video.

        Args:
            video_id: Video ID

        Returns:
            True if successful, False otherwise
        """
        if not video_id:
            return False

        try:
            # Get all concepts affected by this deletion
            affected_concepts = self.db_manager.execute_query("""
            SELECT DISTINCT concept_id FROM occurrences WHERE video_id = ?
            """, (video_id,))

            # Delete occurrences
            self.db_manager.execute_update("""
            DELETE FROM occurrences WHERE video_id = ?
            """, (video_id,))

            # Update occurrence counts for affected concepts
            for concept in affected_concepts:
                self._update_concept_occurrence_counts(concept["concept_id"])

            logger.info(f"Deleted all concept occurrences for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting concept occurrences for video {video_id}: {e}")
            return False

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by ID.

        Args:
            concept_id: Concept ID

        Returns:
            Concept dictionary or None if not found
        """
        if not concept_id:
            return None

        query = "SELECT * FROM concepts WHERE concept_id = ?"
        results = self.db_manager.execute_query(query, (concept_id,))

        return results[0] if results else None

    def get_concept_by_text(self, text: str, domain: str = None) -> Optional[Dict[str, Any]]:
        """
        Get a concept by text.

        Args:
            text: Concept text
            domain: Optional domain filter

        Returns:
            Concept dictionary or None if not found
        """
        if not text:
            return None

        query = "SELECT * FROM concepts WHERE text = ?"
        params = [text]

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        results = self.db_manager.execute_query(query, tuple(params))

        return results[0] if results else None

    def get_concept_occurrences(
        self,
        concept_id: str,
        video_id: str = None,
        context_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get occurrences of a concept.

        Args:
            concept_id: Concept ID
            video_id: Optional video ID filter
            context_type: Optional context type filter
            limit: Maximum number of occurrences to return
            offset: Number of occurrences to skip

        Returns:
            List of occurrence dictionaries
        """
        if not concept_id:
            return []

        query = "SELECT * FROM occurrences WHERE concept_id = ?"
        params = [concept_id]

        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)

        if context_type:
            query += " AND context_type = ?"
            params.append(context_type)

        query += " ORDER BY relevance_score DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        return self.db_manager.execute_query(query, tuple(params))

    def get_concept_relationships(
        self,
        concept_id: str,
        relationship_type: str = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a concept.

        Args:
            concept_id: Concept ID
            relationship_type: Optional relationship type filter
            limit: Maximum number of relationships to return

        Returns:
            List of relationship dictionaries with related concept information
        """
        if not concept_id:
            return []

        query = """
        SELECT r.*, c.text as target_text, c.domain as target_domain,
               c.concept_class as target_class
        FROM concept_relationships r
        JOIN concepts c ON r.target_concept_id = c.concept_id
        WHERE r.source_concept_id = ?
        """
        params = [concept_id]

        if relationship_type:
            query += " AND r.relationship_type = ?"
            params.append(relationship_type)

        query += " ORDER BY r.relationship_strength DESC LIMIT ?"
        params.append(limit)

        return self.db_manager.execute_query(query, tuple(params))

    def search_concepts(
        self,
        query_text: str,
        domain: str = None,
        concept_class: str = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for concepts matching a query.

        Args:
            query_text: Search query text
            domain: Optional domain filter
            concept_class: Optional concept class filter
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            List of matching concept dictionaries
        """
        if not query_text:
            return []

        # Normalize query text
        query_text = query_text.strip().lower()

        # Prepare full-text search query
        fts_query = f'"{query_text}" OR {query_text}'

        # Build SQL query
        query = """
        SELECT c.*, fts.rank
        FROM concepts c
        JOIN concepts_fts fts ON c.concept_id = fts.concept_id
        WHERE concepts_fts MATCH ?
        """
        params = [fts_query]

        if domain:
            query += " AND c.domain = ?"
            params.append(domain)

        if concept_class:
            query += " AND c.concept_class = ?"
            params.append(concept_class)

        query += " ORDER BY fts.rank, c.total_occurrences DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        return self.db_manager.execute_query(query, tuple(params))

    def get_concepts_for_video(
        self,
        video_id: str,
        context_type: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get concepts extracted from a video.

        Args:
            video_id: Video ID
            context_type: Optional context type filter
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            List of concept dictionaries with occurrence information
        """
        if not video_id:
            return []

        query = """
        SELECT c.*, COUNT(o.occurrence_id) as occurrence_count,
               MAX(o.relevance_score) as max_relevance
        FROM concepts c
        JOIN occurrences o ON c.concept_id = o.concept_id
        WHERE o.video_id = ?
        """
        params = [video_id]

        if context_type:
            query += " AND o.context_type = ?"
            params.append(context_type)

        query += " GROUP BY c.concept_id ORDER BY max_relevance DESC, occurrence_count DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        return self.db_manager.execute_query(query, tuple(params))

    def get_videos_for_concept(
        self,
        concept_id: str,
        theory_practice_ratio: float = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get videos containing a concept.

        Args:
            concept_id: Concept ID
            theory_practice_ratio: Optional theory/practice ratio filter
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            List of video dictionaries with occurrence information
        """
        if not concept_id:
            return []

        query = """
        SELECT v.*, COUNT(o.occurrence_id) as occurrence_count,
               MAX(o.relevance_score) as max_relevance
        FROM videos v
        JOIN occurrences o ON v.video_id = o.video_id
        WHERE o.concept_id = ?
        """
        params = [concept_id]

        if theory_practice_ratio is not None:
            # Apply theory/practice filter as a range
            min_ratio = max(0.0, theory_practice_ratio - 0.2)
            max_ratio = min(1.0, theory_practice_ratio + 0.2)
            query += " AND v.theory_practice_ratio BETWEEN ? AND ?"
            params.extend([min_ratio, max_ratio])

        query += " GROUP BY v.video_id ORDER BY max_relevance DESC, occurrence_count DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        return self.db_manager.execute_query(query, tuple(params))

    def analyze_co_occurrences(self, min_co_occurrences: int = 3) -> bool:
        """
        Analyze concept co-occurrences and build concept relationships.
        This is a computationally intensive operation best run as a background task.

        Args:
            min_co_occurrences: Minimum number of co-occurrences to establish a relationship

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all concepts
            concepts = self.db_manager.execute_query(
                "SELECT concept_id, domain FROM concepts"
            )

            if not concepts:
                return True  # No concepts to analyze

            logger.info(f"Analyzing co-occurrences for {len(concepts)} concepts")
            start_time = time.time()

            with self.db_manager.transaction() as cursor:
                # For each concept pair, count co-occurrences in the same video
                for i, concept1 in enumerate(concepts):
                    concept1_id = concept1["concept_id"]
                    concept1_domain = concept1["domain"]

                    # Get videos containing this concept
                    cursor.execute("""
                    SELECT DISTINCT video_id FROM occurrences WHERE concept_id = ?
                    """, (concept1_id,))

                    videos = [row[0] for row in cursor.fetchall()]

                    if not videos:
                        continue

                    # Find co-occurring concepts
                    video_placeholders = ','.join(['?'] * len(videos))
                    query = f"""
                    SELECT concept_id, COUNT(DISTINCT video_id) as co_occurrences
                    FROM occurrences
                    WHERE video_id IN ({video_placeholders})
                    AND concept_id != ?
                    GROUP BY concept_id
                    HAVING co_occurrences >= ?
                    """

                    params = videos + [concept1_id, min_co_occurrences]
                    cursor.execute(query, params)
                    co_occurrences = cursor.fetchall()

                    # Create relationships
                    for row in co_occurrences:
                        concept2_id = row[0]
                        co_occurrence_count = row[1]

                        # Get concept2 domain
                        cursor.execute("""
                        SELECT domain FROM concepts WHERE concept_id = ?
                        """, (concept2_id,))
                        concept2_domain_row = cursor.fetchone()

                        if not concept2_domain_row:
                            continue

                        concept2_domain = concept2_domain_row[0]

                        # Calculate relationship strength based on co-occurrence count
                        relationship_strength = min(1.0, co_occurrence_count / 20)

                        # Determine relationship type
                        if concept1_domain == concept2_domain:
                            relationship_type = "related"
                        else:
                            relationship_type = "cross_domain"

                        # Create relationship
                        relationship_id = hashlib.md5(
                            f"{concept1_id}:{concept2_id}:{relationship_type}".encode()
                        ).hexdigest()

                        current_time = datetime.now().isoformat()

                        cursor.execute("""
                        INSERT OR REPLACE INTO concept_relationships (
                            relationship_id, source_concept_id, target_concept_id,
                            relationship_type, relationship_strength, co_occurrence_count,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            relationship_id,
                            concept1_id,
                            concept2_id,
                            relationship_type,
                            relationship_strength,
                            co_occurrence_count,
                            current_time,
                            current_time
                        ))

                    # Log progress periodically
                    if (i + 1) % 100 == 0 or i == len(concepts) - 1:
                        elapsed = time.time() - start_time
                        logger.info(f"Processed {i + 1}/{len(concepts)} concepts in {elapsed:.2f} seconds")

            logger.info(f"Co-occurrence analysis completed in {time.time() - start_time:.2f} seconds")
            return True

        except Exception as e:
            logger.error(f"Error analyzing concept co-occurrences: {e}")
            return False

    def count_concepts(self, domain: str = None, concept_class: str = None) -> int:
        """
        Count concepts in the repository.

        Args:
            domain: Optional domain filter
            concept_class: Optional concept class filter

        Returns:
            Number of concepts
        """
        query = "SELECT COUNT(*) as count FROM concepts"
        params = []

        conditions = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)

        if concept_class:
            conditions.append("concept_class = ?")
            params.append(concept_class)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        results = self.db_manager.execute_query(query, tuple(params))
        return results[0]["count"] if results else 0

    def get_concept_stats(self) -> Dict[str, Any]:
        """
        Get statistics about concepts in the repository.

        Returns:
            Dictionary of concept statistics
        """
        try:
            stats = {}

            # Total concept count
            total = self.count_concepts()
            stats["total_concepts"] = total

            # Concepts by domain
            domain_query = """
            SELECT domain, COUNT(*) as count
            FROM concepts
            GROUP BY domain
            ORDER BY count DESC
            """
            domain_stats = self.db_manager.execute_query(domain_query)
            stats["concepts_by_domain"] = domain_stats

            # Concepts by class
            class_query = """
            SELECT concept_class, COUNT(*) as count
            FROM concepts
            GROUP BY concept_class
            ORDER BY count DESC
            """
            class_stats = self.db_manager.execute_query(class_query)
            stats["concepts_by_class"] = class_stats

            # Top concepts by occurrences
            top_query = """
            SELECT concept_id, text, domain, concept_class, total_occurrences
            FROM concepts
            ORDER BY total_occurrences DESC
            LIMIT 10
            """
            top_concepts = self.db_manager.execute_query(top_query)
            stats["top_concepts"] = top_concepts

            # Relationship stats
            relationship_query = """
            SELECT relationship_type, COUNT(*) as count
            FROM concept_relationships
            GROUP BY relationship_type
            """
            relationship_stats = self.db_manager.execute_query(relationship_query)
            stats["relationships"] = relationship_stats

            return stats

        except Exception as e:
            logger.error(f"Error getting concept stats: {e}")
            return {"error": str(e)}
