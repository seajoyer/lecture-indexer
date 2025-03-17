"""
Search Engine module for the Lecture Video Content Indexer.
Handles search queries and retrieval with theory/practice filtering.
"""

import os
import json
import logging
import time
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
from pathlib import Path
import threading

# Configure logging
logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Search engine for the Lecture Video Content Indexer.
    Supports theory/practice filtering and domain-specific search.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Search Engine with configuration.

        Args:
            config: Configuration dictionary
        """
        logger.info("Initializing Search Engine")

        self.config = config
        self.index_dir = config.get("index_dir", "data/index")

        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        # Initialize SQLite database for indexing
        self.db_path = Path(self.index_dir) / "index.db"
        self._init_database()

        # Thread lock for database operations
        self.db_lock = threading.Lock()

        logger.info("Search Engine initialized")

    def _init_database(self):
        """Initialize SQLite database for indexing."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create videos table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                channel TEXT,
                publication_date TEXT,
                duration_seconds INTEGER,
                language TEXT,
                domain TEXT,
                domain_confidence REAL,
                theory_practice_ratio REAL,
                theoretical_segments INTEGER,
                practical_segments INTEGER,
                indexed_at TEXT
            )
            ''')

            # Create concepts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id TEXT PRIMARY KEY,
                text TEXT,
                normalized_text TEXT,
                domain TEXT,
                concept_class TEXT,  -- theoretical, practical, both
                total_occurrences INTEGER,
                theoretical_occurrences INTEGER,
                practical_occurrences INTEGER,
                indexed_at TEXT
            )
            ''')

            # Create occurrences table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS occurrences (
                occurrence_id TEXT PRIMARY KEY,
                concept_id TEXT,
                video_id TEXT,
                segment_id TEXT,
                start_time REAL,
                end_time REAL,
                context_type TEXT,  -- theoretical, practical, mixed
                context_text TEXT,
                relevance_score REAL,
                FOREIGN KEY (concept_id) REFERENCES concepts (concept_id),
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
            ''')

            # Create theory_practice_patterns table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS theory_practice_patterns (
                pattern_id TEXT PRIMARY KEY,
                video_id TEXT,
                pattern_type TEXT,  -- theory_to_practice, practice_to_theory
                pattern_subtype TEXT,  -- domain-specific pattern type
                start_segment_id TEXT,
                end_segment_id TEXT,
                start_time REAL,
                end_time REAL,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
            ''')

            # Create FTS (Full-Text Search) virtual table for concepts
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
                concept_id,
                text,
                normalized_text,
                domain,
                content='concepts',
                content_rowid='rowid'
            )
            ''')

            # Create FTS (Full-Text Search) virtual table for segments
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                segment_id,
                video_id,
                text,
                domain,
                context_type
            )
            ''')

            # Create a segments table that includes timestamps
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                video_id TEXT,
                start_time REAL,
                end_time REAL,
                text TEXT,
                domain TEXT,
                context_type TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
            ''')

            # Create index on segments
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id)
            ''')

            # Create index on occurrences
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_occurrences_concept_id ON occurrences(concept_id)
            ''')
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_occurrences_video_id ON occurrences(video_id)
            ''')
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_occurrences_context_type ON occurrences(context_type)
            ''')

            # Create index on theory_practice_patterns
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_video_id ON theory_practice_patterns(video_id)
            ''')
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_pattern_type ON theory_practice_patterns(pattern_type)
            ''')

            # Create a trigger to automatically update the FTS table when concepts are inserted/updated
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS concepts_ai AFTER INSERT ON concepts BEGIN
                INSERT INTO concepts_fts(concept_id, text, normalized_text, domain)
                VALUES (new.concept_id, new.text, new.normalized_text, new.domain);
            END;
            ''')

            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS concepts_ad AFTER DELETE ON concepts BEGIN
                DELETE FROM concepts_fts WHERE concept_id = old.concept_id;
            END;
            ''')

            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS concepts_au AFTER UPDATE ON concepts BEGIN
                DELETE FROM concepts_fts WHERE concept_id = old.concept_id;
                INSERT INTO concepts_fts(concept_id, text, normalized_text, domain)
                VALUES (new.concept_id, new.text, new.normalized_text, new.domain);
            END;
            ''')

            conn.commit()
            conn.close()

            logger.info("Database initialized")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content in the search engine.

        Args:
            processed_result: Processing result dictionary

        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            # Extract key fields
            video_id = processed_result.get("video_id")
            metadata = processed_result.get("metadata", {})
            transcript = processed_result.get("transcript", {})
            domain_features = processed_result.get("domain_features", {})
            theory_practice_results = processed_result.get("theory_practice_results", {})
            theory_practice_patterns = processed_result.get("theory_practice_patterns", {})

            if not video_id or not metadata:
                logger.error("Missing required fields for indexing")
                return False

            # Log domain features
            key_concepts = domain_features.get("key_concepts", [])
            logger.info(f"Indexing video {video_id} with {len(key_concepts)} key concepts")

            # If no key concepts were found, try to extract them from domain keywords
            if not key_concepts and metadata.get("domain"):
                logger.warning(f"No key concepts found for video {video_id}, attempting domain-based extraction")
                key_concepts = self._extract_fallback_concepts(
                    transcript.get("segments", []),
                    metadata.get("domain"),
                    metadata.get("language", "en")
                )
                domain_features["key_concepts"] = key_concepts
                logger.info(f"Extracted {len(key_concepts)} fallback concepts based on domain keywords")

            # Lock database for thread safety
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                try:
                    # Begin transaction
                    conn.execute("BEGIN TRANSACTION")

                    # Index video
                    self._index_video(cursor, video_id, metadata, theory_practice_results)

                    # Index segments and extract concepts
                    segments = transcript.get("segments", [])
                    self._index_segments(cursor, video_id, segments)

                    # Index concepts
                    key_concepts = domain_features.get("key_concepts", [])
                    self._index_concepts(cursor, video_id, key_concepts, segments)

                    # Index theory-practice patterns
                    self._index_theory_practice_patterns(cursor, video_id, theory_practice_patterns)

                    # Commit transaction
                    conn.commit()
                    logger.info(f"Successfully indexed content for video {video_id}")
                    return True

                except Exception as e:
                    # Rollback on error
                    conn.rollback()
                    logger.error(f"Error during indexing: {e}")
                    return False

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return False

    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a search query against the index.

        Args:
            query: Structured query dictionary

        Returns:
            Search results dictionary
        """
        start_time = time.time()

        try:
            # Extract query fields
            query_text = query.get("original_text", "").strip()
            filters = query.get("filters", {})
            theory_practice_ratio = query.get("theory_practice_ratio")
            domain = query.get("domain")
            pagination = query.get("pagination", {})

            offset = pagination.get("offset", 0)
            limit = pagination.get("limit", 10)

            if not query_text:
                logger.warning("Empty search query")
                return {
                    "results": [],
                    "totalResults": 0,
                    "theoreticalResults": 0,
                    "practicalResults": 0,
                    "executionTimeMs": 0,
                    "query": query
                }

            # Log search parameters
            logger.info(f"Searching for '{query_text}' with theory/practice ratio: {theory_practice_ratio}, domain: {domain}")

            # Lock database for thread safety
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row  # Use row factory for dict-like results
                cursor = conn.cursor()

                try:
                    # Try direct concept search first
                    sql_query, params = self._build_concept_search_query(
                        query_text, filters, theory_practice_ratio, domain
                    )

                    # Get total count
                    count_sql = f"SELECT COUNT(*) FROM ({sql_query})"
                    cursor.execute(count_sql, params)
                    total_results = cursor.fetchone()[0]

                    # If no direct concept matches, try segment text search
                    segment_search_used = False
                    if total_results == 0:
                        logger.info(f"No direct concept matches for '{query_text}', trying segment text search")
                        try:
                            sql_query, params = self._build_segment_search_query(
                                query_text, filters, theory_practice_ratio, domain
                            )
                            segment_search_used = True

                            # Get total count again
                            count_sql = f"SELECT COUNT(*) FROM ({sql_query})"
                            cursor.execute(count_sql, params)
                            total_results = cursor.fetchone()[0]
                        except Exception as e:
                            logger.error(f"Error in segment search: {e}")
                            # Fallback to empty results
                            total_results = 0
                            segment_search_used = False
                            sql_query = "SELECT 1 WHERE 0"  # Empty query

                    # Get counts by context type
                    theoretical_results = 0
                    practical_results = 0

                    if total_results > 0:
                        theoretical_sql = f"SELECT COUNT(*) FROM ({sql_query}) WHERE context_type = 'theoretical'"
                        try:
                            cursor.execute(theoretical_sql, params)
                            theoretical_results = cursor.fetchone()[0]
                        except Exception as e:
                            logger.warning(f"Error counting theoretical results: {e}")

                        practical_sql = f"SELECT COUNT(*) FROM ({sql_query}) WHERE context_type = 'practical'"
                        try:
                            cursor.execute(practical_sql, params)
                            practical_results = cursor.fetchone()[0]
                        except Exception as e:
                            logger.warning(f"Error counting practical results: {e}")

                    # Add pagination
                    if total_results > 0:
                        paginated_sql = f"{sql_query} LIMIT ? OFFSET ?"
                        paginated_params = params + [limit, offset]

                        # Execute query
                        cursor.execute(paginated_sql, paginated_params)
                        rows = cursor.fetchall()
                    else:
                        rows = []

                    # Convert to list of dicts
                    results = []
                    for row in rows:
                        result = dict(row)

                        # Enhance with related concepts if this is a concept result
                        if not segment_search_used and result.get("concept_id"):
                            result["related_concepts"] = self._get_related_concepts(
                                cursor, result.get("concept_id")
                            )

                        # Get video title if needed
                        if "video_title" not in result and result.get("video_id"):
                            video_id = result.get("video_id")
                            try:
                                cursor.execute("SELECT title FROM videos WHERE video_id = ?", (video_id,))
                                video_row = cursor.fetchone()
                                if video_row:
                                    result["video_title"] = video_row[0]
                            except Exception as e:
                                logger.warning(f"Error getting video title: {e}")

                        results.append(result)

                    # Calculate execution time
                    execution_time_ms = int((time.time() - start_time) * 1000)

                    # Prepare response
                    response = {
                        "results": results,
                        "totalResults": total_results,
                        "theoreticalResults": theoretical_results,
                        "practicalResults": practical_results,
                        "executionTimeMs": execution_time_ms,
                        "query": query
                    }

                    logger.info(f"Search for '{query_text}' returned {total_results} results in {execution_time_ms}ms")
                    return response

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error executing search query: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())

            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "totalResults": 0,
                "theoreticalResults": 0,
                "practicalResults": 0,
                "executionTimeMs": execution_time_ms,
                "error": str(e),
                "query": query
            }

    def get_concept_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept.

        Args:
            concept_id: Concept ID

        Returns:
            Concept details dictionary if found, None otherwise
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                try:
                    # Get concept details
                    cursor.execute(
                        "SELECT * FROM concepts WHERE concept_id = ?",
                        (concept_id,)
                    )
                    concept_row = cursor.fetchone()

                    if not concept_row:
                        return None

                    concept = dict(concept_row)

                    # Get occurrences
                    cursor.execute(
                        """
                        SELECT o.*, v.title AS video_title
                        FROM occurrences o
                        JOIN videos v ON o.video_id = v.video_id
                        WHERE o.concept_id = ?
                        ORDER BY o.relevance_score DESC
                        """,
                        (concept_id,)
                    )
                    occurrence_rows = cursor.fetchall()
                    occurrences = [dict(row) for row in occurrence_rows]

                    # Get related concepts
                    related_concepts = self._get_related_concepts(cursor, concept_id)

                    # Get theoretical foundations (for practical concepts)
                    theoretical_foundations = []
                    if concept.get("concept_class") == "practical":
                        cursor.execute(
                            """
                            SELECT c.* FROM concepts c
                            JOIN occurrences o1 ON c.concept_id = o1.concept_id
                            JOIN occurrences o2 ON o1.video_id = o2.video_id
                            WHERE o2.concept_id = ? AND c.concept_class = 'theoretical'
                            GROUP BY c.concept_id
                            ORDER BY COUNT(*) DESC
                            LIMIT 10
                            """,
                            (concept_id,)
                        )
                        foundation_rows = cursor.fetchall()
                        theoretical_foundations = [dict(row) for row in foundation_rows]

                    # Get practical applications (for theoretical concepts)
                    practical_applications = []
                    if concept.get("concept_class") == "theoretical":
                        cursor.execute(
                            """
                            SELECT c.* FROM concepts c
                            JOIN occurrences o1 ON c.concept_id = o1.concept_id
                            JOIN occurrences o2 ON o1.video_id = o2.video_id
                            WHERE o2.concept_id = ? AND c.concept_class = 'practical'
                            GROUP BY c.concept_id
                            ORDER BY COUNT(*) DESC
                            LIMIT 10
                            """,
                            (concept_id,)
                        )
                        application_rows = cursor.fetchall()
                        practical_applications = [dict(row) for row in application_rows]

                    # Compile result
                    result = {
                        "concept": concept,
                        "occurrences": occurrences,
                        "related": related_concepts,
                        "theoretical_foundations": theoretical_foundations,
                        "practical_applications": practical_applications
                    }

                    return result

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error getting concept details: {e}")
            return None

    def get_video_concepts(self,
        video_id: str,
        context_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get concepts extracted from a video.

        Args:
            video_id: YouTube video ID
            context_type: Content type filter (theoretical, practical, mixed)

        Returns:
            Video concepts dictionary if found, None otherwise
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                try:
                    # Get video details
                    cursor.execute(
                        "SELECT * FROM videos WHERE video_id = ?",
                        (video_id,)
                    )
                    video_row = cursor.fetchone()

                    if not video_row:
                        return None

                    video = dict(video_row)

                    # Get concepts
                    sql = """
                    SELECT c.*, COUNT(o.occurrence_id) AS occurrence_count
                    FROM concepts c
                    JOIN occurrences o ON c.concept_id = o.concept_id
                    WHERE o.video_id = ?
                    """
                    params = [video_id]

                    if context_type:
                        sql += " AND o.context_type = ?"
                        params.append(context_type)

                    sql += " GROUP BY c.concept_id ORDER BY occurrence_count DESC"

                    cursor.execute(sql, params)
                    concept_rows = cursor.fetchall()
                    concepts = [dict(row) for row in concept_rows]

                    # Get theory-practice patterns
                    cursor.execute(
                        """
                        SELECT * FROM theory_practice_patterns
                        WHERE video_id = ?
                        ORDER BY start_time
                        """,
                        (video_id,)
                    )
                    pattern_rows = cursor.fetchall()
                    patterns = [dict(row) for row in pattern_rows]

                    # Compile result
                    result = {
                        "video": video,
                        "concepts": concepts,
                        "theory_practice_patterns": patterns,
                        "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
                    }

                    return result

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error getting video concepts: {e}")
            return None

    def generate_learning_path(
        self,
        concept_ids: List[str],
        theory_practice_ratio: float = 0.5,
        domain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a learning path for a set of concepts.

        Args:
            concept_ids: List of concept IDs
            theory_practice_ratio: Desired ratio of theoretical to practical content
            domain: Optional domain filter

        Returns:
            Learning path dictionary if successful, None otherwise
        """
        try:
            if not concept_ids:
                return None

            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                try:
                    # Get concepts
                    placeholders = ", ".join(["?"] * len(concept_ids))
                    cursor.execute(
                        f"SELECT * FROM concepts WHERE concept_id IN ({placeholders})",
                        concept_ids
                    )
                    concept_rows = cursor.fetchall()
                    target_concepts = [dict(row) for row in concept_rows]

                    if not target_concepts:
                        return None

                    # Determine the domain if not specified
                    if not domain:
                        # Use the most common domain among target concepts
                        domains = {}
                        for concept in target_concepts:
                            concept_domain = concept.get("domain")
                            if concept_domain:
                                domains[concept_domain] = domains.get(concept_domain, 0) + 1

                        if domains:
                            domain = max(domains.items(), key=lambda x: x[1])[0]

                    # Get prerequisite concepts
                    prerequisite_concepts = []

                    # For each target concept, find potential prerequisites
                    for concept in target_concepts:
                        concept_id = concept.get("concept_id")

                        # Find concepts that frequently appear before this concept
                        cursor.execute(
                            """
                            SELECT c.*, COUNT(*) AS co_occurrence
                            FROM concepts c
                            JOIN occurrences o1 ON c.concept_id = o1.concept_id
                            JOIN occurrences o2 ON o1.video_id = o2.video_id
                                AND o1.start_time < o2.start_time
                            WHERE o2.concept_id = ?
                            AND (? IS NULL OR c.domain = ?)
                            AND c.concept_id NOT IN ({})
                            GROUP BY c.concept_id
                            ORDER BY co_occurrence DESC
                            LIMIT 5
                            """.format(placeholders),
                            [concept_id, domain, domain] + concept_ids
                        )

                        prereq_rows = cursor.fetchall()

                        for row in prereq_rows:
                            prereq = dict(row)
                            if prereq not in prerequisite_concepts:
                                prerequisite_concepts.append(prereq)

                    # Combine target and prerequisite concepts
                    all_concepts = target_concepts + prerequisite_concepts

                    # Sort concepts by theoretical/practical class based on desired ratio
                    theoretical_concepts = [c for c in all_concepts if c.get("concept_class") == "theoretical"]
                    practical_concepts = [c for c in all_concepts if c.get("concept_class") == "practical"]
                    mixed_concepts = [c for c in all_concepts if c.get("concept_class") not in ("theoretical", "practical")]

                    # Calculate actual numbers based on ratio
                    total_concepts = len(all_concepts)
                    target_theoretical = int(total_concepts * theory_practice_ratio)
                    target_practical = total_concepts - target_theoretical

                    # Adjust for available concepts
                    available_theoretical = len(theoretical_concepts)
                    available_practical = len(practical_concepts)

                    if available_theoretical < target_theoretical:
                        # Add some mixed concepts to theoretical
                        needed = min(target_theoretical - available_theoretical, len(mixed_concepts))
                        theoretical_concepts.extend(mixed_concepts[:needed])
                        mixed_concepts = mixed_concepts[needed:]

                    if available_practical < target_practical:
                        # Add remaining mixed concepts to practical
                        needed = min(target_practical - available_practical, len(mixed_concepts))
                        practical_concepts.extend(mixed_concepts[:needed])

                    # Create learning path
                    learning_path = []

                    # Add concepts in an alternating pattern based on ratio
                    t_index = 0
                    p_index = 0

                    # If ratio < 0.5, start with practical, otherwise start with theoretical
                    start_with_theoretical = theory_practice_ratio >= 0.5

                    for i in range(total_concepts):
                        if start_with_theoretical:
                            # Add theoretical, then practical
                            if i % 2 == 0 and t_index < len(theoretical_concepts):
                                learning_path.append(theoretical_concepts[t_index])
                                t_index += 1
                            elif p_index < len(practical_concepts):
                                learning_path.append(practical_concepts[p_index])
                                p_index += 1
                            elif t_index < len(theoretical_concepts):
                                learning_path.append(theoretical_concepts[t_index])
                                t_index += 1
                        else:
                            # Add practical, then theoretical
                            if i % 2 == 0 and p_index < len(practical_concepts):
                                learning_path.append(practical_concepts[p_index])
                                p_index += 1
                            elif t_index < len(theoretical_concepts):
                                learning_path.append(theoretical_concepts[t_index])
                                t_index += 1
                            elif p_index < len(practical_concepts):
                                learning_path.append(practical_concepts[p_index])
                                p_index += 1

                    # Add any remaining concepts
                    while t_index < len(theoretical_concepts):
                        learning_path.append(theoretical_concepts[t_index])
                        t_index += 1

                    while p_index < len(practical_concepts):
                        learning_path.append(practical_concepts[p_index])
                        p_index += 1

                    # Estimate time for each concept
                    for i, concept in enumerate(learning_path):
                        cursor.execute(
                            """
                            SELECT AVG(o.end_time - o.start_time) as avg_duration
                            FROM occurrences o
                            WHERE o.concept_id = ?
                            """,
                            (concept.get("concept_id"),)
                        )

                        row = cursor.fetchone()
                        avg_duration = row[0] if row and row[0] else 300  # Default to 5 minutes

                        concept["order"] = i + 1
                        concept["estimated_time_minutes"] = round(avg_duration / 60)

                    # Calculate total time
                    total_time = sum(c.get("estimated_time_minutes", 0) for c in learning_path)

                    # Count theoretical and practical concepts
                    path_theoretical = sum(1 for c in learning_path if c.get("concept_class") == "theoretical")
                    path_practical = sum(1 for c in learning_path if c.get("concept_class") == "practical")

                    # Calculate actual ratio
                    actual_ratio = path_theoretical / total_concepts if total_concepts > 0 else 0.5

                    # Compile result
                    result = {
                        "concepts": learning_path,
                        "theory_practice_ratio": actual_ratio,
                        "total_theoretical_concepts": path_theoretical,
                        "total_practical_concepts": path_practical,
                        "estimated_total_time_minutes": total_time,
                        "domain": domain
                    }

                    return result

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return None

    def _index_video(
        self,
        cursor: sqlite3.Cursor,
        video_id: str,
        metadata: Dict[str, Any],
        theory_practice_results: Dict[str, Any]
    ):
        """
        Index video metadata in the database.

        Args:
            cursor: Database cursor
            video_id: YouTube video ID
            metadata: Video metadata
            theory_practice_results: Theory/practice classification results
        """
        # Extract fields
        title = metadata.get("title", "")
        description = metadata.get("description", "")
        channel = metadata.get("channel", "")
        publication_date = metadata.get("publication_date", "")
        duration_seconds = metadata.get("duration_seconds", 0)
        language = metadata.get("language", "en")
        domain = metadata.get("domain", "unknown")
        domain_confidence = metadata.get("domain_confidence", 0.0)

        # Theory/practice data
        theory_practice_ratio = theory_practice_results.get("theory_practice_ratio", 0.5)
        theoretical_segments = theory_practice_results.get("theoretical_segments", 0)
        practical_segments = theory_practice_results.get("practical_segments", 0)

        # Current timestamp
        indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # Insert or update video
        cursor.execute(
            """
            INSERT OR REPLACE INTO videos (
                video_id, title, description, channel, publication_date,
                duration_seconds, language, domain, domain_confidence,
                theory_practice_ratio, theoretical_segments, practical_segments,
                indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id, title, description, channel, publication_date,
                duration_seconds, language, domain, domain_confidence,
                theory_practice_ratio, theoretical_segments, practical_segments,
                indexed_at
            )
        )

    def _index_segments(
        self,
        cursor: sqlite3.Cursor,
        video_id: str,
        segments: List[Dict[str, Any]]
    ):
        """
        Index transcript segments in the database.

        Args:
            cursor: Database cursor
            video_id: YouTube video ID
            segments: List of transcript segments
        """
        # Clear previous segments for this video
        cursor.execute("DELETE FROM segments_fts WHERE video_id = ?", (video_id,))
        cursor.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))

        # Insert new segments
        for segment in segments:
            segment_id = segment.get("id")
            text = segment.get("text", "")
            context_type = segment.get("content_type", "mixed")
            domain = segment.get("domain", "unknown")
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)

            if not segment_id or not text:
                continue

            # Insert into segments_fts for full-text search
            cursor.execute(
                """
                INSERT INTO segments_fts (
                    segment_id, video_id, text, domain, context_type
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (segment_id, video_id, text, domain, context_type)
            )

            # Insert into segments table with timestamps
            cursor.execute(
                """
                INSERT INTO segments (
                    segment_id, video_id, start_time, end_time, text, domain, context_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (segment_id, video_id, start_time, end_time, text, domain, context_type)
            )

    def _index_concepts(
        self,
        cursor: sqlite3.Cursor,
        video_id: str,
        key_concepts: List[Dict[str, Any]],
        segments: List[Dict[str, Any]]
    ):
        """
        Index concepts in the database.

        Args:
            cursor: Database cursor
            video_id: YouTube video ID
            key_concepts: List of key concepts
            segments: List of transcript segments
        """
        # Current timestamp
        indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # Clear previous occurrences for this video
        cursor.execute("DELETE FROM occurrences WHERE video_id = ?", (video_id,))

        # Log the number of concepts being indexed
        logger.info(f"Indexing {len(key_concepts)} concepts for video {video_id}")

        # Index each concept
        for concept in key_concepts:
            concept_text = concept.get("text", "")
            normalized_text = concept_text.lower().strip()
            domain = concept.get("domain", "unknown")
            theoretical = concept.get("theoretical", False)

            # Skip empty concepts
            if not concept_text or len(concept_text) < 2:
                continue

            # Generate concept ID based on text and domain
            import hashlib
            concept_id = hashlib.md5(f"{normalized_text}:{domain}".encode()).hexdigest()

            # Determine concept class (theoretical, practical, both)
            concept_class = "theoretical" if theoretical else "practical"

            # Find occurrences in segments
            occurrences = []
            for segment in segments:
                segment_text = segment.get("text", "").lower()
                # Use word boundary to ensure we're matching the whole concept, not part of a word
                pattern = r'\b' + re.escape(normalized_text) + r'\b'
                if re.search(pattern, segment_text, re.IGNORECASE):
                    segment_id = segment.get("id")
                    start_time = segment.get("start_time", 0)
                    end_time = segment.get("end_time", 0)
                    context_type = segment.get("content_type", "mixed")

                    # Calculate relevance score based on concept frequency in segment
                    matches = re.findall(pattern, segment_text, re.IGNORECASE)
                    match_count = len(matches)
                    word_count = len(segment_text.split())
                    relevance_score = min(1.0, (match_count / max(1, word_count / 10)) * 0.3 + 0.7)

                    occurrences.append({
                        "segment_id": segment_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "context_type": context_type,
                        "context_text": segment.get("text", ""),
                        "relevance_score": relevance_score
                    })

            # Skip concepts with no occurrences
            if not occurrences:
                continue

            # Count occurrences by context type
            total_occurrences = len(occurrences)
            theoretical_occurrences = sum(1 for o in occurrences if o["context_type"] == "theoretical")
            practical_occurrences = sum(1 for o in occurrences if o["context_type"] == "practical")

            # Insert or update concept
            cursor.execute(
                """
                INSERT OR REPLACE INTO concepts (
                    concept_id, text, normalized_text, domain, concept_class,
                    total_occurrences, theoretical_occurrences, practical_occurrences,
                    indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept_id, concept_text, normalized_text, domain, concept_class,
                    total_occurrences, theoretical_occurrences, practical_occurrences,
                    indexed_at
                )
            )

            # Insert occurrences
            for occurrence in occurrences:
                occurrence_id = hashlib.md5(f"{concept_id}:{occurrence['segment_id']}".encode()).hexdigest()

                cursor.execute(
                    """
                    INSERT INTO occurrences (
                        occurrence_id, concept_id, video_id, segment_id,
                        start_time, end_time, context_type, context_text, relevance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        occurrence_id, concept_id, video_id, occurrence["segment_id"],
                        occurrence["start_time"], occurrence["end_time"], occurrence["context_type"],
                        occurrence["context_text"], occurrence["relevance_score"]
                    )
                )

    def _index_theory_practice_patterns(
        self,
        cursor: sqlite3.Cursor,
        video_id: str,
        theory_practice_patterns: Dict[str, Any]
    ):
        """
        Index theory-practice patterns in the database.

        Args:
            cursor: Database cursor
            video_id: YouTube video ID
            theory_practice_patterns: Theory-practice patterns dictionary
        """
        # Clear previous patterns for this video
        cursor.execute("DELETE FROM theory_practice_patterns WHERE video_id = ?", (video_id,))

        # Extract patterns
        theory_to_practice = theory_practice_patterns.get("theory_to_practice_sequences", [])
        practice_to_theory = theory_practice_patterns.get("practice_to_theory_sequences", [])

        # Index theory-to-practice patterns
        for pattern in theory_to_practice:
            pattern_id = str(uuid.uuid4())
            pattern_type = "theory_to_practice"
            pattern_subtype = pattern.get("pattern_type", "general_theory_to_practice")

            segments = pattern.get("segments", [])
            if not segments:
                continue

            start_segment_id = segments[0].get("id") if segments else None
            end_segment_id = segments[-1].get("id") if segments else None
            start_time = segments[0].get("start_time", 0) if segments else 0
            end_time = segments[-1].get("end_time", 0) if segments else 0

            cursor.execute(
                """
                INSERT INTO theory_practice_patterns (
                    pattern_id, video_id, pattern_type, pattern_subtype,
                    start_segment_id, end_segment_id, start_time, end_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_id, video_id, pattern_type, pattern_subtype,
                    start_segment_id, end_segment_id, start_time, end_time
                )
            )

        # Index practice-to-theory patterns
        for pattern in practice_to_theory:
            pattern_id = str(uuid.uuid4())
            pattern_type = "practice_to_theory"
            pattern_subtype = pattern.get("pattern_type", "general_practice_to_theory")

            segments = pattern.get("segments", [])
            if not segments:
                continue

            start_segment_id = segments[0].get("id") if segments else None
            end_segment_id = segments[-1].get("id") if segments else None
            start_time = segments[0].get("start_time", 0) if segments else 0
            end_time = segments[-1].get("end_time", 0) if segments else 0

            cursor.execute(
                """
                INSERT INTO theory_practice_patterns (
                    pattern_id, video_id, pattern_type, pattern_subtype,
                    start_segment_id, end_segment_id, start_time, end_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_id, video_id, pattern_type, pattern_subtype,
                    start_segment_id, end_segment_id, start_time, end_time
                )
            )

    def _build_search_query(
            self,
            query_text: str,
            filters: Dict[str, Any],
            theory_practice_ratio: Optional[float],
            domain: Optional[str]
        ) -> Tuple[str, List[Any]]:
            """
            Build SQL query for searching.

            Note: This method exists for backward compatibility with tests.

            Args:
                query_text: Search query text
                filters: Additional filters
                theory_practice_ratio: Theory/practice ratio filter
                domain: Domain filter

            Returns:
                Tuple of (SQL query, parameters)
            """
            # For test compatibility, call the concept search query builder
            return self._build_concept_search_query(query_text, filters, theory_practice_ratio, domain)

    def _build_concept_search_query(
        self,
        query_text: str,
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> Tuple[str, List[Any]]:
        """
        Build SQL query for searching concepts.

        Args:
            query_text: Search query text
            filters: Additional filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            Tuple of (SQL query, parameters)
        """

        # Base query
        sql = """
        SELECT c.*, o.video_id, o.segment_id, o.start_time, o.end_time,
               o.context_type, o.context_text, o.relevance_score, o.occurrence_id
        FROM concepts c
        JOIN concepts_fts f ON c.concept_id = f.concept_id
        JOIN occurrences o ON c.concept_id = o.concept_id
        WHERE concepts_fts MATCH ?
        """
        params = [query_text]

        # Apply domain filter
        if domain:
            sql += " AND c.domain = ?"
            params.append(domain)

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.8:
                # Mostly theoretical
                sql += " AND (o.context_type = 'theoretical' OR c.concept_class = 'theoretical')"
            elif theory_practice_ratio < 0.2:
                # Mostly practical
                sql += " AND (o.context_type = 'practical' OR c.concept_class = 'practical')"
            elif theory_practice_ratio < 0.5:
                # Favor practical
                sql += " ORDER BY CASE WHEN o.context_type = 'practical' THEN 1 ELSE 2 END, o.relevance_score DESC"
            else:
                # Favor theoretical
                sql += " ORDER BY CASE WHEN o.context_type = 'theoretical' THEN 1 ELSE 2 END, o.relevance_score DESC"
        else:
            # Default ordering by relevance
            sql += " ORDER BY o.relevance_score DESC"

        return sql, params

    def _build_segment_search_query(
        self,
        query_text: str,
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> Tuple[str, List[Any]]:
        """
        Build SQL query for searching segments.

        Args:
            query_text: Search query text
            filters: Additional filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            Tuple of (SQL query, parameters)
        """
        # Build a query that uses the segments table (with timestamps)
        # but searches through segments_fts
        sql = """
            SELECT s.segment_id, s.video_id, s.text AS context_text,
                s.context_type, v.title, v.domain,
                NULL as concept_id, NULL as text, NULL as normalized_text,
                NULL as concept_class, NULL as occurrence_id,
                0.8 as relevance_score, s.start_time, s.end_time
            FROM segments s
            JOIN segments_fts fts ON s.segment_id = fts.segment_id
            JOIN videos v ON s.video_id = v.video_id
            WHERE segments_fts MATCH ?
            """
        params = [query_text]

        # Apply domain filter
        if domain:
            sql += " AND s.domain = ?"
            params.append(domain)

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.8:
                # Mostly theoretical
                sql += " AND s.context_type = 'theoretical'"
            elif theory_practice_ratio < 0.2:
                # Mostly practical
                sql += " AND s.context_type = 'practical'"
            elif theory_practice_ratio < 0.5:
                # Favor practical
                sql += " ORDER BY CASE WHEN s.context_type = 'practical' THEN 1 ELSE 2 END"
            else:
                # Favor theoretical
                sql += " ORDER BY CASE WHEN s.context_type = 'theoretical' THEN 1 ELSE 2 END"
        else:
            # No specific ordering
            sql += " ORDER BY fts.rowid"

        return sql, params

    def _get_related_concepts(self, cursor: sqlite3.Cursor, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get concepts related to the given concept.

        Args:
            cursor: Database cursor
            concept_id: Concept ID

        Returns:
            List of related concept dictionaries
        """
        if not concept_id:
            return []

        cursor.execute(
            """
            SELECT c.*, COUNT(*) AS co_occurrence
            FROM concepts c
            JOIN occurrences o1 ON c.concept_id = o1.concept_id
            JOIN occurrences o2 ON o1.video_id = o2.video_id
            WHERE o2.concept_id = ? AND c.concept_id != ?
            GROUP BY c.concept_id
            ORDER BY co_occurrence DESC
            LIMIT 5
            """,
            (concept_id, concept_id)
        )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def _extract_fallback_concepts(
        self,
        segments: List[Dict[str, Any]],
        domain: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Extract fallback concepts based on domain keywords when no concepts were extracted.

        Args:
            segments: List of transcript segments
            domain: Content domain (mathematics, programming, physics)
            language: Language of content (en, ru)

        Returns:
            List of extracted concepts
        """
        # Domain-specific keywords to look for
        domain_keywords = {
            "mathematics": {
                "en": [
                    "theorem", "lemma", "proof", "equation", "function", "calculus",
                    "derivative", "integral", "algebra", "geometry", "matrix",
                    "vector", "set", "topology", "analysis", "group", "field"
                ],
                "ru": [
                    "теорема", "лемма", "доказательство", "уравнение", "функция",
                    "анализ", "производная", "интеграл", "алгебра", "геометрия",
                    "матрица", "вектор", "множество", "топология", "группа", "поле"
                ]
            },
            "programming": {
                "en": [
                    "algorithm", "function", "method", "class", "object", "variable",
                    "array", "list", "tree", "graph", "stack", "queue", "heap", "sort"
                ],
                "ru": [
                    "алгоритм", "функция", "метод", "класс", "объект", "переменная",
                    "массив", "список", "дерево", "граф", "стек", "очередь", "куча",
                    "сортировка"
                ]
            },
            "physics": {
                "en": [
                    "force", "energy", "momentum", "mass", "velocity", "acceleration",
                    "gravity", "field", "wave", "particle", "quantum", "relativity",
                    "electron", "proton", "neutron", "photon", "atomic", "nuclear",
                    "state", "entanglement", "superposition", "spin", "collapse"
                ],
                "ru": [
                    "сила", "энергия", "импульс", "масса", "скорость", "ускорение",
                    "гравитация", "поле", "волна", "частица", "квант", "относительность",
                    "электрон", "протон", "нейтрон", "фотон", "атомный", "ядерный",
                    "состояние", "запутанность", "суперпозиция", "спин", "коллапс",
                    "квантовый", "квантовая", "квантовое", "состояние", "измерение",
                    "квантовая механика"
                ]
            }
        }

        # Use keywords for specific domain and language
        keywords = domain_keywords.get(domain, {}).get(language, [])
        if not keywords and language == "ru":
            # Fallback to English if no Russian keywords
            keywords = domain_keywords.get(domain, {}).get("en", [])

        if not keywords:
            logger.warning(f"No keywords found for domain {domain} and language {language}")
            return []

        # Combine all segment texts
        full_text = " ".join([s.get("text", "") for s in segments])

        # Find all occurrences of each keyword
        concepts = []
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, full_text.lower())
            frequency = len(matches)

            if frequency > 0:
                # Determine if theoretical based on segments where it appears
                theoretical = self._is_concept_theoretical_in_segments(keyword, segments)

                concepts.append({
                    "text": keyword,
                    "domain": domain,
                    "frequency": frequency,
                    "theoretical": theoretical
                })

        # Return top concepts by frequency
        return sorted(concepts, key=lambda x: x["frequency"], reverse=True)[:20]

    def _is_concept_theoretical_in_segments(self, concept: str, segments: List[Dict[str, Any]]) -> bool:
        """Determine if a concept is predominantly theoretical based on the segments it appears in."""
        theoretical_count = 0
        practical_count = 0

        for segment in segments:
            text = segment.get("text", "").lower()
            if re.search(r'\b' + re.escape(concept.lower()) + r'\b', text):
                content_type = segment.get("content_type", "")
                if content_type == "theoretical":
                    theoretical_count += 1
                elif content_type == "practical":
                    practical_count += 1

        # If the concept appears more in theoretical segments, classify it as theoretical
        return theoretical_count >= practical_count
