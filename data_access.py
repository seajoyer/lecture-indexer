"""
Enhanced data access layer for the Video Lecture Content Indexer.

Provides optimized database operations with support for the new concept model.
Handles persistence of concepts, occurrences, and relationships.
"""

import os
import sqlite3
import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

class DataAccess:
    """
    Enhanced data access class for database operations.
    Provides connection pooling, improved security, and optimized queries.
    """

    def __init__(self, db_path: str = "data/index/indexer.db"):
        """
        Initialize the data access layer with database path.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize connection pool
        self._pool_size = 5  # Maximum number of connections
        self._pool = []
        self._pool_lock = threading.Lock()

        # Initialize database schema
        self._ensure_schema()

        logger.info(f"DataAccess initialized with database at {db_path}")

    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new database connection with appropriate settings.

        Returns:
            SQLite connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Performance optimizations
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous = NORMAL")  # Better performance with reasonable safety
        conn.execute("PRAGMA cache_size = 10000")  # Larger cache for better performance
        conn.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory
        conn.execute("PRAGMA mmap_size = 30000000")  # Memory-mapped I/O (30MB)

        return conn

    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection from the pool with automatic return.

        Returns:
            SQLite connection context manager
        """
        connection = None

        # Try to get a connection from the pool
        with self._pool_lock:
            if self._pool:
                connection = self._pool.pop()

        # Create new connection if none available from pool
        if connection is None:
            connection = self._create_connection()

        try:
            # Yield the connection for use
            yield connection
        except sqlite3.Error as e:
            # Log error and re-raise
            logger.error(f"Database error: {e}")
            raise
        finally:
            # Return connection to pool if it's still operational
            try:
                # Test connection with a simple query
                connection.execute("SELECT 1")

                # Add connection back to pool if not already full
                with self._pool_lock:
                    if len(self._pool) < self._pool_size:
                        self._pool.append(connection)
                    else:
                        connection.close()
            except sqlite3.Error:
                # Connection is broken, don't return to pool
                try:
                    connection.close()
                except:
                    pass

    def _ensure_schema(self) -> None:
        """
        Ensure all necessary tables exist in the database with optimized schema
        for the new concept model.
        """
        schema_script = """
        -- Enable foreign key constraints
        PRAGMA foreign_keys = ON;

        -- Videos table for storing basic video metadata
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            channel TEXT,
            publication_date TEXT,
            duration_seconds INTEGER,
            language TEXT,
            indexed_at TEXT,
            processing_status TEXT
        );

        -- Create indexes for videos
        CREATE INDEX IF NOT EXISTS idx_videos_language ON videos(language);
        CREATE INDEX IF NOT EXISTS idx_videos_processing_status ON videos(processing_status);

        -- Segments table for storing transcript segments
        CREATE TABLE IF NOT EXISTS segments (
            segment_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            start_time REAL,
            text TEXT,
            language TEXT,
            educational_value REAL,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );

        -- Create indexes for segments
        CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);
        CREATE INDEX IF NOT EXISTS idx_segments_start_time ON segments(start_time);
        CREATE INDEX IF NOT EXISTS idx_segments_language ON segments(language);
        CREATE INDEX IF NOT EXISTS idx_segments_educational_value ON segments(educational_value);

        -- Concepts table for storing repository concepts
        CREATE TABLE IF NOT EXISTS repository_concepts (
            concept_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,  -- JSON data with representations and relationships
            language TEXT,       -- Primary language
            last_updated TEXT
        );

        -- Create index on concept language
        CREATE INDEX IF NOT EXISTS idx_concepts_language ON repository_concepts(language);

        -- Concept representations table for searchable representations
        CREATE TABLE IF NOT EXISTS concept_representations (
            representation_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            language TEXT NOT NULL,
            text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            FOREIGN KEY (concept_id) REFERENCES repository_concepts(concept_id) ON DELETE CASCADE
        );

        -- Create indexes for concept representations
        CREATE INDEX IF NOT EXISTS idx_representations_concept_id ON concept_representations(concept_id);
        CREATE INDEX IF NOT EXISTS idx_representations_language ON concept_representations(language);
        CREATE INDEX IF NOT EXISTS idx_representations_text ON concept_representations(text);
        CREATE INDEX IF NOT EXISTS idx_representations_normalized_text ON concept_representations(normalized_text);

        -- Concept relationships table for the concept graph
        CREATE TABLE IF NOT EXISTS concept_relationships (
            relationship_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,  -- 'prerequisite' or 'related'
            FOREIGN KEY (source_id) REFERENCES repository_concepts(concept_id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES repository_concepts(concept_id) ON DELETE CASCADE
        );

        -- Create indexes for concept relationships
        CREATE INDEX IF NOT EXISTS idx_relationships_source_id ON concept_relationships(source_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_target_id ON concept_relationships(target_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON concept_relationships(relationship_type);

        -- Occurrences table for concept-segment associations with educational significance
        CREATE TABLE IF NOT EXISTS occurrences (
            occurrence_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            start_time REAL,
            educational_significance REAL,
            occurrence_type TEXT,  -- 'comprehensive' or 'passing'
            similarity REAL,
            context_text TEXT,
            FOREIGN KEY (concept_id) REFERENCES repository_concepts(concept_id) ON DELETE CASCADE,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE
        );

        -- Create indexes for occurrences
        CREATE INDEX IF NOT EXISTS idx_occurrences_concept_id ON occurrences(concept_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_video_id ON occurrences(video_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_segment_id ON occurrences(segment_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_type ON occurrences(occurrence_type);
        CREATE INDEX IF NOT EXISTS idx_occurrences_significance ON occurrences(educational_significance);

        -- Search index using FTS5
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            id,
            text,
            language,
            item_type,
            video_id,
            educational_significance,
            tokenize='unicode61 remove_diacritics 1'
        );
        """

        try:
            with self.get_connection() as conn:
                conn.executescript(schema_script)
                conn.commit()

            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise

    def execute_query(self, query: str, params: Union[tuple, list] = ()) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as a list of dictionaries.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                columns = [col[0] for col in cursor.description] if cursor.description else []
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
        except sqlite3.Error as e:
            logger.error(f"Query error: {e}, Query: {query}, Params: {params}")
            raise

    def execute_update(self, query: str, params: Union[tuple, list] = ()) -> int:
        """
        Execute an update query and return the number of affected rows.

        Args:
            query: SQL update query
            params: Query parameters

        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Update error: {e}, Query: {query}, Params: {params}")
            raise

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute multiple updates with different parameters.

        Args:
            query: SQL query
            params_list: List of parameter tuples

        Returns:
            Number of affected rows
        """
        if not params_list:
            return 0

        try:
            with self.get_connection() as conn:
                # Use batching for better performance with large param lists
                batch_size = 1000  # Process in batches to avoid memory issues
                total_affected = 0

                for i in range(0, len(params_list), batch_size):
                    batch = params_list[i:i + batch_size]
                    cursor = conn.cursor()
                    cursor.executemany(query, batch)
                    total_affected += cursor.rowcount

                conn.commit()
                return total_affected
        except sqlite3.Error as e:
            logger.error(f"Batch update error: {e}, Query: {query}, "
                        f"Params count: {len(params_list)}")
            raise

    # VIDEO OPERATIONS

    def save_video(self, video_data: Dict[str, Any]) -> bool:
        """
        Save or update video metadata.

        Args:
            video_data: Video metadata dictionary

        Returns:
            True if successful, False otherwise
        """
        video_id = video_data.get("video_id")
        if not video_id:
            logger.error("Cannot save video without video_id")
            return False

        try:
            # Check if video exists
            existing = self.get_video(video_id)

            if existing:
                # Update existing video
                query = """
                UPDATE videos SET
                    title = ?,
                    description = ?,
                    channel = ?,
                    publication_date = ?,
                    duration_seconds = ?,
                    language = ?,
                    indexed_at = ?,
                    processing_status = ?
                WHERE video_id = ?
                """
                self.execute_update(query, (
                    video_data.get("title", ""),
                    video_data.get("description", ""),
                    video_data.get("channel", ""),
                    video_data.get("publication_date", ""),
                    video_data.get("duration_seconds", 0),
                    video_data.get("language", ""),
                    video_data.get("indexed_at", ""),
                    video_data.get("processing_status", "completed"),
                    video_id
                ))
            else:
                # Insert new video
                query = """
                INSERT INTO videos (
                    video_id, title, description, channel, publication_date,
                    duration_seconds, language, indexed_at, processing_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.execute_update(query, (
                    video_id,
                    video_data.get("title", ""),
                    video_data.get("description", ""),
                    video_data.get("channel", ""),
                    video_data.get("publication_date", ""),
                    video_data.get("duration_seconds", 0),
                    video_data.get("language", ""),
                    video_data.get("indexed_at", ""),
                    video_data.get("processing_status", "completed")
                ))

            logger.info(f"Saved video metadata for {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving video {video_id}: {e}")
            return False

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata by ID.

        Args:
            video_id: Video ID

        Returns:
            Video metadata dictionary or None if not found
        """
        query = "SELECT * FROM videos WHERE video_id = ?"
        results = self.execute_query(query, (video_id,))

        if not results:
            return None

        return results[0]

    def save_segments(self, video_id: str, segments: List[Dict[str, Any]]) -> bool:
        """
        Save video transcript segments.

        Args:
            video_id: Video ID
            segments: List of segment dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not video_id or not segments:
            logger.error("Cannot save segments without video_id or segments")
            return False

        try:
            # Delete existing segments
            self.execute_update("DELETE FROM segments WHERE video_id = ?", (video_id,))

            # Delete segment occurrences from search index
            self.execute_update(
                "DELETE FROM search_index WHERE item_type = 'segment' AND video_id = ?",
                (video_id,)
            )

            # Prepare batch insert
            query = """
            INSERT INTO segments (
                segment_id, video_id, start_time, text, language, educational_value
            ) VALUES (?, ?, ?, ?, ?, ?)
            """

            # Transform segments into parameter tuples
            params_list = []
            search_index_params = []

            for segment in segments:
                segment_id = segment.get("id")
                if not segment_id:
                    segment_id = f"segment_{video_id}_{time.time()}_{len(params_list)}"

                language = segment.get("language", "")
                educational_value = segment.get("educational_value", 0.0)

                params_list.append((
                    segment_id,
                    video_id,
                    segment.get("start_time", 0.0),
                    segment.get("text", ""),
                    language,
                    educational_value
                ))

                # Prepare search index parameters
                search_index_params.append((
                    segment_id,
                    segment.get("text", ""),
                    language,
                    "segment",
                    video_id,
                    educational_value
                ))

            # Execute batch insert for segments
            self.execute_many(query, params_list)

            # Insert into search index
            if search_index_params:
                search_query = """
                INSERT INTO search_index (
                    id, text, language, item_type, video_id, educational_significance
                ) VALUES (?, ?, ?, ?, ?, ?)
                """
                self.execute_many(search_query, search_index_params)

            logger.info(f"Saved {len(segments)} segments for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving segments for video {video_id}: {e}")
            return False

    def get_video_segments(
        self,
        video_id: str,
        min_educational_value: Optional[float] = None,
        start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get segments for a video with optional filtering.

        Args:
            video_id: Video ID
            min_educational_value: Optional minimum educational value
            start_time: Optional filter for minimum start time

        Returns:
            List of segment dictionaries
        """
        # Build query parameters
        query = "SELECT * FROM segments WHERE video_id = ?"
        params = [video_id]

        if min_educational_value is not None:
            query += " AND educational_value >= ?"
            params.append(min_educational_value)

        if start_time is not None:
            query += " AND start_time >= ?"
            params.append(start_time)

        query += " ORDER BY start_time"

        results = self.execute_query(query, tuple(params))
        return results

    # CONCEPT OPERATIONS

    def save_repository_concept(self, concept_data: Dict[str, Any]) -> bool:
        """
        Save a concept to the repository.

        Args:
            concept_data: Concept data dictionary

        Returns:
            True if successful, False otherwise
        """
        concept_id = concept_data.get("concept_id")
        if not concept_id:
            logger.error("Cannot save concept without concept_id")
            return False

        try:
            # Get primary language - first available language or 'en'
            languages = list(concept_data.get("representations", {}).keys())
            primary_language = languages[0] if languages else "en"

            # Serialize entire concept data as JSON
            serialized_data = json.dumps(concept_data, ensure_ascii=False)

            # Check if concept exists
            existing_query = "SELECT concept_id FROM repository_concepts WHERE concept_id = ?"
            existing = self.execute_query(existing_query, (concept_id,))

            now = time.strftime("%Y-%m-%d %H:%M:%S")

            if existing:
                # Update existing concept
                query = """
                UPDATE repository_concepts SET
                    data = ?,
                    language = ?,
                    last_updated = ?
                WHERE concept_id = ?
                """
                self.execute_update(query, (
                    serialized_data,
                    primary_language,
                    now,
                    concept_id
                ))
            else:
                # Insert new concept
                query = """
                INSERT INTO repository_concepts (
                    concept_id, data, language, last_updated
                ) VALUES (?, ?, ?, ?)
                """
                self.execute_update(query, (
                    concept_id,
                    serialized_data,
                    primary_language,
                    now
                ))

            # Try to update representations - continue even if this fails
            try:
                self._update_concept_representations(concept_id, concept_data)
            except Exception as e:
                logger.error(f"Error updating representations for concept {concept_id}: {e}")
                # Continue despite error - the core concept is saved

            # Try to update relationships - continue even if this fails
            try:
                self._update_concept_relationships(concept_id, concept_data)
            except Exception as e:
                logger.error(f"Error updating relationships for concept {concept_id}: {e}")
                # Continue despite error - the core concept is saved

            logger.info(f"Saved repository concept {concept_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving repository concept {concept_id}: {e}")
            return False

    def _update_concept_representations(self, concept_id: str, concept_data: Dict[str, Any]) -> None:
        """
        Update concept representations in the database.

        Args:
            concept_id: Concept ID
            concept_data: Concept data dictionary
        """
        try:
            # Delete existing representations
            delete_query = "DELETE FROM concept_representations WHERE concept_id = ?"
            self.execute_update(delete_query, (concept_id,))

            # Insert new representations
            insert_query = """
            INSERT INTO concept_representations (
                representation_id, concept_id, language, text, normalized_text
            ) VALUES (?, ?, ?, ?, ?)
            """

            params_list = []
            representations = concept_data.get("representations", {})

            for language, texts in representations.items():
                for text in texts:
                    # Skip empty texts
                    if not text:
                        continue

                    # Create a unique ID for this representation
                    representation_id = f"{concept_id}_{language}_{hash(text)}"

                    # Normalize text
                    normalized_text = self._normalize_text(text)

                    params_list.append((
                        representation_id,
                        concept_id,
                        language,
                        text,
                        normalized_text
                    ))

            # Insert all representations
            if params_list:
                self.execute_many(insert_query, params_list)

        except Exception as e:
            logger.error(f"Error updating representations for concept {concept_id}: {e}")
            raise

    def _update_concept_relationships(self, concept_id: str, concept_data: Dict[str, Any]) -> None:
        """
        Update concept relationships in the database.

        Args:
            concept_id: Concept ID
            concept_data: Concept data dictionary
        """
        try:
            # Delete existing relationships where this concept is the source
            delete_query = "DELETE FROM concept_relationships WHERE source_id = ?"
            self.execute_update(delete_query, (concept_id,))

            # Get all existing concept IDs from the database
            existing_concepts_query = "SELECT concept_id FROM repository_concepts"
            existing_concepts_result = self.execute_query(existing_concepts_query)
            existing_concept_ids = set(row.get("concept_id") for row in existing_concepts_result)

            # Make sure the source concept exists
            if concept_id not in existing_concept_ids:
                # This should never happen, but just in case
                logger.warning(f"Source concept {concept_id} doesn't exist in the database")
                return

            # Insert new relationships
            insert_query = """
            INSERT INTO concept_relationships (
                relationship_id, source_id, target_id, relationship_type
            ) VALUES (?, ?, ?, ?)
            """

            params_list = []
            skipped_prereqs = []
            skipped_related = []

            # Add prerequisites
            prerequisites = concept_data.get("prerequisites", [])
            for prereq_id in prerequisites:
                # Skip if target concept doesn't exist
                if prereq_id not in existing_concept_ids:
                    skipped_prereqs.append(prereq_id)
                    continue

                relationship_id = f"{concept_id}_prereq_{prereq_id}"
                params_list.append((
                    relationship_id,
                    concept_id,
                    prereq_id,
                    "prerequisite"
                ))

            # Add related concepts
            related = concept_data.get("related", [])
            for related_id in related:
                # Skip if target concept doesn't exist
                if related_id not in existing_concept_ids:
                    skipped_related.append(related_id)
                    continue

                relationship_id = f"{concept_id}_related_{related_id}"
                params_list.append((
                    relationship_id,
                    concept_id,
                    related_id,
                    "related"
                ))

            # Insert all valid relationships
            if params_list:
                self.execute_many(insert_query, params_list)
                logger.info(f"Added {len(params_list)} relationships for concept {concept_id}")

            # Log skipped relationships
            if skipped_prereqs:
                logger.warning(f"Skipped prerequisite relationships for concept {concept_id} to non-existent concepts: {', '.join(skipped_prereqs)}")
            if skipped_related:
                logger.warning(f"Skipped related relationships for concept {concept_id} to non-existent concepts: {', '.join(skipped_related)}")

        except Exception as e:
            logger.error(f"Error updating relationships for concept {concept_id}: {e}")
            # Don't re-raise, just log - this allows the concept to be saved even if relationships fail

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for concept matching.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    def get_repository_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept from the repository.

        Args:
            concept_id: Concept ID

        Returns:
            Concept dictionary or None if not found
        """
        query = "SELECT * FROM repository_concepts WHERE concept_id = ?"
        results = self.execute_query(query, (concept_id,))

        if not results:
            return None

        # Parse JSON data
        concept_data = results[0]
        try:
            return json.loads(concept_data.get("data", "{}"))
        except json.JSONDecodeError:
            logger.error(f"Error decoding concept data for {concept_id}")
            return None

    def find_concepts_by_text(
        self,
        text: str,
        language: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find concepts by text using both exact and fuzzy matching.

        Args:
            text: Text to search for
            language: Optional language filter
            limit: Maximum number of results to return

        Returns:
            List of matching concepts
        """
        # Normalize the search text
        normalized_text = self._normalize_text(text)

        # Build the query
        query = """
        SELECT cr.concept_id, cr.language, cr.text,
               rc.data,
               'exact' as match_type,
               1.0 as similarity
        FROM concept_representations cr
        JOIN repository_concepts rc ON cr.concept_id = rc.concept_id
        WHERE cr.normalized_text = ?
        """

        params = [normalized_text]

        # Add language filter if specified
        if language:
            query += " AND cr.language = ?"
            params.append(language)

        # Add limit
        query += f" LIMIT {limit}"

        # Execute query
        results = self.execute_query(query, tuple(params))

        # Process results
        concept_results = []
        for row in results:
            try:
                concept_data = json.loads(row.get("data", "{}"))
                concept_results.append({
                    "concept_id": row.get("concept_id"),
                    "concept": concept_data,
                    "language": row.get("language"),
                    "text": row.get("text"),
                    "match_type": row.get("match_type"),
                    "similarity": row.get("similarity")
                })
            except json.JSONDecodeError:
                logger.error(f"Error decoding concept data for {row.get('concept_id')}")

        return concept_results

    def list_repository_concepts(
        self,
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List concepts in the repository with pagination.

        Args:
            language: Optional language filter
            limit: Maximum number of concepts to return
            offset: Pagination offset

        Returns:
            List of concepts
        """
        # Build the query
        query = "SELECT * FROM repository_concepts"

        params = []

        # Add language filter if specified
        if language:
            query += " WHERE language = ?"
            params.append(language)

        # Add order and pagination
        query += " ORDER BY last_updated DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute query
        results = self.execute_query(query, tuple(params))

        # Process results
        concept_results = []
        for row in results:
            try:
                concept_data = json.loads(row.get("data", "{}"))
                concept_results.append(concept_data)
            except json.JSONDecodeError:
                logger.error(f"Error decoding concept data for {row.get('concept_id')}")

        return concept_results

    # OCCURRENCE OPERATIONS

    def save_occurrences(self, occurrences: List[Dict[str, Any]]) -> bool:
        """
        Save concept occurrences.

        Args:
            occurrences: List of occurrence dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not occurrences:
            return True  # Nothing to save

        try:
            # First, validate each occurrence to ensure foreign keys exist
            valid_occurrences = []

            # Get all valid concept_ids, video_ids, and segment_ids that exist in the database
            concept_ids_query = "SELECT concept_id FROM repository_concepts"
            video_ids_query = "SELECT video_id FROM videos"
            segment_ids_query = "SELECT segment_id FROM segments"

            valid_concept_ids = set(row['concept_id'] for row in self.execute_query(concept_ids_query))
            valid_video_ids = set(row['video_id'] for row in self.execute_query(video_ids_query))
            valid_segment_ids = set(row['segment_id'] for row in self.execute_query(segment_ids_query))

            # Check each occurrence against valid IDs
            for occurrence in occurrences:
                occurrence_id = occurrence.get("occurrence_id")
                if not occurrence_id:
                    occurrence_id = str(time.time()) + "_" + str(len(valid_occurrences))
                    occurrence["occurrence_id"] = occurrence_id

                concept_id = occurrence.get("concept_id")
                video_id = occurrence.get("video_id")
                segment_id = occurrence.get("segment_id")

                # Skip if any required field is missing
                if not concept_id or not video_id or not segment_id:
                    logger.warning(f"Skipping occurrence with missing required fields: {occurrence}")
                    continue

                # Skip if foreign keys don't exist
                if concept_id not in valid_concept_ids:
                    logger.warning(f"Skipping occurrence with invalid concept_id: {concept_id}")
                    continue

                if video_id not in valid_video_ids:
                    logger.warning(f"Skipping occurrence with invalid video_id: {video_id}")
                    continue

                if segment_id not in valid_segment_ids:
                    logger.warning(f"Skipping occurrence with invalid segment_id: {segment_id}")
                    continue

                # Get educational significance and type
                edu_significance = occurrence.get("educational_significance", 0.0)
                occurrence_type = occurrence.get("occurrence_type", "passing")
                if edu_significance >= 2.5 and occurrence_type == "passing":
                    occurrence_type = "comprehensive"  # Ensure consistency

                # Add validated occurrence
                valid_occurrences.append((
                    occurrence_id,
                    concept_id,
                    video_id,
                    segment_id,
                    occurrence.get("start_time", 0.0),
                    edu_significance,
                    occurrence_type,
                    occurrence.get("similarity", 0.0),
                    occurrence.get("context_text", "")
                ))

            # Prepare batch insert
            query = """
            INSERT OR REPLACE INTO occurrences (
                occurrence_id, concept_id, video_id, segment_id,
                start_time, educational_significance, occurrence_type,
                similarity, context_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            # Execute batch inserts for valid occurrences only
            if valid_occurrences:
                self.execute_many(query, valid_occurrences)

                # Add to search index for high educational significance occurrences
                search_index_params = []
                for occ in valid_occurrences:
                    occurrence_id, concept_id, video_id, segment_id, start_time, edu_significance, occurrence_type, similarity, context_text = occ

                    # Get language from related segment
                    segment_query = "SELECT language FROM segments WHERE segment_id = ?"
                    segment_result = self.execute_query(segment_query, (segment_id,))
                    language = segment_result[0].get('language', 'en') if segment_result else 'en'

                    if edu_significance >= 1.5:  # Include borderline educational content too
                        search_index_params.append((
                            occurrence_id,
                            context_text,
                            language,
                            "occurrence",
                            video_id,
                            edu_significance
                        ))

                # Insert into search index
                if search_index_params:
                    search_query = """
                    INSERT INTO search_index (
                        id, text, language, item_type, video_id, educational_significance
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """
                    self.execute_many(search_query, search_index_params)

            logger.info(f"Saved {len(valid_occurrences)} occurrences (skipped {len(occurrences) - len(valid_occurrences)})")
            return True

        except Exception as e:
            logger.error(f"Error saving occurrences: {e}")
            return False

    def get_concept_occurrences(
        self,
        concept_id: str,
        min_educational_significance: Optional[float] = None,
        occurrence_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get occurrences of a concept.

        Args:
            concept_id: Concept ID
            min_educational_significance: Optional minimum educational significance
            occurrence_type: Optional occurrence type filter ('comprehensive' or 'passing')
            limit: Maximum number of results to return

        Returns:
            List of occurrence dictionaries
        """
        try:
            # Log query parameters for debugging
            logger.info(f"Getting occurrences for concept_id={concept_id}, min_significance={min_educational_significance}, type={occurrence_type}")

            # First check if any occurrences exist for this concept
            check_query = "SELECT COUNT(*) as count FROM occurrences WHERE concept_id = ?"
            check_result = self.execute_query(check_query, (concept_id,))

            if check_result and check_result[0]["count"] == 0:
                logger.warning(f"No occurrences found for concept_id={concept_id} in the database")
                return []
            else:
                logger.info(f"Found {check_result[0]['count']} total occurrences for concept_id={concept_id}")

            # Build the query
            query = """
            SELECT o.*, v.title as video_title, s.text as segment_text
            FROM occurrences o
            JOIN videos v ON o.video_id = v.video_id
            JOIN segments s ON o.segment_id = s.segment_id
            WHERE o.concept_id = ?
            """

            params = [concept_id]

            # Add educational significance filter if specified
            if min_educational_significance is not None:
                query += " AND o.educational_significance >= ?"
                params.append(min_educational_significance)

            # Add occurrence type filter if specified
            if occurrence_type:
                query += " AND o.occurrence_type = ?"
                params.append(occurrence_type)

            # Add order and limit
            query += " ORDER BY o.educational_significance DESC LIMIT ?"
            params.append(limit)

            # Execute query
            results = self.execute_query(query, tuple(params))

            # Additional logging for debugging
            logger.info(f"Query returned {len(results)} occurrences after filtering")

            return results

        except Exception as e:
            logger.error(f"Error retrieving concept occurrences: {e}")
            return []

    def get_video_concepts(
        self,
        video_id: str,
        min_educational_significance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get concepts found in a video.

        Args:
            video_id: Video ID
            min_educational_significance: Optional minimum educational significance

        Returns:
            Dictionary with video and concept information
        """
        # Get video information
        video = self.get_video(video_id)
        if not video:
            return {"video": None, "concepts": []}

        # Get occurrences for this video
        query = """
        SELECT o.*, rc.data as concept_data
        FROM occurrences o
        JOIN repository_concepts rc ON o.concept_id = rc.concept_id
        WHERE o.video_id = ?
        """

        params = [video_id]

        # Add educational significance filter if specified
        if min_educational_significance is not None:
            query += " AND o.educational_significance >= ?"
            params.append(min_educational_significance)

        # Execute query
        occurrences = self.execute_query(query, tuple(params))

        # Group occurrences by concept
        concepts_dict = {}
        for occurrence in occurrences:
            concept_id = occurrence.get("concept_id")
            if not concept_id:
                continue

            # Parse concept data
            try:
                concept_data = json.loads(occurrence.get("concept_data", "{}"))
            except json.JSONDecodeError:
                logger.error(f"Error decoding concept data for {concept_id}")
                continue

            # Add to concepts dictionary
            if concept_id not in concepts_dict:
                concepts_dict[concept_id] = {
                    "concept_id": concept_id,
                    "data": concept_data,
                    "occurrences": []
                }

            # Add occurrence
            concepts_dict[concept_id]["occurrences"].append({
                "occurrence_id": occurrence.get("occurrence_id"),
                "segment_id": occurrence.get("segment_id"),
                "start_time": occurrence.get("start_time"),
                "educational_significance": occurrence.get("educational_significance"),
                "occurrence_type": occurrence.get("occurrence_type"),
                "context_text": occurrence.get("context_text")
            })

        # Convert to list and sort by educational significance
        concepts = list(concepts_dict.values())
        concepts.sort(key=lambda c: max([o.get("educational_significance", 0) for o in c.get("occurrences", [])]), reverse=True)

        # Separate comprehensive and passing concepts
        comprehensive_concepts = []
        passing_concepts = []

        for concept in concepts:
            # Calculate max educational significance
            max_significance = max([o.get("educational_significance", 0) for o in concept.get("occurrences", [])])

            if max_significance >= 2.5:
                comprehensive_concepts.append(concept)
            else:
                passing_concepts.append(concept)

        return {
            "video": video,
            "concepts": concepts,
            "comprehensive_concepts": comprehensive_concepts,
            "passing_concepts": passing_concepts
        }

    # SEARCH OPERATIONS

    def search(
        self,
        query_text: str,
        language: Optional[str] = None,
        min_educational_significance: Optional[float] = None,
        item_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search for concepts, segments, and occurrences.

        Args:
            query_text: Text to search for
            language: Optional language filter
            min_educational_significance: Optional minimum educational significance
            item_types: Optional list of item types to include ('concept', 'segment', 'occurrence')
            limit: Maximum number of results to return

        Returns:
            Dictionary with search results
        """
        if not query_text:
            return {"results": [], "total": 0}

        try:
            # Normalize query text
            normalized_query = self._normalize_text(query_text)

            # Build FTS query
            fts_query = f'"{normalized_query}"'  # Exact phrase match

            # Build query
            query = """
            SELECT si.*, v.title as video_title
            FROM search_index si
            JOIN videos v ON si.video_id = v.video_id
            WHERE si.text MATCH ?
            """

            params = [fts_query]

            # Add language filter if specified
            if language:
                query += " AND si.language = ?"
                params.append(language)

            # Add educational significance filter if specified
            if min_educational_significance is not None:
                query += " AND si.educational_significance >= ?"
                params.append(min_educational_significance)

            # Add item type filter if specified
            if item_types:
                placeholders = ", ".join(["?"] * len(item_types))
                query += f" AND si.item_type IN ({placeholders})"
                params.extend(item_types)

            # Add order and limit
            query += " ORDER BY si.educational_significance DESC LIMIT ?"
            params.append(limit)

            # Execute query
            results = self.execute_query(query, tuple(params))

            # Enhance results with additional information
            enhanced_results = []
            for result in results:
                item_type = result.get("item_type")
                item_id = result.get("id")

                enhanced_result = {
                    "id": item_id,
                    "item_type": item_type,
                    "text": result.get("text"),
                    "language": result.get("language"),
                    "video_id": result.get("video_id"),
                    "video_title": result.get("video_title"),
                    "educational_significance": result.get("educational_significance")
                }

                # Add item-specific details
                if item_type == "concept":
                    # Get concept details
                    concept = self.get_repository_concept(item_id)
                    if concept:
                        enhanced_result["concept"] = concept

                elif item_type == "segment":
                    # Get segment details
                    segment_query = "SELECT * FROM segments WHERE segment_id = ?"
                    segment_results = self.execute_query(segment_query, (item_id,))
                    if segment_results:
                        enhanced_result["segment"] = segment_results[0]

                elif item_type == "occurrence":
                    # Get occurrence details
                    occurrence_query = """
                    SELECT o.*, rc.data as concept_data
                    FROM occurrences o
                    JOIN repository_concepts rc ON o.concept_id = rc.concept_id
                    WHERE o.occurrence_id = ?
                    """
                    occurrence_results = self.execute_query(occurrence_query, (item_id,))
                    if occurrence_results:
                        occurrence = occurrence_results[0]
                        try:
                            concept_data = json.loads(occurrence.get("concept_data", "{}"))
                        except json.JSONDecodeError:
                            concept_data = {}

                        enhanced_result["occurrence"] = occurrence
                        enhanced_result["concept"] = concept_data

                enhanced_results.append(enhanced_result)

            return {
                "results": enhanced_results,
                "total": len(enhanced_results)
            }

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {"results": [], "total": 0, "error": str(e)}

    def optimize_database(self) -> bool:
        """
        Optimize the database for better performance.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Run VACUUM
            with self.get_connection() as conn:
                conn.execute("VACUUM")

            # Run ANALYZE
            with self.get_connection() as conn:
                conn.execute("ANALYZE")

            # Optimize FTS index
            with self.get_connection() as conn:
                conn.execute("INSERT INTO search_index(search_index) VALUES('optimize')")

            logger.info("Database optimized")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False


# Singleton instance for global access
_data_access = None

def get_data_access(db_path: str = "data/index/indexer.db") -> DataAccess:
    """
    Get or create the DataAccess singleton instance.

    Args:
        db_path: Path to SQLite database

    Returns:
        DataAccess instance
    """
    global _data_access

    if _data_access is None:
        _data_access = DataAccess(db_path)

    return _data_access
