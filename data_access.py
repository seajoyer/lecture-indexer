"""
Enhanced data access layer for the Lecture Video Content Indexer.
Provides optimized database operations with improved security, performance,
and reliability.
"""

import os
import sqlite3
import logging
import time
import threading
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from contextlib import contextmanager

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

        # Initialize cache settings
        self.cache_ttl = 300  # Cache time-to-live in seconds
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize database schema
        self._ensure_schema()

        logger.info(f"DataAccess initialized with database at {db_path} (connection pooling enabled)")

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
        Ensure all necessary tables exist in the database with optimized schema.
        Consolidates schema creation and adds indexes for performance.
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
            domain TEXT,
            domain_confidence REAL,
            theory_practice_ratio REAL,
            theoretical_segments INTEGER,
            practical_segments INTEGER,
            indexed_at TEXT,
            processing_status TEXT,
            processing_errors TEXT
        );

        -- Create index on domain for filtering
        CREATE INDEX IF NOT EXISTS idx_videos_domain ON videos(domain);

        -- Create index on theory_practice_ratio for filtering
        CREATE INDEX IF NOT EXISTS idx_videos_theory_practice ON videos(theory_practice_ratio);

        -- Segments table for storing transcript segments
        CREATE TABLE IF NOT EXISTS segments (
            segment_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            start_time REAL,
            end_time REAL,
            text TEXT,
            context_type TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );

        -- Create index on video_id in segments for faster queries
        CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);

        -- Create index on context_type for filtering
        CREATE INDEX IF NOT EXISTS idx_segments_context_type ON segments(context_type);

        -- Create index on start_time for timeline ordering
        CREATE INDEX IF NOT EXISTS idx_segments_start_time ON segments(start_time);

        -- Concepts table for storing concept information
        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            domain TEXT,
            concept_class TEXT,
            total_occurrences INTEGER DEFAULT 0
        );

        -- Create indexes for concept filtering
        CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain);
        CREATE INDEX IF NOT EXISTS idx_concepts_class ON concepts(concept_class);
        CREATE INDEX IF NOT EXISTS idx_concepts_text ON concepts(text);

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
            FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE
        );

        -- Create indexes for occurrence queries
        CREATE INDEX IF NOT EXISTS idx_occurrences_concept_id ON occurrences(concept_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_video_id ON occurrences(video_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_context_type ON occurrences(context_type);

        -- Create FTS table for search with improved tokenization and ranking
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            id,
            text,
            domain,
            context_type,
            item_type,
            video_id,
            tokenize='porter unicode61 remove_diacritics 2'
        );

        -- Create playlists table
        CREATE TABLE IF NOT EXISTS playlists (
            playlist_id TEXT PRIMARY KEY,
            title TEXT,
            channel TEXT,
            video_ids TEXT,
            video_count INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        """

        try:
            with self.get_connection() as conn:
                conn.executescript(schema_script)
                conn.commit()
            logger.info("Database schema initialized with optimizations")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise

    def execute_query(self, query: str, params: Union[tuple, list] = ()) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as a list of dictionaries.
        Improved with better error handling and parameter normalization.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        start_time = time.time()

        # Normalize parameters to a tuple
        if isinstance(params, list):
            params = tuple(params)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                columns = [col[0] for col in cursor.description] if cursor.description else []
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]

                # Log slow queries for optimization
                query_time = time.time() - start_time
                if query_time > 0.5:  # Log queries taking more than 0.5 seconds
                    logger.warning(f"Slow query ({query_time:.3f}s): {query}")

                return results
        except sqlite3.Error as e:
            logger.error(f"Query error: {e}, Query: {query}, Params: {params}")
            raise

    def execute_update(self, query: str, params: Union[tuple, list] = ()) -> int:
        """
        Execute an update query and return the number of affected rows.
        Improved with better error handling and parameter normalization.

        Args:
            query: SQL update query
            params: Query parameters

        Returns:
            Number of affected rows
        """
        start_time = time.time()

        # Normalize parameters to a tuple
        if isinstance(params, list):
            params = tuple(params)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()

                # Log slow updates for optimization
                update_time = time.time() - start_time
                if update_time > 0.5:  # Log updates taking more than 0.5 seconds
                    logger.warning(f"Slow update ({update_time:.3f}s): {query}")

                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Update error: {e}, Query: {query}, Params: {params}")
            raise

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute multiple updates with different parameters.
        Improved with batch processing for better performance.

        Args:
            query: SQL query
            params_list: List of parameter tuples

        Returns:
            Number of affected rows
        """
        if not params_list:
            return 0

        start_time = time.time()

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

                # Log slow batch operations for optimization
                batch_time = time.time() - start_time
                if batch_time > 1.0:  # Log batch operations taking more than 1 second
                    logger.warning(f"Slow batch operation ({batch_time:.3f}s): {query} "
                                  f"with {len(params_list)} parameters")

                return total_affected
        except sqlite3.Error as e:
            logger.error(f"Batch update error: {e}, Query: {query}, "
                        f"Params count: {len(params_list)}")
            raise

    # CACHE MANAGEMENT

    def _get_cache_key(self, prefix: str, args: tuple) -> str:
        """
        Generate a cache key from a prefix and arguments.

        Args:
            prefix: Cache key prefix
            args: Arguments to include in the key

        Returns:
            Cache key string
        """
        # Convert arguments to strings
        arg_strs = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                arg_strs.append(str(arg))
            elif arg is None:
                arg_strs.append("None")
            else:
                arg_strs.append(str(hash(str(arg))))

        # Join with underscore
        return f"{prefix}_{'_'.join(arg_strs)}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get a value from cache with TTL check.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        now = time.time()

        if key in self.cache:
            timestamp = self.cache_timestamps.get(key, 0)

            # Check if expired
            if now - timestamp <= self.cache_ttl:
                self.cache_hits += 1
                return self.cache[key]
            else:
                # Expired entry
                del self.cache[key]
                del self.cache_timestamps[key]

        self.cache_misses += 1
        return None

    def _set_in_cache(self, key: str, value: Any) -> None:
        """
        Store a value in cache with timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

        # Check cache size and clear old entries if too large
        if len(self.cache) > 1000:  # Limit cache size
            self._clean_cache()

    def _clean_cache(self) -> None:
        """Clean expired cache entries."""
        now = time.time()
        expired_keys = [k for k, ts in self.cache_timestamps.items()
                        if now - ts > self.cache_ttl]

        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]

        logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries. "
                     f"Cache size: {len(self.cache)}")

    def clear_cache(self, prefix: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            prefix: Optional prefix to clear selective cache entries
        """
        if prefix:
            # Clear specific cache entries
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                if k in self.cache:
                    del self.cache[k]
                if k in self.cache_timestamps:
                    del self.cache_timestamps[k]

            logger.info(f"Cleared {len(keys_to_remove)} cache entries with prefix '{prefix}'")
        else:
            # Clear all cache
            self.cache.clear()
            self.cache_timestamps.clear()
            self.cache_hits = 0
            self.cache_misses = 0

            logger.info("Cleared entire cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.cache_ttl
        }

    # VIDEO OPERATIONS

    def save_video(self, video_data: Dict[str, Any]) -> bool:
        """
        Save or update video metadata with improved validation.

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
                    domain = ?,
                    domain_confidence = ?,
                    theory_practice_ratio = ?,
                    theoretical_segments = ?,
                    practical_segments = ?,
                    indexed_at = ?,
                    processing_status = ?,
                    processing_errors = ?
                WHERE video_id = ?
                """
                self.execute_update(query, (
                    video_data.get("title", ""),
                    video_data.get("description", ""),
                    video_data.get("channel", ""),
                    video_data.get("publication_date", ""),
                    video_data.get("duration_seconds", 0),
                    video_data.get("language", ""),
                    video_data.get("domain", "unknown"),
                    video_data.get("domain_confidence", 0.0),
                    video_data.get("theory_practice_ratio", 0.5),
                    video_data.get("theoretical_segments", 0),
                    video_data.get("practical_segments", 0),
                    video_data.get("indexed_at", ""),
                    video_data.get("processing_status", "completed"),
                    video_data.get("processing_errors"),
                    video_id
                ))
            else:
                # Insert new video
                query = """
                INSERT INTO videos (
                    video_id, title, description, channel, publication_date,
                    duration_seconds, language, domain, domain_confidence,
                    theory_practice_ratio, theoretical_segments, practical_segments,
                    indexed_at, processing_status, processing_errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.execute_update(query, (
                    video_id,
                    video_data.get("title", ""),
                    video_data.get("description", ""),
                    video_data.get("channel", ""),
                    video_data.get("publication_date", ""),
                    video_data.get("duration_seconds", 0),
                    video_data.get("language", ""),
                    video_data.get("domain", "unknown"),
                    video_data.get("domain_confidence", 0.0),
                    video_data.get("theory_practice_ratio", 0.5),
                    video_data.get("theoretical_segments", 0),
                    video_data.get("practical_segments", 0),
                    video_data.get("indexed_at", ""),
                    video_data.get("processing_status", "completed"),
                    video_data.get("processing_errors")
                ))

            # Clear cache for this video
            self.clear_cache(f"video_{video_id}")

            logger.info(f"Saved video metadata for {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving video {video_id}: {e}")
            return False

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata by ID with caching.

        Args:
            video_id: Video ID

        Returns:
            Video metadata dictionary or None if not found
        """
        # Check cache first
        cache_key = self._get_cache_key("video", (video_id,))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        query = "SELECT * FROM videos WHERE video_id = ?"
        results = self.execute_query(query, (video_id,))

        if not results:
            return None

        # Cache the result
        self._set_in_cache(cache_key, results[0])

        return results[0]

    def save_segments(self, video_id: str, segments: List[Dict[str, Any]]) -> bool:
        """
        Save video transcript segments with optimized batch processing.

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
                segment_id, video_id, start_time, end_time, text, context_type
            ) VALUES (?, ?, ?, ?, ?, ?)
            """

            # Transform segments into parameter tuples
            params_list = []
            search_index_params = []

            for segment in segments:
                segment_id = segment.get("id")
                if not segment_id:
                    continue

                params_list.append((
                    segment_id,
                    video_id,
                    segment.get("start_time", 0.0),
                    segment.get("end_time", 0.0),
                    segment.get("text", ""),
                    segment.get("content_type", "mixed")
                ))

                # Prepare search index parameters
                search_index_params.append((
                    segment_id,
                    segment.get("text", ""),
                    segment.get("domain", "unknown"),
                    segment.get("content_type", "mixed"),
                    "segment",
                    video_id
                ))

            # Execute batch insert for segments
            self.execute_many(query, params_list)

            # Insert into search index
            if search_index_params:
                search_query = """
                INSERT INTO search_index (
                    id, text, domain, context_type, item_type, video_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """
                self.execute_many(search_query, search_index_params)

            # Clear cache for this video's segments
            self.clear_cache(f"segments_{video_id}")

            logger.info(f"Saved {len(segments)} segments for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving segments for video {video_id}: {e}")
            return False

    def get_video_segments(
        self,
        video_id: str,
        context_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get segments for a video with optional filtering.

        Args:
            video_id: Video ID
            context_type: Optional filter for context type
            start_time: Optional filter for minimum start time
            end_time: Optional filter for maximum end time

        Returns:
            List of segment dictionaries
        """
        # Build cache key based on all parameters
        cache_key = self._get_cache_key("segments", (video_id, context_type, start_time, end_time))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Build query parameters
        query = "SELECT * FROM segments WHERE video_id = ?"
        params = [video_id]

        if context_type:
            query += " AND context_type = ?"
            params.append(context_type)

        if start_time is not None:
            query += " AND end_time >= ?"
            params.append(start_time)

        if end_time is not None:
            query += " AND start_time <= ?"
            params.append(end_time)

        query += " ORDER BY start_time"

        results = self.execute_query(query, tuple(params))

        # Cache the result
        self._set_in_cache(cache_key, results)

        return results

    # CONCEPT OPERATIONS

    def save_concept(self, concept_data: Dict[str, Any]) -> Optional[str]:
        """
        Save or update a concept with improved validation.

        Args:
            concept_data: Concept data dictionary

        Returns:
            Concept ID if successful, None otherwise
        """
        import hashlib

        # Extract concept information
        concept_text = concept_data.get("text", "")
        if not concept_text:
            logger.error("Cannot save concept without text")
            return None

        domain = concept_data.get("domain", "unknown")

        # Generate concept ID if not provided
        concept_id = concept_data.get("concept_id")
        if not concept_id:
            # Create deterministic ID based on text and domain
            text_for_hash = concept_text.lower().strip()
            concept_id = hashlib.md5(f"{text_for_hash}:{domain}".encode()).hexdigest()

        # Determine concept class
        concept_class = concept_data.get("concept_class", "")
        if not concept_class:
            theoretical = concept_data.get("theoretical", False)
            concept_class = "theoretical" if theoretical else "practical"

        try:
            # Check if concept exists
            existing = self.get_concept(concept_id)

            if existing:
                # Update existing concept
                query = """
                UPDATE concepts SET
                    text = ?,
                    domain = ?,
                    concept_class = ?,
                    total_occurrences = ?
                WHERE concept_id = ?
                """
                self.execute_update(query, (
                    concept_text,
                    domain,
                    concept_class,
                    concept_data.get("total_occurrences", 0),
                    concept_id
                ))
            else:
                # Insert new concept
                query = """
                INSERT INTO concepts (
                    concept_id, text, domain, concept_class, total_occurrences
                ) VALUES (?, ?, ?, ?, ?)
                """
                self.execute_update(query, (
                    concept_id,
                    concept_text,
                    domain,
                    concept_class,
                    concept_data.get("total_occurrences", 0)
                ))

            # Index for search - delete and reinsert to ensure freshness
            self.execute_update(
                "DELETE FROM search_index WHERE id = ? AND item_type = 'concept'",
                (concept_id,)
            )

            self.execute_update(
                """
                INSERT INTO search_index (id, text, domain, context_type, item_type, video_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (concept_id, concept_text, domain, concept_class, "concept", None)
            )

            # Save occurrences if provided
            video_id = concept_data.get("video_id")
            if video_id:
                # Find segments containing this concept
                segments = self.get_video_segments(video_id)

                occurrences = []
                for segment in segments:
                    segment_text = segment["text"].lower()
                    concept_text_lower = concept_text.lower()

                    # Check if concept appears in segment
                    # Split the concept text to handle multi-word concepts
                    concept_parts = concept_text_lower.split()
                    if len(concept_parts) == 1:
                        # For single-word concepts, ensure it's a complete word
                        if re.search(r'\b' + re.escape(concept_text_lower) + r'\b', segment_text):
                            occurrence_id = hashlib.md5(
                                f"{concept_id}:{segment['segment_id']}".encode()
                            ).hexdigest()

                            occurrences.append({
                                "occurrence_id": occurrence_id,
                                "concept_id": concept_id,
                                "video_id": video_id,
                                "segment_id": segment["segment_id"],
                                "start_time": segment["start_time"],
                                "end_time": segment["end_time"],
                                "context_type": segment["context_type"],
                                "context_text": segment["text"]
                            })
                    else:
                        # For multi-word concepts, check if all parts appear in the right order
                        if all(part in segment_text for part in concept_parts):
                            # More thorough check for sequence
                            if " ".join(concept_parts) in segment_text:
                                occurrence_id = hashlib.md5(
                                    f"{concept_id}:{segment['segment_id']}".encode()
                                ).hexdigest()

                                occurrences.append({
                                    "occurrence_id": occurrence_id,
                                    "concept_id": concept_id,
                                    "video_id": video_id,
                                    "segment_id": segment["segment_id"],
                                    "start_time": segment["start_time"],
                                    "end_time": segment["end_time"],
                                    "context_type": segment["context_type"],
                                    "context_text": segment["text"]
                                })

                if occurrences:
                    self.save_occurrences(concept_id, occurrences)

            # Clear cache for this concept
            self.clear_cache(f"concept_{concept_id}")

            logger.info(f"Saved concept {concept_id}: {concept_text}")
            return concept_id

        except Exception as e:
            logger.error(f"Error saving concept {concept_text}: {e}")
            return None

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by ID with caching.

        Args:
            concept_id: Concept ID

        Returns:
            Concept dictionary or None if not found
        """
        # Check cache first
        cache_key = self._get_cache_key("concept", (concept_id,))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        query = "SELECT * FROM concepts WHERE concept_id = ?"
        results = self.execute_query(query, (concept_id,))

        if not results:
            return None

        # Cache the result
        self._set_in_cache(cache_key, results[0])

        return results[0]

    def save_occurrences(self, concept_id: str, occurrences: List[Dict[str, Any]]) -> bool:
        """
        Save concept occurrences with improved batching.

        Args:
            concept_id: Concept ID
            occurrences: List of occurrence dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not concept_id or not occurrences:
            return False

        try:
            # Check for existing occurrences to avoid duplicates
            existing_query = """
            SELECT occurrence_id FROM occurrences WHERE concept_id = ?
            """
            existing = self.execute_query(existing_query, (concept_id,))
            existing_ids = {row["occurrence_id"] for row in existing}

            # Filter out existing occurrences
            new_occurrences = [
                occ for occ in occurrences
                if occ.get("occurrence_id") not in existing_ids
            ]

            if not new_occurrences:
                logger.debug(f"No new occurrences to save for concept {concept_id}")
                return True

            # Prepare batch insert
            query = """
            INSERT INTO occurrences (
                occurrence_id, concept_id, video_id, segment_id,
                start_time, end_time, context_type, context_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            params_list = []
            for occurrence in new_occurrences:
                video_id = occurrence.get("video_id")
                segment_id = occurrence.get("segment_id")
                occurrence_id = occurrence.get("occurrence_id")

                if not video_id or not segment_id or not occurrence_id:
                    continue

                params_list.append((
                    occurrence_id,
                    concept_id,
                    video_id,
                    segment_id,
                    occurrence.get("start_time", 0.0),
                    occurrence.get("end_time", 0.0),
                    occurrence.get("context_type", "mixed"),
                    occurrence.get("context_text", "")
                ))

            if params_list:
                self.execute_many(query, params_list)

                # Update concept occurrence count
                self.execute_update(
                    "UPDATE concepts SET total_occurrences = total_occurrences + ? WHERE concept_id = ?",
                    (len(params_list), concept_id)
                )

                # Clear cache for related items
                self.clear_cache(f"concept_{concept_id}")
                for occurrence in new_occurrences:
                    video_id = occurrence.get("video_id")
                    if video_id:
                        self.clear_cache(f"video_concepts_{video_id}")

            logger.info(f"Saved {len(params_list)} occurrences for concept {concept_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving occurrences for concept {concept_id}: {e}")
            return False

    def get_concepts_for_video(
        self,
        video_id: str,
        context_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get concepts extracted from a video with caching.

        Args:
            video_id: Video ID
            context_type: Optional context type filter

        Returns:
            List of concept dictionaries with occurrence information
        """
        # Check cache first
        cache_key = self._get_cache_key("video_concepts", (video_id, context_type))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        query = """
        SELECT c.*, COUNT(o.occurrence_id) as occurrence_count,
               MAX(o.start_time) as last_occurrence_time
        FROM concepts c
        JOIN occurrences o ON c.concept_id = o.concept_id
        WHERE o.video_id = ?
        """
        params = [video_id]

        if context_type:
            query += " AND o.context_type = ?"
            params.append(context_type)

        query += " GROUP BY c.concept_id ORDER BY occurrence_count DESC, last_occurrence_time"

        results = self.execute_query(query, tuple(params))

        # Cache the result
        self._set_in_cache(cache_key, results)

        return results

    def get_video_concepts(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get all concept and pattern information for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with video concepts and patterns
        """
        # Check cache
        cache_key = self._get_cache_key("video_concept_data", (video_id,))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        video = self.get_video(video_id)
        if not video:
            return None

        concepts = self.get_concepts_for_video(video_id)

        result = {
            "video": video,
            "concepts": concepts,
            "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
        }

        # Cache the result
        self._set_in_cache(cache_key, result)

        return result

    # SEARCH OPERATIONS

    def build_enhanced_search_query(self, query_text: str, domain: Optional[str] = None) -> str:
        """
        Build an enhanced search query with improved handling of different languages
        and special characters.

        Args:
            query_text: Original query text
            domain: Optional domain for domain-specific handling

        Returns:
            Enhanced search query for SQLite FTS5
        """
        # Basic cleaning
        query_text = query_text.strip()
        if not query_text:
            return ""

        # For non-Latin queries, use a simpler approach to avoid syntax errors
        if any(ord(c) > 127 for c in query_text):
            # For non-Latin text (e.g., Russian), use a safe approach
            # Split by whitespace and quote each term
            terms = query_text.split()
            if not terms:
                return ""

            # For each term, create both an exact match and a prefix match
            query_parts = []
            for term in terms:
                # Clean the term of any special characters
                term = ''.join(c for c in term if c.isalnum() or ord(c) > 127)
                if term:
                    # Add both exact and prefix match options
                    query_parts.append(f'"{term}" OR {term}*')

            # Join with OR for better recall
            if len(query_parts) == 1:
                return query_parts[0]
            else:
                # Combine with both AND and OR for balanced precision/recall
                and_query = " AND ".join(f"({part})" for part in query_parts)
                or_query = " OR ".join(f"({part})" for part in query_parts)
                return f"({and_query}) OR ({or_query})"

        # For Latin-based queries, use the more sophisticated approach
        # Extract quoted phrases for exact matching
        quoted_phrases = re.findall(r'"([^"]+)"', query_text)

        # Remove quoted phrases from query for separate processing
        clean_query = query_text
        for phrase in quoted_phrases:
            clean_query = clean_query.replace(f'"{phrase}"', '')

        # Clean and tokenize remaining text
        # Replace special chars that might interfere with FTS query syntax
        clean_query = re.sub(r'[^\w\s]', ' ', clean_query)
        tokens = [token.strip() for token in clean_query.split() if token.strip()]

        query_parts = []

        # Add exact phrases
        for phrase in quoted_phrases:
            # Escape any special chars in the phrase
            safe_phrase = re.sub(r'[^\w\s]', ' ', phrase).strip()
            if safe_phrase:
                query_parts.append(f'"{safe_phrase}"^2')  # Give higher weight to exact matches

        # Add token-based matching with prefix
        if tokens:
            token_parts = []
            for token in tokens:
                # For very short tokens (2 chars or less), don't use prefix matching
                if len(token) <= 2:
                    token_parts.append(token)
                else:
                    token_parts.append(f"{token}*")  # Prefix matching for longer tokens

            # For single tokens, just use the token with prefix
            if len(token_parts) == 1:
                query_parts.append(token_parts[0])
            else:
                # For multiple tokens, try both exact phrase and individual token matching
                exact_tokens = " ".join(tokens)
                token_match = " ".join(token_parts)  # Implicit AND in SQLite FTS5

                # Add both with exact phrase having higher weight
                query_parts.append(f'"{exact_tokens}"^1.5 OR {token_match}')

        # Combine all parts with OR if multiple parts, otherwise return as is
        if len(query_parts) > 1:
            return " OR ".join(query_parts)
        elif query_parts:
            return query_parts[0]
        else:
            return ""

    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced search using SQLite FTS with improvements.

        Args:
            query: Query parameters dictionary

        Returns:
            Search results dictionary
        """
        start_time = time.time()

        try:
            # Extract query parameters
            query_text = query.get("original_text", "").strip()
            filters = query.get("filters", {})
            theory_practice_ratio = query.get("theory_practice_ratio")
            domain = query.get("domain")
            pagination = query.get("pagination", {})

            offset = pagination.get("offset", 0)
            limit = pagination.get("limit", 10)

            if not query_text:
                return {
                    "results": [],
                    "totalResults": 0,
                    "executionTimeMs": 0
                }

            # Generate cache key
            cache_key = f"search_{query_text}_{domain}_{theory_practice_ratio}_{offset}_{limit}"
            if filters:
                cache_key += f"_filters:{hash(str(filters))}"

            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached search results for: {query_text}")
                return cached_result

            # Build enhanced search query for non-exact matching
            search_query = self.build_enhanced_search_query(query_text, domain)

            # Build SQL query with improved JOINs for better performance
            sql = """
            WITH matched_items AS (
                SELECT si.id, si.text, si.domain, si.context_type, si.item_type, si.video_id,
                    rank
                FROM search_index si
                WHERE search_index MATCH ?
            )

            SELECT mi.id, mi.text, mi.domain, mi.context_type, mi.item_type, mi.video_id,
                v.title as video_title,
                CASE WHEN mi.item_type = 'segment' THEN s.start_time ELSE NULL END as start_time,
                CASE WHEN mi.item_type = 'segment' THEN s.end_time ELSE NULL END as end_time,
                CASE WHEN mi.item_type = 'segment' THEN s.text ELSE mi.text END as context_text
            FROM matched_items mi
            LEFT JOIN videos v ON mi.video_id = v.video_id
            LEFT JOIN segments s ON mi.item_type = 'segment' AND mi.id = s.segment_id
            LEFT JOIN concepts c ON mi.item_type = 'concept' AND mi.id = c.concept_id
            """

            params = [search_query]

            # Apply filters
            where_clauses = []

            if domain:
                where_clauses.append("mi.domain = ?")
                params.append(domain)

            if "video_id" in filters:
                where_clauses.append("mi.video_id = ?")
                params.append(filters["video_id"])

            if "video_ids" in filters and filters["video_ids"]:
                placeholders = ", ".join(["?"] * len(filters["video_ids"]))
                where_clauses.append(f"mi.video_id IN ({placeholders})")
                params.extend(filters["video_ids"])

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            # Apply theory/practice ratio filter
            context_type_order = ""
            if theory_practice_ratio is not None:
                if theory_practice_ratio > 0.7:
                    context_type_order = " ORDER BY CASE WHEN mi.context_type = 'theoretical' THEN 1 ELSE 2 END, rank"
                elif theory_practice_ratio < 0.3:
                    context_type_order = " ORDER BY CASE WHEN mi.context_type = 'practical' THEN 1 ELSE 2 END, rank"

            # Apply ordering - use FTS5 rank for relevance
            if context_type_order:
                sql += context_type_order
            else:
                sql += " ORDER BY rank"

            # Apply pagination
            sql += f" LIMIT {limit} OFFSET {offset}"

            # Execute search query
            results = self.execute_query(sql, tuple(params))

            # Count total results more efficiently
            count_sql = """
            SELECT COUNT(*) as count FROM search_index
            WHERE search_index MATCH ?
            """
            count_params = [search_query]

            if domain:
                count_sql += " AND domain = ?"
                count_params.append(domain)

            if "video_id" in filters:
                count_sql += " AND video_id = ?"
                count_params.append(filters["video_id"])

            if "video_ids" in filters and filters["video_ids"]:
                placeholders = ", ".join(["?"] * len(filters["video_ids"]))
                count_sql += f" AND video_id IN ({placeholders})"
                count_params.extend(filters["video_ids"])

            count_result = self.execute_query(count_sql, tuple(count_params))
            total_count = count_result[0]["count"] if count_result else 0

            # Count theoretical and practical results
            theoretical_count = 0
            practical_count = 0

            for r in results:
                if r["context_type"] == "theoretical":
                    theoretical_count += 1
                elif r["context_type"] == "practical":
                    practical_count += 1

            # Format the results
            formatted_results = []
            for r in results:
                result = {
                    "result_type": r["item_type"],
                    "text": r["text"],
                    "domain": r["domain"],
                    "context_type": r["context_type"],
                    "video_id": r["video_id"],
                    "video_title": r["video_title"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "context_text": r["context_text"]
                }

                # Get additional data for concepts
                if r["item_type"] == "concept":
                    concept = self.get_concept(r["id"])
                    if concept:
                        result["concept_id"] = r["id"]
                        result["concept_class"] = concept["concept_class"]

                formatted_results.append(result)

            execution_time_ms = int((time.time() - start_time) * 1000)

            search_result = {
                "results": formatted_results,
                "totalResults": total_count,
                "theoreticalResults": theoretical_count,
                "practicalResults": practical_count,
                "executionTimeMs": execution_time_ms
            }

            # Cache the result
            self._set_in_cache(cache_key, search_result)

            return search_result

        except Exception as e:
            logger.error(f"Error executing search: {e}")
            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "totalResults": 0,
                "executionTimeMs": execution_time_ms,
                "error": str(e)
            }

    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content for search with improved error handling and batching.

        Args:
            processed_result: Processing result dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract key fields
            video_id = processed_result.get("video_id")
            if not video_id:
                logger.error("Missing video_id for indexing")
                return False

            metadata = processed_result.get("metadata", {})
            transcript = processed_result.get("transcript", {})
            domain_features = processed_result.get("domain_features", {})

            # Clear existing content for this video to avoid duplicates
            self.execute_update("DELETE FROM segments WHERE video_id = ?", (video_id,))
            self.execute_update("DELETE FROM occurrences WHERE video_id = ?", (video_id,))
            self.execute_update(
                "DELETE FROM search_index WHERE item_type = 'segment' AND video_id = ?",
                (video_id,)
            )

            # Save video metadata
            video_data = {
                "video_id": video_id,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "channel": metadata.get("channel", ""),
                "publication_date": metadata.get("publication_date", ""),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "language": metadata.get("language", ""),
                "domain": metadata.get("domain", "unknown"),
                "domain_confidence": metadata.get("domain_confidence", 0.0),
                "theory_practice_ratio": processed_result.get("theory_practice_results", {}).get("theory_practice_ratio", 0.5),
                "theoretical_segments": processed_result.get("theory_practice_results", {}).get("theoretical_segments", 0),
                "practical_segments": processed_result.get("theory_practice_results", {}).get("practical_segments", 0),
                "processing_status": "completed",
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            video_saved = self.save_video(video_data)
            if not video_saved:
                logger.error(f"Failed to save video metadata for {video_id}")
                return False

            # Save segments
            segments = transcript.get("segments", [])
            segments_saved = self.save_segments(video_id, segments)
            if not segments_saved:
                logger.error(f"Failed to save segments for {video_id}")
                return False

            # Save concepts in batches
            key_concepts = domain_features.get("key_concepts", [])

            # Process concepts in batches of 20 for better performance
            batch_size = 20
            successful_concepts = 0

            for i in range(0, len(key_concepts), batch_size):
                batch = key_concepts[i:i + batch_size]
                for concept in batch:
                    concept_data = concept.copy()
                    concept_data["video_id"] = video_id
                    concept_id = self.save_concept(concept_data)
                    if concept_id:
                        successful_concepts += 1

            # Log success and clear related caches
            logger.info(f"Successfully indexed {successful_concepts}/{len(key_concepts)} concepts for video {video_id}")

            # Clear caches related to this video
            self.clear_cache(f"video_{video_id}")
            self.clear_cache(f"segments_{video_id}")
            self.clear_cache(f"video_concepts_{video_id}")
            self.clear_cache(f"video_concept_data_{video_id}")

            # Optimize database if appropriate
            self._maybe_optimize_database()

            return True

        except Exception as e:
            logger.error(f"Error indexing content: {e}")

            # Log detailed exception but only in debug mode
            import traceback
            logger.debug(f"Indexing error details: {traceback.format_exc()}")

            return False

    def _maybe_optimize_database(self) -> None:
        """Occasionally optimize database for better performance."""
        # Only optimize occasionally based on random chance (approx. 5%)
        import random
        if random.random() < 0.05:
            try:
                logger.info("Running periodic database optimization")
                self.execute_update("PRAGMA optimize")
                self.execute_update("INSERT INTO search_index(search_index) VALUES('optimize')")
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        stats = {}

        try:
            # Get table counts
            table_stats = {}
            for table in ["videos", "segments", "concepts", "occurrences"]:
                count = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                table_stats[table] = count[0]["count"] if count else 0

            stats["tables"] = table_stats

            # Get domain distribution
            domains = self.execute_query(
                "SELECT domain, COUNT(*) as count FROM videos GROUP BY domain ORDER BY count DESC"
            )
            stats["domains"] = domains

            # Get theory/practice distribution
            theory_practice = self.execute_query("""
                SELECT
                    SUM(CASE WHEN concept_class = 'theoretical' THEN 1 ELSE 0 END) as theoretical,
                    SUM(CASE WHEN concept_class = 'practical' THEN 1 ELSE 0 END) as practical
                FROM concepts
            """)
            stats["concepts"] = theory_practice[0] if theory_practice else {}

            # Get cache stats
            stats["cache"] = self.get_cache_stats()

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

# Global instance for singleton access
_data_access = None

def get_data_access(db_path: str = "data/index/indexer.db") -> DataAccess:
    """
    Get or create the DataAccess instance.

    Args:
        db_path: Database path

    Returns:
        DataAccess instance
    """
    global _data_access

    if _data_access is None:
        _data_access = DataAccess(db_path)

    return _data_access
