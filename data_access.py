"""
Enhanced data access layer for the Lecture Video Content Indexer.
Provides optimized database operations with improved security, performance,
and reliability. Theory/practice categorization removed for improved design.
"""

import os
import sqlite3
import logging
import time
import threading
import re
import difflib  # For string similarity comparison
from typing import Dict, List, Any, Optional, Tuple, Set, Union
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
        Ensure all necessary tables exist in the database with optimized schema.
        Theory/practice categories removed for simplicity and clarity.
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
            indexed_at TEXT,
            processing_status TEXT,
            processing_errors TEXT
        );

        -- Create index on domain for filtering
        CREATE INDEX IF NOT EXISTS idx_videos_domain ON videos(domain);

        -- Create index on theory_practice_ratio for filtering
        CREATE INDEX IF NOT EXISTS idx_videos_theory_practice ON videos(theory_practice_ratio);

        -- Create index on language for multilingual filtering
        CREATE INDEX IF NOT EXISTS idx_videos_language ON videos(language);

        -- Segments table for storing transcript segments
        CREATE TABLE IF NOT EXISTS segments (
            segment_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            start_time REAL,
            end_time REAL,
            text TEXT,
            language TEXT,
            educational_value REAL,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );

        -- Create index on video_id in segments for faster queries
        CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);

        -- Create index on start_time for timeline ordering
        CREATE INDEX IF NOT EXISTS idx_segments_start_time ON segments(start_time);

        -- Add language index for multilingual search
        CREATE INDEX IF NOT EXISTS idx_segments_language ON segments(language);

        -- Add index for educational value
        CREATE INDEX IF NOT EXISTS idx_segments_educational_value ON segments(educational_value);

        -- Concepts table for storing concept information
        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            normalized_text TEXT,
            domain TEXT,
            language TEXT,
            total_occurrences INTEGER DEFAULT 0,
            canonical_concept_id TEXT, -- Reference to canonical concept if this is a variant
            educational_weight REAL DEFAULT 0, -- Measure of educational significance
            is_educational INTEGER DEFAULT 0   -- Flag for educational vs passing mention
        );

        -- Create indexes for concept filtering
        CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain);
        CREATE INDEX IF NOT EXISTS idx_concepts_text ON concepts(text);
        CREATE INDEX IF NOT EXISTS idx_concepts_language ON concepts(language);
        CREATE INDEX IF NOT EXISTS idx_concepts_normalized_text ON concepts(normalized_text);
        CREATE INDEX IF NOT EXISTS idx_concepts_educational ON concepts(is_educational);
        CREATE INDEX IF NOT EXISTS idx_concepts_educational_weight ON concepts(educational_weight);

        -- Occurrences table for concept-segment associations
        CREATE TABLE IF NOT EXISTS occurrences (
            occurrence_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            start_time REAL,
            end_time REAL,
            context_text TEXT,
            FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE
        );

        -- Create indexes for occurrence queries
        CREATE INDEX IF NOT EXISTS idx_occurrences_concept_id ON occurrences(concept_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_video_id ON occurrences(video_id);

        -- Add a composite index for efficient related concept lookup
        CREATE INDEX IF NOT EXISTS idx_occurrences_segment_concept ON occurrences(segment_id, concept_id);
        CREATE INDEX IF NOT EXISTS idx_occurrences_video_segment ON occurrences(video_id, segment_id);

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

        -- Create language_resources table for storing language-specific resources
        CREATE TABLE IF NOT EXISTS language_resources (
            language_code TEXT PRIMARY KEY,
            stopwords TEXT,  -- JSON array of stopwords
            educational_patterns TEXT,  -- JSON array of educational content patterns
            updated_at TEXT
        );
        """

        try:
            with self.get_connection() as conn:
                conn.executescript(schema_script)

                # Check if old columns exist in segments
                segments_columns = conn.execute("PRAGMA table_info(segments)").fetchall()
                segment_column_names = [col[1] for col in segments_columns]

                if "context_type" in segment_column_names:
                    # Add educational_value column if needed
                    if "educational_value" not in segment_column_names:
                        conn.execute("ALTER TABLE segments ADD COLUMN educational_value REAL DEFAULT 0")

                    # Migrate context_type data to avoid data loss during transition
                    conn.execute("""
                    UPDATE segments
                    SET educational_value = CASE
                        WHEN context_type = 'theoretical' THEN 2.0
                        WHEN context_type = 'practical' THEN 1.0
                        ELSE 0.0
                    END
                    WHERE educational_value = 0
                    """)

                    logger.info("Migrated segment context_type data to educational_value")

                # Check if old columns exist in concepts
                concepts_columns = conn.execute("PRAGMA table_info(concepts)").fetchall()
                concept_column_names = [col[1] for col in concepts_columns]

                if "concept_class" in concept_column_names and "educational_weight" in concept_column_names:
                    # Migrate concept_class data to educational_weight
                    conn.execute("""
                    UPDATE concepts
                    SET educational_weight = CASE
                        WHEN concept_class = 'theoretical' THEN educational_weight + 1.0
                        ELSE educational_weight
                    END
                    """)

                    logger.info("Migrated concept_class data to educational_weight")

                # RECREATE THE SEARCH INDEX TABLE - This is a critical fix
                try:
                    # First, check if search_index exists and drop it
                    conn.execute("DROP TABLE IF EXISTS search_index")
                    logger.info("Dropped existing search_index table")

                    # Create the search index table with proper tokenization options
                    conn.execute("""
                    CREATE VIRTUAL TABLE search_index USING fts5(
                        id,
                        text,
                        domain,
                        item_type,
                        video_id,
                        language,
                        educational_weight,
                        tokenize='unicode61 remove_diacritics 1'
                    )
                    """)
                    logger.info("Created new search_index table with improved tokenization")
                except Exception as e:
                    logger.error(f"Error recreating search_index: {e}")
                    # Try with a simpler configuration if the first attempt fails
                    try:
                        conn.execute("DROP TABLE IF EXISTS search_index")
                        conn.execute("""
                        CREATE VIRTUAL TABLE search_index USING fts5(
                            id,
                            text,
                            domain,
                            item_type,
                            video_id,
                            language,
                            educational_weight
                        )
                        """)
                        logger.info("Created new search_index table with basic configuration")
                    except Exception as e2:
                        logger.error(f"Error creating basic search_index: {e2}")
                        raise

                conn.commit()

            logger.info("Database schema initialized with optimizations and educational metrics")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise

    def list_concepts(self, domain_filter=None, video_filter=None, playlist_filter=None, language=None):
        """
        List all concepts with improved query performance

        Args:
            domain_filter: Optional domain filter
            video_filter: Optional video ID filter
            playlist_filter: Optional playlist ID filter
            language: Optional language filter

        Returns:
            List of concepts with stats
        """
        try:
            # First, check if any concepts exist at all
            basic_count_query = "SELECT COUNT(*) as count FROM concepts"
            count_result = self.execute_query(basic_count_query)

            if count_result and count_result[0]["count"] == 0:
                logger.info("No concepts found in database")
                return []

            # Get video IDs from playlist if specified
            video_ids = []
            if playlist_filter:
                video_ids = self._get_playlist_video_ids(playlist_filter)
                if not video_ids:
                    logger.warning(f"No videos found for playlist {playlist_filter}")
                    return []

            # First get concepts matching filters without requiring occurrences
            query = """
            SELECT c.*,
                COUNT(DISTINCT o.video_id) as video_count,
                COUNT(DISTINCT o.occurrence_id) as occurrence_count
            FROM concepts c
            LEFT JOIN occurrences o ON c.concept_id = o.concept_id
            """

            if video_filter:
                query += " LEFT JOIN videos v ON o.video_id = v.video_id"

            query += " WHERE 1=1"  # Start WHERE clause

            # Add WHERE clause if we have filters
            params = []

            if domain_filter:
                query += " AND c.domain = ?"
                params.append(domain_filter)

            if video_filter:
                query += " AND o.video_id = ?"
                params.append(video_filter)

            if language:
                query += " AND (c.language = ? OR c.language IS NULL)"
                params.append(language)

            if video_ids:
                placeholders = ",".join(['?'] * len(video_ids))
                query += f" AND o.video_id IN ({placeholders})"
                params.extend(video_ids)

            # Primary canonical concept filter - only include canonical concepts
            query += " AND (c.canonical_concept_id IS NULL OR c.canonical_concept_id = '')"

            # Group and order with enhanced sorting
            query += """
            GROUP BY c.concept_id
            ORDER BY
                c.domain,
                educational_weight DESC,
                occurrence_count DESC
            """

            # Execute the query with a custom timeout to avoid long-running queries
            concepts = self._execute_query_with_timeout(query, tuple(params), 10)  # 10 second timeout

            if not concepts:
                # If no results with the filters, try a simple query to see if we have any concepts
                basic_query = "SELECT * FROM concepts LIMIT 10"
                basic_results = self.execute_query(basic_query)

                if basic_results:
                    logger.info(f"Found {len(basic_results)} concepts in database, but none match the filters")
                else:
                    logger.info("No concepts found in database")

            return concepts

        except Exception as e:
            logger.error(f"Error listing concepts: {e}")
            return []

    def get_language_resources(self, language_code):
        """
        Get language-specific resources from the database.

        Args:
            language_code: Language code (e.g., 'en', 'ru')

        Returns:
            Dictionary with language resources
        """
        import json

        query = "SELECT * FROM language_resources WHERE language_code = ?"
        results = self.execute_query(query, (language_code,))

        if not results:
            return None

        # Parse JSON fields
        resources = dict(results[0])

        # Convert JSON strings to Python objects
        for field in ['stopwords', 'educational_patterns']:
            if resources.get(field):
                try:
                    resources[field] = json.loads(resources[field])
                except:
                    resources[field] = []

        return resources

    def save_language_resources(self, language_code, resources):
        """
        Save language-specific resources to the database.

        Args:
            language_code: Language code
            resources: Dictionary with language resources

        Returns:
            True if successful, False otherwise
        """
        import json

        try:
            # Convert Python objects to JSON strings
            data = {
                'language_code': language_code,
                'stopwords': json.dumps(resources.get('stopwords', [])),
                'educational_patterns': json.dumps(resources.get('educational_patterns', [])),
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # Check if language exists
            existing = self.get_language_resources(language_code)

            if existing:
                # Update existing
                query = """
                UPDATE language_resources SET
                    stopwords = ?,
                    educational_patterns = ?,
                    updated_at = ?
                WHERE language_code = ?
                """
                self.execute_update(query, (
                    data['stopwords'],
                    data['educational_patterns'],
                    data['updated_at'],
                    language_code
                ))
            else:
                # Insert new
                query = """
                INSERT INTO language_resources (
                    language_code, stopwords, educational_patterns, updated_at
                ) VALUES (?, ?, ?, ?)
                """
                self.execute_update(query, (
                    language_code,
                    data['stopwords'],
                    data['educational_patterns'],
                    data['updated_at']
                ))

            return True
        except Exception as e:
            logger.error(f"Error saving language resources: {e}")
            return False

    def _execute_query_with_timeout(self, query, params=(), timeout_seconds=5):
        """
        Execute a query with a timeout to avoid long-running queries.

        Args:
            query: SQL query
            params: Query parameters
            timeout_seconds: Maximum execution time in seconds

        Returns:
            Query results or empty list on timeout
        """
        try:
            with self.get_connection() as conn:
                # Set query timeout
                conn.execute(f"PRAGMA busy_timeout = {timeout_seconds * 1000}")

                # Execute with timeout
                cursor = conn.cursor()
                cursor.execute(query, params)

                # Get results
                columns = [col[0] for col in cursor.description] if cursor.description else []
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
        except sqlite3.OperationalError as e:
            if "timeout" in str(e).lower():
                logger.warning(f"Query timed out after {timeout_seconds} seconds: {query[:100]}...")
                return []
            else:
                logger.error(f"Database error: {e}")
                raise
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def _get_playlist_video_ids(self, playlist_id):
        """Get video IDs for a playlist"""
        query = "SELECT video_ids FROM playlists WHERE playlist_id = ?"
        results = self.execute_query(query, (playlist_id,))

        if results and results[0]["video_ids"]:
            return results[0]["video_ids"].split(",")
        return []

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

            logger.info("Cleared all cache")

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
                    theory_practice_ratio, indexed_at, processing_status, processing_errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        Updated to use educational_value instead of content_type.

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
                segment_id, video_id, start_time, end_time, text, language, educational_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            # Transform segments into parameter tuples
            params_list = []
            search_index_params = []

            for segment in segments:
                segment_id = segment.get("id")
                if not segment_id:
                    continue

                language = segment.get("language", "")
                educational_value = segment.get("educational_value", 0.0)

                params_list.append((
                    segment_id,
                    video_id,
                    segment.get("start_time", 0.0),
                    segment.get("end_time", 0.0),
                    segment.get("text", ""),
                    language,
                    educational_value
                ))

                # Prepare search index parameters (with educational_weight instead of context_type)
                search_index_params.append((
                    segment_id,
                    segment.get("text", ""),
                    segment.get("domain", "unknown"),
                    "segment",
                    video_id,
                    language,
                    educational_value  # Use educational_value as educational_weight
                ))

            # Execute batch insert for segments
            self.execute_many(query, params_list)

            # Insert into search index
            if search_index_params:
                search_query = """
                INSERT INTO search_index (
                    id, text, domain, item_type, video_id, language, educational_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
        min_educational_value: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get segments for a video with optional filtering.
        Updated to use educational_value instead of context_type.

        Args:
            video_id: Video ID
            min_educational_value: Optional minimum educational value
            start_time: Optional filter for minimum start time
            end_time: Optional filter for maximum end time

        Returns:
            List of segment dictionaries
        """
        # Build cache key based on all parameters
        cache_key = self._get_cache_key("segments", (video_id, min_educational_value, start_time, end_time))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Build query parameters
        query = "SELECT * FROM segments WHERE video_id = ?"
        params = [video_id]

        if min_educational_value is not None:
            query += " AND educational_value >= ?"
            params.append(min_educational_value)

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

    def normalize_concept_text(self, text: str, language: str = "en") -> str:
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

    def find_similar_concepts(self, concept_text: str, domain: str = None, language: str = None) -> List[Dict[str, Any]]:
        """
        Find concepts similar to the given text.

        Args:
            concept_text: Concept text to match
            domain: Optional domain to limit search
            language: Optional language to limit search

        Returns:
            List of similar concepts
        """
        if not concept_text:
            return []

        # Normalize text for matching
        normalized_text = self.normalize_concept_text(concept_text, language)

        # Base query looking for similar concepts
        query = """
        SELECT c.*,
               CASE
                   WHEN c.normalized_text = ? THEN 3
                   WHEN c.normalized_text LIKE ? THEN 2
                   WHEN ? LIKE '%' || c.normalized_text || '%' THEN 1
                   ELSE 0
               END as match_score
        FROM concepts c
        WHERE (c.canonical_concept_id IS NULL OR c.canonical_concept_id = '')
        """

        params = [normalized_text, normalized_text + "%", normalized_text]

        # Add domain filter if provided
        if domain:
            query += " AND c.domain = ?"
            params.append(domain)

        # Add language filter if provided
        if language:
            query += " AND c.language = ?"
            params.append(language)

        # Order by match quality and limit results
        query += " ORDER BY match_score DESC LIMIT 10"

        # Execute query
        results = self.execute_query(query, tuple(params))

        # Filter to only return actually similar concepts
        similar_concepts = []
        for concept in results:
            # Skip concepts with no real similarity
            if concept['match_score'] == 0:
                continue

            # For concepts that didn't get a perfect match score,
            # perform additional string similarity check
            if concept['match_score'] < 3:
                # Calculate string similarity ratio
                similarity = difflib.SequenceMatcher(None,
                                                    normalized_text,
                                                    concept.get('normalized_text', '')).ratio()

                # Only include if similarity is high enough
                if similarity < 0.6:  # Require at least 60% similarity
                    continue

                # Add similarity score to concept data
                concept['similarity'] = similarity

            similar_concepts.append(concept)

        return similar_concepts

    def save_concept(self, concept_data: Dict[str, Any]) -> Optional[str]:
        """
        Save or update a concept with enhanced educational content tracking.
        Checks for similar existing concepts to avoid duplication.
        Updated to remove concept_class and use educational metrics.

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
        language = concept_data.get("language", "en")

        # Generate concept ID if not provided
        concept_id = concept_data.get("concept_id")
        if not concept_id:
            # Create deterministic ID based on text, domain and language
            text_for_hash = concept_text.lower().strip()
            concept_id = hashlib.md5(f"{text_for_hash}:{domain}:{language}".encode()).hexdigest()

        # Normalize text for better matching
        normalized_text = self.normalize_concept_text(concept_text, language)

        try:
            # STEP 1: Check for similar existing concepts before creating a new one
            similar_concepts = self.find_similar_concepts(concept_text, domain, language)

            canonical_concept_id = concept_data.get("canonical_concept_id")

            if similar_concepts and not canonical_concept_id:
                # Get the best matching concept
                best_match = similar_concepts[0]

                # If we have a very similar match, use its ID as canonical
                match_score = best_match.get('match_score', 0)
                similarity = best_match.get('similarity', 0)

                if match_score >= 3 or similarity > 0.85:
                    # This is essentially the same concept, use the existing one as canonical
                    canonical_concept_id = best_match.get('concept_id')
                    logger.info(f"Using canonical concept {canonical_concept_id} for similar concept: '{concept_text}'")

            # Calculate total_occurrences if missing
            total_occurrences = concept_data.get("total_occurrences", 0)
            if total_occurrences == 0:
                # Try to get from frequency or occurrence_count
                total_occurrences = concept_data.get("frequency", concept_data.get("occurrence_count", 1))

            # Get educational content metrics
            educational_weight = concept_data.get("educational_weight", 0.0)
            is_educational = concept_data.get("is_educational", educational_weight > 2.5)

            # STEP 2: Check if concept exists
            existing = self.get_concept(concept_id)

            if existing:
                # Update existing concept
                query = """
                UPDATE concepts SET
                    text = ?,
                    normalized_text = ?,
                    domain = ?,
                    language = ?,
                    total_occurrences = ?,
                    canonical_concept_id = ?,
                    educational_weight = ?,
                    is_educational = ?
                WHERE concept_id = ?
                """
                self.execute_update(query, (
                    concept_text,
                    normalized_text,
                    domain,
                    language,
                    total_occurrences,
                    canonical_concept_id,
                    educational_weight,
                    1 if is_educational else 0,
                    concept_id
                ))
                logger.debug(f"Updated existing concept: {concept_id} - {concept_text}")
            else:
                # Insert new concept
                query = """
                INSERT INTO concepts (
                    concept_id, text, normalized_text, domain, language,
                    total_occurrences, canonical_concept_id, educational_weight, is_educational
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.execute_update(query, (
                    concept_id,
                    concept_text,
                    normalized_text,
                    domain,
                    language,
                    total_occurrences,
                    canonical_concept_id,
                    educational_weight,
                    1 if is_educational else 0
                ))
                logger.debug(f"Inserted new concept: {concept_id} - {concept_text}")

            # STEP 3: Index for search - delete and reinsert to ensure freshness
            self.execute_update(
                "DELETE FROM search_index WHERE id = ? AND item_type = 'concept'",
                (concept_id,)
            )

            # Use the video_id from concept_data instead of None for proper indexing
            self.execute_update(
                """
                INSERT INTO search_index (id, text, domain, item_type, video_id, language, educational_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (concept_id, concept_text, domain, "concept", concept_data.get("video_id"), language, educational_weight)
            )

            # STEP 4: Clear cache for this concept
            self.clear_cache(f"concept_{concept_id}")

            logger.info(f"Saved concept {concept_id}: {concept_text}")

            # Return the canonical concept ID if this is a variant
            return canonical_concept_id or concept_id

        except Exception as e:
            logger.error(f"Error saving concept {concept_text}: {e}")
            return None

    def _find_concept_occurrences(self, concept_id, concept_text, segments, video_id):
        """
        Find segment occurrences of a concept with improved detection.

        Args:
            concept_id: Concept ID
            concept_text: Concept text
            segments: Video segments
            video_id: Video ID

        Returns:
            List of occurrence dictionaries
        """
        import hashlib
        import re

        occurrences = []
        concept_text_lower = concept_text.lower()
        concept_parts = concept_text_lower.split()

        # Single word vs multi-word search strategies
        if len(concept_parts) == 1:
            # For single words, search for exact word matches with word boundaries
            pattern = re.compile(r'\b' + re.escape(concept_text_lower) + r'\b', re.IGNORECASE)

            for segment in segments:
                segment_text = segment.get("text", "").lower()
                if pattern.search(segment_text):
                    occurrence_id = hashlib.md5(
                        f"{concept_id}:{segment['segment_id']}".encode()
                    ).hexdigest()

                    occurrences.append({
                        "occurrence_id": occurrence_id,
                        "concept_id": concept_id,
                        "video_id": video_id,
                        "segment_id": segment["segment_id"],
                        "start_time": segment.get("start_time", 0),
                        "end_time": segment.get("end_time", 0),
                        "context_text": segment.get("text", "")
                    })
        else:
            # For multi-word concepts, check if all parts appear and at least one exact match
            for segment in segments:
                segment_text = segment.get("text", "").lower()

                # Check for approximate match (all words present)
                if all(part in segment_text for part in concept_parts):
                    # Verify with more strict matching for phrases
                    concept_phrase = " ".join(concept_parts)

                    # Try to find an exact match of the phrase
                    if concept_phrase in segment_text or re.search(r'\b' + re.escape(concept_phrase) + r'\b', segment_text):
                        occurrence_id = hashlib.md5(
                            f"{concept_id}:{segment['segment_id']}".encode()
                        ).hexdigest()

                        occurrences.append({
                            "occurrence_id": occurrence_id,
                            "concept_id": concept_id,
                            "video_id": video_id,
                            "segment_id": segment["segment_id"],
                            "start_time": segment.get("start_time", 0),
                            "end_time": segment.get("end_time", 0),
                            "context_text": segment.get("text", "")
                        })

        return occurrences

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
                start_time, end_time, context_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
        min_educational_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get concepts extracted from a video with caching.
        Only returns canonical concepts (not duplicates/variants).
        Updated to use educational_value instead of context_type.

        Args:
            video_id: Video ID
            min_educational_value: Optional minimum educational value

        Returns:
            List of concept dictionaries with occurrence information
        """
        # Check cache first
        cache_key = self._get_cache_key("video_concepts", (video_id, min_educational_value))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # First check if there are any concepts in the database
        count_query = "SELECT COUNT(*) as count FROM concepts"
        count_result = self.execute_query(count_query)
        if count_result and count_result[0]["count"] == 0:
            logger.warning("No concepts found in database")
            return []

        # Check if there are any occurrences for this video
        occurrence_count_query = "SELECT COUNT(*) as count FROM occurrences WHERE video_id = ?"
        occurrence_count = self.execute_query(occurrence_count_query, (video_id,))
        if occurrence_count and occurrence_count[0]["count"] == 0:
            logger.warning(f"No concept occurrences found for video {video_id}")
            return []

        # Query for concepts in this video with educational filtering
        query = """
        SELECT c.*, COUNT(o.occurrence_id) as occurrence_count,
               MAX(o.start_time) as last_occurrence_time
        FROM concepts c
        JOIN occurrences o ON c.concept_id = o.concept_id
        WHERE o.video_id = ?
        AND (c.canonical_concept_id IS NULL OR c.canonical_concept_id = '') -- Only include canonical concepts
        """
        params = [video_id]

        if min_educational_value is not None:
            query += " AND c.educational_weight >= ?"
            params.append(min_educational_value)

        query += " GROUP BY c.concept_id ORDER BY c.educational_weight DESC, occurrence_count DESC"

        results = self.execute_query(query, tuple(params))

        # Cache the result
        self._set_in_cache(cache_key, results)

        return results

    def get_video_concepts(
        self,
        video_id: str,
        min_educational_value: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get concepts extracted from a video.
        Updated to use educational_value instead of theoretical/practical categories.

        Args:
            video_id: YouTube video ID
            min_educational_value: Optional minimum educational value filter

        Returns:
            Dictionary with video concepts or None if not found
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key("video_concept_data", (video_id, min_educational_value))
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # First check if there are any concepts in the database
            count_query = "SELECT COUNT(*) as count FROM concepts"
            count_result = self.execute_query(count_query)
            if count_result and count_result[0]["count"] == 0:
                logger.warning("No concepts found in database")
                return {}

            # Check if there are any occurrences for this video
            occurrence_count_query = "SELECT COUNT(*) as count FROM occurrences WHERE video_id = ?"
            occurrence_count = self.execute_query(occurrence_count_query, (video_id,))
            if occurrence_count and occurrence_count[0]["count"] == 0:
                logger.warning(f"No concept occurrences found for video {video_id}")
                return {}

            # Query for concepts in this video with educational filtering
            query = """
            SELECT c.*, COUNT(o.occurrence_id) as occurrence_count,
                MAX(o.start_time) as last_occurrence_time
            FROM concepts c
            JOIN occurrences o ON c.concept_id = o.concept_id
            WHERE o.video_id = ?
            AND (c.canonical_concept_id IS NULL OR c.canonical_concept_id = '') -- Only include canonical concepts
            """
            params = [video_id]

            if min_educational_value is not None:
                query += " AND c.educational_weight >= ?"
                params.append(min_educational_value)

            query += " GROUP BY c.concept_id ORDER BY c.educational_weight DESC, occurrence_count DESC"

            all_concepts = self.execute_query(query, tuple(params))

            # Separate concepts based on educational value
            educational_concepts = [c for c in all_concepts if c.get("is_educational") == 1]
            passing_concepts = [c for c in all_concepts if c.get("is_educational") == 0]

            video = self.get_video(video_id)

            result = {
                "video": video,
                "concepts": all_concepts,
                "educational_concepts": educational_concepts,
                "passing_concepts": passing_concepts,
                "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
            }

            # Cache the result
            self._set_in_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error getting video concepts: {e}")
            return None

    def build_enhanced_search_query(self, query_text: str, domain: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Build an enhanced search query with improved handling of different languages,
        special characters, and partial matches/typos.

        Args:
            query_text: Original query text
            domain: Optional domain for domain-specific handling
            language: Optional language for language-specific query handling

        Returns:
            Enhanced search query for SQLite FTS5
        """
        # Basic cleaning
        query_text = query_text.strip()
        if not query_text:
            return ""

        # For non-Latin queries, use a more flexible approach
        is_non_latin = any(ord(c) > 127 for c in query_text) or language == 'ru'

        if is_non_latin:
            # For non-Latin text (e.g., Russian), use a more flexible approach
            # Split by whitespace and quote each term
            terms = query_text.split()
            if not terms:
                return ""

            # For each term, create multiple matching patterns:
            # 1. Exact match
            # 2. Prefix match (for partial completion)
            # 3. Partial match within words (using NEAR operator)
            query_parts = []
            for term in terms:
                # Clean the term of any special characters
                term = ''.join(c for c in term if c.isalnum() or ord(c) > 127)
                if term:
                    if len(term) <= 2:
                        # For very short terms, just use exact matching
                        query_parts.append(f'"{term}"')
                    else:
                        # For longer terms, use both exact and prefix matching with different prefix lengths
                        # This will catch partial matches and some typos
                        prefixes = []
                        if len(term) > 3:
                            # Add shorter prefixes for better partial matching
                            prefixes.append(term[:len(term)-1] + '*')  # Missing last letter

                        # Add standard prefix
                        prefixes.append(f'{term}*')

                        # Combine exact match with prefixes
                        query_parts.append(f'("{term}" OR {" OR ".join(prefixes)})')

            # Combine terms with different strategies for better recall
            if len(query_parts) == 1:
                return query_parts[0]
            else:
                # Use both AND and OR combinations for balanced precision/recall
                and_query = " AND ".join(query_parts)
                or_query = " OR ".join(query_parts)
                return f"({and_query}) OR ({or_query})"

        # For Latin-based queries, use a more sophisticated approach
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

        # Add exact phrases with high weight
        for phrase in quoted_phrases:
            # Escape any special chars in the phrase
            safe_phrase = re.sub(r'[^\w\s]', ' ', phrase).strip()
            if safe_phrase:
                query_parts.append(f'"{safe_phrase}"^2')  # Give higher weight to exact matches

        # Process individual tokens with enhanced fuzzy matching
        if tokens:
            token_parts = []
            for token in tokens:
                if len(token) <= 2:
                    # For very short tokens, use exact matching
                    token_parts.append(token)
                else:
                    # For longer tokens, create multiple matching patterns
                    variations = [token]  # Start with exact token

                    # Add prefix matching (catches partial words and some typos)
                    variations.append(f"{token}*")

                    # For tokens longer than 4 chars, add more fuzzy variations
                    if len(token) > 4:
                        # Add prefix with one less character (handles missing letter)
                        variations.append(f"{token[:-1]}*")

                        # Add prefix with first 3 chars (very permissive matching)
                        if len(token) > 5:
                            variations.append(f"{token[:3]}*")

                    # Combine all variations for this token
                    token_parts.append(f"({' OR '.join(variations)})")

            # For multiple tokens, combine them effectively
            if len(token_parts) > 1:
                # Try as phrase first, then as individual tokens
                exact_tokens = " ".join(tokens)
                token_match = " AND ".join(token_parts)  # Require all tokens
                token_match_any = " OR ".join(token_parts)  # Any token matches

                # Combine with different weights
                query_parts.append(f'"{exact_tokens}"^2 OR ({token_match}) OR ({token_match_any})')
            else:
                # Just add the single token with its variations
                query_parts.append(token_parts[0])

        # Combine all parts with OR if multiple parts
        if len(query_parts) > 1:
            return " OR ".join(f"({part})" for part in query_parts)
        elif query_parts:
            return query_parts[0]
        else:
            return ""

    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced search using SQLite FTS.
        Updated to use educational_weight instead of context_type.

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
            language = query.get("language")  # Extract language filter
            educational_only = query.get("educational_only", False)  # Add educational filter
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
            cache_key = f"search_{query_text}_{domain}_{theory_practice_ratio}_{offset}_{limit}_{language}_{educational_only}"
            if filters:
                cache_key += f"_filters:{hash(str(filters))}"

            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached search results for query: {query_text}")
                return cached_result

            # First check if the search index has any entries
            check_query = "SELECT COUNT(*) as count FROM search_index"
            check_result = self.execute_query(check_query)
            has_search_index = check_result and check_result[0]["count"] > 0

            if not has_search_index:
                # Search index is empty - let's reindex everything to rebuild it
                logger.warning("Search index is empty. Rebuilding index from existing data...")

                # Get all videos
                videos_query = "SELECT video_id FROM videos WHERE processing_status = 'completed'"
                videos = self.execute_query(videos_query)

                if videos:
                    # Rebuild index for each video
                    for video in videos:
                        video_id = video["video_id"]

                        # Get all concepts for this video
                        concepts_query = """
                        SELECT DISTINCT c.*
                        FROM concepts c
                        JOIN occurrences o ON c.concept_id = o.concept_id
                        WHERE o.video_id = ?
                        """
                        concepts = self.execute_query(concepts_query, (video_id,))

                        # Reindex each concept
                        for concept in concepts:
                            concept_id = concept["concept_id"]

                            # Add to search index
                            index_query = """
                            INSERT INTO search_index (id, text, domain, item_type, video_id, language, educational_weight)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """
                            self.execute_update(
                                index_query,
                                (
                                    concept_id,
                                    concept["text"],
                                    concept["domain"],
                                    "concept",
                                    video_id,
                                    concept["language"],
                                    concept["educational_weight"]
                                )
                            )

                    # Check if segments table has domain column
                    try:
                        segments_columns = self.execute_query("PRAGMA table_info(segments)")
                        segment_column_names = [col["name"] for col in segments_columns]
                        has_domain_column = "domain" in segment_column_names
                    except:
                        has_domain_column = False

                    # Also reindex segments
                    if has_domain_column:
                        segments_query = "SELECT segment_id, video_id, text, domain, language, educational_value FROM segments"
                    else:
                        # Use video's domain if segment doesn't have it
                        segments_query = """
                        SELECT s.segment_id, s.video_id, s.text, v.domain, s.language, s.educational_value
                        FROM segments s
                        JOIN videos v ON s.video_id = v.video_id
                        """

                    try:
                        segments = self.execute_query(segments_query)

                        for segment in segments:
                            # Add to search index
                            index_query = """
                            INSERT INTO search_index (id, text, domain, item_type, video_id, language, educational_weight)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """
                            self.execute_update(
                                index_query,
                                (
                                    segment["segment_id"],
                                    segment["text"],
                                    segment.get("domain", "unknown"),
                                    "segment",
                                    segment["video_id"],
                                    segment.get("language", "en"),
                                    segment.get("educational_value", 0)
                                )
                            )

                        logger.info(f"Rebuilt search index with {len(concepts)} concepts and {len(segments)} segments")
                    except Exception as e:
                        logger.error(f"Error indexing segments: {e}")
                else:
                    logger.warning("No videos found to rebuild search index")

            # Check if we have any items in the search index after rebuilding
            count_query = "SELECT COUNT(*) as count FROM search_index"
            count_result = self.execute_query(count_query)
            if count_result and count_result[0]["count"] == 0:
                logger.warning("Search index still empty after rebuild attempt")
                return {
                    "results": [],
                    "totalResults": 0,
                    "executionTimeMs": int((time.time() - start_time) * 1000),
                    "status": "empty_index",
                    "message": "No content has been indexed yet"
                }

            # Build enhanced search query for non-exact matching
            search_query = self.build_enhanced_search_query(query_text, domain, language)
            logger.info(f"Using search query: {search_query}")

            # Simplify the search query to debug issues
            simple_query = f'"{query_text.lower()}"'

            # Build a basic query to see if anything matches
            debug_sql = """
            SELECT COUNT(*) as count FROM search_index
            WHERE search_index MATCH ?
            """
            debug_result = self.execute_query(debug_sql, (simple_query,))
            debug_count = debug_result[0]["count"] if debug_result else 0
            logger.info(f"Simple query '{simple_query}' matches {debug_count} items")

            # If simple query doesn't match anything, try an even simpler approach
            if debug_count == 0:
                # Use the raw text as is without any preprocessing
                raw_query = query_text.lower()
                debug_sql = "SELECT COUNT(*) as count FROM search_index WHERE lower(text) LIKE ?"
                debug_result = self.execute_query(debug_sql, (f"%{raw_query}%",))
                debug_count = debug_result[0]["count"] if debug_result else 0
                logger.info(f"Raw LIKE query '%{raw_query}%' matches {debug_count} items")

                # If we found matches with the raw LIKE query, use that approach
                if debug_count > 0:
                    logger.info(f"Using raw LIKE query instead of FTS for better results")
                    search_query = raw_query
                    use_fts = False
                else:
                    # Try with just the first word of the query
                    if " " in query_text:
                        first_word = query_text.split()[0].lower()
                        debug_sql = "SELECT COUNT(*) as count FROM search_index WHERE lower(text) LIKE ?"
                        debug_result = self.execute_query(debug_sql, (f"%{first_word}%",))
                        debug_count = debug_result[0]["count"] if debug_result else 0
                        logger.info(f"First word LIKE query '%{first_word}%' matches {debug_count} items")

                        if debug_count > 0:
                            search_query = first_word
                            use_fts = False
                        else:
                            use_fts = True
                    else:
                        use_fts = True
            else:
                use_fts = True

            # Use appropriate SQL based on whether we're using FTS or regular LIKE
            if use_fts:
                # Use FTS query
                sql = """
                SELECT
                    si.id, si.text, si.domain, si.item_type, si.video_id,
                    si.language, si.educational_weight,
                    v.title as video_title,
                    CASE WHEN si.item_type = 'segment' THEN s.start_time ELSE NULL END as start_time,
                    CASE WHEN si.item_type = 'segment' THEN s.end_time ELSE NULL END as end_time,
                    CASE WHEN si.item_type = 'segment' THEN s.text ELSE si.text END as context_text,
                    CASE WHEN si.item_type = 'concept' AND c.canonical_concept_id IS NOT NULL
                        THEN c.canonical_concept_id ELSE NULL END as canonical_concept_id
                FROM search_index si
                LEFT JOIN videos v ON si.video_id = v.video_id
                LEFT JOIN segments s ON si.item_type = 'segment' AND si.id = s.segment_id
                LEFT JOIN concepts c ON si.item_type = 'concept' AND si.id = c.concept_id
                WHERE search_index MATCH ?
                """
                params = [search_query]
            else:
                # Use LIKE query instead of FTS
                sql = """
                SELECT
                    si.id, si.text, si.domain, si.item_type, si.video_id,
                    si.language, si.educational_weight,
                    v.title as video_title,
                    CASE WHEN si.item_type = 'segment' THEN s.start_time ELSE NULL END as start_time,
                    CASE WHEN si.item_type = 'segment' THEN s.end_time ELSE NULL END as end_time,
                    CASE WHEN si.item_type = 'segment' THEN s.text ELSE si.text END as context_text,
                    CASE WHEN si.item_type = 'concept' AND c.canonical_concept_id IS NOT NULL
                        THEN c.canonical_concept_id ELSE NULL END as canonical_concept_id
                FROM search_index si
                LEFT JOIN videos v ON si.video_id = v.video_id
                LEFT JOIN segments s ON si.item_type = 'segment' AND si.id = s.segment_id
                LEFT JOIN concepts c ON si.item_type = 'concept' AND si.id = c.concept_id
                WHERE lower(si.text) LIKE ?
                """
                params = [f"%{search_query}%"]

            # Apply filters
            where_clauses = []

            if domain:
                where_clauses.append("si.domain = ?")
                params.append(domain)

            if language:
                where_clauses.append("(si.language = ? OR si.language IS NULL)")
                params.append(language)

            if "video_id" in filters:
                where_clauses.append("si.video_id = ?")
                params.append(filters["video_id"])

            if "video_ids" in filters and filters["video_ids"]:
                placeholders = ", ".join(["?"] * len(filters["video_ids"]))
                where_clauses.append(f"si.video_id IN ({placeholders})")
                params.extend(filters["video_ids"])

            # Apply educational content filter if requested
            if educational_only:
                where_clauses.append("si.educational_weight >= 2.5")

            if where_clauses:
                if "WHERE" in sql:
                    sql = sql.replace("WHERE", "WHERE " + " AND ".join(where_clauses) + " AND ")
                else:
                    sql += " WHERE " + " AND ".join(where_clauses)

            # Apply theory/practice ratio filter (using video-level ratio)
            if theory_practice_ratio is not None:
                # Add theory_practice_ratio filter
                if "WHERE" in sql:
                    if theory_practice_ratio > 0.7:
                        sql += " AND v.theory_practice_ratio >= 0.7"
                    elif theory_practice_ratio < 0.3:
                        sql += " AND v.theory_practice_ratio <= 0.3"
                    else:
                        sql += " AND v.theory_practice_ratio BETWEEN 0.3 AND 0.7"
                else:
                    if theory_practice_ratio > 0.7:
                        sql += " WHERE v.theory_practice_ratio >= 0.7"
                    elif theory_practice_ratio < 0.3:
                        sql += " WHERE v.theory_practice_ratio <= 0.3"
                    else:
                        sql += " WHERE v.theory_practice_ratio BETWEEN 0.3 AND 0.7"

            # Apply ordering by educational weight
            sql += " ORDER BY si.educational_weight DESC, si.rowid"

            # Apply pagination
            sql += f" LIMIT {limit} OFFSET {offset}"

            # Execute search query
            results = self.execute_query(sql, tuple(params))
            logger.info(f"Search found {len(results)} results")

            # Get total count (simplified for better performance)
            total_count = len(results)
            if offset == 0 and len(results) < limit:
                # If we got fewer results than the limit on the first page,
                # we know the total count is just the number of results
                pass
            else:
                # Otherwise, we need a separate count query
                if use_fts:
                    count_sql = "SELECT COUNT(*) as count FROM search_index WHERE search_index MATCH ?"
                    count_params = [search_query]
                else:
                    count_sql = "SELECT COUNT(*) as count FROM search_index WHERE lower(text) LIKE ?"
                    count_params = [f"%{search_query}%"]

                count_result = self.execute_query(count_sql, tuple(count_params))
                total_count = count_result[0]["count"] if count_result else 0

            # Format the results
            formatted_results = []
            for r in results:
                result = {
                    "result_type": r["item_type"],
                    "text": r["text"],
                    "domain": r["domain"],
                    "video_id": r["video_id"],
                    "video_title": r["video_title"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "context_text": r["context_text"],
                    "language": r["language"],
                    "educational_weight": r["educational_weight"]
                }

                # Add additional data for concepts
                if r["item_type"] == "concept":
                    result["concept_id"] = r["id"]

                    # Calculate is_educational from educational_weight
                    result["is_educational"] = r["educational_weight"] >= 2.5

                    # Add canonical relationship info if available
                    if r.get("canonical_concept_id"):
                        result["canonical_concept_id"] = r["canonical_concept_id"]
                        result["is_variant"] = True

                formatted_results.append(result)

            # Count educational vs passing concepts
            educational_count = sum(1 for r in formatted_results
                                if r.get("result_type") == "concept" and
                                    r.get("educational_weight", 0) >= 2.5)

            passing_count = sum(1 for r in formatted_results
                            if r.get("result_type") == "concept" and
                                r.get("educational_weight", 0) < 2.5)

            execution_time_ms = int((time.time() - start_time) * 1000)

            search_result = {
                "results": formatted_results,
                "totalResults": total_count,
                "educationalResults": educational_count,
                "passingResults": passing_count,
                "executionTimeMs": execution_time_ms
            }

            # Cache the result
            self._set_in_cache(cache_key, search_result)

            return search_result

        except Exception as e:
            logger.error(f"Error executing search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        Modified to use educational_weight instead of context_type.

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

            # Extract educational concepts
            concepts = domain_features.get("concepts", [])

            # Extract theory_practice_ratio
            theory_practice_results = processed_result.get("theory_practice_results", {})
            theory_practice_ratio = theory_practice_results.get("theory_practice_ratio", 0.5)

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
                "theory_practice_ratio": theory_practice_ratio,
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

            # Save concepts
            if not concepts:
                logger.warning(f"No concepts found for video {video_id}")
                return True  # Still return success as the segments were saved

            # Process concepts in batches for better performance
            batch_size = 20
            successful_concepts = 0
            concept_ids = []

            for i in range(0, len(concepts), batch_size):
                batch = concepts[i:i + batch_size]
                for concept in batch:
                    concept_data = concept.copy()
                    concept_data["video_id"] = video_id

                    # Ensure educational metadata is present
                    if "educational_weight" not in concept_data:
                        concept_data["educational_weight"] = 0.0
                    if "is_educational" not in concept_data:
                        concept_data["is_educational"] = concept_data["educational_weight"] > 2.5

                    concept_id = self.save_concept(concept_data)
                    if concept_id:
                        successful_concepts += 1
                        concept_ids.append(concept_id)

                        # Process occurrences if available
                        occurrences = concept.get("occurrences", [])
                        if occurrences:
                            # Ensure each occurrence has the right concept_id and video_id
                            for occurrence in occurrences:
                                occurrence["concept_id"] = concept_id
                                occurrence["video_id"] = video_id

                            # Save occurrences
                            self.save_occurrences(concept_id, occurrences)

            # Log success and clear related caches
            logger.info(f"Successfully indexed {successful_concepts}/{len(concepts)} concepts for video {video_id}")

            # Clear caches related to this video
            self.clear_cache(f"video_{video_id}")
            self.clear_cache(f"segments_{video_id}")
            self.clear_cache(f"video_concepts_{video_id}")
            self.clear_cache(f"video_concept_data_{video_id}")

            # Clear search cache
            from cache_manager import cache_clear
            cache_clear("search")

            return True

        except Exception as e:
            logger.error(f"Error indexing content: {e}")

            # Log detailed exception but only in debug mode
            import traceback
            logger.debug(f"Indexing error details: {traceback.format_exc()}")

            return False

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

            # Count canonical vs variant concepts
            canonical_count = self.execute_query(
                "SELECT COUNT(*) as count FROM concepts WHERE canonical_concept_id IS NULL OR canonical_concept_id = ''"
            )
            variant_count = self.execute_query(
                "SELECT COUNT(*) as count FROM concepts WHERE canonical_concept_id IS NOT NULL AND canonical_concept_id != ''"
            )

            table_stats["canonical_concepts"] = canonical_count[0]["count"] if canonical_count else 0
            table_stats["variant_concepts"] = variant_count[0]["count"] if variant_count else 0

            stats["tables"] = table_stats

            # Get domain distribution
            domains = self.execute_query(
                "SELECT domain, COUNT(*) as count FROM videos GROUP BY domain ORDER BY count DESC"
            )
            stats["domains"] = domains

            # Get educational content distribution
            educational_stats = self.execute_query("""
                SELECT
                    SUM(CASE WHEN is_educational = 1 THEN 1 ELSE 0 END) as educational,
                    SUM(CASE WHEN is_educational = 0 THEN 1 ELSE 0 END) as passing
                FROM concepts
                WHERE canonical_concept_id IS NULL OR canonical_concept_id = ''
            """)
            stats["educational_distribution"] = educational_stats[0] if educational_stats else {}

            # Get language distribution
            languages = self.execute_query(
                "SELECT language, COUNT(*) as count FROM concepts GROUP BY language ORDER BY count DESC"
            )
            stats["languages"] = languages

            # Get cache stats
            stats["cache"] = self.get_cache_stats()

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

# Global instance for singleton access
_data_access = None

def get_data_access(db_path: Optional[str] = "data/index/indexer.db") -> DataAccess:
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
