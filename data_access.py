"""
Simplified data access layer for the Lecture Video Content Indexer.
Replaces the complex db_manager.py, db_init.py, and multiple repository classes.
"""

import os
import sqlite3
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class DataAccess:
    """
    Consolidated data access class that handles all database operations.
    Replaces VideoRepository, ConceptRepository, SearchRepository, and DBManager.
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

        # Initialize database schema
        self._ensure_schema()

        # Simple in-memory cache
        self.cache = {}

        logger.info(f"DataAccess initialized with database at {db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row factory for dict-like results.

        Returns:
            SQLite connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """
        Ensure all necessary tables exist in the database.
        Consolidates schema creation from all repositories.
        """
        schema_script = """
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

        -- Concepts table for storing concept information
        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            domain TEXT,
            concept_class TEXT,
            total_occurrences INTEGER DEFAULT 0
        );

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

        -- Create FTS table for search
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            id,
            text,
            domain,
            context_type,
            item_type,
            video_id,
            tokenize='porter unicode61'
        );
        """

        try:
            conn = self.get_connection()
            conn.executescript(schema_script)
            conn.commit()
            conn.close()
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as a list of dictionaries.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return results
        finally:
            conn.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an update query and return the number of affected rows.

        Args:
            query: SQL update query
            params: Query parameters

        Returns:
            Number of affected rows
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

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

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # VIDEO OPERATIONS (from VideoRepository)

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
            self._clear_cache(f"video_{video_id}")

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
        # Check cache first
        cache_key = f"video_{video_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = "SELECT * FROM videos WHERE video_id = ?"
        results = self.execute_query(query, (video_id,))

        if not results:
            return None

        # Cache the result
        self.cache[cache_key] = results[0]

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

            # Prepare batch insert
            query = """
            INSERT INTO segments (
                segment_id, video_id, start_time, end_time, text, context_type
            ) VALUES (?, ?, ?, ?, ?, ?)
            """

            # Transform segments into parameter tuples
            params_list = []
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

                # Index for search
                self._index_segment(
                    segment_id,
                    video_id,
                    segment.get("text", ""),
                    segment.get("domain", "unknown"),
                    segment.get("content_type", "mixed")
                )

            # Execute batch insert
            self.execute_many(query, params_list)

            # Clear cache for this video's segments
            self._clear_cache(f"segments_{video_id}")

            logger.info(f"Saved {len(segments)} segments for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving segments for video {video_id}: {e}")
            return False

    def get_video_segments(self, video_id: str, context_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get segments for a video.

        Args:
            video_id: Video ID
            context_type: Optional filter for context type

        Returns:
            List of segment dictionaries
        """
        # Check cache first
        cache_key = f"segments_{video_id}_{context_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = "SELECT * FROM segments WHERE video_id = ?"
        params = [video_id]

        if context_type:
            query += " AND context_type = ?"
            params.append(context_type)

        query += " ORDER BY start_time"

        results = self.execute_query(query, tuple(params))

        # Cache the result
        self.cache[cache_key] = results

        return results

    # CONCEPT OPERATIONS (from ConceptRepository)

    def save_concept(self, concept_data: Dict[str, Any]) -> Optional[str]:
        """
        Save or update a concept.

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
            concept_id = hashlib.md5(f"{concept_text.lower()}:{domain}".encode()).hexdigest()

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

            # Index for search
            self._index_concept(concept_id, concept_text, domain, concept_class)

            # Save occurrences if provided
            video_id = concept_data.get("video_id")
            if video_id:
                # Find segments containing this concept
                segments = self.get_video_segments(video_id)

                occurrences = []
                for segment in segments:
                    if concept_text.lower() in segment["text"].lower():
                        occurrence_id = hashlib.md5(f"{concept_id}:{segment['segment_id']}".encode()).hexdigest()
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
            self._clear_cache(f"concept_{concept_id}")

            logger.info(f"Saved concept {concept_id}: {concept_text}")
            return concept_id

        except Exception as e:
            logger.error(f"Error saving concept {concept_text}: {e}")
            return None

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by ID.

        Args:
            concept_id: Concept ID

        Returns:
            Concept dictionary or None if not found
        """
        # Check cache first
        cache_key = f"concept_{concept_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = "SELECT * FROM concepts WHERE concept_id = ?"
        results = self.execute_query(query, (concept_id,))

        if not results:
            return None

        # Cache the result
        self.cache[cache_key] = results[0]

        return results[0]

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
                start_time, end_time, context_type, context_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            params_list = []
            for occurrence in occurrences:
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
                self._clear_cache(f"concept_{concept_id}")
                for occurrence in occurrences:
                    video_id = occurrence.get("video_id")
                    if video_id:
                        self._clear_cache(f"video_concepts_{video_id}")

            logger.info(f"Saved {len(params_list)} occurrences for concept {concept_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving occurrences for concept {concept_id}: {e}")
            return False

    def get_concepts_for_video(self, video_id: str, context_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get concepts extracted from a video.

        Args:
            video_id: Video ID
            context_type: Optional context type filter

        Returns:
            List of concept dictionaries with occurrence information
        """
        # Check cache first
        cache_key = f"video_concepts_{video_id}_{context_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = """
        SELECT c.*, COUNT(o.occurrence_id) as occurrence_count
        FROM concepts c
        JOIN occurrences o ON c.concept_id = o.concept_id
        WHERE o.video_id = ?
        """
        params = [video_id]

        if context_type:
            query += " AND o.context_type = ?"
            params.append(context_type)

        query += " GROUP BY c.concept_id ORDER BY occurrence_count DESC"

        results = self.execute_query(query, tuple(params))

        # Cache the result
        self.cache[cache_key] = results

        return results

    def get_video_concepts(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get all concept and pattern information for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with video concepts and patterns
        """
        video = self.get_video(video_id)
        if not video:
            return None

        concepts = self.get_concepts_for_video(video_id)

        result = {
            "video": video,
            "concepts": concepts,
            "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
        }

        return result

    # SEARCH OPERATIONS (simplified from SearchRepository)

    def _index_segment(self, segment_id: str, video_id: str, text: str, domain: str, context_type: str) -> bool:
        """
        Index a segment for search.

        Args:
            segment_id: Segment ID
            video_id: Video ID
            text: Segment text
            domain: Content domain
            context_type: Context type (theoretical, practical, mixed)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing index entry
            self.execute_update(
                "DELETE FROM search_index WHERE id = ?",
                (segment_id,)
            )

            # Insert new index entry
            self.execute_update(
                """
                INSERT INTO search_index (id, text, domain, context_type, item_type, video_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (segment_id, text, domain, context_type, "segment", video_id)
            )

            return True
        except Exception as e:
            logger.error(f"Error indexing segment {segment_id}: {e}")
            return False

    def _index_concept(self, concept_id: str, text: str, domain: str, concept_class: str) -> bool:
        """
        Index a concept for search.

        Args:
            concept_id: Concept ID
            text: Concept text
            domain: Content domain
            concept_class: Concept class (theoretical, practical)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing index entry
            self.execute_update(
                "DELETE FROM search_index WHERE id = ?",
                (concept_id,)
            )

            # Insert new index entry
            self.execute_update(
                """
                INSERT INTO search_index (id, text, domain, context_type, item_type, video_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (concept_id, text, domain, concept_class, "concept", None)
            )

            return True
        except Exception as e:
            logger.error(f"Error indexing concept {concept_id}: {e}")
            return False

    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content for search.

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
                "processing_status": "completed"
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
            key_concepts = domain_features.get("key_concepts", [])
            for concept in key_concepts:
                concept_data = concept.copy()
                concept_data["video_id"] = video_id
                self.save_concept(concept_data)

            logger.info(f"Successfully indexed content for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return False

    def _build_enhanced_search_query(self, query_text: str) -> str:
        """
        Build an enhanced search query that supports non-exact matches.
        Handles multilingual queries including Cyrillic characters.

        Args:
            query_text: Original query text

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
        import re

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
        Perform search using SQLite FTS with support for non-exact matching.

        Args:
            query: Query parameters dictionary

        Returns:
            Search results dictionary
        """
        import time

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

            # Build enhanced search query for non-exact matching
            search_query = self._build_enhanced_search_query(query_text)

            # Build SQL query
            sql = """
            SELECT si.id, si.text, si.domain, si.context_type, si.item_type, si.video_id,
                v.title as video_title,
                s.start_time, s.end_time
            FROM search_index si
            LEFT JOIN videos v ON si.video_id = v.video_id
            LEFT JOIN segments s ON si.id = s.segment_id
            WHERE search_index MATCH ?
            """

            params = [search_query]

            # Apply filters
            if domain:
                sql += " AND si.domain = ?"
                params.append(domain)

            if "video_id" in filters:
                sql += " AND si.video_id = ?"
                params.append(filters["video_id"])

            if "video_ids" in filters and filters["video_ids"]:
                placeholders = ", ".join(["?"] * len(filters["video_ids"]))
                sql += f" AND si.video_id IN ({placeholders})"
                params.extend(filters["video_ids"])

            # Apply theory/practice ratio filter
            context_type_order = ""
            if theory_practice_ratio is not None:
                if theory_practice_ratio > 0.7:
                    context_type_order = " ORDER BY CASE WHEN si.context_type = 'theoretical' THEN 1 ELSE 2 END"
                elif theory_practice_ratio < 0.3:
                    context_type_order = " ORDER BY CASE WHEN si.context_type = 'practical' THEN 1 ELSE 2 END"

            # Apply ordering - use FTS5 rank for relevance
            if context_type_order:
                sql += context_type_order
            else:
                sql += " ORDER BY rank"

            # Apply pagination
            sql += f" LIMIT {limit} OFFSET {offset}"

            # Execute search query
            results = self.execute_query(sql, tuple(params))

            # Count total results
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
            theoretical_count = sum(1 for r in results if r["context_type"] == "theoretical")
            practical_count = sum(1 for r in results if r["context_type"] == "practical")

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
                    "end_time": r["end_time"]
                }

                # Get additional data for concepts
                if r["item_type"] == "concept":
                    concept = self.get_concept(r["id"])
                    if concept:
                        result["concept_id"] = r["id"]
                        result["concept_class"] = concept["concept_class"]

                formatted_results.append(result)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "results": formatted_results,
                "totalResults": total_count,
                "theoreticalResults": theoretical_count,
                "practicalResults": practical_count,
                "executionTimeMs": execution_time_ms
            }

        except Exception as e:
            logger.error(f"Error executing search: {e}")
            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "totalResults": 0,
                "executionTimeMs": execution_time_ms,
                "error": str(e)
            }

    # CACHE MANAGEMENT

    def _clear_cache(self, prefix: str = None) -> None:
        """
        Clear cache entries.

        Args:
            prefix: Optional prefix to clear selective cache entries
        """
        if prefix:
            # Clear specific cache entries
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                self.cache.pop(k, None)
        else:
            # Clear all cache
            self.cache.clear()

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
