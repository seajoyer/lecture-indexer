"""
Video Repository module for the Lecture Video Content Indexer.
Handles persistence operations for video data with optimized queries and caching.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from database.db_manager import DBManager

# Configure logging
logger = logging.getLogger(__name__)

class VideoRepository:
    """
    Repository for video data with optimized persistence operations.
    Provides methods to save, retrieve, and query video information.
    """

    def __init__(self, db_manager: DBManager):
        """
        Initialize the video repository.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._ensure_schema()
        logger.info("VideoRepository initialized")

    def _ensure_schema(self):
        """Ensure video-related tables exist in the database."""
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
            playlist_id TEXT,
            processing_status TEXT,
            processing_errors TEXT,
            processing_time REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Segments table for storing transcript segments
        CREATE TABLE IF NOT EXISTS segments (
            segment_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            start_time REAL,
            end_time REAL,
            text TEXT,
            text_normalized TEXT,
            text_stemmed TEXT,
            domain TEXT,
            context_type TEXT,
            segment_num INTEGER,
            video_title TEXT,
            theory_practice_score REAL,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );

        -- Create index on video_id in segments for faster queries
        CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);

        -- Theory practice patterns table
        CREATE TABLE IF NOT EXISTS theory_practice_patterns (
            pattern_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            pattern_type TEXT,
            pattern_subtype TEXT,
            start_segment_id TEXT,
            end_segment_id TEXT,
            start_time REAL,
            end_time REAL,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
            FOREIGN KEY (start_segment_id) REFERENCES segments(segment_id) ON DELETE SET NULL,
            FOREIGN KEY (end_segment_id) REFERENCES segments(segment_id) ON DELETE SET NULL
        );

        -- Create index on video_id in patterns for faster queries
        CREATE INDEX IF NOT EXISTS idx_patterns_video_id ON theory_practice_patterns(video_id);

        -- Playlists table for organizing videos
        CREATE TABLE IF NOT EXISTS playlists (
            playlist_id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            channel TEXT,
            video_count INTEGER,
            indexed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Playlist videos junction table
        CREATE TABLE IF NOT EXISTS playlist_videos (
            playlist_id TEXT,
            video_id TEXT,
            position INTEGER,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (playlist_id, video_id),
            FOREIGN KEY (playlist_id) REFERENCES playlists(playlist_id) ON DELETE CASCADE,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );

        -- Video processing queue
        CREATE TABLE IF NOT EXISTS video_processing_queue (
            queue_id TEXT PRIMARY KEY,
            video_id TEXT,
            video_url TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
            error TEXT,
            metadata TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );

        -- Create indexes for efficient queue processing
        CREATE INDEX IF NOT EXISTS idx_queue_status ON video_processing_queue(status);
        CREATE INDEX IF NOT EXISTS idx_queue_priority ON video_processing_queue(priority);
        """

        try:
            self.db_manager.execute_script(schema_script)
            logger.info("Video repository schema initialized")
        except Exception as e:
            logger.error(f"Error initializing video repository schema: {e}")
            raise

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

        # Check if video exists
        existing = self.get_video(video_id)
        current_time = datetime.now().isoformat()

        try:
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
                    playlist_id = ?,
                    processing_status = ?,
                    processing_errors = ?,
                    processing_time = ?,
                    updated_at = ?
                WHERE video_id = ?
                """
                self.db_manager.execute_update(query, (
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
                    video_data.get("indexed_at", current_time),
                    video_data.get("playlist_id"),
                    video_data.get("processing_status", "completed"),
                    video_data.get("processing_errors"),
                    video_data.get("processing_time", 0.0),
                    current_time,
                    video_id
                ))
            else:
                # Insert new video
                query = """
                INSERT INTO videos (
                    video_id, title, description, channel, publication_date,
                    duration_seconds, language, domain, domain_confidence,
                    theory_practice_ratio, theoretical_segments, practical_segments,
                    indexed_at, playlist_id, processing_status, processing_errors,
                    processing_time, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.db_manager.execute_update(query, (
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
                    video_data.get("indexed_at", current_time),
                    video_data.get("playlist_id"),
                    video_data.get("processing_status", "completed"),
                    video_data.get("processing_errors"),
                    video_data.get("processing_time", 0.0),
                    current_time,
                    current_time
                ))

            logger.info(f"Saved video metadata for {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving video {video_id}: {e}")
            return False

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
            self.db_manager.execute_update(
                "DELETE FROM segments WHERE video_id = ?",
                (video_id,)
            )

            # Prepare batch insert
            query = """
            INSERT INTO segments (
                segment_id, video_id, start_time, end_time, text,
                text_normalized, text_stemmed, domain, context_type,
                segment_num, video_title, theory_practice_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            # Transform segments into parameter tuples
            params_list = []
            for i, segment in enumerate(segments):
                # Calculate theory_practice_score based on context_type
                if segment.get("content_type") == "theoretical":
                    theory_practice_score = 1.0
                elif segment.get("content_type") == "practical":
                    theory_practice_score = 0.0
                else:
                    theory_practice_score = 0.5

                params_list.append((
                    segment.get("id", ""),
                    video_id,
                    segment.get("start_time", 0.0),
                    segment.get("end_time", 0.0),
                    segment.get("text", ""),
                    segment.get("text_normalized", ""),
                    segment.get("text_stemmed", ""),
                    segment.get("domain", "unknown"),
                    segment.get("content_type", "mixed"),
                    i,
                    segment.get("video_title", ""),
                    theory_practice_score
                ))

            # Execute batch insert
            self.db_manager.execute_many(query, params_list)
            logger.info(f"Saved {len(segments)} segments for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving segments for video {video_id}: {e}")
            return False

    def save_theory_practice_patterns(self, video_id: str, patterns: Dict[str, Any]) -> bool:
        """
        Save theory-practice patterns.

        Args:
            video_id: Video ID
            patterns: Dictionary of theory-practice patterns

        Returns:
            True if successful, False otherwise
        """
        if not video_id or not patterns:
            logger.error("Cannot save patterns without video_id or patterns")
            return False

        try:
            # Delete existing patterns
            self.db_manager.execute_update(
                "DELETE FROM theory_practice_patterns WHERE video_id = ?",
                (video_id,)
            )

            # Prepare lists for batch insert
            all_patterns = []

            # Get theory to practice sequences
            theory_to_practice = patterns.get("theory_to_practice_sequences", [])
            for pattern in theory_to_practice:
                segments = pattern.get("segments", [])
                if not segments:
                    continue

                all_patterns.append({
                    "pattern_id": pattern.get("id", f"tp_{int(time.time())}_{len(all_patterns)}"),
                    "video_id": video_id,
                    "pattern_type": "theory_to_practice",
                    "pattern_subtype": pattern.get("pattern_type", "general_theory_to_practice"),
                    "start_segment_id": segments[0].get("id") if segments else None,
                    "end_segment_id": segments[-1].get("id") if segments else None,
                    "start_time": segments[0].get("start_time", 0) if segments else 0,
                    "end_time": segments[-1].get("end_time", 0) if segments else 0
                })

            # Get practice to theory sequences
            practice_to_theory = patterns.get("practice_to_theory_sequences", [])
            for pattern in practice_to_theory:
                segments = pattern.get("segments", [])
                if not segments:
                    continue

                all_patterns.append({
                    "pattern_id": pattern.get("id", f"pt_{int(time.time())}_{len(all_patterns)}"),
                    "video_id": video_id,
                    "pattern_type": "practice_to_theory",
                    "pattern_subtype": pattern.get("pattern_type", "general_practice_to_theory"),
                    "start_segment_id": segments[0].get("id") if segments else None,
                    "end_segment_id": segments[-1].get("id") if segments else None,
                    "start_time": segments[0].get("start_time", 0) if segments else 0,
                    "end_time": segments[-1].get("end_time", 0) if segments else 0
                })

            # Execute batch insert
            if all_patterns:
                query = """
                INSERT INTO theory_practice_patterns (
                    pattern_id, video_id, pattern_type, pattern_subtype,
                    start_segment_id, end_segment_id, start_time, end_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """

                params_list = [(
                    p["pattern_id"],
                    p["video_id"],
                    p["pattern_type"],
                    p["pattern_subtype"],
                    p["start_segment_id"],
                    p["end_segment_id"],
                    p["start_time"],
                    p["end_time"]
                ) for p in all_patterns]

                self.db_manager.execute_many(query, params_list)

            logger.info(f"Saved {len(all_patterns)} theory-practice patterns for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving theory-practice patterns for video {video_id}: {e}")
            return False

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata by ID.

        Args:
            video_id: Video ID

        Returns:
            Video metadata dictionary or None if not found
        """
        if not video_id:
            return None

        query = "SELECT * FROM videos WHERE video_id = ?"
        results = self.db_manager.execute_query(query, (video_id,))

        if not results:
            return None

        return results[0]

    def get_video_segments(self, video_id: str, context_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get segments for a video.

        Args:
            video_id: Video ID
            context_type: Optional filter for content type (theoretical, practical, mixed)

        Returns:
            List of segment dictionaries
        """
        if not video_id:
            return []

        query = "SELECT * FROM segments WHERE video_id = ?"
        params = [video_id]

        if context_type:
            query += " AND context_type = ?"
            params.append(context_type)

        query += " ORDER BY segment_num"

        return self.db_manager.execute_query(query, tuple(params))

    def get_video_theory_practice_patterns(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get theory-practice patterns for a video.

        Args:
            video_id: Video ID

        Returns:
            List of pattern dictionaries
        """
        if not video_id:
            return []

        query = "SELECT * FROM theory_practice_patterns WHERE video_id = ? ORDER BY start_time"
        return self.db_manager.execute_query(query, (video_id,))

    def get_videos_by_domain(self, domain: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get videos by domain.

        Args:
            domain: Domain to filter by
            limit: Maximum number of videos to return
            offset: Number of videos to skip

        Returns:
            List of video dictionaries
        """
        query = """
        SELECT * FROM videos
        WHERE domain = ?
        ORDER BY indexed_at DESC
        LIMIT ? OFFSET ?
        """
        return self.db_manager.execute_query(query, (domain, limit, offset))

    def get_videos_by_theory_practice_ratio(
        self,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get videos by theory/practice ratio.

        Args:
            min_ratio: Minimum theory/practice ratio
            max_ratio: Maximum theory/practice ratio
            limit: Maximum number of videos to return
            offset: Number of videos to skip

        Returns:
            List of video dictionaries
        """
        query = """
        SELECT * FROM videos
        WHERE theory_practice_ratio BETWEEN ? AND ?
        ORDER BY theory_practice_ratio DESC
        LIMIT ? OFFSET ?
        """
        return self.db_manager.execute_query(query, (min_ratio, max_ratio, limit, offset))

    def get_videos_by_playlist(self, playlist_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get videos in a playlist.

        Args:
            playlist_id: Playlist ID
            limit: Maximum number of videos to return
            offset: Number of videos to skip

        Returns:
            List of video dictionaries
        """
        query = """
        SELECT v.* FROM videos v
        JOIN playlist_videos pv ON v.video_id = pv.video_id
        WHERE pv.playlist_id = ?
        ORDER BY pv.position
        LIMIT ? OFFSET ?
        """
        return self.db_manager.execute_query(query, (playlist_id, limit, offset))

    def save_playlist(self, playlist_data: Dict[str, Any], video_ids: List[str] = None) -> bool:
        """
        Save or update playlist metadata and add videos.

        Args:
            playlist_data: Playlist metadata dictionary
            video_ids: List of video IDs to add to the playlist

        Returns:
            True if successful, False otherwise
        """
        playlist_id = playlist_data.get("playlist_id")
        if not playlist_id:
            logger.error("Cannot save playlist without playlist_id")
            return False

        current_time = datetime.now().isoformat()

        try:
            with self.db_manager.transaction() as cursor:
                # Check if playlist exists
                cursor.execute("SELECT 1 FROM playlists WHERE playlist_id = ?", (playlist_id,))
                exists = cursor.fetchone() is not None

                if exists:
                    # Update existing playlist
                    cursor.execute("""
                    UPDATE playlists SET
                        title = ?,
                        description = ?,
                        channel = ?,
                        video_count = ?,
                        indexed_at = ?,
                        updated_at = ?
                    WHERE playlist_id = ?
                    """, (
                        playlist_data.get("title", ""),
                        playlist_data.get("description", ""),
                        playlist_data.get("channel", ""),
                        playlist_data.get("video_count", 0),
                        playlist_data.get("indexed_at", current_time),
                        current_time,
                        playlist_id
                    ))
                else:
                    # Insert new playlist
                    cursor.execute("""
                    INSERT INTO playlists (
                        playlist_id, title, description, channel,
                        video_count, indexed_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        playlist_id,
                        playlist_data.get("title", ""),
                        playlist_data.get("description", ""),
                        playlist_data.get("channel", ""),
                        playlist_data.get("video_count", 0),
                        playlist_data.get("indexed_at", current_time),
                        current_time,
                        current_time
                    ))

                # Add videos to playlist if provided
                if video_ids:
                    # Clear existing videos if this is an update
                    if exists:
                        cursor.execute("DELETE FROM playlist_videos WHERE playlist_id = ?", (playlist_id,))

                    # Insert new video associations
                    for position, video_id in enumerate(video_ids):
                        cursor.execute("""
                        INSERT INTO playlist_videos (
                            playlist_id, video_id, position, added_at
                        ) VALUES (?, ?, ?, ?)
                        """, (
                            playlist_id,
                            video_id,
                            position,
                            current_time
                        ))

                    # Update video count
                    cursor.execute("""
                    UPDATE playlists SET
                        video_count = ?,
                        updated_at = ?
                    WHERE playlist_id = ?
                    """, (
                        len(video_ids),
                        current_time,
                        playlist_id
                    ))

                    # Update playlist_id in videos table for each video
                    for video_id in video_ids:
                        cursor.execute("""
                        UPDATE videos SET
                            playlist_id = ?,
                            updated_at = ?
                        WHERE video_id = ?
                        """, (
                            playlist_id,
                            current_time,
                            video_id
                        ))

            logger.info(f"Saved playlist {playlist_id} with {len(video_ids) if video_ids else 0} videos")
            return True

        except Exception as e:
            logger.error(f"Error saving playlist {playlist_id}: {e}")
            return False

    def add_to_processing_queue(
        self,
        video_url: str,
        priority: int = 0,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Add a video to the processing queue.

        Args:
            video_url: YouTube video URL
            priority: Processing priority (higher = process sooner)
            metadata: Additional metadata for processing

        Returns:
            Queue ID if successful, None otherwise
        """
        import uuid

        # Generate a unique queue ID
        queue_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        try:
            # Extract video ID from URL if possible
            video_id = None
            if 'youtube.com' in video_url or 'youtu.be' in video_url:
                from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor
                extractor = YouTubeDataExtractor("dummy_key")  # We only need URL validation
                valid, video_id = extractor.validate_video_url(video_url)

                if not valid:
                    video_id = None

            # Convert metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO video_processing_queue (
                queue_id, video_id, video_url, priority, status,
                created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            self.db_manager.execute_update(query, (
                queue_id,
                video_id,
                video_url,
                priority,
                "pending",
                current_time,
                metadata_json
            ))

            logger.info(f"Added video {video_url} to processing queue with ID {queue_id}")
            return queue_id

        except Exception as e:
            logger.error(f"Error adding video to processing queue: {e}")
            return None

    def get_next_from_queue(self) -> Optional[Dict[str, Any]]:
        """
        Get the next video from the processing queue.

        Returns:
            Queue item dictionary or None if queue is empty
        """
        current_time = datetime.now().isoformat()

        try:
            with self.db_manager.transaction() as cursor:
                # Get the highest priority pending item
                cursor.execute("""
                SELECT * FROM video_processing_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """)

                result = cursor.fetchone()
                if not result:
                    return None

                # Convert to dictionary
                queue_item = dict(result)

                # Mark as in progress
                cursor.execute("""
                UPDATE video_processing_queue
                SET status = 'processing', started_at = ?
                WHERE queue_id = ?
                """, (
                    current_time,
                    queue_item["queue_id"]
                ))

                # Parse metadata JSON if present
                if queue_item.get("metadata"):
                    try:
                        queue_item["metadata"] = json.loads(queue_item["metadata"])
                    except:
                        queue_item["metadata"] = {}

                return queue_item

        except Exception as e:
            logger.error(f"Error getting next item from processing queue: {e}")
            return None

    def update_queue_status(
        self,
        queue_id: str,
        status: str,
        error: str = None,
        video_id: str = None
    ) -> bool:
        """
        Update the status of a queue item.

        Args:
            queue_id: Queue ID
            status: New status ('completed', 'error', 'cancelled')
            error: Error message if status is 'error'
            video_id: Video ID if it was determined during processing

        Returns:
            True if successful, False otherwise
        """
        if not queue_id:
            return False

        current_time = datetime.now().isoformat()

        try:
            query = """
            UPDATE video_processing_queue
            SET status = ?, completed_at = ?
            """

            params = [status, current_time]

            if error:
                query += ", error = ?"
                params.append(error)

            if video_id:
                query += ", video_id = ?"
                params.append(video_id)

            query += " WHERE queue_id = ?"
            params.append(queue_id)

            self.db_manager.execute_update(query, tuple(params))
            logger.info(f"Updated queue item {queue_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"Error updating queue item {queue_id} status: {e}")
            return False

    def count_videos(self, domain: str = None) -> int:
        """
        Count videos in the repository.

        Args:
            domain: Optional domain filter

        Returns:
            Number of videos
        """
        query = "SELECT COUNT(*) as count FROM videos"
        params = ()

        if domain:
            query += " WHERE domain = ?"
            params = (domain,)

        results = self.db_manager.execute_query(query, params)
        return results[0]["count"] if results else 0

    def delete_video(self, video_id: str) -> bool:
        """
        Delete a video and all associated data.

        Args:
            video_id: Video ID

        Returns:
            True if successful, False otherwise
        """
        if not video_id:
            return False

        try:
            with self.db_manager.transaction() as cursor:
                # Delete all associated data (cascading delete)
                cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))

            logger.info(f"Deleted video {video_id} and all associated data")
            return True

        except Exception as e:
            logger.error(f"Error deleting video {video_id}: {e}")
            return False
