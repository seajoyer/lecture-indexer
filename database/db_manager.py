"""
Database Manager module for the Lecture Video Content Indexer.
Provides a centralized interface for database operations with connection pooling and transaction management.
"""

import os
import sqlite3
import logging
import threading
import time
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class DBConnection:
    """
    Wrapper for SQLite connection with automatic transaction handling.
    Used as a context manager to ensure proper transaction management.
    """

    def __init__(self, connection: sqlite3.Connection, manager: 'DBManager'):
        """
        Initialize the database connection wrapper.

        Args:
            connection: SQLite connection
            manager: DBManager instance that created this connection
        """
        self.connection = connection
        self.manager = manager
        self.in_transaction = False
        self.cursor = None

    def __enter__(self):
        """Context manager entry point."""
        self.cursor = self.connection.cursor()
        self.in_transaction = True
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point with transaction handling."""
        if self.in_transaction:
            if exc_type is not None:
                logger.debug(f"Rolling back transaction due to {exc_type.__name__}: {exc_val}")
                self.connection.rollback()
            else:
                self.connection.commit()
            self.in_transaction = False

        if self.cursor:
            self.cursor.close()
            self.cursor = None

        # Return the connection to the pool
        self.manager._return_connection(self.connection)
        return False  # Let exceptions propagate

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute a SQL statement with automatic transaction handling.

        Args:
            sql: SQL statement
            params: Parameters for SQL statement

        Returns:
            Database cursor
        """
        if not self.cursor:
            self.cursor = self.connection.cursor()
            self.in_transaction = True

        return self.cursor.execute(sql, params)

    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """
        Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement
            params_list: List of parameter tuples

        Returns:
            Database cursor
        """
        if not self.cursor:
            self.cursor = self.connection.cursor()
            self.in_transaction = True

        return self.cursor.executemany(sql, params_list)

    def executescript(self, sql_script: str) -> sqlite3.Cursor:
        """
        Execute a SQL script.

        Args:
            sql_script: SQL script string

        Returns:
            Database cursor
        """
        if not self.cursor:
            self.cursor = self.connection.cursor()
            self.in_transaction = True

        return self.cursor.executescript(sql_script)

    def commit(self):
        """Commit the current transaction."""
        self.connection.commit()
        self.in_transaction = False

    def rollback(self):
        """Roll back the current transaction."""
        self.connection.rollback()
        self.in_transaction = False


class DBManager:
    """
    Database manager providing connection pooling and schema management.
    Centralized interface for database operations with performance optimizations.
    """

    def __init__(self, db_path: str, pool_size: int = 5, timeout: float = 30.0):
        """
        Initialize the database manager.

        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum number of connections in the pool
            timeout: Timeout for getting a connection from the pool
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize connection pool
        self.connection_pool = queue.Queue(maxsize=pool_size)
        self.active_connections = 0
        self.pool_lock = threading.RLock()

        # Fill the pool with initial connections
        self._fill_pool()

        # Schema version tracking
        self.current_schema_version = self._get_schema_version()

        logger.info(f"DBManager initialized with database at {db_path}")
        logger.info(f"Current schema version: {self.current_schema_version}")

    def _fill_pool(self):
        """Fill the connection pool with new connections up to pool_size."""
        with self.pool_lock:
            while not self.connection_pool.full() and self.active_connections < self.pool_size:
                try:
                    conn = self._create_connection()
                    self.connection_pool.put(conn, block=False)
                    self.active_connections += 1
                except queue.Full:
                    # Pool is full now
                    break
                except Exception as e:
                    logger.error(f"Error creating database connection: {e}")
                    break

    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection with optimal settings.

        Returns:
            Configured SQLite connection
        """
        # Enable URI connection string for more options
        conn = sqlite3.connect(f"file:{self.db_path}?mode=rwc", uri=True,
                              detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                              isolation_level=None,  # We'll manage transactions manually
                              check_same_thread=False)  # Allow usage from multiple threads

        # Optimize connection settings
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous = NORMAL")  # Balance between durability and speed
        conn.execute("PRAGMA cache_size = 10000")  # Larger cache for better performance
        conn.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints

        # Set row factory for dictionary-like results
        conn.row_factory = sqlite3.Row

        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection from the pool or create a new one if needed.

        Returns:
            SQLite connection

        Raises:
            TimeoutError: If timeout is reached while waiting for a connection
        """
        start_time = time.time()

        while True:
            try:
                return self.connection_pool.get(block=True, timeout=0.1)
            except queue.Empty:
                # Check if we can create a new connection
                with self.pool_lock:
                    if self.active_connections < self.pool_size:
                        conn = self._create_connection()
                        self.active_connections += 1
                        return conn

                # Check for timeout
                if time.time() - start_time > self.timeout:
                    logger.error("Timeout waiting for database connection")
                    raise TimeoutError("Timeout waiting for database connection")

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

    def _return_connection(self, conn: sqlite3.Connection):
        """
        Return a connection to the pool.

        Args:
            conn: SQLite connection to return
        """
        try:
            self.connection_pool.put(conn, block=False)
        except queue.Full:
            # Pool is full, close this connection
            with self.pool_lock:
                self.active_connections -= 1
                conn.close()

    def get_connection(self) -> DBConnection:
        """
        Get a database connection wrapped in a context manager.

        Returns:
            DBConnection: Connection wrapper to use as context manager
        """
        conn = self._get_connection()
        return DBConnection(conn, self)

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as a list of dictionaries.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        with self.get_connection() as cursor:
            cursor.execute(query, params)
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an update query and return the number of affected rows.

        Args:
            query: SQL update query
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_connection() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_script(self, script: str) -> None:
        """
        Execute a SQL script.

        Args:
            script: SQL script string
        """
        with self.get_connection() as cursor:
            cursor.executescript(script)

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query
            params_list: List of parameter tuples

        Returns:
            Total number of affected rows
        """
        with self.get_connection() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def transaction(self) -> DBConnection:
        """
        Start a transaction and return a connection for use in a context manager.

        Returns:
            DBConnection: Connection wrapper to use as context manager
        """
        return self.get_connection()

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Table name to check

        Returns:
            True if the table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
        """
        results = self.execute_query(query, (table_name,))
        return len(results) > 0

    def _get_schema_version(self) -> int:
        """
        Get the current schema version.

        Returns:
            Schema version number
        """
        # Create schema_version table if it doesn't exist
        self.execute_script("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Get current version
        results = self.execute_query("SELECT MAX(version) as version FROM schema_version")
        version = results[0]['version'] if results and results[0]['version'] is not None else 0
        return version

    def upgrade_schema(self, target_version: int, scripts: Dict[int, str]) -> bool:
        """
        Upgrade database schema to the target version.

        Args:
            target_version: Target schema version
            scripts: Dictionary mapping version numbers to SQL scripts

        Returns:
            True if upgrade was successful, False otherwise
        """
        current_version = self.current_schema_version

        if current_version >= target_version:
            logger.info(f"Database schema already at version {current_version}")
            return True

        logger.info(f"Upgrading database schema from version {current_version} to {target_version}")

        try:
            with self.transaction() as cursor:
                # Apply each version's script in order
                for version in range(current_version + 1, target_version + 1):
                    if version in scripts:
                        logger.info(f"Applying schema version {version}")
                        cursor.executescript(scripts[version])

                        # Update schema version
                        cursor.execute(
                            "INSERT INTO schema_version (version) VALUES (?)",
                            (version,)
                        )

                # Update current schema version
                self.current_schema_version = target_version

            logger.info(f"Database schema successfully upgraded to version {target_version}")
            return True

        except Exception as e:
            logger.error(f"Error upgrading database schema: {e}")
            return False

    def optimize_database(self) -> bool:
        """
        Optimize database for better performance.

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            with self.transaction() as cursor:
                # Run ANALYZE to update statistics
                cursor.execute("ANALYZE")

                # Run integrity check
                cursor.execute("PRAGMA integrity_check")

                # Optimize indexes
                cursor.execute("PRAGMA optimize")

            logger.info("Database optimization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False

    def vacuum_database(self) -> bool:
        """
        Run VACUUM to rebuild the database and reclaim unused space.

        Returns:
            True if vacuum was successful, False otherwise
        """
        try:
            # Need to get a direct connection for VACUUM
            conn = self._create_connection()
            try:
                conn.execute("VACUUM")
                logger.info("Database vacuum completed successfully")
                return True
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
            return False

    def close_all_connections(self):
        """Close all database connections in the pool."""
        with self.pool_lock:
            # Clear the queue
            while not self.connection_pool.empty():
                try:
                    conn = self.connection_pool.get(block=False)
                    conn.close()
                except queue.Empty:
                    break

            self.active_connections = 0

        logger.info("All database connections closed")

    def __del__(self):
        """Clean up resources when the object is deleted."""
        try:
            self.close_all_connections()
        except:
            pass  # Ignore errors during cleanup
