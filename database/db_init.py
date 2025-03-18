"""
Database Initialization module for the Lecture Video Content Indexer.
Handles database creation, schema management, and initialization of necessary repositories.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from database.db_manager import DBManager
from database.video_repository import VideoRepository
from database.concept_repository import ConceptRepository
from database.search_repository import SearchRepository
from common.utils.cache_manager import CacheManager, CacheRegion

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseContext:
    """
    Central database context managing all repositories and caching.
    Provides a unified interface for database operations.
    """

    def __init__(self, config_path: str = "config/db_config.yaml"):
        """
        Initialize database context with configuration.

        Args:
            config_path: Path to database configuration file
        """
        self.config = self._load_config(config_path)

        # Initialize database manager
        db_path = self._get_db_path()
        pool_size = self.config['sqlite'].get('pool_size', 5)
        connection_timeout = self.config['sqlite'].get('connection_timeout', 30)

        self.db_manager = DBManager(db_path, pool_size, connection_timeout)

        # Apply SQLite configuration
        self._configure_sqlite()

        # Initialize repositories
        self.video_repository = VideoRepository(self.db_manager)
        self.concept_repository = ConceptRepository(self.db_manager)
        self.search_repository = SearchRepository(self.db_manager)

        # Initialize cache manager
        self._init_cache_manager()

        logger.info(f"DatabaseContext initialized with database at {db_path}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load database configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            # Try different locations if file doesn't exist
            potential_paths = [
                config_path,
                os.path.join(os.path.dirname(__file__), '..', config_path),
                os.path.join(os.getcwd(), config_path)
            ]

            config_file = None
            for path in potential_paths:
                if os.path.exists(path):
                    config_file = path
                    break

            if not config_file:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
                return self._get_default_config()

            # Load configuration
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded database configuration from {config_file}")
                return config

        except Exception as e:
            logger.error(f"Error loading database configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default database configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            'sqlite': {
                'db_path': 'data/index/indexer.db',
                'pool_size': 5,
                'connection_timeout': 30,
                'enable_wal': True,
                'journal_mode': 'WAL',
                'synchronous': 1,
                'foreign_keys': True,
                'cache_size': -2000,
                'temp_store': 2
            },
            'cache': {
                'default_ttl': 3600,
                'max_size': 1000,
                'strategy': 'hybrid',
                'memory_limit_mb': 100,
                'cleanup_interval': 300,
                'regions': {
                    'videos': {'ttl': 7200},
                    'concepts': {'ttl': 7200},
                    'search': {'ttl': 300},
                    'metadata': {'ttl': 14400}
                }
            }
        }

    def _get_db_path(self) -> str:
        """
        Get database path from configuration and ensure directory exists.

        Returns:
            Database file path
        """
        db_path = self.config['sqlite'].get('db_path', 'data/index/indexer.db')

        # Make sure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        return db_path

    def _configure_sqlite(self):
        """
        Apply SQLite configuration settings to the database.
        """
        try:
            sqlite_config = self.config.get('sqlite', {})

            with self.db_manager.transaction() as cursor:
                # Set journal mode
                journal_mode = sqlite_config.get('journal_mode', 'WAL')
                cursor.execute(f"PRAGMA journal_mode = {journal_mode}")

                # Set synchronous mode
                synchronous = sqlite_config.get('synchronous', 1)
                cursor.execute(f"PRAGMA synchronous = {synchronous}")

                # Set foreign keys
                foreign_keys = sqlite_config.get('foreign_keys', True)
                cursor.execute(f"PRAGMA foreign_keys = {'ON' if foreign_keys else 'OFF'}")

                # Set cache size
                cache_size = sqlite_config.get('cache_size', -2000)
                cursor.execute(f"PRAGMA cache_size = {cache_size}")

                # Set temp store
                temp_store = sqlite_config.get('temp_store', 2)
                cursor.execute(f"PRAGMA temp_store = {temp_store}")

                # Get current settings to verify
                cursor.execute("PRAGMA journal_mode")
                current_journal = cursor.fetchone()[0]

                cursor.execute("PRAGMA synchronous")
                current_sync = cursor.fetchone()[0]

                logger.info(f"SQLite configured with journal_mode={current_journal}, synchronous={current_sync}")

        except Exception as e:
            logger.error(f"Error configuring SQLite: {e}")

    def _init_cache_manager(self):
        """
        Initialize the cache manager and cache regions.
        """
        try:
            cache_config = self.config.get('cache', {})

            # Create cache manager
            max_size = cache_config.get('max_size', 1000)
            default_ttl = cache_config.get('default_ttl', 3600)
            cleanup_interval = cache_config.get('cleanup_interval', 300)
            strategy = cache_config.get('strategy', 'hybrid')
            memory_limit = cache_config.get('memory_limit_mb', 100)

            self.cache_manager = CacheManager(max_size, default_ttl, cleanup_interval, strategy)
            self.cache_manager.set_memory_limit(memory_limit)

            # Create cache regions
            regions = cache_config.get('regions', {})
            self.cache_regions = {}

            for name, settings in regions.items():
                region = self.cache_manager.region(name)
                self.cache_regions[name] = region

            logger.info(f"Initialized cache manager with {len(regions)} regions")

        except Exception as e:
            logger.error(f"Error initializing cache manager: {e}")
            # Create a default cache manager if initialization fails
            self.cache_manager = CacheManager()
            self.cache_regions = {}

    def get_cache_region(self, name: str) -> CacheRegion:
        """
        Get a cache region by name, creating it if it doesn't exist.

        Args:
            name: Region name

        Returns:
            Cache region
        """
        if name not in self.cache_regions:
            self.cache_regions[name] = self.cache_manager.region(name)

        return self.cache_regions[name]

    def optimize_database(self) -> bool:
        """
        Optimize database for better performance.

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Optimize database
            result = self.db_manager.optimize_database()

            # Optimize search indexes
            self.search_repository.optimize_search_indexes()

            # Flush caches
            self.cache_manager.flush()

            logger.info("Database optimization completed")
            return result

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
            result = self.db_manager.vacuum_database()
            logger.info("Database vacuum completed")
            return result

        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
            return False

    def close(self):
        """
        Close all database connections and clean up resources.
        """
        try:
            self.db_manager.close_all_connections()
            logger.info("Database connections closed")

        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Module-level singleton instance
_db_context = None

def init_database(config_path: str = "config/db_config.yaml") -> DatabaseContext:
    """
    Initialize the database and return the database context.

    Args:
        config_path: Path to database configuration file

    Returns:
        Database context
    """
    global _db_context

    if _db_context is None:
        _db_context = DatabaseContext(config_path)

    return _db_context

def get_db_context() -> Optional[DatabaseContext]:
    """
    Get the current database context.

    Returns:
        Database context or None if not initialized
    """
    return _db_context

def close_database():
    """
    Close the database connection.
    """
    global _db_context

    if _db_context is not None:
        _db_context.close()
        _db_context = None
        logger.info("Database closed")


if __name__ == "__main__":
    """
    Run database initialization when script is run directly.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize database
    db_context = init_database()

    # Run optimization
    db_context.optimize_database()

    logger.info("Database initialization completed successfully")
