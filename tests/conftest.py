"""
Enhanced Pytest configuration for the Lecture Video Content Indexer tests.
Includes database and caching setup for testing.
"""

import os
import sys
import pytest
import logging
import tempfile
import shutil
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Silence noisy loggers during tests
logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)
logging.getLogger('googleapiclient.http').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

# Mock implementations for database and caching
from database.db_manager import DBManager
from database.db_init import DatabaseContext
from common.utils.cache_manager import CacheManager

# Define custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark a test as a slow test"
    )
    config.addinivalue_line(
        "markers", "db: mark a test that needs database access"
    )

# Skip integration tests by default unless --integration flag is used
def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )
    parser.addoption(
        "--with-db", action="store_true", default=False, help="run tests with actual database"
    )

def pytest_collection_modifyitems(config, items):
    # Skip integration tests if --integration flag is not provided
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip database tests if --with-db flag is not provided
    if not config.getoption("--with-db"):
        skip_db = pytest.mark.skip(reason="need --with-db option to run")
        for item in items:
            if "db" in item.keywords:
                item.add_marker(skip_db)

# Test database fixture
@pytest.fixture(scope="session")
def test_db_path():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_indexer.db")
    yield db_path
    # Clean up after tests
    if os.path.exists(db_path):
        os.unlink(db_path)
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_db_manager(test_db_path):
    """Create a test database manager."""
    db_manager = DBManager(test_db_path, pool_size=2, timeout=5.0)
    yield db_manager
    db_manager.close_all_connections()

@pytest.fixture
def test_db_context(test_db_path):
    """Create a test database context with all repositories."""
    # Create test configuration
    config = {
        'sqlite': {
            'db_path': test_db_path,
            'pool_size': 2,
            'connection_timeout': 5.0,
            'enable_wal': False,  # Use simpler journal mode for tests
            'journal_mode': 'MEMORY',
            'synchronous': 0,  # Fastest setting for tests
            'foreign_keys': True,
            'cache_size': -1000,  # Smaller cache for tests
            'temp_store': 2
        },
        'cache': {
            'default_ttl': 60,  # Short TTL for tests
            'max_size': 100,
            'strategy': 'lru',
            'memory_limit_mb': 10,
            'cleanup_interval': 30,
            'regions': {
                'test': {'ttl': 60},
            }
        }
    }

    # Create test database context
    db_context = DatabaseContext(test_db_path)
    db_context.config = config

    yield db_context

    # Clean up
    db_context.close()

@pytest.fixture
def test_cache_manager():
    """Create a test cache manager."""
    cache_manager = CacheManager(max_size=100, default_ttl=60)
    yield cache_manager

@pytest.fixture
def mock_youtube_extractor():
    """Create a mock YouTube data extractor for testing."""
    from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor
    extractor = YouTubeDataExtractor("test_api_key")
    yield extractor

@pytest.fixture
def mock_transcript_processor():
    """Create a mock transcript processor for testing."""
    from data_acquisition.transcript_processor.python.transcript_processor import TranscriptProcessor
    processor = TranscriptProcessor()
    yield processor

@pytest.fixture
def mock_domain_classifier():
    """Create a mock domain classifier for testing."""
    from concept_analysis.concept_extractor.python.domain_concept_extractor import DomainClassifier
    classifier = DomainClassifier()
    yield classifier

@pytest.fixture
def mock_theory_practice_classifier():
    """Create a mock theory-practice classifier for testing."""
    from concept_analysis.relevance_analyzer.python.theory_practice_classifier import TheoryPracticeClassifier
    classifier = TheoryPracticeClassifier()
    yield classifier

@pytest.fixture
def sample_video_data():
    """Return sample video data for testing."""
    return {
        "video_id": "test123",
        "title": "Test Mathematics Lecture",
        "description": "A test lecture on calculus. Course: Test Course. Instructor: Test Instructor.",
        "channel": "Test Academy",
        "publication_date": "2023-01-01T00:00:00Z",
        "duration_seconds": 600,
        "language": "en",
        "domain": "mathematics",
        "domain_confidence": 0.8,
        "theory_practice_ratio": 0.7,
        "theoretical_segments": 7,
        "practical_segments": 3
    }

@pytest.fixture
def sample_transcript_data():
    """Return sample transcript data for testing."""
    return {
        "segments": [
            {
                "id": "seg1",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "Welcome to this mathematics lecture on calculus.",
                "language": "en",
                "content_type": "theoretical"
            },
            {
                "id": "seg2",
                "start_time": 10.0,
                "end_time": 20.0,
                "text": "Today we will learn about derivatives.",
                "language": "en",
                "content_type": "theoretical"
            },
            {
                "id": "seg3",
                "start_time": 20.0,
                "end_time": 30.0,
                "text": "Let's solve an example problem: find the derivative of f(x) = x^2.",
                "language": "en",
                "content_type": "practical"
            }
        ],
        "language": "en",
        "domain": "mathematics",
        "video_id": "test123"
    }
