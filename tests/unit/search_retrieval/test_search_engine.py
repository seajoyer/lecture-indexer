"""
Tests for the SearchEngine class.
"""

import pytest
import os
import json
import tempfile
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch

from search_retrieval.python.search_engine import SearchEngine
from database.db_init import get_db_context, init_database


@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        "output_dir": tempfile.mkdtemp(),
        "youtube_api_key": "test_key",
        "use_stemming": True,
        "use_fuzzy_matching": True,
        "min_ngram_size": 2,
        "max_ngram_size": 3,
        "fuzzy_match_threshold": 0.8
    }


@pytest.fixture
def test_db_path():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_indexer.db")
    yield db_path
    # Clean up
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def db_context(test_db_path):
    """Initialize a test database context."""
    db_context = init_database(test_db_path)
    yield db_context
    db_context.close()


@pytest.fixture
def search_engine(config, db_context):
    """Create a SearchEngine instance with a test database."""
    # Mock the get_db_context function to return our test db_context
    with patch('search_retrieval.python.search_engine.get_db_context', return_value=db_context):
        engine = SearchEngine(config)
        yield engine


@pytest.fixture
def sample_processed_result():
    """Create a sample processed result for testing indexing."""
    return {
        "video_id": "test123",
        "video_url": "https://www.youtube.com/watch?v=test123",
        "metadata": {
            "title": "Test Mathematics Lecture",
            "description": "A test lecture on calculus.",
            "channel": "Test Academy",
            "publication_date": "2023-01-01T00:00:00Z",
            "duration_seconds": 600,
            "language": "en",
            "domain": "mathematics",
            "domain_confidence": 0.8
        },
        "transcript": {
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
        },
        "domain_features": {
            "domain": "mathematics",
            "theoretical_segments": 2,
            "practical_segments": 1,
            "key_concepts": [
                {
                    "text": "calculus",
                    "domain": "mathematics",
                    "frequency": 2,
                    "theoretical": True,
                    "relevance": 0.8
                },
                {
                    "text": "derivative",
                    "domain": "mathematics",
                    "frequency": 2,
                    "theoretical": True,
                    "relevance": 0.7
                }
            ]
        },
        "theory_practice_results": {
            "classification": "theoretical",
            "confidence": 0.7,
            "theoretical_segments": 2,
            "practical_segments": 1,
            "theory_practice_ratio": 0.7
        },
        "theory_practice_patterns": {
            "theory_to_practice_sequences": [
                {
                    "start_index": 1,
                    "end_index": 2,
                    "segments": [
                        {
                            "id": "seg2",
                            "start_time": 10.0,
                            "end_time": 20.0,
                            "text": "Today we will learn about derivatives.",
                            "content_type": "theoretical"
                        },
                        {
                            "id": "seg3",
                            "start_time": 20.0,
                            "end_time": 30.0,
                            "text": "Let's solve an example problem: find the derivative of f(x) = x^2.",
                            "content_type": "practical"
                        }
                    ],
                    "pattern": "theory_to_practice",
                    "pattern_type": "concept_to_example"
                }
            ],
            "practice_to_theory_sequences": []
        }
    }


@pytest.fixture
def sample_search_query():
    """Create a sample search query."""
    return {
        "original_text": "calculus derivative",
        "filters": {},
        "theory_practice_ratio": None,
        "domain": None,
        "pagination": {
            "offset": 0,
            "limit": 10
        }
    }


def test_index_content(search_engine, sample_processed_result):
    """Test indexing content functionality."""
    # Mock the repository save methods to avoid actual database operations
    search_engine.db_context.video_repository.save_video = MagicMock(return_value=True)
    search_engine.db_context.video_repository.save_segments = MagicMock(return_value=True)
    search_engine.db_context.video_repository.save_theory_practice_patterns = MagicMock(return_value=True)
    search_engine.db_context.concept_repository.save_concept = MagicMock(return_value="concept1")
    search_engine.db_context.concept_repository.save_occurrences = MagicMock(return_value=True)
    search_engine.db_context.search_repository.index_video_metadata = MagicMock(return_value=True)
    search_engine.db_context.search_repository.index_segments = MagicMock(return_value=True)

    # Test indexing
    result = search_engine.index_content(sample_processed_result)

    # Assert successful indexing
    assert result is True

    # Verify that the repository methods were called correctly
    search_engine.db_context.video_repository.save_video.assert_called_once()
    search_engine.db_context.video_repository.save_segments.assert_called_once()
    search_engine.db_context.video_repository.save_theory_practice_patterns.assert_called_once()

    # Concept repository should be called for each concept
    assert search_engine.db_context.concept_repository.save_concept.call_count == len(
        sample_processed_result["domain_features"]["key_concepts"])

    # Search repository methods should be called
    search_engine.db_context.search_repository.index_video_metadata.assert_called_once()
    search_engine.db_context.search_repository.index_segments.assert_called_once()


def test_search(search_engine, sample_search_query):
    """Test search functionality."""
    # Mock the search repository search method
    mock_search_results = {
        "results": [
            {
                "result_type": "concept",
                "concept_id": "c1",
                "text": "calculus",
                "domain": "mathematics",
                "relevance_score": 0.9,
                "video_id": "test123",
                "video_title": "Test Mathematics Lecture",
                "segment_id": "seg1",
                "context_text": "Welcome to this mathematics lecture on calculus."
            }
        ],
        "totalResults": 1,
        "theoreticalResults": 1,
        "practicalResults": 0,
        "executionTimeMs": 50
    }

    search_engine.db_context.search_repository.search = MagicMock(return_value=mock_search_results)

    # Test search
    results = search_engine.search(sample_search_query)

    # Assert correct results
    assert results == mock_search_results
    search_engine.db_context.search_repository.search.assert_called_once_with(sample_search_query)


def test_search_with_filters(search_engine):
    """Test search with filters."""
    query_with_filters = {
        "original_text": "calculus",
        "filters": {
            "video_id": "test123"
        },
        "theory_practice_ratio": 0.7,
        "domain": "mathematics",
        "pagination": {
            "offset": 0,
            "limit": 10
        }
    }

    # Mock search to return filtered results
    mock_search_results = {
        "results": [
            {
                "result_type": "concept",
                "concept_id": "c1",
                "text": "calculus",
                "domain": "mathematics",
                "relevance_score": 0.9,
                "video_id": "test123",
                "video_title": "Test Mathematics Lecture",
                "segment_id": "seg1",
                "context_text": "Welcome to this mathematics lecture on calculus."
            }
        ],
        "totalResults": 1,
        "theoreticalResults": 1,
        "practicalResults": 0,
        "executionTimeMs": 50
    }

    search_engine.db_context.search_repository.search = MagicMock(return_value=mock_search_results)

    # Test search with filters
    results = search_engine.search(query_with_filters)

    # Assert correct results
    assert results == mock_search_results
    search_engine.db_context.search_repository.search.assert_called_once_with(query_with_filters)


def test_search_no_query(search_engine):
    """Test search with empty query."""
    empty_query = {
        "original_text": "",
        "filters": {},
        "pagination": {
            "offset": 0,
            "limit": 10
        }
    }

    # Test search with empty query
    results = search_engine.search(empty_query)

    # Assert empty results
    assert results["results"] == []
    assert results["totalResults"] == 0


def test_search_with_database_error(search_engine, sample_search_query):
    """Test search behavior when database encounters an error."""
    # Mock search to raise an exception
    error_message = "Database error"
    search_engine.db_context.search_repository.search = MagicMock(side_effect=Exception(error_message))

    # Test search with database error
    results = search_engine.search(sample_search_query)

    # Assert error results
    assert "error" in results
    assert results["results"] == []
    assert results["totalResults"] == 0
    assert error_message in results["error"]


def test_get_concept_details(search_engine):
    """Test getting concept details."""
    concept_id = "c1"
    mock_concept = {
        "concept_id": concept_id,
        "text": "calculus",
        "domain": "mathematics",
        "theoretical": True
    }
    mock_occurrences = [
        {
            "occurrence_id": "o1",
            "concept_id": concept_id,
            "video_id": "test123",
            "segment_id": "seg1",
            "context_text": "Welcome to this mathematics lecture on calculus."
        }
    ]
    mock_related = [
        {
            "relationship_id": "r1",
            "source_concept_id": concept_id,
            "target_concept_id": "c2",
            "target_text": "derivative",
            "relationship_type": "related"
        }
    ]
    mock_videos = [
        {
            "video_id": "test123",
            "title": "Test Mathematics Lecture",
            "occurrence_count": 1
        }
    ]

    # Mock repository methods
    search_engine.db_context.concept_repository.get_concept = MagicMock(return_value=mock_concept)
    search_engine.db_context.concept_repository.get_concept_occurrences = MagicMock(return_value=mock_occurrences)
    search_engine.db_context.concept_repository.get_concept_relationships = MagicMock(return_value=mock_related)
    search_engine.db_context.concept_repository.get_videos_for_concept = MagicMock(return_value=mock_videos)

    # Test getting concept details
    result = search_engine.get_concept_details(concept_id)

    # Assert correct results
    assert result["concept"] == mock_concept
    assert result["occurrences"] == mock_occurrences
    assert result["related"] == mock_related
    assert result["videos"] == mock_videos


def test_get_concept_details_not_found(search_engine):
    """Test getting details of a non-existent concept."""
    # Mock get_concept to return None
    search_engine.db_context.concept_repository.get_concept = MagicMock(return_value=None)

    # Test getting non-existent concept
    result = search_engine.get_concept_details("non_existent")

    # Assert None is returned
    assert result is None


def test_get_video_concepts(search_engine):
    """Test getting video concepts."""
    video_id = "test123"
    mock_video = {
        "video_id": video_id,
        "title": "Test Mathematics Lecture",
        "domain": "mathematics",
        "theory_practice_ratio": 0.7
    }
    mock_concepts = [
        {
            "concept_id": "c1",
            "text": "calculus",
            "domain": "mathematics",
            "theoretical": True,
            "occurrence_count": 2
        },
        {
            "concept_id": "c2",
            "text": "derivative",
            "domain": "mathematics",
            "theoretical": True,
            "occurrence_count": 2
        }
    ]
    mock_patterns = [
        {
            "pattern_id": "p1",
            "video_id": video_id,
            "pattern_type": "theory_to_practice",
            "start_time": 10.0,
            "end_time": 30.0
        }
    ]

    # Mock repository methods
    search_engine.db_context.video_repository.get_video = MagicMock(return_value=mock_video)
    search_engine.db_context.concept_repository.get_concepts_for_video = MagicMock(return_value=mock_concepts)
    search_engine.db_context.video_repository.get_video_theory_practice_patterns = MagicMock(return_value=mock_patterns)

    # Test getting video concepts
    result = search_engine.get_video_concepts(video_id)

    # Assert correct results
    assert result["video"] == mock_video
    assert result["concepts"] == mock_concepts
    assert result["theory_practice_patterns"] == mock_patterns
    assert result["theory_practice_ratio"] == 0.7


def test_get_video_concepts_not_found(search_engine):
    """Test getting concepts for a non-existent video."""
    # Mock get_video to return None
    search_engine.db_context.video_repository.get_video = MagicMock(return_value=None)

    # Test getting concepts for non-existent video
    result = search_engine.get_video_concepts("non_existent")

    # Assert None is returned
    assert result is None


def test_generate_learning_path(search_engine):
    """Test generating a learning path."""
    concept_ids = ["c1", "c2"]
    mock_concepts = [
        {
            "concept_id": "c1",
            "text": "calculus",
            "domain": "mathematics",
            "concept_class": "theoretical",
            "relevance": 0.8
        },
        {
            "concept_id": "c2",
            "text": "derivative",
            "domain": "mathematics",
            "concept_class": "theoretical",
            "relevance": 0.7
        }
    ]
    mock_related = [
        {
            "target_concept_id": "c3",
            "target_text": "integral",
            "target_domain": "mathematics",
            "target_class": "theoretical"
        }
    ]

    # Mock repository methods
    search_engine.db_context.concept_repository.get_concept = MagicMock(side_effect=lambda cid: next(
        (c for c in mock_concepts if c["concept_id"] == cid), None))
    search_engine.db_context.concept_repository.get_concept_relationships = MagicMock(return_value=mock_related)

    # Test generating a learning path
    result = search_engine.generate_learning_path(concept_ids)

    # Assert correct structure
    assert "concepts" in result
    assert "theory_practice_ratio" in result
    assert "domain" in result
    assert len(result["concepts"]) > 0

    # Assert each concept in the path has correct fields
    for concept in result["concepts"]:
        assert "text" in concept
        assert "domain" in concept
        assert "order" in concept


def test_generate_learning_path_empty_concepts(search_engine):
    """Test generating a learning path with empty concept list."""
    # Test generating a learning path with empty concept list
    result = search_engine.generate_learning_path([])

    # Assert None is returned
    assert result is None


def test_generate_learning_path_non_existent_concepts(search_engine):
    """Test generating a learning path with non-existent concepts."""
    # Mock get_concept to return None
    search_engine.db_context.concept_repository.get_concept = MagicMock(return_value=None)

    # Test generating a learning path with non-existent concepts
    result = search_engine.generate_learning_path(["non_existent"])

    # Assert None is returned
    assert result is None


def test_generate_learning_path_custom_ratio(search_engine):
    """Test generating a learning path with custom theory-practice ratio."""
    concept_ids = ["c1", "c2"]
    theory_practice_ratio = 0.3  # Prefer practical content

    mock_concepts = [
        {
            "concept_id": "c1",
            "text": "calculus",
            "domain": "mathematics",
            "concept_class": "theoretical",
            "relevance": 0.8
        },
        {
            "concept_id": "c2",
            "text": "derivative",
            "domain": "mathematics",
            "concept_class": "practical",
            "relevance": 0.7
        }
    ]

    # Mock repository methods
    search_engine.db_context.concept_repository.get_concept = MagicMock(side_effect=lambda cid: next(
        (c for c in mock_concepts if c["concept_id"] == cid), None))
    search_engine.db_context.concept_repository.get_concept_relationships = MagicMock(return_value=[])

    # Test generating a learning path with custom ratio
    result = search_engine.generate_learning_path(concept_ids, theory_practice_ratio)

    # Assert ratio is used
    assert result["theory_practice_ratio"] == theory_practice_ratio

    # Assert the learning path contains the concepts
    assert len(result["concepts"]) == 2
    assert any(c["text"] == "calculus" for c in result["concepts"])
    assert any(c["text"] == "derivative" for c in result["concepts"])


def test_index_content_no_video_id(search_engine):
    """Test indexing without a video ID."""
    invalid_result = {
        "metadata": {
            "title": "Test Video"
        }
    }

    # Test indexing invalid content
    result = search_engine.index_content(invalid_result)

    # Assert indexing fails
    assert result is False


def test_index_content_error_handling(search_engine, sample_processed_result):
    """Test error handling during indexing."""
    # Mock save_video to raise an exception
    search_engine.db_context.video_repository.save_video = MagicMock(side_effect=Exception("Test error"))

    # Test indexing with error
    result = search_engine.index_content(sample_processed_result)

    # Assert indexing fails
    assert result is False


def test_batch_index_content(search_engine, sample_processed_result):
    """Test batch indexing of content."""
    # Mock index_content to track calls
    search_engine.index_content = MagicMock(return_value=True)

    # Create a batch of results
    batch_results = [
        sample_processed_result,
        {**sample_processed_result, "video_id": "test456"}
    ]

    # Test batch indexing
    results = search_engine.batch_index_content(batch_results)

    # Verify index_content was called for each item
    assert search_engine.index_content.call_count == 2
    assert all(result for result in results.values())
    assert len(results) == 2


def test_optimize_database(search_engine):
    """Test database optimization."""
    # Mock optimization methods
    search_engine.db_context.search_repository.optimize_search_indexes = MagicMock(return_value=True)
    search_engine.db_context.optimize_database = MagicMock(return_value=True)

    # Test optimization
    result = search_engine.optimize_database()

    # Verify optimization methods were called
    assert result is True
    search_engine.db_context.search_repository.optimize_search_indexes.assert_called_once()
    search_engine.db_context.optimize_database.assert_called_once()


def test_rebuild_search_indexes(search_engine):
    """Test rebuilding search indexes."""
    # Mock rebuild method
    search_engine.db_context.search_repository.rebuild_search_indexes = MagicMock(return_value=True)
    search_engine.cache.flush = MagicMock()

    # Test rebuilding indexes
    result = search_engine.rebuild_search_indexes()

    # Verify rebuild method was called
    assert result is True
    search_engine.db_context.search_repository.rebuild_search_indexes.assert_called_once()
    search_engine.cache.flush.assert_called_once()
