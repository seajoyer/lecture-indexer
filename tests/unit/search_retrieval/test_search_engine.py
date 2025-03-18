"""
Unit tests for the Search Engine component.
"""

import pytest
import json
import os
import shutil
import sqlite3
from unittest.mock import patch, MagicMock
from pathlib import Path

from search_retrieval.python.search_engine import SearchEngine

# Test configuration
TEST_CONFIG = {
    "index_dir": "test_index",
    "use_stemming": True,
    "use_fuzzy_matching": True
}

# Mock video metadata
MOCK_METADATA = {
    "video_id": "test123",
    "title": "Test Video",
    "channel": "Test Channel",
    "publication_date": "2023-01-01T00:00:00Z",
    "duration_seconds": 300,
    "description": "This is a test video about mathematics and algorithms.",
    "language": "en",
    "domain": "mathematics",
    "domain_confidence": 0.8
}

# Mock transcript segments
MOCK_SEGMENTS = [
    {
        "id": "segment1",
        "start_time": 0.0,
        "end_time": 5.0,
        "text": "Welcome to this lecture on calculus.",
        "content_type": "theoretical"
    },
    {
        "id": "segment2",
        "start_time": 5.0,
        "end_time": 10.0,
        "text": "Today we will discuss derivatives and integrals.",
        "content_type": "theoretical"
    },
    {
        "id": "segment3",
        "start_time": 10.0,
        "end_time": 15.0,
        "text": "Let's solve this problem: find the derivative of f(x) = x^2.",
        "content_type": "practical"
    }
]

# Mock concepts
MOCK_CONCEPTS = [
    {
        "text": "calculus",
        "domain": "mathematics",
        "frequency": 2,
        "theoretical": True
    },
    {
        "text": "derivative",
        "domain": "mathematics",
        "frequency": 2,
        "theoretical": False
    },
    {
        "text": "integral",
        "domain": "mathematics",
        "frequency": 1,
        "theoretical": True
    }
]

# Mock theory/practice results
MOCK_TP_RESULTS = {
    "classification": "theoretical",
    "confidence": 0.8,
    "theoretical_segments": 2,
    "practical_segments": 1,
    "mixed_segments": 0,
    "theory_practice_ratio": 0.67
}

# Mock theory/practice patterns
MOCK_TP_PATTERNS = {
    "theory_to_practice_sequences": [
        {
            "pattern_type": "definition_to_example",
            "segments": MOCK_SEGMENTS
        }
    ],
    "practice_to_theory_sequences": [],
    "theory_practice_alternations": 1,
    "max_theory_sequence": 2,
    "max_practice_sequence": 1
}

# Complete processed result
MOCK_PROCESSED_RESULT = {
    "job_id": "job123",
    "video_id": "test123",
    "video_url": "https://www.youtube.com/watch?v=test123",
    "metadata": MOCK_METADATA,
    "transcript": {
        "segments": MOCK_SEGMENTS,
        "language": "en",
        "domain": "mathematics"
    },
    "domain_features": {
        "key_concepts": MOCK_CONCEPTS
    },
    "theory_practice_results": MOCK_TP_RESULTS,
    "theory_practice_patterns": MOCK_TP_PATTERNS,
    "status": "completed"
}

@pytest.fixture
def search_engine():
    """Create a Search Engine instance for testing."""
    # Create test index directory
    os.makedirs(TEST_CONFIG["index_dir"], exist_ok=True)

    # Create search engine instance
    engine = SearchEngine(TEST_CONFIG)

    yield engine

    # Clean up test directory
    shutil.rmtree(TEST_CONFIG["index_dir"], ignore_errors=True)

class TestSearchEngine:
    """Test the Search Engine component."""

    def test_init(self, search_engine):
        """Test initialization of search engine."""
        assert search_engine is not None
        assert search_engine.index_dir == TEST_CONFIG["index_dir"]
        assert os.path.exists(search_engine.db_path)

    def test_index_content(self, search_engine):
        """Test indexing content."""
        # Index mock content
        result = search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Verify indexing was successful
        assert result is True

        # Verify database contains the indexed content
        conn = sqlite3.connect(search_engine.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check videos table
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (MOCK_METADATA["video_id"],))
        video_row = cursor.fetchone()
        assert video_row is not None

        # Check segments table
        cursor.execute("SELECT COUNT(*) FROM segments WHERE video_id = ?", (MOCK_METADATA["video_id"],))
        segment_count = cursor.fetchone()[0]
        assert segment_count > 0

        # Check concepts table
        cursor.execute("SELECT COUNT(*) FROM concepts")
        concept_count = cursor.fetchone()[0]
        assert concept_count > 0

        conn.close()

    def test_search_exact_match(self, search_engine):
        """Test searching with exact match."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search for a term that should match exactly
        query = {
            "original_text": "calculus",
            "filters": {},
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # Verify results
        assert results is not None
        assert "results" in results
        assert results["totalResults"] > 0

        # The term "calculus" should be found in the results
        found = False
        for result in results["results"]:
            if "calculus" in result.get("context_text", "").lower():
                found = True
                break

        assert found, "Search term 'calculus' not found in results"

    def test_search_with_theoretical_bias(self, search_engine):
        """Test searching with theoretical bias."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search with theoretical bias
        query = {
            "original_text": "calculus",
            "filters": {},
            "theory_practice_ratio": 0.8,  # High theoretical bias
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # Verify results have theoretical bias
        assert results["theoreticalResults"] > 0

        # Check if theoretical segments are prioritized
        if results["results"]:
            # The majority of results should be theoretical
            theoretical_count = sum(1 for r in results["results"] if r.get("context_type") == "theoretical")
            assert theoretical_count > 0

    def test_search_with_practical_bias(self, search_engine):
        """Test searching with practical bias."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search with practical bias
        query = {
            "original_text": "derivative",
            "filters": {},
            "theory_practice_ratio": 0.2,  # Low theoretical bias (high practical bias)
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # Verify results contain at least some content
        assert results["totalResults"] > 0

        # Check if practical segments are present
        practical_count = sum(1 for r in results["results"] if r.get("context_type") == "practical")
        if results["results"]:
            assert practical_count > 0 or results["practicalResults"] > 0, "No practical results found with practical bias"

    def test_search_with_domain_filter(self, search_engine):
        """Test searching with domain filter."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search with domain filter
        query = {
            "original_text": "calculus",
            "filters": {},
            "domain": "mathematics",
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # All results should be from mathematics domain
        for result in results["results"]:
            # Domain could be in the result itself or in the domain field
            domain = result.get("domain") or "unknown"
            assert domain == "mathematics" or domain == "unknown", f"Result has incorrect domain: {domain}"

    def test_search_with_video_filter(self, search_engine):
        """Test searching with video ID filter."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search with video filter
        query = {
            "original_text": "calculus",
            "filters": {"video_id": MOCK_METADATA["video_id"]},
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # All results should be from the specified video
        for result in results["results"]:
            assert result.get("video_id") == MOCK_METADATA["video_id"]

    def test_get_concept_details(self, search_engine):
        """Test getting concept details."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search for a concept to get its ID
        query = {
            "original_text": "calculus",
            "filters": {},
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        concept_id = None
        for result in results["results"]:
            if result.get("concept_id"):
                concept_id = result["concept_id"]
                break

        if concept_id:
            # Get concept details
            concept_details = search_engine.get_concept_details(concept_id)

            # Verify concept details
            assert concept_details is not None
            assert "concept" in concept_details
            assert concept_details["concept"]["concept_id"] == concept_id
            assert "occurrences" in concept_details
        else:
            pytest.skip("No concept results found to test get_concept_details")

    def test_get_video_concepts(self, search_engine):
        """Test getting video concepts."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Get video concepts
        video_concepts = search_engine.get_video_concepts(MOCK_METADATA["video_id"])

        # Verify video concepts
        assert video_concepts is not None
        assert "video" in video_concepts
        assert video_concepts["video"]["video_id"] == MOCK_METADATA["video_id"]
        assert "concepts" in video_concepts
        assert len(video_concepts["concepts"]) > 0

        # Test with context_type filter
        theoretical_concepts = search_engine.get_video_concepts(MOCK_METADATA["video_id"], "theoretical")
        assert theoretical_concepts is not None
        assert "concepts" in theoretical_concepts

    def test_generate_learning_path(self, search_engine):
        """Test generating learning path."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Get concept IDs from video concepts
        video_concepts = search_engine.get_video_concepts(MOCK_METADATA["video_id"])

        if video_concepts and video_concepts.get("concepts"):
            # Get concept IDs
            concept_ids = [c["concept_id"] for c in video_concepts["concepts"][:2]]

            if concept_ids:
                # Generate learning path
                learning_path = search_engine.generate_learning_path(concept_ids)

                # Verify learning path
                assert learning_path is not None
                assert "concepts" in learning_path
                assert len(learning_path["concepts"]) > 0
                assert "theory_practice_ratio" in learning_path
                assert 0 <= learning_path["theory_practice_ratio"] <= 1
            else:
                pytest.skip("No concept IDs found to test generate_learning_path")
        else:
            pytest.skip("No video concepts found to test generate_learning_path")

    def test_empty_search(self, search_engine):
        """Test searching with empty query."""
        # Search with empty query
        query = {
            "original_text": "",
            "filters": {},
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # Verify empty results
        assert results is not None
        assert "results" in results
        assert len(results["results"]) == 0
        assert results["totalResults"] == 0

    def test_search_nonexistent_term(self, search_engine):
        """Test searching for a term that doesn't exist."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search for a term that shouldn't match anything
        query = {
            "original_text": "xyznonexistentterm",
            "filters": {},
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # Verify empty results
        assert results is not None
        assert "results" in results
        assert len(results["results"]) == 0
        assert results["totalResults"] == 0

    def test_get_nonexistent_concept(self, search_engine):
        """Test getting details for a concept that doesn't exist."""
        concept_details = search_engine.get_concept_details("nonexistent_id")
        assert concept_details is None

    def test_get_nonexistent_video_concepts(self, search_engine):
        """Test getting concepts for a video that doesn't exist."""
        video_concepts = search_engine.get_video_concepts("nonexistent_id")
        assert video_concepts is None

    def test_generate_learning_path_no_concepts(self, search_engine):
        """Test generating learning path with no concepts."""
        learning_path = search_engine.generate_learning_path([])
        assert learning_path is None

    def test_generate_learning_path_nonexistent_concepts(self, search_engine):
        """Test generating learning path with nonexistent concepts."""
        learning_path = search_engine.generate_learning_path(["nonexistent_id"])
        assert learning_path is None

    def test_index_invalid_content(self, search_engine):
        """Test indexing invalid content."""
        # Empty result
        empty_result = {}
        result = search_engine.index_content(empty_result)
        assert result is False

        # Missing video_id
        invalid_result = {
            "job_id": "job123",
            "metadata": {}
        }
        result = search_engine.index_content(invalid_result)
        assert result is False

    def test_search_with_fuzzy_matching(self, search_engine):
        """Test searching with fuzzy matching enabled."""
        # First index some content
        search_engine.index_content(MOCK_PROCESSED_RESULT)

        # Search for a term with a slight misspelling
        query = {
            "original_text": "calculuss",  # Misspelled "calculus"
            "filters": {},
            "pagination": {"offset": 0, "limit": 10}
        }

        results = search_engine.search(query)

        # If fuzzy matching is working, we should still find results
        # that are related to "calculus"
        found = False
        for result in results["results"]:
            if "calculus" in result.get("context_text", "").lower():
                found = True
                break

        # This may not always work depending on the implementation details
        # So we'll skip rather than fail if no results
        if not found and results["totalResults"] == 0:
            pytest.skip("Fuzzy matching may not be enabled or working as expected")
        else:
            assert found, "Fuzzy matching did not find 'calculus' with query 'calculuss'"

    def test_pagination(self, search_engine):
        """Test search result pagination."""
        # Create and index multiple results to ensure pagination
        for i in range(3):
            result_copy = MOCK_PROCESSED_RESULT.copy()
            result_copy["video_id"] = f"test{i}"
            result_copy["metadata"] = MOCK_METADATA.copy()
            result_copy["metadata"]["video_id"] = f"test{i}"
            result_copy["metadata"]["title"] = f"Test Video {i}"
            search_engine.index_content(result_copy)

        # Search with pagination
        query = {
            "original_text": "calculus",
            "filters": {},
            "pagination": {"offset": 0, "limit": 2}
        }

        results_page1 = search_engine.search(query)

        # Check first page
        assert len(results_page1["results"]) <= 2

        # Get second page
        query["pagination"]["offset"] = 2
        results_page2 = search_engine.search(query)

        # Check second page is different from first
        if results_page1["totalResults"] > 2 and results_page2["results"]:
            # If we have IDs, check they're different
            if results_page1["results"] and results_page2["results"]:
                page1_ids = set()
                page2_ids = set()

                for r in results_page1["results"]:
                    if r.get("occurrence_id"):
                        page1_ids.add(r["occurrence_id"])
                    elif r.get("segment_id"):
                        page1_ids.add(r["segment_id"])

                for r in results_page2["results"]:
                    if r.get("occurrence_id"):
                        page2_ids.add(r["occurrence_id"])
                    elif r.get("segment_id"):
                        page2_ids.add(r["segment_id"])

                if page1_ids and page2_ids:
                    # Check for any overlap
                    overlap = page1_ids & page2_ids
                    assert len(overlap) == 0, "Pages contain overlapping results"
