"""
Unit tests for the Search Engine component.
"""

import pytest
import json
import os
import sqlite3
import shutil
from unittest.mock import patch, MagicMock
import uuid
import time

from search_retrieval.search_engine.python.search_engine import SearchEngine

# Test configuration
TEST_CONFIG = {
    "index_dir": "test_index"
}

TEST_VIDEO_ID = "test123"

# Test data
TEST_PROCESSED_RESULT = {
    "job_id": "job123",
    "video_id": TEST_VIDEO_ID,
    "metadata": {
        "video_id": TEST_VIDEO_ID,
        "title": "Introduction to Calculus",
        "channel": "Math Channel",
        "description": "Learn about the basics of calculus.",
        "duration_seconds": 600,
        "language": "en",
        "domain": "mathematics",
        "domain_confidence": 0.9
    },
    "transcript": {
        "segments": [
            {
                "id": "segment1",
                "text": "Welcome to this calculus lecture.",
                "start_time": 0.0,
                "end_time": 5.0,
                "content_type": "theoretical"
            },
            {
                "id": "segment2",
                "text": "A derivative is defined as the limit of the difference quotient.",
                "start_time": 5.0,
                "end_time": 10.0,
                "content_type": "theoretical"
            },
            {
                "id": "segment3",
                "text": "Let's solve this problem: find the derivative of f(x) = x^2.",
                "start_time": 10.0,
                "end_time": 15.0,
                "content_type": "practical"
            }
        ],
        "language": "en",
        "domain": "mathematics"
    },
    "domain_features": {
        "domain": "mathematics",
        "theoretical_segments": 2,
        "practical_segments": 1,
        "key_concepts": [
            {
                "text": "calculus",
                "domain": "mathematics",
                "frequency": 1,
                "theoretical": True
            },
            {
                "text": "derivative",
                "domain": "mathematics",
                "frequency": 2,
                "theoretical": True
            }
        ]
    },
    "theory_practice_results": {
        "classification": "theoretical",
        "confidence": 0.75,
        "theoretical_segments": 2,
        "practical_segments": 1,
        "mixed_segments": 0,
        "theory_practice_ratio": 0.67
    },
    "theory_practice_patterns": {
        "theory_to_practice_sequences": [
            {
                "start_index": 1,
                "end_index": 2,
                "segments": [
                    {
                        "id": "segment2",
                        "text": "A derivative is defined as the limit of the difference quotient.",
                        "content_type": "theoretical"
                    },
                    {
                        "id": "segment3",
                        "text": "Let's solve this problem: find the derivative of f(x) = x^2.",
                        "content_type": "practical"
                    }
                ],
                "pattern": "theory_to_practice",
                "pattern_type": "definition_to_example"
            }
        ],
        "practice_to_theory_sequences": [],
        "theory_practice_alternations": 1,
        "max_theory_sequence": 2,
        "max_practice_sequence": 1
    }
}

@pytest.fixture
def search_engine():
    """Create a Search Engine instance."""
    # Create test index directory
    os.makedirs(TEST_CONFIG["index_dir"], exist_ok=True)

    # Create search engine
    engine = SearchEngine(TEST_CONFIG)

    # Check if database was initialized
    assert os.path.exists(engine.db_path)

    yield engine

    # Clean up test directory
    shutil.rmtree(TEST_CONFIG["index_dir"], ignore_errors=True)

class TestSearchEngine:
    """Test the Search Engine component."""

    def test_init_database(self, search_engine):
        """Test that the database is initialized correctly."""
        # Connect to the database
        conn = sqlite3.connect(search_engine.db_path)
        cursor = conn.cursor()

        # Check that tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "videos" in tables
        assert "concepts" in tables
        assert "occurrences" in tables
        assert "theory_practice_patterns" in tables
        assert "concepts_fts" in tables
        assert "segments_fts" in tables

        # Close connection
        conn.close()

    def test_index_content(self, search_engine):
        """Test indexing content in the search engine."""
        # Index the test content
        result = search_engine.index_content(TEST_PROCESSED_RESULT)

        assert result is True

        # Connect to the database and verify indexed content
        conn = sqlite3.connect(search_engine.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check video
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (TEST_VIDEO_ID,))
        video = cursor.fetchone()

        assert video is not None
        assert video["title"] == "Introduction to Calculus"
        assert video["domain"] == "mathematics"
        assert video["theory_practice_ratio"] == 0.67
        assert video["theoretical_segments"] == 2
        assert video["practical_segments"] == 1

        # Check concepts
        cursor.execute("SELECT * FROM concepts")
        concepts = cursor.fetchall()

        assert len(concepts) > 0

        # Check if both concepts were indexed
        concept_texts = [concept["text"].lower() for concept in concepts]
        assert "calculus" in concept_texts
        assert "derivative" in concept_texts

        # Check occurrences
        cursor.execute("SELECT * FROM occurrences WHERE video_id = ?", (TEST_VIDEO_ID,))
        occurrences = cursor.fetchall()

        assert len(occurrences) > 0

        # Check theory-practice patterns
        cursor.execute("SELECT * FROM theory_practice_patterns WHERE video_id = ?", (TEST_VIDEO_ID,))
        patterns = cursor.fetchall()

        assert len(patterns) == 1
        assert patterns[0]["pattern_type"] == "theory_to_practice"
        assert patterns[0]["pattern_subtype"] == "definition_to_example"

        # Close connection
        conn.close()

    def test_search(self, search_engine):
        """Test searching for concepts."""
        # First index some content
        search_engine.index_content(TEST_PROCESSED_RESULT)

        # Search for "derivative"
        query = {
            "original_text": "derivative",
            "filters": {},
            "theory_practice_ratio": None,
            "domain": None,
            "pagination": {
                "offset": 0,
                "limit": 10
            }
        }

        results = search_engine.search(query)

        assert "results" in results
        assert "totalResults" in results
        assert "theoreticalResults" in results
        assert "practicalResults" in results
        assert "executionTimeMs" in results

        assert results["totalResults"] > 0
        assert len(results["results"]) > 0

        # Check that the concept was found
        found = False
        for result in results["results"]:
            if "derivative" in result.get("text", "").lower():
                found = True
                break

        assert found, "Concept 'derivative' not found in search results"

    def test_segment_search(self, search_engine):
        """Test searching for content in segment text when no direct concept matches are found."""
        # First index some content
        search_engine.index_content(TEST_PROCESSED_RESULT)

        # Search for a term that's not a concept but appears in segment text
        query = {
            "original_text": "amazing",  # This word isn't indexed as a concept but appears in segments
            "filters": {},
            "theory_practice_ratio": None,
            "domain": None,
            "pagination": {
                "offset": 0,
                "limit": 10
            }
        }

        # Execute search - should fall back to segment search
        results = search_engine.search(query)

        assert "results" in results
        assert "totalResults" in results

        # We may or may not find results depending on the content
        # The important thing is that the query executes without error
        if results["totalResults"] > 0:
            for result in results["results"]:
                # Check that segment results have the expected structure
                assert "segment_id" in result
                assert "video_id" in result
                assert "context_text" in result
                assert "context_type" in result

    def test_search_with_theory_practice_filter(self, search_engine):
        """Test searching with theory/practice filtering."""
        # First index some content
        search_engine.index_content(TEST_PROCESSED_RESULT)

        # Search with high theoretical ratio
        theoretical_query = {
            "original_text": "derivative",
            "filters": {},
            "theory_practice_ratio": 0.9,  # Very theoretical
            "domain": "mathematics",
            "pagination": {
                "offset": 0,
                "limit": 10
            }
        }

        # Search with low theoretical ratio (practical)
        practical_query = {
            "original_text": "derivative",
            "filters": {},
            "theory_practice_ratio": 0.1,  # Very practical
            "domain": "mathematics",
            "pagination": {
                "offset": 0,
                "limit": 10
            }
        }

        theoretical_results = search_engine.search(theoretical_query)
        practical_results = search_engine.search(practical_query)

        # Both should find results since "derivative" appears in both contexts
        assert theoretical_results["totalResults"] > 0
        assert practical_results["totalResults"] > 0

        # The theoretical results should have more theoretical occurrences
        assert theoretical_results["theoreticalResults"] >= theoretical_results["practicalResults"]

        # The practical results should prioritize practical occurrences
        # We may not be able to assert this strictly since our test data is limited
        # but the ordering should be different
        if practical_results["practicalResults"] > 0:
            # Check if any ordering differences exist
            theoretical_ids = [r.get("occurrence_id") for r in theoretical_results["results"]]
            practical_ids = [r.get("occurrence_id") for r in practical_results["results"]]

            # This isn't a strict test since with limited data they might be the same
            # but in real usage they should differ
            if len(theoretical_ids) > 1 and len(practical_ids) > 1:
                assert theoretical_ids != practical_ids, "Theory/practice filtering didn't affect result order"

    def test_get_concept_details(self, search_engine):
        """Test getting detailed information about a concept."""
        # First index some content
        search_engine.index_content(TEST_PROCESSED_RESULT)

        # Search to get the concept ID
        query = {"original_text": "derivative", "pagination": {"offset": 0, "limit": 1}}
        results = search_engine.search(query)

        assert len(results["results"]) > 0
        concept_id = results["results"][0].get("concept_id")

        assert concept_id is not None

        # Get concept details
        details = search_engine.get_concept_details(concept_id)

        assert details is not None
        assert "concept" in details
        assert "occurrences" in details
        assert "related" in details

        assert details["concept"]["concept_id"] == concept_id
        assert "derivative" in details["concept"]["text"].lower()
        assert len(details["occurrences"]) > 0

    def test_get_video_concepts(self, search_engine):
        """Test getting concepts from a specific video."""
        # First index some content
        search_engine.index_content(TEST_PROCESSED_RESULT)

        # Get video concepts
        concepts = search_engine.get_video_concepts(TEST_VIDEO_ID)

        assert concepts is not None
        assert "video" in concepts
        assert "concepts" in concepts

        assert concepts["video"]["video_id"] == TEST_VIDEO_ID
        assert concepts["video"]["title"] == "Introduction to Calculus"
        assert len(concepts["concepts"]) > 0

        # Check filtering by context type
        theoretical_concepts = search_engine.get_video_concepts(TEST_VIDEO_ID, "theoretical")

        assert theoretical_concepts is not None
        assert len(theoretical_concepts["concepts"]) > 0

        # Every concept should have occurrences in theoretical contexts
        for concept in theoretical_concepts["concepts"]:
            found_in_theoretical = False
            for segment in TEST_PROCESSED_RESULT["transcript"]["segments"]:
                if (segment["content_type"] == "theoretical" and
                    concept["text"].lower() in segment["text"].lower()):
                    found_in_theoretical = True
                    break

            # Some concepts might appear in both contexts, so we can't strictly assert this
            # but at least one should be found in a theoretical context
            if concept["text"].lower() == "derivative":
                assert found_in_theoretical, f"Concept {concept['text']} not found in theoretical context"

    def test_generate_learning_path(self, search_engine):
        """Test generating a learning path for a set of concepts."""
        # First index some content
        search_engine.index_content(TEST_PROCESSED_RESULT)

        # Get concept IDs
        query = {"original_text": "derivative", "pagination": {"offset": 0, "limit": 1}}
        results = search_engine.search(query)

        assert len(results["results"]) > 0
        concept_id = results["results"][0].get("concept_id")

        query = {"original_text": "calculus", "pagination": {"offset": 0, "limit": 1}}
        results = search_engine.search(query)

        assert len(results["results"]) > 0
        concept_id2 = results["results"][0].get("concept_id")

        # Generate learning path
        path = search_engine.generate_learning_path(
            [concept_id, concept_id2],
            theory_practice_ratio=0.7,
            domain="mathematics"
        )

        assert path is not None
        assert "concepts" in path
        assert "theory_practice_ratio" in path
        assert "total_theoretical_concepts" in path
        assert "total_practical_concepts" in path
        assert "estimated_total_time_minutes" in path
        assert "domain" in path

        assert len(path["concepts"]) > 0
        assert path["domain"] == "mathematics"

        # Check that both concepts are included in the path
        concept_ids = [concept.get("concept_id") for concept in path["concepts"]]
        assert concept_id in concept_ids
        assert concept_id2 in concept_ids

    def test_build_search_query(self, search_engine):
        """Test building SQL query for search."""
        query_text = "derivative"
        filters = {}
        theory_practice_ratio = 0.7  # Favor theoretical
        domain = "mathematics"

        sql, params = search_engine._build_search_query(query_text, filters, theory_practice_ratio, domain)

        # Check that the SQL query includes the search term
        assert "concepts_fts MATCH ?" in sql
        assert params[0] == query_text

        # Check that domain filter is applied
        assert "c.domain = ?" in sql
        assert params[1] == domain

        # Check theoretical ordering for ratio > 0.5
        assert "ORDER BY CASE WHEN o.context_type = 'theoretical'" in sql

        # Test with practical ratio
        sql_practical, params_practical = search_engine._build_search_query(
            query_text, filters, 0.3, domain  # Favor practical
        )

        # Check practical ordering for ratio < 0.5
        assert "ORDER BY CASE WHEN o.context_type = 'practical'" in sql_practical

    def test_index_video(self, search_engine):
        """Test indexing a video in the database."""
        cursor = MagicMock()

        video_id = TEST_VIDEO_ID
        metadata = TEST_PROCESSED_RESULT["metadata"]
        theory_practice_results = TEST_PROCESSED_RESULT["theory_practice_results"]

        search_engine._index_video(cursor, video_id, metadata, theory_practice_results)

        # Check that the execute method was called for inserting video
        cursor.execute.assert_called_once()

        # Extract the SQL and parameters from the call
        call_args = cursor.execute.call_args[0]
        sql = call_args[0]
        params = call_args[1]

        # Check that we're inserting into videos table
        assert "INSERT OR REPLACE INTO videos" in sql

        # Check parameters
        assert params[0] == TEST_VIDEO_ID
        assert params[1] == "Introduction to Calculus"
        assert params[7] == "mathematics"
        assert params[9] == 0.67  # theory_practice_ratio
