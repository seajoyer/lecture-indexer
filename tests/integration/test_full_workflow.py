"""
Integration test for the full workflow of the Lecture Video Content Indexer.
Tests the complete process from data acquisition to search and retrieval.
"""

import pytest
import os
import tempfile
from typing import Dict, Any, List

from data_acquisition.youtube_api.python.data_pipeline import DataPipeline
from search_retrieval.python.search_engine import SearchEngine

# Test configuration
TEST_CONFIG = {
    "youtube_api_key": "test_api_key",  # Use test key to avoid real API calls
    "output_dir": None  # Will be set to a temporary directory
}

@pytest.fixture
def test_output_dir():
    """Create a temporary output directory for test results."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    import shutil
    shutil.rmtree(temp_dir)

@pytest.mark.integration
def test_complete_workflow(test_db_context, test_output_dir, mock_youtube_extractor,
                         mock_transcript_processor, mock_domain_classifier,
                         mock_theory_practice_classifier):
    """
    Test the complete workflow from data acquisition to search.
    Verifies the entire pipeline from video processing to search functionality.

    This test:
    1. Processes a sample video through the pipeline
    2. Verifies video metadata, transcript, and concept extraction
    3. Tests indexing the content for search
    4. Performs search operations
    5. Retrieves video concepts and concept details
    6. Tests batch processing
    """
    # 1. Set up test configuration
    config = TEST_CONFIG.copy()
    config["output_dir"] = test_output_dir

    # 2. Create pipeline with test components
    pipeline = DataPipeline(config)
    pipeline.db_context = test_db_context

    # Use mock components
    pipeline.youtube_extractor = mock_youtube_extractor
    pipeline.transcript_processor = mock_transcript_processor
    pipeline.domain_classifier = mock_domain_classifier
    pipeline.theory_practice_classifier = mock_theory_practice_classifier

    # 3. Process a sample video
    video_url = "https://www.youtube.com/watch?v=test123"
    result = pipeline.process_video(video_url)

    # 4. Verify the result contains expected components
    assert result is not None, "Processing result should not be None"
    assert "video_id" in result, "Result should contain video_id"
    assert "status" in result, "Result should contain status"
    assert result["status"] == "completed", f"Expected status 'completed', got '{result.get('status')}'"
    assert "metadata" in result, "Result should contain metadata"
    assert "transcript" in result, "Result should contain transcript"
    assert "domain_features" in result, "Result should contain domain_features"
    assert "theory_practice_results" in result, "Result should contain theory_practice_results"

    # Save the video_id for later use
    video_id = result["video_id"]

    # 5. Verify the video was properly classified
    assert result["metadata"]["domain"] in ["mathematics", "programming", "physics", "unknown"], \
        f"Invalid domain: {result['metadata'].get('domain')}"
    assert result["theory_practice_results"]["classification"] in ["theoretical", "practical", "mixed"], \
        f"Invalid classification: {result['theory_practice_results'].get('classification')}"

    # 6. Verify concepts were extracted (if supported by the mock)
    assert "key_concepts" in result["domain_features"], "Domain features should contain key_concepts"

    # 7. Create search engine and perform a search
    search_engine = SearchEngine(config)
    search_engine.db_context = test_db_context  # Use the same test database

    # 8. Index the content for search
    indexing_success = search_engine.index_content(result)
    assert indexing_success is True, "Content indexing should succeed"

    # 9. Perform a search query
    search_query = {
        "original_text": "calculus",  # Should match the test video content
        "filters": {},
        "pagination": {
            "offset": 0,
            "limit": 10
        }
    }

    search_results = search_engine.search(search_query)

    # 10. Verify search results
    assert "results" in search_results, "Search results should contain 'results' field"
    assert "totalResults" in search_results, "Search results should contain 'totalResults' field"

    # 11. Test retrieving a processed video
    processed_result = pipeline.get_processed_result(video_id)
    assert processed_result is not None, "Processed result retrieval should not return None"
    assert processed_result["video_id"] == video_id, f"Expected video_id {video_id}, got {processed_result.get('video_id')}"

    # 12. Test retrieving video concepts
    video_concepts = search_engine.get_video_concepts(video_id)
    assert video_concepts is not None, "Video concepts retrieval should not return None"

    # 13. Test concept details retrieval (if there are concepts)
    concept_id = None
    if "key_concepts" in result["domain_features"] and result["domain_features"]["key_concepts"]:
        # Find first concept with an ID
        for concept in result["domain_features"]["key_concepts"]:
            if "concept_id" in concept:
                concept_id = concept["concept_id"]
                break

    if concept_id:
        concept_details = search_engine.get_concept_details(concept_id)
        assert concept_details is not None, "Concept details retrieval should not return None"

    # 14. Test learning path generation (if there are concepts)
    concept_ids = []
    if "key_concepts" in result["domain_features"]:
        for concept in result["domain_features"]["key_concepts"][:3]:
            if "concept_id" in concept:
                concept_ids.append(concept["concept_id"])

    if concept_ids:
        learning_path = search_engine.generate_learning_path(
            concept_ids,
            theory_practice_ratio=0.5,
            domain=result["metadata"]["domain"]
        )
        # Just verify the call doesn't error - learning path might be None if not enough concepts

    # 15. Test batch processing
    batch_results = pipeline.batch_process_videos(
        ["https://www.youtube.com/watch?v=another123"]
    )
    assert batch_results is not None, "Batch processing should not return None"
    assert len(batch_results) == 1, f"Expected 1 batch result, got {len(batch_results)}"
