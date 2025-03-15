"""
Integration test for the full Lecture Video Content Indexer workflow.
"""

import pytest
import os
import shutil
import json
import time
from unittest.mock import patch, MagicMock

from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor
from data_acquisition.transcript_processor.python.transcript_processor import TranscriptProcessor
from concept_analysis.concept_extractor.python.domain_concept_extractor import DomainClassifier
from concept_analysis.relevance_analyzer.python.theory_practice_classifier import TheoryPracticeClassifier
from data_acquisition.youtube_api.python.data_pipeline import DataPipeline
from search_retrieval.search_engine.python.search_engine import SearchEngine

# Test configuration
TEST_CONFIG = {
    "youtube_api_key": "test_api_key",
    "output_dir": "test_output",
    "index_dir": "test_index"
}

TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TEST_VIDEO_ID = "dQw4w9WgXcQ"

# Mock responses
MOCK_VIDEO_RESPONSE = {
    "items": [{
        "id": TEST_VIDEO_ID,
        "snippet": {
            "title": "Introduction to Calculus",
            "channelTitle": "Math Channel",
            "publishedAt": "2023-01-01T00:00:00Z",
            "description": "Learn about the basics of calculus. Course: Advanced Calculus. Instructor: Dr. Math",
            "tags": ["mathematics", "calculus", "derivatives"],
            "categoryId": "27",  # Education
            "defaultLanguage": "en"
        },
        "contentDetails": {
            "duration": "PT10M30S"  # 10 minutes, 30 seconds
        },
        "statistics": {
            "viewCount": "1000"
        }
    }]
}

MOCK_TRANSCRIPT = [
    {
        "text": "Welcome to this calculus lecture.",
        "start": 0.0,
        "duration": 5.0
    },
    {
        "text": "Today we'll discuss derivatives. A derivative is defined as the limit of the difference quotient.",
        "start": 5.0,
        "duration": 10.0
    },
    {
        "text": "Let's solve this problem: Find the derivative of f(x) = x^2.",
        "start": 15.0,
        "duration": 8.0
    },
    {
        "text": "Using the definition, we find that f'(x) = 2x.",
        "start": 23.0,
        "duration": 7.0
    }
]

class MockTranscriptList:
    def find_manually_created_transcript(self, lang_codes):
        if "en" in lang_codes:
            return MockTranscript()
        raise Exception("No transcript found")

    def find_generated_transcript(self, lang_codes):
        if "en" in lang_codes:
            return MockTranscript()
        raise Exception("No transcript found")

    def find_transcript(self, lang_codes):
        if "en" in lang_codes:
            return MockTranscript()
        raise Exception("No transcript found")

class MockTranscript:
    def fetch(self):
        return MOCK_TRANSCRIPT

@pytest.fixture
def test_environment():
    """Create a test environment with all necessary directories."""
    # Create test directories
    os.makedirs(TEST_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TEST_CONFIG["index_dir"], exist_ok=True)

    yield

    # Clean up test directories
    shutil.rmtree(TEST_CONFIG["output_dir"], ignore_errors=True)
    shutil.rmtree(TEST_CONFIG["index_dir"], ignore_errors=True)

@pytest.mark.integration
class TestFullWorkflow:
    """Integration test for the full workflow."""

    @patch('googleapiclient.discovery.build')
    @patch('youtube_transcript_api.YouTubeTranscriptApi.list_transcripts')
    @patch('spacy.load')
    def test_end_to_end_workflow(self, mock_spacy_load, mock_list_transcripts, mock_build, test_environment):
        """Test the full workflow from video URL to search results."""
        # Set up mocks for YouTube API
        mock_youtube = MagicMock()
        mock_videos = MagicMock()
        mock_list = MagicMock()
        mock_execute = MagicMock(return_value=MOCK_VIDEO_RESPONSE)

        mock_list.execute = mock_execute
        mock_videos.list.return_value = mock_list
        mock_youtube.videos.return_value = mock_videos
        mock_build.return_value = mock_youtube

        # Set up mock for transcript API
        mock_list_transcripts.return_value = MockTranscriptList()

        # Set up mock for spaCy
        mock_en_nlp = MagicMock()
        mock_ru_nlp = MagicMock()

        def mock_spacy_side_effect(model):
            if model == 'en_core_web_sm':
                return mock_en_nlp
            elif model == 'ru_core_news_sm':
                return mock_ru_nlp
            raise ValueError(f"No mock for {model}")

        mock_spacy_load.side_effect = mock_spacy_side_effect

        # Step 1: Create all pipeline components directly
        youtube_extractor = YouTubeDataExtractor(TEST_CONFIG["youtube_api_key"])
        transcript_processor = TranscriptProcessor()

        # Set NLP models directly since mocks are not loaded
        transcript_processor.en_nlp = mock_en_nlp
        transcript_processor.ru_nlp = mock_ru_nlp

        domain_classifier = DomainClassifier()
        theory_practice_classifier = TheoryPracticeClassifier()

        # Step 2: Create data pipeline and search engine
        pipeline_config = {**TEST_CONFIG}
        pipeline = DataPipeline(pipeline_config)

        # Replace pipeline components with our test instances
        pipeline.youtube_extractor = youtube_extractor
        pipeline.transcript_processor = transcript_processor
        pipeline.domain_classifier = domain_classifier
        pipeline.theory_practice_classifier = theory_practice_classifier

        search_engine = SearchEngine({"index_dir": TEST_CONFIG["index_dir"]})

        # Step 3: Process video
        result = pipeline.process_video(TEST_VIDEO_URL)

        # Check results
        assert result["video_id"] == TEST_VIDEO_ID
        assert result["status"] == "completed"
        assert result["metadata"]["title"] == "Introduction to Calculus"
        assert result["metadata"]["domain"] == "mathematics"
        assert "transcript" in result
        assert "domain_features" in result
        assert "theory_practice_results" in result
        assert "theory_practice_patterns" in result

        # Check file was saved
        output_files = os.listdir(TEST_CONFIG["output_dir"])
        assert len(output_files) > 0

        # Step 4: Index the content
        indexed = search_engine.index_content(result)
        assert indexed is True

        # Step 5: Search for concepts
        query = {
            "original_text": "derivative",
            "filters": {},
            "theory_practice_ratio": 0.7,  # Favor theoretical content
            "domain": "mathematics",
            "pagination": {
                "offset": 0,
                "limit": 10
            }
        }

        search_results = search_engine.search(query)

        # Check search results
        assert search_results["totalResults"] > 0
        assert len(search_results["results"]) > 0

        # The search should find "derivative" in the content
        found_derivative = False
        for res in search_results["results"]:
            if "derivative" in res.get("context_text", "").lower():
                found_derivative = True
                break

        assert found_derivative, "Could not find 'derivative' in search results"

        # Step 6: Test theory vs. practice filtering
        # Search for practical content
        practical_query = {
            "original_text": "problem",
            "filters": {},
            "theory_practice_ratio": 0.2,  # Favor practical content
            "domain": "mathematics",
            "pagination": {
                "offset": 0,
                "limit": 10
            }
        }

        practical_results = search_engine.search(practical_query)

        # Check that practical results focus on problem-solving
        assert practical_results["totalResults"] > 0

        # Check that the practical segment appears in results
        found_practical = False
        for res in practical_results["results"]:
            if "solve this problem" in res.get("context_text", "").lower():
                found_practical = True
                break

        assert found_practical, "Could not find practical content in results"

        # Step 7: Get video concepts
        video_concepts = search_engine.get_video_concepts(TEST_VIDEO_ID)

        assert video_concepts is not None
        assert "video" in video_concepts
        assert "concepts" in video_concepts
        assert len(video_concepts["concepts"]) > 0

        # Check that we have both theoretical and practical segments
        theoretical_concepts = search_engine.get_video_concepts(TEST_VIDEO_ID, "theoretical")
        practical_concepts = search_engine.get_video_concepts(TEST_VIDEO_ID, "practical")

        assert len(theoretical_concepts["concepts"]) > 0

        # We might not have enough data to guarantee practical concepts, but we should at least
        # have theoretical concepts

        # Step 8: Generate learning path
        # Get first concept ID
        first_concept = video_concepts["concepts"][0]
        learning_path = search_engine.generate_learning_path(
            [first_concept["concept_id"]],
            theory_practice_ratio=0.5,  # Balanced
            domain="mathematics"
        )

        assert learning_path is not None
        assert "concepts" in learning_path
        assert len(learning_path["concepts"]) > 0
        assert "theory_practice_ratio" in learning_path
        assert "domain" in learning_path
        assert learning_path["domain"] == "mathematics"
