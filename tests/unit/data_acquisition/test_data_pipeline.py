"""
Integration tests for the Data Pipeline component.
"""

import pytest
import json
import os
import shutil
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime

from data_acquisition.youtube_api.python.data_pipeline import DataPipeline

# Test configuration
TEST_CONFIG = {
    "youtube_api_key": "test_api_key",
    "output_dir": "test_output"
}

TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TEST_VIDEO_ID = "dQw4w9WgXcQ"

# Mock data for YouTube and processing components
MOCK_METADATA = {
    "video_id": TEST_VIDEO_ID,
    "title": "Test Video",
    "channel": "Test Channel",
    "publication_date": "2023-01-01T00:00:00Z",
    "duration_seconds": 300,
    "description": "Test description",
    "language": "en",
    "domain": "mathematics",
    "domain_confidence": 0.8
}

MOCK_TRANSCRIPT = [
    {
        "start": 0.0,
        "duration": 5.0,
        "text": "Welcome to this lecture on calculus.",
        "language": "en"
    },
    {
        "start": 5.0,
        "duration": 5.0,
        "text": "Today we'll discuss derivatives.",
        "language": "en"
    }
]

MOCK_PROCESSED_TRANSCRIPT = {
    "segments": [
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
            "text": "Today we'll discuss derivatives.",
            "content_type": "theoretical"
        }
    ],
    "sentences": [],
    "sections": [],
    "language": "en",
    "domain": "mathematics",
    "video_id": TEST_VIDEO_ID
}

MOCK_DOMAIN_FEATURES = {
    "domain": "mathematics",
    "theoretical_segments": 2,
    "practical_segments": 0,
    "key_concepts": [
        {
            "text": "calculus",
            "domain": "mathematics",
            "frequency": 1,
            "theoretical": True
        },
        {
            "text": "derivatives",
            "domain": "mathematics",
            "frequency": 1,
            "theoretical": True
        }
    ]
}

MOCK_THEORY_PRACTICE_RESULTS = {
    "classification": "theoretical",
    "confidence": 0.8,
    "theoretical_segments": 2,
    "practical_segments": 0,
    "mixed_segments": 0,
    "theory_practice_ratio": 0.9
}

MOCK_THEORY_PRACTICE_PATTERNS = {
    "theory_to_practice_sequences": [],
    "practice_to_theory_sequences": [],
    "theory_practice_alternations": 0,
    "max_theory_sequence": 2,
    "max_practice_sequence": 0
}

@pytest.fixture
def data_pipeline():
    """Create a Data Pipeline instance with mocked components."""
    # Create test output directory
    os.makedirs(TEST_CONFIG["output_dir"], exist_ok=True)

    # Create pipeline with mocks
    with patch('data_acquisition.youtube_api.python.youtube_data_extractor.YouTubeDataExtractor') as mock_extractor, \
         patch('data_acquisition.transcript_processor.python.transcript_processor.TranscriptProcessor') as mock_processor, \
         patch('concept_analysis.concept_extractor.python.domain_concept_extractor.DomainClassifier') as mock_domain, \
         patch('concept_analysis.relevance_analyzer.python.theory_practice_classifier.TheoryPracticeClassifier') as mock_tp:

        # Set up mock returns
        mock_extractor_instance = mock_extractor.return_value
        mock_extractor_instance.validate_video_url.return_value = (True, TEST_VIDEO_ID)
        mock_extractor_instance.extract_video_metadata.return_value = MOCK_METADATA
        mock_extractor_instance.extract_transcript.return_value = MOCK_TRANSCRIPT

        mock_processor_instance = mock_processor.return_value
        mock_processor_instance.process_transcript.return_value = MOCK_PROCESSED_TRANSCRIPT

        mock_domain_instance = mock_domain.return_value
        mock_domain_instance.classify_transcript.return_value = ("mathematics", 0.9)
        mock_domain_instance.extract_domain_specific_features.return_value = MOCK_DOMAIN_FEATURES

        mock_tp_instance = mock_tp.return_value
        mock_tp_instance.classify_transcript.return_value = MOCK_THEORY_PRACTICE_RESULTS
        mock_tp_instance.extract_theory_practice_patterns.return_value = MOCK_THEORY_PRACTICE_PATTERNS

        pipeline = DataPipeline(TEST_CONFIG)

        # Store mocks for assertions
        pipeline.mock_extractor = mock_extractor_instance
        pipeline.mock_processor = mock_processor_instance
        pipeline.mock_domain = mock_domain_instance
        pipeline.mock_tp = mock_tp_instance

        yield pipeline

        # Clean up test directory
        shutil.rmtree(TEST_CONFIG["output_dir"], ignore_errors=True)

class TestDataPipeline:
    """Test the Data Pipeline component."""

    def test_init_components(self, data_pipeline):
        """Test initializing pipeline components."""
        assert data_pipeline.youtube_extractor is not None
        assert data_pipeline.transcript_processor is not None
        assert data_pipeline.domain_classifier is not None
        assert data_pipeline.theory_practice_classifier is not None

    def test_process_video(self, data_pipeline):
        """Test processing a video through the pipeline."""
        # Mock uuid.uuid4 to return a predictable value
        with patch('uuid.uuid4', return_value='test-job-id'):
            # Process video
            result = data_pipeline.process_video(TEST_VIDEO_URL)

            # Check that all components were called
            data_pipeline.mock_extractor.validate_video_url.assert_called_once_with(TEST_VIDEO_URL)
            data_pipeline.mock_extractor.extract_video_metadata.assert_called_once_with(TEST_VIDEO_ID)
            data_pipeline.mock_extractor.extract_transcript.assert_called_once()
            data_pipeline.mock_processor.process_transcript.assert_called_once()
            data_pipeline.mock_domain.extract_domain_specific_features.assert_called_once()
            data_pipeline.mock_tp.classify_transcript.assert_called_once()
            data_pipeline.mock_tp.extract_theory_practice_patterns.assert_called_once()

            # Check result structure
            assert result["job_id"] == "test-job-id"
            assert result["video_id"] == TEST_VIDEO_ID
            assert result["video_url"] == TEST_VIDEO_URL
            assert result["metadata"] == MOCK_METADATA
            assert result["transcript"] == MOCK_PROCESSED_TRANSCRIPT
            assert result["domain_features"] == MOCK_DOMAIN_FEATURES
            assert result["theory_practice_results"] == MOCK_THEORY_PRACTICE_RESULTS
            assert result["theory_practice_patterns"] == MOCK_THEORY_PRACTICE_PATTERNS
            assert result["status"] == "completed"

            # Check that result was saved to file
            expected_file = os.path.join(TEST_CONFIG["output_dir"], f"{TEST_VIDEO_ID}_test-job-id.json")
            assert os.path.exists(expected_file)

    def test_process_video_invalid_url(self, data_pipeline):
        """Test processing a video with an invalid URL."""
        # Set up mock to return invalid URL
        data_pipeline.mock_extractor.validate_video_url.return_value = (False, None)

        # Mock uuid.uuid4 to return a predictable value
        with patch('uuid.uuid4', return_value='test-job-id'):
            # Process video (should raise ValueError)
            with pytest.raises(ValueError) as excinfo:
                data_pipeline.process_video("https://invalid-url.com")

            assert "Invalid YouTube URL" in str(excinfo.value)

    def test_process_video_no_metadata(self, data_pipeline):
        """Test processing a video with no metadata."""
        # Set up mock to return empty metadata
        data_pipeline.mock_extractor.extract_video_metadata.return_value = {}

        # Mock uuid.uuid4 to return a predictable value
        with patch('uuid.uuid4', return_value='test-job-id'):
            # Process video (should raise ValueError)
            with pytest.raises(ValueError) as excinfo:
                data_pipeline.process_video(TEST_VIDEO_URL)

            assert "Failed to extract metadata" in str(excinfo.value)

    def test_process_video_no_transcript(self, data_pipeline):
        """Test processing a video with no transcript."""
        # Set up mock to return empty transcript
        data_pipeline.mock_extractor.extract_transcript.return_value = []

        # Mock uuid.uuid4 to return a predictable value
        with patch('uuid.uuid4', return_value='test-job-id'):
            # Process video (should raise ValueError)
            with pytest.raises(ValueError) as excinfo:
                data_pipeline.process_video(TEST_VIDEO_URL)

            assert "Failed to extract transcript" in str(excinfo.value)

    def test_process_video_error_handling(self, data_pipeline):
        """Test error handling during video processing."""
        # Set up mock to raise an exception
        data_pipeline.mock_processor.process_transcript.side_effect = Exception("Processing error")

        # Mock uuid.uuid4 to return a predictable value
        with patch('uuid.uuid4', return_value='test-job-id'):
            # Process video
            result = data_pipeline.process_video(TEST_VIDEO_URL)

            # Check error result
            assert result["job_id"] == "test-job-id"
            assert result["video_url"] == TEST_VIDEO_URL
            assert result["video_id"] == TEST_VIDEO_ID
            assert result["status"] == "error"
            assert "Processing error" in result["error"]

            # Check that error result was saved to file
            expected_file = os.path.join(TEST_CONFIG["output_dir"], f"{TEST_VIDEO_ID}_test-job-id.json")
            assert os.path.exists(expected_file)

    def test_batch_process_videos(self, data_pipeline):
        """Test batch processing multiple videos."""
        # Set up mock uuid.uuid4 to return predictable values
        with patch('uuid.uuid4', side_effect=['job1', 'job2']):
            # Batch process videos
            results = data_pipeline.batch_process_videos([TEST_VIDEO_URL, TEST_VIDEO_URL])

            # Check results
            assert len(results) == 2
            assert results[0]["job_id"] == "job1"
            assert results[1]["job_id"] == "job2"
            assert results[0]["status"] == "completed"
            assert results[1]["status"] == "completed"

    def test_batch_process_with_errors(self, data_pipeline):
        """Test batch processing with some errors."""
        # Set up first call to succeed, second call to fail
        data_pipeline.mock_extractor.validate_video_url.side_effect = [(True, TEST_VIDEO_ID), (False, None)]

        # Set up mock uuid.uuid4 to return predictable values
        with patch('uuid.uuid4', return_value='job1'):
            # Batch process videos
            results = data_pipeline.batch_process_videos([TEST_VIDEO_URL, "https://invalid-url.com"])

            # Check results
            assert len(results) == 2
            assert results[0]["status"] == "completed"
            assert results[1]["status"] == "error"
            assert "Invalid YouTube URL" in results[1]["error"]

    def test_get_processed_result(self, data_pipeline):
        """Test retrieving a previously processed result."""
        # First process a video to create a result file
        with patch('uuid.uuid4', return_value='test-job-id'):
            data_pipeline.process_video(TEST_VIDEO_URL)

        # Now retrieve the result
        result = data_pipeline.get_processed_result(TEST_VIDEO_ID)

        # Check result
        assert result is not None
        assert result["job_id"] == "test-job-id"
        assert result["video_id"] == TEST_VIDEO_ID
        assert result["status"] == "completed"

    def test_get_processed_result_not_found(self, data_pipeline):
        """Test retrieving a result that doesn't exist."""
        result = data_pipeline.get_processed_result("nonexistent-id")

        assert result is None
