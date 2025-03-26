import pytest
import os
import json
import unittest.mock as mock
from data_acquisition.youtube_api.python.data_pipeline import DataPipeline
from common.utils.error_handling import YouTubeAPIError, TranscriptExtractionError

class TestDataPipeline:
    """Tests for DataPipeline class."""

    @pytest.fixture
    def test_config(self):
        """Sample configuration for testing."""
        return {
            "youtube_api_key": "test_api_key",
            "output_dir": "test_output",
            "task_dir": "test_tasks",
            "result_dir": "test_results"
        }

    @pytest.fixture
    def test_pipeline(self, test_config):
        """Create a DataPipeline instance for testing."""
        # Create test directories if they don't exist
        os.makedirs(test_config["output_dir"], exist_ok=True)
        os.makedirs(test_config["task_dir"], exist_ok=True)
        os.makedirs(test_config["result_dir"], exist_ok=True)

        pipeline = DataPipeline(test_config)

        # Use mock components for testing
        pipeline.mock_extractor = mock.MagicMock()
        pipeline.mock_processor = mock.MagicMock()
        pipeline.mock_domain = mock.MagicMock()
        pipeline.mock_tp = mock.MagicMock()

        return pipeline

    @pytest.fixture(autouse=True)
    def cleanup(self, test_config):
        """Clean up test directories after tests."""
        yield
        # Remove test files
        for dir_name in ["output_dir", "task_dir", "result_dir"]:
            dir_path = test_config[dir_name]
            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

    def test_process_video(self, test_pipeline):
        """Test end-to-end video processing with mock components."""
        # Set up mock returns
        test_pipeline.mock_extractor.validate_video_url.return_value = (True, "test_video_id")
        test_pipeline.mock_extractor.extract_video_metadata.return_value = {
            "video_id": "test_video_id",
            "title": "Test Video",
            "description": "A test video about mathematics",
            "domain": "mathematics",
            "domain_confidence": 0.9
        }
        test_pipeline.mock_extractor.extract_transcript.return_value = [
            {
                "start": 0.0,
                "duration": 5.0,
                "text": "Welcome to the mathematics lecture.",
                "language": "en"
            },
            {
                "start": 5.0,
                "duration": 5.0,
                "text": "Today we will explore derivatives.",
                "language": "en"
            }
        ]
        test_pipeline.mock_processor.process_transcript.return_value = {
            "segments": [
                {
                    "id": "seg1",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "Welcome to the mathematics lecture.",
                    "content_type": "theoretical"
                },
                {
                    "id": "seg2",
                    "start_time": 5.0,
                    "end_time": 10.0,
                    "text": "Today we will explore derivatives.",
                    "content_type": "theoretical"
                }
            ],
            "sentences": [
                {"id": "sent1", "text": "Welcome to the mathematics lecture."},
                {"id": "sent2", "text": "Today we will explore derivatives."}
            ],
            "sections": [
                {"id": "section1", "title": None, "segments": ["seg1", "seg2"]}
            ],
            "language": "en",
            "domain": "mathematics",
            "video_id": "test_video_id"
        }
        test_pipeline.mock_domain.classify_transcript.return_value = ("mathematics", 0.9)
        test_pipeline.mock_domain.extract_domain_specific_features.return_value = {
            "domain": "mathematics",
            "theoretical_segments": 2,
            "practical_segments": 0,
            "key_concepts": [
                {"text": "derivative", "domain": "mathematics", "theoretical": True}
            ]
        }
        test_pipeline.mock_tp.classify_transcript.return_value = {
            "classification": "theoretical",
            "confidence": 0.9,
            "theoretical_segments": 2,
            "practical_segments": 0,
            "mixed_segments": 0,
            "theory_practice_ratio": 0.9
        }
        test_pipeline.mock_tp.extract_theory_practice_patterns.return_value = {
            "theory_to_practice_sequences": [],
            "practice_to_theory_sequences": [],
            "theory_practice_alternations": 0
        }

        # Process a video
        result = test_pipeline.process_video("https://www.youtube.com/watch?v=test_id")

        # Check that processing completed successfully
        assert result["status"] == "completed"
        assert result["video_id"] == "test_video_id"
        assert "metadata" in result
        assert "transcript" in result
        assert "domain_features" in result
        assert "theory_practice_results" in result
        assert "theory_practice_patterns" in result

        # Check that metadata was extracted correctly
        assert result["metadata"]["title"] == "Test Video"
        assert result["metadata"]["domain"] == "mathematics"

        # Check that transcript was processed correctly
        assert len(result["transcript"]["segments"]) == 2
        assert result["transcript"]["segments"][0]["content_type"] == "theoretical"

        # Check that domain features were extracted correctly
        assert result["domain_features"]["domain"] == "mathematics"
        assert len(result["domain_features"]["key_concepts"]) == 1
        assert result["domain_features"]["key_concepts"][0]["text"] == "derivative"

        # Check that theory/practice results were calculated correctly
        assert result["theory_practice_results"]["classification"] == "theoretical"
        assert result["theory_practice_results"]["theory_practice_ratio"] == 0.9

    def test_batch_process_videos(self, test_pipeline):
        """Test batch processing of multiple videos."""
        # Set up mock returns same as in test_process_video
        test_pipeline.mock_extractor.validate_video_url.return_value = (True, "test_video_id")
        test_pipeline.mock_extractor.extract_video_metadata.return_value = {
            "video_id": "test_video_id",
            "title": "Test Video",
            "domain": "mathematics"
        }
        test_pipeline.mock_extractor.extract_transcript.return_value = [
            {"text": "Test transcript"}
        ]
        test_pipeline.mock_processor.process_transcript.return_value = {
            "segments": [{"text": "Test segment"}],
            "sentences": [{"text": "Test sentence"}],
            "sections": [{"id": "section1"}],
            "domain": "mathematics",
            "language": "en",
            "video_id": "test_video_id"
        }
        test_pipeline.mock_domain.classify_transcript.return_value = ("mathematics", 0.8)
        test_pipeline.mock_domain.extract_domain_specific_features.return_value = {
            "key_concepts": [{"text": "calculus"}]
        }
        test_pipeline.mock_tp.classify_transcript.return_value = {
            "classification": "theoretical",
            "confidence": 0.7,
            "theory_practice_ratio": 0.8
        }
        test_pipeline.mock_tp.extract_theory_practice_patterns.return_value = {
            "theory_to_practice_sequences": []
        }

        # Process two videos in batch
        urls = [
            "https://www.youtube.com/watch?v=test_id1",
            "https://www.youtube.com/watch?v=test_id2"
        ]

        results = test_pipeline.batch_process_videos(urls)

        # Check that both videos were processed
        assert len(results) == 2
        for result in results:
            assert result["status"] == "completed"
            assert "video_id" in result
            assert "metadata" in result
            assert "transcript" in result

    def test_error_handling(self, test_pipeline):
        """Test handling of errors during processing."""
        # Set up the mock to validate URL but raise error on metadata extraction
        test_pipeline.mock_extractor.validate_video_url.return_value = (True, "error_test_id")
        test_pipeline.mock_extractor.extract_video_metadata.side_effect = Exception("API error")

        # Processing should complete but with error status
        result = test_pipeline.process_video("https://www.youtube.com/watch?v=error_test")

        assert result["status"] == "error"
        assert "error" in result
        assert "API error" in result["error"]

    def test_video_url_validation(self, test_pipeline):
        """Test validation of video URLs."""
        # Set up mock for valid URL
        test_pipeline.mock_extractor.validate_video_url.side_effect = [
            (True, "valid_id"),  # First call returns valid
            (False, None)        # Second call returns invalid
        ]

        # For the valid URL, set up the rest of the mocks
        test_pipeline.mock_extractor.extract_video_metadata.return_value = {
            "video_id": "valid_id",
            "title": "Valid Video",
            "domain": "mathematics"
        }
        test_pipeline.mock_extractor.extract_transcript.return_value = [
            {"text": "Valid transcript"}
        ]
        test_pipeline.mock_processor.process_transcript.return_value = {
            "segments": [{"text": "Valid segment"}],
            "sentences": [{"text": "Valid sentence"}],
            "sections": [{"id": "section1"}],
            "domain": "mathematics",
            "language": "en",
            "video_id": "valid_id"
        }
        test_pipeline.mock_domain.classify_transcript.return_value = ("mathematics", 0.8)
        test_pipeline.mock_domain.extract_domain_specific_features.return_value = {
            "key_concepts": [{"text": "calculus"}]
        }
        test_pipeline.mock_tp.classify_transcript.return_value = {
            "classification": "theoretical",
            "confidence": 0.7,
            "theory_practice_ratio": 0.8
        }
        test_pipeline.mock_tp.extract_theory_practice_patterns.return_value = {
            "theory_to_practice_sequences": []
        }

        # Valid URL
        result = test_pipeline.process_video("https://www.youtube.com/watch?v=valid_id")
        assert "status" in result
        assert result["status"] == "completed"

        # Invalid URL should raise ValueError
        with pytest.raises(ValueError):
            test_pipeline.process_video("https://example.com/not_youtube")

    def test_get_processed_result(self, test_pipeline, test_config):
        """Test retrieving previously processed results."""
        # Create a mock result file
        result_data = {
            "video_id": "test_result_id",
            "status": "completed",
            "metadata": {"title": "Test Result"},
            "transcript": {"segments": [{"text": "Test segment"}]}
        }

        # Save to file
        result_path = os.path.join(test_config["result_dir"], "test_result_id_123.json")
        with open(result_path, 'w') as f:
            json.dump(result_data, f)

        # Mock the _get_processed_result_from_file method to return our result
        test_pipeline._get_processed_result_from_file = mock.MagicMock(return_value=result_data)

        # Retrieve the result
        result = test_pipeline.get_processed_result("test_result_id")

        # Check that result was retrieved
        assert result is not None
        assert result["video_id"] == "test_result_id"
        assert result["status"] == "completed"
        assert "metadata" in result
        assert "transcript" in result

    def test_processing_steps(self, test_pipeline):
        """Test individual processing steps."""
        # Set up mock returns for all steps
        test_pipeline.mock_extractor.validate_video_url.return_value = (True, "step_test_id")
        test_pipeline.mock_extractor.extract_video_metadata.return_value = {
            "video_id": "step_test_id",
            "title": "Step Test Video",
            "domain": "unknown",
            "domain_confidence": 0.0
        }
        test_pipeline.mock_extractor.extract_transcript.return_value = [
            {"text": "Step test transcript"}
        ]
        test_pipeline.mock_processor.process_transcript.return_value = {
            "segments": [{"text": "Step test segment"}],
            "sentences": [{"text": "Step test sentence"}],
            "sections": [{"id": "section1"}],
            "domain": "unknown",
            "language": "en",
            "video_id": "step_test_id"
        }
        test_pipeline.mock_domain.classify_transcript.return_value = ("mathematics", 0.8)
        test_pipeline.mock_domain.extract_domain_specific_features.return_value = {
            "key_concepts": [{"text": "calculus"}]
        }
        test_pipeline.mock_tp.classify_transcript.return_value = {
            "classification": "theoretical",
            "confidence": 0.7,
            "theory_practice_ratio": 0.8
        }
        test_pipeline.mock_tp.extract_theory_practice_patterns.return_value = {
            "theory_to_practice_sequences": []
        }

        # Start processing a video
        result = test_pipeline.process_video("https://www.youtube.com/watch?v=step_test_id")

        # Check that all steps were called
        test_pipeline.mock_extractor.validate_video_url.assert_called_once()
        test_pipeline.mock_extractor.extract_video_metadata.assert_called_once()
        test_pipeline.mock_extractor.extract_transcript.assert_called_once()
        test_pipeline.mock_processor.process_transcript.assert_called_once()
        test_pipeline.mock_domain.classify_transcript.assert_called_once()
        test_pipeline.mock_domain.extract_domain_specific_features.assert_called_once()
        test_pipeline.mock_tp.classify_transcript.assert_called_once()
        test_pipeline.mock_tp.extract_theory_practice_patterns.assert_called_once()

        # Check that processing completed successfully
        assert result["status"] == "completed"

        # Check that domain was updated after classification
        assert result["metadata"]["domain"] == "mathematics"
        assert result["metadata"]["domain_confidence"] == 0.8
