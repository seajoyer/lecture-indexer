import pytest
import unittest.mock as mock
from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor
from common.utils.error_handling import YouTubeAPIError, TranscriptExtractionError

class TestYouTubeDataExtractor:
    """Tests for YouTubeDataExtractor class."""

    def test_validate_video_url(self, mock_youtube_extractor):
        """Test URL validation with various YouTube URL formats."""
        # Regular URL format
        valid, video_id = mock_youtube_extractor.validate_video_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert valid is True
        assert video_id == "dQw4w9WgXcQ"

        # Short URL format
        valid, video_id = mock_youtube_extractor.validate_video_url("https://youtu.be/dQw4w9WgXcQ")
        assert valid is True
        assert video_id == "dQw4w9WgXcQ"

        # Embed URL format
        valid, video_id = mock_youtube_extractor.validate_video_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert valid is True
        assert video_id == "dQw4w9WgXcQ"

        # Old embed URL format
        valid, video_id = mock_youtube_extractor.validate_video_url("https://www.youtube.com/v/dQw4w9WgXcQ")
        assert valid is True
        assert video_id == "dQw4w9WgXcQ"

        # YouTube shorts URL format
        valid, video_id = mock_youtube_extractor.validate_video_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        assert valid is True
        assert video_id == "dQw4w9WgXcQ"

        # Invalid URL
        valid, video_id = mock_youtube_extractor.validate_video_url("https://example.com/video")
        assert valid is False
        assert video_id is None

    def test_extract_video_metadata(self, mock_youtube_extractor):
        """Test extracting metadata with mock API response."""
        metadata = mock_youtube_extractor.extract_video_metadata("dQw4w9WgXcQ")

        # Check basic metadata structure
        assert metadata is not None
        assert "video_id" in metadata
        assert metadata["video_id"] == "dQw4w9WgXcQ"
        assert "title" in metadata
        assert "channel" in metadata
        assert "domain" in metadata
        assert "domain_confidence" in metadata
        assert "educational_metadata" in metadata

        # Check educational metadata
        edu_meta = metadata["educational_metadata"]
        assert "course_name" in edu_meta
        assert "instructor" in edu_meta
        assert "institution" in edu_meta

    def test_extract_transcript(self, mock_youtube_extractor):
        """Test extracting transcript with mock response."""
        transcript = mock_youtube_extractor.extract_transcript("dQw4w9WgXcQ")

        # Check basic transcript structure
        assert transcript is not None
        assert len(transcript) == 2  # The test mock returns exactly 2 segments

        # Check segment structure
        assert "start" in transcript[0]
        assert "duration" in transcript[0]
        assert "text" in transcript[0]
        assert "language" in transcript[0]

        # Check first segment content
        assert transcript[0]["text"] == "Welcome to the mathematics lecture."

    def test_detect_language(self, mock_youtube_extractor):
        """Test language detection from transcript."""
        # English transcript
        en_transcript = [
            {"text": "This is an English transcript."},
            {"text": "It contains English words and phrases."}
        ]
        language = mock_youtube_extractor.detect_language(en_transcript)
        assert language == "en"

        # Russian transcript
        ru_transcript = [
            {"text": "Это русская транскрипция."},
            {"text": "В ней содержатся русские слова и фразы."}
        ]
        language = mock_youtube_extractor.detect_language(ru_transcript)
        assert language == "ru"

        # Empty transcript should default to English
        empty_transcript = []
        language = mock_youtube_extractor.detect_language(empty_transcript)
        assert language == "en"

    def test_parse_duration(self, mock_youtube_extractor):
        """Test parsing of ISO 8601 duration strings."""
        # Access private method for testing
        duration = mock_youtube_extractor._parse_duration("PT1H30M15S")
        assert duration == 5415  # 1h 30m 15s = 5415 seconds

        duration = mock_youtube_extractor._parse_duration("PT30M")
        assert duration == 1800  # 30m = 1800 seconds

        duration = mock_youtube_extractor._parse_duration("PT1H")
        assert duration == 3600  # 1h = 3600 seconds

        duration = mock_youtube_extractor._parse_duration("PT45S")
        assert duration == 45  # 45s = 45 seconds

    def test_extract_educational_metadata(self, mock_youtube_extractor):
        """Test extraction of educational metadata from video description."""
        # Test English metadata extraction
        description = "Course: Advanced Calculus. Instructor: Dr. Smith. University: Example University."
        metadata = mock_youtube_extractor._extract_educational_metadata(description)

        assert metadata["course_name"] == "Advanced Calculus"
        assert metadata["instructor"] == "Dr. Smith"
        assert metadata["institution"] == "Example University"

        # Test Russian metadata extraction
        ru_description = "Курс: Высшая математика. Преподаватель: Проф. Иванов. Университет: МГУ."
        ru_metadata = mock_youtube_extractor._extract_educational_metadata(ru_description)

        assert ru_metadata["course_name"] == "Высшая математика"
        assert ru_metadata["instructor"] == "Проф. Иванов"
        assert ru_metadata["institution"] == "МГУ"

        # Test with missing metadata
        partial_description = "Just a simple video description with no educational metadata."
        partial_metadata = mock_youtube_extractor._extract_educational_metadata(partial_description)

        assert partial_metadata["course_name"] is None
        assert partial_metadata["instructor"] is None
        assert partial_metadata["institution"] is None

    def test_initial_domain_classification(self, mock_youtube_extractor):
        """Test initial domain classification from metadata."""
        # Mathematics content
        domain, confidence = mock_youtube_extractor._initial_domain_classification(
            "Introduction to Calculus",
            ["mathematics", "calculus", "derivatives"],
            "Learn about derivatives and integrals in this calculus course."
        )
        assert domain == "mathematics"
        assert confidence > 0.5

        # Programming content
        domain, confidence = mock_youtube_extractor._initial_domain_classification(
            "Python Programming Tutorial",
            ["python", "programming", "coding"],
            "Learn how to code in Python with this tutorial."
        )
        assert domain == "programming"
        assert confidence > 0.5

        # Physics content
        domain, confidence = mock_youtube_extractor._initial_domain_classification(
            "Introduction to Physics",
            ["physics", "mechanics", "newton"],
            "Learn about Newton's laws of motion in this physics lecture."
        )
        assert domain == "physics"
        assert confidence > 0.5

        # Unknown content
        domain, confidence = mock_youtube_extractor._initial_domain_classification(
            "Random Video Title",
            [],
            "This is just a random video with no specific domain."
        )
        assert domain == "unknown"
        assert confidence == 0.0

    @mock.patch('googleapiclient.discovery.build')
    def test_api_error_handling(self, mock_build, mock_youtube_extractor):
        """Test error handling for API failures."""
        # Force API to raise an error by mocking
        mock_youtube = mock.MagicMock()
        mock_request = mock.MagicMock()
        mock_execute = mock.MagicMock(side_effect=Exception("API error"))

        mock_request.execute = mock_execute
        mock_youtube.videos.return_value.list.return_value = mock_request
        mock_build.return_value = mock_youtube

        # Make extractor use our mock
        mock_youtube_extractor._youtube = mock_youtube

        # When using test API key, it should fall back to mock data instead of raising
        metadata = mock_youtube_extractor.extract_video_metadata("error_test")
        assert metadata is not None
        assert "title" in metadata
        assert metadata["title"] == "Test Video Title"
