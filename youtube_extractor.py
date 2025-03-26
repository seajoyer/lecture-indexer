"""
Simplified YouTube data extractor for the Lecture Video Content Indexer.
Handles extraction of video metadata and transcripts with minimal complexity.
"""

import re
import logging
import googleapiclient.discovery
import googleapiclient.errors
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import Dict, List, Tuple, Any, Optional

# Import simplified modules
from cache_manager import cache_get, cache_set
from performance_utils import time_function

# Configure logging
logger = logging.getLogger(__name__)

class YouTubeExtractor:
    """
    Extracts video metadata and transcripts from YouTube videos.
    Simplified version with reduced complexity and dependencies.
    """

    def __init__(self, api_key: str):
        """
        Initialize the YouTube extractor with API key.

        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key
        self._youtube = None
        logger.info("YouTubeExtractor initialized")

    @property
    def youtube(self):
        """Lazy initialization of YouTube API client."""
        if self._youtube is None:
            try:
                self._youtube = googleapiclient.discovery.build(
                    "youtube", "v3", developerKey=self.api_key, cache_discovery=False
                )
                logger.info("YouTube API client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API client: {e}")
                self._youtube = self._create_mock_client()
        return self._youtube

    def _create_mock_client(self):
        """Create a mock client for testing when API key is invalid."""
        logger.warning("Creating mock YouTube client - API calls will not work")
        mock = type('MockYouTube', (), {})()
        mock_videos = type('MockVideos', (), {})()
        mock_list = type('MockList', (), {})()

        def mock_execute():
            return {"items": []}

        mock_list.execute = mock_execute
        mock_videos.list = lambda **kwargs: mock_list
        mock.videos = lambda: mock_videos
        mock.playlistItems = lambda: mock_videos
        return mock

    def validate_video_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validates a YouTube URL and extracts the video ID.

        Args:
            url: YouTube video URL

        Returns:
            Tuple of (is_valid, video_id)
        """
        # Check cache first
        cache_key = f"url_validation_{url}"
        cached_result = cache_get("video", cache_key)
        if cached_result is not None:
            return cached_result

        # Regular expression patterns for different YouTube URL formats
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',  # Standard URL
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^\?\s]+)',  # Shortened URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\?\s]+)',  # Embedded URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^\?\s]+)',  # Old embed URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([^\?\s]+)'  # YouTube shorts URL
        ]

        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                video_id = match.group(1)
                result = (True, video_id)

                # Cache the result
                cache_set("video", cache_key, result)

                return result

        logger.warning(f"Invalid YouTube URL format: {url}")
        result = (False, None)

        # Cache the negative result too
        cache_set("video", cache_key, result)

        return result

    @time_function(2000)  # Log warning if takes more than 2 seconds
    def extract_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Extracts metadata for a YouTube video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary containing video metadata
        """
        logger.info(f"Extracting metadata for video: {video_id}")

        # Check cache first
        cache_key = f"video_metadata_{video_id}"
        cached_result = cache_get("video", cache_key)
        if cached_result:
            logger.info(f"Using cached metadata for video: {video_id}")
            return cached_result

        # For test mode, return mock data
        is_test_mode = self.api_key == "test_api_key" or not self.api_key
        if is_test_mode:
            logger.warning("Using test mode with mock data")
            mock_data = self._get_mock_metadata(video_id)
            cache_set("video", cache_key, mock_data)
            return mock_data

        try:
            # Request video details from YouTube API
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            response = request.execute()

            if not response.get('items'):
                logger.warning(f"No video found with ID: {video_id}")
                raise ValueError(f"No video found with ID: {video_id}")

            # Extract relevant information from response
            video_data = response['items'][0]
            snippet = video_data.get('snippet', {})
            statistics = video_data.get('statistics', {})
            content_details = video_data.get('contentDetails', {})

            # Format the duration (convert from ISO 8601 format)
            duration_str = content_details.get('duration', 'PT0S')
            duration_seconds = self._parse_duration(duration_str)

            # Detect language from metadata
            language = snippet.get('defaultLanguage', snippet.get('defaultAudioLanguage', ''))
            if language.startswith('ru'):
                language = 'ru'
            elif language.startswith('en'):
                language = 'en'
            else:
                language = ''  # Will be determined from transcript later

            # Perform initial domain classification based on title and description
            domain, domain_confidence = self._initial_domain_classification(
                snippet.get('title', ''),
                snippet.get('description', '')
            )

            # Create metadata object
            metadata = {
                "video_id": video_id,
                "title": snippet.get('title', ''),
                "channel": snippet.get('channelTitle', ''),
                "publication_date": snippet.get('publishedAt', ''),
                "duration_seconds": duration_seconds,
                "category": snippet.get('categoryId', ''),
                "tags": snippet.get('tags', []),
                "description": snippet.get('description', ''),
                "language": language,
                "view_count": int(statistics.get('viewCount', 0)),
                "domain": domain,
                "domain_confidence": domain_confidence
            }

            # Store in cache
            cache_set("video", cache_key, metadata)

            logger.info(f"Successfully extracted metadata for video: {video_id}")
            return metadata

        except googleapiclient.errors.HttpError as e:
            error_message = f"YouTube API error when extracting metadata: {e}"
            logger.error(error_message)

            if is_test_mode:
                mock_data = self._get_mock_metadata(video_id)
                return mock_data

            raise ValueError(error_message)

        except Exception as e:
            error_message = f"Unexpected error when extracting metadata: {e}"
            logger.error(error_message)

            if is_test_mode:
                mock_data = self._get_mock_metadata(video_id)
                return mock_data

            raise ValueError(error_message)

    def _get_mock_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Generate mock metadata for testing purposes.

        Args:
            video_id: YouTube video ID

        Returns:
            Mock metadata dictionary
        """
        logger.info(f"Generating mock metadata for video: {video_id}")
        return {
            "video_id": video_id,
            "title": "Test Video Title",
            "channel": "Test Channel",
            "publication_date": "2023-01-01T00:00:00Z",
            "duration_seconds": 630,  # 10m30s
            "category": "27",  # Education
            "tags": ["mathematics", "calculus", "derivatives"],
            "description": "This is a test video for a mathematics course.",
            "language": "en",
            "view_count": 1000,
            "domain": "mathematics",
            "domain_confidence": 0.8
        }

    @time_function(3000)  # Log warning if takes more than 3 seconds
    def extract_transcript(self, video_id: str, language_preference: List[str] = ['en', 'ru']) -> List[Dict]:
        """
        Extracts transcript for a YouTube video with preference for specified languages.

        Args:
            video_id: YouTube video ID
            language_preference: List of language codes in order of preference

        Returns:
            List of transcript segments
        """
        logger.info(f"Extracting transcript for video: {video_id}")

        # Check cache first
        cache_key = f"video_transcript_{video_id}_{'-'.join(language_preference)}"
        cached_result = cache_get("transcript", cache_key)
        if cached_result:
            logger.info(f"Using cached transcript for video: {video_id}")
            return cached_result

        # Only use mock data for explicit test mode
        is_test_mode = self.api_key == "test_api_key" or not self.api_key
        if is_test_mode:
            logger.warning("Using mock transcript data for testing")
            mock_transcript = self._get_mock_transcript()
            cache_set("transcript", cache_key, mock_transcript)
            return mock_transcript

        try:
            # Get available transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get manually created transcript in preferred languages
            for lang in language_preference:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    logger.info(f"Found manually created transcript in {lang}")
                    transcript_data = self._format_transcript(transcript.fetch(), lang)
                    cache_set("transcript", cache_key, transcript_data)
                    return transcript_data
                except Exception:
                    logger.debug(f"No manually created transcript in {lang}")

            # Try to get generated transcript in preferred languages
            for lang in language_preference:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    logger.info(f"Found generated transcript in {lang}")
                    transcript_data = self._format_transcript(transcript.fetch(), lang)
                    cache_set("transcript", cache_key, transcript_data)
                    return transcript_data
                except Exception:
                    logger.debug(f"No generated transcript in {lang}")

            # If no preferred language transcript is found, get the default one
            try:
                default_transcript = transcript_list.find_transcript(['en'])
                logger.info(f"Using default transcript")
                transcript_data = self._format_transcript(default_transcript.fetch(), 'en')
                cache_set("transcript", cache_key, transcript_data)
                return transcript_data
            except:
                # Try one more fallback: use any available transcript
                available_langs = transcript_list._manually_created_transcripts.keys()
                if available_langs:
                    lang = list(available_langs)[0]
                    transcript = transcript_list._manually_created_transcripts[lang]
                    logger.info(f"Found fallback transcript in {lang}")
                    transcript_data = self._format_transcript(transcript.fetch(), lang)
                    cache_set("transcript", cache_key, transcript_data)
                    return transcript_data

                raise ValueError(f"No transcript found for video: {video_id}")

        except TranscriptsDisabled:
            error_message = f"Transcripts are disabled for video: {video_id}"
            logger.warning(error_message)
            if is_test_mode:
                return self._get_mock_transcript()
            raise ValueError(error_message)

        except NoTranscriptFound:
            error_message = f"No transcript found for video: {video_id}"
            logger.warning(error_message)
            if is_test_mode:
                return self._get_mock_transcript()
            raise ValueError(error_message)

        except Exception as e:
            error_message = f"Error extracting transcript: {e}"
            logger.error(error_message)
            if is_test_mode:
                return self._get_mock_transcript()
            raise ValueError(error_message)

    def _get_mock_transcript(self) -> List[Dict]:
        """
        Generate a mock transcript for testing purposes.

        Returns:
            List of transcript segments
        """
        logger.info("Generating mock transcript data")
        return [
            {
                "start": 0.0,
                "duration": 5.0,
                "text": "Welcome to this calculus lecture.",
                "language": "en"
            },
            {
                "start": 5.0,
                "duration": 10.0,
                "text": "Today we'll discuss derivatives and their applications.",
                "language": "en"
            },
            {
                "start": 15.0,
                "duration": 8.0,
                "text": "Let's start with the definition of a derivative.",
                "language": "en"
            },
            {
                "start": 23.0,
                "duration": 12.0,
                "text": "A derivative is defined as the limit of the difference quotient as the interval approaches zero.",
                "language": "en"
            },
            {
                "start": 35.0,
                "duration": 10.0,
                "text": "Now let's solve a problem. Find the derivative of f(x) = x^2.",
                "language": "en"
            }
        ]

    def _parse_duration(self, duration_str: str) -> int:
        """
        Parses ISO 8601 duration string to seconds.

        Args:
            duration_str: Duration string in ISO 8601 format (e.g., 'PT1H30M15S')

        Returns:
            Duration in seconds
        """
        hours = 0
        minutes = 0
        seconds = 0

        # Extract hours
        hour_match = re.search(r'(\d+)H', duration_str)
        if hour_match:
            hours = int(hour_match.group(1))

        # Extract minutes
        minute_match = re.search(r'(\d+)M', duration_str)
        if minute_match:
            minutes = int(minute_match.group(1))

        # Extract seconds
        second_match = re.search(r'(\d+)S', duration_str)
        if second_match:
            seconds = int(second_match.group(1))

        return hours * 3600 + minutes * 60 + seconds

    def _initial_domain_classification(self, title: str, description: str) -> Tuple[str, float]:
        """
        Simple domain classification based on keywords in title and description.

        Args:
            title: Video title
            description: Video description

        Returns:
            Tuple of (domain, confidence)
        """
        # Combine text for analysis
        combined_text = f"{title} {description}".lower()

        # Define domain-specific keywords
        math_keywords = [
            'math', 'mathematics', 'calculus', 'algebra', 'geometry', 'theorem', 'proof',
            'equation', 'function', 'derivative', 'integral'
        ]

        programming_keywords = [
            'programming', 'algorithm', 'code', 'software', 'development', 'computer science',
            'python', 'java', 'c++', 'javascript', 'data structure'
        ]

        physics_keywords = [
            'physics', 'mechanics', 'dynamics', 'kinematics', 'electromagnetism',
            'thermodynamics', 'quantum', 'relativity', 'force', 'energy'
        ]

        # Count keyword matches for each domain
        math_count = sum(1 for keyword in math_keywords if keyword in combined_text)
        programming_count = sum(1 for keyword in programming_keywords if keyword in combined_text)
        physics_count = sum(1 for keyword in physics_keywords if keyword in combined_text)

        # Get the domain with highest count
        counts = {
            'mathematics': math_count,
            'programming': programming_count,
            'physics': physics_count
        }

        max_count = max(counts.values())
        if max_count == 0:
            return ('unknown', 0.0)

        # Get domain with highest count
        max_domains = [domain for domain, count in counts.items() if count == max_count]
        if len(max_domains) == 1:
            domain = max_domains[0]
            total_count = sum(counts.values())
            confidence = max_count / total_count if total_count > 0 else 0.0
            return (domain, confidence)
        else:
            # If tie, return the first domain with medium confidence
            return (max_domains[0], 0.5)

    def _format_transcript(self, transcript_data: List[Dict], language: str) -> List[Dict]:
        """
        Formats transcript data into a standardized structure.

        Args:
            transcript_data: Raw transcript data from YouTube API
            language: Detected language of transcript

        Returns:
            List of formatted transcript segments
        """
        formatted_transcript = []

        for segment in transcript_data:
            formatted_segment = {
                "start": segment.get('start', 0.0),
                "duration": segment.get('duration', 0.0),
                "text": segment.get('text', ''),
                "language": language
            }
            formatted_transcript.append(formatted_segment)

        return formatted_transcript
