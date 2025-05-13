"""
Enhanced YouTube data extractor for the Lecture Video Content Indexer.
Handles extraction of video metadata and transcripts with improved multilingual support.
Added support for playlist processing and more robust transcript extraction.
"""

import re
import logging
import googleapiclient.discovery
import googleapiclient.errors
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import Dict, List, Tuple, Any, Optional
import json
import unicodedata
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import requests
import time
import random

# Import simplified modules
from cache_manager import cache_get, cache_set
from performance_utils import time_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeExtractor:
    """
    Extracts video metadata and transcripts from YouTube videos.
    Enhanced version with improved multilingual support and language detection.
    Added support for playlist processing.
    """

    def __init__(self, api_key: str):
        """
        Initialize the YouTube extractor with API key.

        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key
        self._youtube = None

        # Initialize language detection
        self._init_language_detection()

        # Maximum retries for transcript fetching
        self.max_retries = 3
        self.retry_delay = 2  # seconds

        logger.info("YouTubeExtractor initialized with enhanced multilingual support and playlist handling")

    def _init_language_detection(self):
        """Initialize language detection capabilities."""
        try:
            # Set a fixed seed for consistent language detection
            langdetect.DetectorFactory.seed = 0

            # Initialize language metadata for supported languages
            self.language_metadata = {
                'en': {
                    'name': 'English',
                    'codes': ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],
                    'script': 'Latin'
                },
                'ru': {
                    'name': 'Russian',
                    'codes': ['ru', 'ru-RU'],
                    'script': 'Cyrillic'
                }
                # Add more languages as needed
            }

            logger.info("Language detection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize language detection: {e}")

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
        mock_playlists = type('MockPlaylists', (), {})()
        mock_list = type('MockList', (), {})()

        def mock_execute():
            return {"items": []}

        mock_list.execute = mock_execute
        mock_videos.list = lambda **kwargs: mock_list
        mock_playlists.list = lambda **kwargs: mock_list
        mock.videos = lambda: mock_videos
        mock.playlists = lambda: mock_playlists
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

    def validate_playlist_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validates a YouTube playlist URL and extracts the playlist ID.

        Args:
            url: YouTube playlist URL

        Returns:
            Tuple of (is_valid, playlist_id)
        """
        # Check cache first
        cache_key = f"playlist_validation_{url}"
        cached_result = cache_get("video", cache_key)
        if cached_result is not None:
            return cached_result

        # Regular expression patterns for different YouTube playlist URL formats
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/playlist\?list=([^&\s]+)',  # Standard playlist URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=[^&\s]+&list=([^&\s]+)'  # Video with playlist
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                playlist_id = match.group(1)
                result = (True, playlist_id)

                # Cache the result
                cache_set("video", cache_key, result)

                return result

        logger.warning(f"Invalid YouTube playlist URL format: {url}")
        result = (False, None)

        # Cache the negative result too
        cache_set("video", cache_key, result)

        return result

    @time_function(2000)  # Log warning if takes more than 2 seconds
    def extract_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Extracts metadata for a YouTube video with enhanced language detection.

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

            # Enhanced language detection from metadata
            language = self._detect_language_from_metadata(snippet)

            # Perform enhanced domain classification based on title and description
            domain, domain_confidence = self._enhanced_domain_classification(
                snippet.get('title', ''),
                snippet.get('description', ''),
                language
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

    def extract_playlist_videos(self, playlist_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Extract video IDs and basic metadata from a YouTube playlist.

        Args:
            playlist_id: YouTube playlist ID
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing video ID and basic metadata
        """
        logger.info(f"Extracting videos from playlist: {playlist_id}")

        # Check cache first
        cache_key = f"playlist_videos_{playlist_id}_{max_results}"
        cached_result = cache_get("video", cache_key)
        if cached_result:
            logger.info(f"Using cached playlist data for playlist: {playlist_id}")
            return cached_result

        # For test mode, return mock data
        is_test_mode = self.api_key == "test_api_key" or not self.api_key
        if is_test_mode:
            logger.warning("Using test mode with mock playlist data")
            mock_data = self._get_mock_playlist_videos(playlist_id)
            cache_set("video", cache_key, mock_data)
            return mock_data

        try:
            # Get playlist details first
            playlist_request = self.youtube.playlists().list(
                part="snippet",
                id=playlist_id
            )
            playlist_response = playlist_request.execute()

            if not playlist_response.get('items'):
                logger.warning(f"No playlist found with ID: {playlist_id}")
                raise ValueError(f"No playlist found with ID: {playlist_id}")

            playlist_title = playlist_response['items'][0]['snippet'].get('title', 'Unknown Playlist')
            playlist_channel = playlist_response['items'][0]['snippet'].get('channelTitle', 'Unknown Channel')

            # Request playlist items from YouTube API
            videos = []
            next_page_token = None

            while len(videos) < max_results:
                request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                )
                response = request.execute()

                if not response.get('items'):
                    break

                # Extract video data
                for item in response.get('items', []):
                    content_details = item.get('contentDetails', {})
                    snippet = item.get('snippet', {})

                    video_id = content_details.get('videoId')
                    if not video_id:
                        continue

                    # Get basic video details
                    video_data = {
                        "video_id": video_id,
                        "title": snippet.get('title', 'Unknown Title'),
                        "position": snippet.get('position', 0),
                        "thumbnail": snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
                        "description": snippet.get('description', ''),
                        "playlist_id": playlist_id,
                        "playlist_title": playlist_title,
                        "playlist_channel": playlist_channel
                    }

                    videos.append(video_data)

                # Check for next page
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

            # Store in cache
            cache_set("video", cache_key, videos)

            logger.info(f"Successfully extracted {len(videos)} videos from playlist: {playlist_id}")
            return videos

        except googleapiclient.errors.HttpError as e:
            error_message = f"YouTube API error when extracting playlist videos: {e}"
            logger.error(error_message)

            if is_test_mode:
                mock_data = self._get_mock_playlist_videos(playlist_id)
                return mock_data

            raise ValueError(error_message)

        except Exception as e:
            error_message = f"Unexpected error when extracting playlist videos: {e}"
            logger.error(error_message)

            if is_test_mode:
                mock_data = self._get_mock_playlist_videos(playlist_id)
                return mock_data

            raise ValueError(error_message)

    def extract_playlist_metadata(self, playlist_id: str) -> Dict[str, Any]:
        """
        Extract metadata for a YouTube playlist.

        Args:
            playlist_id: YouTube playlist ID

        Returns:
            Dictionary containing playlist metadata
        """
        logger.info(f"Extracting metadata for playlist: {playlist_id}")

        # Check cache first
        cache_key = f"playlist_metadata_{playlist_id}"
        cached_result = cache_get("video", cache_key)
        if cached_result:
            logger.info(f"Using cached metadata for playlist: {playlist_id}")
            return cached_result

        # For test mode, return mock data
        is_test_mode = self.api_key == "test_api_key" or not self.api_key
        if is_test_mode:
            logger.warning("Using test mode with mock playlist metadata")
            mock_data = self._get_mock_playlist_metadata(playlist_id)
            cache_set("video", cache_key, mock_data)
            return mock_data

        try:
            # Request playlist details from YouTube API
            request = self.youtube.playlists().list(
                part="snippet,contentDetails,status",
                id=playlist_id
            )
            response = request.execute()

            if not response.get('items'):
                logger.warning(f"No playlist found with ID: {playlist_id}")
                raise ValueError(f"No playlist found with ID: {playlist_id}")

            # Extract relevant information from response
            playlist_data = response['items'][0]
            snippet = playlist_data.get('snippet', {})
            content_details = playlist_data.get('contentDetails', {})
            status = playlist_data.get('status', {})

            # Enhanced language detection from metadata
            language = self._detect_language_from_metadata(snippet)

            # Create metadata object
            metadata = {
                "playlist_id": playlist_id,
                "title": snippet.get('title', ''),
                "description": snippet.get('description', ''),
                "channel": snippet.get('channelTitle', ''),
                "channel_id": snippet.get('channelId', ''),
                "publication_date": snippet.get('publishedAt', ''),
                "item_count": content_details.get('itemCount', 0),
                "privacy_status": status.get('privacyStatus', ''),
                "thumbnails": snippet.get('thumbnails', {}),
                "language": language
            }

            # Store in cache
            cache_set("video", cache_key, metadata)

            logger.info(f"Successfully extracted metadata for playlist: {playlist_id}")
            return metadata

        except googleapiclient.errors.HttpError as e:
            error_message = f"YouTube API error when extracting playlist metadata: {e}"
            logger.error(error_message)

            if is_test_mode:
                mock_data = self._get_mock_playlist_metadata(playlist_id)
                return mock_data

            raise ValueError(error_message)

        except Exception as e:
            error_message = f"Unexpected error when extracting playlist metadata: {e}"
            logger.error(error_message)

            if is_test_mode:
                mock_data = self._get_mock_playlist_metadata(playlist_id)
                return mock_data

            raise ValueError(error_message)

    def _detect_language_from_metadata(self, snippet: Dict) -> str:
        """
        Enhanced language detection from video metadata.

        Args:
            snippet: Video snippet from YouTube API

        Returns:
            Language code (2-letter)
        """
        # First try to get language from YouTube metadata
        meta_language = snippet.get('defaultLanguage', snippet.get('defaultAudioLanguage', ''))

        # Clean up language code
        if meta_language:
            # Extract base language code
            language_base = meta_language.split('-')[0].lower()

            # If it's a supported language, return it
            if language_base in self.language_metadata:
                return language_base

        # If no language in metadata or unsupported, try to detect from title/description
        title = snippet.get('title', '')
        description = snippet.get('description', '')

        # Combine title and description for better detection
        combined_text = f"{title} {description}"

        # Get the dominant script (writing system)
        script = self._detect_script(combined_text)

        # Map scripts to likely languages
        script_to_language = {
            'Cyrillic': 'ru',
            'Latin': 'en'
            # Add more mappings as needed
        }

        if script in script_to_language:
            return script_to_language[script]

        # Fallback - try langdetect
        try:
            if combined_text:
                detected = langdetect.detect(combined_text)
                if detected in self.language_metadata:
                    return detected
        except LangDetectException:
            pass

        # Default to English if detection fails
        return 'en'

    def _detect_script(self, text: str) -> str:
        """
        Detect the dominant script/writing system of a text.

        Args:
            text: Text to analyze

        Returns:
            Script name
        """
        if not text:
            return 'Latin'  # Default

        # Count characters by script
        script_counts = {}

        for char in text:
            if not char.isalpha():
                continue

            # Get character name which includes script information
            try:
                char_name = unicodedata.name(char)

                # Extract script from character name
                for script in ['LATIN', 'CYRILLIC', 'GREEK', 'ARABIC', 'HEBREW', 'CJK']:
                    if script in char_name:
                        script_counts[script] = script_counts.get(script, 0) + 1
                        break
            except ValueError:
                continue

        # Return the dominant script
        if not script_counts:
            return 'Latin'  # Default

        dominant_script = max(script_counts.items(), key=lambda x: x[1])[0]

        # Map to proper case
        script_map = {
            'LATIN': 'Latin',
            'CYRILLIC': 'Cyrillic',
            'GREEK': 'Greek',
            'ARABIC': 'Arabic',
            'HEBREW': 'Hebrew',
            'CJK': 'CJK'
        }

        return script_map.get(dominant_script, 'Latin')

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
            "title": f"Test Video Title for {video_id}",
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

    def _get_mock_playlist_videos(self, playlist_id: str) -> List[Dict[str, Any]]:
        """
        Generate mock playlist videos data for testing purposes.

        Args:
            playlist_id: YouTube playlist ID

        Returns:
            List of mock video dictionaries
        """
        logger.info(f"Generating mock playlist videos for playlist: {playlist_id}")

        # Generate 5 mock videos
        videos = []
        for i in range(5):
            video_id = f"mock_video_{i}_{playlist_id[-6:]}"
            videos.append({
                "video_id": video_id,
                "title": f"Test Video {i+1} in Playlist {playlist_id}",
                "position": i,
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/default.jpg",
                "description": f"This is a test video {i+1} in the playlist about mathematics.",
                "playlist_id": playlist_id,
                "playlist_title": f"Test Playlist {playlist_id}",
                "playlist_channel": "Test Channel"
            })

        return videos

    def _get_mock_playlist_metadata(self, playlist_id: str) -> Dict[str, Any]:
        """
        Generate mock playlist metadata for testing purposes.

        Args:
            playlist_id: YouTube playlist ID

        Returns:
            Mock playlist metadata dictionary
        """
        logger.info(f"Generating mock metadata for playlist: {playlist_id}")
        return {
            "playlist_id": playlist_id,
            "title": f"Test Playlist Title for {playlist_id}",
            "description": "This is a test playlist about mathematics courses.",
            "channel": "Test Channel",
            "channel_id": "UC123456789",
            "publication_date": "2023-01-01T00:00:00Z",
            "item_count": 5,
            "privacy_status": "public",
            "thumbnails": {
                "default": {"url": f"https://img.youtube.com/vi/{playlist_id}/default.jpg"}
            },
            "language": "en"
        }

    @time_function(3000)  # Log warning if takes more than 3 seconds
    def extract_transcript(self, video_id: str, language_preference: List[str] = ['en', 'ru']) -> List[Dict]:
        """
        Extracts transcript for a YouTube video with improved multilingual support.
        Enhanced with multiple extraction methods and fallbacks.

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

        # Use multiple methods with fallbacks to extract transcript
        transcript_data = None

        # Method 1: Try official API first (most reliable but sometimes reports false negatives)
        transcript_data = self._extract_transcript_with_api(video_id, language_preference)

        # Method 2: If API fails, try alternate extraction methods
        if not transcript_data:
            transcript_data = self._extract_transcript_with_fallbacks(video_id, language_preference)

        # If we have transcript data, format it and cache it
        if transcript_data:
            # Format the transcript
            formatted_transcript = self._format_transcript(transcript_data, language_preference[0])
            cache_set("transcript", cache_key, formatted_transcript)
            return formatted_transcript

        # Last resort: return mock data for test mode or raise error
        if is_test_mode:
            return self._get_mock_transcript()

        # If still no transcript, raise error
        raise ValueError(f"No transcript found for video: {video_id}")

    def _extract_transcript_with_api(self, video_id: str, language_preference: List[str]) -> Optional[List[Dict]]:
        """
        Try to extract transcript using the YouTube Transcript API.

        Args:
            video_id: YouTube video ID
            language_preference: List of language codes in order of preference

        Returns:
            Transcript data if successful, None otherwise
        """
        for attempt in range(self.max_retries):
            try:
                # Get available transcript list
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                # Get all available languages
                available_langs = {
                    **transcript_list._manually_created_transcripts,
                    **transcript_list._generated_transcripts
                }

                logger.info(f"Available transcript languages: {list(available_langs.keys())}")

                # Expand language codes to include regional variants
                expanded_preferences = []
                for lang in language_preference:
                    # Add base code
                    expanded_preferences.append(lang)

                    # Add known variants
                    if lang in self.language_metadata:
                        expanded_preferences.extend(self.language_metadata[lang]['codes'])

                # Deduplicate
                expanded_preferences = list(dict.fromkeys(expanded_preferences))

                # Try to match preferred languages
                for lang_code in expanded_preferences:
                    # Try manually created transcripts first
                    for available_lang in transcript_list._manually_created_transcripts:
                        # Check if language code matches or starts with preferred code
                        if available_lang == lang_code or available_lang.startswith(f"{lang_code}-"):
                            try:
                                transcript = transcript_list._manually_created_transcripts[available_lang]
                                logger.info(f"Found manually created transcript in {available_lang}")
                                return transcript.fetch()
                            except Exception as e:
                                logger.debug(f"Error fetching manual transcript in {available_lang}: {e}")

                    # Then try generated transcripts
                    for available_lang in transcript_list._generated_transcripts:
                        if available_lang == lang_code or available_lang.startswith(f"{lang_code}-"):
                            try:
                                transcript = transcript_list._generated_transcripts[available_lang]
                                logger.info(f"Found generated transcript in {available_lang}")
                                result = transcript.fetch()

                                # Add additional check: verify the result is not empty or malformed
                                if not result or not isinstance(result, list) or len(result) == 0:
                                    logger.warning(f"Transcript API returned empty or invalid result for {available_lang}")
                                    continue

                                return result
                            except Exception as e:
                                logger.debug(f"Error fetching generated transcript in {available_lang}: {e}")

                # If no preferred language transcript is found, get any available transcript
                if transcript_list._manually_created_transcripts:
                    # Prefer manually created transcripts
                    lang = list(transcript_list._manually_created_transcripts.keys())[0]
                    transcript = transcript_list._manually_created_transcripts[lang]
                    logger.info(f"Using fallback manual transcript in {lang}")
                    result = transcript.fetch()

                    # Verify result
                    if not result or not isinstance(result, list) or len(result) == 0:
                        logger.warning(f"Fallback manual transcript returned empty or invalid result")
                        continue

                    return result

                elif transcript_list._generated_transcripts:
                    # Use generated transcript if no manual one available
                    lang = list(transcript_list._generated_transcripts.keys())[0]
                    transcript = transcript_list._generated_transcripts[lang]
                    logger.info(f"Using fallback generated transcript in {lang}")
                    result = transcript.fetch()

                    # Verify result
                    if not result or not isinstance(result, list) or len(result) == 0:
                        logger.warning(f"Fallback generated transcript returned empty or invalid result")
                        continue

                    return result

                # If we get here, no transcript was found
                logger.warning(f"No transcript found via API for video: {video_id}")
                return None

            except TranscriptsDisabled:
                logger.warning(f"API reports transcripts are disabled for video: {video_id} (attempt {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except NoTranscriptFound:
                logger.warning(f"API reports no transcript found for video: {video_id} (attempt {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Error extracting transcript with API: {e} (attempt {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # If we reach here, all attempts failed
        return None

    def _extract_transcript_with_fallbacks(self, video_id: str, language_preference: List[str]) -> Optional[List[Dict]]:
        """
        Extract transcript using alternative methods when the API fails.
        This approach can sometimes work when the official API incorrectly reports that transcripts are disabled.

        Args:
            video_id: YouTube video ID
            language_preference: List of language codes in order of preference

        Returns:
            Transcript data if successful, None otherwise
        """
        # Try multiple fallback methods

        # Method 1: Direct transcript extraction using pattern matching
        transcript_data = self._extract_transcript_direct(video_id, language_preference)
        if transcript_data:
            logger.info(f"Successfully extracted transcript using direct method for video: {video_id}")
            return transcript_data

        # Method 2: Timedtext API (used for auto-generated captions)
        transcript_data = self._extract_transcript_timedtext(video_id, language_preference)
        if transcript_data:
            logger.info(f"Successfully extracted transcript using timedtext API for video: {video_id}")
            return transcript_data

        # All fallback methods failed
        logger.warning(f"All transcript extraction methods failed for video: {video_id}")
        return None

    def _extract_transcript_direct(self, video_id: str, language_preference: List[str]) -> Optional[List[Dict]]:
        """
        Extract transcript directly from video page using pattern matching.
        This can sometimes work when the API incorrectly reports that transcripts are disabled.

        Args:
            video_id: YouTube video ID
            language_preference: List of language codes in order of preference

        Returns:
            Transcript data if successful, None otherwise
        """
        try:
            # Try to fetch the video page
            url = f"https://www.youtube.com/watch?v={video_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch video page for transcript extraction: {response.status_code}")
                return None

            content = response.text

            # Look for the caption track data in the page
            captions_regex = r'"captionTracks":\s*(\[.*?\])'
            captions_match = re.search(captions_regex, content)

            if not captions_match:
                logger.warning("Could not find caption tracks in video page")
                return None

            captions_data = captions_match.group(1)
            # Parse the JSON data
            try:
                captions_json = json.loads(captions_data.replace('\\u0026', '&').replace('\\', '\\\\'))
            except json.JSONDecodeError:
                logger.warning("Failed to parse caption tracks data")
                return None

            # Find the best caption track based on language preference
            best_track = None

            # Expand language codes to include regional variants
            expanded_preferences = []
            for lang in language_preference:
                # Add base code
                expanded_preferences.append(lang)
                # Add known variants
                if lang in self.language_metadata:
                    expanded_preferences.extend(self.language_metadata[lang]['codes'])

            # First try to find a track matching preferred languages
            for lang_code in expanded_preferences:
                for track in captions_json:
                    track_lang = track.get('languageCode', '')
                    is_auto = track.get('kind', '') == 'asr'  # Auto-generated captions

                    if track_lang == lang_code or track_lang.startswith(f"{lang_code}-"):
                        # Prefer manual captions over auto-generated
                        if best_track is None or (is_auto and best_track.get('kind') != 'asr'):
                            best_track = track
                            break

            # If no preferred language found, use any available track
            if best_track is None and captions_json:
                best_track = captions_json[0]

            if not best_track:
                logger.warning("No suitable caption track found")
                return None

            # Get the caption track URL
            track_url = best_track.get('baseUrl', '')
            if not track_url:
                logger.warning("Caption track URL not found")
                return None

            # Fetch the caption track
            track_response = requests.get(track_url, timeout=10)
            if track_response.status_code != 200:
                logger.warning(f"Failed to fetch caption track: {track_response.status_code}")
                return None

            # Parse the caption data
            caption_data = track_response.text

            # Verify that the response is not empty
            if not caption_data or len(caption_data.strip()) == 0:
                logger.warning("Caption track response is empty")
                return None

            # XML format parsing - simple approach for now
            segments = []

            # Verify this looks like XML before parsing
            if not caption_data.strip().startswith('<?xml') and not caption_data.strip().startswith('<transcript'):
                logger.warning("Caption track response doesn't appear to be valid XML")
                logger.debug(f"First 100 chars of response: {caption_data[:100]}")
                return None

            # Simple regex-based parsing for caption data
            caption_regex = r'<text start="([\d\.]+)" dur="([\d\.]+)".*?>(.*?)</text>'
            matches = list(re.finditer(caption_regex, caption_data))

            if not matches:
                logger.warning("No caption segments found in the XML response")
                return None

            for match in matches:
                start = float(match.group(1))
                duration = float(match.group(2))
                text = match.group(3).strip()

                # Clean up HTML entities
                text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')

                segments.append({
                    'start': start,
                    'duration': duration,
                    'text': text
                })

            if segments:
                return segments

            logger.warning("Failed to extract segments from caption data")
            return None

        except Exception as e:
            logger.error(f"Error in direct transcript extraction: {e}")
            return None

    def _extract_transcript_timedtext(self, video_id: str, language_preference: List[str]) -> Optional[List[Dict]]:
        """
        Extract transcript using YouTube's timedtext API.
        Used as a fallback method for auto-generated captions.

        Args:
            video_id: YouTube video ID
            language_preference: List of language codes in order of preference

        Returns:
            Transcript data if successful, None otherwise
        """
        try:
            # Try languages in order of preference
            for lang in language_preference:
                # Format for auto-generated captions
                url = f"https://www.youtube.com/api/timedtext?lang={lang}&v={video_id}"

                # Also try alternative format
                alt_url = f"https://www.youtube.com/api/timedtext?lang={lang}&v={video_id}&fmt=srv3"

                # Try both URLs
                for try_url in [url, alt_url]:
                    try:
                        response = requests.get(try_url, timeout=10)
                        if response.status_code == 200 and response.text:
                            # Make sure the response is not empty and is valid XML
                            caption_data = response.text

                            if not caption_data or len(caption_data.strip()) == 0:
                                continue

                            # Verify this looks like XML before parsing
                            if not caption_data.strip().startswith('<?xml') and not caption_data.strip().startswith('<transcript'):
                                logger.debug(f"Response doesn't appear to be valid XML: {caption_data[:100]}")
                                continue

                            # Simple regex-based parsing for caption data
                            segments = []
                            caption_regex = r'<text start="([\d\.]+)" dur="([\d\.]+)".*?>(.*?)</text>'
                            matches = list(re.finditer(caption_regex, caption_data))

                            if not matches:
                                logger.debug(f"No caption segments found in the XML response for {try_url}")
                                continue

                            for match in matches:
                                start = float(match.group(1))
                                duration = float(match.group(2))
                                text = match.group(3).strip()

                                # Clean up HTML entities
                                text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')

                                segments.append({
                                    'start': start,
                                    'duration': duration,
                                    'text': text
                                })

                            if segments:
                                return segments
                    except Exception as e:
                        logger.debug(f"Error in timedtext API call to {try_url}: {e}")

            # If we get here, all attempts failed
            logger.warning("Failed to extract transcript using timedtext API")
            return None

        except Exception as e:
            logger.error(f"Error in timedtext transcript extraction: {e}")
            return None

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

    def _enhanced_domain_classification(self, title: str, description: str, language: str = 'en') -> Tuple[str, float]:
        """
        Enhanced domain classification with multilingual support.

        Args:
            title: Video title
            description: Video description
            language: Detected language code

        Returns:
            Tuple of (domain, confidence)
        """
        # Combine text for analysis
        combined_text = f"{title} {description}".lower()

        # Define multilingual domain-specific keywords
        domain_keywords = {
            "mathematics": {
                "en": [
                    'math', 'mathematics', 'calculus', 'algebra', 'geometry', 'theorem', 'proof',
                    'equation', 'function', 'derivative', 'integral', 'statistics', 'probability'
                ],
                "ru": [
                    'математика', 'матем', 'алгебра', 'геометрия', 'теорема', 'доказательство',
                    'уравнение', 'функция', 'производная', 'интеграл', 'статистика', 'вероятность'
                ]
            },
            "programming": {
                "en": [
                    'programming', 'algorithm', 'code', 'software', 'development', 'computer science',
                    'python', 'java', 'c++', 'javascript', 'data structure', 'database', 'web'
                ],
                "ru": [
                    'программирование', 'алгоритм', 'код', 'программа', 'разработка', 'информатика',
                    'python', 'java', 'с++', 'javascript', 'структура данных', 'база данных', 'веб'
                ]
            },
            "physics": {
                "en": [
                    'physics', 'mechanics', 'dynamics', 'kinematics', 'electromagnetism',
                    'thermodynamics', 'quantum', 'relativity', 'force', 'energy', 'momentum'
                ],
                "ru": [
                    'физика', 'механика', 'динамика', 'кинематика', 'электромагнетизм',
                    'термодинамика', 'квантовая', 'относительность', 'сила', 'энергия', 'импульс'
                ]
            }
        }

        # Get language-specific keywords, fall back to English if not available
        lang_key = language if language in ['en', 'ru'] else 'en'

        # Count keyword matches for each domain
        domain_scores = {}

        for domain, keywords_dict in domain_keywords.items():
            keywords = keywords_dict.get(lang_key, keywords_dict.get('en', []))
            count = sum(1 for keyword in keywords if keyword in combined_text)
            domain_scores[domain] = count

            # Add weighted category ID score for English YouTube categories
            if language == 'en':
                if 'math' in combined_text and domain == 'mathematics':
                    domain_scores[domain] += 2
                elif 'programming' in combined_text and domain == 'programming':
                    domain_scores[domain] += 2
                elif 'physics' in combined_text and domain == 'physics':
                    domain_scores[domain] += 2

        # Get the domain with highest count
        max_count = max(domain_scores.values()) if domain_scores else 0

        if max_count == 0:
            return ('unknown', 0.0)

        # Get domain with highest count
        max_domains = [domain for domain, count in domain_scores.items() if count == max_count]

        if len(max_domains) == 1:
            domain = max_domains[0]
            total_count = sum(domain_scores.values())
            confidence = max_count / total_count if total_count > 0 else 0.0

            # Boost confidence for strong matches
            if max_count >= 3:
                confidence = min(confidence + 0.2, 0.95)

            return (domain, confidence)
        else:
            # If tie, return the first domain with medium confidence
            return (max_domains[0], 0.5)

    def _format_transcript(self, transcript_data: List[Dict], language: str) -> List[Dict]:
        """
        Formats transcript data into a standardized structure with language detection.

        Args:
            transcript_data: Raw transcript data from YouTube API
            language: Detected language of transcript

        Returns:
            List of formatted transcript segments
        """
        # Validate input to avoid errors
        if not transcript_data or not isinstance(transcript_data, list):
            logger.warning("Invalid transcript data format - returning empty transcript")
            return []

        formatted_transcript = []

        # If language wasn't provided, try to detect from transcript text
        if not language:
            # Combine some segments for better language detection
            sample_text = " ".join([segment.get('text', '') for segment in transcript_data[:10]])
            try:
                detected_lang = langdetect.detect(sample_text)
                # Use only the base language code
                language = detected_lang.split('-')[0]
            except:
                # Default to English if detection fails
                language = 'en'

        # Normalize language code to 2-letter code
        language = language[:2].lower()

        for segment in transcript_data:
            # Validate segment format
            if not isinstance(segment, dict):
                logger.warning(f"Invalid segment format: {segment} - skipping")
                continue

            # Get segment fields with defaults
            start = segment.get('start', 0.0)
            duration = segment.get('duration', 0.0)
            text = segment.get('text', '')

            # Skip empty segments
            if not text.strip():
                continue

            formatted_segment = {
                "start": start,
                "duration": duration,
                "text": text,
                "language": language
            }
            formatted_transcript.append(formatted_segment)

        return formatted_transcript
