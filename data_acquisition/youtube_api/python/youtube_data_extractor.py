"""
YouTube Data Extractor module for the Lecture Video Content Indexer.
Handles extraction of video metadata and transcripts from YouTube.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
import googleapiclient.discovery
import googleapiclient.errors
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Configure logging
logger = logging.getLogger(__name__)

class YouTubeDataExtractor:
    """
    Extracts video metadata and transcripts from YouTube videos.
    Optimized for educational content in Russian and English.
    """

    def __init__(self, api_key: str):
        """
        Initialize the YouTube Data Extractor with API credentials.

        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key, cache_discovery=False
        )
        logger.info("YouTube Data Extractor initialized")

    def validate_video_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validates a YouTube URL and extracts the video ID.

        Args:
            url: YouTube video URL

        Returns:
            Tuple of (is_valid, video_id)
        """
        # Regular expression patterns for different YouTube URL formats
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',  # Standard URL
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^\?\s]+)',  # Shortened URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\?\s]+)'  # Embedded URL
        ]

        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                video_id = match.group(1)
                return True, video_id

        logger.warning(f"Invalid YouTube URL format: {url}")
        return False, None

    def extract_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Extracts metadata for a YouTube video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary containing video metadata
        """
        logger.info(f"Extracting metadata for video: {video_id}")

        try:
            # Request video details from YouTube API
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics,status",
                id=video_id
            )
            response = request.execute()

            if not response.get('items'):
                logger.warning(f"No video found with ID: {video_id}")
                return {}

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
            # Simplify language code (we only care about Russian and English)
            if language.startswith('ru'):
                language = 'ru'
            elif language.startswith('en'):
                language = 'en'
            else:
                language = ''  # Will be determined from transcript later

            # Extract educational metadata from description
            description = snippet.get('description', '')
            educational_metadata = self._extract_educational_metadata(description)

            # Perform initial domain classification based on title, tags, and description
            domain, domain_confidence = self._initial_domain_classification(
                snippet.get('title', ''),
                snippet.get('tags', []),
                description
            )

            # Create structured metadata object
            metadata = {
                "video_id": video_id,
                "title": snippet.get('title', ''),
                "channel": snippet.get('channelTitle', ''),
                "publication_date": snippet.get('publishedAt', ''),
                "duration_seconds": duration_seconds,
                "category": snippet.get('categoryId', ''),
                "tags": snippet.get('tags', []),
                "description": description,
                "language": language,
                "view_count": int(statistics.get('viewCount', 0)),
                "educational_metadata": educational_metadata,
                "domain": domain,
                "domain_confidence": domain_confidence
            }

            logger.info(f"Successfully extracted metadata for video: {video_id}")
            return metadata

        except googleapiclient.errors.HttpError as e:
            logger.error(f"YouTube API error when extracting metadata: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error when extracting metadata: {e}")
            return {}

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

        try:
            # Get available transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get manually created transcript in preferred languages
            for lang in language_preference:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    logger.info(f"Found manually created transcript in {lang}")
                    return self._format_transcript(transcript.fetch(), lang)
                except:
                    logger.debug(f"No manually created transcript in {lang}")

            # Try to get generated transcript in preferred languages
            for lang in language_preference:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    logger.info(f"Found generated transcript in {lang}")
                    return self._format_transcript(transcript.fetch(), lang)
                except:
                    logger.debug(f"No generated transcript in {lang}")

            # If no preferred language transcript is found, get the default one
            try:
                default_transcript = transcript_list.find_transcript(['en'])
                logger.info(f"Using default transcript")
                detected_lang = self.detect_language(default_transcript.fetch())
                return self._format_transcript(default_transcript.fetch(), detected_lang)
            except:
                logger.warning(f"No default transcript available")
                return []

        except TranscriptsDisabled:
            logger.warning(f"Transcripts are disabled for video: {video_id}")
            return []
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video: {video_id}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error when extracting transcript: {e}")
            return []

    def detect_language(self, transcript: List[Dict]) -> str:
        """
        Detects language of transcript (focusing on Russian and English).

        Args:
            transcript: List of transcript segments

        Returns:
            Language code ('en' or 'ru')
        """
        if not transcript:
            return 'en'  # Default to English if no transcript

        # Combine some text from transcript for language detection
        # Use at most 10 segments to avoid processing too much text
        text_sample = ' '.join([item.get('text', '') for item in transcript[:10]])

        # Count Cyrillic characters (for Russian detection)
        cyrillic_count = sum(1 for char in text_sample if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in text_sample if '\u0041' <= char <= '\u007A')

        # If significant portion is Cyrillic, consider it Russian
        if cyrillic_count > 0 and cyrillic_count > latin_count * 0.3:
            return 'ru'
        return 'en'

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

    def _extract_educational_metadata(self, description: str) -> Dict[str, Optional[str]]:
        """
        Extracts educational metadata from video description.

        Args:
            description: Video description text

        Returns:
            Dictionary with educational metadata
        """
        metadata = {
            'course_name': None,
            'instructor': None,
            'institution': None
        }

        # Extract course information (English patterns)
        course_match = re.search(r'Course:?\s*([^\n]+)', description)
        if course_match:
            metadata['course_name'] = course_match.group(1).strip()

        # Extract instructor information (English patterns)
        instructor_match = re.search(r'(Instructor|Lecturer|Professor|Teacher):?\s*([^\n]+)', description)
        if instructor_match:
            metadata['instructor'] = instructor_match.group(2).strip()

        # Extract institution information (English patterns)
        institution_match = re.search(r'(University|College|School|Institution):?\s*([^\n]+)', description)
        if institution_match:
            metadata['institution'] = institution_match.group(2).strip()

        # Russian patterns for course
        ru_course_match = re.search(r'(Курс|Предмет):?\s*([^\n]+)', description)
        if ru_course_match and not metadata['course_name']:
            metadata['course_name'] = ru_course_match.group(2).strip()

        # Russian patterns for instructor
        ru_instructor_match = re.search(r'(Преподаватель|Лектор|Профессор):?\s*([^\n]+)', description)
        if ru_instructor_match and not metadata['instructor']:
            metadata['instructor'] = ru_instructor_match.group(2).strip()

        # Russian patterns for institution
        ru_institution_match = re.search(r'(Университет|Институт|ВУЗ|Школа):?\s*([^\n]+)', description)
        if ru_institution_match and not metadata['institution']:
            metadata['institution'] = ru_institution_match.group(2).strip()

        return metadata

    def _initial_domain_classification(self, title: str, tags: List[str], description: str) -> Tuple[str, float]:
        """
        Performs initial domain classification based on video metadata.

        Args:
            title: Video title
            tags: Video tags
            description: Video description

        Returns:
            Tuple of (domain, confidence)
        """
        # Combine text for analysis
        combined_text = f"{title} {' '.join(tags)} {description}".lower()

        # Define domain-specific keywords
        math_keywords = [
            'math', 'mathematics', 'calculus', 'algebra', 'geometry', 'theorem', 'proof',
            'equation', 'function', 'derivative', 'integral', 'математика', 'алгебра',
            'геометрия', 'теорема', 'доказательство', 'уравнение', 'функция', 'матан',
            'матем', 'производная', 'интеграл'
        ]

        programming_keywords = [
            'programming', 'algorithm', 'code', 'software', 'development', 'computer science',
            'python', 'java', 'c++', 'javascript', 'data structure', 'программирование',
            'алгоритм', 'код', 'разработка', 'информатика', 'структуры данных'
        ]

        physics_keywords = [
            'physics', 'mechanics', 'dynamics', 'kinematics', 'electromagnetism', 'thermodynamics',
            'quantum', 'relativity', 'физика', 'механика', 'динамика', 'кинематика',
            'электромагнетизм', 'термодинамика', 'квантовая', 'относительность'
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
            return 'unknown', 0.0

        # Check if we have a clear winner
        top_domains = [domain for domain, count in counts.items() if count == max_count]
        if len(top_domains) == 1:
            domain = top_domains[0]
            # Calculate confidence based on relative frequency
            total_count = sum(counts.values())
            confidence = max_count / total_count if total_count > 0 else 0.0
            return domain, confidence
        else:
            # If tie, return the first domain with low confidence
            return top_domains[0], 0.5

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

        for i, segment in enumerate(transcript_data):
            formatted_segment = {
                "start": segment.get('start', 0.0),
                "duration": segment.get('duration', 0.0),
                "text": segment.get('text', ''),
                "confidence": segment.get('confidence', None),
                "speaker": None,  # YouTube doesn't provide speaker information
                "language": language
            }
            formatted_transcript.append(formatted_segment)

        return formatted_transcript
