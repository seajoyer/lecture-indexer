#!/usr/bin/env python3
"""
YouTube Lecture Batch Processor - Analyzes educational videos for theoretical vs. practical content.

This script takes a text file with YouTube video URLs (one per line) and processes each video to:
1. Extract metadata and transcript
2. Classify the domain (mathematics, programming, physics)
3. Analyze theoretical vs. practical content
4. Output detailed analysis results

Usage:
    python batch_processor.py input_file.txt [--output-dir OUTPUT_DIR] [--api-key API_KEY]

Requirements:
    - YouTube Data API key (set as YOUTUBE_API_KEY environment variable or use --api-key)
    - Required packages: google-api-python-client, youtube-transcript-api
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import requests
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch_processor")

class YouTubeDataExtractor:
    """Extract video metadata and transcripts from YouTube."""

    def __init__(self, api_key: str):
        """
        Initialize with API key.

        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key
        self._youtube = None

    @property
    def youtube(self):
        """Lazy initialization of YouTube API client."""
        if self._youtube is None:
            try:
                self._youtube = googleapiclient.discovery.build(
                    "youtube", "v3", developerKey=self.api_key, cache_discovery=False
                )
                logger.info("Successfully built YouTube API client")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API client: {e}")
                # Return a mock client for testing
                self._youtube = self._create_mock_client()
        return self._youtube

    def _create_mock_client(self):
        """Create a mock client when API initialization fails."""
        logger.warning("Creating mock YouTube client - API calls will not work")
        mock = type('MockYouTube', (), {})()
        mock_videos = type('MockVideos', (), {})()
        mock_list = type('MockList', (), {})()

        def mock_execute():
            return {"items": []}

        mock_list.execute = mock_execute
        mock_videos.list = lambda **kwargs: mock_list
        mock.videos = lambda: mock_videos
        return mock

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
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\?\s]+)',  # Embedded URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^\?\s]+)',  # Old embed URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([^\?\s]+)'  # YouTube shorts URL
        ]

        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                video_id = match.group(1)
                return (True, video_id)

        logger.warning(f"Invalid YouTube URL format: {url}")
        return (False, None)

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
                return {"error": f"No video found with ID: {video_id}"}

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
            # Simplify language code
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

            return metadata

        except Exception as e:
            error_message = f"Error extracting metadata: {e}"
            logger.error(error_message)
            return {"error": error_message}

    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration string to seconds."""
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
        """Extract educational metadata from video description."""
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
        """Classify the domain based on video metadata."""
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
            return ('unknown', 0.0)
        else:
            # Check if we have a clear winner
            top_domains = [domain for domain, count in counts.items() if count == max_count]
            if len(top_domains) == 1:
                domain = top_domains[0]
                # Calculate confidence based on relative frequency
                total_count = sum(counts.values())
                confidence = max_count / total_count if total_count > 0 else 0.0
                return (domain, confidence)
            else:
                # If tie, return the first domain with low confidence
                return (top_domains[0], 0.5)

    def extract_transcript(self, video_id: str, language_preference: List[str] = ['en', 'ru']) -> List[Dict]:
        """
        Extract transcript for a YouTube video with preference for specified languages.

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
                except Exception as e:
                    logger.debug(f"No manually created transcript in {lang}: {e}")

            # Try to get generated transcript in preferred languages
            for lang in language_preference:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    logger.info(f"Found generated transcript in {lang}")
                    return self._format_transcript(transcript.fetch(), lang)
                except Exception as e:
                    logger.debug(f"No generated transcript in {lang}: {e}")

            # If no preferred language transcript is found, get the default one
            try:
                default_transcript = transcript_list.find_transcript(['en'])
                logger.info("Using default transcript")
                raw_transcript = default_transcript.fetch()
                detected_lang = self._detect_language(raw_transcript)
                return self._format_transcript(raw_transcript, detected_lang)
            except:
                logger.warning("No default transcript available")

                # Try one more fallback: use any available transcript
                try:
                    available_langs = transcript_list._manually_created_transcripts.keys()
                    if available_langs:
                        lang = list(available_langs)[0]
                        transcript = transcript_list._manually_created_transcripts[lang]
                        logger.info(f"Found fallback transcript in {lang}")
                        raw_transcript = transcript.fetch()
                        detected_lang = self._detect_language(raw_transcript)
                        return self._format_transcript(raw_transcript, detected_lang)
                except Exception as e:
                    logger.error(f"Failed to get fallback transcript: {e}")

            return {"error": "No transcript found in any language"}

        except TranscriptsDisabled:
            error_message = f"Transcripts are disabled for video: {video_id}"
            logger.warning(error_message)
            return {"error": error_message}

        except NoTranscriptFound:
            error_message = f"No transcript found for video: {video_id}"
            logger.warning(error_message)
            return {"error": error_message}

        except Exception as e:
            error_message = f"Error extracting transcript: {e}"
            logger.error(error_message)
            return {"error": error_message}

    def _detect_language(self, transcript: List[Dict]) -> str:
        """Detect language of transcript (focusing on Russian and English)."""
        if not transcript:
            return 'en'  # Default to English if no transcript

        # Combine some text from transcript for language detection
        text_sample = ' '.join([item.get('text', '') for item in transcript[:10]])

        # Count Cyrillic characters (for Russian detection)
        cyrillic_count = sum(1 for char in text_sample if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in text_sample if '\u0041' <= char <= '\u007A')

        # If significant portion is Cyrillic, consider it Russian
        return 'ru' if cyrillic_count > 0 and cyrillic_count > latin_count * 0.3 else 'en'

    def _format_transcript(self, transcript_data: List[Dict], language: str) -> List[Dict]:
        """Format transcript data into a standardized structure."""
        formatted_transcript = []

        for i, segment in enumerate(transcript_data):
            formatted_segment = {
                "start": segment.get('start', 0.0),
                "duration": segment.get('duration', 0.0),
                "text": segment.get('text', ''),
                "confidence": segment.get('confidence', 1.0),  # Set default confidence
                "speaker": None,  # YouTube doesn't provide speaker information
                "language": language
            }
            formatted_transcript.append(formatted_segment)

        return formatted_transcript


class TranscriptProcessor:
    """Process and analyze transcript data."""

    def __init__(self):
        """Initialize the transcript processor."""
        logger.info("Initializing Transcript Processor")

    def process_transcript(self, transcript: List[Dict], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw transcript segments into a structured format.

        Args:
            transcript: List of raw transcript segments
            metadata: Video metadata

        Returns:
            Dictionary containing processed transcript data
        """
        video_id = metadata.get("video_id", "")

        if not transcript or isinstance(transcript, dict) and "error" in transcript:
            logger.warning("Empty or error transcript provided")
            return {
                "segments": [],
                "sentences": [],
                "language": metadata.get("language", "en"),
                "domain": metadata.get("domain", "unknown"),
                "video_id": video_id
            }

        # Determine language
        language = transcript[0].get("language", "en") if transcript else "en"

        # Domain from metadata
        domain = metadata.get("domain", "unknown")

        # Process segments
        processed_segments = []

        for i, segment in enumerate(transcript):
            processed_segment = {
                "id": f"segment_{video_id}_{i}",
                "start_time": segment.get("start", 0),
                "end_time": segment.get("start", 0) + segment.get("duration", 0),
                "text": segment.get("text", ""),
                "language": language,
                "video_id": video_id,
                "domain": domain,
                "content_type": self._classify_segment_content_type(segment.get("text", ""), domain)
            }
            processed_segments.append(processed_segment)

        # Group segments into sentences
        sentences = self._extract_sentences(processed_segments)

        # Create processed transcript object
        processed_transcript = {
            "segments": processed_segments,
            "sentences": sentences,
            "language": language,
            "domain": domain,
            "video_id": video_id
        }

        return processed_transcript

    def _classify_segment_content_type(self, text: str, domain: str) -> str:
        """
        Classify segment as theoretical or practical based on text content.

        Args:
            text: Segment text
            domain: Content domain

        Returns:
            Classification: "theoretical", "practical", or "mixed"
        """
        # Count theoretical and practical indicators
        theoretical_count = 0
        practical_count = 0

        # Theoretical indicators
        theoretical_patterns = [
            r'(?:definition|define|theory|concept|theorem|proof)',
            r'(?:principle|axiom|lemma|postulate|hypothesis)',
            r'(?:formally|theoretically|conceptually|abstractly)',
            r'(?:consider|assume|suppose|let)',
            r'(?:we know that|is defined as|is called)'
        ]

        # Practical indicators
        practical_patterns = [
            r'(?:example|exercise|problem|task|implementation)',
            r'(?:let\'s try|let\'s see|we use|try this|see how)',
            r'(?:apply|implement|demonstrate|show how|build)',
            r'(?:calculate|compute|evaluate|solve|find)',
            r'(?:in practice|practically|in the real world|step-by-step)'
        ]

        # Domain-specific patterns
        if domain == "mathematics":
            theoretical_patterns.extend([
                r'(?:theorem|proposition|corollary|equation|formula)',
                r'(?:given that|therefore|thus|hence|we conclude)'
            ])
            practical_patterns.extend([
                r'(?:calculate|solve for|find the value|determine)',
                r'(?:substituting|plugging in|let\'s try|working through)'
            ])
        elif domain == "programming":
            theoretical_patterns.extend([
                r'(?:algorithm|complexity|time complexity|space complexity)',
                r'(?:data structure|object-oriented|functional|paradigm)'
            ])
            practical_patterns.extend([
                r'(?:code|program|function|method|class|parameter)',
                r'(?:write|compile|execute|run|debug|test|output)'
            ])
        elif domain == "physics":
            theoretical_patterns.extend([
                r'(?:law|theory|principle|conservation|field)',
                r'(?:relativity|quantum|conservation of|according to)'
            ])
            practical_patterns.extend([
                r'(?:measure|experiment|observe|laboratory|setup)',
                r'(?:force|energy|momentum|velocity|calculate the)'
            ])

        # Count matches
        for pattern in theoretical_patterns:
            theoretical_count += len(re.findall(pattern, text, re.IGNORECASE))

        for pattern in practical_patterns:
            practical_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Classify based on counts
        if theoretical_count > practical_count * 2:
            return "theoretical"
        elif practical_count > theoretical_count * 2:
            return "practical"
        else:
            return "mixed"

    def _extract_sentences(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract sentences from transcript segments.

        Args:
            segments: List of transcript segments

        Returns:
            List of sentence dictionaries
        """
        sentences = []

        for segment in segments:
            text = segment.get("text", "")

            # Simple sentence splitting based on punctuation
            # In a full implementation, use NLTK or spaCy for better sentence segmentation
            sentence_texts = re.split(r'(?<=[.!?])\s+', text)

            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time

            # Process each sentence
            for i, sentence_text in enumerate(sentence_texts):
                if not sentence_text.strip():
                    continue

                # Calculate time proportionally
                ratio = i / max(1, len(sentence_texts))
                sentence_start = start_time + duration * ratio
                sentence_end = start_time + duration * (i + 1) / max(1, len(sentence_texts))

                # Create sentence object
                sentence = {
                    "id": f"{segment.get('id')}_sentence_{i}",
                    "start_time": sentence_start,
                    "end_time": sentence_end,
                    "text": sentence_text.strip(),
                    "content_type": segment.get("content_type", "mixed"),
                    "domain": segment.get("domain", "unknown"),
                    "segment_id": segment.get("id"),
                    "video_id": segment.get("video_id")
                }

                sentences.append(sentence)

        return sentences


class TheoryPracticeClassifier:
    """Classify educational content as theoretical or practical."""

    def __init__(self):
        """Initialize the classifier."""
        logger.info("Initializing Theory/Practice Classifier")

    def classify_transcript(self, processed_transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify transcript as theoretical, practical, or mixed.

        Args:
            processed_transcript: Processed transcript data

        Returns:
            Classification results
        """
        segments = processed_transcript.get("segments", [])

        if not segments:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "theory_practice_ratio": 0.5,
                "theoretical_segments": 0,
                "practical_segments": 0,
                "mixed_segments": 0
            }

        # Count by type
        theoretical_segments = sum(1 for s in segments if s.get("content_type") == "theoretical")
        practical_segments = sum(1 for s in segments if s.get("content_type") == "practical")
        mixed_segments = sum(1 for s in segments if s.get("content_type") == "mixed")

        # Calculate ratio
        total_segments = len(segments)
        theory_practice_ratio = theoretical_segments / max(1, total_segments)

        # Determine classification
        if theory_practice_ratio > 0.7:
            classification = "theoretical"
            confidence = min(1.0, theory_practice_ratio)
        elif theory_practice_ratio < 0.3:
            classification = "practical"
            confidence = min(1.0, 1.0 - theory_practice_ratio)
        else:
            classification = "mixed"
            confidence = 0.5 + abs(0.5 - theory_practice_ratio)

        return {
            "classification": classification,
            "confidence": confidence,
            "theory_practice_ratio": theory_practice_ratio,
            "theoretical_segments": theoretical_segments,
            "practical_segments": practical_segments,
            "mixed_segments": mixed_segments
        }

    def extract_theory_practice_patterns(self, processed_transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract patterns of theory-to-practice and practice-to-theory transitions.

        Args:
            processed_transcript: Processed transcript data

        Returns:
            Patterns dictionary
        """
        segments = processed_transcript.get("segments", [])

        if not segments:
            return {
                "theory_to_practice_sequences": [],
                "practice_to_theory_sequences": []
            }

        # Find transitions
        theory_to_practice = []
        practice_to_theory = []

        for i in range(len(segments) - 1):
            current_type = segments[i].get("content_type")
            next_type = segments[i+1].get("content_type")

            # Theory to practice transition
            if current_type == "theoretical" and next_type == "practical":
                # Get sequence of segments (current and next)
                sequence = {
                    "id": f"t2p_{i}",
                    "start_time": segments[i].get("start_time"),
                    "end_time": segments[i+1].get("end_time"),
                    "pattern_type": "theory_to_practice",
                    "segments": [segments[i], segments[i+1]]
                }
                theory_to_practice.append(sequence)

            # Practice to theory transition
            elif current_type == "practical" and next_type == "theoretical":
                # Get sequence of segments (current and next)
                sequence = {
                    "id": f"p2t_{i}",
                    "start_time": segments[i].get("start_time"),
                    "end_time": segments[i+1].get("end_time"),
                    "pattern_type": "practice_to_theory",
                    "segments": [segments[i], segments[i+1]]
                }
                practice_to_theory.append(sequence)

        return {
            "theory_to_practice_sequences": theory_to_practice,
            "practice_to_theory_sequences": practice_to_theory
        }


class DomainClassifier:
    """Classify educational content domain and extract domain-specific features."""

    def __init__(self):
        """Initialize the classifier."""
        logger.info("Initializing Domain Classifier")

    def classify_transcript(self, processed_transcript: Dict[str, Any]) -> Tuple[str, float]:
        """
        Classify the domain of a transcript.

        Args:
            processed_transcript: Processed transcript data

        Returns:
            Tuple of (domain, confidence)
        """
        # Use existing domain if already present
        domain = processed_transcript.get("domain")
        if domain and domain != "unknown":
            return domain, 1.0

        # Extract text for analysis
        segments = processed_transcript.get("segments", [])
        text = " ".join([s.get("text", "") for s in segments])

        # Define domain-specific keywords
        domains = {
            "mathematics": [
                r'\bmath(ematics)?\b', r'\bcalculus\b', r'\balgebra\b', r'\bgeometry\b',
                r'\btheorem\b', r'\bproof\b', r'\bequation\b', r'\bfunction\b',
                r'\bderivative\b', r'\bintegral\b', r'\blimit\b', r'\bvector\b',
                r'\bmatrix\b', r'\btopology\b'
            ],
            "programming": [
                r'\bprogramming\b', r'\bcode\b', r'\balgorithm\b', r'\bfunction\b',
                r'\bvariable\b', r'\bdevelopment\b', r'\bcomputer science\b',
                r'\bpython\b', r'\bjava\b', r'\bc\+\+\b', r'\bjavascript\b',
                r'\bdata structure\b', r'\bclass\b', r'\bobject\b', r'\bmethod\b'
            ],
            "physics": [
                r'\bphysics\b', r'\bmechanics\b', r'\bdynamics\b', r'\bkinematics\b',
                r'\belectromagnetism\b', r'\bthermodynamics\b', r'\bquantum\b',
                r'\brelativity\b', r'\bforce\b', r'\bmomentum\b', r'\benergy\b',
                r'\belectric\b', r'\bmagnetic\b', r'\bwave\b', r'\bparticle\b'
            ]
        }

        # Count matches for each domain
        domain_counts = {}
        for domain_name, patterns in domains.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            domain_counts[domain_name] = count

        # Find domain with highest count
        max_count = max(domain_counts.values())

        if max_count == 0:
            return "unknown", 0.0

        # Get domains with max count
        max_domains = [domain for domain, count in domain_counts.items() if count == max_count]

        if len(max_domains) == 1:
            domain = max_domains[0]
            total = sum(domain_counts.values())
            confidence = max_count / total if total > 0 else 0.0
            return domain, confidence
        else:
            # If tie, return the first domain with medium confidence
            return max_domains[0], 0.5

    def extract_domain_specific_features(self, processed_transcript: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Extract domain-specific features and concepts from the transcript.

        Args:
            processed_transcript: Processed transcript data
            domain: Content domain

        Returns:
            Domain-specific features
        """
        segments = processed_transcript.get("segments", [])
        text = " ".join([s.get("text", "") for s in segments])

        # Extract key concepts based on domain
        key_concepts = self._extract_key_concepts(text, domain, segments)

        # Count theoretical and practical segments
        theoretical_segments = sum(1 for s in segments if s.get("content_type") == "theoretical")
        practical_segments = sum(1 for s in segments if s.get("content_type") == "practical")

        return {
            "domain": domain,
            "key_concepts": key_concepts,
            "theoretical_segments": theoretical_segments,
            "practical_segments": practical_segments
        }

    def _extract_key_concepts(self, text: str, domain: str, segments: List[Dict]) -> List[Dict]:
        """Extract key domain concepts from text."""
        concepts = []

        # Domain-specific concept patterns
        concept_patterns = {
            "mathematics": [
                (r'(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:theorem|equation|formula|law)', True),
                (r'(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:function|series|transform)', True),
                (r'(?:solving|calculate|compute|find)\s+(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})', False)
            ],
            "programming": [
                (r'(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:algorithm|data structure|paradigm)', True),
                (r'(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:function|method|pattern)', True),
                (r'(?:implementing|coding|writing)\s+(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})', False)
            ],
            "physics": [
                (r'(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:law|principle|effect)', True),
                (r'(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:field|force|energy)', True),
                (r'(?:measuring|calculating|observing)\s+(?:the|a)\s+([a-z]+(?:\s+[a-z]+){0,2})', False)
            ]
        }

        # Domain-specific terms
        domain_terms = {
            "mathematics": [
                ("calculus", True), ("algebra", True), ("geometry", True),
                ("function", True), ("derivative", True), ("integral", True),
                ("theorem", True), ("equation", True), ("vector", True),
                ("matrix", True), ("probability", True), ("statistics", True),
                ("calculation", False), ("solve", False), ("problem", False)
            ],
            "programming": [
                ("algorithm", True), ("data structure", True), ("object-oriented", True),
                ("function", True), ("method", True), ("class", True),
                ("inheritance", True), ("recursion", True), ("loop", True),
                ("code", False), ("implement", False), ("debug", False),
                ("compile", False), ("program", False), ("application", False)
            ],
            "physics": [
                ("mechanics", True), ("quantum", True), ("relativity", True),
                ("electromagnetism", True), ("thermodynamics", True), ("energy", True),
                ("force", True), ("particle", True), ("field", True),
                ("experiment", False), ("measure", False), ("observe", False),
                ("laboratory", False), ("instrument", False), ("equipment", False)
            ]
        }

        # Extract concepts from patterns
        if domain in concept_patterns:
            for pattern, is_theoretical in concept_patterns[domain]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    concept_text = match.group(1)
                    if len(concept_text) > 2:  # Minimum concept length
                        concepts.append({
                            "text": concept_text,
                            "theoretical": is_theoretical,
                            "frequency": 1,
                            "domain": domain
                        })

        # Add domain-specific terms if found in text
        if domain in domain_terms:
            for term, is_theoretical in domain_terms[domain]:
                term_lower = term.lower()
                count = len(re.findall(r'\b' + re.escape(term_lower) + r'\b', text.lower()))
                if count > 0:
                    # Check if this term is already in concepts list
                    found = False
                    for c in concepts:
                        if term_lower == c["text"].lower():
                            c["frequency"] += count
                            found = True
                            break

                    if not found:
                        concepts.append({
                            "text": term,
                            "theoretical": is_theoretical,
                            "frequency": count,
                            "domain": domain
                        })

        # Also check each segment individually for better classification
        for segment in segments:
            segment_text = segment.get("text", "").lower()
            segment_type = segment.get("content_type")
            is_theoretical = segment_type == "theoretical"

            # Look for multi-word concepts in theoretical segments
            if is_theoretical and len(segment_text.split()) > 3:
                # Extract noun phrases (simplified approach)
                noun_phrases = re.findall(r'(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+){1,2})', segment_text)
                for np in noun_phrases:
                    if len(np) > 5:  # Minimum length for a concept
                        # Check if already added
                        found = False
                        for c in concepts:
                            if np.lower() == c["text"].lower():
                                c["frequency"] += 1
                                found = True
                                break

                        if not found:
                            concepts.append({
                                "text": np,
                                "theoretical": is_theoretical,
                                "frequency": 1,
                                "domain": domain
                            })

        # Remove duplicates and sort by frequency
        unique_concepts = []
        seen_texts = set()

        for concept in sorted(concepts, key=lambda x: x["frequency"], reverse=True):
            text = concept["text"].lower()
            if text not in seen_texts and len(text) > 2:
                seen_texts.add(text)
                unique_concepts.append(concept)

        return unique_concepts


class ContentIndexer:
    """Coordinate the end-to-end content indexing process."""

    def __init__(self, api_key: str, output_dir: str = "output"):
        """
        Initialize with API key.

        Args:
            api_key: YouTube Data API key
            output_dir: Directory for saving output
        """
        self.youtube_extractor = YouTubeDataExtractor(api_key)
        self.transcript_processor = TranscriptProcessor()
        self.theory_practice_classifier = TheoryPracticeClassifier()
        self.domain_classifier = DomainClassifier()
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def process_video(self, url: str) -> Dict[str, Any]:
        """
        Process a single YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            Processing result
        """
        start_time = time.time()

        # Validate URL and extract video ID
        is_valid, video_id = self.youtube_extractor.validate_video_url(url)
        if not is_valid or not video_id:
            return {
                "status": "error",
                "error": f"Invalid YouTube URL: {url}",
                "video_url": url
            }

        logger.info(f"Processing video: {url} (ID: {video_id})")

        try:
            # Step 1: Extract video metadata
            metadata = self.youtube_extractor.extract_video_metadata(video_id)
            if "error" in metadata:
                return {
                    "status": "error",
                    "error": metadata["error"],
                    "video_id": video_id,
                    "video_url": url
                }

            # Step 2: Extract transcript
            transcript = self.youtube_extractor.extract_transcript(video_id)
            if isinstance(transcript, dict) and "error" in transcript:
                return {
                    "status": "error",
                    "error": transcript["error"],
                    "video_id": video_id,
                    "video_url": url,
                    "metadata": metadata
                }

            # Step 3: Process transcript
            processed_transcript = self.transcript_processor.process_transcript(transcript, metadata)

            # Step 4: Classify domain if not already determined
            if metadata.get("domain") == "unknown" or metadata.get("domain_confidence", 0) < 0.5:
                domain, confidence = self.domain_classifier.classify_transcript(processed_transcript)
                metadata["domain"] = domain
                metadata["domain_confidence"] = confidence

            # Step 5: Extract domain-specific features
            domain_features = self.domain_classifier.extract_domain_specific_features(
                processed_transcript, metadata["domain"]
            )

            # Step 6: Classify theory vs practice
            theory_practice_results = self.theory_practice_classifier.classify_transcript(processed_transcript)

            # Step 7: Extract theory-practice patterns
            theory_practice_patterns = self.theory_practice_classifier.extract_theory_practice_patterns(
                processed_transcript
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare result
            result = {
                "status": "completed",
                "video_id": video_id,
                "video_url": url,
                "metadata": metadata,
                "transcript": processed_transcript,
                "domain_features": domain_features,
                "theory_practice_results": theory_practice_results,
                "theory_practice_patterns": theory_practice_patterns,
                "processing_time": processing_time
            }

            # Save results to file
            self._save_result(result)

            logger.info(f"Successfully processed video {video_id} in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "video_id": video_id,
                "video_url": url
            }

    def process_batch(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process multiple videos in batch.

        Args:
            urls: List of YouTube video URLs

        Returns:
            Batch processing results
        """
        batch_id = f"batch_{int(time.time())}"
        start_time = time.time()

        logger.info(f"Starting batch processing {batch_id} with {len(urls)} videos")

        results = []
        successful = 0
        failed = 0

        for i, url in enumerate(urls):
            logger.info(f"[{i+1}/{len(urls)}] Processing: {url}")

            try:
                result = self.process_video(url)
                results.append(result)

                if result.get("status") == "completed":
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                failed += 1
                results.append({
                    "status": "error",
                    "error": str(e),
                    "video_url": url
                })

        # Prepare batch summary
        total_time = time.time() - start_time

        batch_summary = {
            "batch_id": batch_id,
            "total_videos": len(urls),
            "successful": successful,
            "failed": failed,
            "total_processing_time": total_time,
            "average_processing_time": total_time / max(1, len(urls)),
            "results": results
        }

        # Save batch summary
        self._save_batch_summary(batch_summary)

        logger.info(f"Batch processing completed: {successful}/{len(urls)} videos processed successfully")
        return batch_summary

    def _save_result(self, result: Dict[str, Any]) -> None:
        """Save processing result to file."""
        video_id = result.get("video_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{video_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved result to {filepath}")
        except Exception as e:
            logger.error(f"Error saving result: {e}")

    def _save_batch_summary(self, batch_summary: Dict[str, Any]) -> None:
        """Save batch processing summary to file."""
        batch_id = batch_summary.get("batch_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"batch_{batch_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        try:
            # Create a version with reduced size by removing large transcript data
            summary_to_save = batch_summary.copy()
            for result in summary_to_save.get("results", []):
                if "transcript" in result:
                    # Keep only minimal transcript information
                    transcript = result["transcript"]
                    if isinstance(transcript, dict) and "segments" in transcript:
                        segment_count = len(transcript["segments"])
                        result["transcript"] = {
                            "segment_count": segment_count,
                            "language": transcript.get("language", "")
                        }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary_to_save, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved batch summary to {filepath}")
        except Exception as e:
            logger.error(f"Error saving batch summary: {e}")


def format_timecode(seconds):
    """Format seconds into a timecode (HH:MM:SS)."""
    if seconds is None:
        return "00:00:00"

    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def print_theory_practice_analysis(result: Dict[str, Any]):
    """Print analysis of theory vs practice for a video."""
    if result.get("status") != "completed":
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    video_id = result.get("video_id", "unknown")
    metadata = result.get("metadata", {})
    theory_practice = result.get("theory_practice_results", {})
    domain_features = result.get("domain_features", {})

    print("\n===== VIDEO INFORMATION =====")
    print(f"Title: {metadata.get('title', 'Unknown')}")
    print(f"Channel: {metadata.get('channel', 'Unknown')}")
    print(f"Domain: {metadata.get('domain', 'Unknown')} (confidence: {metadata.get('domain_confidence', 0):.2f})")
    print(f"URL: https://www.youtube.com/watch?v={video_id}")

    print("\n===== THEORY/PRACTICE ANALYSIS =====")
    print(f"Classification: {theory_practice.get('classification', 'Unknown')}")
    print(f"Confidence: {theory_practice.get('confidence', 0):.2f}")

    theoretical = theory_practice.get("theoretical_segments", 0)
    practical = theory_practice.get("practical_segments", 0)
    mixed = theory_practice.get("mixed_segments", 0)
    total = theoretical + practical + mixed

    print(f"Theoretical segments: {theoretical} ({theoretical/max(1,total)*100:.1f}%)")
    print(f"Practical segments: {practical} ({practical/max(1,total)*100:.1f}%)")
    print(f"Mixed segments: {mixed} ({mixed/max(1,total)*100:.1f}%)")
    print(f"Theory/Practice ratio: {theory_practice.get('theory_practice_ratio', 0):.2f}")

    # Print concept analysis
    print("\n===== KEY CONCEPTS =====")
    key_concepts = domain_features.get("key_concepts", [])

    theoretical_concepts = [c for c in key_concepts if c.get("theoretical", False)]
    practical_concepts = [c for c in key_concepts if not c.get("theoretical", False)]

    print(f"Theoretical concepts ({len(theoretical_concepts)}):")
    for i, concept in enumerate(theoretical_concepts[:10], 1):
        print(f"  {i}. {concept.get('text')} (frequency: {concept.get('frequency', 0)})")

    print(f"\nPractical concepts ({len(practical_concepts)}):")
    for i, concept in enumerate(practical_concepts[:10], 1):
        print(f"  {i}. {concept.get('text')} (frequency: {concept.get('frequency', 0)})")

    # Print transitions
    print("\n===== THEORY/PRACTICE TRANSITIONS =====")
    patterns = result.get("theory_practice_patterns", {})
    t2p = patterns.get("theory_to_practice_sequences", [])
    p2t = patterns.get("practice_to_theory_sequences", [])

    print(f"Theory → Practice transitions: {len(t2p)}")
    for i, seq in enumerate(t2p[:3], 1):
        start_time = seq.get("start_time", 0)
        print(f"  {i}. At {format_timecode(start_time)} - https://www.youtube.com/watch?v={video_id}&t={int(start_time)}")

    print(f"\nPractice → Theory transitions: {len(p2t)}")
    for i, seq in enumerate(p2t[:3], 1):
        start_time = seq.get("start_time", 0)
        print(f"  {i}. At {format_timecode(start_time)} - https://www.youtube.com/watch?v={video_id}&t={int(start_time)}")


def process_url_file(filename: str) -> List[str]:
    """
    Process a file containing YouTube URLs.

    Args:
        filename: Path to file with URLs

    Returns:
        List of valid YouTube URLs
    """
    valid_urls = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Skip empty lines and comments
                line = line.strip()
                if line and not line.startswith('#'):
                    # Basic URL validation
                    if 'youtube.com' in line or 'youtu.be' in line:
                        valid_urls.append(line)
                    else:
                        logger.warning(f"Skipping invalid URL: {line}")
    except Exception as e:
        logger.error(f"Error reading URL file: {e}")

    return valid_urls


def batch_process_urls(indexer: ContentIndexer, urls: List[str], max_videos: Optional[int] = None):
    """
    Process multiple YouTube videos and display results.

    Args:
        indexer: ContentIndexer instance
        urls: List of URLs to process
        max_videos: Maximum number of videos to process (None for all)
    """
    if max_videos is not None and max_videos > 0:
        urls = urls[:max_videos]

    batch_result = indexer.process_batch(urls)

    print("\n===== BATCH PROCESSING RESULTS =====")
    print(f"Total videos: {batch_result.get('total_videos', 0)}")
    print(f"Successfully processed: {batch_result.get('successful', 0)}")
    print(f"Failed: {batch_result.get('failed', 0)}")
    print(f"Total processing time: {batch_result.get('total_processing_time', 0):.2f}s")
    print(f"Average processing time: {batch_result.get('average_processing_time', 0):.2f}s")

    # Print summary of each video
    print("\n===== VIDEO SUMMARIES =====")
    for i, result in enumerate(batch_result.get("results", []), 1):
        if result.get("status") == "completed":
            metadata = result.get("metadata", {})
            theory_practice = result.get("theory_practice_results", {})

            print(f"\n{i}. {metadata.get('title', 'Unknown')}")
            print(f"   Domain: {metadata.get('domain', 'Unknown')}")
            print(f"   Classification: {theory_practice.get('classification', 'Unknown')}")
            print(f"   Theory/Practice ratio: {theory_practice.get('theory_practice_ratio', 0):.2f}")
        else:
            print(f"\n{i}. Error: {result.get('error', 'Unknown error')}")


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YouTube Lecture Batch Processor')
    parser.add_argument('input_file', help='File containing YouTube URLs (one per line)')
    parser.add_argument('--output-dir', default='output', help='Directory for output files')
    parser.add_argument('--api-key', help='YouTube Data API key (or set YOUTUBE_API_KEY environment variable)')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Get API key from arguments or environment
    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: No YouTube API key provided. Please set YOUTUBE_API_KEY environment variable or use --api-key.")
        return 1

    try:
        # Create content indexer
        indexer = ContentIndexer(api_key, args.output_dir)

        # Process URL file
        urls = process_url_file(args.input_file)

        if not urls:
            print(f"No valid YouTube URLs found in {args.input_file}")
            return 1

        print(f"Found {len(urls)} YouTube URLs in {args.input_file}")

        # Process batch
        batch_process_urls(indexer, urls, args.max_videos)

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
