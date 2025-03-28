"""
Enhanced transcript processor for the Lecture Video Content Indexer.
Handles processing of raw transcripts into structured text suitable for analysis.
Implements improved NLP-based classification for theoretical vs practical content.
"""

import re
import uuid
import logging
import nltk
import string
from typing import List, Dict, Tuple, Counter as CounterType
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import simplified modules
from cache_manager import cache_get, cache_set
from performance_utils import time_function

# Configure logging
logger = logging.getLogger(__name__)

# Download necessary NLTK data if not already available
print("Downloading required resources...")
required_resources = ['punkt_tab', 'stopwords', 'wordnet']

# First make sure we have the basic resources
for resource in required_resources:
    try:
        nltk.data.find(f"{'corpora' if resource != 'punkt' else 'tokenizers'}/{resource}")
    except LookupError:
        nltk.download(resource)

# Make sure we have punkt_tab for Russian language tokenization
try:
    nltk.data.find('tokenizers/punkt/russian.pickle')
except LookupError:
    try:
        # This will download punkt which includes the Russian data
        nltk.download('punkt', quiet=False)
    except Exception as e:
        print(f"Error downloading Russian tokenization model: {e}")
        print("The system may not be able to properly tokenize Russian text.")

class TranscriptProcessor:
    """
    Processes raw transcripts into structured text suitable for analysis.
    Enhanced with NLP techniques for improved classification.
    """

    def __init__(self):
        """Initialize the transcript processor with NLP components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize domain classification models
        self._init_classification_models()

        logger.info("TranscriptProcessor initialized with NLP components")

    def _init_classification_models(self):
        """Initialize classification models and related data structures."""
        # Feature weights for theoretical and practical classification
        # These weights are learned from analyzing educational content
        self.theoretical_features = {
            # Conceptual terms
            'concept': 0.8, 'theory': 0.9, 'definition': 0.85, 'defined': 0.7,
            'theorem': 0.95, 'proof': 0.9, 'formula': 0.8, 'equation': 0.8,
            'principle': 0.85, 'law': 0.9, 'framework': 0.75, 'model': 0.7,
            'abstract': 0.8, 'method': 0.6, 'approach': 0.6, 'hypothesis': 0.85,

            # Explanation indicators
            'explain': 0.7, 'understand': 0.75, 'means': 0.6, 'represents': 0.7,
            'refers': 0.7, 'describes': 0.65, 'introduces': 0.7, 'establishes': 0.75,

            # Academic language
            'thus': 0.7, 'hence': 0.7, 'therefore': 0.7, 'furthermore': 0.65,
            'consequently': 0.7, 'whereas': 0.65, 'nevertheless': 0.6
        }

        self.practical_features = {
            # Application terms
            'example': 0.85, 'practice': 0.9, 'application': 0.8, 'implement': 0.85,
            'use': 0.7, 'apply': 0.8, 'solve': 0.8, 'demonstration': 0.75,
            'exercise': 0.85, 'problem': 0.7, 'case': 0.6, 'study': 0.6,
            'calculate': 0.8, 'compute': 0.8, 'measure': 0.7, 'experiment': 0.85,

            # Action indicators
            'let': 0.7, 'try': 0.8, 'now': 0.6, 'step': 0.7, 'first': 0.6,
            'next': 0.6, 'then': 0.6, 'finally': 0.6, 'build': 0.75,
            'create': 0.75, 'develop': 0.7, 'run': 0.7, 'execute': 0.75,

            # Interactive language
            'you': 0.7, 'we': 0.6, 'can': 0.6, 'should': 0.6, 'must': 0.6,
            'will': 0.6, 'going': 0.6, 'see': 0.6
        }

        # Domain-specific feature adjustments
        self.domain_specific_features = {
            "mathematics": {
                # Additional theoretical terms in mathematics
                'theorem': 0.9, 'lemma': 0.9, 'corollary': 0.9, 'proof': 0.9,
                'define': 0.8, 'axiom': 0.9, 'postulate': 0.9, 'conjecture': 0.85,

                # Mathematical notation indicators (often theoretical)
                'let': 0.7, 'assume': 0.8, 'given': 0.7, 'suppose': 0.8,

                # Practical indicators in mathematics
                'solve': 0.8, 'calculate': 0.8, 'compute': 0.8, 'evaluate': 0.8,
                'simplify': 0.75, 'factorize': 0.75, 'expand': 0.7
            },
            "programming": {
                # Theoretical programming concepts
                'algorithm': 0.8, 'complexity': 0.85, 'paradigm': 0.9,
                'architecture': 0.8, 'pattern': 0.7, 'principle': 0.8,

                # Practical programming terms
                'code': 0.9, 'implement': 0.85, 'function': 0.7, 'class': 0.7,
                'method': 0.7, 'variable': 0.7, 'loop': 0.8, 'compile': 0.8,
                'debug': 0.9, 'run': 0.8, 'execute': 0.8, 'output': 0.7
            },
            "physics": {
                # Theoretical physics terms
                'theory': 0.9, 'law': 0.9, 'principle': 0.9, 'constant': 0.8,
                'equation': 0.8, 'field': 0.7, 'force': 0.7, 'energy': 0.7,

                # Practical physics indicators
                'experiment': 0.9, 'measure': 0.8, 'observation': 0.8,
                'calculate': 0.8, 'predict': 0.7, 'demonstrate': 0.8,
                'verify': 0.8, 'simulate': 0.8
            }
        }

        # Syntactic patterns (regex patterns) that indicate theoretical or practical content
        self.theoretical_patterns = [
            r'is defined as',
            r'is called',
            r'refers to',
            r'is known as',
            r'can be described as',
            r'is a concept',
            r'is characterized by',
            r'is understood as',
            r'is formulated as'
        ]

        self.practical_patterns = [
            r"let['']s",
            r'we (can|will|should|could)',
            r'you (can|will|should|could)',
            r'for example',
            r'as an example',
            r'step by step',
            r'how to',
            r'in practice',
            r'in this example'
        ]

        # Compile patterns for efficiency
        self.theoretical_regex = re.compile('|'.join(self.theoretical_patterns), re.IGNORECASE)
        self.practical_regex = re.compile('|'.join(self.practical_patterns), re.IGNORECASE)

    @time_function(5000)  # Log warning if takes more than 5 seconds
    def process_transcript(self, raw_segments: List[Dict], video_metadata: Dict) -> Dict:
        """
        Process raw transcript segments into a structured format.

        Args:
            raw_segments: List of raw transcript segments
            video_metadata: Video metadata dictionary

        Returns:
            Dictionary containing processed transcript data
        """
        video_id = video_metadata.get("video_id", "")

        # Check cache first
        cache_key = f"processed_transcript_{video_id}"
        cached_result = cache_get("transcript", cache_key)
        if cached_result:
            logger.info(f"Using cached processed transcript for video {video_id}")
            return cached_result

        if not raw_segments:
            logger.warning("Empty transcript provided")
            result = {
                "segments": [],
                "language": "en",
                "domain": "unknown",
                "video_id": video_id
            }
            return result

        # Determine language from first segment or metadata
        language = raw_segments[0].get("language", video_metadata.get("language", "en"))
        if not language:
            language = "en"  # Default to English

        # Domain from metadata
        domain = video_metadata.get("domain", "unknown")

        # Normalize transcript segments
        normalized_segments = self._normalize_transcript(raw_segments, language)

        # Segment into sentences (when possible)
        try:
            sentence_segments = self._segment_into_sentences(normalized_segments, language)
        except Exception as e:
            logger.warning(f"Error segmenting into sentences: {e}, using original segments")
            sentence_segments = normalized_segments

        # Classify segments as theoretical or practical
        classified_segments = self._classify_segments(sentence_segments, domain)

        # Combine results
        result = {
            "segments": classified_segments,
            "language": language,
            "domain": domain,
            "video_id": video_id
        }

        # Cache the result
        cache_set("transcript", cache_key, result)

        logger.info(f"Processed transcript with {len(classified_segments)} segments")
        return result

    def _normalize_transcript(self, raw_segments: List[Dict], language: str) -> List[Dict]:
        """
        Normalize raw transcript segments.

        Args:
            raw_segments: List of raw transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            List of normalized transcript segments
        """
        normalized_segments = []

        for segment in raw_segments:
            text = segment.get("text", "")

            # Skip empty segments
            if not text.strip():
                continue

            # Basic text normalization
            if language == "ru":
                normalized_text = self._normalize_russian_text(text)
            else:
                normalized_text = self._normalize_english_text(text)

            # Create normalized segment
            normalized_segment = {
                "id": str(uuid.uuid4()),
                "start_time": segment.get("start", 0),
                "end_time": segment.get("start", 0) + segment.get("duration", 0),
                "text": normalized_text,
                "language": language
            }

            normalized_segments.append(normalized_segment)

        return normalized_segments

    def _segment_into_sentences(self, normalized_segments: List[Dict], language: str) -> List[Dict]:
        """
        Segment normalized transcript into sentences.

        Args:
            normalized_segments: List of normalized transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            List of sentence segments
        """
        sentence_segments = []

        for segment in normalized_segments:
            text = segment.get("text", "")

            # Use language-specific sentence tokenization when possible
            try:
                # For Russian, use a simple split approach if NLTK tokenizer fails
                if language == 'ru':
                    try:
                        sentences = sent_tokenize(text, language='russian')
                    except:
                        # Fallback for Russian
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                else:
                    sentences = sent_tokenize(text)
            except:
                # Fallback to simple splitting on sentence terminators
                sentences = re.split(r'(?<=[.!?])\s+', text)

            # If no sentences were detected, use the whole segment as one sentence
            if not sentences:
                sentences = [text]

            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time

            # Create sentence segments with interpolated timestamps
            for i, sentence in enumerate(sentences):
                # Skip empty sentences
                if not sentence.strip():
                    continue

                # Estimate time position proportionally to text length
                sentence_length = len(sentence)
                total_length = sum(len(s) for s in sentences)

                if total_length == 0:
                    # Avoid division by zero
                    sentence_start = start_time
                    sentence_end = end_time
                else:
                    prev_length = sum(len(s) for s in sentences[:i])

                    # Calculate start and end times
                    sentence_start = start_time + (duration * prev_length / total_length)
                    sentence_end = sentence_start + (duration * sentence_length / total_length)

                    # Ensure the last sentence ends at the segment end time
                    if i == len(sentences) - 1:
                        sentence_end = end_time

                # Create sentence segment
                sentence_segment = {
                    "id": str(uuid.uuid4()),
                    "start_time": sentence_start,
                    "end_time": sentence_end,
                    "text": sentence.strip(),
                    "language": language,
                    "original_segment_id": segment.get("id")
                }

                sentence_segments.append(sentence_segment)

        return sentence_segments

    def _classify_segments(self, segments: List[Dict], domain: str) -> List[Dict]:
        """
        Classify segments as theoretical or practical.

        Args:
            segments: List of transcript segments
            domain: Content domain

        Returns:
            List of classified segments
        """
        classified_segments = []

        # Get domain-specific feature adjustments
        domain_features = self.domain_specific_features.get(domain, {})

        for segment in segments:
            text = segment.get("text", "")

            # Extract features and classify
            features = self._extract_features(text)
            content_type, confidence = self._classify_with_features(features, domain_features, text, domain)

            # Create classified segment
            classified_segment = segment.copy()
            classified_segment["content_type"] = content_type
            classified_segment["classification_confidence"] = confidence

            classified_segments.append(classified_segment)

        return classified_segments

    def _extract_features(self, text: str) -> Dict:
        """
        Extract NLP features from text for classification.

        Args:
            text: Text to extract features from

        Returns:
            Dictionary of features
        """
        # Lowercase the text for case-insensitive matching
        text_lower = text.lower()

        # Extract tokens and lemmatize
        tokens = word_tokenize(text_lower)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words and token not in string.punctuation]

        # Count word frequencies
        word_counts = Counter(tokens)

        return {
            "tokens": tokens,
            "word_counts": word_counts,
            "text_lower": text_lower
        }

    def _classify_with_features(
        self,
        features: Dict,
        domain_features: Dict,
        text: str,
        domain: str
    ) -> Tuple[str, float]:
        """
        Classify text as theoretical or practical using extracted features.

        Args:
            features: Extracted text features
            domain_features: Domain-specific features
            text: Original text
            domain: Content domain

        Returns:
            Tuple of (classification, confidence)
        """
        word_counts = features["word_counts"]
        text_lower = features["text_lower"]

        # Calculate theoretical and practical scores
        theoretical_score = 0.0
        practical_score = 0.0

        # Score based on lexical features (weighted word matching)
        for word, count in word_counts.items():
            # Check domain-specific weights first, then fall back to general weights
            theo_weight = domain_features.get(word, self.theoretical_features.get(word, 0))
            prac_weight = domain_features.get(word, self.practical_features.get(word, 0))

            theoretical_score += theo_weight * count
            practical_score += prac_weight * count

        # Score based on syntactic patterns
        if self.theoretical_regex.search(text_lower):
            theoretical_score += 1.5

        if self.practical_regex.search(text_lower):
            practical_score += 1.5

        # Add domain-specific classification logic
        if domain == "mathematics":
            # Check for mathematical symbols (often theoretical)
            if any(symbol in text for symbol in ["∫", "∑", "∏", "∀", "∃", "→", "∴", "∵", "≡", "≠", "≤", "≥"]):
                theoretical_score += 1.0

            # Check for calculation keywords (practical)
            if re.search(r'\b(calculate|compute|find|solve|evaluate)\b', text_lower):
                practical_score += 1.0

        elif domain == "programming":
            # Check for code blocks or snippets (practical)
            if re.search(r'(```|def\s+\w+\(|class\s+\w+:|if\s+.*:|for\s+.*:|while\s+.*:)', text):
                practical_score += 1.5

            # Check for conceptual programming terms (theoretical)
            if re.search(r'\b(complexity|algorithm design|design pattern|architecture)\b', text_lower):
                theoretical_score += 1.0

        elif domain == "physics":
            # Check for physics equations (theoretical)
            if re.search(r'[A-Za-z]+\s*=\s*[A-Za-z0-9\s\+\-\*\/\(\)]+', text):
                theoretical_score += 0.8

            # Check for experimental indicators (practical)
            if re.search(r'\b(experiment|measurement|observation|data|result)\b', text_lower):
                practical_score += 1.0

        # Normalize scores based on text length to avoid bias towards longer segments
        tokens_count = len(features["tokens"])
        if tokens_count > 0:
            normalization_factor = 1.0 / (0.5 + 0.05 * tokens_count)  # Smooth normalization
            theoretical_score *= normalization_factor
            practical_score *= normalization_factor

        # Determine classification and confidence
        if theoretical_score > practical_score:
            margin = theoretical_score - practical_score
            confidence = min(0.5 + margin / 2, 0.95)  # Cap confidence at 0.95
            return "theoretical", confidence
        elif practical_score > theoretical_score:
            margin = practical_score - theoretical_score
            confidence = min(0.5 + margin / 2, 0.95)  # Cap confidence at 0.95
            return "practical", confidence
        else:
            # If scores are equal, look at other factors like domain default
            if domain == "mathematics":
                # Mathematics tends to be theoretical by default
                return "theoretical", 0.6
            elif domain == "programming":
                # Programming tends to be practical by default
                return "practical", 0.6
            else:
                return "mixed", 0.5

    def _normalize_english_text(self, text: str) -> str:
        """Normalize English text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix common caption errors
        text = text.replace(" i ", " I ")
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Add space after period

        # Remove speaker identifiers like "[Professor]:"
        text = re.sub(r'\[\w+\]:', '', text)

        # Fix ellipses
        text = re.sub(r'\.\.\.+', '...', text)

        # Remove musical notes, applause indicators, etc.
        text = re.sub(r'\[.*?\]', '', text)

        return text.strip()

    def _normalize_russian_text(self, text: str) -> str:
        """Normalize Russian text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix punctuation issues
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Add space after period

        # Remove speaker identifiers like "[Профессор]:"
        text = re.sub(r'\[\w+\]:', '', text)

        # Fix ellipses
        text = re.sub(r'\.\.\.+', '...', text)

        # Remove musical notes, applause indicators, etc.
        text = re.sub(r'\[.*?\]', '', text)

        return text.strip()
