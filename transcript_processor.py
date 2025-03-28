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
import os
from typing import List, Dict, Tuple, Counter as CounterType
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Import simplified modules
from cache_manager import cache_get, cache_set
from performance_utils import time_function

# Configure logging
logger = logging.getLogger(__name__)

class TranscriptProcessor:
    """
    Processes raw transcripts into structured text suitable for analysis.
    Enhanced with NLP techniques for improved classification.
    """

    def __init__(self):
        """Initialize the transcript processor with NLP components."""
        # Download necessary NLTK data if not already available
        self._ensure_nltk_resources()

        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()

        # Initialize stemmers for multiple languages
        self.stemmers = {}
        for lang in ['english', 'russian']:
            try:
                self.stemmers[lang[:2]] = SnowballStemmer(lang)
            except:
                logger.warning(f"Could not initialize stemmer for {lang}")

        # Load stopwords for multiple languages
        self.stopwords = {}
        self._load_stopwords()

        # Initialize domain classification models
        self._init_classification_models()

        logger.info("TranscriptProcessor initialized with multilingual NLP components")

    def _ensure_nltk_resources(self):
        """Ensure all required NLTK resources are available."""
        required_resources = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet')
        ]

        for resource, path in required_resources:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)

    def _load_stopwords(self):
        """Load stopwords for multiple languages with extended sets."""
        # Core language stopwords
        languages = {
            'en': 'english',
            'ru': 'russian'
        }

        for code, lang in languages.items():
            try:
                # Load NLTK stopwords
                self.stopwords[code] = set(stopwords.words(lang))

                # Add language-specific common words that should be filtered
                if code == 'en':
                    english_extras = {
                        "uh", "um", "like", "so", "well", "actually", "basically",
                        "literally", "sort", "kind", "really", "very", "quite",
                        "okay", "ok", "yeah", "yes", "no", "right", "let", "just",
                        "gonna", "going", "let's", "now", "here", "there", "this",
                        "that", "these", "those", "will", "shall", "should", "would",
                        "could", "can", "may", "might", "must", "although", "however"
                    }
                    self.stopwords[code].update(english_extras)

                elif code == 'ru':
                    russian_extras = {
                        "это", "что", "как", "так", "вот", "просто", "если",
                        "там", "здесь", "сейчас", "тут", "ну", "да", "нет", "уже",
                        "значит", "такой", "такая", "такое", "давайте", "есть", "был",
                        "была", "были", "будет", "будут", "потому", "ещё", "еще",
                        "нас", "меня", "можно", "всё", "они", "только", "для"
                    }
                    self.stopwords[code].update(russian_extras)
            except:
                logger.warning(f"Failed to load stopwords for {lang}")
                self.stopwords[code] = set()

    def _init_classification_models(self):
        """
        Initialize classification models and related data structures
        with improved multilingual support and domain-specific patterns.
        """
        # Language-specific patterns for theoretical content
        self.theoretical_patterns = {
            'en': [
                r'is defined as',
                r'is called',
                r'refers to',
                r'is known as',
                r'can be described as',
                r'is a concept',
                r'is characterized by',
                r'is understood as',
                r'is formulated as'
            ],
            'ru': [
                r'определяется как',
                r'называется',
                r'обозначает',
                r'известен как',
                r'может быть описан как',
                r'является концепцией',
                r'характеризуется',
                r'понимается как',
                r'формулируется как'
            ]
        }

        # Language-specific patterns for practical content
        self.practical_patterns = {
            'en': [
                r"let['']s",
                r'we (can|will|should|could)',
                r'you (can|will|should|could)',
                r'for example',
                r'as an example',
                r'step by step',
                r'how to',
                r'in practice',
                r'in this example'
            ],
            'ru': [
                r'давайте',
                r'мы (можем|будем|должны|могли)',
                r'вы (можете|будете|должны|могли)',
                r'например',
                r'в качестве примера',
                r'шаг за шагом',
                r'как сделать',
                r'на практике',
                r'в этом примере'
            ]
        }

        # Compile patterns for efficiency
        self.theoretical_regex = {
            lang: re.compile('|'.join(patterns), re.IGNORECASE)
            for lang, patterns in self.theoretical_patterns.items()
        }

        self.practical_regex = {
            lang: re.compile('|'.join(patterns), re.IGNORECASE)
            for lang, patterns in self.practical_patterns.items()
        }

        # Domain-specific linguistic features by language
        self.domain_features = {
            "mathematics": {
                "en": {
                    # Theoretical indicators
                    "theorem": 0.9, "proof": 0.9, "lemma": 0.9, "define": 0.8,
                    "equation": 0.8, "formula": 0.8, "function": 0.7, "property": 0.7,

                    # Practical indicators
                    "calculate": 0.8, "compute": 0.8, "solve": 0.8, "example": 0.7,
                    "problem": 0.7, "find": 0.7, "evaluate": 0.7
                },
                "ru": {
                    # Theoretical indicators
                    "теорема": 0.9, "доказательство": 0.9, "лемма": 0.9, "определение": 0.8,
                    "уравнение": 0.8, "формула": 0.8, "функция": 0.7, "свойство": 0.7,

                    # Practical indicators
                    "вычислить": 0.8, "рассчитать": 0.8, "решить": 0.8, "пример": 0.7,
                    "задача": 0.7, "найти": 0.7, "определить": 0.7
                }
            },
            "programming": {
                "en": {
                    # Theoretical indicators
                    "algorithm": 0.8, "complexity": 0.85, "paradigm": 0.9,
                    "architecture": 0.8, "pattern": 0.7, "principle": 0.8,

                    # Practical indicators
                    "code": 0.9, "implement": 0.85, "function": 0.7, "class": 0.7,
                    "debug": 0.9, "run": 0.8, "execute": 0.8
                },
                "ru": {
                    # Theoretical indicators
                    "алгоритм": 0.8, "сложность": 0.85, "парадигма": 0.9,
                    "архитектура": 0.8, "шаблон": 0.7, "принцип": 0.8,

                    # Practical indicators
                    "код": 0.9, "реализовать": 0.85, "функция": 0.7, "класс": 0.7,
                    "отладка": 0.9, "запустить": 0.8, "выполнить": 0.8
                }
            },
            "physics": {
                "en": {
                    # Theoretical indicators
                    "theory": 0.9, "law": 0.9, "principle": 0.9, "constant": 0.8,
                    "equation": 0.8, "field": 0.7, "force": 0.7, "energy": 0.7,

                    # Practical indicators
                    "experiment": 0.9, "measure": 0.8, "observation": 0.8,
                    "calculate": 0.8, "predict": 0.7, "demonstrate": 0.8
                },
                "ru": {
                    # Theoretical indicators
                    "теория": 0.9, "закон": 0.9, "принцип": 0.9, "константа": 0.8,
                    "уравнение": 0.8, "поле": 0.7, "сила": 0.7, "энергия": 0.7,

                    # Practical indicators
                    "эксперимент": 0.9, "измерение": 0.8, "наблюдение": 0.8,
                    "рассчитать": 0.8, "предсказать": 0.7, "демонстрировать": 0.8
                }
            }
        }

        # Initialize TF-IDF vectorizers for domain detection
        self.domain_vectorizers = {}
        self.domain_centroids = {}

        # Load pre-trained domain vectorizers if available
        # This would be implemented in production with saved models
        self._load_domain_models()

    def _load_domain_models(self):
        """Load or train TF-IDF domain classifiers if possible."""
        # This would load pre-trained models in production
        # For now, we'll use a simple training approach
        sample_texts = {
            "mathematics": [
                "Mathematics is the study of numbers, quantity, space, structure, and change.",
                "Calculus is the mathematical study of continuous change.",
                "A derivative measures the sensitivity to change of a function value."
            ],
            "programming": [
                "Programming is the process of creating instructions for computers.",
                "Python is a high-level programming language for general-purpose programming.",
                "Object-oriented programming is a programming paradigm based on objects."
            ],
            "physics": [
                "Physics is the natural science that studies matter and its motion.",
                "Quantum mechanics is a fundamental theory in physics.",
                "Energy is the quantitative property that must be transferred to an object."
            ]
        }

        try:
            # Initialize a general vectorizer
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

            # Fit each domain separately
            for domain, texts in sample_texts.items():
                domain_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                domain_vectors = domain_vectorizer.fit_transform(texts)

                # Create a centroid (average vector)
                if domain_vectors.shape[0] > 0:
                    centroid = np.mean(domain_vectors.toarray(), axis=0)

                    self.domain_vectorizers[domain] = domain_vectorizer
                    self.domain_centroids[domain] = centroid
        except Exception as e:
            logger.warning(f"Failed to initialize domain vectorizers: {e}")

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
        if not language or language not in ['en', 'ru']:
            language = self._detect_language([s.get("text", "") for s in raw_segments[:5]])

        # Normalize to 2-letter code
        language = language[:2]
        logger.info(f"Detected language: {language}")

        # Get domain from metadata or detect it
        domain = video_metadata.get("domain", "unknown")
        if domain == "unknown":
            domain = self._detect_domain([s.get("text", "") for s in raw_segments], language)
            logger.info(f"Detected domain: {domain}")

        # Normalize transcript segments
        normalized_segments = self._normalize_transcript(raw_segments, language)

        # Segment into sentences (when possible)
        try:
            sentence_segments = self._segment_into_sentences(normalized_segments, language)
        except Exception as e:
            logger.warning(f"Error segmenting into sentences: {e}, using original segments")
            sentence_segments = normalized_segments

        # Classify segments as theoretical or practical
        classified_segments = self._classify_segments(sentence_segments, domain, language)

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

    def _detect_language(self, text_samples: List[str]) -> str:
        """
        Detect language from text samples using character frequency analysis.

        Args:
            text_samples: List of text samples

        Returns:
            Language code ('en' or 'ru')
        """
        if not text_samples:
            return 'en'

        # Join samples
        full_text = ' '.join(text_samples)

        # Count Cyrillic characters
        cyrillic_count = sum(1 for c in full_text if 'а' <= c.lower() <= 'я' or c.lower() in 'ёэіїєґў')

        # Count Latin characters
        latin_count = sum(1 for c in full_text if 'a' <= c.lower() <= 'z')

        # Determine language based on character distribution
        if cyrillic_count > latin_count:
            return 'ru'
        else:
            return 'en'

    def _detect_domain(self, text_samples: List[str], language: str) -> str:
        """
        Detect domain using TF-IDF similarity to domain centroids or keyword analysis.

        Args:
            text_samples: List of text samples
            language: Language code

        Returns:
            Domain name
        """
        if not text_samples:
            return 'unknown'

        # Join samples
        full_text = ' '.join(text_samples)

        # Try machine learning approach first (for English content)
        if language == 'en' and self.domain_vectorizers:
            try:
                # Compute similarity to each domain
                similarities = {}
                for domain, vectorizer in self.domain_vectorizers.items():
                    # Transform the text
                    vector = vectorizer.transform([full_text])

                    # Calculate cosine similarity with domain centroid
                    centroid = self.domain_centroids.get(domain)
                    if centroid is not None and vector.shape[1] == len(centroid):
                        # Calculate cosine similarity
                        similarity = np.dot(vector.toarray()[0], centroid) / (
                            np.linalg.norm(vector.toarray()[0]) * np.linalg.norm(centroid) + 1e-10  # Avoid division by zero
                        )
                        similarities[domain] = similarity

                if similarities:
                    # Return domain with highest similarity if above threshold
                    max_domain, max_sim = max(similarities.items(), key=lambda x: x[1])
                    if max_sim > 0.2:  # Threshold for confidence
                        return max_domain
            except Exception as e:
                logger.warning(f"TF-IDF domain detection error: {e}")

        # Fallback: Keyword-based detection (multilingual)
        domain_keywords = {
            "mathematics": {
                "en": ["math", "mathematics", "calculus", "algebra", "geometry", "theorem", "equation", "function"],
                "ru": ["математика", "алгебра", "геометрия", "теорема", "уравнение", "функция"]
            },
            "programming": {
                "en": ["programming", "algorithm", "code", "software", "python", "java", "javascript"],
                "ru": ["программирование", "алгоритм", "код", "программа", "python", "java"]
            },
            "physics": {
                "en": ["physics", "mechanics", "dynamics", "quantum", "relativity", "force", "energy"],
                "ru": ["физика", "механика", "динамика", "квантовая", "сила", "энергия"]
            }
        }

        # Get keywords for detected language, falling back to English if necessary
        lang_key = language if language in ["en", "ru"] else "en"

        # Count keyword occurrences
        domain_scores = {domain: 0 for domain in domain_keywords}
        lowered_text = full_text.lower()

        for domain, keywords_dict in domain_keywords.items():
            keywords = keywords_dict.get(lang_key, keywords_dict.get("en", []))
            for keyword in keywords:
                domain_scores[domain] += lowered_text.count(keyword.lower())

        # Return domain with highest score if any found
        if sum(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]

        return 'unknown'

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

            # Basic text normalization based on language
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
        Segment normalized transcript into sentences with improved language handling.

        Args:
            normalized_segments: List of normalized transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            List of sentence segments
        """
        sentence_segments = []

        for segment in normalized_segments:
            text = segment.get("text", "")

            # Use language-specific sentence tokenization
            try:
                # For Russian, handle specially
                if language == 'ru':
                    try:
                        # Try with Russian-specific tokenizer if available
                        sentences = sent_tokenize(text, language='russian')
                    except:
                        # Fallback for Russian using simple rules
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                else:
                    # For English and other languages
                    sentences = sent_tokenize(text)
            except:
                # Fallback to simple regex for all languages
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

    def _classify_segments(self, segments: List[Dict], domain: str, language: str = 'en') -> List[Dict]:
        """
        Classify segments as theoretical or practical with improved multilingual support.

        Args:
            segments: List of transcript segments
            domain: Content domain
            language: Language code ('en' or 'ru')

        Returns:
            List of classified segments
        """
        classified_segments = []

        # Ensure we have pattern matchers for this language
        lang = language if language in self.theoretical_regex else 'en'

        # Get domain-specific features for this language
        domain_features = {}
        if domain in self.domain_features:
            # Try to get language-specific features first
            if language in self.domain_features[domain]:
                domain_features = self.domain_features[domain][language]
            # Fall back to English if language-specific features not available
            elif 'en' in self.domain_features[domain]:
                domain_features = self.domain_features[domain]['en']

        for segment in segments:
            text = segment.get("text", "")

            # Extract features and classify
            features = self._extract_features(text, language)
            content_type, confidence = self._classify_with_features(features, domain_features, text, domain, language)

            # Create classified segment
            classified_segment = segment.copy()
            classified_segment["content_type"] = content_type
            classified_segment["classification_confidence"] = confidence

            classified_segments.append(classified_segment)

        return classified_segments

    def _extract_features(self, text: str, language: str = 'en') -> Dict:
        """
        Extract NLP features from text for classification with language support.

        Args:
            text: Text to extract features from
            language: Language code

        Returns:
            Dictionary of features
        """
        # Lowercase the text for case-insensitive matching
        text_lower = text.lower()

        # Get correct stopwords and stemmer
        lang_code = language if language in self.stopwords else 'en'
        stop_words = self.stopwords.get(lang_code, set())
        stemmer = self.stemmers.get(lang_code, self.stemmers.get('en', None))

        # Extract tokens
        try:
            tokens = word_tokenize(text_lower)
        except:
            # Fallback tokenization (simple whitespace split)
            tokens = text_lower.split()

        # Remove stopwords and punctuation, then stem tokens
        filtered_tokens = []
        for token in tokens:
            if token not in stop_words and token not in string.punctuation:
                # Apply stemming if available
                try:
                    if stemmer:
                        stemmed = stemmer.stem(token)
                        filtered_tokens.append(stemmed)
                    else:
                        filtered_tokens.append(token)
                except:
                    filtered_tokens.append(token)

        # Count word frequencies
        word_counts = Counter(filtered_tokens)

        return {
            "tokens": filtered_tokens,
            "word_counts": word_counts,
            "text_lower": text_lower
        }

    def _classify_with_features(
        self,
        features: Dict,
        domain_features: Dict,
        text: str,
        domain: str,
        language: str = 'en'
    ) -> Tuple[str, float]:
        """
        Classify text as theoretical or practical using extracted features.

        Args:
            features: Extracted text features
            domain_features: Domain-specific language features
            text: Original text
            domain: Content domain
            language: Language code

        Returns:
            Tuple of (classification, confidence)
        """
        word_counts = features["word_counts"]
        text_lower = features["text_lower"]

        # Get correct language for patterns
        lang = language if language in self.theoretical_regex else 'en'

        # Calculate theoretical and practical scores
        theoretical_score = 0.0
        practical_score = 0.0

        # Score based on linguistic features/domain features
        for word, count in word_counts.items():
            # Check if word is in domain features
            if word in domain_features:
                # Use the feature weight directly - higher weights for theoretical terms
                if domain_features[word] >= 0.75:  # Threshold for theoretical
                    theoretical_score += domain_features[word] * count
                else:
                    practical_score += domain_features[word] * count

        # Score based on syntactic patterns
        if self.theoretical_regex[lang].search(text_lower):
            theoretical_score += 1.5

        if self.practical_regex[lang].search(text_lower):
            practical_score += 1.5

        # Add domain-specific pattern matching
        if domain == "mathematics":
            # Check for mathematical symbols (theoretical)
            math_symbols = ["∫", "∑", "∏", "∀", "∃", "→", "∴", "∵", "≡", "≠", "≤", "≥"]
            if any(symbol in text for symbol in math_symbols):
                theoretical_score += 1.0

            # Check for calculation keywords (practical)
            calc_pattern = r'\b(calculate|compute|find|solve|evaluate|вычислить|рассчитать|решить)\b'
            if re.search(calc_pattern, text_lower):
                practical_score += 1.0

        elif domain == "programming":
            # Check for code blocks or snippets (practical)
            code_pattern = r'(```|def\s+\w+\(|class\s+\w+:|if\s+.*:|for\s+.*:|while\s+.*:)'
            if re.search(code_pattern, text):
                practical_score += 1.5

            # Check for conceptual programming terms (theoretical)
            concept_pattern = r'\b(complexity|algorithm design|design pattern|architecture|сложность|проектирование алгоритмов|шаблон проектирования)\b'
            if re.search(concept_pattern, text_lower):
                theoretical_score += 1.0

        elif domain == "physics":
            # Check for physics equations (theoretical)
            equation_pattern = r'[A-Za-z]+\s*=\s*[A-Za-z0-9\s\+\-\*\/\(\)]+'
            if re.search(equation_pattern, text):
                theoretical_score += 0.8

            # Check for experimental indicators (practical)
            experiment_pattern = r'\b(experiment|measurement|observation|data|result|эксперимент|измерение|наблюдение|данные|результат)\b'
            if re.search(experiment_pattern, text_lower):
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
