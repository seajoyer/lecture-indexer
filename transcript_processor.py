"""
Redesigned transcript processor for the Lecture Video Content Indexer.
Implements a hybrid global+local processing approach for improved accuracy and performance.
"""

import re
import uuid
import logging
import string
import os
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import Counter
import time
import concurrent.futures
from functools import lru_cache

# NLP libraries - with graceful degradation if not available
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available - using simplified text processing")

try:
    import spacy
    # Load models only when needed to minimize memory usage
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("Spacy not available - using alternative NLP processing")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - using simplified text analysis")

# Import project modules
from cache_manager import cache_get, cache_set
from performance_utils import time_function, Timer

# Configure logging
logger = logging.getLogger(__name__)

# Cache version - increment when processing logic changes
CACHE_VERSION = "1.0.0"

class TranscriptProcessor:
    """
    Processes YouTube video transcripts using a hybrid global+local approach.
    First processes the complete transcript as a unified document to establish
    global context, then applies that knowledge to individual segments.
    """

    def __init__(self):
        """Initialize the transcript processor with NLP components."""
        # Initialize NLP components with lazy loading
        self._nlp_initialized = False
        self._spacy_models = {}
        self._stemmer_cache = {}
        self._stopwords_cache = {}

        # Load basic resources immediately for lightweight initialization
        self._load_basic_resources()

        # Version for cache invalidation
        self.processor_version = CACHE_VERSION

        # Set maximum workers for parallel processing
        self.max_workers = os.cpu_count() or 4

        logger.info("TranscriptProcessor initialized with hybrid processing approach")

    def _load_basic_resources(self):
        """Load minimal language resources for basic operations."""
        # Simple language detection patterns
        self.language_patterns = {
            "en": re.compile(r'[a-zA-Z]'),  # English: Latin characters
            "ru": re.compile(r'[а-яА-ЯёЁ]')  # Russian: Cyrillic characters
        }

        # Basic stopwords for minimal filtering (used even without NLTK)
        self.basic_stopwords = {
            'en': {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'by', 'for', 'with', 'about'},
            'ru': {'и', 'в', 'на', 'с', 'по', 'к', 'у', 'от', 'из', 'для', 'это', 'так', 'что', 'как'}
        }

        # Content classification indicators
        self.classification_indicators = {
            "theoretical": {
                "en": ["definition", "concept", "theory", "principle", "defined as", "refers to", "means"],
                "ru": ["определение", "концепция", "теория", "принцип", "определяется как", "означает"]
            },
            "practical": {
                "en": ["example", "application", "practice", "how to", "implement", "demonstrate", "let's"],
                "ru": ["пример", "применение", "практика", "как", "реализовать", "показать", "давайте"]
            }
        }

        # Educational significance indicators
        self.educational_indicators = {
            "en": ["important", "essential", "key", "fundamental", "core", "critical", "main"],
            "ru": ["важный", "существенный", "ключевой", "фундаментальный", "основной", "критический"]
        }

    def _ensure_nlp_resources(self, language: str = 'en'):
        """
        Ensure NLP resources are available for the specified language.
        Uses lazy initialization to minimize startup time and memory usage.

        Args:
            language: Two-letter language code
        """
        if self._nlp_initialized:
            return

        # Initialize NLTK resources if available
        if NLTK_AVAILABLE:
            try:
                # Download necessary resources
                for resource in ['punkt', 'stopwords', 'wordnet']:
                    try:
                        nltk.data.find(f'tokenizers/{resource}')
                    except LookupError:
                        nltk.download(resource, quiet=True)

                # Initialize stemmers and lemmatizers
                self.lemmatizer = WordNetLemmatizer()

                # Load language-specific resources
                if language == 'en':
                    self._stopwords_cache['en'] = set(stopwords.words('english'))
                    self._stemmer_cache['en'] = SnowballStemmer('english')
                elif language == 'ru':
                    self._stopwords_cache['ru'] = set(stopwords.words('russian'))
                    self._stemmer_cache['ru'] = SnowballStemmer('russian')

            except Exception as e:
                logger.warning(f"Error initializing NLTK resources: {e}")

        # Initialize Spacy models if available
        if SPACY_AVAILABLE and language in ['en', 'ru']:
            try:
                model_name = 'en_core_web_sm' if language == 'en' else 'ru_core_news_sm'
                try:
                    self._spacy_models[language] = spacy.load(model_name)
                except OSError:
                    logger.warning(f"Spacy model {model_name} not found. Make sure it's installed in your Nix environment.")
                    # Don't try to download - it won't work in NixOS
                    raise
            except Exception as e:
                logger.warning(f"Error initializing Spacy model for {language}: {e}")

        # Initialize scikit-learn models if available
        if SKLEARN_AVAILABLE:
            # Initialize TF-IDF vectorizers for domain detection
            self.domain_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

            # Sample data for domain detection - improved with more examples
            self.domain_samples = {
                "mathematics": [
                    "Mathematics is the study of numbers, shapes, quantities and patterns.",
                    "Linear algebra studies vector spaces and linear mappings between them.",
                    "Calculus is the mathematical study of continuous change.",
                    "Geometry is concerned with properties of space and figures.",
                    "Number theory studies the properties of integers and number systems."
                ],
                "physics": [
                    "Physics is the natural science of matter, energy, and motion.",
                    "Quantum mechanics describes nature at the smallest scales of energy.",
                    "Electromagnetic theory explains the interaction between particles.",
                    "Thermodynamics deals with heat and temperature and their relation to energy.",
                    "Relativity describes the relationship between space and time."
                ],
                "programming": [
                    "Programming is the process of creating a set of instructions for computers.",
                    "Algorithms are step-by-step procedures for calculations and data processing.",
                    "Object-oriented programming organizes code around data and objects.",
                    "Data structures are specialized formats for organizing and storing data.",
                    "Software development includes coding, testing, and maintaining source code."
                ]
            }

            # Fit domain vectorizers
            self.domain_vectors = {}
            for domain, texts in self.domain_samples.items():
                try:
                    vectors = self.domain_vectorizer.fit_transform(texts)
                    self.domain_vectors[domain] = vectors.mean(axis=0)
                except:
                    logger.warning(f"Error creating domain vector for {domain}")

        self._nlp_initialized = True
        logger.info(f"NLP resources initialized for language: {language}")

    @time_function(5000)  # Log warning if takes more than 5 seconds
    def process_transcript(self, raw_segments: List[Dict], video_metadata: Dict) -> Dict:
        """
        Process raw transcript segments into a structured format using hybrid approach.
        First processes the complete transcript as a unified document, then
        processes individual segments with that global context.

        Args:
            raw_segments: List of raw transcript segments
            video_metadata: Video metadata dictionary

        Returns:
            Dictionary containing processed transcript data
        """
        video_id = video_metadata.get("video_id", "unknown")

        # Generate cache key with version info
        cache_key = f"processed_transcript_{video_id}_v{self.processor_version}"
        cached_result = cache_get("transcript", cache_key)
        if cached_result:
            logger.info(f"Using cached processed transcript for video {video_id}")
            return cached_result

        if not raw_segments:
            logger.warning("Empty transcript provided")
            result = {
                "segments": [],
                "global_text": "",
                "language": "en",
                "domain": "unknown",
                "video_id": video_id
            }
            return result

        # Create a timer for performance tracking
        process_timer = Timer("transcript_processing")
        process_timer.start()

        # STAGE 1: Global Processing
        # Concatenate all segments to create a unified document
        global_text = " ".join([s.get("text", "") for s in raw_segments if s.get("text")])

        # 1.1: Language Detection (using the full text for better accuracy)
        language = self._detect_language(global_text)
        logger.info(f"Detected language: {language}")

        # 1.2: Ensure NLP resources are loaded for this language
        self._ensure_nlp_resources(language)

        # 1.3: Domain Detection (using the full text)
        domain = video_metadata.get("domain", "unknown")
        if domain == "unknown":
            domain = self._detect_domain(global_text, language)
            logger.info(f"Detected domain: {domain}")

        # 1.4: Global text analysis
        global_analysis = self._analyze_global_text(global_text, language, domain)

        # STAGE 2: Segment Processing
        # 2.1: Normalize segments
        normalized_segments = self._normalize_segments(raw_segments, language)

        # 2.2: Sentence segmentation - convert transcript segments to sentence segments
        sentence_segments = self._create_sentence_segments(normalized_segments, language)

        # 2.3: Classify segments using global context
        classified_segments = self._classify_segments(
            sentence_segments,
            global_analysis,
            domain,
            language
        )

        # Prepare result with both global and segment-level information
        result = {
            "segments": classified_segments,
            "global_analysis": {
                "language": language,
                "domain": domain,
                "word_count": global_analysis.get("word_count", 0),
                "sentence_count": global_analysis.get("sentence_count", 0),
                "key_terms": global_analysis.get("key_terms", []),
                "theoretical_indicators": global_analysis.get("theoretical_indicators", 0),
                "practical_indicators": global_analysis.get("practical_indicators", 0)
            },
            "language": language,
            "domain": domain,
            "video_id": video_id
        }

        # Cache the result
        cache_set("transcript", cache_key, result)

        process_time = process_timer.stop() / 1000  # Convert to seconds
        logger.info(f"Processed transcript for video {video_id} in {process_time:.2f}s")

        return result

    def _detect_language(self, text: str) -> str:
        """
        Detect text language using character frequency analysis.
        Enhanced to be more accurate with mixed-language text.

        Args:
            text: Text to analyze

        Returns:
            Two-letter language code ('en' or 'ru')
        """
        if not text:
            return 'en'  # Default to English for empty text

        # Count character frequencies
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        cyrillic_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))

        # Calculate ratios
        total_chars = max(1, len(re.findall(r'[a-zA-Zа-яА-ЯёЁ]', text)))
        latin_ratio = latin_chars / total_chars
        cyrillic_ratio = cyrillic_chars / total_chars

        # Use thresholds to determine language
        if cyrillic_ratio > 0.4:  # If text is at least 40% Cyrillic
            return 'ru'
        elif latin_ratio > 0.4:  # If text is at least 40% Latin
            return 'en'

        # If no clear pattern, try additional detection methods
        if SPACY_AVAILABLE:
            # Use spaCy's language detection if available
            try:
                # Sample the text (for performance)
                sample = text[:1000]
                from spacy.language import Language
                from spacy_langdetect import LanguageDetector

                # Only set up language detector if needed
                if not hasattr(self, 'language_detector'):
                    Language.factory("language_detector", func=lambda nlp, name: LanguageDetector())
                    # Use English model as base
                    if 'en' not in self._spacy_models:
                        self._spacy_models['en'] = spacy.load("en_core_web_sm")
                    self._spacy_models['en'].add_pipe('language_detector', last=True)

                doc = self._spacy_models['en'](sample)
                detected_lang = doc._.language.get('language')

                if detected_lang in ['en', 'ru']:
                    return detected_lang
            except:
                logger.warning("spaCy language detection failed, using character-based fallback")

        # Default to English if detection is inconclusive
        return 'en'

    def _detect_domain(self, text: str, language: str) -> str:
        """
        Detect the content domain using improved classification techniques.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            Domain name
        """
        if not text:
            return 'unknown'

        # Use TF-IDF and cosine similarity if scikit-learn is available
        if SKLEARN_AVAILABLE:
            try:
                # Transform the text
                text_vector = self.domain_vectorizer.transform([text])

                # Calculate similarity to each domain
                similarities = {}
                for domain, domain_vector in self.domain_vectors.items():
                    similarity = cosine_similarity(text_vector, domain_vector)[0][0]
                    similarities[domain] = similarity

                # Get the most similar domain if confidence is sufficient
                if similarities:
                    best_domain, best_score = max(similarities.items(), key=lambda x: x[1])
                    if best_score > 0.1:  # Minimum confidence threshold
                        return best_domain
            except Exception as e:
                logger.warning(f"TF-IDF domain detection failed: {e}")

        # Fallback: Keyword-based detection
        domain_keywords = {
            "mathematics": {
                "en": ["math", "mathematics", "calculus", "algebra", "geometry", "theorem", "equation"],
                "ru": ["математика", "алгебра", "геометрия", "теорема", "уравнение"]
            },
            "physics": {
                "en": ["physics", "quantum", "mechanics", "energy", "force", "particle", "wave"],
                "ru": ["физика", "квантовый", "механика", "энергия", "сила", "частица", "волна"]
            },
            "programming": {
                "en": ["programming", "code", "algorithm", "function", "class", "object", "data"],
                "ru": ["программирование", "код", "алгоритм", "функция", "класс", "объект", "данные"]
            }
        }

        # Get keywords for detected language, fallback to English if needed
        lang_key = language if language in ["en", "ru"] else "en"

        # Count keyword occurrences with improved algorithm
        domain_scores = {domain: 0 for domain in domain_keywords}
        lowered_text = text.lower()

        for domain, keywords_dict in domain_keywords.items():
            keywords = keywords_dict.get(lang_key, keywords_dict.get("en", []))

            for keyword in keywords:
                # Count occurrences
                count = lowered_text.count(keyword.lower())

                # Weigh multi-word terms higher (they're more specific)
                if " " in keyword:
                    count *= 2

                domain_scores[domain] += count

        # Return domain with highest score
        if sum(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]

        return 'unknown'

    def _analyze_global_text(self, text: str, language: str, domain: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on the global text to extract features
        that will inform segment-level processing.

        Args:
            text: Full transcript text
            language: Detected language code
            domain: Content domain

        Returns:
            Dictionary of global text features
        """
        # Initialize result structure
        analysis = {
            "word_count": 0,
            "sentence_count": 0,
            "key_terms": [],
            "theoretical_indicators": 0,
            "practical_indicators": 0,
            "educational_indicators": 0,
            "domain_terms": [],
            "term_frequencies": {},
            "sentence_lengths": [],
            "average_sentence_length": 0
        }

        if not text.strip():
            return analysis

        # Use spaCy for advanced analysis if available
        if SPACY_AVAILABLE and language in self._spacy_models:
            try:
                # Process text with spaCy (in chunks to manage memory)
                doc = self._process_text_with_spacy(text, language)

                # Extract basic statistics
                analysis["word_count"] = len([token for token in doc if not token.is_punct and not token.is_space])
                analysis["sentence_count"] = len(list(doc.sents))

                # Extract key terms - handle language-specific differences
                key_terms = []

                # Check if noun_chunks attribute is available for this language
                has_noun_chunks = True
                try:
                    # Just test if we can iterate through noun chunks
                    next(iter(doc.noun_chunks), None)
                except (ValueError, AttributeError) as e:
                    has_noun_chunks = False
                    logger.debug(f"Noun chunks not available for language '{language}': {e}")

                if has_noun_chunks:
                    # Use noun chunks for languages that support it (like English)
                    for chunk in doc.noun_chunks:
                        if len(chunk.text) > 2:  # Filter out very short terms
                            key_terms.append(chunk.text.lower())
                else:
                    # For languages without noun chunks (like Russian),
                    # extract nouns and noun+adjective combinations
                    nouns = []
                    for token in doc:
                        # Extract single nouns
                        if token.pos_ == "NOUN" and len(token.text) > 2:
                            nouns.append(token.text.lower())
                            key_terms.append(token.text.lower())

                        # Extract adjective+noun combinations
                        if token.pos_ == "NOUN" and token.i > 0:
                            prev_token = doc[token.i - 1]
                            if prev_token.pos_ == "ADJ":
                                phrase = f"{prev_token.text} {token.text}".lower()
                                key_terms.append(phrase)

                # Count frequencies
                term_counter = Counter(key_terms)
                analysis["term_frequencies"] = {term: count for term, count in term_counter.most_common(50)}
                analysis["key_terms"] = [term for term, _ in term_counter.most_common(20)]

                # Get sentence lengths
                analysis["sentence_lengths"] = [len([t for t in sent if not t.is_punct and not t.is_space])
                                            for sent in doc.sents]
                if analysis["sentence_lengths"]:
                    analysis["average_sentence_length"] = sum(analysis["sentence_lengths"]) / len(analysis["sentence_lengths"])

            except Exception as e:
                logger.warning(f"spaCy analysis failed: {e} - using fallback")
                # Continue with fallback analysis

        # Fallback analysis for when spaCy isn't available or fails
        if not analysis["word_count"]:
            # Basic tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            analysis["word_count"] = len(words)

            # Simple sentence segmentation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            analysis["sentence_count"] = len(sentences)

            # Get word frequencies
            word_counter = Counter(words)
            # Filter out stopwords if available
            if language in self._stopwords_cache:
                word_counter = {word: count for word, count in word_counter.items()
                            if word not in self._stopwords_cache[language]}

            analysis["term_frequencies"] = {term: count for term, count in word_counter.most_common(50)}
            analysis["key_terms"] = [term for term, _ in word_counter.most_common(20)]

        # Count theoretical and practical indicators
        theoretical_patterns = self.classification_indicators["theoretical"].get(language, [])
        practical_patterns = self.classification_indicators["practical"].get(language, [])
        educational_indicators = self.educational_indicators.get(language, [])

        text_lower = text.lower()

        # Count pattern occurrences
        analysis["theoretical_indicators"] = sum(text_lower.count(pattern.lower()) for pattern in theoretical_patterns)
        analysis["practical_indicators"] = sum(text_lower.count(pattern.lower()) for pattern in practical_patterns)
        analysis["educational_indicators"] = sum(text_lower.count(pattern.lower()) for pattern in educational_indicators)

        # Determine overall content type based on indicators ratio
        content_type_ratio = 0.5  # Default balanced
        if analysis["theoretical_indicators"] + analysis["practical_indicators"] > 0:
            content_type_ratio = analysis["theoretical_indicators"] / (analysis["theoretical_indicators"] + analysis["practical_indicators"])

        analysis["content_type_ratio"] = content_type_ratio
        analysis["primary_content_type"] = "theoretical" if content_type_ratio > 0.6 else "practical" if content_type_ratio < 0.4 else "mixed"

        return analysis

    def _process_text_with_spacy(self, text: str, language: str) -> Any:
        """
        Process text with spaCy in a memory-efficient way.

        Args:
            text: Text to process
            language: Language code

        Returns:
            spaCy Doc object
        """
        # Use appropriate spaCy model based on language
        nlp = self._spacy_models.get(language)
        if not nlp:
            raise ValueError(f"No spaCy model available for language: {language}")

        # For very long text, we need to process in chunks to avoid memory issues
        if len(text) > 100000:  # 100KB threshold
            # Split into roughly sentence-sized chunks
            chunks = re.split(r'(?<=[.!?])\s+', text)
            processed_docs = []

            # Process chunks in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_text = " ".join(chunks[i:i+batch_size])
                # For Russian specifically, disable parser to avoid noun_chunks errors
                if language == 'ru':
                    # Temporarily disable parser for Russian to avoid the noun_chunks error
                    with nlp.select_pipes(disable=["parser"]):
                        processed_docs.append(nlp(batch_text))
                else:
                    processed_docs.append(nlp(batch_text))

            # Combine results - need a way to merge spaCy docs
            # For simplicity, we'll just return the first chunk for now
            return processed_docs[0]
        else:
            # For Russian specifically, disable parser to avoid noun_chunks errors if not needed
            if language == 'ru':
                # Disable parser for Russian to avoid the noun_chunks error
                with nlp.select_pipes(disable=["parser"]):
                    return nlp(text)
            else:
                return nlp(text)

    def _normalize_segments(self, raw_segments: List[Dict], language: str) -> List[Dict]:
        """
        Normalize raw transcript segments.

        Args:
            raw_segments: List of raw transcript segments
            language: Language code

        Returns:
            List of normalized segments
        """
        normalized_segments = []

        for segment in raw_segments:
            text = segment.get("text", "")

            # Skip empty segments
            if not text.strip():
                continue

            # Apply language-specific normalization
            if language == "ru":
                normalized_text = self._normalize_russian_text(text)
            else:
                normalized_text = self._normalize_english_text(text)

            # Create normalized segment
            normalized_segment = {
                "id": segment.get("id", str(uuid.uuid4())),
                "start_time": segment.get("start", 0.0),
                "end_time": segment.get("start", 0.0) + segment.get("duration", 0.0),
                "text": normalized_text,
                "language": language,
                "raw_data": segment  # Keep original data for reference
            }

            normalized_segments.append(normalized_segment)

        return normalized_segments

    def _create_sentence_segments(self, normalized_segments: List[Dict], language: str) -> List[Dict]:
        """
        Convert transcript segments into sentence-based segments.
        Improves sentence boundary detection by considering the global context.

        Args:
            normalized_segments: List of normalized transcript segments
            language: Language code

        Returns:
            List of sentence-based segments
        """
        sentence_segments = []

        # Configure sentence tokenization based on language
        tokenize_func = self._get_sentence_tokenizer(language)

        # Process each segment to extract sentences
        for segment in normalized_segments:
            segment_id = segment.get("id", "")
            segment_text = segment.get("text", "")
            start_time = segment.get("start_time", 0.0)
            end_time = segment.get("end_time", 0.0)

            # Skip empty segments
            if not segment_text.strip():
                continue

            # Extract sentences
            sentences = tokenize_func(segment_text)

            # If no sentences were detected, use the whole segment
            if not sentences:
                sentences = [segment_text]

            # Calculate timing for each sentence based on character count
            segment_duration = end_time - start_time
            total_chars = sum(len(s) for s in sentences)

            current_pos = 0
            for sentence in sentences:
                sentence_len = len(sentence)

                # Skip empty sentences
                if not sentence.strip():
                    continue

                # Calculate proportional timing
                if total_chars > 0:
                    sentence_portion = sentence_len / total_chars
                    sentence_duration = segment_duration * sentence_portion
                    sentence_start = start_time + (segment_duration * current_pos / total_chars)
                    sentence_end = sentence_start + sentence_duration
                    current_pos += sentence_len
                else:
                    # Handle empty segments gracefully
                    sentence_start = start_time
                    sentence_end = end_time

                # Create sentence segment
                sentence_segment = {
                    "id": f"{segment_id}_{len(sentence_segments)}",
                    "start_time": sentence_start,
                    "end_time": sentence_end,
                    "text": sentence.strip(),
                    "language": language,
                    "segment_id": segment_id,  # Track the original segment
                    "is_sentence": True  # Flag as a sentence-level segment
                }

                sentence_segments.append(sentence_segment)

        return sentence_segments

    def _get_sentence_tokenizer(self, language: str):
        """
        Get the appropriate sentence tokenizer function for the language.

        Args:
            language: Language code

        Returns:
            Tokenizer function
        """
        # If NLTK is available, use its sentence tokenizer
        if NLTK_AVAILABLE:
            if language == 'ru':
                return lambda text: sent_tokenize(text, language='russian')
            else:
                return lambda text: sent_tokenize(text)

        # If spaCy is available, use its sentence tokenizer
        if SPACY_AVAILABLE and language in self._spacy_models:
            nlp = self._spacy_models[language]
            return lambda text: [sent.text for sent in nlp(text).sents]

        # Fallback to simple regex-based tokenization
        return lambda text: re.split(r'(?<=[.!?])\s+', text)

    def _classify_segments(
        self,
        segments: List[Dict],
        global_analysis: Dict,
        domain: str,
        language: str
    ) -> List[Dict]:
        """
        Classify segments as theoretical or practical, and score educational significance,
        using both local features and global context.

        Args:
            segments: List of sentence segments
            global_analysis: Global text analysis results
            domain: Content domain
            language: Language code

        Returns:
            List of classified segments
        """
        # Extract global context that will inform classification
        global_context = {
            "primary_content_type": global_analysis.get("primary_content_type", "mixed"),
            "content_type_ratio": global_analysis.get("content_type_ratio", 0.5),
            "key_terms": set(global_analysis.get("key_terms", [])),
            "educational_indicators": global_analysis.get("educational_indicators", 0)
        }

        # Determine if parallel processing is beneficial and safe
        # Deactivate parallel processing for Russian to avoid potential spaCy issues
        use_parallel = (len(segments) > 20 and
                    self.max_workers > 1 and
                    language != "ru")  # Skip parallel for Russian

        if use_parallel:
            try:
                # Process segments in parallel for better performance
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Create tasks for each segment
                    futures = {
                        executor.submit(
                            self._classify_single_segment,
                            segment,
                            global_context,
                            domain,
                            language
                        ): segment for segment in segments
                    }

                    # Set a timeout for the entire operation
                    timeout = 60  # 60 seconds timeout to prevent hanging

                    # Collect results as they complete
                    classified_segments = []

                    try:
                        for future in concurrent.futures.as_completed(futures, timeout=timeout):
                            try:
                                classified_segment = future.result(timeout=2)  # 2 second timeout per segment
                                classified_segments.append(classified_segment)
                            except Exception as e:
                                logger.error(f"Error classifying segment: {e}")
                                # If classification fails, include the original segment
                                segment = futures[future]
                                segment["content_type"] = "mixed"
                                segment["classification_confidence"] = 0.5
                                segment["educational_score"] = 0.0
                                classified_segments.append(segment)
                    except concurrent.futures.TimeoutError:
                        logger.warning("Timeout in parallel segment classification, falling back to sequential")
                        # If we got a timeout, use the results we have and process the rest sequentially
                        processed_ids = {s.get("id") for s in classified_segments}
                        remaining = [s for s in segments if s.get("id") not in processed_ids]

                        # Process remaining segments sequentially
                        for segment in remaining:
                            try:
                                classified_segment = self._classify_single_segment(
                                    segment,
                                    global_context,
                                    domain,
                                    language
                                )
                                classified_segments.append(classified_segment)
                            except Exception as e:
                                logger.error(f"Error in sequential fallback: {e}")
                                segment["content_type"] = "mixed"
                                segment["classification_confidence"] = 0.5
                                segment["educational_score"] = 0.0
                                classified_segments.append(segment)

                    # Sort by start time to maintain order
                    classified_segments.sort(key=lambda x: x.get("start_time", 0))
            except Exception as e:
                logger.error(f"Parallel processing error: {e}, falling back to sequential")
                use_parallel = False  # Fall back to sequential

        # If parallel processing is disabled or failed, use sequential processing
        if not use_parallel:
            # Process segments sequentially
            classified_segments = []
            for segment in segments:
                try:
                    classified_segment = self._classify_single_segment(
                        segment,
                        global_context,
                        domain,
                        language
                    )
                    classified_segments.append(classified_segment)
                except Exception as e:
                    logger.error(f"Error classifying segment: {e}")
                    # If classification fails, include the original segment
                    segment["content_type"] = "mixed"
                    segment["classification_confidence"] = 0.5
                    segment["educational_score"] = 0.0
                    classified_segments.append(segment)

        # Second pass: enhance classification with context from adjacent segments
        enhanced_segments = self._enhance_with_context(classified_segments)

        return enhanced_segments

    def _classify_single_segment(
        self,
        segment: Dict,
        global_context: Dict,
        domain: str,
        language: str
    ) -> Dict:
        """
        Classify a single segment based on its content and global context.

        Args:
            segment: Segment dictionary
            global_context: Global context information
            domain: Content domain
            language: Language code

        Returns:
            Classified segment dictionary
        """
        # Copy segment to avoid modifying original
        result = segment.copy()
        text = segment.get("text", "")

        # Skip empty segments
        if not text.strip():
            result["content_type"] = "mixed"
            result["classification_confidence"] = 0.5
            result["educational_score"] = 0.0
            return result

        # Extract features for classification
        features = self._extract_segment_features(text, language)

        # Combine local features with global context
        classification = self._classify_with_features(
            features,
            global_context,
            domain,
            language
        )

        # Calculate educational significance
        educational_score = self._calculate_educational_score(
            text,
            features,
            global_context,
            domain,
            language
        )

        # Update segment with classification
        result["content_type"] = classification["content_type"]
        result["classification_confidence"] = classification["confidence"]
        result["educational_score"] = educational_score
        result["is_educational"] = educational_score > 2.5  # Threshold for educational vs passing mention

        # Save extracted features for debugging/visualization
        result["features"] = {
            "theoretical_score": classification.get("theoretical_score", 0),
            "practical_score": classification.get("practical_score", 0),
            "key_terms": features.get("key_terms", []),
            "domain_terms": features.get("domain_terms", [])
        }

        return result

    def _extract_segment_features(self, text: str, language: str) -> Dict[str, Any]:
        """
        Extract features from a segment for classification.

        Args:
            text: Segment text
            language: Language code

        Returns:
            Dictionary of features
        """
        # Initialize features
        features = {
            "word_count": 0,
            "key_terms": [],
            "domain_terms": [],
            "theoretical_indicators": 0,
            "practical_indicators": 0,
            "educational_indicators": 0,
            "tokens": [],
            "word_counts": {}
        }

        if not text.strip():
            return features

        # Use spaCy for feature extraction if available
        if SPACY_AVAILABLE and language in self._spacy_models:
            try:
                # Process with spaCy
                nlp = self._spacy_models[language]
                doc = nlp(text)

                # Extract tokens
                features["tokens"] = [token.text.lower() for token in doc
                                    if not token.is_punct and not token.is_space]
                features["word_count"] = len(features["tokens"])

                # Extract key terms - handle language-specific differences
                key_terms = []

                # Check if noun_chunks attribute is available
                has_noun_chunks = True
                try:
                    # Just test if we can iterate through noun chunks
                    next(iter(doc.noun_chunks), None)
                except (ValueError, AttributeError):
                    has_noun_chunks = False

                if has_noun_chunks:
                    # Use noun chunks for languages that support it
                    for chunk in doc.noun_chunks:
                        if len(chunk.text) > 2:
                            key_terms.append(chunk.text.lower())
                else:
                    # For languages without noun chunks (like Russian),
                    # extract nouns and noun+adjective combinations
                    for token in doc:
                        # Extract single nouns
                        if token.pos_ == "NOUN" and len(token.text) > 2:
                            key_terms.append(token.text.lower())

                        # Extract adjective+noun combinations
                        if token.pos_ == "NOUN" and token.i > 0:
                            prev_token = doc[token.i - 1]
                            if prev_token.pos_ == "ADJ":
                                phrase = f"{prev_token.text} {token.text}".lower()
                                key_terms.append(phrase)

                features["key_terms"] = key_terms

                # Count words
                features["word_counts"] = Counter(features["tokens"])

            except Exception as e:
                logger.warning(f"spaCy feature extraction failed: {e} - using fallback")

        # Fallback feature extraction
        if not features["word_count"]:
            # Basic tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
            features["tokens"] = tokens
            features["word_count"] = len(tokens)
            features["word_counts"] = Counter(tokens)

        # Count theoretical and practical indicators
        theoretical_patterns = self.classification_indicators["theoretical"].get(language, [])
        practical_patterns = self.classification_indicators["practical"].get(language, [])
        educational_indicators = self.educational_indicators.get(language, [])

        text_lower = text.lower()

        # Count pattern occurrences
        features["theoretical_indicators"] = sum(text_lower.count(pattern.lower()) for pattern in theoretical_patterns)
        features["practical_indicators"] = sum(text_lower.count(pattern.lower()) for pattern in practical_patterns)
        features["educational_indicators"] = sum(text_lower.count(pattern.lower()) for pattern in educational_indicators)

        return features

    def _classify_with_features(
        self,
        features: Dict[str, Any],
        global_context: Dict,
        domain: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Classify segment based on features and global context.

        Args:
            features: Segment features
            global_context: Global context information
            domain: Content domain
            language: Language code

        Returns:
            Classification result dictionary
        """
        # Calculate theoretical and practical scores
        theoretical_score = 0.0
        practical_score = 0.0

        # 1. Score based on indicator patterns
        theoretical_score += features["theoretical_indicators"] * 1.0
        practical_score += features["practical_indicators"] * 1.0

        # 2. Score based on domain-specific keywords
        # Domain-specific keywords for theoretical/practical scoring
        domain_keywords = {
            "physics": {
                "theoretical": {
                    "en": ["theory", "principle", "law", "equation", "model", "concept"],
                    "ru": ["теория", "принцип", "закон", "уравнение", "модель", "концепция"]
                },
                "practical": {
                    "en": ["example", "experiment", "demonstration", "application", "measurement"],
                    "ru": ["пример", "эксперимент", "демонстрация", "применение", "измерение"]
                }
            },
            "mathematics": {
                "theoretical": {
                    "en": ["theorem", "proof", "definition", "lemma", "corollary", "axiom"],
                    "ru": ["теорема", "доказательство", "определение", "лемма", "следствие", "аксиома"]
                },
                "practical": {
                    "en": ["example", "problem", "calculation", "application", "solution"],
                    "ru": ["пример", "задача", "вычисление", "применение", "решение"]
                }
            },
            "programming": {
                "theoretical": {
                    "en": ["concept", "paradigm", "principle", "architecture", "algorithm"],
                    "ru": ["концепция", "парадигма", "принцип", "архитектура", "алгоритм"]
                },
                "practical": {
                    "en": ["code", "implementation", "example", "function", "method"],
                    "ru": ["код", "реализация", "пример", "функция", "метод"]
                }
            }
        }

        # Get domain-specific keywords
        theoretical_keywords = domain_keywords.get(domain, {}).get("theoretical", {}).get(language, [])
        practical_keywords = domain_keywords.get(domain, {}).get("practical", {}).get(language, [])

        # Score based on domain keywords
        text_lower = " ".join(features["tokens"]).lower()
        theoretical_score += sum(text_lower.count(kw.lower()) for kw in theoretical_keywords) * 0.5
        practical_score += sum(text_lower.count(kw.lower()) for kw in practical_keywords) * 0.5

        # 3. Consider global context
        # Bias classification based on global primary content type
        if global_context["primary_content_type"] == "theoretical":
            theoretical_score += 0.5
        elif global_context["primary_content_type"] == "practical":
            practical_score += 0.5

        # 4. Normalize scores based on text length to avoid bias towards longer segments
        word_count = features["word_count"]
        if word_count > 0:
            normalization_factor = 1.0 / (0.5 + 0.05 * min(word_count, 50))  # Cap for very long segments
            theoretical_score *= normalization_factor
            practical_score *= normalization_factor

        # Determine classification and confidence
        content_type = "mixed"
        confidence = 0.5

        if theoretical_score > practical_score:
            content_type = "theoretical"
            margin = theoretical_score - practical_score
            confidence = min(0.5 + margin / 4, 0.95)  # Cap confidence at 0.95
        elif practical_score > theoretical_score:
            content_type = "practical"
            margin = practical_score - theoretical_score
            confidence = min(0.5 + margin / 4, 0.95)  # Cap confidence at 0.95

        return {
            "content_type": content_type,
            "confidence": confidence,
            "theoretical_score": theoretical_score,
            "practical_score": practical_score
        }

    def _calculate_educational_score(
        self,
        text: str,
        features: Dict,
        global_context: Dict,
        domain: str,
        language: str
    ) -> float:
        """
        Calculate an educational significance score for a segment.
        Determines whether this is a substantive explanation vs. passing mention.

        Args:
            text: Segment text
            features: Segment features
            global_context: Global context information
            domain: Content domain
            language: Language code

        Returns:
            Educational score (higher = more educational significance)
        """
        # Start with no educational significance
        educational_score = 0.0

        # Factor 1: Length of content - longer segments tend to have more educational value
        word_count = features["word_count"]
        if word_count > 0:
            # Logarithmic scale to give diminishing returns for very long segments
            import math
            length_score = min(1.0 + math.log10(word_count / 10 + 1), 2.0)
            educational_score += length_score

        # Factor 2: Educational markers presence
        educational_score += features["educational_indicators"] * 0.5

        # Factor 3: Domain-specific term density
        # Higher density of domain terms indicates educational content
        domain_keywords = {
            "physics": {
                "en": ["quantum", "mechanics", "energy", "force", "particle", "wave", "theory"],
                "ru": ["квантовый", "механика", "энергия", "сила", "частица", "волна", "теория"]
            },
            "mathematics": {
                "en": ["theorem", "proof", "equation", "function", "derivative", "integral"],
                "ru": ["теорема", "доказательство", "уравнение", "функция", "производная", "интеграл"]
            },
            "programming": {
                "en": ["algorithm", "function", "class", "object", "method", "variable"],
                "ru": ["алгоритм", "функция", "класс", "объект", "метод", "переменная"]
            }
        }

        # Get domain terms
        domain_terms = domain_keywords.get(domain, {}).get(language, [])

        # Count domain terms in text
        domain_term_count = sum(text.lower().count(term) for term in domain_terms)

        # Calculate domain term density
        if word_count > 0:
            domain_density = domain_term_count / word_count
            domain_score = min(domain_density * 5.0, 2.0)  # Cap at 2.0
            educational_score += domain_score

        # Factor 4: Contains key terms from global context
        segment_terms = set(features["key_terms"])
        global_terms = global_context["key_terms"]

        # If segment contains global key terms, it's more likely to be educational
        shared_terms = segment_terms.intersection(global_terms)
        if shared_terms:
            term_score = min(len(shared_terms) * 0.3, 1.0)
            educational_score += term_score

        # Factor 5: Contains terminology patterns that suggest explanation
        explanation_patterns = {
            'en': [
                r'(is|are) defined as', r'refers to', r'means that', r'represents',
                r'consists of', r'comprises', r'described as', r'characterized by',
                r'explanation of', r'understanding of', r'concept of'
            ],
            'ru': [
                r'определяется как', r'означает', r'обозначает', r'представляет',
                r'состоит из', r'включает в себя', r'описывается как', r'характеризуется',
                r'объяснение', r'понимание', r'концепция'
            ]
        }

        # Count explanatory patterns
        explanation_patterns_lang = explanation_patterns.get(language, explanation_patterns['en'])
        explanation_score = 0

        for pattern in explanation_patterns_lang:
            if re.search(pattern, text.lower()):
                explanation_score += 0.5

        educational_score += min(explanation_score, 1.5)  # Cap at 1.5

        return educational_score

    def _enhance_with_context(self, segments: List[Dict]) -> List[Dict]:
        """
        Enhance segment classification by considering context from adjacent segments.

        Args:
            segments: List of classified segments

        Returns:
            Enhanced segments with improved classification
        """
        if len(segments) <= 1:
            return segments

        enhanced_segments = []

        # Use a sliding window approach to consider context
        for i, segment in enumerate(segments):
            # Get original values
            content_type = segment.get("content_type", "mixed")
            confidence = segment.get("classification_confidence", 0.5)
            educational_score = segment.get("educational_score", 0.0)

            # Get preceding and following segments when available
            preceding = segments[i-1] if i > 0 else None
            following = segments[i+1] if i < len(segments)-1 else None

            # Enhance confidence based on adjacent segments with same classification
            if preceding and following:
                # If both adjacent segments have same classification, boost confidence
                if (preceding.get("content_type") == content_type and
                    following.get("content_type") == content_type):
                    confidence = min(confidence + 0.15, 0.95)

                # If part of a continuous educational section, boost educational score
                if (preceding.get("is_educational", False) and
                    following.get("is_educational", False)):
                    educational_score += 0.5
            elif preceding:
                # If only preceding segment matches, small boost
                if preceding.get("content_type") == content_type:
                    confidence = min(confidence + 0.05, 0.95)

                # Educational content boost if preceded by educational content
                if preceding.get("is_educational", False):
                    educational_score += 0.2
            elif following:
                # If only following segment matches, small boost
                if following.get("content_type") == content_type:
                    confidence = min(confidence + 0.05, 0.95)

                # Educational content boost if followed by educational content
                if following.get("is_educational", False):
                    educational_score += 0.2

            # Update segment with enhanced values
            enhanced_segment = segment.copy()
            enhanced_segment["classification_confidence"] = confidence
            enhanced_segment["educational_score"] = educational_score
            enhanced_segment["is_educational"] = educational_score > 2.5  # Threshold for educational vs passing mention

            enhanced_segments.append(enhanced_segment)

        return enhanced_segments

    def _normalize_english_text(self, text: str) -> str:
        """
        Normalize English text.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
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

        # Simple grammar fixes
        text = re.sub(r'\s+,', ',', text)  # Remove space before comma
        text = re.sub(r'\s+\.', '.', text)  # Remove space before period

        return text.strip()

    def _normalize_russian_text(self, text: str) -> str:
        """
        Normalize Russian text.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
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

        # Simple grammar fixes
        text = re.sub(r'\s+,', ',', text)  # Remove space before comma
        text = re.sub(r'\s+\.', '.', text)  # Remove space before period

        # Common problematic phrases in Russian transcripts
        text = text.replace("то обсуждений давайте", "")
        text = text.replace("то состояние второго определённо такое", "")
        text = text.replace("вакуумное состояние оно", "вакуумное состояние")
        text = text.replace("эрмитово оператора", "эрмитов оператор")
        text = text.replace("любое собственное состояние оно", "собственное состояние")

        return text.strip()

# Helper functions for testing and standalone usage
def calculate_theory_practice_ratio(segments: List[Dict]) -> Dict[str, Any]:
    """
    Calculate theory/practice ratio from classified segments.

    Args:
        segments: Classified transcript segments

    Returns:
        Dictionary with theory/practice analysis
    """
    if not segments:
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "theoretical_segments": 0,
            "practical_segments": 0,
            "mixed_segments": 0,
            "theory_practice_ratio": 0.5
        }

    # Count segment types with confidence weighting
    theoretical_count = 0
    practical_count = 0
    mixed_count = 0

    # Track total confidence-weighted counts
    theoretical_weighted = 0
    practical_weighted = 0
    mixed_weighted = 0

    # Track time distribution
    total_duration = 0
    theoretical_duration = 0
    practical_duration = 0
    mixed_duration = 0

    for segment in segments:
        segment_type = segment.get("content_type", "mixed")
        confidence = segment.get("classification_confidence", 0.6)  # Default confidence if not present

        # Calculate segment duration
        start_time = segment.get("start_time", 0)
        end_time = segment.get("end_time", 0)
        duration = end_time - start_time
        total_duration += duration

        if segment_type == "theoretical":
            theoretical_count += 1
            theoretical_weighted += confidence
            theoretical_duration += duration
        elif segment_type == "practical":
            practical_count += 1
            practical_weighted += confidence
            practical_duration += duration
        else:  # mixed
            mixed_count += 1
            mixed_weighted += confidence
            mixed_duration += duration

    total_segments = theoretical_count + practical_count + mixed_count

    # Calculate theory/practice ratio with improved weighting
    if total_segments > 0:
        # Apply a weighted formula with confidence
        total_weighted = theoretical_weighted + practical_weighted + mixed_weighted

        if total_weighted > 0:
            # Apply confidence-weighted formula
            theory_weight = theoretical_weighted + (mixed_weighted * 0.5)
            theory_practice_ratio = theory_weight / total_weighted
        else:
            theory_practice_ratio = 0.5

        # Factor in duration-based ratio
        if total_duration > 0:
            duration_theory_ratio = (theoretical_duration + (mixed_duration * 0.5)) / total_duration

            # Final ratio is an average of count-based and duration-based ratios
            theory_practice_ratio = (theory_practice_ratio + duration_theory_ratio) / 2

    else:
        theory_practice_ratio = 0.5

    # Determine overall classification
    if theory_practice_ratio > 0.7:
        classification = "theoretical"
        confidence = 0.8 if theory_practice_ratio > 0.85 else 0.7
    elif theory_practice_ratio < 0.3:
        classification = "practical"
        confidence = 0.8 if theory_practice_ratio < 0.15 else 0.7
    else:
        classification = "mixed"
        closeness_to_half = 1.0 - abs(theory_practice_ratio - 0.5) * 2
        confidence = 0.6 + (closeness_to_half * 0.3)

    return {
        "classification": classification,
        "confidence": confidence,
        "theoretical_segments": theoretical_count,
        "practical_segments": practical_count,
        "mixed_segments": mixed_count,
        "theory_practice_ratio": theory_practice_ratio,
        "duration_analysis": {
            "total_duration": total_duration,
            "theoretical_duration": theoretical_duration,
            "practical_duration": practical_duration,
            "mixed_duration": mixed_duration
        }
    }

def test_transcript_processor(raw_segments: List[Dict], video_metadata: Dict) -> Dict[str, Any]:
    """
    Test the transcript processor with a sample transcript.

    Args:
        raw_segments: List of raw transcript segments
        video_metadata: Video metadata dictionary

    Returns:
        Processing results
    """
    processor = TranscriptProcessor()
    start_time = time.time()

    # Process the transcript
    result = processor.process_transcript(raw_segments, video_metadata)

    # Calculate timing metrics
    processing_time = time.time() - start_time

    # Add test metrics
    result["test_metrics"] = {
        "processing_time_seconds": processing_time,
        "segments_processed": len(raw_segments),
        "output_segments": len(result["segments"])
    }

    # Calculate theory/practice ratio for the processed segments
    theory_practice_results = calculate_theory_practice_ratio(result["segments"])
    result["theory_practice_results"] = theory_practice_results

    return result
