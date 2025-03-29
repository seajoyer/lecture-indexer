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
from typing import List, Dict, Tuple, Counter as CounterType, Set, Optional
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

        # Load stopwords for multiple languages with enhanced lists
        self.stopwords = {}
        self._load_enhanced_stopwords()

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

    def _load_enhanced_stopwords(self):
        """Load comprehensive stopwords for multiple languages with extended sets."""
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
                        # Common English filler words and discourse markers
                        "uh", "um", "like", "so", "well", "actually", "basically",
                        "literally", "sort", "kind", "really", "very", "quite",
                        "okay", "ok", "yeah", "yes", "no", "right", "let", "just",
                        "gonna", "going", "let's", "now", "here", "there", "this",
                        "that", "these", "those", "will", "shall", "should", "would",
                        "could", "can", "may", "might", "must", "although", "however",

                        # Common pronouns and determiners
                        "i", "me", "my", "mine", "myself",
                        "you", "your", "yours", "yourself",
                        "he", "him", "his", "himself",
                        "she", "her", "hers", "herself",
                        "it", "its", "itself",
                        "we", "us", "our", "ours", "ourselves",
                        "they", "them", "their", "theirs", "themselves",
                        "what", "which", "who", "whom", "whose",
                        "this", "that", "these", "those", "such",
                        "the", "a", "an", "some", "any", "all", "most", "every", "many", "much",

                        # Common verbs
                        "am", "is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "having", "do", "does", "did", "doing",
                        "get", "gets", "got", "getting", "go", "goes", "went", "gone", "going",
                        "make", "makes", "made", "making", "take", "takes", "took", "taken", "taking",
                        "come", "comes", "came", "coming", "see", "sees", "saw", "seen", "seeing",
                        "use", "uses", "used", "using",

                        # Common adverbs and prepositions
                        "up", "down", "in", "out", "on", "off", "over", "under", "at", "by",
                        "for", "from", "to", "with", "about", "against", "between", "into",
                        "through", "during", "before", "after", "as", "since", "until",
                        "above", "below", "near", "far", "then", "also", "even", "only",
                        "already", "still", "always", "never", "sometimes", "usually",

                        # Conjunctions and other function words
                        "and", "but", "or", "nor", "yet", "so", "because", "if", "unless",
                        "while", "where", "when", "how", "why", "whether", "though",
                        "although", "since", "then", "than", "etc", "ie", "eg"
                    }
                    self.stopwords[code].update(english_extras)

                elif code == 'ru':
                    russian_extras = {
                        # Common Russian filler words and discourse markers
                        "это", "что", "как", "так", "вот", "просто", "если",
                        "там", "здесь", "сейчас", "тут", "ну", "да", "нет", "уже",
                        "значит", "такой", "такая", "такое", "давайте", "есть", "был",
                        "была", "были", "будет", "будут", "потому", "ещё", "еще",
                        "нас", "меня", "можно", "всё", "все", "они", "только", "для",

                        # Common Russian pronouns and determiners
                        "я", "мне", "меня", "мой", "моя", "моё", "мои", "мною",
                        "ты", "тебя", "тебе", "твой", "твоя", "твоё", "твои", "тобой",
                        "он", "его", "ему", "им", "него", "нему", "ним",
                        "она", "её", "ей", "ею", "неё", "ней",
                        "оно", "нас", "нам", "нами", "них", "ими",
                        "вы", "вас", "вам", "вами",
                        "они", "их", "им", "ими",
                        "кто", "что", "какой", "какая", "какое", "какие", "чей", "который",
                        "тот", "та", "то", "те", "этот", "эта", "это", "эти",
                        "весь", "вся", "всё", "все", "каждый", "любой", "самый", "другой",

                        # Common Russian verbs
                        "быть", "есть", "буду", "будешь", "будет", "будем", "будете", "будут",
                        "был", "была", "было", "были",
                        "иметь", "имею", "имеешь", "имеет", "имеем", "имеете", "имеют",
                        "делать", "делаю", "делаешь", "делает", "делаем", "делаете", "делают",
                        "идти", "иду", "идёшь", "идёт", "идём", "идёте", "идут",
                        "сказать", "скажу", "скажешь", "скажет", "скажем", "скажете", "скажут",
                        "видеть", "вижу", "видишь", "видит", "видим", "видите", "видят",
                        "знать", "знаю", "знаешь", "знает", "знаем", "знаете", "знают",
                        "мочь", "могу", "можешь", "может", "можем", "можете", "могут",
                        "хотеть", "хочу", "хочешь", "хочет", "хотим", "хотите", "хотят",

                        # Common Russian adverbs and prepositions
                        "в", "на", "с", "к", "у", "от", "из", "по", "за", "о", "об", "без", "до",
                        "над", "под", "при", "через", "между", "около", "перед", "после",
                        "сейчас", "потом", "всегда", "никогда", "иногда", "обычно", "вверх", "вниз",
                        "внутри", "снаружи", "здесь", "там", "далеко", "близко", "очень", "слишком",
                        "более", "менее", "почти", "совсем", "где", "когда", "куда", "откуда", "зачем",

                        # Additional common words that appear in the output
                        "поэтому", "равно", "нужно", "получается", "означает", "должна", "вами",
                        "можем", "какой-то", "что-то", "стоит", "хочу", "буду", "видим",
                        "понятно", "сделать", "например", "должны", "какие-то", "сюда",
                        "плюс", "минус", "будем", "результат", "такое"
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
                r'is formulated as',
                r'is represented by',
                r'is expressed as',
                r'is given by',
                r'is derived from',
                r'is related to',
                r'the definition of',
                r'the concept of',
                r'the theory of',
                r'the principle of',
                r'the law of',
                r'the equation for',
                r'according to the theory',
                r'in theoretical terms',
                r'from a theoretical perspective',
                r'a fundamental principle',
                r'the basis of'
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
                r'формулируется как',
                r'представлен как',
                r'выражается как',
                r'дается как',
                r'выводится из',
                r'связан с',
                r'определение',
                r'концепция',
                r'теория',
                r'принцип',
                r'закон',
                r'уравнение для',
                r'согласно теории',
                r'с теоретической точки зрения',
                r'фундаментальный принцип',
                r'основа',
                r'теоретически'
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
                r'in this example',
                r'to solve this',
                r'to implement this',
                r'to calculate',
                r'to compute',
                r'let me show you',
                r'I\'ll demonstrate',
                r'try to',
                r'let\'s try',
                r'in our case',
                r'the procedure is',
                r'application of',
                r'when working with',
                r'in real world',
                r'practically speaking',
                r'to illustrate',
                r'case study'
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
                r'в этом примере',
                r'чтобы решить',
                r'для реализации',
                r'для вычисления',
                r'позвольте показать',
                r'я продемонстрирую',
                r'попробуйте',
                r'давайте попробуем',
                r'в нашем случае',
                r'процедура',
                r'применение',
                r'при работе с',
                r'в реальном мире',
                r'практически говоря',
                r'для иллюстрации',
                r'пример из практики'
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
                    "axiom": 0.9, "postulate": 0.9, "corollary": 0.9, "proposition": 0.9,
                    "identity": 0.8, "inequality": 0.8, "relation": 0.7, "topology": 0.9,
                    "algebra": 0.8, "calculus": 0.8, "geometry": 0.8, "analysis": 0.8,
                    "theory": 0.9, "definition": 0.9,

                    # Practical indicators
                    "calculate": 0.8, "compute": 0.8, "solve": 0.8, "example": 0.7,
                    "problem": 0.7, "find": 0.7, "evaluate": 0.7, "simplify": 0.7,
                    "demonstrate": 0.7, "show": 0.6, "practice": 0.8, "exercise": 0.8,
                    "application": 0.7, "plug in": 0.8, "substitute": 0.7, "result": 0.6,
                    "numeric": 0.7, "estimate": 0.7, "approximate": 0.7, "implement": 0.8
                },
                "ru": {
                    # Theoretical indicators
                    "теорема": 0.9, "доказательство": 0.9, "лемма": 0.9, "определение": 0.8,
                    "уравнение": 0.8, "формула": 0.8, "функция": 0.7, "свойство": 0.7,
                    "аксиома": 0.9, "постулат": 0.9, "следствие": 0.9, "предложение": 0.9,
                    "тождество": 0.8, "неравенство": 0.8, "отношение": 0.7, "топология": 0.9,
                    "алгебра": 0.8, "анализ": 0.8, "геометрия": 0.8, "теория": 0.9,

                    # Practical indicators
                    "вычислить": 0.8, "рассчитать": 0.8, "решить": 0.8, "пример": 0.7,
                    "задача": 0.7, "найти": 0.7, "определить": 0.7, "упростить": 0.7,
                    "показать": 0.6, "демонстрировать": 0.7, "практика": 0.8, "упражнение": 0.8,
                    "применение": 0.7, "подставить": 0.7, "результат": 0.6,
                    "числовой": 0.7, "оценить": 0.7, "приблизить": 0.7, "реализовать": 0.8
                }
            },
            "programming": {
                "en": {
                    # Theoretical indicators
                    "algorithm": 0.8, "complexity": 0.85, "paradigm": 0.9,
                    "architecture": 0.8, "pattern": 0.7, "principle": 0.8,
                    "framework": 0.7, "abstraction": 0.9, "encapsulation": 0.9,
                    "inheritance": 0.8, "polymorphism": 0.9, "recursion": 0.8,
                    "structure": 0.7, "interface": 0.7, "protocol": 0.8,
                    "syntax": 0.8, "semantics": 0.9, "compiler": 0.7,

                    # Practical indicators
                    "code": 0.9, "implement": 0.85, "function": 0.7, "class": 0.7,
                    "debug": 0.9, "run": 0.8, "execute": 0.8, "compile": 0.7,
                    "install": 0.9, "library": 0.7, "framework": 0.7, "API": 0.8,
                    "build": 0.8, "deploy": 0.9, "test": 0.8, "version": 0.7,
                    "package": 0.7, "dependency": 0.7, "configuration": 0.7
                },
                "ru": {
                    # Theoretical indicators
                    "алгоритм": 0.8, "сложность": 0.85, "парадигма": 0.9,
                    "архитектура": 0.8, "шаблон": 0.7, "принцип": 0.8,
                    "фреймворк": 0.7, "абстракция": 0.9, "инкапсуляция": 0.9,
                    "наследование": 0.8, "полиморфизм": 0.9, "рекурсия": 0.8,
                    "структура": 0.7, "интерфейс": 0.7, "протокол": 0.8,
                    "синтаксис": 0.8, "семантика": 0.9, "компилятор": 0.7,

                    # Practical indicators
                    "код": 0.9, "реализовать": 0.85, "функция": 0.7, "класс": 0.7,
                    "отладка": 0.9, "запустить": 0.8, "выполнить": 0.8, "компилировать": 0.7,
                    "установить": 0.9, "библиотека": 0.7, "фреймворк": 0.7, "API": 0.8,
                    "сборка": 0.8, "развертывание": 0.9, "тест": 0.8, "версия": 0.7,
                    "пакет": 0.7, "зависимость": 0.7, "конфигурация": 0.7
                }
            },
            "physics": {
                "en": {
                    # Theoretical indicators
                    "theory": 0.9, "law": 0.9, "principle": 0.9, "constant": 0.8,
                    "equation": 0.8, "field": 0.7, "force": 0.7, "energy": 0.7,
                    "quantum": 0.9, "relativity": 0.9, "mechanics": 0.8, "dynamics": 0.8,
                    "thermodynamics": 0.9, "electromagnetism": 0.9, "oscillation": 0.8,
                    "particle": 0.8, "wave": 0.7, "momentum": 0.8, "conservation": 0.9,
                    "operator": 0.9, "eigenvalue": 0.9, "eigenstate": 0.9, "hamiltonian": 0.9,
                    "schrodinger": 0.9, "dirac": 0.9, "commutator": 0.9, "symmetry": 0.8,

                    # Practical indicators
                    "experiment": 0.9, "measure": 0.8, "observation": 0.8,
                    "calculate": 0.8, "predict": 0.7, "demonstrate": 0.8,
                    "laboratory": 0.9, "setup": 0.8, "device": 0.8, "apparatus": 0.9,
                    "probe": 0.8, "detector": 0.9, "sensor": 0.8, "signal": 0.7,
                    "data": 0.7, "instrument": 0.8, "calibration": 0.8, "procedure": 0.7
                },
                "ru": {
                    # Theoretical indicators
                    "теория": 0.9, "закон": 0.9, "принцип": 0.9, "константа": 0.8,
                    "уравнение": 0.8, "поле": 0.7, "сила": 0.7, "энергия": 0.7,
                    "квантовый": 0.9, "относительность": 0.9, "механика": 0.8, "динамика": 0.8,
                    "термодинамика": 0.9, "электромагнетизм": 0.9, "колебание": 0.8,
                    "частица": 0.8, "волна": 0.7, "импульс": 0.8, "сохранение": 0.9,
                    "оператор": 0.9, "собственное значение": 0.9, "собственное состояние": 0.9,
                    "гамильтониан": 0.9, "шредингер": 0.9, "дирак": 0.9, "коммутатор": 0.9,
                    "симметрия": 0.8, "состояние": 0.8, "представление": 0.8, "базис": 0.7,

                    # Practical indicators
                    "эксперимент": 0.9, "измерение": 0.8, "наблюдение": 0.8,
                    "рассчитать": 0.8, "предсказать": 0.7, "демонстрировать": 0.8,
                    "лаборатория": 0.9, "установка": 0.8, "устройство": 0.8, "аппарат": 0.9,
                    "зонд": 0.8, "детектор": 0.9, "датчик": 0.8, "сигнал": 0.7,
                    "данные": 0.7, "инструмент": 0.8, "калибровка": 0.8, "процедура": 0.7
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
                "en": ["math", "mathematics", "calculus", "algebra", "geometry", "theorem", "equation", "function",
                       "derivative", "integral", "linear", "quadratic", "polynomial", "vector", "matrix",
                       "differential", "series", "sequence", "limit", "continuous", "discrete"],
                "ru": ["математика", "алгебра", "геометрия", "теорема", "уравнение", "функция",
                       "производная", "интеграл", "линейный", "квадратичный", "полином", "вектор", "матрица",
                       "дифференциал", "ряд", "последовательность", "предел", "непрерывный", "дискретный"]
            },
            "programming": {
                "en": ["programming", "algorithm", "code", "software", "python", "java", "javascript",
                       "function", "class", "object", "method", "variable", "data structure", "loop",
                       "recursion", "compiler", "debugging", "framework", "library", "API"],
                "ru": ["программирование", "алгоритм", "код", "программа", "python", "java",
                       "функция", "класс", "объект", "метод", "переменная", "структура данных",
                       "цикл", "рекурсия", "компилятор", "отладка", "фреймворк", "библиотека", "API"]
            },
            "physics": {
                "en": ["physics", "mechanics", "dynamics", "quantum", "relativity", "force", "energy",
                       "momentum", "particle", "wave", "field", "nuclear", "atomic", "electromagnetic",
                       "thermodynamics", "velocity", "acceleration", "mass", "gravity", "charge"],
                "ru": ["физика", "механика", "динамика", "квантовая", "относительность", "сила", "энергия",
                       "импульс", "частица", "волна", "поле", "ядерный", "атомный", "электромагнитный",
                       "термодинамика", "скорость", "ускорение", "масса", "гравитация", "заряд",
                       "оператор", "состояние", "гамильтониан", "шредингер", "коммутатор", "представление"]
            }
        }

        # Get keywords for detected language, falling back to English if necessary
        lang_key = language if language in ["en", "ru"] else "en"

        # Count keyword occurrences with improved algorithm giving more weight to specialized terms
        domain_scores = {domain: 0 for domain in domain_keywords}
        lowered_text = full_text.lower()

        for domain, keywords_dict in domain_keywords.items():
            keywords = keywords_dict.get(lang_key, keywords_dict.get("en", []))

            for keyword in keywords:
                # Count occurrences
                count = lowered_text.count(keyword.lower())

                # Weigh multi-word terms higher (they're more specific)
                if " " in keyword:
                    count *= 2

                # Add to domain score
                domain_scores[domain] += count

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

    def normalize_word(self, word: str, language: str = 'en') -> str:
        """
        Normalize a word using appropriate stemmer for the language.

        Args:
            word: Word to normalize
            language: Language code

        Returns:
            Normalized word
        """
        # Skip very short words
        if len(word) <= 2:
            return word

        # Skip if word contains digits (likely a formula or equation)
        if any(char.isdigit() for char in word):
            return word

        # Get appropriate stemmer
        lang_code = language if language in self.stemmers else 'en'
        stemmer = self.stemmers.get(lang_code)

        if not stemmer:
            return word

        try:
            return stemmer.stem(word.lower())
        except:
            return word.lower()

    def get_bigrams(self, text: str, language: str = 'en') -> List[str]:
        """
        Extract meaningful bigrams from text.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of bigrams
        """
        # Get appropriate stopwords
        lang_code = language if language in self.stopwords else 'en'
        stop_words = self.stopwords.get(lang_code, set())

        # Tokenize text
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()

        # Filter out stopwords and punctuation
        filtered_tokens = [token for token in tokens if token not in stop_words
                          and token not in string.punctuation
                          and len(token) > 2]

        # Extract bigrams
        bigrams = []
        for i in range(len(filtered_tokens) - 1):
            # Don't include bigrams where both words are the same
            if filtered_tokens[i] != filtered_tokens[i+1]:
                bigrams.append(f"{filtered_tokens[i]} {filtered_tokens[i+1]}")

        return bigrams

    def get_trigrams(self, text: str, language: str = 'en') -> List[str]:
        """
        Extract meaningful trigrams from text.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of trigrams
        """
        # Get appropriate stopwords
        lang_code = language if language in self.stopwords else 'en'
        stop_words = self.stopwords.get(lang_code, set())

        # Tokenize text
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()

        # Filter out stopwords and punctuation
        filtered_tokens = [token for token in tokens if token not in stop_words
                          and token not in string.punctuation
                          and len(token) > 2]

        # Extract trigrams
        trigrams = []
        for i in range(len(filtered_tokens) - 2):
            # Don't include trigrams with repeated words
            if len(set([filtered_tokens[i], filtered_tokens[i+1], filtered_tokens[i+2]])) == 3:
                trigrams.append(f"{filtered_tokens[i]} {filtered_tokens[i+1]} {filtered_tokens[i+2]}")

        return trigrams

    def get_domain_specific_patterns(self, text: str, domain: str, language: str = 'en') -> List[str]:
        """
        Extract domain-specific patterns from text.

        Args:
            text: Input text
            domain: Domain (mathematics, programming, physics)
            language: Language code

        Returns:
            List of matched patterns
        """
        lang_key = language if language in ['en', 'ru'] else 'en'

        # Get appropriate patterns for domain and language
        if domain not in self.compiled_domain_patterns:
            return []

        domain_patterns = self.compiled_domain_patterns.get(domain, {}).get(lang_key, [])
        if not domain_patterns:
            domain_patterns = self.compiled_domain_patterns.get(domain, {}).get('en', [])

        # Extract matches
        matches = []
        for pattern in domain_patterns:
            for match in pattern.finditer(text):
                # Extract the matched phrase
                phrase = text[match.start():match.end()]
                if phrase and len(phrase) > 3:  # Ensure non-empty matches
                    matches.append(phrase.lower())

        return matches
