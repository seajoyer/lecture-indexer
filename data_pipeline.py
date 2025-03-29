"""
Enhanced data pipeline for the Lecture Video Content Indexer.
Coordinates the end-to-end process of video extraction, transcript processing,
domain classification, and theory-practice analysis with improved concept extraction.
"""

import os
import logging
import uuid
import re
import nltk
from typing import Dict, List, Set, Any, Optional, Tuple, Counter as CounterType
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import string

# Make sure NLTK resources are available
required_resources = ['punkt', 'stopwords']
for resource in required_resources:
    try:
        nltk.data.find(f"{'corpora' if resource != 'punkt' else 'tokenizers'}/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder

# Configure logging
logger = logging.getLogger(__name__)

# Import project modules
from youtube_extractor import YouTubeExtractor
from transcript_processor import TranscriptProcessor
from performance_utils import time_function, Timer
from cache_manager import cache_get, cache_set
try:
    from concept_signature_generator import enhance_data_pipeline
except ImportError:
    # Handle import error gracefully
    logger.warning("ConceptSignatureGenerator not available - concept signatures will not be used")
    enhance_data_pipeline = lambda x: x  # No-op function


class DataPipeline:
    """
    Coordinates the end-to-end process of video data acquisition and analysis.
    Enhanced with improved concept extraction and domain analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get("output_dir", "data/processed")

        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self._init_components()

        # Initialize NLP resources
        self._init_nlp_resources()

        enhance_data_pipeline(self)

        logger.info("DataPipeline initialized with enhanced concept extraction")

    def _init_components(self):
        """Initialize pipeline components."""
        # Get YouTube API key from config
        youtube_api_key = self.config.get("youtube_api_key")
        if not youtube_api_key:
            logger.warning("No YouTube API key provided, using test mode")
            youtube_api_key = "test_api_key"

        # Initialize components
        self.youtube_extractor = YouTubeExtractor(youtube_api_key)
        self.transcript_processor = TranscriptProcessor()

        logger.info("Pipeline components initialized")

    def _init_nlp_resources(self):
        """Initialize NLP resources for concept extraction."""
        # Initialize stopwords for different languages with comprehensive lists
        self.stopwords = {
            'en': set(stopwords.words('english')),
            'ru': set()
        }

        # Try to load Russian stopwords if available
        try:
            self.stopwords['ru'] = set(stopwords.words('russian'))
        except:
            logger.warning("Russian stopwords not available, using empty set")

        # Add common filler words not in NLTK's stopwords
        additional_stopwords_en = {
            # Common English filler words and discourse markers
            "uh", "um", "like", "so", "well", "actually", "basically",
            "literally", "sort", "kind", "really", "very", "quite",
            "okay", "ok", "yeah", "yes", "no", "right", "let", "just",
            "gonna", "going", "let's", "now", "here", "there", "this",
            "that", "these", "those", "will", "shall", "should", "would",
            "could", "can", "may", "might", "must",

            # Common pronouns and determiners
            "i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "whose",

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

            # Conjunctions and other function words
            "and", "but", "or", "nor", "yet", "so", "because", "if", "unless",
            "while", "where", "when", "how", "why", "whether", "though",
            "although", "since"
        }
        self.stopwords['en'].update(additional_stopwords_en)

        # Add Russian filler words
        additional_stopwords_ru = {
            # Common Russian filler words and discourse markers
            "это", "вот", "так", "как", "ну", "да", "нет", "просто",
            "значит", "сейчас", "здесь", "тут", "уже", "если", "все", "всё",
            "хорошо", "там", "кстати", "давайте", "итак", "будет", "ещё", "еще",
            "нас", "меня", "можно", "они", "только", "для",

            # Common Russian pronouns and determiners
            "я", "мне", "меня", "мой", "моя", "моё", "мои", "мною",
            "ты", "тебя", "тебе", "твой", "твоя", "твоё", "твои", "тобой",
            "он", "его", "ему", "им", "него", "нему", "ним",
            "она", "её", "ей", "ею", "неё", "ней",
            "оно", "нас", "нам", "нами", "них", "ими",
            "вы", "вас", "вам", "вами",
            "они", "их", "им", "ими",
            "кто", "что", "какой", "какая", "какое", "какие", "чей", "который",

            # Common Russian verbs
            "быть", "есть", "буду", "будешь", "будет", "будем", "будете", "будут",
            "был", "была", "было", "были",
            "иметь", "имею", "имеешь", "имеет", "имеем", "имеете", "имеют",
            "делать", "делаю", "делаешь", "делает", "делаем", "делаете", "делают",
            "идти", "иду", "идёшь", "идёт", "идём", "идёте", "идут",
            "сказать", "скажу", "скажешь", "скажет", "скажем", "скажете", "скажут",

            # Common Russian adverbs and prepositions
            "в", "на", "с", "к", "у", "от", "из", "по", "за", "о", "об", "без", "до",
            "над", "под", "при", "через", "между", "около", "перед", "после",
            "сейчас", "потом", "всегда", "никогда", "иногда", "обычно", "вверх", "вниз",

            # Problem words from the example output
            "поэтому", "равно", "нужно", "получается", "означает", "должна", "вами",
            "можем", "какой-то", "что-то", "стоит", "хочу", "буду", "видим",
            "понятно", "сделать", "например", "должны", "какие-то", "сюда",
            "плюс", "минус", "будем", "результат", "такое"
        }
        self.stopwords['ru'].update(additional_stopwords_ru)

        # Domain-specific keywords that are important for each domain
        # These should NOT be filtered out as stopwords
        self.domain_keywords = {
            "mathematics": {
                "en": {"function", "variable", "equation", "theorem", "proof",
                      "integral", "derivative", "limit", "series", "vector",
                      "matrix", "algebra", "geometry", "calculus", "topology",
                      "probability", "statistics", "set", "group", "field",
                      "differential", "discrete", "continuous", "infinite", "finite"},
                "ru": {"функция", "переменная", "уравнение", "теорема", "доказательство",
                      "интеграл", "производная", "предел", "ряд", "вектор",
                      "матрица", "алгебра", "геометрия", "анализ", "топология",
                      "вероятность", "статистика", "множество", "группа", "поле",
                      "дифференциал", "дискретный", "непрерывный", "бесконечный", "конечный"}
            },
            "programming": {
                "en": {"algorithm", "function", "class", "object", "method",
                      "variable", "array", "list", "loop", "recursion",
                      "data", "structure", "stack", "queue", "tree", "graph",
                      "hash", "sort", "search", "complexity", "database",
                      "interface", "inheritance", "polymorphism", "encapsulation"},
                "ru": {"алгоритм", "функция", "класс", "объект", "метод",
                      "переменная", "массив", "список", "цикл", "рекурсия",
                      "данные", "структура", "стек", "очередь", "дерево", "граф",
                      "хэш", "сортировка", "поиск", "сложность", "база данных",
                      "интерфейс", "наследование", "полиморфизм", "инкапсуляция"}
            },
            "physics": {
                "en": {"force", "energy", "momentum", "mass", "velocity", "acceleration",
                      "gravity", "electromagnetism", "quantum", "relativity", "particle",
                      "wave", "field", "potential", "nuclear", "atomic", "thermodynamics",
                      "fluid", "mechanics", "dynamics", "kinematics", "statics"},
                "ru": {"сила", "энергия", "импульс", "масса", "скорость", "ускорение",
                      "гравитация", "электромагнетизм", "квантовый", "относительность", "частица",
                      "волна", "поле", "потенциал", "ядерный", "атомный", "термодинамика",
                      "жидкость", "механика", "динамика", "кинематика", "статика",
                      "оператор", "гамильтониан", "состояние", "собственное", "коммутатор",
                      "представление", "базис", "шредингер", "симметрия"}
            }
        }

        # Domain-specific n-gram patterns for concept extraction
        self.domain_concept_patterns = {
            "mathematics": {
                "en": [
                    # Patterns for mathematical concepts
                    r'\b(?:the|a) (\w+) (theorem|lemma|property|identity|formula|equation|inequality|principle)\b',
                    r'\b(differential|partial|ordinary) (equation)\b',
                    r'\b(linear|quadratic|polynomial|exponential|logarithmic|trigonometric) (function|equation|identity)\b',
                    r'\b(convergent|divergent|infinite|finite) (series|sequence)\b',
                    r'\b(vector|matrix|tensor) (space|field|algebra|calculus)\b',
                    r'\b(probability|statistical) (distribution|model|test|analysis)\b'
                ],
                "ru": [
                    # Russian patterns
                    r'\b(теорема|лемма|свойство|формула|уравнение|неравенство|принцип) (\w+)\b',
                    r'\b(дифференциальное|частное|обыкновенное) (уравнение)\b',
                    r'\b(линейная|квадратичная|полиномиальная|экспоненциальная|логарифмическая|тригонометрическая) (функция|уравнение)\b',
                    r'\b(сходящийся|расходящийся|бесконечный|конечный) (ряд|последовательность)\b',
                    r'\b(векторное|матричное|тензорное) (пространство|поле|алгебра)\b'
                ]
            },
            "programming": {
                "en": [
                    # Patterns for programming concepts
                    r'\b(data|abstract) (structure|type)\b',
                    r'\b(sorting|search|graph|tree) (algorithm)\b',
                    r'\b(time|space) (complexity)\b',
                    r'\b(object[\-\s]oriented|functional|procedural|declarative) (programming|approach|paradigm)\b',
                    r'\b(design|architectural) (pattern)\b',
                    r'\b(binary|linear|hash) (search|table)\b',
                    r'\b(linked|array|circular) (list)\b'
                ],
                "ru": [
                    # Russian patterns
                    r'\b(структура|тип) (данных)\b',
                    r'\b(алгоритм) (сортировки|поиска|обхода)\b',
                    r'\b(временная|пространственная) (сложность)\b',
                    r'\b(объектно[\-\s]ориентированное|функциональное|процедурное) (программирование|подход|парадигма)\b',
                    r'\b(шаблон|паттерн) (проектирования)\b',
                    r'\b(бинарный|линейный|хеш) (поиск|таблица)\b',
                    r'\b(связный|массив|циклический) (список)\b'
                ]
            },
            "physics": {
                "en": [
                    # Patterns for physics concepts
                    r'\b(gravitational|electric|magnetic|electromagnetic) (field|force|potential)\b',
                    r'\b(kinetic|potential|mechanical|thermal|nuclear) (energy)\b',
                    r'\b(newton\'?s|coulomb\'?s|faraday\'?s|ohm\'?s|ampere\'?s|kepler\'?s) (law|principle)\b',
                    r'\b(special|general) (relativity)\b',
                    r'\b(quantum) (mechanics|field theory|chromodynamics|electrodynamics)\b',
                    r'\b(wave|particle) (function|duality|theory)\b',
                    r'\b(string|m|supersymmetric) (theory)\b'
                ],
                "ru": [
                    # Russian patterns
                    r'\b(гравитационное|электрическое|магнитное|электромагнитное) (поле|сила|потенциал)\b',
                    r'\b(кинетическая|потенциальная|механическая|тепловая|ядерная) (энергия)\b',
                    r'\b(закон|принцип) (ньютона|кулона|фарадея|ома|ампера|кеплера)\b',
                    r'\b(специальная|общая) (теория относительности)\b',
                    r'\b(квантовая) (механика|теория поля|хромодинамика|электродинамика)\b',
                    r'\b(волновая|корпускулярная) (функция|дуализм|теория)\b',
                    r'\b(оператор) (гамильтониана|шредингера|рождения|уничтожения|импульса|энергии)\b',
                    r'\b(собственное) (значение|состояние|функция)\b',
                    r'\b(координатное|импульсное) (представление)\b'
                ]
            }
        }

        # Compile the patterns for efficiency
        self.compiled_domain_patterns = {}
        for domain, lang_patterns in self.domain_concept_patterns.items():
            self.compiled_domain_patterns[domain] = {}
            for lang, patterns in lang_patterns.items():
                self.compiled_domain_patterns[domain][lang] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

        # Define patterns for definitional contexts
        self.definition_patterns = {
            'en': [
                r'(\w+[\s\w]*) is defined as ([\s\w]+)',
                r'(\w+[\s\w]*) refers to ([\s\w]+)',
                r'(\w+[\s\w]*) is a ([\s\w]+)',
                r'(\w+[\s\w]*) is an ([\s\w]+)',
                r'(\w+[\s\w]*) means ([\s\w]+)',
                r'the concept of (\w+[\s\w]*)',
                r'the principle of (\w+[\s\w]*)',
                r'the theory of (\w+[\s\w]*)',
                r'(\w+[\s\w]*) is characterized by ([\s\w]+)',
                r'(\w+[\s\w]*) is called ([\s\w]+)'
            ],
            'ru': [
                r'(\w+[\s\w]*) определяется как ([\s\w]+)',
                r'(\w+[\s\w]*) означает ([\s\w]+)',
                r'(\w+[\s\w]*) это ([\s\w]+)',
                r'(\w+[\s\w]*) является ([\s\w]+)',
                r'понятие (\w+[\s\w]*)',
                r'концепция (\w+[\s\w]*)',
                r'принцип (\w+[\s\w]*)',
                r'теория (\w+[\s\w]*)',
                r'(\w+[\s\w]*) характеризуется ([\s\w]+)',
                r'(\w+[\s\w]*) называется ([\s\w]+)'
            ]
        }

        # Compile definition patterns
        self.compiled_definition_patterns = {
            lang: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for lang, patterns in self.definition_patterns.items()
        }

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

        # Compile the regex patterns
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

        logger.info("NLP resources initialized for concept extraction")

    @time_function(10000)  # Log warning if takes more than 10 seconds
    def process_video(self, video_url: str, language_preference: List[str] = ['en', 'ru']) -> Dict[str, Any]:
        """
        Process a YouTube video through the entire pipeline.

        Args:
            video_url: YouTube video URL
            language_preference: List of language codes in order of preference

        Returns:
            Dictionary with processing results
        """
        # Create a timer for overall process
        timer = Timer("process_video").start()

        # Generate a unique job ID based on timestamp and UUID
        job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Starting video processing job {job_id} for URL: {video_url}")

        try:
            # Step 1: Validate URL and extract video ID
            valid, video_id = self.youtube_extractor.validate_video_url(video_url)
            if not valid or not video_id:
                error_msg = f"Invalid YouTube URL: {video_url}"
                logger.error(error_msg)
                return {
                    "job_id": job_id,
                    "status": "error",
                    "error": error_msg,
                    "video_url": video_url
                }

            logger.info(f"Validated YouTube URL, video ID: {video_id}")

            # Check cache for previously processed result
            cache_key = f"processed_video_{video_id}"
            cached_result = cache_get("video", cache_key)
            if cached_result:
                logger.info(f"Using cached processing result for video {video_id}")
                return cached_result

            # Step 2: Extract video metadata
            metadata = self.youtube_extractor.extract_video_metadata(video_id)
            logger.info(f"Extracted metadata for video: {video_id}")

            # Step 3: Extract transcript
            raw_transcript = self.youtube_extractor.extract_transcript(video_id, language_preference)
            logger.info(f"Extracted transcript with {len(raw_transcript)} segments")

            # Step 4: Process transcript
            processed_transcript = self.transcript_processor.process_transcript(raw_transcript, metadata)
            logger.info(f"Processed transcript with {len(processed_transcript['segments'])} segments")

            # Step 5: Calculate theory/practice ratio
            theory_practice_results = self._calculate_theory_practice_ratio(processed_transcript['segments'])
            logger.info(f"Calculated theory/practice ratio: {theory_practice_results['theory_practice_ratio']:.2f}")

            # Step 6: Extract key concepts
            domain_features = self._extract_domain_features(processed_transcript, metadata["domain"])
            logger.info(f"Extracted {len(domain_features['key_concepts'])} key concepts")

            # Prepare result
            processing_time = timer.stop() / 1000  # Convert from ms to seconds

            result = {
                "job_id": job_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "video_id": video_id,
                "video_url": video_url,
                "metadata": metadata,
                "transcript": processed_transcript,
                "domain_features": domain_features,
                "theory_practice_results": theory_practice_results,
                "processing_time": processing_time
            }

            # Cache the result
            cache_set("video", cache_key, result)

            # Save result to file (for backward compatibility)
            self._save_result(result)

            logger.info(f"Successfully processed video {video_id} in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")
            error_result = {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "video_url": video_url,
                "timestamp": datetime.now().isoformat()
            }

            # If we have a video_id, include it
            if 'video_id' in locals() and video_id:
                error_result["video_id"] = video_id

            # If we have metadata, include it
            if 'metadata' in locals() and metadata:
                error_result["metadata"] = metadata

            # Save error result
            self._save_result(error_result)

            return error_result

    def _calculate_theory_practice_ratio(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Calculate theory/practice ratio from segments with enhanced accuracy.

        Args:
            segments: Processed transcript segments

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

        # Determine overall classification with improved confidence calculation
        if theory_practice_ratio > 0.7:
            classification = "theoretical"
            # Higher confidence if there is a strong bias towards theoretical
            if theory_practice_ratio > 0.85:
                confidence = 0.9
            elif theoretical_count > practical_count * 2:
                confidence = 0.8
            else:
                confidence = 0.7
        elif theory_practice_ratio < 0.3:
            classification = "practical"
            # Higher confidence if there is a strong bias towards practical
            if theory_practice_ratio < 0.15:
                confidence = 0.9
            elif practical_count > theoretical_count * 2:
                confidence = 0.8
            else:
                confidence = 0.7
        else:
            classification = "mixed"
            # Higher confidence when theory/practice ratio is near 0.5
            closeness_to_half = 1.0 - abs(theory_practice_ratio - 0.5) * 2  # 1.0 at 0.5, 0.0 at 0.0/1.0
            confidence = 0.6 + (closeness_to_half * 0.3)  # Maps to 0.6-0.9 range

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

    def _extract_domain_features(self, processed_transcript: Dict, domain: str) -> Dict[str, Any]:
        """
        Extract domain-specific features from processed transcript with enhanced concept extraction.

        Args:
            processed_transcript: Processed transcript dictionary
            domain: Content domain

        Returns:
            Dictionary with domain-specific features
        """
        segments = processed_transcript.get("segments", [])
        language = processed_transcript.get("language", "en")

        # Extract combined text for analysis
        combined_text = " ".join([segment.get("text", "") for segment in segments])

        # Perform concept extraction with significantly improved algorithm
        key_concepts = self._extract_key_concepts_enhanced(combined_text, segments, domain, language)

        # Organize concepts by segment context types
        theoretical_concepts = []
        practical_concepts = []

        for concept in key_concepts:
            # Add to appropriate list based on concept_class
            if concept["concept_class"] == "theoretical":
                theoretical_concepts.append(concept)
            else:
                practical_concepts.append(concept)

        # Find relationships between concepts
        concept_relationships = self._find_concept_relationships(key_concepts, segments)

        return {
            "domain": domain,
            "key_concepts": key_concepts,
            "theoretical_concepts": theoretical_concepts,
            "practical_concepts": practical_concepts,
            "concept_relationships": concept_relationships
        }

    def _extract_key_concepts_enhanced(
        self,
        combined_text: str,
        segments: List[Dict],
        domain: str,
        language: str = "en"
    ) -> List[Dict]:
        """
        Enhanced concept extraction using advanced NLP techniques with multilingual support.
        Implements significant improvements for concept recognition.

        Args:
            combined_text: Combined text from all segments
            segments: List of transcript segments
            domain: Content domain
            language: Language code

        Returns:
            List of concept dictionaries
        """
        # Skip if text is empty
        if not combined_text.strip():
            return []

        # Get stopwords for the language
        lang_code = language if language in self.stopwords else 'en'
        stopwords_set = self.stopwords.get(lang_code, set())

        # Get domain keywords to exclude from stopwords
        domain_keywords = self._get_domain_keywords(domain, language)
        filtered_stopwords = stopwords_set - domain_keywords

        # Split text into sentences for better context analysis
        try:
            sentences = sent_tokenize(combined_text)
        except:
            # Fallback for non-English content
            sentences = re.split(r'(?<=[.!?])\s+', combined_text)

        # 1. Extract candidate concepts using multiple methods
        candidates = {}

        # 1.1 Extract n-grams with TF-IDF weighting
        ngram_concepts = self._extract_ngram_concepts(segments, filtered_stopwords, language)
        for term, score_data in ngram_concepts.items():
            candidates[term] = {
                "text": term,
                "frequency": score_data.get("frequency", 1),
                "ngram_type": score_data.get("type", "ngram"),
                "score": score_data.get("score", 0) * 1.2,  # Weight n-grams slightly higher
                "source": "ngram_extraction"
            }

        # 1.2 Extract domain-specific pattern matches
        pattern_matches = self._extract_domain_patterns(sentences, domain, language)
        for pattern, count in pattern_matches.items():
            if pattern in candidates:
                candidates[pattern]["score"] += count * 2.0  # Boost score for pattern matches
                candidates[pattern]["pattern_match"] = True
                candidates[pattern]["source"] = "domain_pattern"
            else:
                candidates[pattern] = {
                    "text": pattern,
                    "frequency": count,
                    "ngram_type": "pattern",
                    "pattern_match": True,
                    "score": count * 2.0,  # Higher weight for domain patterns
                    "source": "domain_pattern"
                }

        # 1.3 Extract concepts from definitional contexts
        definitional_concepts = self._extract_definitional_concepts(sentences, language)
        for concept, info in definitional_concepts.items():
            if concept in candidates:
                candidates[concept]["score"] += info["count"] * 2.5  # Highest weight for definitional contexts
                candidates[concept]["definitional"] = True
                candidates[concept]["source"] = "definitional_context"
            else:
                candidates[concept] = {
                    "text": concept,
                    "frequency": info["count"],
                    "ngram_type": "definitional",
                    "definitional": True,
                    "score": info["count"] * 2.5,  # Highest weight for definitional concepts
                    "definition": info.get("definition", ""),
                    "source": "definitional_context"
                }

        # 1.4 Collocations extraction (words that frequently appear together)
        collocations = self._extract_collocations(combined_text, language)
        for collocation, score in collocations.items():
            if collocation in candidates:
                candidates[collocation]["score"] += score
                candidates[collocation]["collocation"] = True
            else:
                candidates[collocation] = {
                    "text": collocation,
                    "frequency": 1,  # We don't have exact frequency here
                    "ngram_type": "collocation",
                    "collocation": True,
                    "score": score,
                    "source": "collocation"
                }

        # 2. Apply filters to remove unlikely concepts
        filtered_candidates = {}

        # Get language-specific minimum score threshold and word length
        min_score_threshold = 2.0 if language == 'ru' else 1.0
        min_word_length = 4 if language == 'ru' else 3

        for concept_text, concept_data in candidates.items():
            # Skip concepts with very low scores
            if concept_data.get("score", 0) < min_score_threshold:
                continue

            # Skip very short concepts (likely not meaningful)
            words = concept_text.split()
            if len(words) == 1 and len(concept_text) < min_word_length:
                continue

            # Skip concepts that are just numbers or mathematical symbols
            if re.match(r'^[\d\s\+\-\*\/\=]+$', concept_text):
                continue

            # Skip concepts with too many stopwords
            if len(words) > 1:
                stopword_count = sum(1 for word in words if word.lower() in filtered_stopwords)
                if stopword_count / len(words) > 0.5:  # More than half are stopwords
                    continue

            # Skip concepts that appear too frequently (likely common words)
            if concept_data.get("frequency", 0) / len(segments) > 0.7:
                continue

            # Keep the concept
            filtered_candidates[concept_text] = concept_data

        # 3. Determine if each concept is theoretical or practical based on context
        for concept_text, concept_data in filtered_candidates.items():
            # Determine if concept is theoretical or practical based on context
            is_theoretical = self._is_theoretical_concept_enhanced(concept_text, segments, language, domain)

            # Update concept data
            concept_data["theoretical"] = is_theoretical
            concept_data["concept_class"] = "theoretical" if is_theoretical else "practical"

        # 4. Final ranking and selection
        # Convert to list and sort by score
        ranked_concepts = []
        for concept_text, concept_data in filtered_candidates.items():
            ranked_concepts.append({
                "text": concept_text,
                "frequency": concept_data.get("frequency", 0),
                "domain": domain,
                "theoretical": concept_data.get("theoretical", True),
                "concept_class": concept_data.get("concept_class", "theoretical"),
                "ngram_type": concept_data.get("ngram_type", ""),
                "pattern_match": concept_data.get("pattern_match", False),
                "definitional": concept_data.get("definitional", False),
                "collocation": concept_data.get("collocation", False),
                "score": concept_data.get("score", 0),
                "source": concept_data.get("source", ""),
                "language": language
            })

        # Sort by score
        ranked_concepts.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Take top concepts with a adaptive limit based on content length
        max_concepts = min(50, max(10, len(sentences) // 10))  # Scale with content length
        return ranked_concepts[:max_concepts]

    def _extract_ngram_concepts(
        self,
        segments: List[Dict],
        stopwords: set,
        language: str
    ) -> Dict[str, Dict]:
        """
        Extract concepts using n-gram analysis with TF-IDF weighting.
        Significantly improved to handle multi-word concepts and different languages.

        Args:
            segments: List of transcript segments
            stopwords: Set of stopwords to filter
            language: Language code

        Returns:
            Dictionary of concepts with their scores and metadata
        """
        # Extract all text and create a corpus of segments
        segment_texts = [segment.get("text", "") for segment in segments]
        all_text = " ".join(segment_texts)

        # Initialize results
        results = {}

        # Process unigrams (single words)
        unigrams = self._extract_significant_unigrams(all_text, segment_texts, stopwords, language)
        for term, data in unigrams.items():
            results[term] = {
                "frequency": data["frequency"],
                "score": data["score"] * 0.8,  # Lower weight for single words
                "type": "unigram"
            }

        # Process bigrams (two-word phrases)
        bigrams = self._extract_significant_bigrams(all_text, segment_texts, stopwords, language)
        for term, data in bigrams.items():
            results[term] = {
                "frequency": data["frequency"],
                "score": data["score"] * 1.2,  # Higher weight for bigrams
                "type": "bigram"
            }

        # Process trigrams (three-word phrases)
        trigrams = self._extract_significant_trigrams(all_text, segment_texts, stopwords, language)
        for term, data in trigrams.items():
            results[term] = {
                "frequency": data["frequency"],
                "score": data["score"] * 1.5,  # Even higher weight for trigrams
                "type": "trigram"
            }

        # Apply domain-specific boosting
        results = self._boost_domain_terms(results, language)

        return results

    def _extract_significant_unigrams(
        self,
        all_text: str,
        segment_texts: List[str],
        stopwords: set,
        language: str
    ) -> Dict[str, Dict]:
        """
        Extract significant single words using TF-IDF like weighting.

        Args:
            all_text: Combined text
            segment_texts: List of segment texts
            stopwords: Stopwords to filter
            language: Language code

        Returns:
            Dictionary of unigrams with scores
        """
        # Tokenize combined text
        try:
            tokens = word_tokenize(all_text.lower())
        except:
            tokens = all_text.lower().split()

        # Filter tokens
        filtered_tokens = [token for token in tokens
                          if token not in stopwords
                          and token not in string.punctuation
                          and len(token) > 2
                          and not token.isdigit()]

        # Count frequencies
        token_counter = Counter(filtered_tokens)

        # Calculate document frequency (in how many segments a token appears)
        doc_freq = {}
        for token in set(filtered_tokens):
            doc_freq[token] = sum(1 for text in segment_texts if token in text.lower())

        # Calculate TF-IDF like score
        results = {}
        num_segments = len(segment_texts)

        for token, freq in token_counter.items():
            if freq < 2:  # Skip tokens that appear only once
                continue

            # Calculate TF-IDF
            tf = freq / len(filtered_tokens)
            idf = math.log(num_segments / (1 + doc_freq.get(token, 1)))
            score = tf * idf

            # Language-specific boosting
            if language == 'ru':
                # For Russian, boost longer words as they tend to be more significant
                score *= (1 + min(len(token) / 10, 0.5))

            results[token] = {
                "frequency": freq,
                "score": score
            }

        return results

    def _extract_significant_bigrams(
        self,
        all_text: str,
        segment_texts: List[str],
        stopwords: set,
        language: str
    ) -> Dict[str, Dict]:
        """
        Extract significant bigrams (two-word phrases).

        Args:
            all_text: Combined text
            segment_texts: List of segment texts
            stopwords: Stopwords to filter
            language: Language code

        Returns:
            Dictionary of bigrams with scores
        """
        # Use the TranscriptProcessor's get_bigrams method
        all_bigrams = self.transcript_processor.get_bigrams(all_text, language)
        bigram_counter = Counter(all_bigrams)

        # Calculate document frequency for bigrams
        doc_freq = {}
        for bigram in set(all_bigrams):
            doc_freq[bigram] = sum(1 for text in segment_texts if bigram in text.lower())

        # Calculate scores
        results = {}
        num_segments = len(segment_texts)

        for bigram, freq in bigram_counter.items():
            if freq < 2:  # Skip bigrams that appear only once
                continue

            # Skip if all words are stopwords
            words = bigram.split()
            if all(word in stopwords for word in words):
                continue

            # Calculate TF-IDF
            tf = freq / len(all_bigrams) if all_bigrams else 0
            idf = math.log(num_segments / (1 + doc_freq.get(bigram, 1)))
            score = tf * idf

            # Boost bigrams with domain-specific words
            words = bigram.split()
            if any(len(word) > 5 for word in words):  # Longer words tend to be more technical
                score *= 1.2

            results[bigram] = {
                "frequency": freq,
                "score": score
            }

        return results

    def _extract_significant_trigrams(
        self,
        all_text: str,
        segment_texts: List[str],
        stopwords: set,
        language: str
    ) -> Dict[str, Dict]:
        """
        Extract significant trigrams (three-word phrases).

        Args:
            all_text: Combined text
            segment_texts: List of segment texts
            stopwords: Stopwords to filter
            language: Language code

        Returns:
            Dictionary of trigrams with scores
        """
        # Use the TranscriptProcessor's get_trigrams method
        all_trigrams = self.transcript_processor.get_trigrams(all_text, language)
        trigram_counter = Counter(all_trigrams)

        # Calculate document frequency for trigrams
        doc_freq = {}
        for trigram in set(all_trigrams):
            doc_freq[trigram] = sum(1 for text in segment_texts if trigram in text.lower())

        # Calculate scores
        results = {}
        num_segments = len(segment_texts)

        for trigram, freq in trigram_counter.items():
            if freq < 2:  # Skip trigrams that appear only once
                continue

            # Skip if most words are stopwords
            words = trigram.split()
            stopword_count = sum(1 for word in words if word in stopwords)
            if stopword_count >= 2:  # Skip if 2 or more words are stopwords
                continue

            # Calculate TF-IDF
            tf = freq / len(all_trigrams) if all_trigrams else 0
            idf = math.log(num_segments / (1 + doc_freq.get(trigram, 1)))
            score = tf * idf

            # Boost trigrams that form noun phrases
            # This is a simplified approach without POS tagging
            score *= 1.5  # Generally boost trigrams as they're more specific

            results[trigram] = {
                "frequency": freq,
                "score": score
            }

        return results

    def _boost_domain_terms(self, results: Dict[str, Dict], language: str) -> Dict[str, Dict]:
        """
        Boost scores for domain-specific terminology.

        Args:
            results: Dictionary of terms with their data
            language: Language code

        Returns:
            Updated dictionary with boosted scores
        """
        # Check if language has relevant domain keywords
        if language not in ['en', 'ru']:
            return results

        # Boost scores for domain-specific terms
        for term, data in results.items():
            # Check if any word in the term is a domain keyword
            words = term.split()

            # Check across all domains for domain-specific terminology
            for domain, domain_keywords in self.domain_keywords.items():
                lang_keywords = domain_keywords.get(language, set())

                # Check if any word in the term is a domain keyword
                if any(word in lang_keywords for word in words):
                    data["score"] *= 1.5
                    data["domain_specific"] = True

                # Extra boost for multi-word domain terms
                if len(words) > 1:
                    domain_word_count = sum(1 for word in words if word in lang_keywords)
                    if domain_word_count >= 2:
                        data["score"] *= 1.3

        return results

    def _extract_domain_patterns(
        self,
        sentences: List[str],
        domain: str,
        language: str
    ) -> Dict[str, int]:
        """
        Extract domain-specific patterns from sentences.

        Args:
            sentences: List of sentences
            domain: Content domain
            language: Language code

        Returns:
            Dictionary mapping extracted patterns to their frequency
        """
        patterns = {}

        # Use appropriate language patterns, fall back to English if not available
        lang_key = language if language in ['en', 'ru'] else 'en'

        # Get domain-specific patterns
        if domain in self.compiled_domain_patterns:
            domain_patterns = self.compiled_domain_patterns[domain].get(lang_key,
                                                                      self.compiled_domain_patterns[domain].get('en', []))

            # Extract patterns from sentences
            for sentence in sentences:
                for pattern in domain_patterns:
                    matches = pattern.findall(sentence)
                    for match in matches:
                        if isinstance(match, tuple):
                            # Join multi-word matches
                            concept = ' '.join(match)
                        else:
                            concept = match

                        # Clean up concept
                        concept = concept.lower().strip()
                        if concept:
                            patterns[concept] = patterns.get(concept, 0) + 1

        return patterns

    def _extract_definitional_concepts(self, sentences: List[str], language: str) -> Dict[str, Dict]:
        """
        Extract concepts from definitional contexts in the text.

        Args:
            sentences: List of sentences
            language: Language code

        Returns:
            Dictionary mapping concepts to their definitional information
        """
        definitional_concepts = {}

        # Get patterns for the language
        lang_key = language if language in self.compiled_definition_patterns else 'en'
        patterns = self.compiled_definition_patterns.get(lang_key, [])

        # Extract concepts from definitional contexts
        for sentence in sentences:
            for pattern in patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 1:
                        # Extract concept from match
                        concept = match[0].lower().strip()

                        # Skip very short concepts
                        if len(concept) < 3:
                            continue

                        # Get definition if available (in position 1)
                        definition = match[1].lower().strip() if len(match) > 1 else ""

                        # Update concept information
                        if concept in definitional_concepts:
                            definitional_concepts[concept]["count"] += 1
                            if definition and not definitional_concepts[concept].get("definition"):
                                definitional_concepts[concept]["definition"] = definition
                        else:
                            definitional_concepts[concept] = {
                                "count": 1,
                                "definition": definition
                            }

        return definitional_concepts

    def _extract_collocations(self, text: str, language: str) -> Dict[str, float]:
        """
        Extract statistically significant word collocations from text.

        Args:
            text: Input text
            language: Language code

        Returns:
            Dictionary mapping collocations to their scores
        """
        # Get stopwords for the language
        lang_code = language if language in self.stopwords else 'en'
        stopwords_set = self.stopwords.get(lang_code, set())

        # Tokenize text
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()

        # Filter tokens
        filtered_tokens = [token for token in tokens
                         if token not in stopwords_set
                         and token not in string.punctuation
                         and len(token) > 2]

        # Skip if too few tokens
        if len(filtered_tokens) < 10:
            return {}

        # Extract bigram collocations
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(filtered_tokens)

        # Apply frequency filter
        finder.apply_freq_filter(2)

        # Find collocations using different measures
        try:
            # Get top collocations
            bigram_scores = {}

            # PMI (Pointwise Mutual Information)
            pmi_bigrams = finder.score_ngrams(bigram_measures.pmi)
            for bigram, score in pmi_bigrams:
                bigram_text = ' '.join(bigram)
                bigram_scores[bigram_text] = score

            # Likelihood ratio
            lr_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
            for bigram, score in lr_bigrams:
                bigram_text = ' '.join(bigram)
                if bigram_text in bigram_scores:
                    bigram_scores[bigram_text] += score
                else:
                    bigram_scores[bigram_text] = score

            # Normalize scores
            max_score = max(bigram_scores.values()) if bigram_scores else 1
            normalized_scores = {bigram: score/max_score for bigram, score in bigram_scores.items()}

            return normalized_scores

        except Exception as e:
            logger.warning(f"Error extracting collocations: {e}")
            return {}

    def _get_domain_keywords(self, domain: str, language: str) -> set:
        """
        Get domain-specific keywords that should not be filtered as stopwords.

        Args:
            domain: Content domain
            language: Language code

        Returns:
            Set of domain keywords
        """
        # Use appropriate language keywords, fall back to English if not available
        lang_key = language if language in ['en', 'ru'] else 'en'

        # Get domain-specific keywords
        if domain in self.domain_keywords:
            return self.domain_keywords[domain].get(lang_key, self.domain_keywords[domain].get('en', set()))

        return set()

    def _is_theoretical_concept_enhanced(
        self,
        term: str,
        segments: List[Dict],
        language: str,
        domain: str
    ) -> bool:
        """
        Determine if a concept is theoretical using a comprehensive approach.
        Enhanced with multiple detection methods and domain awareness.

        Args:
            term: Concept term
            segments: Processed transcript segments
            language: Language code
            domain: Content domain

        Returns:
            True if theoretical, False if practical
        """
        # Normalize term for matching
        term_lower = term.lower()
        term_parts = term_lower.split()

        # Count segment types where this term appears
        theoretical_count = 0
        practical_count = 0
        theoretical_confidence_sum = 0
        practical_confidence_sum = 0

        # Count segments containing this term by their classification
        segments_with_term = []
        for segment in segments:
            segment_text = segment.get("text", "").lower()
            context_type = segment.get("content_type", "mixed")
            confidence = segment.get("classification_confidence", 0.6)

            # Check if term appears in this segment
            term_in_segment = False

            if len(term_parts) == 1:
                # For single words, use word boundary matching
                if re.search(r'\b' + re.escape(term_lower) + r'\b', segment_text):
                    term_in_segment = True
            else:
                # For phrases, check if all parts appear in order
                if term_lower in segment_text:
                    term_in_segment = True

            if term_in_segment:
                segments_with_term.append(segment)
                if context_type == "theoretical":
                    theoretical_count += 1
                    theoretical_confidence_sum += confidence
                elif context_type == "practical":
                    practical_count += 1
                    practical_confidence_sum += confidence

        # Get surrounding context for the term
        term_contexts = self._get_term_contexts(term_lower, segments_with_term)

        # Apply heuristics based on context patterns
        context_theoretical_score = 0
        context_practical_score = 0

        # Get language-appropriate patterns
        lang = language if language in self.theoretical_regex else 'en'

        for context in term_contexts:
            # Check for theoretical patterns in context
            if self.theoretical_regex[lang].search(context):
                context_theoretical_score += 1

            # Check for practical patterns in context
            if self.practical_regex[lang].search(context):
                context_practical_score += 1

        # Apply domain-specific terminology heuristics
        if domain in self.domain_features:
            domain_features = self.domain_features[domain].get(language, {})
            if not domain_features:
                domain_features = self.domain_features[domain].get('en', {})

            # Check if any word in the term is a domain feature
            for word in term_parts:
                if word in domain_features:
                    feature_weight = domain_features[word]
                    if feature_weight >= 0.75:  # Threshold for theoretical
                        context_theoretical_score += 1
                    else:
                        context_practical_score += 1

        # Count definitional contexts (strong indicator of theoretical content)
        definitional_score = self._count_definitional_contexts(term_lower, term_contexts, language)
        context_theoretical_score += definitional_score * 2  # Double weight for definitional contexts

        # Consider term structure (multi-word terms more likely to be theoretical)
        if len(term_parts) >= 3:
            context_theoretical_score += 0.5

        # If no segments contain the term, use context scores only
        if theoretical_count == 0 and practical_count == 0:
            return context_theoretical_score >= context_practical_score

        # Weighted classification based on all factors
        theoretical_score = (
            theoretical_confidence_sum * 1.0 +  # Segment classifications
            context_theoretical_score * 1.5     # Context patterns
        )

        practical_score = (
            practical_confidence_sum * 1.0 +    # Segment classifications
            context_practical_score * 1.5       # Context patterns
        )

        # Determine final classification
        return theoretical_score >= practical_score

    def _get_term_contexts(self, term: str, segments: List[Dict]) -> List[str]:
        """
        Get surrounding contexts for a term from segments.

        Args:
            term: The term to find contexts for
            segments: List of segments containing the term

        Returns:
            List of context texts
        """
        contexts = []

        for segment in segments:
            segment_text = segment.get("text", "").lower()

            # Find all occurrences of the term
            term_positions = []
            start = 0
            while True:
                pos = segment_text.find(term, start)
                if pos == -1:
                    break
                term_positions.append(pos)
                start = pos + len(term)

            # Extract contexts around the term
            for pos in term_positions:
                # Get context before and after the term
                context_start = max(0, pos - 50)
                context_end = min(len(segment_text), pos + len(term) + 50)
                context = segment_text[context_start:context_end]
                contexts.append(context)

        return contexts

    def _count_definitional_contexts(self, term: str, contexts: List[str], language: str) -> int:
        """
        Count how many contexts are definitional for the term.

        Args:
            term: The term to check
            contexts: List of context texts
            language: Language code

        Returns:
            Count of definitional contexts
        """
        definitional_count = 0

        # Get patterns for the language
        lang_key = language if language in self.compiled_definition_patterns else 'en'
        patterns = self.compiled_definition_patterns.get(lang_key, [])

        for context in contexts:
            # Check each pattern
            for pattern in patterns:
                matches = pattern.findall(context)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 1:
                        match_term = match[0].lower().strip()
                        # Check if the extracted term matches our term
                        if match_term == term or term in match_term or match_term in term:
                            definitional_count += 1
                            break

        return definitional_count

    def _find_concept_relationships(self, concepts: List[Dict], segments: List[Dict]) -> List[Dict]:
        """
        Find relationships between concepts based on co-occurrence and context.

        Args:
            concepts: List of extracted concepts
            segments: List of transcript segments

        Returns:
            List of concept relationship dictionaries
        """
        # Skip if not enough concepts
        if len(concepts) < 2:
            return []

        # Create a map of concepts to their texts for easier lookup
        concept_texts = {concept["text"].lower(): concept for concept in concepts}

        # Track co-occurrences with context types
        co_occurrences = {}

        # Analyze each segment for co-occurring concepts
        for segment in segments:
            segment_text = segment.get("text", "").lower()
            context_type = segment.get("content_type", "mixed")

            # Find all concepts in this segment
            concepts_in_segment = []
            for concept_text in concept_texts:
                if concept_text in segment_text:
                    concepts_in_segment.append(concept_text)

            # Record co-occurrences for each pair with context type
            for i, concept1 in enumerate(concepts_in_segment):
                for concept2 in concepts_in_segment[i+1:]:
                    pair = tuple(sorted([concept1, concept2]))

                    if pair not in co_occurrences:
                        co_occurrences[pair] = {
                            "count": 0,
                            "theoretical": 0,
                            "practical": 0,
                            "mixed": 0
                        }

                    co_occurrences[pair]["count"] += 1
                    co_occurrences[pair][context_type] += 1

        # Create relationship records
        relationships = []
        for (concept1, concept2), data in co_occurrences.items():
            count = data["count"]

            # Only include significant co-occurrences
            if count >= 2:
                c1 = concept_texts[concept1]
                c2 = concept_texts[concept2]

                # Determine relationship type based on context types
                rel_type = "related"

                if data["theoretical"] > data["practical"]:
                    rel_type = "related_theoretical"
                elif data["practical"] > data["theoretical"]:
                    rel_type = "related_practical"
                elif c1["concept_class"] == c2["concept_class"]:
                    rel_type = "related_" + c1["concept_class"]
                else:
                    rel_type = "theory_practice_pair"

                relationship = {
                    "source_concept": concept1,
                    "target_concept": concept2,
                    "co_occurrence_count": count,
                    "theoretical_contexts": data["theoretical"],
                    "practical_contexts": data["practical"],
                    "relationship_type": rel_type,
                    "source_class": c1["concept_class"],
                    "target_class": c2["concept_class"]
                }
                relationships.append(relationship)

        # Sort by co-occurrence count
        relationships.sort(key=lambda x: x["co_occurrence_count"], reverse=True)

        return relationships

    def _save_result(self, result: Dict[str, Any]):
        """
        Save processing result to file.

        Args:
            result: Processing result dictionary
        """
        # Create filename using video_id or job_id
        video_id = result.get("video_id", "unknown")
        job_id = result.get("job_id", "unknown")
        filename = f"{video_id}_{job_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved processing result to {filepath}")
        except Exception as e:
            logger.error(f"Error saving result to file: {e}")

    def batch_process_videos(self, video_urls: List[str], language_preference: List[str] = ['en', 'ru']) -> List[Dict[str, Any]]:
        """
        Process multiple YouTube videos.

        Args:
            video_urls: List of YouTube video URLs
            language_preference: List of language codes in order of preference

        Returns:
            List of processing result dictionaries
        """
        logger.info(f"Starting batch processing for {len(video_urls)} videos")

        results = []
        for i, url in enumerate(video_urls):
            try:
                # Process video
                logger.info(f"Processing video {i+1}/{len(video_urls)}: {url}")
                result = self.process_video(url, language_preference)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing video {url}: {e}")
                results.append({
                    "video_url": url,
                    "status": "error",
                    "error": str(e)
                })

        logger.info(f"Batch processing completed for {len(video_urls)} videos")
        return results
