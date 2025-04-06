"""
Enhanced Unified Concept Extractor for the Lecture Video Content Indexer.
Provides robust concept extraction from video transcripts with optimized
language processing for both English and Russian content.
"""

import re
import uuid
import logging
from typing import Dict, List, Set, Any, Optional
from collections import Counter, defaultdict
import string
import hashlib
import time

# Configure logging
logger = logging.getLogger(__name__)

class UnifiedConceptExtractor:
    """
    Unified concept extractor with enhanced support for multiple languages
    and optimized extraction for academic content.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the concept extractor.

        Args:
            language: Default language code ('en' or 'ru')
        """
        self.language = language

        # Load enhanced stopwords and domain-specific patterns
        self._load_nlp_resources()

        logger.info(f"UnifiedConceptExtractor initialized for language: {language}")

    def _load_nlp_resources(self):
        """Load NLP resources including stopwords and domain-specific patterns."""
        # Enhanced stopwords for multiple languages
        self.stopwords = {
            'en': self._load_english_stopwords(),
            'ru': self._load_russian_stopwords()
        }

        # Domain-specific keywords that are important (should NOT be filtered)
        self.domain_keywords = {
            "physics": {
                "en": {
                    "quantum", "mechanics", "wave", "function", "operator", "state",
                    "eigenvalue", "eigenstate", "hamiltonian", "commutator", "hermitian",
                    "observable", "measurement", "probability", "amplitude", "schrodinger",
                    "dirac", "bra", "ket", "hilbert", "space", "vector", "momentum", "energy",
                    "position", "uncertainty", "principle", "entanglement", "superposition",
                    "degeneracy", "symmetry", "invariant", "transformation",
                    "spin", "angular", "potential", "barrier", "time-dependent", "time-independent"
                },
                "ru": {
                    "квантовый", "квантовая", "квантовое", "квантовые",
                    "механика", "волновая", "функция", "оператор", "состояние",
                    "собственное", "значение", "собственный", "вектор",
                    "гамильтониан", "коммутатор", "эрмитов", "эрмитово", "эрмитова", "эрмитовый",
                    "наблюдаемая", "измерение", "вероятность", "амплитуда", "шредингер",
                    "дирак", "бра", "кет", "гильбертово", "пространство", "вектор", "импульс", "энергия",
                    "положение", "неопределенность", "принцип", "запутанность", "суперпозиция",
                    "вырождение", "вырожденный", "симметрия", "инвариант", "преобразование",
                    "спин", "угловой", "потенциал", "барьер", "временной", "стационарный",
                    "волновой", "матрица", "плотности", "чистое", "смешанное"
                }
            },
            "mathematics": {
                "en": {"function", "variable", "equation", "theorem", "proof", "integral",
                      "derivative", "limit", "series", "vector", "matrix", "algebra",
                      "geometry", "calculus", "topology", "group", "ring", "field",
                      "manifold", "transformation", "linear", "differential", "algebraic"},
                "ru": {"функция", "переменная", "уравнение", "теорема", "доказательство",
                      "интеграл", "производная", "предел", "ряд", "вектор",
                      "матрица", "алгебра", "геометрия", "анализ", "топология",
                      "группа", "кольцо", "поле", "многообразие", "преобразование",
                      "линейный", "дифференциальный", "алгебраический"}
            },
            "programming": {
                "en": {"algorithm", "function", "class", "object", "method", "variable",
                      "array", "list", "loop", "recursion", "data", "structure", "complexity",
                      "runtime", "memory", "interface", "inheritance", "polymorphism"},
                "ru": {"алгоритм", "функция", "класс", "объект", "метод", "переменная",
                      "массив", "список", "цикл", "рекурсия", "данные", "структура",
                      "сложность", "время", "память", "интерфейс", "наследование", "полиморфизм"}
            }
        }

        # Patterns for theoretical/practical content
        self.theoretical_patterns = {
            'en': [
                r'is defined as', r'is called', r'refers to', r'is known as',
                r'can be described as', r'is a concept', r'is characterized by',
                r'is understood as', r'is formulated as', r'is represented by',
                r'is expressed as', r'is given by', r'is derived from', r'is related to',
                r'the definition of', r'the concept of', r'the theory of', r'the principle of',
                r'the law of', r'the equation for', r'according to the theory'
            ],
            'ru': [
                r'определяется как', r'называется', r'обозначает', r'известен как',
                r'можно описать как', r'является концепцией', r'характеризуется',
                r'понимается как', r'формулируется как', r'представлен как',
                r'выражается как', r'задается как', r'выводится из', r'связан с',
                r'определение', r'концепция', r'теория', r'принцип',
                r'закон', r'уравнение для', r'согласно теории'
            ]
        }

        self.practical_patterns = {
            'en': [
                r"let['']s", r'we (can|will|should|could)', r'you (can|will|should|could)',
                r'for example', r'as an example', r'step by step', r'how to',
                r'in practice', r'in this example', r'to solve this', r'to implement this',
                r'to calculate', r'to compute', r'let me show you', r'I\'ll demonstrate'
            ],
            'ru': [
                r'давайте', r'мы (можем|будем|должны|могли)', r'вы (можете|будете|должны|могли)',
                r'например', r'в качестве примера', r'шаг за шагом', r'как сделать',
                r'на практике', r'в этом примере', r'чтобы решить', r'для реализации',
                r'для вычисления', r'позвольте показать', r'я продемонстрирую', r'рассмотрим'
            ]
        }

        # Compile regex patterns
        self.theoretical_regex = {
            lang: re.compile('|'.join(patterns), re.IGNORECASE)
            for lang, patterns in self.theoretical_patterns.items()
        }

        self.practical_regex = {
            lang: re.compile('|'.join(patterns), re.IGNORECASE)
            for lang, patterns in self.practical_patterns.items()
        }

        # Filler phrases to remove by language
        self.filler_phrases = {
            "en": [
                r'^the\s+', r'^a\s+', r'^an\s+', r'^this\s+', r'^that\s+',
                r'^just\s+', r'^so\s+', r'^only\s+', r'^about\s+', r'^there\s+',
                r'^here\s+', r'^these\s+', r'^those\s+', r'^such\s+', r'^like\s+',
                r'^what\s+', r'^which\s+', r'^where\s+', r'^when\s+', r'^why\s+',
                r'^how\s+', r'^who\s+', r'^I\s+', r'^we\s+', r'^you\s+', r'^it\s+',
                r'\s+is$', r'\s+are$', r'\s+was$', r'\s+were$', r'\s+been$',
                r'\s+can$', r'\s+will$', r'\s+should$', r'\s+could$', r'\s+would$',
                r'\s+have$', r'\s+has$', r'\s+had$'
            ],
            "ru": [
                # Starting filler phrases
                r'^это\s+', r'^вот\s+', r'^та\s+', r'^тот\s+', r'^те\s+', r'^та\s+',
                r'^такая\s+', r'^такой\s+', r'^такое\s+', r'^такие\s+', r'^просто\s+',
                r'^только\s+', r'^лишь\s+', r'^да\s+', r'^ну\s+', r'^и\s+',
                r'^в\s+', r'^но\s+', r'^на\s+', r'^по\s+', r'^у\s+нас\s+',
                r'^мы\s+', r'^я\s+', r'^вы\s+', r'^они\s+', r'^он\s+', r'^она\s+',
                r'^оно\s+', r'^как\s+', r'^что\s+', r'^когда\s+', r'^где\s+',
                r'^потому\s+', r'^причин\s+', r'^здесь\s+', r'^тут\s+',
                r'^значит\s+', r'^теперь\s+', r'^итак\s+', r'^тогда\s+', r'^дальше\s+',
                r'^там\s+', r'^вообще\s+', r'^кстати\s+', r'^собственно\s+', r'^фактически\s+',

                # Problematic phrases explicitly identified
                r'^то\s+обсуждений\s+', r'^то\s+состояние\s+второго\s+определённо\s+',
                r'^состояние\s+едини\s+на2\s+', r'^гравитации\s+эйнштейна\s+',
                r'^этом\s+источнике\s+', r'^были\s+помере\s+',
                r'^ну\s+можно\s+убедиться\s+', r'^уже\s+содержится\s+',
                r'^потом\s+обсужу\s+', r'^сейчас\s+скажу\s+',

                # Ending phrases
                r'\s+должна$', r'\s+должен$', r'\s+должно$', r'\s+должны$',
                r'\s+может$', r'\s+могут$', r'\s+будет$', r'\s+будут$', r'\s+было$',
                r'\s+были$', r'\s+есть$', r'\s+имеет$', r'\s+имеют$', r'\s+нужно$',
                r'\s+нужна$', r'\s+надо$', r'\s+необходимо$', r'\s+требуется$',
                r'\s+следует$', r'\s+стоит$', r'\s+хочет$', r'\s+хотят$',
                r'\s+являются$', r'\s+является$'
            ]
        }

        # Invalid concepts lookup (direct terms that shouldn't be considered valid concepts)
        self.invalid_concepts = {
            "ru": {
                "то обсуждений давайте", "обсуждений давайте", "то обсуждений",
                "состояние едини на2", "состоянии вверх", "гравитации эйнштейна",
                "этом источнике", "были помере", "ну можно убедиться",
                "некоторого некоторой", "то состояние второго определённо такое",
                "то состояние второго", "состояние второго", "второго определённо",
                "сейчас скажу", "потом обсужу", "чем одно состояние",
                "приравняют формуле", "тета получается", "случаев равно",
                "потом эти", "можно убедиться", "некоторой функцией",
                "теперь рассмотрим", "рассмотрим теперь", "давайте вспомним",
                "давайте рассмотрим", "давайте теперь", "давайте сначала",
                "вспомним что", "возьмём тот", "можем заменить", "будем дальше",
                "буду получать", "запутанность давайте", "единицу поэтому",
                "давайте тогда", "эта процедура", "они отвечают", "должны тогда",
                "является давайте", "быть пропорциональна", "давайте возьмём",
                "слов давайте", "можно сказать", "теперь если", "стоит отметить",
                "все равно", "нет смысла", "это да", "да нет", "теперь давайте"
            },
            "en": {
                "we can see", "we can say", "this is", "that is", "it is", "it's",
                "there is", "there are", "we know", "let's", "we will",
                "as we know", "you can see", "you can find", "you know",
                "now let's", "now we can", "now let us", "let us now",
                "we can now", "we now", "we then", "first we", "then we"
            }
        }

    def _load_english_stopwords(self) -> Set[str]:
        """Load and return enhanced English stopwords."""
        try:
            # Try to import NLTK
            from nltk.corpus import stopwords
            nltk_stopwords = set(stopwords.words('english'))
        except:
            # Fallback to basic stopwords
            nltk_stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                't', 'can', 'will', 'just', 'don', 'should', 'now'
            }

        # Add more English stopwords and fillers
        additional_stopwords = {
            "uh", "um", "like", "so", "well", "actually", "basically",
            "literally", "sort", "kind", "really", "very", "quite",
            "okay", "ok", "yeah", "yes", "no", "right", "let", "just",
            "gonna", "going", "let's", "now", "here", "there", "this",
            "that", "these", "those", "will", "shall", "should", "would",
            "could", "can", "may", "might", "must", "although", "however"
        }

        return nltk_stopwords.union(additional_stopwords)

    def _load_russian_stopwords(self) -> Set[str]:
        """Load and return enhanced Russian stopwords."""
        try:
            # Try to import NLTK
            from nltk.corpus import stopwords
            nltk_stopwords = set(stopwords.words('russian'))
        except:
            # Fallback to basic stopwords
            nltk_stopwords = {
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
                'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
                'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
                'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
                'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него',
                'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
                'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо',
                'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без',
                'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
                'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
                'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'кажется',
                'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
                'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше',
                'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
                'разве', 'сказать', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою',
                'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой',
                'им', 'более', 'всегда', 'конечно', 'всю', 'между'
            }

        # Add enhanced Russian stopwords and fillers
        additional_stopwords = {
            "это", "вот", "так", "как", "ну", "да", "нет", "просто",
            "значит", "сейчас", "здесь", "тут", "уже", "если", "все", "всё",
            "хорошо", "там", "кстати", "итак", "будет", "ещё", "еще",
            "нас", "меня", "можно", "они", "только", "для", "поэтому", "равно",
            "нужно", "получается", "означает", "должна", "вами", "можем",
            "какой-то", "что-то", "стоит", "хочу", "буду", "видим", "понятно",
            "сделать", "например", "должны", "какие-то", "сюда", "плюс", "минус",
            "будем", "результат", "такое", "давайте", "рассмотрим"
        }

        return nltk_stopwords.union(additional_stopwords)

    def normalize_concept_text(self, text: str, language: str = None) -> str:
        """
        Normalize concept text with enhanced language-specific processing.

        Args:
            text: Concept text
            language: Language code

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Use provided language or default
        lang = language or self.language

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # First, check against invalid concepts list
        lang_key = lang if lang in self.invalid_concepts else 'en'
        if normalized in self.invalid_concepts.get(lang_key, {}):
            return ""  # Invalid concept

        # Apply filler phrase removal
        lang_key = lang if lang in self.filler_phrases else 'en'
        patterns = self.filler_phrases.get(lang_key, [])

        for pattern in patterns:
            normalized = re.sub(pattern, '', normalized)

        # Special handling for Russian
        if lang == "ru":
            # Fix common problematic phrases
            normalized = normalized.replace("то обсуждений давайте", "")
            normalized = normalized.replace("то состояние второго определённо такое", "")
            normalized = normalized.replace("вакуумное состояние оно", "вакуумное состояние")
            normalized = normalized.replace("эрмитово оператора", "эрмитов оператор")
            normalized = normalized.replace("любое собственное состояние оно", "собственное состояние")
            normalized = normalized.replace("любое состояние оно", "состояние")
            normalized = normalized.replace("состояние оно", "состояние")
            normalized = normalized.replace("второго определённо такое", "")
            normalized = normalized.replace("обсуждений давайте", "")
            normalized = normalized.replace("состояние едини на2", "")
            normalized = normalized.replace("некоторого некоторой", "")
            normalized = normalized.replace("приравняют формуле", "")

            # Fix partial removal of phrases that might leave dangling words
            normalized = re.sub(r'\s+(это|оно|вот|так|такое|такой|такая)$', '', normalized)
            normalized = re.sub(r'^(это|оно|вот|так|такое|такой|такая)\s+', '', normalized)

        # Remove any remaining leading/trailing whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Final check: if normalized text is just a simple conjunction or preposition, invalidate it
        simple_terms = {
            'en': {"the", "a", "an", "and", "or", "but", "if", "of", "in", "on", "at", "by", "for", "with", "about"},
            'ru': {"и", "или", "но", "если", "в", "на", "под", "над", "при", "у", "для", "о", "об", "к", "от", "из", "до", "с", "со"}
        }

        lang_key = lang if lang in simple_terms else 'en'
        if normalized in simple_terms.get(lang_key, set()):
            return ""

        return normalized

    def is_valid_concept(self, text: str, language: str = None) -> bool:
        """
        Check if text represents a valid concept with enhanced validation rules.

        Args:
            text: Concept text
            language: Language code

        Returns:
            True if valid concept, False otherwise
        """
        # Normalize and check validity
        normalized = self.normalize_concept_text(text, language)

        if not normalized:
            return False

        # Use provided language or default
        lang = language or self.language

        # Check minimum length
        if len(normalized) < 3:
            return False

        # Check against invalid concepts list
        invalid_concepts = self.invalid_concepts.get(lang, set())
        if normalized in invalid_concepts:
            return False

        # Check word count
        words = normalized.split()
        word_count = len(words)

        # Valid concept typically has 1-5 words
        if word_count < 1 or word_count > 5:
            return False

        # Check if it's mostly numbers
        if sum(c.isdigit() for c in normalized) / len(normalized) > 0.3:
            return False

        # Check for domain keywords
        domain_keywords = set()
        for domain, lang_keywords in self.domain_keywords.items():
            domain_keywords.update(lang_keywords.get(lang, set()))

        # If word count is 1, require it to be a domain keyword
        if word_count == 1 and normalized not in domain_keywords:
            # Single words must be domain keywords (physics, math terms)
            stopwords_set = self.stopwords.get(lang, self.stopwords.get('en', set()))
            if normalized in stopwords_set:
                return False

            # For Russian single words, additional validation
            if lang == 'ru':
                # Common verb endings that aren't usually concepts
                invalid_endings = ["ают", "еют", "ить", "ать", "еть", "уть", "еть", "ает", "ует"]
                if any(normalized.endswith(suffix) for suffix in invalid_endings):
                    return False

        # For multi-word concepts, check if at least one word is a domain keyword
        if word_count >= 2:
            # For Russian, specific validation for problematic phrases
            if lang == 'ru':
                # Check for phrases with "давайте" (let's) which are often invalid concepts
                if "давайте" in normalized:
                    return False

                # Check for phrases with "будем" (we will) which are often invalid concepts
                if "будем" in normalized:
                    return False

                # Check for phrases with forms of "мочь" (can) which are often invalid
                if any(word in normalized for word in ["можно", "можем", "могу", "могут", "могли"]):
                    return False

        return True

    def extract_concepts(
        self,
        text: str,
        domain: str = "physics",
        language: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from text with improved multilingual support.

        Args:
            text: Input text
            domain: Content domain
            language: Language code

        Returns:
            List of concept dictionaries
        """
        # Use provided language or default
        lang = language or self.language

        # Skip if text is empty
        if not text.strip():
            return []

        # Extract candidate concepts using multiple methods
        candidates = {}

        # 1. Extract domain-specific patterns (priority)
        pattern_matches = self._extract_domain_patterns(text, domain, lang)
        for pattern, count in pattern_matches.items():
            candidates[pattern] = {
                "text": pattern,
                "frequency": count,
                "score": count * 2.5,  # Higher weight for domain patterns
                "source": "domain_pattern",
                "domain_match": True
            }

        # 2. Extract n-grams (bigrams, trigrams)
        bigrams = self._extract_significant_bigrams(text, lang)
        for bigram, score in bigrams.items():
            if bigram in candidates:
                candidates[bigram]["score"] += score
            else:
                candidates[bigram] = {
                    "text": bigram,
                    "frequency": 1,
                    "score": score,
                    "source": "bigram"
                }

        trigrams = self._extract_significant_trigrams(text, lang)
        for trigram, score in trigrams.items():
            if trigram in candidates:
                candidates[trigram]["score"] += score * 1.2  # Higher weight for trigrams
            else:
                candidates[trigram] = {
                    "text": trigram,
                    "frequency": 1,
                    "score": score * 1.2,  # Higher weight for trigrams
                    "source": "trigram"
                }

        # 3. Extract definitional concepts
        definitions = self._extract_definitions(text, lang)
        for term, definition in definitions.items():
            score = 3.0  # High score for definitional contexts

            if term in candidates:
                candidates[term]["score"] += score
                candidates[term]["definition"] = definition
                candidates[term]["source"] = "definition"
            else:
                candidates[term] = {
                    "text": term,
                    "frequency": 1,
                    "score": score,
                    "definition": definition,
                    "source": "definition"
                }

        # 4. Direct search for important domain terms
        if domain == "physics":
            important_terms = self.domain_keywords.get("physics", {}).get(lang, set())

            for term in important_terms:
                if term.lower() in text.lower():
                    term_score = 3.0  # High score for important domain terms
                    if term in candidates:
                        candidates[term]["score"] += term_score
                    else:
                        candidates[term] = {
                            "text": term,
                            "frequency": 1,
                            "score": term_score,
                            "source": "direct_match",
                            "domain_match": True
                        }

        # 5. Filter and validate candidates
        filtered_candidates = {}

        for term, data in candidates.items():
            # Strict validation of concepts
            if not self.is_valid_concept(term, lang):
                continue

            # Apply higher score threshold for improved quality
            if data["score"] < 1.0:  # Increased threshold for better quality
                continue

            # Normalize concept text
            normalized_text = self.normalize_concept_text(term, lang)
            if not normalized_text:
                continue

            # Generate a concept ID
            concept_id = hashlib.md5(f"{normalized_text}:{domain}:{lang}".encode()).hexdigest()

            # Classify as theoretical or practical
            is_theoretical = self._is_theoretical_concept(term, text, domain, lang)

            # Create the concept entry
            filtered_candidates[normalized_text] = {
                "text": term,
                "normalized_text": normalized_text,
                "concept_id": concept_id,
                "frequency": data.get("frequency", 1),
                "score": data.get("score", 0),
                "source": data.get("source", ""),
                "definition": data.get("definition", ""),
                "domain": domain,
                "language": lang,
                "theoretical": is_theoretical,
                "concept_class": "theoretical" if is_theoretical else "practical"
            }

        # 6. Convert to list and sort by score
        concepts = list(filtered_candidates.values())
        concepts.sort(key=lambda x: x["score"], reverse=True)

        # 7. Limit to top concepts
        max_concepts = 50  # Reduced from 100 to focus on highest quality

        return concepts[:max_concepts]

    def _extract_domain_patterns(
        self,
        text: str,
        domain: str,
        language: str
    ) -> Dict[str, int]:
        """
        Extract domain-specific patterns from text.

        Args:
            text: Input text
            domain: Content domain
            language: Language code

        Returns:
            Dictionary of matched patterns and their counts
        """
        matches = {}

        # Physics domain patterns
        patterns = {
            "physics": {
                'en': [
                    r'(wave|quantum) (function|state|mechanics)',
                    r'(eigen)(value|state|vector|function)',
                    r'(hermitian|linear|unitary) (operator)',
                    r'(hamiltonian|momentum|position|energy) (operator)',
                    r'(time[\-\s])(dependent|independent|evolution)',
                    r'(uncertainty) (principle|relation)',
                    r'(quantum) (entanglement|superposition|measurement)',
                    r'(probability) (amplitude|density|distribution)',
                    r'(schrodinger|dirac) (equation|notation|formalism)',
                    r'(hilbert) (space)',
                    r'(bra|ket) (vector|notation)'
                ],
                'ru': [
                    r'(волнов[а-я]+) (функци[а-я]+|состояни[а-я]+|механик[а-я]+)',
                    r'(квантов[а-я]+) (механик[а-я]+|состояни[а-я]+|теори[а-я]+)',
                    r'(собственн[а-я]+) (значени[а-я]+|состояни[а-я]+|вектор[а-я]+|функци[а-я]+)',
                    r'(эрмитов[а-я]*) (оператор[а-я]*)',
                    r'(гамильтониан[а-я]*|импульс[а-я]*|координат[а-я]*|энерги[а-я]*) (оператор[а-я]*)',
                    r'(временн[а-я]+) (зависимост[а-я]+|независимост[а-я]+|эволюци[а-я]+)',
                    r'(принцип|соотношение) (неопределенност[а-я]+)',
                    r'(квантов[а-я]+) (запутанност[а-я]+|суперпозици[а-я]+|измерени[а-я]+)',
                    r'(вероятностн[а-я]+) (амплитуд[а-я]+|плотност[а-я]+|распределени[а-я]+)',
                    r'(шредингер[а-я]+|дирак[а-я]+) (уравнени[а-я]+|обозначени[а-я]+|формализм[а-я]*)',
                    r'(гильбертов[а-я]+) (пространств[а-я]+)',
                    r'(бра|кет) (вектор[а-я]+|обозначени[а-я]+)',
                    r'(матриц[а-я]+) (плотност[а-я]+)',
                    r'(квантов[а-я]+) (числ[а-я]+)',
                    r'(скалярн[а-я]+) (произведени[а-я]+)',
                    r'(вакуумное) (состояние)',
                    r'(основн[а-я]+) (состояни[а-я]+)',
                    r'(возбужденн[а-я]+) (состояни[а-я]+)',
                    r'(чист[а-я]+) (состояни[а-я]+)',
                    r'(смешанн[а-я]+) (состояни[а-я]+)',
                    r'(номерн[а-я]+) (базис[а-я]*)',
                    r'(тензорн[а-я]+) (произведени[а-я]+)',
                    r'(прям[а-я]+) (произведени[а-я]+)',
                    r'(унитарн[а-я]+) (оператор[а-я]*|преобразовани[а-я]*)'
                ]
            },
            "mathematics": {
                'en': [
                    r'(linear) (algebra|transformation|map|operator)',
                    r'(differential) (equation|form|geometry|calculus)',
                    r'(partial) (derivative|differential)',
                    r'(vector) (space|field|bundle|calculus)',
                    r'(matrix) (multiplication|algebra|theory|decomposition)',
                    r'(function) (space|theory|analysis)'
                ],
                'ru': [
                    r'(линейн[а-я]+) (алгебр[а-я]+|преобразовани[а-я]+|отображени[а-я]+|оператор[а-я]+)',
                    r'(дифференциальн[а-я]+) (уравнени[а-я]+|форм[а-я]+|геометри[а-я]+|исчислени[а-я]+)',
                    r'(частн[а-я]+) (производн[а-я]+|дифференциал[а-я]+)',
                    r'(вектор[а-я]+) (пространств[а-я]+|пол[а-я]+|расслоени[а-я]+|исчислени[а-я]+)',
                    r'(матричн[а-я]+) (умножени[а-я]+|алгебр[а-я]+|теори[а-я]+|разложени[а-я]+)',
                    r'(функциональн[а-я]+) (пространств[а-я]+|теори[а-я]+|анализ[а-я]+)'
                ]
            }
        }

        # Get patterns for this domain and language
        domain_patterns = patterns.get(domain, {}).get(language, patterns.get(domain, {}).get('en', []))

        if not domain_patterns:
            return matches

        # Compile patterns
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in domain_patterns]

        # Find all matches
        for pattern in compiled_patterns:
            for match in pattern.finditer(text.lower()):
                # Get the matched text
                match_text = text[match.start():match.end()]

                # Normalize to remove extra whitespace
                match_text = ' '.join(match_text.split()).strip()

                if match_text:
                    # Validate before adding
                    normalized = self.normalize_concept_text(match_text, language)
                    if normalized and len(normalized) >= 3:
                        if self.is_valid_concept(normalized, language):
                            matches[match_text] = matches.get(match_text, 0) + 1

        # Special processing for Russian physics terms
        if domain == "physics" and language == "ru":
            # List of important quantum physics bigrams/trigrams
            quantum_phrases = [
                "волновая функция", "собственное состояние", "собственное значение",
                "эрмитов оператор", "операторы рождения", "операторы уничтожения",
                "квантовое состояние", "квантовая механика", "принцип неопределенности",
                "волновой пакет", "энергетический уровень", "стационарное состояние",
                "квантовый осциллятор", "вакуумное состояние", "квантовая система",
                "гамильтониан системы", "скалярное произведение", "матрица плотности",
                "спиновое состояние", "угловой момент", "оператор энергии",
                "оператор импульса", "оператор координаты", "уравнение Шредингера",
                "базис состояний", "дискретный спектр", "непрерывный спектр",
                "коэффициент разложения", "нормировка функции", "гармонический осциллятор"
            ]

            # Search for phrases
            for phrase in quantum_phrases:
                count = text.lower().count(phrase)
                if count > 0 and self.is_valid_concept(phrase, language):
                    matches[phrase] = matches.get(phrase, 0) + count * 2  # Higher weight for phrases

        return matches

    def _extract_significant_bigrams(self, text: str, language: str = "en") -> Dict[str, float]:
        """
        Extract significant bigrams from text.

        Args:
            text: Input text
            language: Language code

        Returns:
            Dictionary of bigrams with their scores
        """
        # Get stopwords for the language
        stopwords_set = self.stopwords.get(language, self.stopwords.get('en', set()))

        # Tokenize text
        tokens = text.lower().split()

        # Filter stopwords and short tokens
        filtered_tokens = [token for token in tokens
                          if token not in stopwords_set
                          and token not in string.punctuation
                          and len(token) > 2]

        # Skip if too few tokens
        if len(filtered_tokens) < 3:
            return {}

        # Extract bigrams
        bigrams = []
        for i in range(len(filtered_tokens) - 1):
            # Skip bigrams where both tokens are the same
            if filtered_tokens[i] != filtered_tokens[i+1]:
                # Create the bigram
                bigram = f"{filtered_tokens[i]} {filtered_tokens[i+1]}"
                # Only add if it's long enough
                if len(bigram) >= 3:
                    bigrams.append((filtered_tokens[i], filtered_tokens[i+1]))

        # Count frequencies
        bigram_counts = Counter(bigrams)

        # Skip if no repeated bigrams
        if len(bigram_counts) == 0:
            return {}

        # Calculate scores based on frequency
        max_count = max(bigram_counts.values()) if bigram_counts else 1

        # Convert to string format and calculate scores
        bigram_scores = {}
        for (word1, word2), count in bigram_counts.items():
            # Include bigrams that appear at least once
            bigram_text = f"{word1} {word2}"

            # Score is based on frequency and normalized by max count
            score = (count / max_count) * 2.0

            # Boost score for domain-specific terms
            if language == "ru" and any(keyword in [word1, word2] for keyword in
                                      ["квантовый", "квантовая", "собственное", "эрмитов", "эрмитово",
                                       "волновая", "функция", "состояние", "оператор", "гамильтониан"]):
                score *= 1.5

            # Validate bigram and only add if valid
            if self.is_valid_concept(bigram_text, language):
                bigram_scores[bigram_text] = score

        return bigram_scores

    def _extract_significant_trigrams(self, text: str, language: str = "en") -> Dict[str, float]:
        """
        Extract significant trigrams from text.

        Args:
            text: Input text
            language: Language code

        Returns:
            Dictionary of trigrams with their scores
        """
        # Get stopwords for the language
        stopwords_set = self.stopwords.get(language, self.stopwords.get('en', set()))

        # Tokenize text
        tokens = text.lower().split()

        # Filter stopwords and short tokens
        filtered_tokens = [token for token in tokens
                          if token not in stopwords_set
                          and token not in string.punctuation
                          and len(token) > 2]

        # Skip if too few tokens
        if len(filtered_tokens) < 4:
            return {}

        # Extract trigrams
        trigrams = []
        for i in range(len(filtered_tokens) - 2):
            # Only use trigrams with unique tokens
            if len(set([filtered_tokens[i], filtered_tokens[i+1], filtered_tokens[i+2]])) >= 2:
                # Create the trigram
                trigram = f"{filtered_tokens[i]} {filtered_tokens[i+1]} {filtered_tokens[i+2]}"
                # Only add if it's long enough
                if len(trigram) >= 5:
                    trigrams.append((filtered_tokens[i], filtered_tokens[i+1], filtered_tokens[i+2]))

        # Count frequencies
        trigram_counts = Counter(trigrams)

        # Skip if no repeated trigrams
        if len(trigram_counts) == 0:
            return {}

        # Calculate scores based on frequency
        max_count = max(trigram_counts.values()) if trigram_counts else 1

        # Convert to string format and calculate scores
        trigram_scores = {}
        for (word1, word2, word3), count in trigram_counts.items():
            # Include all trigrams
            trigram_text = f"{word1} {word2} {word3}"

            # Score is based on frequency and normalized by max count, with a boost for trigrams
            score = (count / max_count) * 2.5

            # Boost score for domain-specific terms
            if language == "ru" and any(keyword in [word1, word2, word3] for keyword in
                                      ["квантовый", "квантовая", "собственное", "эрмитов", "эрмитово",
                                       "волновая", "функция", "состояние", "оператор", "гамильтониан"]):
                score *= 1.5

            # Validate trigram and only add if valid
            if self.is_valid_concept(trigram_text, language):
                trigram_scores[trigram_text] = score

        return trigram_scores

    def _extract_definitions(self, text: str, language: str) -> Dict[str, str]:
        """
        Extract definitions from text with enhanced pattern recognition.

        Args:
            text: Input text
            language: Language code

        Returns:
            Dictionary mapping terms to their definitions
        """
        definitions = {}

        # Definition patterns by language
        patterns = {
            'en': [
                r'([\w\s]+) (?:is|are) defined as ([\w\s,]+)',
                r'([\w\s]+) (?:refers to|means|is called) ([\w\s,]+)',
                r'(?:the|a) (?:concept|definition) of ([\w\s]+) is ([\w\s,]+)',
                r'([\w\s]+) is (?:a|an) ([\w\s,]+)',  # Simple "is a" definition
                r'([\w\s]+) (?:is|are) (?:understood as|characterized by|represented by) ([\w\s,]+)'
            ],
            'ru': [
                r'([\w\s]+) (?:определяется как|это|является) ([\w\s,]+)',
                r'([\w\s]+) (?:называется|обозначает) ([\w\s,]+)',
                r'(?:понятие|определение) ([\w\s]+) (?:это|есть) ([\w\s,]+)',
                # Added additional patterns
                r'([\w\s]+) (?:означает|представляет собой|подразумевает) ([\w\s,]+)',
                r'под (?:термином|понятием)? ([\w\s]+) (?:понимается|подразумевается) ([\w\s,]+)',
                r'([\w\s]+) — это ([\w\s,]+)',  # Em dash definition
                r'([\w\s]+) - это ([\w\s,]+)'   # Regular dash definition
            ]
        }

        # Use patterns for this language or fall back to English
        lang_patterns = patterns.get(language, patterns['en'])

        # Find definitions
        for pattern in lang_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    term = match.group(1).strip().lower()
                    definition = match.group(2).strip()

                    # Skip very short terms
                    if len(term) < 3:
                        continue

                    # Skip if term doesn't pass validation
                    if not self.is_valid_concept(term, language):
                        continue

                    definitions[term] = definition

        return definitions

    def _is_theoretical_concept(
        self,
        concept: str,
        context: str,
        domain: str,
        language: str
    ) -> bool:
        """
        Determine if a concept is theoretical based on its context.

        Args:
            concept: Concept text
            context: Context text
            domain: Content domain
            language: Language code

        Returns:
            True if theoretical, False if practical
        """
        # Use language-specific patterns
        lang = language if language in self.theoretical_regex else 'en'

        # Check for theoretical and practical patterns in context
        theoretical_match = bool(self.theoretical_regex[lang].search(context))
        practical_match = bool(self.practical_regex[lang].search(context))

        # If clear match in one category, use that
        if theoretical_match and not practical_match:
            return True
        if practical_match and not theoretical_match:
            return False

        # Count words to estimate complexity - longer concepts tend to be more theoretical
        word_count = len(concept.split())

        # Domain-specific defaults
        if domain == "physics":
            # Physics concepts are more likely theoretical by default,
            # especially for more complex terms with multiple words
            if word_count >= 2:
                # For quantum physics concepts, most multi-word terms are theoretical
                return True

            # Check if concept contains domain-specific terms
            domain_keywords = self.domain_keywords.get(domain, {}).get(language, set())
            for word in concept.lower().split():
                if word in domain_keywords:
                    return True

        # For single words, depend on domain
        return True  # Default to theoretical for academic content

    def extract_concepts_from_segments(
        self,
        segments: List[Dict[str, Any]],
        domain: str = "physics",
        language: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from transcript segments with improved validation.

        Args:
            segments: List of transcript segments
            domain: Content domain
            language: Language code

        Returns:
            List of extracted concepts
        """
        # Use specified language or default
        lang = language or self.language

        # Track time
        start_time = time.time()

        # Combine all segment texts for initial extraction
        combined_text = " ".join([segment.get("text", "") for segment in segments])

        # Extract initial concepts from combined text
        combined_concepts = self.extract_concepts(combined_text, domain, lang)
        logger.info(f"Extracted {len(combined_concepts)} initial concepts from combined text")

        # Map each segment text to a unique ID for quick lookups
        segment_map = {segment.get("id", str(uuid.uuid4())): segment for segment in segments}

        # Process segments in batches to reduce memory pressure
        batch_size = 10
        all_segment_concepts = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]

            # Process each segment in the batch
            for segment in batch:
                segment_text = segment.get("text", "")
                if len(segment_text) > 30:  # Only process substantial segments
                    segment_concepts = self.extract_concepts(segment_text, domain, lang)
                    all_segment_concepts.extend(segment_concepts)

        logger.info(f"Extracted {len(all_segment_concepts)} additional concepts from segments")

        # Merge concepts from combined text and individual segments
        all_concepts = combined_concepts + all_segment_concepts

        # Deduplicate and consolidate - map by concept ID
        concept_map = {}
        for concept in all_concepts:
            concept_id = concept.get("concept_id")
            if not concept_id:
                continue

            if concept_id not in concept_map or concept.get("score", 0) > concept_map[concept_id].get("score", 0):
                concept_map[concept_id] = concept.copy()

                # Initialize occurrences list if not present
                if "occurrences" not in concept_map[concept_id]:
                    concept_map[concept_id]["occurrences"] = []

        # Find occurrences in segments
        logger.info("Finding concept occurrences in segments")
        for concept_id, concept in concept_map.items():
            concept_text = concept.get("text", "").lower()

            # Track segments containing this concept
            for segment_id, segment in segment_map.items():
                segment_text = segment.get("text", "").lower()

                if concept_text in segment_text:
                    # Create occurrence record
                    occurrence = {
                        "segment_id": segment_id,
                        "start_time": segment.get("start_time", 0),
                        "end_time": segment.get("end_time", 0),
                        "context_type": segment.get("content_type", "mixed"),
                        "context_text": segment.get("text", "")
                    }

                    # Add to concept's occurrences
                    concept["occurrences"].append(occurrence)

        # Update concept frequency based on actual occurrences
        for concept in concept_map.values():
            concept["frequency"] = len(concept.get("occurrences", []))

            # Verify theoretical vs practical based on occurrences
            occurrences = concept.get("occurrences", [])
            theoretical_count = sum(1 for o in occurrences if o.get("context_type") == "theoretical")
            practical_count = sum(1 for o in occurrences if o.get("context_type") == "practical")

            # Use majority vote across segments
            if theoretical_count > practical_count:
                concept["theoretical"] = True
                concept["concept_class"] = "theoretical"
            elif practical_count > theoretical_count:
                concept["theoretical"] = False
                concept["concept_class"] = "practical"

        # Convert to list and sort by frequency and score
        result_concepts = list(concept_map.values())
        result_concepts.sort(key=lambda x: (x.get("frequency", 0) * 2 + x.get("score", 0)), reverse=True)

        processing_time = time.time() - start_time
        logger.info(f"Concept extraction completed in {processing_time:.2f} seconds, found {len(result_concepts)} unique concepts")

        return result_concepts

    def is_domain_keyword(self, word: str, domain: str, language: str = None) -> bool:
        """
        Check if a word is a domain-specific keyword.

        Args:
            word: Word to check
            domain: Domain to check against
            language: Language code

        Returns:
            True if domain keyword, False otherwise
        """
        lang = language or self.language
        domain_keywords = self.domain_keywords.get(domain, {}).get(lang, set())

        return word.lower() in domain_keywords
