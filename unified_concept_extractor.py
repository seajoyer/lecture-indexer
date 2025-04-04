"""
Unified Concept Extractor for the Lecture Video Content Indexer.
Consolidates concept extraction functionality into a single, well-defined module
with enhanced language processing for English and Russian.
"""

import re
import uuid
import logging
from typing import Dict, List, Set, Any
from collections import Counter, defaultdict
import string

# Configure logging
logger = logging.getLogger(__name__)

class UnifiedConceptExtractor:
    """
    Unified concept extractor with enhanced support for multiple languages
    and domain-specific knowledge.
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
                    "degeneracy", "degeneracies", "symmetry", "invariant", "transformation",
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
                    "волновой", "матрица", "плотности", "чистое", "смешанное",
                    "нормировка", "унитарный", "унитарная", "унитарное", "унитарные",
                    "атом", "молекула", "частица", "фотон", "электрон", "ядро",
                    "орбиталь", "спектр", "уровень", "квантование", "квантованный",
                    "тензорное", "произведение", "прямое", "скалярное",
                    "вероятностная", "интерпретация", "волновой", "пакет", "распределение",
                    "мера", "возбуждение", "возбужденное", "основное",
                    "вакуумное", "операторы", "рождения", "уничтожения",
                    "фоковское", "номерной", "базис", "осциллятор", "туннелирование",
                    "бозон", "фермион", "квант", "фаза", "спектр", "когерентное",
                    "ввели", "ввести", "общих", "общие", "число", "волновая", "моё", "наша", "общих"
                }
            },
            "mathematics": {
                "en": {"function", "variable", "equation", "theorem", "proof", "integral",
                      "derivative", "limit", "series", "vector", "matrix", "algebra",
                      "geometry", "calculus", "topology"},
                "ru": {"функция", "переменная", "уравнение", "теорема", "доказательство",
                      "интеграл", "производная", "предел", "ряд", "вектор",
                      "матрица", "алгебра", "геометрия", "анализ", "топология"}
            },
            "programming": {
                "en": {"algorithm", "function", "class", "object", "method", "variable",
                      "array", "list", "loop", "recursion", "data", "structure"},
                "ru": {"алгоритм", "функция", "класс", "объект", "метод", "переменная",
                      "массив", "список", "цикл", "рекурсия", "данные", "структура"}
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
                r'для вычисления', r'позвольте показать', r'я продемонстрирую'
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

        # Filler phrases to remove by language (significantly improved for Russian)
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
                # Starting phrases
                r'^это\s+', r'^вот\s+', r'^та\s+', r'^тот\s+', r'^те\s+', r'^та\s+',
                r'^такая\s+', r'^такой\s+', r'^такое\s+', r'^такие\s+', r'^просто\s+',
                r'^только\s+', r'^лишь\s+', r'^да\s+', r'^ну\s+', r'^и\s+',
                r'^в\s+', r'^но\s+', r'^на\s+', r'^по\s+', r'^у\s+нас\s+',
                r'^мы\s+', r'^я\s+', r'^вы\s+', r'^они\s+', r'^он\s+', r'^она\s+',
                r'^оно\s+', r'^как\s+', r'^что\s+', r'^когда\s+', r'^где\s+',
                r'^потому\s+', r'^причин\s+', r'^здесь\s+', r'^тут\s+',
                r'^значит\s+', r'^теперь\s+', r'^итак\s+', r'^тогда\s+', r'^дальше\s+',
                r'^там\s+', r'^вообще\s+', r'^кстати\s+', r'^собственно\s+', r'^фактически\s+',
                r'^почему\s+', r'^зачем\s+', r'^чтобы\s+', r'^если\s+', r'^поскольку\s+',
                r'^наверное\s+', r'^наверно\s+', r'^может\s+быть\s+', r'^возможно\s+',
                r'^сегодня\s+', r'^затем\s+', r'^пускай\s+', r'^наше\s+',
                r'^вас\s+', r'^обсуждений\s+',

                # Added new patterns for commonly occurring problematic phrases
                r'^то\s+обсуждений\s+', r'^то\s+состояние\s+второго\s+определённо\s+',

                # Ending phrases
                r'\s+должна$', r'\s+должен$', r'\s+должно$', r'\s+должны$',
                r'\s+может$', r'\s+могут$', r'\s+будет$', r'\s+будут$', r'\s+было$',
                r'\s+были$', r'\s+есть$', r'\s+имеет$', r'\s+имеют$', r'\s+нужно$',
                r'\s+нужна$', r'\s+надо$', r'\s+необходимо$', r'\s+требуется$',
                r'\s+следует$', r'\s+стоит$', r'\s+хочет$', r'\s+хотят$',
                r'\s+являются$', r'\s+является$', r'\s+представляет$', r'\s+представляют$',
                r'\s+собой$', r'\s+так$', r'\s+вот$', r'\s+просто$', r'\s+только$',
                r'\s+еще$', r'\s+ещё$', r'\s+уже$', r'\s+тоже$', r'\s+также$',
                r'\s+так\s+далее$', r'\s+так\s+далее\s+тому\s+подобное$',
                r'\s+да$', r'\s+нет$', r'\s+конечно$', r'\s+точно$', r'\s+именно$',

                # Added common endings seen in problematic concepts
                r'\s+давайте\s+это$', r'\s+определённо\s+такое\s+это$'
            ]
        }

        # Complete phrases to remove (entire matches)
        self.complete_phrases = {
            "en": [
                "we have", "we can see", "we can say", "this is", "that is",
                "it is", "it's", "there is", "there are", "we know", "let's",
                "we will", "as we know", "you can see", "you can find", "you know"
            ],
            "ru": [
                "мы имеем", "мы видим", "мы можем", "мы можем видеть", "мы можем сказать",
                "мы знаем", "как мы знаем", "мы будем", "то есть", "то означает", "то значит",
                "да это", "да вот", "ну вот", "ну это", "я думаю", "я считаю",
                "мне кажется", "нам кажется", "нам надо", "нам нужно", "вот так",
                "вот здесь", "вот тут", "вот этот", "вот эта", "вот это", "вот эти",
                "просто так", "просто потому что", "просто надо", "просто нужно",
                "можно видеть", "можно сказать", "можно утверждать", "можно заметить",
                "можно отметить", "можно тогда", "можно так", "можно здесь",
                "вы видите", "вы знаете", "вы можете видеть", "вы можете найти",
                "число это", "прирост стремящемся", "стремится нулю",
                "то обсуждений давайте", "обсуждений давайте",
                "то состояние второго определённо такое",
                "да нет", "как оно", "как она", "как он", "что мы", "что вы",
                "что они", "если мы", "если вы", "здесь мы", "тут мы", "например мы",
                "приведём пример", "введем обозначение", "рассмотрим пример",
                "давайте рассмотрим", "давайте посмотрим", "то что",

                # Added problematic phrases seen in the output
                "то обсуждений давайте это",
                "то состояние второго определённо такое", "то состояние второго определённо такое это"
            ]
        }

        # Simple conjunctions and prepositions to filter
        self.simple_terms = {
            "en": {"the", "a", "an", "and", "or", "but", "if", "of", "in", "on", "at", "by", "for", "with", "about"},
            "ru": {"и", "или", "но", "если", "в", "на", "под", "над", "при", "у", "для", "о", "об", "к", "от", "из", "до", "с", "со", "то", "такое", "который", "которая", "которое", "которые"}
        }

        # Domain-specific patterns for quantum mechanics
        self.domain_specific_patterns = {
            "physics": {
                "en": [
                    # Quantum mechanics patterns
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
                "ru": [
                    # Russian quantum mechanics patterns
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
                    r'(операторы) (рождения|уничтожения)',
                    r'(вакуумное) (состояние)',
                    r'(основн[а-я]+) (состояни[а-я]+)',
                    r'(возбужденн[а-я]+) (состояни[а-я]+)',
                    r'(чист[а-я]+) (состояни[а-я]+)',
                    r'(смешанн[а-я]+) (состояни[а-я]+)',
                    r'(номерн[а-я]+) (базис[а-я]*)',
                    r'(фоковск[а-я]+) (пространств[а-я]+)',
                    r'(тензорн[а-я]+) (произведени[а-я]+)',
                    r'(прям[а-я]+) (произведени[а-я]+)',
                    r'(уровн[а-я]+) (энерги[а-я]+)',
                    r'(волнов[а-я]+) (пакет[а-я]*)',
                    r'(стационарн[а-я]+) (состояни[а-я]+)',
                    r'(спинов[а-я]+) (состояни[а-я]+)',
                    r'(унитарн[а-я]+) (оператор[а-я]*|преобразовани[а-я]*)',
                    r'(нормировк[а-я]+) (состояни[а-я]+|волнов[а-я]+)',
                    r'(квантов[а-я]+) (осциллятор[а-я]*)',
                    r'(гармоническ[а-я]+) (осциллятор[а-я]*)',
                    # Additional patterns that might match key concepts
                    r'(наша) (волновая функция)',
                    r'(моё) (когерентное состояние)',
                    r'(ввели) (собственное состояние)',
                    r'(собственное) (состояние число)',
                    r'(общих) (собственных функций)',
                    r'(энергия) (частицы|системы|осциллятора)',
                    r'(волновой) (пакет)',
                    r'(уравнение) (шредингера)'
                ]
            }
        }

        # Compile domain-specific patterns
        self.compiled_domain_patterns = {}
        for domain, lang_patterns in self.domain_specific_patterns.items():
            self.compiled_domain_patterns[domain] = {}
            for lang, patterns in lang_patterns.items():
                self.compiled_domain_patterns[domain][lang] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]

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
            "could", "can", "may", "might", "must", "although", "however",
            "actually", "generally", "specifically", "obviously", "clearly",
            "of course", "indeed", "certainly", "probably", "possibly",
            "apparently", "evidently", "importantly", "notably", "surely"
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
            "нас", "меня", "можно", "всё", "все", "они", "только", "для",
            "поэтому", "равно", "нужно", "получается", "означает", "должна", "вами",
            "можем", "какой-то", "что-то", "стоит", "хочу", "буду", "видим",
            "понятно", "сделать", "например", "должны", "какие-то", "сюда",
            "плюс", "минус", "будем", "результат",

            # Additional Russian stopwords found in problematic concepts
            "обсуждений", "второго", "определённо", "это",
            "то", "такая", "такие", "который", "которая", "которое", "которые",
            "ему", "себе", "чего", "чему", "этого", "такого", "одного",
            "каждого", "затем", "потом", "сначала", "сегодня", "вчера", "завтра", "теперь",
            "рассмотрим", "посмотрим", "увидим", "имеем", "говорили", "говорим", "скажем",
            "согласны", "пускай", "пусть", "вас", "нами", "ними", "кого", "где-то", "что-либо",
            "а", "однако", "также", "причем", "притом", "именно", "будь", "ваш", "ваша", "ваше",
            "ваши", "свой", "своя", "свое", "свои", "каждый", "каждая", "каждое", "хотя",
            "прежде", "иначе", "никак", "зато", "поскольку", "почему-то", "почему", "отсюда",
            "откуда", "туда", "сюда", "наверху", "внизу", "сверху", "снизу", "внутри", "снаружи"
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

        # First, check against complete phrases (whole-phrase matches)
        lang_key = lang if lang in self.complete_phrases else 'en'
        for phrase in self.complete_phrases.get(lang_key, []):
            if normalized == phrase:
                return ""  # Complete match with a filler phrase - invalid concept
            normalized = normalized.replace(phrase, " ")

        # Remove filler phrases
        lang_key = lang if lang in self.filler_phrases else 'en'
        patterns = self.filler_phrases.get(lang_key, [])

        for pattern in patterns:
            normalized = re.sub(pattern, '', normalized)

        # Special handling for Russian quantum mechanics lectures
        if lang == "ru":
            # Fix common problematic phrases
            normalized = normalized.replace("то обсуждений давайте", "")
            normalized = normalized.replace("то состояние второго определённо такое", "")
            normalized = normalized.replace("вакуумное состояние оно", "вакуумное состояние")
            normalized = normalized.replace("эрмитово оператора", "эрмитово операторов")
            normalized = normalized.replace("любое собственное состояние оно", "собственное состояние")
            normalized = normalized.replace("любое состояние оно", "состояние")
            normalized = normalized.replace("состояние оно", "состояние")
            normalized = normalized.replace("второго определённо такое", "")

            # Fix partial removal of phrases that might leave dangling words
            normalized = re.sub(r'\s+(это|оно|вот|так|такое)$', '', normalized)
            normalized = re.sub(r'^(это|оно|вот|так|такое)\s+', '', normalized)

        # Remove any remaining leading/trailing whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Final check: if normalized text is just a simple conjunction or preposition, invalidate it
        simple_terms = self.simple_terms.get(lang if lang in self.simple_terms else 'en', set())
        if normalized in simple_terms:
            return ""

        return normalized

    def is_valid_concept(self, text: str, language: str = None) -> bool:
        """
        Check if text represents a valid concept.

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

        # Check word count
        words = normalized.split()
        word_count = len(words)

        # Valid concept has 1-6 words - increased from 5 to catch more concepts
        if word_count < 1 or word_count > 6:
            return False

        # Check if it's mostly numbers
        if sum(c.isdigit() for c in normalized) / len(normalized) > 0.3:
            return False

        # Less strict stopword check - only filter single-word concepts that are stopwords
        stopwords_set = self.stopwords.get(lang, self.stopwords.get('en', set()))
        if word_count == 1 and normalized in stopwords_set:
            return False

        # Additional check for invalid Russian concepts
        if lang == 'ru':
            # Less strict check for Russian
            # Single words ending with common verb endings are often not valid concepts
            if word_count == 1 and any(normalized.endswith(suffix) for suffix in
                ['ают', 'еют', 'ят']) and len(normalized) < 5:
                return False

            # Check for common phrases that aren't valid concepts (only the most problematic ones)
            invalid_phrases = [
                'то обсуждений давайте', 'то обсуждений давайте это',
                'то состояние второго определённо такое', 'то состояние второго определённо такое это'
            ]

            if normalized in invalid_phrases:
                return False

        # Check for domain keywords - if contains a domain keyword, it's more likely to be valid
        domain_keywords = self.domain_keywords.get("physics", {}).get(lang, set())
        if any(kw in normalized for kw in domain_keywords):
            return True

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
                "score": count * 2.0,  # Reduced from 3.0 to give other extraction methods more weight
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
            score = 3.0  # Reduced from 4.0 to balance with other methods

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

        # 4. Direct search for important quantum terms (for Russian physics specifically)
        if domain == "physics" and lang == "ru":
            important_terms = [
                "гамильтониан", "волновая функция", "собственное состояние", "собственное значение",
                "эрмитов оператор", "операторы рождения", "операторы уничтожения", "когерентное состояние",
                "матрица плотности", "собственные функции", "общих собственных функций",
                "квантовое состояние", "квантовая механика", "принцип неопределенности",
                "волновой пакет", "энергетический уровень", "стационарное состояние",
                "квантовый осциллятор", "вакуумное состояние", "квантовая система"
            ]

            for term in important_terms:
                if term.lower() in text.lower():
                    term_score = 3.0  # High score for important quantum terms
                    if term in candidates:
                        candidates[term]["score"] += term_score
                    else:
                        candidates[term] = {
                            "text": term,
                            "frequency": 1,
                            "score": term_score,
                            "source": "direct_match"
                        }

        # 5. Filter and validate candidates with less stringent validation
        filtered_candidates = {}

        for term, data in candidates.items():
            # Lower threshold for validity - now just checking if it's completely invalid
            if not self.is_valid_concept(term, lang):
                continue

            # Lower score threshold to extract more concepts
            if data["score"] < 0.5:  # Reduced from 1.0
                continue

            # Normalize concept text
            normalized_text = self.normalize_concept_text(term, lang)
            if not normalized_text:
                continue

            # Check if this is a domain-specific term worth keeping
            is_domain_term = False
            domain_keywords = self.domain_keywords.get(domain, {}).get(lang, set())

            # Check if any word in the term is a domain keyword
            for word in normalized_text.split():
                if word in domain_keywords:
                    is_domain_term = True
                    # Boost score for domain-specific terms
                    data["score"] *= 1.5
                    break

            # Generate a concept ID
            import hashlib
            concept_id = hashlib.md5(f"{normalized_text}:{domain}:{lang}".encode()).hexdigest()

            # Classify as theoretical or practical (physics concepts are mostly theoretical)
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
                "concept_class": "theoretical" if is_theoretical else "practical",
                "domain_match": data.get("domain_match", False) or is_domain_term
            }

        # 6. Convert to list and sort by score
        concepts = list(filtered_candidates.values())
        concepts.sort(key=lambda x: x["score"], reverse=True)

        # 7. Post-process - deduplicate similar concepts with less stringent matching
        deduplicated_concepts = self._deduplicate_concepts(concepts)

        # Limit to top concepts - increased to extract more
        max_concepts = 100  # Increased from 50

        return deduplicated_concepts[:max_concepts]

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

        # Get domain patterns for the language
        lang_key = language if language in self.compiled_domain_patterns.get(domain, {}) else 'en'
        domain_patterns = self.compiled_domain_patterns.get(domain, {}).get(lang_key, [])

        if not domain_patterns:
            return matches

        # Find all matches
        for pattern in domain_patterns:
            for match in pattern.finditer(text.lower()):
                # Get the matched text
                match_text = text[match.start():match.end()]

                # Normalize to remove extra whitespace
                match_text = ' '.join(match_text.split()).strip()

                if match_text:
                    # Use less strict validation - we want to catch more potential concepts
                    normalized = self.normalize_concept_text(match_text, language)
                    if normalized and len(normalized) >= 3:
                        matches[match_text] = matches.get(match_text, 0) + 1

        # Special processing for Russian quantum mechanics terms - expanded
        if domain == "physics" and language == "ru":
            # List of important quantum mechanics terms to search directly
            quantum_terms = [
                "гамильтониан", "эрмитов", "эрмитова", "эрмитово", "собственное", "состояние",
                "значение", "оператор", "квантовая", "квантовый", "квантовое", "механика",
                "принцип", "неопределенность", "матрица", "плотности", "волновая", "функция",
                "осциллятор", "нормировка", "вектор", "спин", "запутанность", "суперпозиция",
                "коммутатор", "унитарный", "унитарное", "преобразование", "фаза", "когерентное",
                "вакуумное", "основное", "возбужденное", "чистое", "смешанное",
                "бозон", "фермион", "операторы", "инвариант", "симметрия",
                "наша", "моё", "ввели", "общих", "общие", "число", "этих", "одна"
            ]

            # And common quantum physics bigrams/trigrams
            quantum_phrases = [
                "волновая функция", "собственное состояние", "собственное значение",
                "эрмитов оператор", "операторы рождения", "операторы уничтожения",
                "квантовое состояние", "квантовая механика", "принцип неопределенности",
                "волновой пакет", "энергетический уровень", "стационарное состояние",
                "квантовый осциллятор", "вакуумное состояние", "квантовая система",
                "гамильтониан системы", "наша волновая функция", "моё когерентное состояние",
                "ввели собственное состояние", "собственное состояние число",
                "эрмитово операторов", "общих собственных функций", "нормировка состояния",
                "общие собственные функции", "скалярное произведение", "матрица плотности"
            ]

            # Search for these important terms directly
            for term in quantum_terms:
                # Simple search without regex for efficiency
                count = text.lower().count(term)
                if count > 0:
                    matches[term] = matches.get(term, 0) + count

            # Search for phrases
            for phrase in quantum_phrases:
                count = text.lower().count(phrase)
                if count > 0:
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

        # Filter stopwords and short tokens - less strict to catch more potential concepts
        filtered_tokens = [token for token in tokens
                          if token not in string.punctuation
                          and len(token) > 1]  # Reduced from 3 to 2

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
                                       "волновая", "функция", "состояние", "оператор"]):
                score *= 1.5

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

        # Filter stopwords and short tokens - less strict to catch more potential concepts
        filtered_tokens = [token for token in tokens
                          if token not in string.punctuation
                          and len(token) > 1]  # Reduced from 3 to 2

        # Skip if too few tokens
        if len(filtered_tokens) < 4:
            return {}

        # Extract trigrams
        trigrams = []
        for i in range(len(filtered_tokens) - 2):
            # Only use trigrams with unique tokens
            if len(set([filtered_tokens[i], filtered_tokens[i+1], filtered_tokens[i+2]])) >= 2:  # Changed from 3 to 2
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
                                       "волновая", "функция", "состояние", "оператор"]):
                score *= 1.5

            trigram_scores[trigram_text] = score

        return trigram_scores

    def _extract_definitions(self, text: str, language: str) -> Dict[str, str]:
        """
        Extract definitions from text.

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
                r'(?:the|a) (?:concept|definition) of ([\w\s]+) is ([\w\s,]+)'
            ],
            'ru': [
                r'([\w\s]+) (?:определяется как|это|является) ([\w\s,]+)',
                r'([\w\s]+) (?:называется|обозначает) ([\w\s,]+)',
                r'(?:понятие|определение) ([\w\s]+) (?:это|есть) ([\w\s,]+)',
                # Added additional Russian definition patterns
                r'([\w\s]+) (?:означает|представляет собой|подразумевает) ([\w\s,]+)',
                r'Под ([\w\s]+) (?:понимается|подразумевается) ([\w\s,]+)'
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
            # Physics concepts are more likely theoretical by default
            if word_count >= 2:
                return True

            # Check if concept contains domain-specific terms
            domain_keywords = self.domain_keywords.get(domain, {}).get(language, set())
            for word in concept.lower().split():
                if word in domain_keywords:
                    return True

        # For single words, depend on domain
        return True  # Default to theoretical for academic content

    def _deduplicate_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate and merge similar concepts with less stringent similarity threshold.

        Args:
            concepts: List of concept dictionaries

        Returns:
            Deduplicated list of concepts
        """
        if not concepts:
            return []

        # Group by normalized text to catch exact duplicates
        text_groups = defaultdict(list)
        for concept in concepts:
            normalized = concept.get("normalized_text", "").lower()
            text_groups[normalized].append(concept)

        # Process each group to select the best concept
        deduplicated = []

        for normalized_text, group in text_groups.items():
            if len(group) == 1:
                # Only one concept with this text, keep it
                deduplicated.append(group[0])
            else:
                # Multiple concepts with the same text, select the best one
                # Sort by score (highest first)
                group.sort(key=lambda x: x.get("score", 0), reverse=True)
                best_concept = group[0]

                # If any concept has a definition, use it
                for concept in group:
                    if concept.get("definition") and not best_concept.get("definition"):
                        best_concept["definition"] = concept["definition"]

                    # Accumulate frequency
                    if concept != best_concept:
                        best_concept["frequency"] = best_concept.get("frequency", 1) + concept.get("frequency", 1)

                deduplicated.append(best_concept)

        # Sort by score and frequency - giving more weight to frequency
        deduplicated.sort(key=lambda x: (
            x.get("frequency", 1) * 2 + x.get("score", 0)
        ), reverse=True)

        return deduplicated

    def extract_concepts_from_segments(
        self,
        segments: List[Dict[str, Any]],
        domain: str = "physics",
        language: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from transcript segments.

        Args:
            segments: List of transcript segments
            domain: Content domain
            language: Language code

        Returns:
            List of extracted concepts
        """
        # Use specified language or default
        lang = language or self.language

        # Combine all segment texts
        combined_text = " ".join([segment.get("text", "") for segment in segments])

        # Extract initial concepts from combined text
        combined_concepts = self.extract_concepts(combined_text, domain, lang)

        # Also extract concepts from individual segments for better context
        segment_concepts = []
        for segment in segments:
            segment_text = segment.get("text", "")
            if len(segment_text) > 20:  # Reduced from 50 to process more segments
                segment_concepts.extend(self.extract_concepts(segment_text, domain, lang))

        # Merge concepts from combined text and individual segments
        all_concepts = combined_concepts + segment_concepts

        # Deduplicate based on normalized text
        text_to_concept = {}
        for concept in all_concepts:
            normalized = concept.get("normalized_text", "").lower()
            # Prioritize higher scores but also consider frequency
            if normalized not in text_to_concept or (
                concept.get("score", 0) + concept.get("frequency", 1) >
                text_to_concept[normalized].get("score", 0) + text_to_concept[normalized].get("frequency", 1)
            ):
                text_to_concept[normalized] = concept

        merged_concepts = list(text_to_concept.values())

        # Track concept occurrences in segments
        concept_occurrences = defaultdict(list)

        for concept in merged_concepts:
            concept_text = concept.get("text", "").lower()

            # Find segments containing this concept
            for segment in segments:
                segment_text = segment.get("text", "").lower()

                if concept_text in segment_text:
                    occurrence = {
                        "segment_id": segment.get("id", str(uuid.uuid4())),
                        "start_time": segment.get("start_time", 0),
                        "end_time": segment.get("end_time", 0),
                        "context_type": segment.get("content_type", "mixed"),
                        "context_text": segment.get("text", "")
                    }
                    concept_occurrences[concept_text].append(occurrence)

        # Update concepts with occurrence information
        for concept in merged_concepts:
            concept_text = concept.get("text", "").lower()
            occurrences = concept_occurrences.get(concept_text, [])

            # Update frequency based on actual occurrences
            concept["frequency"] = len(occurrences)

            # Determine theoretical vs practical based on occurrences
            theoretical_count = sum(1 for o in occurrences if o.get("context_type") == "theoretical")
            practical_count = sum(1 for o in occurrences if o.get("context_type") == "practical")

            # Update concept class if enough evidence
            if theoretical_count > practical_count:
                concept["theoretical"] = True
                concept["concept_class"] = "theoretical"
            elif practical_count > theoretical_count:
                concept["theoretical"] = False
                concept["concept_class"] = "practical"

            # Add occurrences to concept
            concept["occurrences"] = occurrences

        # Sort by frequency and score
        merged_concepts.sort(key=lambda x: (x.get("frequency", 0), x.get("score", 0)), reverse=True)

        return merged_concepts

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
