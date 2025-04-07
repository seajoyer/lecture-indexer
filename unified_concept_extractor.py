"""
Enhanced Unified Concept Extractor for the Lecture Video Content Indexer.
Redesigned with improved concept validation, educational significance detection,
and language-specific processing, particularly for Russian content.
"""

import re
import uuid
import logging
import time
import hashlib
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter, defaultdict
import string
import json

# Configure logging
logger = logging.getLogger(__name__)

class UnifiedConceptExtractor:
    """
    Enhanced concept extractor with robust validation and improved language handling.
    Significantly improves extraction quality, especially for Russian content.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the concept extractor.

        Args:
            language: Default language code ('en' or 'ru')
        """
        self.language = language

        # Load enhanced NLP resources
        self._load_nlp_resources()

        logger.info(f"UnifiedConceptExtractor initialized for language: {language}")

    def _load_nlp_resources(self):
        """Load comprehensive NLP resources including stopwords and domain-specific patterns."""
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
                    # Core quantum physics terms
                    "квантовый", "квантовая", "квантовое", "квантовые", "квантовость",
                    "механика", "волновая", "функция", "оператор", "состояние",
                    "собственное", "значение", "собственный", "вектор", "собственная",
                    "гамильтониан", "коммутатор", "эрмитов", "эрмитово", "эрмитова",
                    "наблюдаемая", "измерение", "вероятность", "амплитуда", "шредингер",
                    "дирак", "бра", "кет", "гильбертово", "пространство",
                    "импульс", "энергия", "положение", "координата", "координаты",
                    "неопределенность", "принцип", "запутанность", "суперпозиция",
                    "вырождение", "симметрия", "инвариант", "преобразование",
                    "спин", "угловой", "момент", "потенциал", "барьер",
                    "волновой", "матрица", "плотности", "чистое", "смешанное",

                    # Additional specific quantum terms
                    "операторы", "эрмитовский", "гамильтона", "волновую", "волновая",
                    "базис", "базисные", "вектора", "векторы", "матрицы", "матричный",
                    "стационарное", "нестационарное", "вероятностный", "амплитуда",
                    "амплитуды", "квадрат", "модуль", "нормировка", "нормированный",
                    "унитарный", "унитарное", "унитарная", "эволюция", "когерентность"
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
                      "линейный", "дифференциальный", "алгебраический", "оператор"}
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

        # Patterns for theoretical/practical content (used for concept classification)
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

        # Educational content markers for substantive explanations
        self.educational_markers = {
            "en": [
                r'important concept',
                r'key principle',
                r'fundamental idea',
                r'essential to understand',
                r'core concept',
                r'critical to',
                r'central idea',
                r'primarily concerned with',
                r'focuses on',
                r'the main',
                r'in depth',
                r'thoroughly',
                r'explain in detail',
                r'explore the',
                r'analyze',
                r'examine',
                r'investigate',
                r'detailed',
                r'significant',
                r'important',
                r'crucial',
                r'vital',
                r'key',
                r'central',
                r'underlying',
                r'foundation',
                r'basis',
                r'fundamental',
                r'primary',
                r'comprehensive',
                r'thorough',
                r'elaborate',
                r'rigorous',
                r'systematic',
                r'precise',
                r'specific',
                r'in-depth',
                r'detailed analysis',
                r'extensive discussion'
            ],
            "ru": [
                r'важная концепция',
                r'ключевой принцип',
                r'фундаментальная идея',
                r'необходимо понять',
                r'основная концепция',
                r'критически важно',
                r'центральная идея',
                r'в первую очередь',
                r'фокусируется на',
                r'главный',
                r'подробно',
                r'тщательно',
                r'объяснить детально',
                r'исследовать',
                r'анализировать',
                r'изучить',
                r'исследовать',
                r'детальный',
                r'значительный',
                r'важный',
                r'существенный',
                r'жизненно важный',
                r'ключевой',
                r'центральный',
                r'лежащий в основе',
                r'фундамент',
                r'основа',
                r'фундаментальный',
                r'главный',
                r'всесторонний',
                r'тщательный',
                r'подробный',
                r'строгий',
                r'систематический',
                r'точный',
                r'специфический',
                r'углубленный',
                r'детальный анализ',
                r'обширное обсуждение'
            ]
        }

        # Compile educational markers patterns
        self.educational_markers_regex = {
            lang: re.compile('|'.join(patterns), re.IGNORECASE)
            for lang, patterns in self.educational_markers.items()
        }

        # Comprehensive list of invalid concepts and patterns
        # These are direct terms or patterns that should never be considered valid concepts
        self.invalid_concepts = {
            "ru": {
                # Common invalid Russian phrases from lecture transcripts
                # Speech artifacts and transitional phrases
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
                "все равно", "нет смысла", "это да", "да нет", "теперь давайте",

                # Additional common fragments in Russian physics lectures
                "следующей лекции", "скажу сразу", "хотелось бы", "посмотрим далее",
                "видно что", "сделать вывод", "мы заметим", "я говорю", "что говорить",
                "сложно сказать", "хочу сказать", "напишем здесь", "пишем дальше",
                "пишем следующее", "заметим что", "понятно что", "очевидно что",
                "обратите внимание", "указывает на", "добавим еще", "подводя итоги",
                "перейдем к", "перейдём к", "переходим к", "будем использовать",
                "нам понадобится", "которые будут", "переходим к", "возвращаемся к",
                "буду использовать", "мы видим", "можем видеть", "как видим",
                "видим что", "хочу показать", "хочу объяснить", "необходимо отметить",
                "нужно понимать", "имеем ввиду", "что имеется", "писать буду",
                "выписываем здесь", "принимаем во", "имеем дело", "мы знаем",
                "хочу напомнить", "напомню что", "напомню вам", "вспомним здесь",
                "повторим что", "повторюсь что", "давайте вернемся", "давайте вернёмся",
                "вернёмся к", "вернемся к", "тогда имеем", "получаем сразу",
                "ещё раз", "еще раз", "подставляя в", "подставим сюда",
                "смотрим на", "посмотрим на", "видим здесь", "помним что",
                "вы видите", "посмотрите на", "посмотрите здесь", "смотрите сюда",
                "сейчас объясню", "сейчас покажу", "приступим к", "сначала рассмотрим",
                "соответственно получаем", "таким образом", "имеется ввиду"
            },
            "en": {
                "we can see", "we can say", "this is", "that is", "it is", "it's",
                "there is", "there are", "we know", "let's", "we will",
                "as we know", "you can see", "you can find", "you know",
                "now let's", "now we can", "now let us", "let us now",
                "we can now", "we now", "we then", "first we", "then we"
            }
        }

        # Create invalid patterns matching - these are regex patterns that will be used to invalidate concepts
        self.invalid_patterns = {
            "ru": [
                # Patterns that match Russian speech artifacts and lecture fragments
                r'^давайте\s+\w+',  # Any phrase starting with "давайте" (let's)
                r'^будем\s+\w+',    # Any phrase starting with "будем" (we will)
                r'^будет\s+\w+',    # Any phrase starting with "будет" (will be)
                r'^будут\s+\w+',    # Any phrase starting with "будут" (will be pl.)
                r'^мы\s+\w+',       # Any phrase starting with "мы" (we)
                r'^вы\s+\w+',       # Any phrase starting with "вы" (you)
                r'^я\s+\w+',        # Any phrase starting with "я" (I)
                r'^сейчас\s+\w+',   # Any phrase starting with "сейчас" (now)
                r'^теперь\s+\w+',   # Any phrase starting with "теперь" (now)
                r'^тогда\s+\w+',    # Any phrase starting with "тогда" (then)
                r'^здесь\s+\w+',    # Any phrase starting with "здесь" (here)
                r'^тут\s+\w+',      # Any phrase starting with "тут" (here)
                r'^это\s+\w+',      # Any phrase starting with "это" (this is)
                r'^то\s+\w+',       # Any phrase starting with "то" (that is)
                r'^так\s+\w+',      # Any phrase starting with "так" (so)
                r'^итак\s+\w+',     # Any phrase starting with "итак" (so)
                r'^далее\s+\w+',    # Any phrase starting with "далее" (further)
                r'^потом\s+\w+',    # Any phrase starting with "потом" (then)
                r'^если\s+\w+',     # Any phrase starting with "если" (if)
                r'^когда\s+\w+',    # Any phrase starting with "когда" (when)
                r'^зачем\s+\w+',    # Any phrase starting with "зачем" (why)
                r'^почему\s+\w+',   # Any phrase starting with "почему" (why)
                r'^какой\s+\w+',    # Any phrase starting with "какой" (which)
                r'^какая\s+\w+',    # Any phrase starting with "какая" (which)
                r'^какое\s+\w+',    # Any phrase starting with "какое" (which)
                r'^какие\s+\w+',    # Any phrase starting with "какие" (which)
                r'^как\s+\w+',      # Any phrase starting with "как" (how)
                r'^где\s+\w+',      # Any phrase starting with "где" (where)
                r'^куда\s+\w+',     # Any phrase starting with "куда" (where to)
                r'^откуда\s+\w+',   # Any phrase starting with "откуда" (where from)
                r'^можно\s+\w+',    # Any phrase starting with "можно" (can)
                r'^нужно\s+\w+',    # Any phrase starting with "нужно" (need to)
                r'^надо\s+\w+',     # Any phrase starting with "надо" (have to)
                r'^следует\s+\w+',  # Any phrase starting with "следует" (follows)
                r'^получается\s+\w+', # Any phrase starting with "получается" (it turns out)
                r'^значит\s+\w+',   # Any phrase starting with "значит" (means)

                # Invalid phrase endings
                r'\w+\s+давайте$',  # Any phrase ending with "давайте" (let's)
                r'\w+\s+например$', # Any phrase ending with "например" (for example)
                r'\w+\s+скажем$',   # Any phrase ending with "скажем" (let's say)
                r'\w+\s+видим$',    # Any phrase ending with "видим" (we see)
                r'\w+\s+смотрим$',  # Any phrase ending with "смотрим" (we look)
                r'\w+\s+думаем$',   # Any phrase ending with "думаем" (we think)
                r'\w+\s+заметим$',  # Any phrase ending with "заметим" (we note)
                r'\w+\s+также$',    # Any phrase ending with "также" (also)
                r'\w+\s+тоже$',     # Any phrase ending with "тоже" (also)
                r'\w+\s+ещё$',      # Any phrase ending with "ещё" (more)
                r'\w+\s+еще$',      # Any phrase ending with "еще" (more)
                r'\w+\s+будет$',    # Any phrase ending with "будет" (will be)
                r'\w+\s+дальше$',   # Any phrase ending with "дальше" (further)

                # Verb and verb forms that are often in invalid concepts
                r'\w+\s+получается$',  # Any phrase ending with "получается" (it turns out)
                r'\w+\s+получаем$',    # Any phrase ending with "получаем" (we get)
                r'\w+\s+видно$',       # Any phrase ending with "видно" (it's visible)
                r'\w+\s+знаем$',       # Any phrase ending with "знаем" (we know)
                r'\w+\s+помним$',      # Any phrase ending with "помним" (we remember)
                r'\w+\s+рассмотрим$',  # Any phrase ending with "рассмотрим" (we'll consider)
                r'\w+\s+увидим$',      # Any phrase ending with "увидим" (we'll see)
                r'\w+\s+запишем$',     # Any phrase ending with "запишем" (we'll write)
                r'\w+\s+запишем$',     # Any phrase ending with "запишем" (we'll write)
                r'\w+\s+пишем$',       # Any phrase ending with "пишем" (we write)
                r'\w+\s+понимаем$',    # Any phrase ending with "понимаем" (we understand)
                r'\w+\s+определяем$',  # Any phrase ending with "определяем" (we define)

                # Invalid verb phrase fragments in middle of text
                r'\w+\s+будем\s+\w+',  # Contains "будем" (we will)
                r'\w+\s+давайте\s+\w+', # Contains "давайте" (let's)
                r'\w+\s+можем\s+\w+',  # Contains "можем" (we can)
                r'\w+\s+могли\s+\w+',  # Contains "могли" (we could)
                r'\w+\s+следует\s+\w+', # Contains "следует" (it follows)
                r'\w+\s+посмотрим\s+\w+', # Contains "посмотрим" (let's look)
                r'\w+\s+покажем\s+\w+', # Contains "покажем" (we'll show)
                r'\w+\s+объясним\s+\w+', # Contains "объясним" (we'll explain)

                # Specific problematic Russian physicist lecture fragments
                r'записываем\s+\w+',   # Any phrases with "записываем" (we write down)
                r'рассмотрим\s+\w+',   # Any phrases with "рассмотрим" (let's consider)
                r'примем\s+\w+',       # Any phrases with "примем" (we'll assume)
                r'будем\s+считать',    # "будем считать" (we'll assume)
                r'будем\s+полагать',   # "будем полагать" (we'll assume)
                r'предположим\s+\w+',  # Any phrases with "предположим" (let's suppose)
                r'допустим\s+\w+',     # Any phrases with "допустим" (let's assume)
                r'заметим\s+\w+',      # Any phrases with "заметим" (we note)
                r'отметим\s+\w+',      # Any phrases with "отметим" (we note)
                r'вернемся\s+\w+',     # Any phrases with "вернемся" (let's return)
                r'вернёмся\s+\w+',     # Any phrases with "вернёмся" (let's return)
                r'перейдем\s+\w+',     # Any phrases with "перейдем" (let's move)
                r'перейдём\s+\w+',     # Any phrases with "перейдём" (let's move)
                r'упростим\s+\w+',     # Any phrases with "упростим" (let's simplify)
                r'обозначим\s+\w+',    # Any phrases with "обозначим" (let's denote)
                r'назовем\s+\w+',      # Any phrases with "назовем" (let's call)
                r'назовём\s+\w+',      # Any phrases with "назовём" (let's call)

                # Patterns of lecturer self-correction
                r'нет\s+не\s+то',      # "nет не то" (no not that)
                r'не\s+совсем\s+так',  # "не совсем так" (not quite like that)
                r'это\s+не\s+совсем',  # "это не совсем" (this is not quite)
                r'ой\s+\w+',           # Any phrases with "ой" (oops)
                r'извините\s+\w+',     # Any phrases with "извините" (sorry)
                r'прошу\s+прощения',   # "прошу прощения" (I apologize)

                # Specifically problematic quantum physics fragments from examples
                r'столбик\s+обсуждали', # "столбик обсуждали"
                r'отсюда\s+перепутать',  # "отсюда перепутать"
                r'величины\s+букву\s+выбрала', # "величины букву выбрала"
                r'ищутся\s+очевидны',   # "ищутся очевидны"
                r'привожу\s+пример',    # "привожу пример"
                r'молчите\s+показывайте', # "молчите показывайте"
                r'устремляя\s+нулю',     # "устремляя нулю"
                r'менять\s+условия',     # "менять условия"
                r'напутала\s+самое',     # "напутала самое"
                r'первоначально\s+сначала', # "первоначально сначала"
                r'представлении\s+поскольку', # "представлении поскольку"
                r'писать\s+дискретном',     # "писать дискретном"
                r'попробуем\s+операторы',   # "попробуем операторы"
                r'возникает\s+неё',         # "возникает неё"
                r'элементы\s+вычислять',    # "элементы вычислять"
                r'думаю\s+имейлы\s+свои',   # "думаю имейлы свои"

                # Common ending forms from invalid examples
                r'оператор\s+называемое',   # "оператор называемое"
                r'оператора\s+именно\s+величине', # "оператора именно величине"
                r'квадрат\s+такого',        # "квадрат такого"
                r'состояний\s+базис\s+энергий', # "состояний базис энергий"
                r'представлении\s+непрерывной', # "представлении непрерывной"
                r'вместо\s+индексов',       # "вместо индексов"
                r'поскольку\s+аналогия',    # "поскольку аналогия"
                r'выбрали\s+тепер\s+мым',   # "выбрали тепер мым"
                r'большинстве\s+случаев',   # "большинстве случаев"
                r'суммы\s+перейти',         # "суммы перейти"
                r'функции\s+слева\s+сопряжённое', # "функции слева сопряжённое"
                r'неважно\s+любой\s+оператор', # "неважно любой оператор"
                r'число\s+будут',           # "число будут"
                r'заканчивается\s+первая\s+глава', # "заканчивается первая глава"
                r'пропадёт\s+точно',        # "пропадёт точно"
                r'спин\s+вверх\s+проекцию', # "спин вверх проекцию"
                r'неделе\s+содержание',     # "неделе содержание"

                # Additional math/physics terms that aren't valid concepts
                r'противоречия\s+никакого', # "противоречия никакого"
                r'коэффициент\s+разложения', # "коэффициент разложения"
                r'следующей\s+главе\s+соответственно' # "следующей главе соответственно"
            ],
            "en": [
                # Patterns for English that would match speech artifacts, not concepts
                r'^let\'?s\s+\w+',     # Any phrase starting with "let's"
                r'^we\s+will\s+\w+',   # Any phrase starting with "we will"
                r'^we\s+can\s+\w+',    # Any phrase starting with "we can"
                r'^we\s+should\s+\w+', # Any phrase starting with "we should"
                r'^we\s+could\s+\w+',  # Any phrase starting with "we could"
                r'^we\s+need\s+to\s+\w+', # Any phrase starting with "we need to"
                r'^we\s+have\s+to\s+\w+', # Any phrase starting with "we have to"
                r'^you\s+can\s+\w+',   # Any phrase starting with "you can"
                r'^you\s+should\s+\w+', # Any phrase starting with "you should"
                r'^you\s+need\s+to\s+\w+', # Any phrase starting with "you need to"
                r'^i\s+will\s+\w+',    # Any phrase starting with "i will"
                r'^i\s+want\s+to\s+\w+', # Any phrase starting with "i want to"
                r'^now\s+\w+',         # Any phrase starting with "now"
                r'^here\s+\w+',        # Any phrase starting with "here"
                r'^there\s+\w+',       # Any phrase starting with "there"
                r'^this\s+is\s+\w+',   # Any phrase starting with "this is"
                r'^that\s+is\s+\w+',   # Any phrase starting with "that is"
                r'^it\s+is\s+\w+',     # Any phrase starting with "it is"
                r'^for\s+example\s+\w+', # Any phrase starting with "for example"
                r'^as\s+an\s+example\s+\w+', # Any phrase starting with "as an example"
                r'^in\s+this\s+case\s+\w+', # Any phrase starting with "in this case"
                r'^so\s+\w+',          # Any phrase starting with "so"
                r'^then\s+\w+',        # Any phrase starting with "then"
                r'^thus\s+\w+',        # Any phrase starting with "thus"
                r'^therefore\s+\w+',   # Any phrase starting with "therefore"
                r'^first\s+\w+',       # Any phrase starting with "first"
                r'^second\s+\w+',      # Any phrase starting with "second"
                r'^next\s+\w+',        # Any phrase starting with "next"
                r'^finally\s+\w+',     # Any phrase starting with "finally"
                r'^to\s+summarize\s+\w+', # Any phrase starting with "to summarize"

                # Ending patterns
                r'\w+\s+for\s+example$', # Any phrase ending with "for example"
                r'\w+\s+here$',         # Any phrase ending with "here"
                r'\w+\s+there$',        # Any phrase ending with "there"
                r'\w+\s+now$',          # Any phrase ending with "now"
                r'\w+\s+later$',        # Any phrase ending with "later"
                r'\w+\s+first$',        # Any phrase ending with "first"
                r'\w+\s+next$',         # Any phrase ending with "next"
                r'\w+\s+then$',         # Any phrase ending with "then"

                # Patterns with problematic verbs
                r'\w+\s+will\s+see\s+\w+', # Contains "will see"
                r'\w+\s+can\s+see\s+\w+',  # Contains "can see"
                r'\w+\s+will\s+show\s+\w+', # Contains "will show"
                r'\w+\s+can\s+show\s+\w+',  # Contains "can show"
                r'\w+\s+will\s+demonstrate\s+\w+', # Contains "will demonstrate"
                r'\w+\s+can\s+demonstrate\s+\w+',  # Contains "can demonstrate"
                r'\w+\s+will\s+explain\s+\w+',     # Contains "will explain"
                r'\w+\s+can\s+explain\s+\w+',      # Contains "can explain"
                r'\w+\s+will\s+describe\s+\w+',    # Contains "will describe"
                r'\w+\s+can\s+describe\s+\w+',     # Contains "can describe"
                r'\w+\s+will\s+define\s+\w+',      # Contains "will define"
                r'\w+\s+can\s+define\s+\w+',       # Contains "can define"
                r'\w+\s+will\s+write\s+\w+',       # Contains "will write"
                r'\w+\s+can\s+write\s+\w+',        # Contains "can write"
                r'\w+\s+will\s+use\s+\w+',         # Contains "will use"
                r'\w+\s+can\s+use\s+\w+',          # Contains "can use"
                r'\w+\s+will\s+discuss\s+\w+',     # Contains "will discuss"
                r'\w+\s+can\s+discuss\s+\w+',      # Contains "can discuss"

                # Specific lecturer patterns
                r'remember\s+\w+',        # Any phrases with "remember"
                r'recall\s+\w+',          # Any phrases with "recall"
                r'consider\s+\w+',        # Any phrases with "consider"
                r'suppose\s+\w+',         # Any phrases with "suppose"
                r'assume\s+\w+',          # Any phrases with "assume"
                r'note\s+that\s+\w+',     # Any phrases with "note that"
                r'observe\s+that\s+\w+',  # Any phrases with "observe that"
                r'notice\s+that\s+\w+',   # Any phrases with "notice that"
                r'let\s+us\s+\w+',        # Any phrases with "let us"
                r'let\s+me\s+\w+',        # Any phrases with "let me"
                r'shall\s+\w+',           # Any phrases with "shall"
                r'would\s+\w+',           # Any phrases with "would"
                r'could\s+\w+',           # Any phrases with "could"
                r'should\s+\w+',          # Any phrases with "should"
                r'might\s+\w+',           # Any phrases with "might"
                r'may\s+\w+',             # Any phrases with "may"
            ]
        }

        # Compile invalid pattern regexes for efficiency
        self.invalid_pattern_regexes = {
            lang: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for lang, patterns in self.invalid_patterns.items()
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
            "could", "can", "may", "might", "must", "although", "however",
            "thus", "therefore", "hence", "whereby", "wherein", "whereof",
            "whereas", "whither", "thither", "hither", "thence", "hence",
            "thereof", "therein", "thereby", "thereupon", "thereafter",
            "nevertheless", "nonetheless", "notwithstanding", "consequently",
            "accordingly", "furthermore", "moreover", "meanwhile", "afterward",
            "beforehand", "meanwhile", "anyway", "anyhow", "besides", "instead",
            "further", "rather", "yet", "still", "even", "also", "again", "then",
            "always", "often", "seldom", "never", "ever", "perhaps", "maybe",
            "possibly", "probably", "certainly", "definitely", "absolutely",
            "indeed", "surely", "obviously", "clearly", "evidently", "apparently",
            "seemingly", "reportedly", "reputedly", "supposedly", "allegedly",
            "presumably", "purportedly", "ostensibly", "outwardly", "superficially",
            "firstly", "secondly", "thirdly", "lastly", "finally", "ultimately",
            "eventually", "subsequently", "formerly", "previously", "recently",
            "lately", "nowadays", "today", "tomorrow", "yesterday", "earlier", "later",
            "soon", "immediately", "instantly", "presently", "currently", "formerly",
            "previously", "subsequently", "thereafter", "beforehand", "hereafter",
            "already", "yet", "still", "anymore", "anytime", "sometimes", "occasionally",
            "frequently", "regularly", "usually", "normally", "commonly", "generally",
            "typically", "traditionally", "historically", "culturally", "socially",
            "politically", "economically", "financially", "commercially", "industrially",
            "technologically", "scientifically", "medically", "academically", "educationally",
            "legally", "morally", "ethically", "philosophically", "psychologically",
            "emotionally", "physically", "mentally", "spiritually", "religiously",
            "theoretically", "practically", "effectively", "efficiently", "successfully",
            "hopefully", "thankfully", "fortunately", "unfortunately", "regrettably",
            "sadly", "happily", "luckily", "unluckily", "interestingly", "surprisingly",
            "amazingly", "astonishingly", "remarkably", "notably", "noticeably",
            "significantly", "considerably", "substantially", "marginally", "slightly",
            "somewhat", "fairly", "rather", "quite", "relatively", "comparatively",
            "approximately", "roughly", "about", "around", "nearly", "almost", "exactly",
            "precisely", "specifically", "particularly", "especially", "notably",
            "significantly", "markedly", "decidedly", "definitely", "certainly",
            "undoubtedly", "undeniably", "unquestionably", "indisputably", "indubitably",
            "doubtlessly", "decidedly", "positively", "absolutely", "totally", "completely",
            "entirely", "wholly", "fully", "thoroughly", "utterly", "perfectly", "purely",
            "simply", "merely", "just", "only", "solely", "exclusively", "specifically",
            "particularly", "peculiarly", "uniquely", "distinctly", "differently",
            "alternatively", "otherwise", "similarly", "likewise", "equally", "correspondingly",
            "analogously", "comparably", "equivalently", "identically", "uniformly", "consistently",
            "constantly", "continually", "continuously", "perpetually", "eternally", "everlastingly",
            "incessantly", "unceasingly", "unremittingly", "unrelentingly", "relentlessly",
            "persistently", "steadily", "steadfastly", "unfailingly", "invariably", "inevitably",
            "necessarily", "unavoidably", "inescapably", "inexorably", "irrevocably", "irreversibly",
            "irretrievably", "irremediably", "irreparably", "irredeemably", "hopelessly", "helplessly",
            "powerlessly", "impotently", "ineffectually", "vainly", "uselessly", "fruitlessly", "futilely"
        }

        # Specific terms for physics lectures that are not valid concepts by themselves
        physics_lecture_stopwords = {
            "lecture", "lesson", "course", "class", "semester", "topic",
            "section", "subsection", "chapter", "part", "example", "exercise",
            "homework", "solution", "problem", "question", "answer", "explanation",
            "derivation", "proof", "demonstration", "illustration", "figure", "diagram",
            "graph", "plot", "table", "equation", "formula", "expression", "relation",
            "identity", "rule", "law", "theorem", "lemma", "corollary", "proposition",
            "statement", "assertion", "claim", "argument", "reasoning", "logic", "approach",
            "method", "technique", "procedure", "process", "algorithm", "calculation",
            "computation", "analysis", "evaluation", "assessment", "examination", "investigation",
            "exploration", "study", "research", "review", "summary", "recap", "conclusion",
            "introduction", "background", "context", "framework", "structure", "organization",
            "arrangement", "configuration", "setup", "specification", "requirement", "constraint",
            "limitation", "restriction", "condition", "assumption", "hypothesis", "conjecture",
            "speculation", "theory", "model", "paradigm", "concept", "idea", "notion",
            "thought", "impression", "perception", "understanding", "interpretation",
            "meaning", "definition", "description", "explanation", "characterization",
            "formulation", "statement", "articulation", "expression", "representation",
            "depiction", "portrayal", "illustration"
        }

        return nltk_stopwords.union(additional_stopwords).union(physics_lecture_stopwords)

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
            # Basic stopwords
            "это", "вот", "так", "как", "ну", "да", "нет", "просто",
            "значит", "сейчас", "здесь", "тут", "уже", "если", "все", "всё",
            "хорошо", "там", "кстати", "итак", "будет", "ещё", "еще",
            "нас", "меня", "можно", "они", "только", "для", "поэтому", "равно",
            "нужно", "получается", "означает", "должна", "вами", "можем",
            "какой-то", "что-то", "стоит", "хочу", "буду", "видим", "понятно",
            "сделать", "например", "должны", "какие-то", "сюда", "плюс", "минус",
            "будем", "результат", "такое", "давайте", "рассмотрим",

            # Extended Russian lecture-specific stopwords
            "такой", "такая", "такие", "каждый", "каждая", "каждое", "каждые",
            "любой", "любая", "любое", "любые", "весь", "вся", "всё", "все",
            "наш", "наша", "наше", "наши", "ваш", "ваша", "ваше", "ваши",
            "свой", "своя", "своё", "свои", "тот", "та", "то", "те",
            "этот", "эта", "это", "эти", "такой", "такая", "такое", "такие",
            "который", "которая", "которое", "которые", "кто", "что", "чей", "чья",
            "чьё", "чьи", "какой", "какая", "какое", "какие", "сколько", "где",
            "куда", "откуда", "когда", "зачем", "почему", "как", "который", "которая",
            "которое", "которые", "чей", "чья", "чье", "чьи", "кем", "чем", "кого",
            "чего", "кому", "чему", "ком", "чем", "собой", "мной", "тобой", "нами",
            "вами", "ими", "тогда", "туда", "сюда", "оттуда", "отсюда", "везде",
            "всюду", "нигде", "никуда", "отовсюду", "повсюду", "всюду", "тут", "там",
            "здесь", "сейчас", "теперь", "потом", "затем", "далее", "дальше", "ещё",
            "еще", "снова", "опять", "уже", "пока", "покуда", "доколе", "насколько",
            "много", "мало", "немного", "немало", "несколько", "столько", "настолько",
            "весьма", "слишком", "очень", "совсем", "совершенно", "почти", "примерно",
            "приблизительно", "около", "именно", "точно", "ровно", "прямо", "просто",
            "только", "лишь", "исключительно", "единственно", "почти", "почти что",
            "едва", "чуть", "чуть-чуть", "немного", "слегка", "еле", "едва", "чисто",
            "совсем", "совершенно", "абсолютно", "полностью", "целиком", "полностью",
            "вполне", "отнюдь", "вовсе", "более", "менее", "меньше", "больше",

            # Academic Russian stopwords
            "лекция", "глава", "раздел", "тема", "подраздел", "параграф", "пункт",
            "пример", "задача", "решение", "доказательство", "упражнение", "вопрос",
            "ответ", "объяснение", "рассуждение", "вывод", "заключение", "следствие",
            "утверждение", "формула", "формулировка", "выражение", "равенство",
            "тождество", "уравнение", "неравенство", "система", "метод", "подход",
            "способ", "алгоритм", "процедура", "правило", "закон", "принцип",
            "аксиома", "лемма", "гипотеза", "концепция", "определение", "понятие",
            "термин", "обозначение", "запись", "нотация", "формализм", "описание"
        }

        # Russian lecture fillers and phrases that often appear in bad concepts
        russian_lecture_fillers = {
            # Common filler verbs and phrases that appear in lectures
            "давайте", "возьмем", "рассмотрим", "обсудим", "проанализируем", "сравним",
            "начнем", "продолжим", "закончим", "перейдем", "вернемся", "остановимся",
            "сосредоточимся", "обратимся", "посмотрим", "увидим", "заметим", "отметим",
            "напомним", "вспомним", "забудем", "пренебрежем", "учтем", "примем",
            "поймем", "допустим", "предположим", "устремим", "положим", "обозначим",
            "запишем", "выпишем", "вычислим", "применим", "используем", "найдем",
            "определим", "опустим", "сократим", "упростим", "раскроем", "развернем",
            "построим", "нарисуем", "изобразим", "начертим", "проведем", "соединим",
            "установим", "подтвердим", "проверим", "докажем", "покажем", "убедимся",
            "сверим", "уточним", "детализируем", "удостоверимся", "усвоим", "освоим",
            "выявим", "обнаружим", "придём", "прийдём", "подведём", "сделаем", "скажем",

            # Common filler phrases and fragments
            "на самом деле", "собственно говоря", "вообще говоря", "строго говоря",
            "кстати говоря", "честно говоря", "короче говоря", "как говорится",
            "так сказать", "как бы", "своего рода", "в общем", "в общем-то",
            "в принципе", "в сущности", "по сути", "по существу", "фактически",
            "практически", "реально", "буквально", "конкретно", "собственно",
            "откровенно говоря", "между прочим", "между тем", "вместе с тем",
            "тем не менее", "тем более", "на данный момент", "в данном случае",
            "в первую очередь", "прежде всего", "главным образом", "в основном",
            "как правило", "обычно", "в целом", "как всегда", "как обычно",
            "довольно", "достаточно", "почему-то", "зачем-то", "где-то", "когда-то",
            "как-то", "что-то", "почему-либо", "отчего-то", "вроде", "якобы",
            "будто бы", "как будто", "словно", "примерно", "приблизительно",
            "более-менее", "так далее", "так сказать", "иначе говоря", "другими словами",
            "то есть", "вернее", "точнее", "вернее сказать", "точнее сказать",
            "попросту говоря", "коротко говоря", "мягко говоря", "мягко выражаясь",
            "грубо говоря", "между нами говоря", "сказать по правде", "по правде говоря",
            "если можно так выразиться", "если можно так сказать"
        }

        # Combine all Russian stopwords
        return nltk_stopwords.union(additional_stopwords).union(russian_lecture_fillers)

    def normalize_concept_text(self, text: str, language: str = None) -> str:
        """
        Normalize concept text with enhanced language-specific processing.
        This is a critical step for improving concept quality.

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
        # Common filler phrases by language
        filler_phrases = {
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
                r'\s+являются$', r'\s+является$',

                # Extended Russian-specific end phrases
                r'\s+получается$', r'\s+получаем$', r'\s+получится$', r'\s+получим$',
                r'\s+получилось$', r'\s+получилась$', r'\s+получились$', r'\s+получилось$',
                r'\s+видим$', r'\s+увидим$', r'\s+видно$', r'\s+заметим$', r'\s+замечаем$',
                r'\s+отметим$', r'\s+отмечаем$', r'\s+узнаем$', r'\s+узнали$', r'\s+узнаём$',
                r'\s+узнал$', r'\s+узнала$', r'\s+узнало$', r'\s+узнали$', r'\s+понимаем$',
                r'\s+поняли$', r'\s+пишем$', r'\s+напишем$', r'\s+записываем$', r'\s+запишем$',
                r'\s+рассматриваем$', r'\s+рассмотрим$', r'\s+обсуждаем$', r'\s+обсудим$',
                r'\s+выясним$', r'\s+выясняем$', r'\s+выяснили$', r'\s+определили$',
                r'\s+определяем$', r'\s+определим$', r'\s+покажем$', r'\s+показываем$',
                r'\s+вычисляем$', r'\s+вычислим$', r'\s+считаем$', r'\s+посчитаем$',
                r'\s+упрощаем$', r'\s+упростим$', r'\s+раскрываем$', r'\s+раскроем$',
                r'\s+сокращаем$', r'\s+сократим$', r'\s+думаем$', r'\s+подумаем$',
                r'\s+решаем$', r'\s+решим$', r'\s+найдём$', r'\s+находим$',
                r'\s+доказываем$', r'\s+докажем$', r'\s+следуем$', r'\s+применяем$',
                r'\s+применим$', r'\s+используем$', r'\s+начинаем$', r'\s+начнём$',
                r'\s+продолжим$', r'\s+продолжаем$', r'\s+перейдём$', r'\s+переходим$'
            ]
        }

        # Get patterns for this language
        lang_key = lang if lang in filler_phrases else 'en'
        patterns = filler_phrases.get(lang_key, [])

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
            normalized = normalized.replace("потом эти", "")
            normalized = normalized.replace("можно убедиться", "")
            normalized = normalized.replace("операторы давайте", "операторы")
            normalized = normalized.replace("давайте посмотрим", "")
            normalized = normalized.replace("давайте напишем", "")
            normalized = normalized.replace("давайте представим", "")
            normalized = normalized.replace("давайте определим", "")
            normalized = normalized.replace("отсюда перепутать", "")
            normalized = normalized.replace("величины букву выбрала", "")
            normalized = normalized.replace("молчите показывайте", "")
            normalized = normalized.replace("думаю имейлы свои", "")
            normalized = normalized.replace("ищутся очевидны", "")
            normalized = normalized.replace("привожу пример", "")
            normalized = normalized.replace("устремляя нулю", "")
            normalized = normalized.replace("менять условия", "")
            normalized = normalized.replace("напутала самое", "")

            # Remove specific bad word combinations
            bad_combinations = [
                "оператор называемое", "оператора именно величине", "квадрат такого",
                "столбик обсуждали", "представлении поскольку", "писать дискретном",
                "попробуем операторы", "возникает неё", "элементы вычислять",
                "представлении непрерывной", "вместо индексов", "поскольку аналогия",
                "выбрали тепер мым", "большинстве случаев", "суммы перейти",
                "функции слева сопряжённое", "неважно любой оператор", "число будут"
            ]

            for bad_combo in bad_combinations:
                if bad_combo in normalized:
                    normalized = normalized.replace(bad_combo, "")

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
        Significantly enhanced to improve concept quality.

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

        # Valid concept typically has 1-5 words
        if word_count < 1 or word_count > 5:
            return False

        # Check if it's mostly numbers
        if sum(c.isdigit() for c in normalized) / len(normalized) > 0.3:
            return False

        # Check against invalid concepts list
        invalid_concepts = self.invalid_concepts.get(lang, set())
        if normalized in invalid_concepts:
            return False

        # Check against invalid patterns
        lang_key = lang if lang in self.invalid_pattern_regexes else 'en'
        for pattern in self.invalid_pattern_regexes[lang_key]:
            if pattern.search(normalized):
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
                invalid_endings = ["ают", "еют", "ить", "ать", "еть", "уть", "еть", "ает", "ует",
                                  "ывать", "ивать", "овать", "евать", "ывал", "ивал", "овал", "евал",
                                  "нный", "тый", "емый", "имый", "ющий", "ящий", "вший", "ший"]
                if any(normalized.endswith(suffix) for suffix in invalid_endings):
                    return False

        # For multi-word concepts, check if at least one word is a domain keyword or has min. length
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

                # Need at least one substantial word (>3 chars) or domain keyword
                substantial_word = False
                for word in words:
                    if word in domain_keywords or len(word) > 3:
                        substantial_word = True
                        break

                if not substantial_word:
                    return False

        return True

    def extract_concepts_from_transcript(
        self,
        processed_transcript: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from a processed transcript using the hybrid approach.
        Uses both global text and individual segments for better concept detection.

        Args:
            processed_transcript: Processed transcript from TranscriptProcessor

        Returns:
            Dictionary containing theoretical and practical concepts
        """
        segments = processed_transcript.get("segments", [])
        language = processed_transcript.get("language", "en")
        domain = processed_transcript.get("domain", "unknown")
        global_analysis = processed_transcript.get("global_analysis", {})

        # Set language for processing
        self.language = language

        # Extract concepts using the hybrid approach
        return self.extract_concepts_from_segments(
            segments,
            domain,
            language,
            global_analysis
        )

    def extract_concepts_from_segments(
        self,
        segments: List[Dict[str, Any]],
        domain: str = "physics",
        language: str = None,
        global_analysis: Dict[str, Any] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from transcript segments with improved validation.
        Uses hybrid global+local approach for better concept extraction.

        Args:
            segments: List of transcript segments
            domain: Content domain
            language: Language code
            global_analysis: Optional global text analysis data

        Returns:
            Dictionary containing theoretical and practical concepts with occurrences
        """
        # Use specified language or default
        lang = language or self.language

        # Track time
        start_time = time.time()

        # First, extract concepts from the full combined transcript text
        # This helps identify concepts that might span segment boundaries
        combined_text = " ".join([segment.get("text", "") for segment in segments])
        combined_concepts = self.extract_concepts(combined_text, domain, lang, global_analysis)
        logger.info(f"Extracted {len(combined_concepts)} initial concepts from combined text")

        # Map each segment text to a unique ID for quick lookups
        segment_map = {segment.get("id", str(uuid.uuid4())): segment for segment in segments}

        # Process segments in batches to reduce memory pressure
        batch_size = 10
        all_segment_concepts = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]

            # Process each segment
            for segment in batch:
                segment_text = segment.get("text", "")
                if len(segment_text) > 30:  # Only process substantial segments
                    segment_concepts = self.extract_concepts(
                        segment_text,
                        domain,
                        lang,
                        global_analysis,
                        # Pass segment educational score to inform concept extraction
                        segment.get("educational_score", 0)
                    )
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

        # Update concept scores based on educational content metrics:
        # 1. Frequency of occurrences (concepts that appear multiple times are more likely educational)
        # 2. Distribution across segments (concepts that appear in multiple segments get higher weight)
        # 3. Duration of segments where concept appears (longer segments = more extensive explanation)
        # 4. Educational scores of containing segments (segments marked as educational boost the concept)
        self._score_educational_occurrences(concept_map, segments)

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

        # Filter concepts with our improved validation
        valid_concepts = []
        for concept in result_concepts:
            concept_text = concept.get("text", "")
            if self.is_valid_concept(concept_text, lang) and len(concept.get("occurrences", [])) > 0:
                valid_concepts.append(concept)
            else:
                logger.debug(f"Filtered out invalid concept: {concept_text}")

        # Separate concepts into theoretical and practical
        theoretical_concepts = [c for c in valid_concepts if c.get("concept_class") == "theoretical"]
        practical_concepts = [c for c in valid_concepts if c.get("concept_class") == "practical"]

        processing_time = time.time() - start_time
        logger.info(f"Concept extraction completed in {processing_time:.2f} seconds")
        logger.info(f"Found {len(valid_concepts)} valid concepts from {len(result_concepts)} candidates")
        logger.info(f"Theoretical concepts: {len(theoretical_concepts)}, Practical concepts: {len(practical_concepts)}")

        # Return dictionary with theoretical and practical concepts
        return {
            "theoretical_concepts": theoretical_concepts,
            "practical_concepts": practical_concepts
        }

    def extract_concepts(
        self,
        text: str,
        domain: str = "physics",
        language: str = None,
        global_analysis: Dict[str, Any] = None,
        educational_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from text with improved multilingual support.
        Uses global analysis information to improve extraction quality.

        Args:
            text: Input text
            domain: Content domain
            language: Language code
            global_analysis: Optional global text analysis
            educational_score: Optional educational score of the segment

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

        # 3. Direct search for important domain terms
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

        # 4. Check for educational content markers
        self._score_educational_content(candidates, text, lang, educational_score, global_analysis)

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

            # Create the concept entry with educational metrics
            filtered_candidates[normalized_text] = {
                "text": term,
                "normalized_text": normalized_text,
                "concept_id": concept_id,
                "frequency": data.get("frequency", 1),
                "score": data.get("score", 0),
                "source": data.get("source", ""),
                "domain": domain,
                "language": lang,
                "theoretical": is_theoretical,
                "concept_class": "theoretical" if is_theoretical else "practical",
                "educational_weight": data.get("educational_weight", 0),
                "is_educational": data.get("is_educational", False)
            }

        # 6. Convert to list and sort by score
        concepts = list(filtered_candidates.values())
        concepts.sort(key=lambda x: x["score"], reverse=True)

        # 7. Limit to top concepts
        max_concepts = 50  # Reduced from 100 to focus on highest quality

        return concepts[:max_concepts]

    def _score_educational_content(
        self,
        candidates: Dict[str, Dict[str, Any]],
        text: str,
        language: str,
        segment_educational_score: float = 0.0,
        global_analysis: Dict[str, Any] = None
    ):
        """
        Score candidate concepts based on educational content markers.
        Uses segment educational score and global analysis to improve accuracy.

        Args:
            candidates: Dictionary of candidate concepts
            text: Source text
            language: Language code
            segment_educational_score: Educational score of the segment
            global_analysis: Global text analysis information
        """
        # Get appropriate language or fallback to English
        lang = language if language in self.educational_markers_regex else 'en'

        # Check for educational content markers in the text
        has_educational_markers = bool(self.educational_markers_regex[lang].search(text))

        # Add educational weight to concepts
        for term, data in candidates.items():
            # Base educational weight - higher for all concepts in educational contexts
            educational_weight = 2.0 if has_educational_markers else 0.0

            # Add segment's educational score as a factor
            educational_weight += segment_educational_score * 0.5

            # Check frequency/repetition - concepts repeated multiple times are more likely educational
            if data["frequency"] > 1:
                educational_weight += min(data["frequency"], 5) * 0.5

            # Check for concept appearing in proximity to educational markers
            text_lower = text.lower()
            term_lower = term.lower()

            # See if term is near educational markers (within ~100 chars)
            markers = self.educational_markers[lang]
            for marker in markers:
                marker_pos = text_lower.find(marker)
                if marker_pos >= 0:
                    term_pos = text_lower.find(term_lower)
                    if term_pos >= 0 and abs(term_pos - marker_pos) < 100:
                        educational_weight += 1.5
                        break

            # If global analysis is provided, check if this concept appears in key terms
            if global_analysis and "key_terms" in global_analysis:
                if term_lower in global_analysis["key_terms"]:
                    educational_weight += 1.0

            # Add educational weight to the concept's score
            data["educational_weight"] = educational_weight
            data["score"] += educational_weight

            # Flag as educational if weight exceeds threshold
            data["is_educational"] = educational_weight > 2.5

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
                    r'(унитарн[а-я]+) (оператор[а-я]*|преобразовани[а-я]*)',
                    r'(оператор[а-я]*) (энерги[а-я]+|импульс[а-я]+|координат[а-я]+)',
                    r'(принцип) (суперпозици[а-я]+)',
                    r'(стационарн[а-я]+) (состояни[а-я]+)'
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
                    r'(функциональн[а-я]+) (пространств[а-я]+|теори[а-я]+|анализ[а-я]+)',
                    r'(интеграл[а-я]*) (фурье|лебега|римана)',
                    r'(ряд[а-я]*) (фурье|тейлора|лорана)',
                    r'(непрерывн[а-я]+) (функци[а-я]+)',
                    r'(дифференцируем[а-я]+) (функци[а-я]+)',
                    r'(комплексн[а-я]+) (перемен[а-я]+)'
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
                "коэффициент разложения", "нормировка функции", "гармонический осциллятор",
                "квантовая теория", "спиновая система", "спин-орбитальное взаимодействие",
                "атомный спектр", "молекулярная орбиталь", "базисный вектор",
                "функция распределения", "вероятностная интерпретация", "принцип суперпозиции",
                "уравнение непрерывности", "соотношение неопределенностей", "граничные условия",
                "начальные условия", "квантовый эффект", "пространство состояний",
                "тензорное произведение", "прямое произведение", "комплексное сопряжение",
                "импульсное представление", "координатное представление"
            ]

            # Search for phrases
            for phrase in quantum_phrases:
                count = text.lower().count(phrase)
                if count > 0 and self.is_valid_concept(phrase, language):
                    matches[phrase] = matches.get(phrase, 0) + count * 2  # Higher weight for phrases

        return matches

    def _extract_significant_bigrams(self, text: str, language: str = "en") -> Dict[str, float]:
        """
        Extract significant bigrams from text with better filtering.

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

            # Only keep valid bigrams
            if self.is_valid_concept(bigram_text, language):
                bigram_scores[bigram_text] = score

        return bigram_scores

    def _extract_significant_trigrams(self, text: str, language: str = "en") -> Dict[str, float]:
        """
        Extract significant trigrams from text with better filtering.

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

    def _score_educational_occurrences(self, concept_map: Dict[str, Dict[str, Any]], segments: List[Dict] = None):
        """
        Score concepts based on their occurrences to distinguish educational content from passing mentions.
        Uses segment educational scores to improve accuracy.

        Args:
            concept_map: Dictionary mapping concept_id to concept data
            segments: List of transcript segments with educational scores
        """
        # Create segment map for efficient lookup if segments provided
        segment_map = {}
        if segments:
            segment_map = {seg["id"]: seg for seg in segments if "id" in seg}

        for concept_id, concept in concept_map.items():
            occurrences = concept.get("occurrences", [])

            if not occurrences:
                continue

            # Calculate educational metrics
            frequency = len(occurrences)
            unique_segments = len(set(occ.get("segment_id") for occ in occurrences))
            total_duration = sum(occ.get("end_time", 0) - occ.get("start_time", 0) for occ in occurrences)

            # Metrics for educational vs passing mention:
            # 1. Frequency bonus - concepts mentioned multiple times
            frequency_factor = min(frequency, 5) * 0.5

            # 2. Segment distribution - concepts in multiple segments
            segment_factor = min(unique_segments, 3) * 0.7

            # 3. Duration bonus - longer total discussion time
            duration_factor = min(total_duration / 10.0, 3.0)

            # 4. Use educational scores from segments if available
            segments_educational_score = 0.0
            if segment_map:
                # Get educational scores from segments containing this concept
                segment_scores = []
                for occ in occurrences:
                    segment_id = occ.get("segment_id")
                    if segment_id in segment_map:
                        segment_scores.append(segment_map[segment_id].get("educational_score", 0.0))

                # Calculate average educational score across segments
                if segment_scores:
                    segments_educational_score = sum(segment_scores) / len(segment_scores)

            # Calculate educational score
            educational_score = frequency_factor + segment_factor + duration_factor + segments_educational_score

            # Add educational weight to concept score
            concept["educational_weight"] = educational_score
            concept["score"] += educational_score

            # Mark concepts with high educational weight
            concept["is_educational"] = educational_score > 2.5

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
