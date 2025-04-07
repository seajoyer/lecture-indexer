"""
Enhanced Linear Multiple Longest Common Subsequence (MLCS) algorithm implementation.
Optimized for both character-level and token-level comparison, efficiently handling
both concept similarity detection and concept signature generation with improved
language-specific processing, especially for Russian content.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Set

# Configure logging
logger = logging.getLogger(__name__)

class MLCSAlgorithm:
    """
    Efficient Linear Multiple Longest Common Subsequence algorithm implementation.

    This implementation provides both character-level and token-level MLCS algorithms
    with enhanced language-specific processing for concept detection and signature extraction.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the MLCS algorithm with language-specific resources.

        Args:
            language: Language code ('en' or 'ru')
        """
        self.language = language
        self._load_language_resources()

    def _load_language_resources(self):
        """Load enhanced language-specific resources for preprocessing."""
        # Enhanced stopwords by language - words to filter out
        self.stopwords = {
            'en': {
                'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should',
                'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'also', 'actually', 'like', 'basically', 'obviously', 'simply', 'certainly',
                'definitely', 'really', 'probably', 'possibly', 'perhaps', 'indeed', 'furthermore',
                'moreover', 'however', 'nevertheless', 'nonetheless', 'therefore', 'thus',
                'meanwhile', 'subsequently', 'consequently', 'alternatively', 'similarly',
                'likewise', 'accordingly', 'hence', 'besides', 'anyway', 'actually', 'incidentally',
                'by the way', 'in fact', 'as a matter of fact', 'in any case', 'in either case',
                'in both cases', 'either way', 'otherwise', 'rather', 'instead', 'conversely',
                'in contrast', 'on the contrary', 'on the other hand', 'at any rate', 'in any event',
                'in conclusion', 'to conclude', 'to summarize', 'to sum up', 'finally'
            },
            'ru': {
                # Core Russian stopwords
                "это", "вот", "так", "как", "ну", "да", "нет", "просто",
                "значит", "сейчас", "здесь", "тут", "уже", "если", "все", "всё",
                "хорошо", "там", "кстати", "итак", "будет", "ещё", "еще",
                "нас", "меня", "можно", "они", "только", "для", "поэтому", "равно",
                "нужно", "получается", "означает", "должна", "вами", "можем",
                "какой-то", "что-то", "стоит", "хочу", "буду", "видим", "понятно",
                "сделать", "например", "должны", "какие-то", "сюда", "плюс", "минус",
                "будем", "результат", "такое", "и", "или", "но", "а", "в", "на", "при",
                "с", "со", "к", "от", "из", "по", "о", "об", "за", "под", "над", "через",
                "между", "перед", "после", "около", "у", "возле", "то", "ты", "он", "она",
                "оно", "мы", "вы", "они", "мой", "твой", "его", "её", "наш", "ваш", "их",
                "тот", "этот", "тем", "тогда", "себя", "свой", "весь", "каждый", "любой",
                "другой", "иной", "всякий", "который", "чей", "самый", "что", "кто", "где",
                "когда", "почему", "зачем", "оттуда", "туда", "там", "здесь", "теперь",
                "потом", "затем", "впоследствии", "сначала", "сперва", "вначале", "прежде",
                "давайте", "рассмотрим", "посмотрим", "будем", "далее", "было", "быть",
                "есть", "суть", "именно", "лишь", "даже", "ведь", "ещё", "уже", "опять",
                "снова", "никогда", "всегда", "часто", "иногда", "редко",

                # Extended Russian lecture stopwords - important for Russian physics lectures
                "то есть", "так сказать", "вот так", "так вот", "как бы", "в общем",
                "короче", "собственно", "фактически", "практически", "видите ли",
                "знаете ли", "понимаете", "представляете", "допустим", "предположим",
                "пускай", "дальше", "имеется", "получаем", "видим", "замечаем",
                "смотрим", "рассматриваем", "обсуждаем", "обсудим", "рассмотрим",
                "остановимся", "вернемся", "перейдем", "продолжим", "начнем",
                "закончим", "заканчиваем", "начинаем", "следует", "надо", "необходимо",
                "нужно", "хочется", "можем", "должны", "хотим", "будем", "попробуем",
                "пробуем", "пытаемся", "хотелось бы", "допустим", "предположим",
                "представим", "например", "положим", "обозначим", "тогда", "потом",
                "затем", "следовательно", "таким образом", "значит", "поэтому",
                "соответственно", "итак", "наконец", "кстати", "между прочим", "кстати",
                "кроме того", "также", "причем", "заметим", "отметим", "напомним",
                "запомним", "подчеркнем", "выделим", "скажем", "говорим", "помним",
                "знаем", "помните", "знаете", "думаю", "считаю", "полагаю", "на мой взгляд",
                "собственно говоря", "честно говоря", "образно говоря"
            }
        }

        # Domain-specific keywords to keep during preprocessing - expanded for each domain
        self.domain_keywords = {
            "physics": {
                "en": [
                    # Core quantum physics terms
                    "quantum", "mechanics", "wave", "function", "operator", "state",
                    "eigenvalue", "eigenstate", "hamiltonian", "commutator", "hermitian",
                    "observable", "measurement", "probability", "amplitude", "schrodinger",
                    "dirac", "bra", "ket", "hilbert", "space", "vector", "momentum", "energy",
                    "position", "uncertainty", "principle", "entanglement", "superposition",

                    # Extended physics terminology
                    "fermion", "boson", "photon", "electron", "proton", "neutron",
                    "spin", "charge", "field", "potential", "barrier", "well",
                    "particle", "wave", "duality", "interference", "diffraction",
                    "quantization", "discrete", "continuous", "spectrum", "matrix",
                    "tensor", "eigenfunction", "ground", "excited", "stationary",
                    "time-dependent", "time-independent", "perturbation", "vacuum",
                    "vacuum", "density", "angular", "linear", "orbital", "interaction",
                    "coupling", "coherence", "decoherence", "collapse", "emission",
                    "absorption", "tunneling", "radiation", "nucleus", "atomic", "molecular"
                ],
                "ru": [
                    # Core quantum physics terms in Russian
                    "квантовый", "квантовая", "квантовое", "квантовые", "квантовость",
                    "механика", "волновая", "функция", "оператор", "состояние",
                    "собственное", "значение", "собственный", "вектор", "собственная",
                    "гамильтониан", "коммутатор", "эрмитов", "эрмитово", "эрмитова",
                    "наблюдаемая", "измерение", "вероятность", "амплитуда", "шредингер",
                    "дирак", "бра", "кет", "гильбертово", "пространство",
                    "импульс", "энергия", "положение", "координата", "координаты",
                    "неопределенность", "принцип", "запутанность", "суперпозиция",

                    # Extended physics terminology in Russian
                    "фермион", "бозон", "фотон", "электрон", "протон", "нейтрон",
                    "спин", "заряд", "поле", "потенциал", "барьер", "яма",
                    "частица", "волна", "дуализм", "интерференция", "дифракция",
                    "квантование", "дискретный", "непрерывный", "спектр", "матрица",
                    "тензор", "собственная функция", "основное", "возбужденное", "стационарное",
                    "зависящий от времени", "не зависящий от времени", "возмущение", "вакуум",
                    "плотность", "угловой", "линейный", "орбитальный", "взаимодействие",
                    "связь", "когерентность", "декогеренция", "коллапс", "излучение",
                    "поглощение", "туннелирование", "радиация", "ядро", "атомный", "молекулярный",

                    # Specific Russian physics terms
                    "волновой пакет", "операторы рождения", "операторы уничтожения",
                    "эрмитово сопряженный", "эрмитово сопряженная", "перестановочный",
                    "гильбертово пространство", "матрица плотности", "унитарный",
                    "унитарная", "унитарное", "нормировка", "нормировочный",
                    "томографическое", "распределение", "матричный", "векторный",
                    "спектральный", "уровень", "уровни", "энергетический", "сферический"
                ]
            },
            "mathematics": {
                "en": [
                    "function", "derivative", "integral", "differential", "equation",
                    "theorem", "lemma", "proof", "corollary", "proposition", "axiom",
                    "definition", "variable", "constant", "expression", "formula",
                    "identity", "inequality", "transformation", "mapping", "morphism",
                    "isomorphism", "homomorphism", "bijection", "surjection", "injection",
                    "domain", "codomain", "range", "image", "kernel", "vector", "scalar",
                    "tensor", "matrix", "determinant", "trace", "eigenvalue", "eigenvector",
                    "basis", "dimension", "topology", "continuous", "limit", "convergence",
                    "divergence", "sequence", "series", "summation", "product", "induction",
                    "recursion", "recurrence", "algorithm", "computation", "geometry",
                    "trigonometry", "angle", "triangle", "circle", "sphere", "polynomial",
                    "monomial", "binomial", "exponential", "logarithm", "asymptotic",
                    "approximation", "error", "prime", "factorization", "divisor", "multiple",
                    "greatest", "least", "remainder", "quotient", "factor", "fraction",
                    "decimal", "rational", "irrational", "real", "complex", "imaginary",
                    "conjugate", "magnitude", "argument", "sin", "cos", "tan", "arcsin",
                    "arccos", "arctan", "sinh", "cosh", "tanh", "absolute", "maximum",
                    "minimum", "critical", "stationary", "inflection", "concave", "convex"
                ],
                "ru": [
                    "функция", "производная", "интеграл", "дифференциал", "уравнение",
                    "теорема", "лемма", "доказательство", "следствие", "предложение", "аксиома",
                    "определение", "переменная", "постоянная", "выражение", "формула",
                    "тождество", "неравенство", "преобразование", "отображение", "морфизм",
                    "изоморфизм", "гомоморфизм", "биекция", "сюръекция", "инъекция",
                    "область определения", "область значений", "образ", "ядро", "вектор", "скаляр",
                    "тензор", "матрица", "определитель", "след", "собственное значение", "собственный вектор",
                    "базис", "размерность", "топология", "непрерывный", "предел", "сходимость",
                    "расходимость", "последовательность", "ряд", "сумма", "произведение", "индукция",
                    "рекурсия", "рекуррентность", "алгоритм", "вычисление", "геометрия",
                    "тригонометрия", "угол", "треугольник", "круг", "сфера", "многочлен",
                    "одночлен", "бином", "экспонента", "логарифм", "асимптотика",
                    "приближение", "погрешность", "простое число", "факторизация", "делитель", "кратное",
                    "наибольший", "наименьший", "остаток", "частное", "множитель", "дробь",
                    "десятичный", "рациональный", "иррациональный", "действительный", "комплексный", "мнимый",
                    "сопряженный", "модуль", "аргумент", "синус", "косинус", "тангенс", "арксинус",
                    "арккосинус", "арктангенс", "гиперболический синус", "гиперболический косинус",
                    "гиперболический тангенс", "абсолютный", "максимум", "минимум", "критический",
                    "стационарный", "перегиб", "вогнутый", "выпуклый"
                ]
            }
        }

    def normalize_token(self, token: str, language: Optional[str] = None) -> str:
        """
        Normalize a token using language-specific rules.

        Args:
            token: Token to normalize
            language: Optional language code

        Returns:
            Normalized token
        """
        # Use provided language or default
        lang = language or self.language

        # Lowercase the token
        normalized = token.lower()

        # If Russian, apply Russian-specific normalizations
        if lang == 'ru':
            # Replace 'ё' with 'е' (common in Russian text normalization)
            normalized = normalized.replace('ё', 'е')

            # Remove Russian soft sign (ь) and hard sign (ъ) at the end of words
            # These don't change the meaning but are grammatical markers
            normalized = re.sub(r'[ьъ]$', '', normalized)

            # Normalize common variant endings for adjectives
            normalized = re.sub(r'(ого|его)$', 'ый', normalized)  # masculine genitive to nominative
            normalized = re.sub(r'(ому|ему)$', 'ый', normalized)  # masculine dative to nominative
            normalized = re.sub(r'(ую|юю)$', 'ая', normalized)    # feminine accusative to nominative

            # Normalize common case endings for nouns
            normalized = re.sub(r'(ом|ем|ам|ям|ей|ов|ев|ьев)$', '', normalized)  # plural forms

            # Special case for Russian plurals
            normalized = re.sub(r'и$', '', normalized) if len(normalized) > 4 else normalized

        else:  # English and other languages
            # Remove common English suffixes for normalization
            if len(normalized) > 3:
                # Plurals and verb forms
                if normalized.endswith('s') and not normalized.endswith('ss'):
                    normalized = normalized[:-1]
                elif normalized.endswith('es'):
                    normalized = normalized[:-2]
                elif normalized.endswith('ing'):
                    normalized = normalized[:-3]
                elif normalized.endswith('ed') and len(normalized) > 4:
                    normalized = normalized[:-2]

        # Remove non-alphanumeric characters except for hyphens
        normalized = re.sub(r'[^\w\-]', '', normalized)

        return normalized

    def preprocess_text(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Preprocess text by tokenizing, removing stopwords, and normalizing tokens.
        Enhanced with language-specific processing, especially for Russian.

        Args:
            text: Input text
            language: Optional language code

        Returns:
            List of preprocessed tokens
        """
        if not text:
            return []

        # Use provided language or default
        lang = language or self.language

        # Get all domain-specific keywords for better filtering decisions
        domain_keywords = set()
        for domain, lang_keywords in self.domain_keywords.items():
            domain_keywords.update(lang_keywords.get(lang, set()))
            # Fallback to English keywords if the language isn't available
            if not lang_keywords.get(lang):
                domain_keywords.update(lang_keywords.get('en', set()))

        # Get stopwords for this language
        stop_words = self.stopwords.get(lang, set())
        if not stop_words:
            # Fallback to English stopwords
            stop_words = self.stopwords.get('en', set())

        # Apply language-specific tokenization
        if lang == 'ru':
            # Russian tokenization with specific patterns
            tokens = re.findall(r'[\w\-]+', text.lower())
        else:
            # Default tokenization
            tokens = re.findall(r'\b[\w\-\']+\b', text.lower())

        # Filter out stopwords and short tokens, but keep domain keywords
        preprocessed = []
        for token in tokens:
            normalized = self.normalize_token(token, lang)

            # Skip empty tokens
            if not normalized:
                continue

            # Always keep domain keywords
            if normalized in domain_keywords:
                preprocessed.append(normalized)
                continue

            # Skip stopwords and very short tokens
            if normalized in stop_words or len(normalized) <= 2:
                continue

            # Skip tokens that are just numbers
            if normalized.isdigit():
                continue

            preprocessed.append(normalized)

        return preprocessed

    def find_mlcs(self, sequences: List[List[str]], min_length: int = 2) -> List[str]:
        """
        Find the Multiple Longest Common Subsequence across sequences.
        Works with both character and token sequences.

        Args:
            sequences: List of token/character sequences
            min_length: Minimum length of common subsequence

        Returns:
            MLCS as a list of tokens/characters
        """
        if not sequences:
            return []

        if len(sequences) == 1:
            return sequences[0]

        # For two sequences, use the efficient LCS algorithm
        if len(sequences) == 2:
            lcs = self._lcs(sequences[0], sequences[1])
            return lcs if len(lcs) >= min_length else []

        # For more than two sequences, use optimized approach
        return self._find_mlcs_linear(sequences, min_length)

    def _lcs(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """
        Find the Longest Common Subsequence between two sequences.
        Optimized implementation for both character and token sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            LCS as a list of tokens/characters
        """
        # Optimization: Empty sequence check
        if not seq1 or not seq2:
            return []

        # Optimization: If sequences are very long (e.g., character sequences)
        # use more memory-efficient dynamic programming approach
        if len(seq1) > 200 or len(seq2) > 200:
            return self._lcs_efficient(seq1, seq2)

        # Standard LCS dynamic programming for moderate-sized sequences
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Reconstruct the LCS
        lcs = []
        i, j = m, n

        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                lcs.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        # Reverse the LCS (since we built it backwards)
        lcs.reverse()

        return lcs

    def _lcs_efficient(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """
        Memory-efficient LCS implementation for very long sequences.
        Uses space optimization to reduce memory consumption.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            LCS as a list of tokens/characters
        """
        # Ensure seq1 is the shorter sequence for efficiency
        if len(seq1) > len(seq2):
            seq1, seq2 = seq2, seq1

        m, n = len(seq1), len(seq2)

        # Use two rows instead of full matrix
        current = [0] * (n + 1)
        previous = [0] * (n + 1)

        # Track the choices made for reconstruction
        choices = {}  # (i, j) -> direction (diagonal, up, left)

        # Fill the dp table with just two rows
        for i in range(1, m + 1):
            previous, current = current, [0] * (n + 1)

            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    current[j] = previous[j-1] + 1
                    choices[(i, j)] = 'diagonal'
                elif previous[j] >= current[j-1]:
                    current[j] = previous[j]
                    choices[(i, j)] = 'up'
                else:
                    current[j] = current[j-1]
                    choices[(i, j)] = 'left'

        # Reconstruct the LCS
        lcs = []
        i, j = m, n

        while i > 0 and j > 0:
            direction = choices.get((i, j))

            if direction == 'diagonal':
                lcs.append(seq1[i-1])
                i -= 1
                j -= 1
            elif direction == 'up':
                i -= 1
            else:
                j -= 1

        # Reverse the LCS
        lcs.reverse()

        return lcs

    def _find_mlcs_linear(self, sequences: List[List[str]], min_length: int = 2) -> List[str]:
        """
        Find MLCS using a linear approach optimized for academic terms.
        Enhanced for multilingual support.

        Args:
            sequences: List of token/character sequences
            min_length: Minimum length of common subsequence

        Returns:
            MLCS as a list of tokens/characters
        """
        # Optimization for character-level sequences
        is_char_level = all(isinstance(seq[0], str) and len(seq[0]) == 1 for seq in sequences if seq)

        # For character-level comparison (typical in concept deduplication)
        if is_char_level:
            return self._find_mlcs_character(sequences, min_length)

        # For token-level comparison (typical in concept signature extraction)
        return self._find_mlcs_token(sequences, min_length)

    def _find_mlcs_character(self, sequences: List[List[str]], min_length: int = 2) -> List[str]:
        """
        Optimized MLCS for character-level sequences.
        Used primarily for concept deduplication.

        Args:
            sequences: List of character sequences
            min_length: Minimum length of common subsequence

        Returns:
            MLCS as a list of characters
        """
        if not sequences:
            return []

        # Progressively find common subsequence
        current_lcs = sequences[0]

        for i in range(1, len(sequences)):
            current_lcs = self._lcs(current_lcs, sequences[i])

            # Early termination if LCS becomes too short
            if len(current_lcs) < min_length:
                return []

        return current_lcs if len(current_lcs) >= min_length else []

    def _find_mlcs_token(self, sequences: List[List[str]], min_length: int = 2) -> List[str]:
        """
        Optimized MLCS for token-level sequences with enhanced language handling.
        Used primarily for concept signature extraction.

        Args:
            sequences: List of token sequences
            min_length: Minimum length of common subsequence

        Returns:
            MLCS as a list of tokens
        """
        # Calculate the frequency of each token in all sequences
        all_tokens = set()
        for seq in sequences:
            all_tokens.update(seq)

        # Track token positions in each sequence
        token_positions = {token: [] for token in all_tokens}

        for seq_idx, seq in enumerate(sequences):
            for pos, token in enumerate(seq):
                token_positions[token].append((seq_idx, pos))

        # Find tokens that appear in all sequences
        common_tokens = [token for token, positions in token_positions.items()
                        if len(set(pos[0] for pos in positions)) == len(sequences)]

        if not common_tokens:
            # If no common tokens across all sequences, try a more relaxed approach
            # Find tokens that appear in at least half of the sequences
            min_seq_count = max(2, len(sequences) // 2)
            common_tokens = [token for token, positions in token_positions.items()
                            if len(set(pos[0] for pos in positions)) >= min_seq_count]

            if not common_tokens:
                return []

        # Extract n-grams from each sequence
        ngrams = {}

        for n in range(min_length, min(10, max(len(seq) for seq in sequences)) + 1):
            for seq_idx, seq in enumerate(sequences):
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i+n])
                    if ngram not in ngrams:
                        ngrams[ngram] = []
                    ngrams[ngram].append(seq_idx)

        # Find n-grams that appear in at least half of the sequences
        min_seq_count = max(2, len(sequences) // 2)
        common_ngrams = []

        for ngram, seq_indices in ngrams.items():
            if len(set(seq_indices)) >= min_seq_count:
                # Score the n-gram by length and number of sequences it appears in
                score = len(ngram) * len(set(seq_indices)) / len(sequences)

                # Boost score for domain-specific terms in the ngram
                domain_term_count = sum(1 for token in ngram if self._is_domain_term(token))
                if domain_term_count > 0:
                    score *= (1 + 0.2 * domain_term_count)

                common_ngrams.append((ngram, score))

        # Sort by score (higher score first)
        common_ngrams.sort(key=lambda x: x[1], reverse=True)

        # Return the highest scoring n-gram if any
        return list(common_ngrams[0][0]) if common_ngrams else []

    def _is_domain_term(self, token: str) -> bool:
        """
        Check if a token is a domain-specific term based on loaded resources.

        Args:
            token: Token to check

        Returns:
            True if it's a domain term, False otherwise
        """
        # Check all domains for this token
        for domain, lang_keywords in self.domain_keywords.items():
            if token in lang_keywords.get(self.language, set()):
                return True
            # Try English as fallback
            if self.language != 'en' and token in lang_keywords.get('en', set()):
                return True
        return False

    def extract_concept_signature(
        self,
        concept_text: str,
        contexts: List[str],
        language: Optional[str] = None
    ) -> Tuple[List[str], float]:
        """
        Extract a concept signature from its context occurrences with enhanced language processing.

        Args:
            concept_text: Concept text
            contexts: List of context texts where the concept appears
            language: Optional language code

        Returns:
            Tuple of (signature_pattern, confidence)
        """
        # Use provided language or default
        lang = language or self.language

        # If no contexts, use the concept text itself
        if not contexts:
            preprocessed = self.preprocess_text(concept_text, lang)
            return preprocessed, 0.5

        # Extract significant sequences from contexts
        significant_sequences = self.find_significant_patterns(
            contexts, min_length=2, min_frequency=max(2, len(contexts) // 3), language=lang
        )

        # If we found significant sequences
        if significant_sequences:
            # Use the highest scoring sequence as the signature pattern
            signature_pattern, score = significant_sequences[0]

            # Ensure the extracted signature actually relates to the concept
            concept_tokens = self.preprocess_text(concept_text, lang)

            # Check if there's overlap between signature pattern and concept tokens
            overlap = set(signature_pattern).intersection(set(concept_tokens))

            if overlap or len(signature_pattern) <= 2:
                # Calculate confidence based on score
                confidence = min(score / 10.0, 0.95)  # Normalize confidence
                return signature_pattern, confidence

            # If no overlap, try the next highest scoring sequence
            if len(significant_sequences) > 1:
                signature_pattern, score = significant_sequences[1]
                confidence = min(score / 10.0, 0.9)  # Slightly lower confidence
                return signature_pattern, confidence

        # If no significant sequences found or no good match, use the preprocessed concept text
        preprocessed = self.preprocess_text(concept_text, lang)
        return preprocessed, 0.5

    def find_significant_patterns(
        self,
        texts: List[str],
        min_length: int = 2,
        min_frequency: int = 2,
        language: Optional[str] = None
    ) -> List[Tuple[List[str], float]]:
        """
        Extract significant common patterns from multiple texts with enhanced language processing.

        Args:
            texts: List of text strings
            min_length: Minimum pattern length
            min_frequency: Minimum pattern frequency
            language: Optional language code

        Returns:
            List of (pattern, score) tuples
        """
        # Use provided language or default
        lang = language or self.language

        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text, lang) for text in texts]

        # Filter out empty or very short texts
        preprocessed_texts = [tokens for tokens in preprocessed_texts if len(tokens) >= min_length]

        if not preprocessed_texts:
            return []

        # Find common n-grams across texts
        ngrams = {}

        for tokens in preprocessed_texts:
            text_ngrams = set()  # Use set to avoid counting duplicates within the same text

            for n in range(min_length, min(10, len(tokens)) + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    text_ngrams.add(ngram)

            # Count each unique n-gram once per text
            for ngram in text_ngrams:
                ngrams[ngram] = ngrams.get(ngram, 0) + 1

        # Filter by frequency and sort by score
        significant_ngrams = []

        for ngram, count in ngrams.items():
            if count >= min_frequency:
                # Calculate base score based on length and frequency
                length_weight = len(ngram) * 0.3
                frequency_weight = (count / len(texts)) * 2.0

                # Count domain terms in the n-gram
                domain_term_count = sum(1 for term in ngram if self._is_domain_term(term))
                domain_weight = domain_term_count * 0.5

                # Language-specific scoring adjustments
                if lang == 'ru':
                    # For Russian, give higher weights to multi-word terms
                    # that correspond to important physics concepts
                    if len(ngram) >= 2:
                        # Check for important Russian physics bigrams/trigrams
                        term = " ".join(ngram)
                        important_terms = [
                            "волновая функция", "квантовая механика", "собственное значение",
                            "собственное состояние", "гильбертово пространство", "принцип неопределенности",
                            "оператор энергии", "оператор импульса", "оператор координаты",
                            "эрмитов оператор", "унитарное преобразование", "стационарное состояние",
                            "квантовая теория", "вакуумное состояние", "матрица плотности",
                            "квантовый осциллятор", "уравнение шредингера"
                        ]

                        for important in important_terms:
                            if important in term:
                                domain_weight += 1.0  # Significant boost

                # Calculate final score
                score = length_weight + frequency_weight + domain_weight

                significant_ngrams.append((list(ngram), score))

        # Sort by score
        significant_ngrams.sort(key=lambda x: x[1], reverse=True)

        return significant_ngrams
