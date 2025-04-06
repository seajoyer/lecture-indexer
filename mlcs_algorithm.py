"""
Enhanced Linear Multiple Longest Common Subsequence (MLCS) algorithm implementation.
Optimized for both character-level and token-level comparison, efficiently handling
both concept similarity detection and concept signature generation.
"""

import re
import logging
from typing import List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)

class MLCSAlgorithm:
    """
    Efficient Linear Multiple Longest Common Subsequence algorithm implementation.

    This implementation provides both character-level and token-level MLCS algorithms
    for different use cases - character-level for concept deduplication and
    token-level for concept signature extraction.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the MLCS algorithm.

        Args:
            language: Language code ('en' or 'ru')
        """
        self.language = language
        self._load_language_resources()

    def _load_language_resources(self):
        """Load language-specific resources for preprocessing."""
        # Stopwords by language - words to filter out
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
                'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
            },
            'ru': {
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
                "снова", "никогда", "всегда", "часто", "иногда", "редко"
            }
        }

        # Domain-specific keywords to keep during preprocessing
        self.domain_keywords = {
            "physics": {
                "en": [
                    "quantum", "mechanics", "wave", "function", "operator", "state",
                    "eigenvalue", "eigenstate", "hamiltonian", "commutator", "hermitian",
                    "observable", "measurement", "probability", "amplitude", "schrodinger",
                    "dirac", "bra", "ket", "hilbert", "space", "vector", "momentum", "energy",
                    "position", "uncertainty", "principle", "entanglement", "superposition"
                ],
                "ru": [
                    "квантовый", "квантовая", "квантовое", "квантовые", "механика",
                    "волновая", "функция", "оператор", "состояние", "собственное",
                    "значение", "собственный", "вектор", "гамильтониан", "коммутатор",
                    "эрмитов", "эрмитово", "эрмитова", "эрмитовый", "наблюдаемая",
                    "измерение", "вероятность", "амплитуда", "шредингер", "дирак",
                    "бра", "кет", "гильбертово", "пространство", "вектор", "импульс",
                    "энергия", "положение", "неопределенность", "принцип", "запутанность",
                    "суперпозиция", "матрица", "плотности", "чистое", "смешанное"
                ]
            },
            "mathematics": {
                "en": ["function", "derivative", "integral", "differential", "equation"],
                "ru": ["функция", "производная", "интеграл", "дифференциал", "уравнение"]
            }
        }

    def normalize_token(self, token: str, language: Optional[str] = None) -> str:
        """
        Normalize a token by lowercasing and removing non-alphanumeric characters.

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

        # Remove non-alphanumeric characters except for hyphens
        normalized = re.sub(r'[^\w\-]', '', normalized)

        return normalized

    def preprocess_text(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Preprocess text by tokenizing, removing stopwords, and normalizing tokens.
        Used for token-level MLCS in concept signature extraction.

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

        # Get domain keywords for physics (we're focusing on this domain)
        physics_keywords = self.domain_keywords.get("physics", {}).get(lang, [])
        if not physics_keywords:
            # Fallback to English keywords
            physics_keywords = self.domain_keywords.get("physics", {}).get("en", [])

        # Mathematics keywords
        math_keywords = self.domain_keywords.get("mathematics", {}).get(lang, [])
        if not math_keywords:
            math_keywords = self.domain_keywords.get("mathematics", {}).get("en", [])

        # Combine all domain keywords
        domain_keywords = set(physics_keywords + math_keywords)

        # Get stopwords for this language
        stop_words = self.stopwords.get(lang, set())
        if not stop_words:
            # Fallback to English stopwords
            stop_words = self.stopwords.get("en", set())

        # Tokenize text
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
        Works with both character and token sequences.

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
        Optimized MLCS for token-level sequences with more sophisticated matching.
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
                common_ngrams.append((ngram, score))

        # Sort by score (higher score first)
        common_ngrams.sort(key=lambda x: x[1], reverse=True)

        # Return the highest scoring n-gram if any
        return list(common_ngrams[0][0]) if common_ngrams else []

    def extract_concept_signature(
        self,
        concept_text: str,
        contexts: List[str],
        language: Optional[str] = None
    ) -> Tuple[List[str], float]:
        """
        Extract a concept signature from its context occurrences.

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
        Extract significant common patterns from multiple texts.

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
                # Score based on length, frequency, and domain term count
                # Get domain keywords for this language
                physics_keywords = self.domain_keywords.get("physics", {}).get(lang, [])
                if not physics_keywords:
                    physics_keywords = self.domain_keywords.get("physics", {}).get("en", [])

                # Count domain terms in the n-gram
                domain_term_count = sum(1 for term in ngram if term in physics_keywords)

                # Adjust score calculation to favor domain terms and longer patterns
                domain_term_bonus = domain_term_count * 0.5
                length_bonus = len(ngram) * 0.3
                frequency_factor = count / len(texts)

                # Calculate final score
                score = length_bonus + (frequency_factor * 2.0) + domain_term_bonus

                significant_ngrams.append((list(ngram), score))

        # Sort by score
        significant_ngrams.sort(key=lambda x: x[1], reverse=True)

        return significant_ngrams
