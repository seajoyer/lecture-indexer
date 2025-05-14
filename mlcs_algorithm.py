"""
MLCS Algorithm Python wrapper for the C++ implementation.
This module provides a seamless interface with the existing codebase by
wrapping the C++ implementation with a compatible Python interface.
"""

import logging
from typing import List, Tuple, Set, Dict, Any, Optional, Union
import re

# Import C++ module
try:
    from mlcs_cpp import MLCSAlgorithm as MLCSAlgorithmCpp
    CPP_IMPLEMENTATION = True
except ImportError:
    CPP_IMPLEMENTATION = False
    logging.warning("C++ implementation not available - using Python fallback")

# Configure logging
logger = logging.getLogger(__name__)

class MLCSAlgorithm:
    """
    Efficient Linear Multiple Longest Common Subsequence algorithm implementation.

    This implementation is a wrapper around the optimized C++ implementation,
    with a fallback to the original Python implementation if the C++ module
    is not available.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the MLCS algorithm with language-specific resources.

        Args:
            language: Language code ('en' or 'ru')
        """
        self.language = language

        if CPP_IMPLEMENTATION:
            # Use C++ implementation
            self._cpp_impl = MLCSAlgorithmCpp(language)
            logger.info(f"Using C++ MLCS implementation for language: {language}")
        else:
            # Fallback to original Python implementation
            self._load_language_resources()
            logger.info(f"Using Python MLCS implementation for language: {language}")

    def _load_language_resources(self):
        """Load language-specific resources for preprocessing."""
        # This function is only used in the Python fallback implementation
        # For the C++ implementation, resources are loaded in the C++ constructor

        # Import the original Python implementation if needed
        from mlcs_algorithm import MLCSAlgorithm as MLCSAlgorithmPy

        # Create an instance to access its resources
        py_impl = MLCSAlgorithmPy(self.language)

        # Copy resources from Python implementation
        self.educational_markers = py_impl.educational_markers
        self.educational_markers_regex = py_impl.educational_markers_regex
        self.domain_keywords = py_impl.domain_keywords

    def normalize_token(self, token: str, language: Optional[str] = None) -> str:
        """
        Normalize a token using language-specific rules.

        Args:
            token: Token to normalize
            language: Optional language code

        Returns:
            Normalized token
        """
        lang = language if language else self.language

        if CPP_IMPLEMENTATION:
            return self._cpp_impl.normalize_token(token, lang)
        else:
            # Import the original Python implementation
            from mlcs_algorithm import MLCSAlgorithm as MLCSAlgorithmPy
            py_impl = MLCSAlgorithmPy(self.language)
            return py_impl.normalize_token(token, lang)

    def preprocess_text(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Preprocess text by tokenizing, removing stopwords, and normalizing tokens.

        Args:
            text: Input text
            language: Optional language code

        Returns:
            List of preprocessed tokens
        """
        lang = language if language else self.language

        if CPP_IMPLEMENTATION:
            return self._cpp_impl.preprocess_text(text, lang)
        else:
            # Import the original Python implementation
            from mlcs_algorithm import MLCSAlgorithm as MLCSAlgorithmPy
            py_impl = MLCSAlgorithmPy(self.language)
            return py_impl.preprocess_text(text, lang)

    def generate_variants(self, text: str) -> Set[str]:
        """
        Generate possible morphological variants of a term.

        Args:
            text: The original text

        Returns:
            Set of possible variants
        """
        if CPP_IMPLEMENTATION:
            return self._cpp_impl.generate_variants(text)
        else:
            # Import the original Python implementation
            from mlcs_algorithm import MLCSAlgorithm as MLCSAlgorithmPy
            py_impl = MLCSAlgorithmPy(self.language)
            return py_impl.generate_variants(text)

    def match_variants(self, text: str, target: str) -> float:
        """
        Check if text matches any variant of the target.

        Args:
            text: Text to check
            target: Target concept

        Returns:
            Similarity score (0.0-1.0), 1.0 if exact match
        """
        if CPP_IMPLEMENTATION:
            return self._cpp_impl.match_variants(text, target)
        else:
            # Import the original Python implementation
            from mlcs_algorithm import MLCSAlgorithm as MLCSAlgorithmPy
            py_impl = MLCSAlgorithmPy(self.language)
            return py_impl.match_variants(text, target)

    def find_mlcs(self, sequences: List[List[str]], min_length: int = 2) -> List[str]:
        """
        Find the Multiple Longest Common Subsequence across sequences.

        Args:
            sequences: List of token/character sequences
            min_length: Minimum length of common subsequence

        Returns:
            MLCS as a list of tokens/characters
        """
        if CPP_IMPLEMENTATION:
            return self._cpp_impl.find_mlcs(sequences, min_length)
        else:
            # Import the original Python implementation
            from mlcs_algorithm import MLCSAlgorithm as MLCSAlgorithmPy
            py_impl = MLCSAlgorithmPy(self.language)
            return py_impl.find_mlcs(sequences, min_length)

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
        lang = language if language else self.language

        if CPP_IMPLEMENTATION:
            return self._cpp_impl.extract_concept_signature(concept_text, contexts, lang)
        else:
            # Import the original Python implementation
            from mlcs_algorithm import MLCSAlgorithm as MLCSAlgorithmPy
            py_impl = MLCSAlgorithmPy(self.language)
            return py_impl.extract_concept_signature(concept_text, contexts, lang)

# Instantiate global algorithm instance
_default_mlcs_algorithm = None

def get_mlcs_algorithm(language: str = "en") -> MLCSAlgorithm:
    """
    Get or create the MLCSAlgorithm singleton instance.

    Args:
        language: Language code

    Returns:
        MLCSAlgorithm instance
    """
    global _default_mlcs_algorithm

    if _default_mlcs_algorithm is None:
        _default_mlcs_algorithm = MLCSAlgorithm(language)

    return _default_mlcs_algorithm
