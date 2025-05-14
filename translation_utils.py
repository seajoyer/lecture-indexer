"""
Translation utility for concept extraction and repository management.
Provides translation between languages with graceful degradation.
"""

import logging
import re
import json
import os
from typing import Dict, Optional, Tuple

# Try to import translation libraries with fallbacks
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    logging.warning("deep_translator not available - using fallback translation")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available - using local dictionary only")

# Configure logging
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Service for translating between languages with multiple fallback mechanisms.
    """

    def __init__(self):
        """Initialize the translation service with available translators."""
        self.translators = {}
        self.dictionaries = {}
        self.cache = {}  # Simple in-memory cache

        # Dictionary file paths
        self.dict_dir = os.path.join(os.path.dirname(__file__), "translation_dicts")
        os.makedirs(self.dict_dir, exist_ok=True)

        # Load or create dictionaries
        self._load_dictionaries()

        # Initialize available translators
        self._init_translators()

        logger.info("TranslationService initialized")

    def _init_translators(self):
        """Initialize available translation services."""
        # Initialize deep_translator if available (Google Translate)
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                # Pre-initialize common language pairs
                self.translators['en_to_ru'] = GoogleTranslator(source='en', target='ru')
                self.translators['ru_to_en'] = GoogleTranslator(source='ru', target='en')
                logger.info("GoogleTranslator initialized")
            except Exception as e:
                logger.warning(f"Error initializing GoogleTranslator: {e}")

    def _load_dictionaries(self):
        """Load translation dictionaries from files or create new ones."""
        # Check for existing dictionary files
        dict_files = {
            'en_to_ru': os.path.join(self.dict_dir, "en_to_ru_dict.json"),
            'ru_to_en': os.path.join(self.dict_dir, "ru_to_en_dict.json")
        }

        # Load or create dictionaries
        for key, file_path in dict_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.dictionaries[key] = json.load(f)
                    logger.info(f"Loaded dictionary {key} with {len(self.dictionaries[key])} entries")
                except Exception as e:
                    logger.warning(f"Error loading dictionary {key}: {e}")
                    self.dictionaries[key] = {}
            else:
                self.dictionaries[key] = {}
                logger.info(f"Created new dictionary {key}")

    def _save_dictionary(self, dict_type: str):
        """
        Save a dictionary to file.

        Args:
            dict_type: Dictionary type ('en_to_ru' or 'ru_to_en')
        """
        if dict_type not in self.dictionaries:
            return

        file_path = os.path.join(self.dict_dir, f"{dict_type}_dict.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.dictionaries[dict_type], f, ensure_ascii=False, indent=2)
            logger.info(f"Saved dictionary {dict_type} with {len(self.dictionaries[dict_type])} entries")
        except Exception as e:
            logger.warning(f"Error saving dictionary {dict_type}: {e}")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language.

        Args:
            text: Text to translate
            source_lang: Source language code ('en' or 'ru')
            target_lang: Target language code ('en' or 'ru')

        Returns:
            Translated text
        """
        if not text:
            return ""

        # Check if languages are the same
        if source_lang == target_lang:
            return text

        # Make sure source and target are supported
        if source_lang not in ['en', 'ru'] or target_lang not in ['en', 'ru']:
            logger.warning(f"Unsupported language pair: {source_lang} to {target_lang}")
            return text

        # Create a cache key
        cache_key = f"{source_lang}_{target_lang}_{text}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Determine dictionary type
        dict_type = f"{source_lang}_to_{target_lang}"

        # Normalize text for dictionary lookup (lowercase)
        normalized_text = text.lower()

        # Method 1: Check the dictionary first
        if dict_type in self.dictionaries and normalized_text in self.dictionaries[dict_type]:
            # Return the cached translation
            result = self.dictionaries[dict_type][normalized_text]
            self.cache[cache_key] = result
            return result

        # Method 2: Try deep_translator if available
        if DEEP_TRANSLATOR_AVAILABLE and dict_type in self.translators:
            try:
                translator = self.translators[dict_type]
                translation = translator.translate(text)

                if translation:
                    # Add to dictionary for future use
                    self.dictionaries[dict_type][normalized_text] = translation.lower()
                    self._save_dictionary(dict_type)

                    # Add to cache
                    self.cache[cache_key] = translation.lower()
                    return translation.lower()
            except Exception as e:
                logger.warning(f"Translation error with deep_translator: {e}")

        # Method 3: If requests is available, try a public translation API
        if REQUESTS_AVAILABLE:
            try:
                translation = self._translate_with_fallback_api(text, source_lang, target_lang)
                if translation:
                    # Add to dictionary for future use
                    self.dictionaries[dict_type][normalized_text] = translation.lower()
                    self._save_dictionary(dict_type)

                    # Add to cache
                    self.cache[cache_key] = translation.lower()
                    return translation.lower()
            except Exception as e:
                logger.warning(f"Translation error with fallback API: {e}")

        # Method 4: Rule-based transliteration for common terms
        transliteration = self._transliterate(text, source_lang, target_lang)
        if transliteration != text:
            # Add to dictionary for future use
            self.dictionaries[dict_type][normalized_text] = transliteration.lower()
            self._save_dictionary(dict_type)

            # Add to cache
            self.cache[cache_key] = transliteration.lower()
            return transliteration.lower()

        # If all methods fail, return the original text
        logger.warning(f"Translation failed for '{text}' from {source_lang} to {target_lang}")
        return text.lower()

    def _translate_with_fallback_api(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Try to translate using publicly available APIs as fallback.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text or empty string if failed
        """
        # Simple fallback to a public API
        try:
            # This is a placeholder for a real implementation
            # In a production environment, you would use a proper API with authentication
            api_url = "https://api.mymemory.translated.net/get"
            params = {
                "q": text,
                "langpair": f"{source_lang}|{target_lang}"
            }

            response = requests.get(api_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and "responseData" in data and "translatedText" in data["responseData"]:
                    return data["responseData"]["translatedText"].lower()
        except Exception as e:
            logger.warning(f"Fallback API translation error: {e}")

        return ""

    def _transliterate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Perform simple rule-based transliteration between languages.

        Args:
            text: Text to transliterate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Transliterated text
        """
        # Only handle Russian to English and vice versa
        if source_lang == 'ru' and target_lang == 'en':
            return self._transliterate_russian_to_english(text)
        elif source_lang == 'en' and target_lang == 'ru':
            return self._transliterate_english_to_russian(text)

        return text

    def _transliterate_russian_to_english(self, text: str) -> str:
        """
        Transliterate Russian text to English.

        Args:
            text: Russian text

        Returns:
            Transliterated text
        """
        # Mapping of Russian characters to English
        russian_to_english = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
            # Upper case
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
            'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch',
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
        }

        # Common Russian physics/math terms with direct English translations
        russian_terms = {
            'квантовая механика': 'quantum mechanics',
            'квантовый': 'quantum',
            'волновая функция': 'wave function',
            'уравнение': 'equation',
            'принцип неопределенности': 'uncertainty principle',
            'оператор': 'operator',
            'собственное значение': 'eigenvalue',
            'собственный вектор': 'eigenvector',
            'состояние': 'state',
            'измерение': 'measurement',
            'вероятность': 'probability',
            'энергия': 'energy',
            'импульс': 'momentum',
            'координата': 'coordinate',
            'положение': 'position',
            'скорость': 'velocity',
            'ускорение': 'acceleration',
            'масса': 'mass',
            'сила': 'force',
            'поле': 'field',
            'заряд': 'charge',
            'частица': 'particle',
            'волна': 'wave',
            'спин': 'spin',
            'орбитальный': 'orbital',
            'ядро': 'nucleus',
            'электрон': 'electron',
            'протон': 'proton',
            'нейтрон': 'neutron',
            'фотон': 'photon',
            'атом': 'atom',
            'молекула': 'molecule',
            'теорема': 'theorem',
            'теория': 'theory',
            'закон': 'law',
            'физика': 'physics',
            'математика': 'mathematics',
            'алгоритм': 'algorithm',
            'функция': 'function',
            'производная': 'derivative',
            'интеграл': 'integral',
            'дифференциал': 'differential',
            'матрица': 'matrix',
            'вектор': 'vector'
        }

        # Check for direct term match first
        text_lower = text.lower()
        if text_lower in russian_terms:
            return russian_terms[text_lower]

        # Try to match parts of the text
        for ru_term, en_term in russian_terms.items():
            if ru_term in text_lower:
                return text_lower.replace(ru_term, en_term)

        # If no direct match, do character-by-character transliteration
        result = ""
        for char in text:
            result += russian_to_english.get(char, char)

        return result

    def _transliterate_english_to_russian(self, text: str) -> str:
        """
        Transliterate English text to Russian.

        Args:
            text: English text

        Returns:
            Transliterated text
        """
        # Mapping of English sounds to Russian characters
        english_to_russian = {
            'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
            'g': 'г', 'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л',
            'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
            's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
            'y': 'й', 'z': 'з',
            # Special combinations
            'ch': 'ч', 'sh': 'ш', 'sch': 'щ', 'zh': 'ж',
            'th': 'т', 'ph': 'ф', 'kh': 'х', 'ts': 'ц',
            'ya': 'я', 'yo': 'ё', 'yu': 'ю', 'ye': 'е'
        }

        # Common English physics/math terms with direct Russian translations
        english_terms = {
            'quantum mechanics': 'квантовая механика',
            'quantum': 'квантовый',
            'wave function': 'волновая функция',
            'equation': 'уравнение',
            'uncertainty principle': 'принцип неопределенности',
            'operator': 'оператор',
            'eigenvalue': 'собственное значение',
            'eigenvector': 'собственный вектор',
            'state': 'состояние',
            'measurement': 'измерение',
            'probability': 'вероятность',
            'energy': 'энергия',
            'momentum': 'импульс',
            'coordinate': 'координата',
            'position': 'положение',
            'velocity': 'скорость',
            'acceleration': 'ускорение',
            'mass': 'масса',
            'force': 'сила',
            'field': 'поле',
            'charge': 'заряд',
            'particle': 'частица',
            'wave': 'волна',
            'spin': 'спин',
            'orbital': 'орбитальный',
            'nucleus': 'ядро',
            'electron': 'электрон',
            'proton': 'протон',
            'neutron': 'нейтрон',
            'photon': 'фотон',
            'atom': 'атом',
            'molecule': 'молекула',
            'theorem': 'теорема',
            'theory': 'теория',
            'law': 'закон',
            'physics': 'физика',
            'mathematics': 'математика',
            'algorithm': 'алгоритм',
            'function': 'функция',
            'derivative': 'производная',
            'integral': 'интеграл',
            'differential': 'дифференциал',
            'matrix': 'матрица',
            'vector': 'вектор'
        }

        # Check for direct term match first
        text_lower = text.lower()
        if text_lower in english_terms:
            return english_terms[text_lower]

        # Try to match parts of the text
        for en_term, ru_term in english_terms.items():
            if en_term in text_lower:
                return text_lower.replace(en_term, ru_term)

        # If no direct match, better to return original than try character transliteration
        # as English->Russian transliteration is more complex
        return text

# Singleton instance
_translation_service = None

def get_translation_service() -> TranslationService:
    """
    Get the singleton instance of TranslationService.

    Returns:
        TranslationService instance
    """
    global _translation_service

    if _translation_service is None:
        _translation_service = TranslationService()

    return _translation_service

# Helper functions
def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source language to target language.

    Args:
        text: Text to translate
        source_lang: Source language code ('en' or 'ru')
        target_lang: Target language code ('en' or 'ru')

    Returns:
        Translated text
    """
    return get_translation_service().translate(text, source_lang, target_lang)

def generate_concept_id(text: str) -> str:
    """
    Generate a concept ID from text.

    Args:
        text: Text to generate ID from

    Returns:
        Concept ID
    """
    # Ensure text is in English - if it's Russian, translate it
    if any(char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for char in text.lower()):
        text = translate(text, 'ru', 'en')

    # Convert to lowercase
    text = text.lower()

    # Replace spaces with underscores
    text = text.replace(' ', '_')

    # Remove any non-alphanumeric characters except underscores
    text = re.sub(r'[^a-z0-9_]', '', text)

    # Ensure the ID is not empty
    if not text:
        text = "concept_" + str(hash(text) % 10000)

    return text
