"""
Domain Classifier module for the Lecture Video Content Indexer.
Classifies content into mathematics, programming, or physics domains.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Configure logging
logger = logging.getLogger(__name__)

class DomainClassifier:
    """
    Classifies educational content into domains (mathematics, programming, physics).
    Supports both rule-based classification and machine learning approaches.
    """

    def __init__(self):
        """Initialize the Domain Classifier."""
        logger.info("Initializing Domain Classifier")

        # Initialize keyword dictionaries for each domain and language
        self.domain_keywords = {
            "mathematics": {
                "en": [
                    "math", "mathematics", "calculus", "algebra", "geometry", "theorem",
                    "proof", "equation", "function", "derivative", "integral", "limit",
                    "vector", "matrix", "topology", "analysis", "discrete", "statistics",
                    "probability", "trigonometry", "polynomial", "set theory", "group theory",
                    "number theory", "optimization", "differential", "numerical",
                    "logarithm", "exponential", "quadratic", "linear", "inequality",
                    "coordinate", "graph", "formula", "axiom", "lemma", "corollary"
                ],
                "ru": [
                    "математика", "алгебра", "геометрия", "теорема", "доказательство",
                    "уравнение", "функция", "производная", "интеграл", "предел",
                    "вектор", "матрица", "топология", "анализ", "дискретная", "статистика",
                    "вероятность", "тригонометрия", "многочлен", "теория множеств",
                    "теория групп", "теория чисел", "оптимизация", "дифференциальный",
                    "численный", "логарифм", "экспоненциальный", "квадратный", "линейный",
                    "неравенство", "координата", "график", "формула", "аксиома", "лемма"
                ]
            },
            "programming": {
                "en": [
                    "programming", "code", "algorithm", "data structure", "function",
                    "variable", "class", "object", "method", "development", "software",
                    "python", "java", "c++", "javascript", "html", "css", "database",
                    "api", "library", "framework", "interface", "abstraction", "inheritance",
                    "polymorphism", "encapsulation", "loop", "condition", "compiler",
                    "interpreter", "debugging", "runtime", "memory", "exception", "testing",
                    "git", "version control", "frontend", "backend", "fullstack", "web"
                ],
                "ru": [
                    "программирование", "код", "алгоритм", "структура данных", "функция",
                    "переменная", "класс", "объект", "метод", "разработка", "программное обеспечение",
                    "питон", "python", "java", "джава", "с++", "c++", "javascript", "джаваскрипт",
                    "html", "css", "база данных", "апи", "api", "библиотека", "фреймворк",
                    "интерфейс", "абстракция", "наследование", "полиморфизм", "инкапсуляция",
                    "цикл", "условие", "компилятор", "интерпретатор", "отладка", "дебаг",
                    "время выполнения", "память", "исключение", "тестирование",
                    "гит", "git", "контроль версий", "фронтенд", "бэкенд", "веб"
                ]
            },
            "physics": {
                "en": [
                    "physics", "mechanics", "dynamics", "kinematics", "force", "energy",
                    "momentum", "newton", "electromagnetism", "thermodynamics", "quantum",
                    "relativity", "fluid", "wave", "particle", "gravity", "magnetic",
                    "electric", "optics", "light", "heat", "temperature", "pressure",
                    "velocity", "acceleration", "mass", "weight", "density", "torque",
                    "friction", "elasticity", "oscillation", "vibration", "sound",
                    "nuclear", "atomic", "subatomic", "plasma", "radiation", "field"
                ],
                "ru": [
                    "физика", "механика", "динамика", "кинематика", "сила", "энергия",
                    "импульс", "ньютон", "электромагнетизм", "термодинамика", "квантовая",
                    "относительность", "жидкость", "волна", "частица", "гравитация", "магнитный",
                    "электрический", "оптика", "свет", "тепло", "температура", "давление",
                    "скорость", "ускорение", "масса", "вес", "плотность", "крутящий момент",
                    "трение", "упругость", "колебание", "вибрация", "звук",
                    "ядерный", "атомный", "субатомный", "плазма", "излучение", "поле"
                ]
            }
        }

        # Initialize ML model
        self.ml_model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["mathematics", "programming", "physics", "unknown"])

    def train_model(self, training_data: List[Dict[str, Any]]):
        """
        Train a machine learning model for domain classification.

        Args:
            training_data: List of training examples with "text" and "domain" fields
        """
        if not training_data:
            logger.warning("No training data provided for domain classifier")
            return

        texts = [item.get("text", "") for item in training_data]
        domains = [item.get("domain", "unknown") for item in training_data]

        logger.info(f"Training domain classifier with {len(texts)} examples")

        try:
            # Create and train pipeline
            self.ml_model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )),
                ('classifier', LogisticRegression(
                    C=10,
                    max_iter=1000,
                    class_weight='balanced',
                    solver='lbfgs',
                    multi_class='multinomial'
                ))
            ])

            self.ml_model.fit(texts, domains)
            logger.info("Domain classifier trained successfully")

        except Exception as e:
            logger.error(f"Error training domain classifier: {e}")
            self.ml_model = None

    def classify_text(self, text: str, language: str = "en") -> Tuple[str, float]:
        """
        Classify text into a domain.

        Args:
            text: Text to classify
            language: Language code ('en' or 'ru')

        Returns:
            Tuple of (domain, confidence)
        """
        # Use ML model if available
        if self.ml_model is not None:
            try:
                # Get prediction and probability
                domain = self.ml_model.predict([text])[0]
                probs = self.ml_model.predict_proba([text])[0]
                confidence = max(probs)

                logger.debug(f"ML classification: {domain} with confidence {confidence:.2f}")

                # If confidence is high enough, return ML result
                if confidence > 0.7:
                    return domain, confidence

                # Otherwise, fall back to rule-based method
                logger.debug("Low ML confidence, falling back to rule-based classification")
            except Exception as e:
                logger.warning(f"Error in ML classification: {e}")

        # Rule-based classification
        return self._rule_based_classification(text, language)

    def classify_transcript(self, transcript: Dict[str, Any]) -> Tuple[str, float]:
        """
        Classify a full transcript into a domain.

        Args:
            transcript: Transcript dictionary

        Returns:
            Tuple of (domain, confidence)
        """
        language = transcript.get("language", "en")
        segments = transcript.get("segments", [])

        if not segments:
            logger.warning("Empty transcript provided for classification")
            return "unknown", 0.0

        # Combine segment texts for classification
        full_text = " ".join([segment.get("text", "") for segment in segments])

        # Use enhanced classification for full transcript
        return self._enhanced_classification(full_text, segments, language)

    def _rule_based_classification(self, text: str, language: str) -> Tuple[str, float]:
        """
        Perform rule-based domain classification.

        Args:
            text: Text to classify
            language: Language code ('en' or 'ru')

        Returns:
            Tuple of (domain, confidence)
        """
        lang_key = "ru" if language == "ru" else "en"
        text = text.lower()

        # Count keyword matches for each domain
        domain_scores = {}

        for domain, lang_keywords in self.domain_keywords.items():
            keywords = lang_keywords.get(lang_key, lang_keywords.get("en", []))
            score = 0

            for keyword in keywords:
                # Count keyword occurrences with word boundary check
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = re.findall(pattern, text)
                score += len(matches)

            domain_scores[domain] = score

        # Find domain with highest score
        max_score = max(domain_scores.values())

        if max_score == 0:
            return "unknown", 0.0

        # Get domains with max score
        max_domains = [domain for domain, score in domain_scores.items() if score == max_score]

        if len(max_domains) == 1:
            domain = max_domains[0]
            total = sum(domain_scores.values())
            confidence = max_score / total if total > 0 else 0.0
            return domain, confidence
        else:
            # If tie, return the first domain with medium confidence
            return max_domains[0], 0.5

    def _enhanced_classification(self, full_text: str, segments: List[Dict], language: str) -> Tuple[str, float]:
        """
        Perform enhanced domain classification using segment-level information.

        Args:
            full_text: Combined text from all segments
            segments: List of transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            Tuple of (domain, confidence)
        """
        # First, get base classification from full text
        base_domain, base_confidence = self._rule_based_classification(full_text, language)

        # If we have high confidence, return result
        if base_confidence > 0.8:
            return base_domain, base_confidence

        # Otherwise, classify each segment individually
        segment_classifications = []

        for segment in segments:
            segment_text = segment.get("text", "")
            if segment_text:
                domain, confidence = self._rule_based_classification(segment_text, language)
                if domain != "unknown" and confidence > 0.0:
                    segment_classifications.append((domain, confidence))

        if not segment_classifications:
            return base_domain, base_confidence

        # Aggregate segment classifications
        domain_weights = {}

        for domain, confidence in segment_classifications:
            if domain not in domain_weights:
                domain_weights[domain] = 0
            domain_weights[domain] += confidence

        # Find domain with highest total weight
        if domain_weights:
            top_domain = max(domain_weights.items(), key=lambda x: x[1])
            total_weight = sum(domain_weights.values())
            confidence = top_domain[1] / total_weight if total_weight > 0 else 0.0
            return top_domain[0], confidence

        # Fallback to base classification
        return base_domain, base_confidence

    def extract_domain_specific_features(self, transcript: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Extract domain-specific features from a transcript.

        Args:
            transcript: Transcript dictionary
            domain: Domain of the transcript

        Returns:
            Dictionary of domain-specific features
        """
        language = transcript.get("language", "en")
        segments = transcript.get("segments", [])

        features = {
            "domain": domain,
            "theoretical_segments": 0,
            "practical_segments": 0,
            "key_concepts": [],
            "domain_specific_metadata": {}
        }

        if not segments:
            return features

        # Count theoretical and practical segments
        for segment in segments:
            content_type = segment.get("content_type", "")
            if content_type == "theoretical":
                features["theoretical_segments"] += 1
            elif content_type == "practical":
                features["practical_segments"] += 1

        # Extract domain-specific concepts based on domain
        if domain == "mathematics":
            features["domain_specific_metadata"] = self._extract_math_features(transcript, language)
        elif domain == "programming":
            features["domain_specific_metadata"] = self._extract_programming_features(transcript, language)
        elif domain == "physics":
            features["domain_specific_metadata"] = self._extract_physics_features(transcript, language)

        # Extract key concepts using simple frequency analysis
        features["key_concepts"] = self._extract_key_concepts(transcript, domain, language)

        return features

    def _extract_math_features(self, transcript: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Extract mathematics-specific features."""
        segments = transcript.get("segments", [])

        features = {
            "formulas_count": 0,
            "theorems_count": 0,
            "proofs_count": 0,
            "definitions_count": 0,
            "topics": []
        }

        # Extract formulas
        for segment in segments:
            nlp_data = segment.get("nlp_data", {})
            formulas = nlp_data.get("formulas", [])
            features["formulas_count"] += len(formulas)

            # Count sentence types
            sentence_type = nlp_data.get("sentence_type", "")
            if sentence_type == "definition":
                features["definitions_count"] += 1

            # Check for theorems and proofs
            text = segment.get("text", "").lower()

            if language == "ru":
                if "теорема" in text:
                    features["theorems_count"] += 1
                if "доказательство" in text or "докажем" in text:
                    features["proofs_count"] += 1
            else:
                if "theorem" in text:
                    features["theorems_count"] += 1
                if "proof" in text or "prove" in text:
                    features["proofs_count"] += 1

        # Identify math topics
        math_topics = {
            "en": {
                "algebra": r'\b(?:algebra|algebraic)\b',
                "calculus": r'\b(?:calculus|derivative|integral)\b',
                "geometry": r'\b(?:geometry|geometric|triangle|circle)\b',
                "statistics": r'\b(?:statistics|probability|distribution)\b',
                "linear_algebra": r'\b(?:linear algebra|matrix|vector space)\b',
                "number_theory": r'\b(?:number theory|prime|divisor)\b'
            },
            "ru": {
                "algebra": r'\b(?:алгебра|алгебраич)\b',
                "calculus": r'\b(?:матанализ|производная|интеграл)\b',
                "geometry": r'\b(?:геометри|треугольник|окружность)\b',
                "statistics": r'\b(?:статистик|вероятност|распределени)\b',
                "linear_algebra": r'\b(?:линейная алгебра|матриц|векторн)\b',
                "number_theory": r'\b(?:теори[яю] чисел|прост[ыое]|делител)\b'
            }
        }

        lang_key = "ru" if language == "ru" else "en"
        full_text = " ".join([segment.get("text", "") for segment in segments])

        topics = []
        for topic, pattern in math_topics.get(lang_key, {}).items():
            if re.search(pattern, full_text, re.IGNORECASE):
                topics.append(topic)

        features["topics"] = topics

        return features

    def _extract_programming_features(self, transcript: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Extract programming-specific features."""
        segments = transcript.get("segments", [])

        features = {
            "code_snippets_count": 0,
            "languages_mentioned": [],
            "algorithms_count": 0,
            "data_structures_count": 0,
            "topics": []
        }

        # Programming languages to detect
        prog_languages = {
            "python": r'\bpython\b',
            "java": r'\bjava\b',
            "cpp": r'\b(?:c\+\+|cpp)\b',
            "javascript": r'\b(?:javascript|js)\b',
            "csharp": r'\b(?:c#|csharp)\b',
            "php": r'\bphp\b',
            "ruby": r'\bruby\b',
            "go": r'\bgo lang\b',
            "rust": r'\b(?:rust|rustlang)\b',
            "swift": r'\bswift\b'
        }

        # Extract code snippets and languages
        full_text = ""
        for segment in segments:
            text = segment.get("text", "")
            full_text += " " + text

            nlp_data = segment.get("nlp_data", {})
            code_snippets = nlp_data.get("code_snippets", [])
            features["code_snippets_count"] += len(code_snippets)

            # Check for languages in code snippets
            for snippet in code_snippets:
                lang = snippet.get("language", "").lower()
                if lang and lang != "unknown" and lang not in features["languages_mentioned"]:
                    features["languages_mentioned"].append(lang)

        # Check for languages in text
        for lang, pattern in prog_languages.items():
            if re.search(pattern, full_text, re.IGNORECASE) and lang not in features["languages_mentioned"]:
                features["languages_mentioned"].append(lang)

        # Check for algorithm and data structure mentions
        programming_topics = {
            "en": {
                "algorithms": r'\b(?:algorithm|sorting|searching|recursion)\b',
                "data_structures": r'\b(?:data structure|array|linked list|tree|graph|hash|stack|queue)\b',
                "web_dev": r'\b(?:web|html|css|frontend|backend|api|http|ajax)\b',
                "databases": r'\b(?:database|sql|nosql|query|mongodb|postgresql)\b',
                "oop": r'\b(?:object.?oriented|class|inheritance|polymorphism|encapsulation)\b',
                "functional": r'\b(?:functional programming|lambda|closure|immutable)\b'
            },
            "ru": {
                "algorithms": r'\b(?:алгоритм|сортировк|поиск|рекурси)\b',
                "data_structures": r'\b(?:структур[аы] данных|массив|список|дерево|граф|хеш|стек|очередь)\b',
                "web_dev": r'\b(?:веб|html|css|фронтенд|бэкенд|api|http|ajax)\b',
                "databases": r'\b(?:баз[аы] данных|sql|nosql|запрос|mongodb|postgresql)\b',
                "oop": r'\b(?:объектно.?ориентированн|класс|наследовани|полиморфизм|инкапсуляци)\b',
                "functional": r'\b(?:функциональное программирование|лямбда|замыкани|неизменяем)\b'
            }
        }

        lang_key = "ru" if language == "ru" else "en"

        topics = []
        for topic, pattern in programming_topics.get(lang_key, {}).items():
            if re.search(pattern, full_text, re.IGNORECASE):
                topics.append(topic)

                # Count algorithms and data structures
                if topic == "algorithms":
                    features["algorithms_count"] += len(re.findall(pattern, full_text, re.IGNORECASE))
                elif topic == "data_structures":
                    features["data_structures_count"] += len(re.findall(pattern, full_text, re.IGNORECASE))

        features["topics"] = topics

        return features

    def _extract_physics_features(self, transcript: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Extract physics-specific features."""
        segments = transcript.get("segments", [])

        features = {
            "equations_count": 0,
            "experiments_count": 0,
            "laws_count": 0,
            "constants_count": 0,
            "topics": []
        }

        # Extract physics-specific patterns
        full_text = ""
        for segment in segments:
            text = segment.get("text", "").lower()
            full_text += " " + text

            nlp_data = segment.get("nlp_data", {})
            formulas = nlp_data.get("formulas", [])
            features["equations_count"] += len(formulas)

            # Check for experiments
            if language == "ru":
                if re.search(r'\bэксперимент|\bопыт|\bизмерени', text):
                    features["experiments_count"] += 1
            else:
                if re.search(r'\bexperiment|\blab|\bmeasurement', text):
                    features["experiments_count"] += 1

            # Check for physical laws
            if language == "ru":
                if re.search(r'закон[а-я]*\s[А-Я][а-я]+', text):  # "закон Ньютона", etc.
                    features["laws_count"] += 1
            else:
                if re.search(r'law\s+of\s+[A-Z][a-z]+|[A-Z][a-z]+\'s\s+law', text):
                    features["laws_count"] += 1

            # Check for physical constants
            constants_ru = [r'скорость света', r'постоянная планка', r'гравитационная постоянная']
            constants_en = [r'speed of light', r'planck\'s constant', r'gravitational constant']

            if language == "ru":
                for constant in constants_ru:
                    if re.search(constant, text):
                        features["constants_count"] += 1
            else:
                for constant in constants_en:
                    if re.search(constant, text):
                        features["constants_count"] += 1

        # Identify physics topics
        physics_topics = {
            "en": {
                "mechanics": r'\b(?:mechanics|force|motion|momentum|acceleration)\b',
                "electromagnetism": r'\b(?:electromagnet|electric|magnetic|field|charge)\b',
                "thermodynamics": r'\b(?:thermodynamics|heat|temperature|entropy|energy)\b',
                "quantum": r'\b(?:quantum|wave.?particle|uncertainty|superposition)\b',
                "relativity": r'\b(?:relativity|einstein|spacetime|gravity|dilation)\b',
                "optics": r'\b(?:optics|light|lens|refraction|reflection|diffraction)\b'
            },
            "ru": {
                "mechanics": r'\b(?:механик|сил|движени|импульс|ускорени)\b',
                "electromagnetism": r'\b(?:электромагнет|электрическ|магнитн|пол[ея]|заряд)\b',
                "thermodynamics": r'\b(?:термодинамик|тепл|температур|энтропи|энерги)\b',
                "quantum": r'\b(?:квантов|волн.?частиц|неопределенност|суперпозици)\b',
                "relativity": r'\b(?:относительност|эйнштейн|пространство.?врем|гравитаци|дилатаци)\b',
                "optics": r'\b(?:оптик|свет|линз|преломлени|отражени|дифракци)\b'
            }
        }

        lang_key = "ru" if language == "ru" else "en"

        topics = []
        for topic, pattern in physics_topics.get(lang_key, {}).items():
            if re.search(pattern, full_text, re.IGNORECASE):
                topics.append(topic)

        features["topics"] = topics

        return features

    def _extract_key_concepts(self, transcript: Dict[str, Any], domain: str, language: str) -> List[Dict[str, Any]]:
        """Extract key concepts from transcript based on domain."""
        segments = transcript.get("segments", [])
        full_text = " ".join([segment.get("text", "") for segment in segments])

        # Define domain-specific concept extraction patterns
        if domain == "mathematics":
            if language == "ru":
                patterns = [
                    r'(?:понятие|концепция|определение)\s+([а-яА-ЯёЁ\s]+)',
                    r'([а-яА-ЯёЁ\s]+)\s+(?:называется|определяется как)',
                    r'теорема\s+([а-яА-ЯёЁ\s]+)'
                ]
            else:
                patterns = [
                    r'(?:concept|definition)\s+of\s+([a-zA-Z\s]+)',
                    r'([a-zA-Z\s]+)\s+(?:is defined as|is called)',
                    r'theorem\s+of\s+([a-zA-Z\s]+)',
                    r'([a-zA-Z\s]+)\s+theorem'
                ]
        elif domain == "programming":
            if language == "ru":
                patterns = [
                    r'алгоритм\s+([а-яА-ЯёЁ\s]+)',
                    r'структура данных\s+([а-яА-ЯёЁ\s]+)',
                    r'парадигма\s+([а-яА-ЯёЁ\s]+)',
                    r'(?:класс|функция|метод)\s+([а-яА-Я0-9_]+)'
                ]
            else:
                patterns = [
                    r'algorithm\s+(?:of\s+)?([a-zA-Z\s]+)',
                    r'data structure\s+(?:of\s+)?([a-zA-Z\s]+)',
                    r'paradigm\s+of\s+([a-zA-Z\s]+)',
                    r'(?:class|function|method)\s+([a-zA-Z0-9_]+)'
                ]
        elif domain == "physics":
            if language == "ru":
                patterns = [
                    r'закон\s+([а-яА-ЯёЁ\s]+)',
                    r'принцип\s+([а-яА-ЯёЁ\s]+)',
                    r'явление\s+([а-яА-ЯёЁ\s]+)',
                    r'([а-яА-ЯёЁ\s]+)\s+эффект'
                ]
            else:
                patterns = [
                    r'law\s+of\s+([a-zA-Z\s]+)',
                    r'principle\s+of\s+([a-zA-Z\s]+)',
                    r'phenomenon\s+of\s+([a-zA-Z\s]+)',
                    r'([a-zA-Z\s]+)\s+effect'
                ]
        else:
            patterns = []

        # Extract concepts
        concepts = []
        for pattern in patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                concept_text = match.group(1).strip()
                if concept_text and len(concept_text) > 3 and concept_text not in [c["text"] for c in concepts]:
                    concepts.append({
                        "text": concept_text,
                        "domain": domain,
                        "frequency": self._count_concept_occurrences(concept_text, full_text),
                        "theoretical": self._is_concept_theoretical(concept_text, segments, language)
                    })

        # Return top concepts by frequency
        return sorted(concepts, key=lambda x: x["frequency"], reverse=True)[:10]

    def _count_concept_occurrences(self, concept: str, text: str) -> int:
        """Count occurrences of a concept in text."""
        # Create pattern with word boundaries for whole concept
        pattern = r'\b' + re.escape(concept) + r'\b'
        return len(re.findall(pattern, text, re.IGNORECASE))

    def _is_concept_theoretical(self, concept: str, segments: List[Dict], language: str) -> bool:
        """Determine if a concept is predominantly theoretical or practical."""
        theoretical_count = 0
        practical_count = 0

        for segment in segments:
            text = segment.get("text", "")
            if concept.lower() in text.lower():
                content_type = segment.get("content_type", "")
                if content_type == "theoretical":
                    theoretical_count += 1
                elif content_type == "practical":
                    practical_count += 1

        # If the concept appears more in theoretical segments, classify it as theoretical
        return theoretical_count >= practical_count
