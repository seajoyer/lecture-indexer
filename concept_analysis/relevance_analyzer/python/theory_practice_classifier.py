"""
Theory Practice Classifier module for the Lecture Video Content Indexer.
Classifies content as theoretical or practical based on linguistic markers and content analysis.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Configure logging
logger = logging.getLogger(__name__)

class TheoryPracticeClassifier:
    """
    Classifies educational content as theoretical or practical.
    Distinguishes between abstract explanations and concrete problem-solving.
    Supports Russian and English language content.
    """

    def __init__(self):
        """Initialize the Theory Practice Classifier."""
        logger.info("Initializing Theory Practice Classifier")

        # Initialize rule-based classification patterns
        self._init_classification_patterns()

        # Initialize ML model
        self.ml_model = None

        # Initialize semantic features extraction
        self._init_semantic_features()

    def _init_classification_patterns(self):
        """Initialize patterns for rule-based classification."""
        # Patterns for theoretical content
        self.theoretical_patterns = {
            "en": [
                # Definition patterns
                r'(?:is defined as|refers to|means|is called|is known as)',
                # Theorem/proof structures
                r'(?:we can prove|it follows that|we will show|theorem|lemma|corollary)',
                # Abstract discussion
                r'(?:concept of|theory|abstract|fundamental|principle)',
                # Explanation patterns
                r'(?:understand|conceptualize|comprehend|grasp|consider)'
            ],
            "ru": [
                # Definition patterns
                r'(?:определяется как|означает|называется|известен как)',
                # Theorem/proof structures
                r'(?:докажем|следовательно|покажем|теорема|лемма|следствие)',
                # Abstract discussion
                r'(?:понятие|теория|абстрактный|фундаментальный|принцип)',
                # Explanation patterns
                r'(?:понять|осмыслить|постичь|представить|рассмотрим)'
            ]
        }

        # Patterns for practical content
        self.practical_patterns = {
            "en": [
                # Problem statements
                r'(?:solve for|find the|calculate|compute|determine|evaluate)',
                # Application contexts
                r'(?:in practice|real-world example|application|implementation|use case)',
                # Step-by-step solutions
                r'(?:first step|next we|then|following steps|procedure)',
                # Code implementation
                r'(?:implement|coding|program|function|method|algorithm)',
                # Numerical examples
                r'(?:\d+\s*[+\-*/]\s*\d+|result is|output|value)'
            ],
            "ru": [
                # Problem statements
                r'(?:решите|найдите|вычислите|определите|оцените)',
                # Application contexts
                r'(?:на практике|пример из жизни|применение|реализация|пример использования)',
                # Step-by-step solutions
                r'(?:сперва|затем|далее|следующие шаги|процедура)',
                # Code implementation
                r'(?:реализация|программирование|функция|метод|алгоритм)',
                # Numerical examples
                r'(?:\d+\s*[+\-*/]\s*\d+|результат|вывод|значение)'
            ]
        }

        # Domain-specific patterns for theory and practice
        self.domain_patterns = {
            "mathematics": {
                "theoretical": {
                    "en": [
                        r'(?:definition|axiom|postulate|theorem|proof)',
                        r'(?:let us define|consider|given that|assume that)',
                        r'(?:mathematical structure|abstract algebra|topology)'
                    ],
                    "ru": [
                        r'(?:определение|аксиома|постулат|теорема|доказательство)',
                        r'(?:определим|рассмотрим|дано что|предположим что)',
                        r'(?:математическая структура|абстрактная алгебра|топология)'
                    ]
                },
                "practical": {
                    "en": [
                        r'(?:solve the equation|calculate|compute|find the value)',
                        r'(?:example|exercise|problem|application)',
                        r'(?:step by step|method|approach|technique)'
                    ],
                    "ru": [
                        r'(?:решите уравнение|вычислите|найдите значение)',
                        r'(?:пример|упражнение|задача|приложение)',
                        r'(?:шаг за шагом|метод|подход|техника)'
                    ]
                }
            },
            "programming": {
                "theoretical": {
                    "en": [
                        r'(?:computer science|computational theory|algorithm analysis)',
                        r'(?:complexity|asymptotic|big O notation)',
                        r'(?:paradigm|principle|concept|design pattern)'
                    ],
                    "ru": [
                        r'(?:информатика|теория вычислений|анализ алгоритмов)',
                        r'(?:сложность|асимптотический|O-большое)',
                        r'(?:парадигма|принцип|концепция|шаблон проектирования)'
                    ]
                },
                "practical": {
                    "en": [
                        r'(?:code|implementation|writing|programming)',
                        r'(?:function|method|class|object|library)',
                        r'(?:compile|run|execute|debug|deploy)',
                        r'(?:syntax|error|bug|exception)',
                        r'```'  # Code blocks in markdown
                    ],
                    "ru": [
                        r'(?:код|реализация|написание|программирование)',
                        r'(?:функция|метод|класс|объект|библиотека)',
                        r'(?:компилировать|запускать|выполнять|отлаживать|развертывать)',
                        r'(?:синтаксис|ошибка|баг|исключение)',
                        r'```'  # Code blocks in markdown
                    ]
                }
            },
            "physics": {
                "theoretical": {
                    "en": [
                        r'(?:theory|law|principle|hypothesis)',
                        r'(?:conceptual framework|theoretical model)',
                        r'(?:derivation|mathematical formulation)'
                    ],
                    "ru": [
                        r'(?:теория|закон|принцип|гипотеза)',
                        r'(?:концептуальная основа|теоретическая модель)',
                        r'(?:вывод|математическая формулировка)'
                    ]
                },
                "practical": {
                    "en": [
                        r'(?:experiment|measurement|observation|data)',
                        r'(?:laboratory|apparatus|setup|equipment)',
                        r'(?:calculate|compute|determine|estimate)',
                        r'(?:practical application|real-world example)'
                    ],
                    "ru": [
                        r'(?:эксперимент|измерение|наблюдение|данные)',
                        r'(?:лаборатория|аппарат|установка|оборудование)',
                        r'(?:вычислить|рассчитать|определить|оценить)',
                        r'(?:практическое применение|пример из реальной жизни)'
                    ]
                }
            }
        }

    def _init_semantic_features(self):
        """Initialize semantic feature extraction."""
        # Theoretical semantic markers
        self.theoretical_semantic = {
            "en": {
                "high_abstraction": ["abstract", "concept", "theory", "general", "universal", "fundamental"],
                "definitions": ["define", "definition", "meaning", "denote", "represent"],
                "reasoning": ["proof", "prove", "theorem", "derive", "imply", "deduce"],
                "understanding": ["understand", "comprehend", "grasp", "insight", "intuition"]
            },
            "ru": {
                "high_abstraction": ["абстрактный", "концепция", "теория", "общий", "универсальный", "фундаментальный"],
                "definitions": ["определять", "определение", "значение", "обозначать", "представлять"],
                "reasoning": ["доказательство", "доказывать", "теорема", "выводить", "подразумевать", "делать вывод"],
                "understanding": ["понимать", "постигать", "схватывать", "понимание", "интуиция"]
            }
        }

        # Practical semantic markers
        self.practical_semantic = {
            "en": {
                "application": ["apply", "application", "use", "implement", "practical"],
                "problem_solving": ["problem", "solution", "solve", "approach", "method"],
                "concrete_examples": ["example", "instance", "case", "specific", "particular"],
                "action_steps": ["step", "procedure", "action", "operation", "activity"]
            },
            "ru": {
                "application": ["применять", "применение", "использовать", "реализовывать", "практический"],
                "problem_solving": ["проблема", "решение", "решать", "подход", "метод"],
                "concrete_examples": ["пример", "экземпляр", "случай", "конкретный", "частный"],
                "action_steps": ["шаг", "процедура", "действие", "операция", "деятельность"]
            }
        }

    def train_model(self, training_data: List[Dict[str, Any]]):
        """
        Train a machine learning model for theory/practice classification.

        Args:
            training_data: List of training examples with "text" and "classification" fields
        """
        if not training_data:
            logger.warning("No training data provided for theory/practice classifier")
            return

        texts = [item.get("text", "") for item in training_data]
        labels = [item.get("classification", "unknown") for item in training_data]

        logger.info(f"Training theory/practice classifier with {len(texts)} examples")

        try:
            # Create and train an ensemble model
            self.ml_model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )),
                ('classifier', VotingClassifier(estimators=[
                    ('nb', MultinomialNB()),
                    ('lr', LogisticRegression(
                        C=10,
                        max_iter=1000,
                        class_weight='balanced'
                    ))
                ], voting='soft'))
            ])

            self.ml_model.fit(texts, labels)
            logger.info("Theory/practice classifier trained successfully")

        except Exception as e:
            logger.error(f"Error training theory/practice classifier: {e}")
            self.ml_model = None

    def classify_text(self, text: str, language: str = "en", domain: str = None) -> Tuple[str, float]:
        """
        Classify text as theoretical or practical.

        Args:
            text: Text to classify
            language: Language code ('en' or 'ru')
            domain: Domain of the text (mathematics, programming, physics)

        Returns:
            Tuple of (classification, confidence)
        """
        # Use ML model if available
        if self.ml_model is not None:
            try:
                # Get prediction and probability
                prediction = self.ml_model.predict([text])[0]
                proba = self.ml_model.predict_proba([text])[0]
                confidence = max(proba)

                logger.debug(f"ML classification: {prediction} with confidence {confidence:.2f}")

                # If confidence is high enough, return ML result
                if confidence > 0.7:
                    return prediction, confidence

                # Otherwise, combine with rule-based method
                rule_classification, rule_confidence = self._rule_based_classification(text, language, domain)

                # Weight ML higher than rule-based
                if prediction == rule_classification:
                    return prediction, max(confidence, rule_confidence)
                else:
                    # If they disagree, take the one with higher confidence
                    if confidence >= rule_confidence:
                        return prediction, confidence
                    else:
                        return rule_classification, rule_confidence

            except Exception as e:
                logger.warning(f"Error in ML classification: {e}")

        # Rule-based classification
        return self._rule_based_classification(text, language, domain)

    def _rule_based_classification(self, text: str, language: str, domain: str = None) -> Tuple[str, float]:
        """
        Perform rule-based classification of text as theoretical or practical.

        Args:
            text: Text to classify
            language: Language code ('en' or 'ru')
            domain: Domain of the text (mathematics, programming, physics)

        Returns:
            Tuple of (classification, confidence)
        """
        lang_key = "ru" if language == "ru" else "en"
        text = text.lower()

        # Calculate scores
        theoretical_score = self._calculate_theoretical_score(text, lang_key, domain)
        practical_score = self._calculate_practical_score(text, lang_key, domain)

        # Get linguistic features for more nuanced scoring
        linguistic_features = self._extract_linguistic_features(text, lang_key)

        # Adjust scores based on linguistic features
        theoretical_score += linguistic_features.get("theoretical_weight", 0)
        practical_score += linguistic_features.get("practical_weight", 0)

        # Get semantic features
        semantic_features = self._extract_semantic_features(text, lang_key)

        # Adjust scores based on semantic features
        theoretical_score += semantic_features.get("theoretical_weight", 0)
        practical_score += semantic_features.get("practical_weight", 0)

        # Special case: code blocks almost always indicate practical content
        if "```" in text or "<code>" in text:
            practical_score += 5

        # Special case: direct references to proofs and definitions are strong theoretical indicators
        if lang_key == "en":
            if re.search(r'\bproof\b|\btheorem\b|\bdefinition\b', text):
                theoretical_score += 3
        else:
            if re.search(r'\bдоказательство\b|\bтеорема\b|\bопределение\b', text):
                theoretical_score += 3

        # Calculate confidence based on score difference
        score_diff = abs(theoretical_score - practical_score)
        total_score = theoretical_score + practical_score

        # Normalize confidence to [0, 1]
        confidence = min(score_diff / max(total_score, 1), 1.0) * 0.8 + 0.2  # 0.2 is base confidence

        # Determine classification
        if theoretical_score > practical_score:
            return "theoretical", confidence
        elif practical_score > theoretical_score:
            return "practical", confidence
        else:
            # If scores are equal, default to "mixed" with low confidence
            return "mixed", 0.5

    def _calculate_theoretical_score(self, text: str, lang_key: str, domain: str = None) -> float:
        """Calculate theoretical score based on pattern matching."""
        score = 0

        # Apply general theoretical patterns
        for pattern in self.theoretical_patterns.get(lang_key, []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            score += len(matches)

        # Apply domain-specific theoretical patterns if domain is provided
        if domain and domain in self.domain_patterns:
            domain_theoretical_patterns = self.domain_patterns[domain]["theoretical"].get(lang_key, [])
            for pattern in domain_theoretical_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches) * 1.5  # Domain-specific patterns have higher weight

        return score

    def _calculate_practical_score(self, text: str, lang_key: str, domain: str = None) -> float:
        """Calculate practical score based on pattern matching."""
        score = 0

        # Apply general practical patterns
        for pattern in self.practical_patterns.get(lang_key, []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            score += len(matches)

        # Apply domain-specific practical patterns if domain is provided
        if domain and domain in self.domain_patterns:
            domain_practical_patterns = self.domain_patterns[domain]["practical"].get(lang_key, [])
            for pattern in domain_practical_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches) * 1.5  # Domain-specific patterns have higher weight

        return score

    def _extract_linguistic_features(self, text: str, lang_key: str) -> Dict[str, float]:
        """Extract linguistic features for theory/practice classification."""
        features = {
            "theoretical_weight": 0,
            "practical_weight": 0
        }

        # Check for mathematical notation (often theoretical)
        if re.search(r'\$.*\$|\\\(.*\\\)|\\\[.*\\\]', text):
            features["theoretical_weight"] += 1

        # Check for concrete numbers (often practical)
        number_matches = re.findall(r'\b\d+(\.\d+)?\b', text)
        if len(number_matches) > 3:  # If multiple numbers are present
            features["practical_weight"] += 1

        # Check sentence structure
        if lang_key == "en":
            # Imperative sentences (often practical)
            imperative_matches = re.findall(r'\b(Calculate|Find|Solve|Compute|Determine)\b', text)
            features["practical_weight"] += len(imperative_matches) * 0.5

            # Passive voice (often theoretical)
            passive_matches = re.findall(r'\b(is defined|is called|is known|is considered|is shown)\b', text)
            features["theoretical_weight"] += len(passive_matches) * 0.5

        else:  # Russian
            # Imperative sentences (often practical)
            imperative_matches = re.findall(r'\b(Вычислите|Найдите|Решите|Определите)\b', text)
            features["practical_weight"] += len(imperative_matches) * 0.5

            # Passive voice (often theoretical)
            passive_matches = re.findall(r'\b(определяется|называется|известен|рассматривается|показывается)\b', text)
            features["theoretical_weight"] += len(passive_matches) * 0.5

        return features

    def _extract_semantic_features(self, text: str, lang_key: str) -> Dict[str, float]:
        """Extract semantic features for theory/practice classification."""
        features = {
            "theoretical_weight": 0,
            "practical_weight": 0
        }

        # Check theoretical semantic categories
        for category, terms in self.theoretical_semantic.get(lang_key, {}).items():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    features["theoretical_weight"] += 0.3

        # Check practical semantic categories
        for category, terms in self.practical_semantic.get(lang_key, {}).items():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    features["practical_weight"] += 0.3

        return features

    def classify_segment(self, segment: Dict[str, Any], domain: str = None) -> Tuple[str, float]:
        """
        Classify a transcript segment as theoretical or practical.

        Args:
            segment: Transcript segment dictionary
            domain: Domain of the segment

        Returns:
            Tuple of (classification, confidence)
        """
        text = segment.get("text", "")
        language = segment.get("language", "en")

        # Use NLP data if available
        nlp_data = segment.get("nlp_data", {})
        sentence_type = nlp_data.get("sentence_type", "")

        # Some sentence types are inherently theoretical or practical
        if sentence_type == "definition" or sentence_type == "proof":
            # Still do classification but with a bias
            classification, confidence = self.classify_text(text, language, domain)
            if classification == "mixed":
                return "theoretical", max(confidence, 0.7)
            return classification, confidence

        elif sentence_type == "problem_statement" or sentence_type == "solution":
            # Still do classification but with a bias
            classification, confidence = self.classify_text(text, language, domain)
            if classification == "mixed":
                return "practical", max(confidence, 0.7)
            return classification, confidence

        # Check for formulas and code snippets
        formulas = nlp_data.get("formulas", [])
        code_snippets = nlp_data.get("code_snippets", [])

        # If segment has only formulas, it's more likely to be theoretical
        if formulas and not code_snippets:
            classification, confidence = self.classify_text(text, language, domain)
            if classification == "mixed":
                return "theoretical", max(confidence, 0.6)
            return classification, confidence

        # If segment has code snippets, it's more likely to be practical
        if code_snippets:
            classification, confidence = self.classify_text(text, language, domain)
            if classification == "mixed":
                return "practical", max(confidence, 0.7)
            return classification, confidence

        # Otherwise, do standard classification
        return self.classify_text(text, language, domain)

    def classify_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an entire transcript and provide theory/practice statistics.

        Args:
            transcript: Transcript dictionary

        Returns:
            Dictionary with classification results
        """
        language = transcript.get("language", "en")
        domain = transcript.get("domain", None)
        segments = transcript.get("segments", [])

        if not segments:
            logger.warning("Empty transcript provided for classification")
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "theoretical_segments": 0,
                "practical_segments": 0,
                "mixed_segments": 0,
                "theory_practice_ratio": 0.5
            }

        # Classify each segment
        theoretical_segments = 0
        practical_segments = 0
        mixed_segments = 0

        for segment in segments:
            classification, confidence = self.classify_segment(segment, domain)

            # Update segment with classification
            segment["content_type"] = classification
            segment["classification_confidence"] = confidence

            # Update counts
            if classification == "theoretical":
                theoretical_segments += 1
            elif classification == "practical":
                practical_segments += 1
            else:
                mixed_segments += 1

        # Calculate overall statistics
        total_segments = theoretical_segments + practical_segments + mixed_segments

        # Theory-practice ratio (0 = all practical, 1 = all theoretical)
        if total_segments > 0:
            theory_practice_ratio = (theoretical_segments + mixed_segments * 0.5) / total_segments
        else:
            theory_practice_ratio = 0.5

        # Determine overall classification
        if theory_practice_ratio > 0.7:
            overall_classification = "theoretical"
            confidence = theory_practice_ratio
        elif theory_practice_ratio < 0.3:
            overall_classification = "practical"
            confidence = 1 - theory_practice_ratio
        else:
            overall_classification = "mixed"
            confidence = 1 - abs(theory_practice_ratio - 0.5) * 2  # 0.5 = max confidence for mixed

        # Compile results
        results = {
            "classification": overall_classification,
            "confidence": confidence,
            "theoretical_segments": theoretical_segments,
            "practical_segments": practical_segments,
            "mixed_segments": mixed_segments,
            "theory_practice_ratio": theory_practice_ratio
        }

        logger.info(f"Classified transcript: {overall_classification} (ratio: {theory_practice_ratio:.2f})")

        return results

    def extract_theory_practice_patterns(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract theory-practice sequence patterns from a transcript.

        Args:
            transcript: Transcript dictionary

        Returns:
            Dictionary with theory-practice sequence patterns
        """
        segments = transcript.get("segments", [])
        domain = transcript.get("domain", None)

        if not segments:
            return {
                "theory_to_practice_sequences": [],
                "practice_to_theory_sequences": [],
                "theory_practice_alternations": 0,
                "max_theory_sequence": 0,
                "max_practice_sequence": 0
            }

        # Extract segment classifications
        segment_types = []
        for segment in segments:
            content_type = segment.get("content_type", "mixed")
            if content_type == "mixed":
                continue  # Skip mixed segments for pattern analysis
            segment_types.append(content_type)

        if not segment_types:
            return {
                "theory_to_practice_sequences": [],
                "practice_to_theory_sequences": [],
                "theory_practice_alternations": 0,
                "max_theory_sequence": 0,
                "max_practice_sequence": 0
            }

        # Find theory-to-practice transitions
        theory_to_practice = []
        practice_to_theory = []
        alternations = 0

        for i in range(1, len(segment_types)):
            if segment_types[i-1] == "theoretical" and segment_types[i] == "practical":
                theory_to_practice.append(i)
                alternations += 1
            elif segment_types[i-1] == "practical" and segment_types[i] == "theoretical":
                practice_to_theory.append(i)
                alternations += 1

        # Find longest sequences
        max_theory_sequence = 0
        max_practice_sequence = 0
        current_theory_sequence = 0
        current_practice_sequence = 0

        for segment_type in segment_types:
            if segment_type == "theoretical":
                current_theory_sequence += 1
                current_practice_sequence = 0
                max_theory_sequence = max(max_theory_sequence, current_theory_sequence)
            else:
                current_practice_sequence += 1
                current_theory_sequence = 0
                max_practice_sequence = max(max_practice_sequence, current_practice_sequence)

        # Analyze theory-to-practice transitions
        theory_to_practice_sequences = []
        for transition_index in theory_to_practice:
            # Get 1 segment before and 2 segments after transition
            start_idx = max(0, transition_index - 1)
            end_idx = min(len(segments), transition_index + 3)

            sequence = {
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "segments": segments[start_idx:end_idx],
                "pattern": "theory_to_practice"
            }

            # Check if it's a domain-specific pattern
            if domain == "mathematics":
                if any(self._is_math_definition(seg) for seg in sequence["segments"][:2]) and \
                   any(self._is_math_example(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "definition_to_example"
                elif any(self._is_math_theorem(seg) for seg in sequence["segments"][:2]) and \
                     any(self._is_math_application(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "theorem_to_application"
                else:
                    sequence["pattern_type"] = "general_theory_to_practice"

            elif domain == "programming":
                if any(self._is_programming_concept(seg) for seg in sequence["segments"][:2]) and \
                   any(self._is_programming_implementation(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "concept_to_implementation"
                elif any(self._is_programming_algorithm(seg) for seg in sequence["segments"][:2]) and \
                     any(self._is_programming_code(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "algorithm_to_code"
                else:
                    sequence["pattern_type"] = "general_theory_to_practice"

            elif domain == "physics":
                if any(self._is_physics_law(seg) for seg in sequence["segments"][:2]) and \
                   any(self._is_physics_problem(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "law_to_problem"
                elif any(self._is_physics_concept(seg) for seg in sequence["segments"][:2]) and \
                     any(self._is_physics_experiment(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "concept_to_experiment"
                else:
                    sequence["pattern_type"] = "general_theory_to_practice"

            else:
                sequence["pattern_type"] = "general_theory_to_practice"

            theory_to_practice_sequences.append(sequence)

        # Analyze practice-to-theory transitions
        practice_to_theory_sequences = []
        for transition_index in practice_to_theory:
            # Get 1 segment before and 2 segments after transition
            start_idx = max(0, transition_index - 1)
            end_idx = min(len(segments), transition_index + 3)

            sequence = {
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "segments": segments[start_idx:end_idx],
                "pattern": "practice_to_theory"
            }

            # Check if it's a domain-specific pattern
            if domain == "mathematics":
                if any(self._is_math_example(seg) for seg in sequence["segments"][:2]) and \
                   any(self._is_math_generalization(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "example_to_generalization"
                else:
                    sequence["pattern_type"] = "general_practice_to_theory"

            elif domain == "programming":
                if any(self._is_programming_code(seg) for seg in sequence["segments"][:2]) and \
                   any(self._is_programming_explanation(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "code_to_explanation"
                else:
                    sequence["pattern_type"] = "general_practice_to_theory"

            elif domain == "physics":
                if any(self._is_physics_experiment(seg) for seg in sequence["segments"][:2]) and \
                   any(self._is_physics_theory(seg) for seg in sequence["segments"][2:]):
                    sequence["pattern_type"] = "experiment_to_theory"
                else:
                    sequence["pattern_type"] = "general_practice_to_theory"

            else:
                sequence["pattern_type"] = "general_practice_to_theory"

            practice_to_theory_sequences.append(sequence)

        # Compile results
        patterns = {
            "theory_to_practice_sequences": theory_to_practice_sequences,
            "practice_to_theory_sequences": practice_to_theory_sequences,
            "theory_practice_alternations": alternations,
            "max_theory_sequence": max_theory_sequence,
            "max_practice_sequence": max_practice_sequence
        }

        return patterns

    def _is_math_definition(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is a mathematical definition."""
        text = segment.get("text", "").lower()
        nlp_data = segment.get("nlp_data", {})

        if nlp_data.get("sentence_type") == "definition":
            return True

        language = segment.get("language", "en")
        if language == "en":
            return bool(re.search(r'\bdefinition\b|\bis defined\b|\bmeans\b', text))
        else:
            return bool(re.search(r'\bопределение\b|\bопределяется\b|\bозначает\b', text))

    def _is_math_theorem(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is a mathematical theorem."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\btheorem\b|\blemma\b|\bcorollary\b', text))
        else:
            return bool(re.search(r'\bтеорема\b|\bлемма\b|\bследствие\b', text))

    def _is_math_example(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is a mathematical example."""
        text = segment.get("text", "").lower()
        nlp_data = segment.get("nlp_data", {})

        if nlp_data.get("sentence_type") == "example":
            return True

        language = segment.get("language", "en")
        if language == "en":
            return bool(re.search(r'\bexample\b|\binstance\b|\bconsider\b', text))
        else:
            return bool(re.search(r'\bпример\b|\bслучай\b|\bрассмотрим\b', text))

    def _is_math_application(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is a mathematical application."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\bapplication\b|\bapply\b|\buse\b|\bsolve\b', text))
        else:
            return bool(re.search(r'\bприменение\b|\bприменять\b|\bиспользовать\b|\bрешать\b', text))

    def _is_math_generalization(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is a mathematical generalization."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\bgeneral\b|\bgeneralize\b|\babstract\b', text))
        else:
            return bool(re.search(r'\bобщий\b|\bобобщить\b|\bабстрактный\b', text))

    def _is_programming_concept(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about programming concepts."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\bconcept\b|\bparadigm\b|\bprinciple\b', text))
        else:
            return bool(re.search(r'\bконцепция\b|\bпарадигма\b|\bпринцип\b', text))

    def _is_programming_algorithm(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about programming algorithms."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\balgorithm\b|\bcomplexity\b|\bpseudocode\b', text))
        else:
            return bool(re.search(r'\bалгоритм\b|\bсложность\b|\bпсевдокод\b', text))

    def _is_programming_implementation(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about programming implementation."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\bimplementation\b|\bimplementing\b|\bcoding\b', text))
        else:
            return bool(re.search(r'\bреализация\b|\bреализовать\b|\bкодирование\b', text))

    def _is_programming_code(self, segment: Dict[str, Any]) -> bool:
        """Check if segment contains programming code."""
        nlp_data = segment.get("nlp_data", {})
        code_snippets = nlp_data.get("code_snippets", [])

        if code_snippets:
            return True

        text = segment.get("text", "").lower()
        return "```" in text or "<code>" in text

    def _is_programming_explanation(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is a programming explanation."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\bexplain\b|\bunderstand\b|\bmeans\b|\bworks\b', text))
        else:
            return bool(re.search(r'\bобъяснять\b|\bпонимать\b|\bозначает\b|\bработает\b', text))

    def _is_physics_law(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about physics laws."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\blaw\b|\bprinciple\b|\btheory\b', text))
        else:
            return bool(re.search(r'\bзакон\b|\bпринцип\b|\bтеория\b', text))

    def _is_physics_concept(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about physics concepts."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\bconcept\b|\bphenomenon\b|\bmodel\b', text))
        else:
            return bool(re.search(r'\bконцепция\b|\bявление\b|\bмодель\b', text))

    def _is_physics_problem(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about physics problems."""
        text = segment.get("text", "").lower()
        nlp_data = segment.get("nlp_data", {})

        if nlp_data.get("sentence_type") == "problem_statement":
            return True

        language = segment.get("language", "en")
        if language == "en":
            return bool(re.search(r'\bproblem\b|\bcalculate\b|\bfind\b|\bsolve\b', text))
        else:
            return bool(re.search(r'\bзадача\b|\bвычислить\b|\bнайти\b|\bрешить\b', text))

    def _is_physics_experiment(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about physics experiments."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\bexperiment\b|\bmeasurement\b|\blab\b|\bobserve\b', text))
        else:
            return bool(re.search(r'\bэксперимент\b|\bизмерение\b|\bлаборатор\b|\bнаблюдать\b', text))

    def _is_physics_theory(self, segment: Dict[str, Any]) -> bool:
        """Check if segment is about physics theory."""
        text = segment.get("text", "").lower()
        language = segment.get("language", "en")

        if language == "en":
            return bool(re.search(r'\btheory\b|\btheoretical\b|\bhypothesis\b', text))
        else:
            return bool(re.search(r'\bтеория\b|\bтеоретический\b|\bгипотеза\b', text))
