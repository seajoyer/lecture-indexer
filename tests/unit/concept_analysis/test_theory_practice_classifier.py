"""
Unit tests for the Theory Practice Classifier component.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import os
import re

from concept_analysis.relevance_analyzer.python.theory_practice_classifier import TheoryPracticeClassifier

# Test data
THEORETICAL_TEXT_EN = "In theoretical calculus, we define the derivative as the limit of the difference quotient as the interval approaches zero. This concept forms the foundation of differential calculus."

PRACTICAL_TEXT_EN = "Let's solve this problem: Find the derivative of f(x) = x^2 + 3x + 2. We take the derivative of each term: f'(x) = 2x + 3. Now we can evaluate at specific points."

THEORETICAL_TEXT_RU = "В теоретическом исчислении мы определяем производную как предел разностного отношения, когда интервал приближается к нулю. Эта концепция формирует основу дифференциального исчисления."

PRACTICAL_TEXT_RU = "Давайте решим эту задачу: Найдите производную f(x) = x^2 + 3x + 2. Мы берем производную каждого члена: f'(x) = 2x + 3. Теперь мы можем вычислить в конкретных точках."

MIXED_TEXT = "While the concept of a derivative is defined theoretically as a limit, we can apply it to solve practical problems like finding the slope of a tangent line."

CODE_TEXT = "Here's a Python implementation: ```python\ndef calculate_derivative(x):\n    return 2*x + 3\n```"

MATH_TRANSCRIPT = {
    "language": "en",
    "domain": "mathematics",
    "segments": [
        {
            "id": "s1",
            "text": "Today we'll discuss the theoretical foundations of calculus.",
            "nlp_data": {
                "sentence_type": "explanation",
                "formulas": [],
                "code_snippets": []
            }
        },
        {
            "id": "s2",
            "text": "A derivative is defined as the limit of the difference quotient.",
            "nlp_data": {
                "sentence_type": "definition",
                "formulas": [{"text": "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}"}],
                "code_snippets": []
            }
        },
        {
            "id": "s3",
            "text": "Now, let's solve some problems. Find the derivative of f(x) = x^2.",
            "nlp_data": {
                "sentence_type": "problem_statement",
                "formulas": [{"text": "f(x) = x^2"}],
                "code_snippets": []
            }
        },
        {
            "id": "s4",
            "text": "The derivative is f'(x) = 2x. We can verify this with examples.",
            "nlp_data": {
                "sentence_type": "solution",
                "formulas": [{"text": "f'(x) = 2x"}],
                "code_snippets": []
            }
        }
    ]
}

@pytest.fixture
def theory_practice_classifier():
    """Create a Theory Practice Classifier instance."""
    return TheoryPracticeClassifier()

class TestTheoryPracticeClassifier:
    """Test the Theory Practice Classifier component."""

    def test_classify_text_theoretical_english(self, theory_practice_classifier):
        """Test classifying theoretical English text."""
        classification, confidence = theory_practice_classifier.classify_text(THEORETICAL_TEXT_EN, "en")

        assert classification == "theoretical"
        assert confidence > 0.5

    def test_classify_text_practical_english(self, theory_practice_classifier):
        """Test classifying practical English text."""
        classification, confidence = theory_practice_classifier.classify_text(PRACTICAL_TEXT_EN, "en")

        assert classification == "practical"
        assert confidence > 0.5

    def test_classify_text_theoretical_russian(self, theory_practice_classifier):
        """Test classifying theoretical Russian text."""
        classification, confidence = theory_practice_classifier.classify_text(THEORETICAL_TEXT_RU, "ru")

        assert classification == "theoretical"
        assert confidence > 0.5

    def test_classify_text_practical_russian(self, theory_practice_classifier):
        """Test classifying practical Russian text."""
        classification, confidence = theory_practice_classifier.classify_text(PRACTICAL_TEXT_RU, "ru")

        assert classification == "practical"
        assert confidence > 0.5

    def test_classify_text_mixed(self, theory_practice_classifier):
        """Test classifying mixed content text."""
        classification, confidence = theory_practice_classifier.classify_text(MIXED_TEXT, "en")

        # Could be classified as either, but often "mixed"
        assert classification in ("theoretical", "practical", "mixed")

        # If classified as mixed, confidence should be low
        if classification == "mixed":
            assert confidence <= 0.6

    def test_classify_text_with_code(self, theory_practice_classifier):
        """Test classifying text with code."""
        classification, confidence = theory_practice_classifier.classify_text(CODE_TEXT, "en")

        assert classification == "practical"
        assert confidence > 0.6

    def test_rule_based_classification(self, theory_practice_classifier):
        """Test the rule-based classification method."""
        classification, confidence = theory_practice_classifier._rule_based_classification(
            THEORETICAL_TEXT_EN, "en"
        )

        assert classification == "theoretical"
        assert confidence > 0.5

        classification, confidence = theory_practice_classifier._rule_based_classification(
            PRACTICAL_TEXT_EN, "en"
        )

        assert classification == "practical"
        assert confidence > 0.5

    def test_calculate_theoretical_score(self, theory_practice_classifier):
        """Test calculating theoretical score."""
        score = theory_practice_classifier._calculate_theoretical_score(
            THEORETICAL_TEXT_EN, "en", "mathematics"
        )

        assert score > 0

        # Theoretical text should score higher than practical text
        theoretical_score = theory_practice_classifier._calculate_theoretical_score(
            THEORETICAL_TEXT_EN, "en"
        )
        practical_score = theory_practice_classifier._calculate_theoretical_score(
            PRACTICAL_TEXT_EN, "en"
        )

        assert theoretical_score > practical_score

    def test_calculate_practical_score(self, theory_practice_classifier):
        """Test calculating practical score."""
        score = theory_practice_classifier._calculate_practical_score(
            PRACTICAL_TEXT_EN, "en", "mathematics"
        )

        assert score > 0

        # Practical text should score higher than theoretical text
        theoretical_score = theory_practice_classifier._calculate_practical_score(
            THEORETICAL_TEXT_EN, "en"
        )
        practical_score = theory_practice_classifier._calculate_practical_score(
            PRACTICAL_TEXT_EN, "en"
        )

        assert practical_score > theoretical_score

    def test_extract_linguistic_features(self, theory_practice_classifier):
        """Test extracting linguistic features."""
        features = theory_practice_classifier._extract_linguistic_features(
            "The formula $f(x) = x^2$ is defined in theoretical calculus.", "en"
        )

        assert "theoretical_weight" in features
        assert "practical_weight" in features
        assert features["theoretical_weight"] > 0

        # Text with numbers should get practical weight
        features = theory_practice_classifier._extract_linguistic_features(
            "Let's compute: 5 + 3 = 8, then 8 * 2 = 16, and finally 16 / 4 = 4.", "en"
        )

        assert features["practical_weight"] > 0

    def test_extract_semantic_features(self, theory_practice_classifier):
        """Test extracting semantic features."""
        features = theory_practice_classifier._extract_semantic_features(
            "In theory, we understand concepts through abstract definitions.", "en"
        )

        assert "theoretical_weight" in features
        assert "practical_weight" in features
        assert features["theoretical_weight"] > 0

        features = theory_practice_classifier._extract_semantic_features(
            "In practice, we apply methods to solve specific problems.", "en"
        )

        assert features["practical_weight"] > 0

    def test_train_model(self, theory_practice_classifier):
        """Test training the classification model."""
        training_data = [
            {"text": "In theoretical mathematics, we define concepts abstractly.", "classification": "theoretical"},
            {"text": "Theoretical concepts form the foundation of mathematics.", "classification": "theoretical"},
            {"text": "Let's solve this equation: 2x + 3 = 7", "classification": "practical"},
            {"text": "In practice, we can implement this algorithm in code.", "classification": "practical"}
        ]

        # Mock the ML components
        with patch('sklearn.feature_extraction.text.TfidfVectorizer'), \
             patch('sklearn.naive_bayes.MultinomialNB'), \
             patch('sklearn.linear_model.LogisticRegression'), \
             patch('sklearn.pipeline.Pipeline'), \
             patch('sklearn.ensemble.VotingClassifier'):

            theory_practice_classifier.train_model(training_data)

            assert theory_practice_classifier.ml_model is not None

    @patch('sklearn.pipeline.Pipeline')
    def test_classify_text_with_ml_model(self, mock_pipeline, theory_practice_classifier):
        """Test classifying text with ML model."""
        # Set up mock ML model
        mock_model = MagicMock()
        mock_model.predict.return_value = ["theoretical"]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]  # [practical, theoretical]

        theory_practice_classifier.ml_model = mock_model

        classification, confidence = theory_practice_classifier.classify_text(THEORETICAL_TEXT_EN, "en")

        assert mock_model.predict.called
        assert mock_model.predict_proba.called

        assert classification == "theoretical"
        assert confidence == 0.8

    def test_classify_segment(self, theory_practice_classifier):
        """Test classifying a transcript segment."""
        # Create a definition segment (inherently theoretical)
        definition_segment = {
            "text": "A derivative is defined as the limit of the difference quotient.",
            "language": "en",
            "nlp_data": {
                "sentence_type": "definition",
                "formulas": [{"text": "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}"}],
                "code_snippets": []
            }
        }

        # Create a problem segment (inherently practical)
        problem_segment = {
            "text": "Find the derivative of f(x) = x^2 + 3x + 2.",
            "language": "en",
            "nlp_data": {
                "sentence_type": "problem_statement",
                "formulas": [{"text": "f(x) = x^2 + 3x + 2"}],
                "code_snippets": []
            }
        }

        # Create a code segment (practical)
        code_segment = {
            "text": "Here's how to implement it in Python.",
            "language": "en",
            "nlp_data": {
                "sentence_type": "explanation",
                "formulas": [],
                "code_snippets": [{"text": "def f(x): return x**2 + 3*x + 2", "language": "python"}]
            }
        }

        # Mock the classify_text method to avoid real classification
        with patch.object(theory_practice_classifier, 'classify_text', return_value=("mixed", 0.5)):
            # A definition should be classified as theoretical
            classification, confidence = theory_practice_classifier.classify_segment(definition_segment)
            assert classification == "theoretical"

            # A problem should be classified as practical
            classification, confidence = theory_practice_classifier.classify_segment(problem_segment)
            assert classification == "practical"

            # A segment with code should be classified as practical
            classification, confidence = theory_practice_classifier.classify_segment(code_segment)
            assert classification == "practical"

    def test_classify_transcript(self, theory_practice_classifier):
        """Test classifying a full transcript."""
        # Mock the classify_segment method
        with patch.object(theory_practice_classifier, 'classify_segment') as mock_classify:
            # Set up the mock to return different classifications
            mock_classify.side_effect = [
                ("theoretical", 0.8),
                ("theoretical", 0.9),
                ("practical", 0.7),
                ("practical", 0.8)
            ]

            result = theory_practice_classifier.classify_transcript(MATH_TRANSCRIPT)

            assert "classification" in result
            assert "confidence" in result
            assert "theoretical_segments" in result
            assert "practical_segments" in result
            assert "mixed_segments" in result
            assert "theory_practice_ratio" in result

            assert result["theoretical_segments"] == 2
            assert result["practical_segments"] == 2

            # Theory/practice ratio should be around 0.5 (balanced)
            assert 0.4 <= result["theory_practice_ratio"] <= 0.6

    def test_extract_theory_practice_patterns(self, theory_practice_classifier):
        """Test extracting theory-practice patterns from a transcript."""
        # Mock the relevant test methods
        with patch.object(theory_practice_classifier, '_is_math_definition', return_value=True), \
             patch.object(theory_practice_classifier, '_is_math_example', return_value=True), \
             patch.object(theory_practice_classifier, '_is_programming_concept', return_value=False), \
             patch.object(theory_practice_classifier, '_is_physics_law', return_value=False):

            # Create a transcript with segments containing content_type
            transcript = {
                "language": "en",
                "domain": "mathematics",
                "segments": [
                    {"id": "s1", "content_type": "theoretical", "text": "Definition of a derivative."},
                    {"id": "s2", "content_type": "theoretical", "text": "Theoretical properties of derivatives."},
                    {"id": "s3", "content_type": "practical", "text": "Let's compute some derivatives."},
                    {"id": "s4", "content_type": "practical", "text": "Here's another example."},
                    {"id": "s5", "content_type": "theoretical", "text": "Now back to theory."}
                ]
            }

            patterns = theory_practice_classifier.extract_theory_practice_patterns(transcript)

            assert "theory_to_practice_sequences" in patterns
            assert "practice_to_theory_sequences" in patterns
            assert "theory_practice_alternations" in patterns
            assert "max_theory_sequence" in patterns
            assert "max_practice_sequence" in patterns

            # Should detect one theory-to-practice transition (s2 to s3)
            assert len(patterns["theory_to_practice_sequences"]) == 1

            # Should detect one practice-to-theory transition (s4 to s5)
            assert len(patterns["practice_to_theory_sequences"]) == 1

            # Should detect 2 alternations total
            assert patterns["theory_practice_alternations"] == 2

            # Check the detected pattern type
            assert patterns["theory_to_practice_sequences"][0]["pattern_type"] == "definition_to_example"

    def test_is_math_definition(self, theory_practice_classifier):
        """Test detecting math definitions."""
        segment = {
            "text": "A derivative is defined as the limit of the difference quotient.",
            "language": "en",
            "nlp_data": {"sentence_type": "definition"}
        }

        assert theory_practice_classifier._is_math_definition(segment) is True

        segment["nlp_data"]["sentence_type"] = "explanation"
        assert theory_practice_classifier._is_math_definition(segment) is True  # Still true because of the text

        segment["text"] = "Let's solve a problem."
        assert theory_practice_classifier._is_math_definition(segment) is False

    def test_is_programming_code(self, theory_practice_classifier):
        """Test detecting programming code."""
        segment = {
            "text": "Here's some code to implement it.",
            "nlp_data": {
                "code_snippets": [{"text": "def example(): pass", "language": "python"}]
            }
        }

        assert theory_practice_classifier._is_programming_code(segment) is True

        segment["nlp_data"]["code_snippets"] = []
        segment["text"] = "Here's some code: ```python\ndef example(): pass\n```"
        assert theory_practice_classifier._is_programming_code(segment) is True

        segment["text"] = "No code here."
        assert theory_practice_classifier._is_programming_code(segment) is False

    def test_is_physics_experiment(self, theory_practice_classifier):
        """Test detecting physics experiments."""
        segment = {
            "text": "In this experiment, we measure the acceleration due to gravity.",
            "language": "en"
        }

        assert theory_practice_classifier._is_physics_experiment(segment) is True

        segment["text"] = "Let's discuss the theoretical concept of gravity."
        assert theory_practice_classifier._is_physics_experiment(segment) is False
