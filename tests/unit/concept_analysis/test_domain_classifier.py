"""
Unit tests for the Domain Classifier component.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import os
import numpy as np

from concept_analysis.concept_extractor.python.domain_concept_extractor import DomainClassifier

# Test data
MATH_TEXT = "In this mathematics lecture, we discuss calculus, derivatives, integrals, and theorem proofs. We explore algebraic structures and geometric concepts."

PROGRAMMING_TEXT = "This programming tutorial covers Python coding, data structures, algorithms, and software development. We'll implement classes, functions, and methods."

PHYSICS_TEXT = "In our physics class, we study mechanics, dynamics, forces, and energy. We'll explore Newton's laws, electromagnetism, and thermodynamics."

MIXED_TEXT = "This lecture covers both mathematical concepts and programming implementations. We'll derive formulas and write code to solve problems."

MATH_TRANSCRIPT = {
    "language": "en",
    "segments": [
        {
            "id": "segment1",
            "text": "Welcome to our mathematics lecture on calculus.",
            "content_type": "theoretical",
            "nlp_data": {
                "formulas": [{"text": "f(x) = x^2", "start": 0, "end": 10}],
                "code_snippets": []
            }
        },
        {
            "id": "segment2",
            "text": "Let's discuss the definition of a derivative.",
            "content_type": "theoretical",
            "nlp_data": {
                "formulas": [{"text": "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}", "start": 0, "end": 10}],
                "code_snippets": []
            }
        },
        {
            "id": "segment3",
            "text": "Now let's solve some calculus problems.",
            "content_type": "practical",
            "nlp_data": {
                "formulas": [{"text": "\\int x^2 dx = \\frac{x^3}{3} + C", "start": 0, "end": 10}],
                "code_snippets": []
            }
        }
    ]
}

PROGRAMMING_TRANSCRIPT = {
    "language": "en",
    "segments": [
        {
            "id": "segment1",
            "text": "Welcome to our programming tutorial on Python.",
            "content_type": "theoretical",
            "nlp_data": {
                "formulas": [],
                "code_snippets": []
            }
        },
        {
            "id": "segment2",
            "text": "Object-oriented programming involves classes and inheritance.",
            "content_type": "theoretical",
            "nlp_data": {
                "formulas": [],
                "code_snippets": []
            }
        },
        {
            "id": "segment3",
            "text": "Let's write a Python function to calculate factorials.",
            "content_type": "practical",
            "nlp_data": {
                "formulas": [],
                "code_snippets": [{"text": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", "language": "python", "start": 0, "end": 10}]
            }
        }
    ]
}

@pytest.fixture
def domain_classifier():
    """Create a Domain Classifier instance."""
    return DomainClassifier()

class TestDomainClassifier:
    """Test the Domain Classifier component."""

    def test_classify_text_mathematics(self, domain_classifier):
        """Test classifying mathematics text."""
        domain, confidence = domain_classifier.classify_text(MATH_TEXT, "en")

        assert domain == "mathematics"
        assert confidence > 0.5

    def test_classify_text_programming(self, domain_classifier):
        """Test classifying programming text."""
        domain, confidence = domain_classifier.classify_text(PROGRAMMING_TEXT, "en")

        assert domain == "programming"
        assert confidence > 0.5

    def test_classify_text_physics(self, domain_classifier):
        """Test classifying physics text."""
        domain, confidence = domain_classifier.classify_text(PHYSICS_TEXT, "en")

        assert domain == "physics"
        assert confidence > 0.5

    def test_classify_text_mixed(self, domain_classifier):
        """Test classifying mixed content text."""
        domain, confidence = domain_classifier.classify_text(MIXED_TEXT, "en")

        # This could be classified as either mathematics or programming
        # depending on the implementation, but should have lower confidence
        assert domain in ("mathematics", "programming")
        assert confidence <= 0.7  # Confidence should be lower for mixed content

    def test_rule_based_classification(self, domain_classifier):
        """Test rule-based classification method."""
        domain, confidence = domain_classifier._rule_based_classification(MATH_TEXT, "en")

        assert domain == "mathematics"
        assert confidence > 0.5

        domain, confidence = domain_classifier._rule_based_classification(PROGRAMMING_TEXT, "en")

        assert domain == "programming"
        assert confidence > 0.5

    def test_train_model(self, domain_classifier):
        """Test training the classification model."""
        training_data = [
            {"text": "This is a calculus lecture about derivatives.", "domain": "mathematics"},
            {"text": "We'll discuss integrals and differential equations.", "domain": "mathematics"},
            {"text": "Let's write a Python program to sort an array.", "domain": "programming"},
            {"text": "Object-oriented programming with classes and methods.", "domain": "programming"},
            {"text": "We'll study Newton's laws and kinematics.", "domain": "physics"},
            {"text": "Thermodynamics and energy conservation principles.", "domain": "physics"}
        ]

        # Mock the sklearn components
        with patch('sklearn.feature_extraction.text.TfidfVectorizer'), \
             patch('sklearn.linear_model.LogisticRegression'), \
             patch('sklearn.pipeline.Pipeline'):

            # Train the model
            domain_classifier.train_model(training_data)

            # Check that the model was created
            assert domain_classifier.ml_model is not None

    def test_classify_transcript(self, domain_classifier):
        """Test classifying a full transcript."""
        # Mock the enhanced classification method
        with patch.object(domain_classifier, '_enhanced_classification', return_value=("mathematics", 0.9)):
            domain, confidence = domain_classifier.classify_transcript(MATH_TRANSCRIPT)

            assert domain == "mathematics"
            assert confidence == 0.9

    @patch('sklearn.pipeline.Pipeline')
    def test_classify_text_with_ml_model(self, mock_pipeline, domain_classifier):
        """Test classifying text using the ML model."""
        # Set up the mock ML model
        mock_model = MagicMock()
        mock_model.predict.return_value = ["mathematics"]
        mock_model.predict_proba.return_value = [[0.2, 0.7, 0.1]]  # Probabilities for each class

        # Assign the mock model
        domain_classifier.ml_model = mock_model

        # Call the method
        domain, confidence = domain_classifier.classify_text(MATH_TEXT, "en")

        # Check that the ML model was used
        assert mock_model.predict.called
        assert mock_model.predict_proba.called

        assert domain == "mathematics"
        assert confidence == 0.7

    def test_extract_domain_specific_features_math(self, domain_classifier):
        """Test extracting mathematics-specific features."""
        features = domain_classifier.extract_domain_specific_features(MATH_TRANSCRIPT, "mathematics")

        assert features["domain"] == "mathematics"
        assert features["theoretical_segments"] == 2
        assert features["practical_segments"] == 1

        # Check domain-specific metadata
        assert "domain_specific_metadata" in features
        assert "formulas_count" in features["domain_specific_metadata"]
        assert features["domain_specific_metadata"]["formulas_count"] == 3

    def test_extract_domain_specific_features_programming(self, domain_classifier):
        """Test extracting programming-specific features."""
        features = domain_classifier.extract_domain_specific_features(PROGRAMMING_TRANSCRIPT, "programming")

        assert features["domain"] == "programming"
        assert features["theoretical_segments"] == 2
        assert features["practical_segments"] == 1

        # Check domain-specific metadata
        assert "domain_specific_metadata" in features
        assert "code_snippets_count" in features["domain_specific_metadata"]
        assert features["domain_specific_metadata"]["code_snippets_count"] == 1

    def test_extract_math_features(self, domain_classifier):
        """Test extracting mathematics-specific features."""
        features = domain_classifier._extract_math_features(MATH_TRANSCRIPT, "en")

        assert "formulas_count" in features
        assert features["formulas_count"] == 3
        assert "theorems_count" in features
        assert "proofs_count" in features
        assert "definitions_count" in features
        assert "topics" in features

    def test_extract_programming_features(self, domain_classifier):
        """Test extracting programming-specific features."""
        features = domain_classifier._extract_programming_features(PROGRAMMING_TRANSCRIPT, "en")

        assert "code_snippets_count" in features
        assert features["code_snippets_count"] == 1
        assert "languages_mentioned" in features
        assert "algorithms_count" in features
        assert "data_structures_count" in features
        assert "topics" in features

        # Check if Python is detected
        assert "python" in [lang.lower() for lang in features["languages_mentioned"]]

    def test_extract_key_concepts(self, domain_classifier):
        """Test extracting key concepts from a transcript."""
        with patch.object(domain_classifier, '_count_concept_occurrences', return_value=3), \
             patch.object(domain_classifier, '_is_concept_theoretical', return_value=True):

            concepts = domain_classifier._extract_key_concepts(MATH_TRANSCRIPT, "mathematics", "en")

            assert isinstance(concepts, list)
            assert len(concepts) <= 10  # Should return at most 10 concepts

            # Check concept structure if any were found
            if concepts:
                assert "text" in concepts[0]
                assert "domain" in concepts[0]
                assert "frequency" in concepts[0]
                assert "theoretical" in concepts[0]
                assert concepts[0]["domain"] == "mathematics"

    def test_is_concept_theoretical(self, domain_classifier):
        """Test determining if a concept is theoretical or practical."""
        # Create segments where "calculus" appears in both theoretical and practical segments
        segments = [
            {"text": "Calculus is the study of change.", "content_type": "theoretical"},
            {"text": "Calculus has many real-world applications.", "content_type": "theoretical"},
            {"text": "Let's solve this calculus problem.", "content_type": "practical"}
        ]

        # Concept appears more in theoretical segments
        is_theoretical = domain_classifier._is_concept_theoretical("calculus", segments, "en")
        assert is_theoretical is True

        # Create segments where "problem" appears more in practical segments
        segments = [
            {"text": "Understanding problem statements is important.", "content_type": "theoretical"},
            {"text": "Let's solve this problem step by step.", "content_type": "practical"},
            {"text": "Here's another problem to practice with.", "content_type": "practical"}
        ]

        # Concept appears more in practical segments
        is_theoretical = domain_classifier._is_concept_theoretical("problem", segments, "en")
        assert is_theoretical is False

    def test_count_concept_occurrences(self, domain_classifier):
        """Test counting occurrences of a concept in text."""
        text = "Calculus is important. We use calculus to solve problems. Differential calculus focuses on rates of change."

        count = domain_classifier._count_concept_occurrences("calculus", text)
        assert count == 3

        count = domain_classifier._count_concept_occurrences("derivative", text)
        assert count == 0
