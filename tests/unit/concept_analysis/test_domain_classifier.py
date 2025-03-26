"""
Unit tests for the Domain Classifier component of the Lecture Video Content Indexer.
Tests domain classification, feature extraction, and model training functionality.
"""

import pytest
import os
import json
from typing import Dict, List, Any
import logging

# Import the component to test
from concept_analysis.concept_extractor.python.domain_concept_extractor import DomainClassifier

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDomainClassifier:
    """Test suite for the DomainClassifier component."""

    @pytest.fixture
    def domain_classifier(self):
        """Fixture to provide a DomainClassifier instance."""
        return DomainClassifier()

    @pytest.fixture
    def sample_transcript(self):
        """Fixture to provide a sample transcript dictionary."""
        return {
            "segments": [
                {
                    "id": "seg1",
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "text": "Welcome to this mathematics lecture on calculus.",
                    "language": "en"
                },
                {
                    "id": "seg2",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "text": "Today we will learn about derivatives and integration.",
                    "language": "en"
                },
                {
                    "id": "seg3",
                    "start_time": 20.0,
                    "end_time": 30.0,
                    "text": "The derivative is defined as the limit as delta x approaches zero.",
                    "language": "en"
                }
            ],
            "language": "en",
            "domain": "unknown",
            "video_id": "test123"
        }

    @pytest.fixture
    def sample_russian_transcript(self):
        """Fixture to provide a sample Russian transcript dictionary."""
        return {
            "segments": [
                {
                    "id": "seg1",
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "text": "Добро пожаловать на лекцию по программированию.",
                    "language": "ru"
                },
                {
                    "id": "seg2",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "text": "Сегодня мы изучим алгоритмы и структуры данных.",
                    "language": "ru"
                },
                {
                    "id": "seg3",
                    "start_time": 20.0,
                    "end_time": 30.0,
                    "text": "Мы напишем код на языке Python.",
                    "language": "ru"
                }
            ],
            "language": "ru",
            "domain": "unknown",
            "video_id": "test456"
        }

    @pytest.fixture
    def sample_physics_text(self):
        """Fixture to provide a sample physics text."""
        return "In this lecture, we will discuss Newton's laws of motion and the conservation of energy in classical mechanics."

    @pytest.fixture
    def sample_training_data(self):
        """Fixture to provide sample training data for the classifier."""
        return [
            {"text": "In calculus, we study the concept of limits and derivatives.", "classification": "mathematics"},
            {"text": "The function f(x) = x^2 has a derivative f'(x) = 2x.", "classification": "mathematics"},
            {"text": "Python is a programming language with dynamic typing.", "classification": "programming"},
            {"text": "We can implement a binary search tree in Java.", "classification": "programming"},
            {"text": "Newton's laws describe the motion of objects.", "classification": "physics"},
            {"text": "Quantum mechanics is the study of subatomic particles.", "classification": "physics"}
        ]

    def test_classify_text_mathematics(self, domain_classifier):
        """Test classification of mathematical text."""
        text = "In calculus, the derivative of a function f(x) = x^2 is f'(x) = 2x."
        domain, confidence = domain_classifier.classify_text(text, "en")

        assert domain == "mathematics"
        assert confidence > 0.5

    def test_classify_text_programming(self, domain_classifier):
        """Test classification of programming text."""
        text = "In Python, we can write a function to implement a binary search algorithm."
        domain, confidence = domain_classifier.classify_text(text, "en")

        assert domain == "programming"
        assert confidence > 0.5

    def test_classify_text_physics(self, domain_classifier, sample_physics_text):
        """Test classification of physics text."""
        domain, confidence = domain_classifier.classify_text(sample_physics_text, "en")

        assert domain == "physics"
        assert confidence > 0.5

    def test_classify_text_russian(self, domain_classifier):
        """Test classification of Russian text."""
        text = "В программировании на Python мы используем классы и объекты."
        domain, confidence = domain_classifier.classify_text(text, "ru")

        assert domain == "programming"
        assert confidence > 0.5

    def test_classify_mixed_domain(self, domain_classifier):
        """Test classification of text with mixed domain signals."""
        text = "This lecture covers both mathematical concepts and programming implementations."
        domain, confidence = domain_classifier.classify_text(text, "en")

        # Either mathematics or programming could be determined as the primary domain
        assert domain in ["mathematics", "programming"]
        # Confidence should be lower for mixed content
        assert confidence <= 0.8

    def test_classify_empty_text(self, domain_classifier):
        """Test classification of empty text."""
        domain, confidence = domain_classifier.classify_text("", "en")

        assert domain == "unknown"
        assert confidence == 0.0

    def test_classify_transcript(self, domain_classifier, sample_transcript):
        """Test classification of a complete transcript."""
        domain, confidence = domain_classifier.classify_transcript(sample_transcript)

        assert domain == "mathematics"
        assert confidence > 0.5

    def test_classify_russian_transcript(self, domain_classifier, sample_russian_transcript):
        """Test classification of a Russian transcript."""
        domain, confidence = domain_classifier.classify_transcript(sample_russian_transcript)

        assert domain == "programming"
        assert confidence > 0.5

    def test_extract_domain_features_mathematics(self, domain_classifier, sample_transcript):
        """Test extraction of domain-specific features for mathematics."""
        features = domain_classifier.extract_domain_specific_features(sample_transcript, "mathematics")

        assert features["domain"] == "mathematics"
        assert "key_concepts" in features
        assert len(features["key_concepts"]) > 0

    def test_extract_domain_features_programming(self, domain_classifier, sample_russian_transcript):
        """Test extraction of domain-specific features for programming."""
        features = domain_classifier.extract_domain_specific_features(sample_russian_transcript, "programming")

        assert features["domain"] == "programming"
        assert "key_concepts" in features
        assert len(features["key_concepts"]) > 0

    def test_extract_domain_features_physics(self, domain_classifier):
        """Test extraction of domain-specific features for physics."""
        transcript = {
            "segments": [
                {
                    "id": "seg1",
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "text": "Today we will study the laws of thermodynamics.",
                    "language": "en"
                },
                {
                    "id": "seg2",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "text": "Energy cannot be created or destroyed, only transformed.",
                    "language": "en"
                }
            ],
            "language": "en",
            "domain": "physics",
            "video_id": "test789"
        }

        features = domain_classifier.extract_domain_specific_features(transcript, "physics")

        assert features["domain"] == "physics"
        assert "key_concepts" in features
        assert len(features["key_concepts"]) > 0

    def test_train_model(self, domain_classifier, sample_training_data):
        """Test training of the classification model."""
        # Train the model
        domain_classifier.train_model(sample_training_data)

        # Verify model exists after training
        assert domain_classifier.ml_model is not None

        # Test the trained model
        text = "The integral of x^2 is x^3/3 + C."
        domain, confidence = domain_classifier.classify_text(text, "en")

        assert domain == "mathematics"
        assert confidence > 0.5

    @pytest.mark.integration
    def test_db_integration(self, domain_classifier, test_db_context):
        """Test integration with database for caching and persistence."""
        # This test requires the database context
        if not hasattr(domain_classifier, 'db_context') or domain_classifier.db_context is None:
            domain_classifier.db_context = test_db_context
            domain_classifier.cache = test_db_context.get_cache_region("domain_classifier")

        # Classify text and check that it gets cached
        text = "In physics, we study the motion of objects."
        domain, confidence = domain_classifier.classify_text(text, "en")

        assert domain == "physics"

        # Check if the result is cached
        cache_key = f"classify_{hashlib.md5(text.encode()).hexdigest()}_en_None"
        cached_result = domain_classifier.cache.get(cache_key)

        assert cached_result is not None
        assert cached_result[0] == domain
        assert cached_result[1] == confidence

    def test_theoretical_content_detection(self, domain_classifier):
        """Test detection of theoretical content."""
        text = "In theoretical calculus, we define the concept of a limit."
        domain, _ = domain_classifier.classify_text(text, "en")

        assert domain == "mathematics"

        # Create a segment for theoretical detection
        segment = {
            "text": text,
            "language": "en"
        }

        # In a real implementation, this would be included in domain-specific features
        features = domain_classifier.extract_domain_specific_features(
            {"segments": [segment], "language": "en"}, "mathematics"
        )

        # Check for theoretical key concepts
        theoretical_concepts = [c for c in features["key_concepts"] if c.get("theoretical", False)]
        assert len(theoretical_concepts) > 0

    def test_practical_content_detection(self, domain_classifier):
        """Test detection of practical content."""
        text = "Let's implement a sorting algorithm in Python."
        domain, _ = domain_classifier.classify_text(text, "en")

        assert domain == "programming"

        # Create a segment for practical detection
        segment = {
            "text": text,
            "language": "en"
        }

        # In a real implementation, this would be included in domain-specific features
        features = domain_classifier.extract_domain_specific_features(
            {"segments": [segment], "language": "en"}, "programming"
        )

        # Check for practical key concepts
        practical_concepts = [c for c in features["key_concepts"] if not c.get("theoretical", False)]
        assert len(practical_concepts) > 0

    def test_russian_theoretical_content(self, domain_classifier):
        """Test detection of theoretical content in Russian."""
        text = "В теоретическом исчислении мы рассматриваем понятие предела."
        domain, _ = domain_classifier.classify_text(text, "ru")

        assert domain == "mathematics"

        # Create a segment for theoretical detection
        segment = {
            "text": text,
            "language": "ru"
        }

        # In a real implementation, this would be included in domain-specific features
        features = domain_classifier.extract_domain_specific_features(
            {"segments": [segment], "language": "ru"}, "mathematics"
        )

        # Check for theoretical key concepts
        theoretical_concepts = [c for c in features["key_concepts"] if c.get("theoretical", False)]
        assert len(theoretical_concepts) > 0
