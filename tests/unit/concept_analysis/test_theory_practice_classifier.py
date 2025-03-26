"""
Tests for the Theory/Practice Classifier module.
"""

import pytest
import hashlib
from concept_analysis.relevance_analyzer.python.theory_practice_classifier import TheoryPracticeClassifier

# Basic classification tests

def test_classify_theoretical_text_english(mock_theory_practice_classifier):
    """Test classification of theoretical text in English."""
    classifier = mock_theory_practice_classifier
    text = "In theoretical calculus, we define the derivative as the limit of the difference quotient."
    classification, confidence = classifier.classify_text(text, language="en")
    assert classification == "theoretical"
    assert confidence > 0.5

def test_classify_practical_text_english(mock_theory_practice_classifier):
    """Test classification of practical text in English."""
    classifier = mock_theory_practice_classifier
    text = "Let's solve this problem: find the derivative of f(x) = x^2 by using the power rule."
    classification, confidence = classifier.classify_text(text, language="en")
    assert classification == "practical"
    assert confidence > 0.5

def test_classify_theoretical_text_russian(mock_theory_practice_classifier):
    """Test classification of theoretical text in Russian."""
    classifier = mock_theory_practice_classifier
    text = "В теоретическом исчислении, мы определяем производную как предел разностного отношения."
    classification, confidence = classifier.classify_text(text, language="ru")
    assert classification == "theoretical"
    assert confidence > 0.5

def test_classify_practical_text_russian(mock_theory_practice_classifier):
    """Test classification of practical text in Russian."""
    classifier = mock_theory_practice_classifier
    text = "Давайте решим задачу: найдите производную функции f(x) = x^2, используя правило степени."
    classification, confidence = classifier.classify_text(text, language="ru")
    assert classification == "practical"
    assert confidence > 0.5

# Segment classification tests

def test_classify_segment_with_nlp_data(mock_theory_practice_classifier):
    """Test classification of segment with NLP data."""
    classifier = mock_theory_practice_classifier
    segment = {
        "text": "Let's look at the formula for the derivative: f'(x) = lim_{h→0} [f(x+h) - f(x)] / h",
        "language": "en",
        "nlp_data": {
            "sentence_type": "definition",
            "formulas": [
                {
                    "formula": "f'(x) = lim_{h→0} [f(x+h) - f(x)] / h",
                    "text": "f'(x) = lim_{h→0} [f(x+h) - f(x)] / h",
                    "start": 38,
                    "end": 75
                }
            ],
            "code_snippets": []
        }
    }
    classification, confidence = classifier.classify_segment(segment)
    assert classification == "theoretical"
    assert confidence > 0.5

def test_classify_segment_with_code(mock_theory_practice_classifier):
    """Test classification of segment with code snippet."""
    classifier = mock_theory_practice_classifier
    segment = {
        "text": "Here's how to implement a derivative function in Python: def derivative(f, x, h=0.0001): return (f(x+h) - f(x))/h",
        "language": "en",
        "nlp_data": {
            "sentence_type": "example",
            "formulas": [],
            "code_snippets": [
                {
                    "code": "def derivative(f, x, h=0.0001): return (f(x+h) - f(x))/h",
                    "language": "python",
                    "text": "def derivative(f, x, h=0.0001): return (f(x+h) - f(x))/h",
                    "start": 53,
                    "end": 107
                }
            ]
        }
    }
    classification, confidence = classifier.classify_segment(segment)
    assert classification == "practical"
    assert confidence > 0.5

# Transcript classification tests

def test_classify_transcript(mock_theory_practice_classifier, sample_transcript_data):
    """Test classification of a transcript."""
    classifier = mock_theory_practice_classifier

    # Ensure the transcript has specified content types
    sample_transcript_data["segments"][0]["content_type"] = "theoretical"
    sample_transcript_data["segments"][1]["content_type"] = "theoretical"
    sample_transcript_data["segments"][2]["content_type"] = "practical"

    result = classifier.classify_transcript(sample_transcript_data)

    assert "classification" in result
    assert "theory_practice_ratio" in result
    assert 0 <= result["theory_practice_ratio"] <= 1
    assert result["theoretical_segments"] == 2
    assert result["practical_segments"] == 1

# Pattern extraction tests

def test_extract_theory_practice_patterns(mock_theory_practice_classifier, sample_transcript_data):
    """Test extraction of theory/practice patterns from a transcript."""
    classifier = mock_theory_practice_classifier

    # Ensure the transcript has theoretical segments followed by a practical segment
    sample_transcript_data["segments"][0]["content_type"] = "theoretical"
    sample_transcript_data["segments"][1]["content_type"] = "theoretical"
    sample_transcript_data["segments"][2]["content_type"] = "practical"

    patterns = classifier.extract_theory_practice_patterns(sample_transcript_data)

    assert "theory_to_practice_sequences" in patterns
    assert "practice_to_theory_sequences" in patterns
    assert len(patterns["theory_to_practice_sequences"]) > 0
    assert isinstance(patterns["theory_practice_alternations"], int)
    assert patterns["theory_practice_alternations"] >= 1  # At least one alternation
    assert patterns["max_theory_sequence"] >= 2  # At least 2 theoretical segments in a row
    assert patterns["max_practice_sequence"] >= 1  # At least 1 practical segment in a row

# Domain-specific tests

def test_classification_for_mathematics_domain(mock_theory_practice_classifier):
    """Test classification for mathematics domain content."""
    classifier = mock_theory_practice_classifier
    text = "The Pythagorean theorem states that a^2 + b^2 = c^2 for right triangles."
    classification, confidence = classifier.classify_text(text, language="en", domain="mathematics")
    assert classification in ["theoretical", "practical"]
    assert confidence > 0

def test_classification_for_programming_domain(mock_theory_practice_classifier):
    """Test classification for programming domain content."""
    classifier = mock_theory_practice_classifier
    text = "Let's create a function that implements binary search in Python."
    classification, confidence = classifier.classify_text(text, language="en", domain="programming")
    assert classification == "practical"
    assert confidence > 0.5

def test_classification_for_physics_domain(mock_theory_practice_classifier):
    """Test classification for physics domain content."""
    classifier = mock_theory_practice_classifier
    text = "Newton's Third Law states that for every action, there is an equal and opposite reaction."
    classification, confidence = classifier.classify_text(text, language="en", domain="physics")
    assert classification == "theoretical"
    assert confidence > 0.5

# Caching test

def test_caching_behavior(mock_theory_practice_classifier, test_cache_manager):
    """Test that results are properly cached."""
    classifier = mock_theory_practice_classifier

    # Override cache with a test one that we can check
    classifier.cache = test_cache_manager.region("test_cache")

    # First classification should cache the result
    text = "This is a test of the caching system with theoretical content."
    classifier.classify_text(text, language="en")

    # Check if result is in cache
    cache_key = f"classify_{hashlib.md5(text.encode()).hexdigest()}_en_None"
    assert classifier.cache.get(cache_key) is not None

# Optional: DB integration test

@pytest.mark.db
def test_database_integration(mock_theory_practice_classifier, test_db_context):
    """Test database integration for theory/practice classification."""
    classifier = mock_theory_practice_classifier

    # Override classifier's db_context with the test one
    classifier.db_context = test_db_context

    # Create a sample transcript with video_id
    transcript = {
        "video_id": "test_video_123",
        "segments": [
            {
                "id": "seg1",
                "text": "Let's define what a derivative is in calculus.",
                "language": "en"
            },
            {
                "id": "seg2",
                "text": "Now we'll calculate some derivatives using the power rule.",
                "language": "en"
            }
        ],
        "language": "en",
        "domain": "mathematics"
    }

    # Classify the transcript
    result = classifier.classify_transcript(transcript)

    # Check if video data was stored in database
    video = test_db_context.video_repository.get_video("test_video_123")

    assert video is not None
    assert "theory_practice_ratio" in video
    assert video["theory_practice_ratio"] == result["theory_practice_ratio"]
