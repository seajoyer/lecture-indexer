"""
Unit tests for the Transcript Processor component.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import os
import uuid

from data_acquisition.transcript_processor.python.transcript_processor import TranscriptProcessor

# Test data
ENGLISH_RAW_SEGMENTS = [
    {
        "start": 0.0,
        "duration": 5.0,
        "text": "Welcome to this lecture on calculus.",
        "language": "en"
    },
    {
        "start": 5.0,
        "duration": 10.0,
        "text": "Today we'll discuss derivatives and their applications.",
        "language": "en"
    },
    {
        "start": 15.0,
        "duration": 8.0,
        "text": "Let's start with the definition of a derivative.",
        "language": "en"
    },
    {
        "start": 23.0,
        "duration": 12.0,
        "text": "A derivative is defined as the limit of the difference quotient as the interval approaches zero.",
        "language": "en"
    },
    {
        "start": 35.0,
        "duration": 10.0,
        "text": "Now let's solve a problem. Find the derivative of f(x) = x^2.",
        "language": "en"
    }
]

RUSSIAN_RAW_SEGMENTS = [
    {
        "start": 0.0,
        "duration": 5.0,
        "text": "Добро пожаловать на лекцию по исчислению.",
        "language": "ru"
    },
    {
        "start": 5.0,
        "duration": 10.0,
        "text": "Сегодня мы обсудим производные и их применения.",
        "language": "ru"
    },
    {
        "start": 15.0,
        "duration": 8.0,
        "text": "Начнем с определения производной.",
        "language": "ru"
    },
    {
        "start": 23.0,
        "duration": 12.0,
        "text": "Производная определяется как предел разностного частного при стремлении интервала к нулю.",
        "language": "ru"
    },
    {
        "start": 35.0,
        "duration": 10.0,
        "text": "Теперь решим задачу. Найдите производную f(x) = x^2.",
        "language": "ru"
    }
]

MATH_VIDEO_METADATA = {
    "video_id": "test123",
    "title": "Introduction to Calculus",
    "domain": "mathematics",
    "domain_confidence": 0.9,
    "language": "en"
}

@pytest.fixture
def transcript_processor():
    """Create a Transcript Processor instance."""
    with patch('spacy.load') as mock_load:
        # Mock the spacy language models
        mock_en_nlp = MagicMock()
        mock_ru_nlp = MagicMock()

        # Make mock_load return different mocks based on the argument
        def side_effect(model_name):
            if model_name == 'en_core_web_sm':
                return mock_en_nlp
            elif model_name == 'ru_core_news_sm':
                return mock_ru_nlp
            raise ValueError(f"No mock for {model_name}")

        mock_load.side_effect = side_effect

        processor = TranscriptProcessor()

        # Manually set the NLP models
        processor.en_nlp = mock_en_nlp
        processor.ru_nlp = mock_ru_nlp

        return processor

class TestTranscriptProcessor:
    """Test the Transcript Processor component."""

    def test_process_transcript_english(self, transcript_processor):
        """Test processing an English transcript."""
        # Mock the uuid.uuid4 function to return predictable values
        with patch('uuid.uuid4', return_value='test-uuid'):
            processed = transcript_processor.process_transcript(ENGLISH_RAW_SEGMENTS, MATH_VIDEO_METADATA)

            assert processed["language"] == "en"
            assert processed["domain"] == "mathematics"
            assert processed["video_id"] == "test123"
            assert len(processed["segments"]) == len(ENGLISH_RAW_SEGMENTS)

            # Check that we have sentences, sections, and segments
            assert "sentences" in processed
            assert "sections" in processed
            assert len(processed["sections"]) > 0

    def test_process_transcript_russian(self, transcript_processor):
        """Test processing a Russian transcript."""
        # Update metadata for Russian
        russian_metadata = MATH_VIDEO_METADATA.copy()
        russian_metadata["language"] = "ru"

        with patch('uuid.uuid4', return_value='test-uuid'):
            processed = transcript_processor.process_transcript(RUSSIAN_RAW_SEGMENTS, russian_metadata)

            assert processed["language"] == "ru"
            assert processed["domain"] == "mathematics"
            assert processed["video_id"] == "test123"
            assert len(processed["segments"]) == len(RUSSIAN_RAW_SEGMENTS)

            # Check that we have sentences, sections, and segments
            assert "sentences" in processed
            assert "sections" in processed
            assert len(processed["sections"]) > 0

    def test_process_transcript_empty(self, transcript_processor):
        """Test processing an empty transcript."""
        processed = transcript_processor.process_transcript([], MATH_VIDEO_METADATA)

        assert processed["language"] == "en"
        assert processed["domain"] == "unknown"
        assert processed["video_id"] == "test123"
        assert len(processed["segments"]) == 0
        assert len(processed["sentences"]) == 0
        assert len(processed["sections"]) == 0

    def test_normalize_transcript_english(self, transcript_processor):
        """Test normalizing an English transcript."""
        with patch('uuid.uuid4', return_value='test-uuid'):
            normalized = transcript_processor.normalize_transcript(ENGLISH_RAW_SEGMENTS, "en")

            assert len(normalized) == len(ENGLISH_RAW_SEGMENTS)
            assert normalized[0]["id"] == "test-uuid"
            assert normalized[0]["language"] == "en"
            assert normalized[0]["text"] == "Welcome to this lecture on calculus."
            assert normalized[0]["start_time"] == 0.0
            assert normalized[0]["end_time"] == 5.0

    def test_normalize_transcript_russian(self, transcript_processor):
        """Test normalizing a Russian transcript."""
        with patch('uuid.uuid4', return_value='test-uuid'):
            normalized = transcript_processor.normalize_transcript(RUSSIAN_RAW_SEGMENTS, "ru")

            assert len(normalized) == len(RUSSIAN_RAW_SEGMENTS)
            assert normalized[0]["id"] == "test-uuid"
            assert normalized[0]["language"] == "ru"
            assert normalized[0]["text"] == "Добро пожаловать на лекцию по исчислению."
            assert normalized[0]["start_time"] == 0.0
            assert normalized[0]["end_time"] == 5.0

    def test_segment_into_sentences(self, transcript_processor):
        """Test segmenting transcript into sentences."""
        # Create a normalized segment with multiple sentences
        normalized_segment = [{
            "id": "test-segment",
            "start_time": 0.0,
            "end_time": 10.0,
            "text": "This is the first sentence. This is the second sentence. And this is the third.",
            "speaker": None,
            "section_id": None,
            "is_section_boundary": False,
            "language": "en"
        }]

        with patch('uuid.uuid4', return_value='test-uuid'):
            sentences = transcript_processor.segment_into_sentences(normalized_segment, "en")

            assert len(sentences) == 3
            assert sentences[0]["text"] == "This is the first sentence."
            assert sentences[1]["text"] == "This is the second sentence."
            assert sentences[2]["text"] == "And this is the third."

            # Check that timestamps are interpolated
            assert sentences[0]["start_time"] == 0.0
            assert sentences[2]["end_time"] == 10.0
            assert sentences[0]["end_time"] < sentences[1]["start_time"]
            assert sentences[1]["end_time"] < sentences[2]["start_time"]

    def test_detect_sections(self, transcript_processor):
        """Test detecting sections in transcript."""
        # Create sentence segments with section boundaries
        sentences = [
            {
                "id": "s1",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Welcome to this lecture.",
                "speaker": None,
                "section_id": None,
                "is_section_boundary": False,
                "language": "en"
            },
            {
                "id": "s2",
                "start_time": 5.0,
                "end_time": 10.0,
                "text": "Chapter 1: Introduction to Calculus",
                "speaker": None,
                "section_id": None,
                "is_section_boundary": False,
                "language": "en"
            },
            {
                "id": "s3",
                "start_time": 10.0,
                "end_time": 15.0,
                "text": "Calculus is the study of change.",
                "speaker": None,
                "section_id": None,
                "is_section_boundary": False,
                "language": "en"
            },
            {
                "id": "s4",
                "start_time": 20.0,  # Note the gap to trigger a section boundary
                "end_time": 25.0,
                "text": "Let's move on to derivatives.",
                "speaker": None,
                "section_id": None,
                "is_section_boundary": False,
                "language": "en"
            }
        ]

        with patch('uuid.uuid4', side_effect=['section1', 'section2']):
            sections = transcript_processor.detect_sections(sentences, "en")

            assert len(sections) == 2

            # First section should include s1 and s2
            assert sections[0]["id"] == "section1"
            assert sections[0]["start_time"] == 0.0
            assert "segments" in sections[0]

            # Second section should include s3 and s4
            assert sections[1]["id"] == "section2"
            assert sections[1]["start_time"] in (10.0, 20.0)  # Could be either depending on implementation

            # Check that segments have been updated with section_id
            assert sentences[0]["section_id"] is not None
            assert sentences[2]["section_id"] is not None

    def test_enhance_with_nlp(self, transcript_processor):
        """Test enhancing segments with NLP data."""
        # Create a basic sentence segment
        sentence = {
            "id": "s1",
            "start_time": 0.0,
            "end_time": 5.0,
            "text": "The derivative of f(x) = x^2 is 2x.",
            "speaker": None,
            "section_id": "section1",
            "is_section_boundary": False,
            "language": "en",
            "original_segment_id": "orig1"
        }

        # Mock the NLP processing
        with patch.object(transcript_processor, '_process_english_nlp', return_value={"pos_tags": []}), \
             patch.object(transcript_processor, '_extract_formulas', return_value=[{"text": "f(x) = x^2", "start": 15, "end": 23}]), \
             patch.object(transcript_processor, '_extract_code_snippets', return_value=[]), \
             patch.object(transcript_processor, '_classify_sentence_type', return_value="explanation"), \
             patch.object(transcript_processor, '_classify_content_type', return_value="theoretical"):

            enhanced = transcript_processor.enhance_with_nlp([sentence], "en")

            assert len(enhanced) == 1
            assert "nlp_data" in enhanced[0]
            assert enhanced[0]["nlp_data"]["sentence_type"] == "explanation"
            assert enhanced[0]["content_type"] == "theoretical"
            assert len(enhanced[0]["nlp_data"]["formulas"]) == 1
            assert enhanced[0]["nlp_data"]["formulas"][0]["text"] == "f(x) = x^2"

    def test_classify_domain_mathematics(self, transcript_processor):
        """Test domain classification for mathematics content."""
        text = "In this lecture, we'll discuss calculus, derivatives, and integrals. We'll solve equations and work with functions."

        domain, confidence = transcript_processor.classify_domain(text, "en")

        assert domain == "mathematics"
        assert confidence > 0.5

    def test_classify_domain_programming(self, transcript_processor):
        """Test domain classification for programming content."""
        text = "In this programming tutorial, we'll write Python code and create algorithms. We'll implement data structures and use object-oriented programming."

        domain, confidence = transcript_processor.classify_domain(text, "en")

        assert domain == "programming"
        assert confidence > 0.5

    def test_classify_domain_physics(self, transcript_processor):
        """Test domain classification for physics content."""
        text = "In this physics lecture, we'll study mechanics, forces, and motion. We'll explore Newton's laws and solve problems with kinematics."

        domain, confidence = transcript_processor.classify_domain(text, "en")

        assert domain == "physics"
        assert confidence > 0.5

    def test_normalize_english_text(self, transcript_processor):
        """Test normalizing English text."""
        text = "  This is a   text with extra  spaces. It has speaker [Professor]: identification. And it has... ellipses.  "

        normalized = transcript_processor._normalize_english_text(text)

        assert normalized == "This is a text with extra spaces. It has speaker identification. And it has... ellipses."

    def test_normalize_russian_text(self, transcript_processor):
        """Test normalizing Russian text."""
        text = "  Это текст с   лишними  пробелами. В нём есть [Профессор]: обозначение говорящего. И в нём есть... многоточие.  "

        normalized = transcript_processor._normalize_russian_text(text)

        assert normalized == "Это текст с лишними пробелами. В нём есть обозначение говорящего. И в нём есть... многоточие."

    def test_extract_formulas(self, transcript_processor):
        """Test extracting mathematical formulas from text."""
        text = "The formula is $f(x) = x^2$. Another formula is \\(E = mc^2\\). And one more: $$\\int_0^1 x^2 dx = \\frac{1}{3}$$"

        formulas = transcript_processor._extract_formulas(text)

        assert len(formulas) == 3
        assert formulas[0]["formula"] == "f(x) = x^2"
        assert formulas[1]["formula"] == "E = mc^2"
        assert "\\int_0^1 x^2 dx = \\frac{1}{3}" in formulas[2]["formula"]

    def test_extract_code_snippets(self, transcript_processor):
        """Test extracting code snippets from text."""
        text = "Here's a code snippet: `print('Hello')`. And a code block:\n```python\ndef add(a, b):\n    return a + b\n```"

        snippets = transcript_processor._extract_code_snippets(text)

        assert len(snippets) == 2
        assert snippets[0]["code"] == "print('Hello')"
        assert snippets[1]["code"] == "def add(a, b):\n    return a + b"
        assert snippets[1]["language"] == "python"

    def test_classify_sentence_type(self, transcript_processor):
        """Test classifying sentence types."""
        # Definition
        text1 = "A derivative is defined as the limit of the difference quotient."
        type1 = transcript_processor._classify_sentence_type(text1, "en")
        assert type1 == "definition"

        # Example
        text2 = "For example, let's calculate the derivative of f(x) = x^2."
        type2 = transcript_processor._classify_sentence_type(text2, "en")
        assert type2 == "example"

        # Problem statement
        text3 = "Find the derivative of the function g(x) = sin(x)."
        type3 = transcript_processor._classify_sentence_type(text3, "en")
        assert type3 == "problem_statement"

        # Solution
        text4 = "To solve this problem, we first apply the chain rule."
        type4 = transcript_processor._classify_sentence_type(text4, "en")
        assert type4 == "solution"

        # Default to explanation
        text5 = "The value of x increases over time."
        type5 = transcript_processor._classify_sentence_type(text5, "en")
        assert type5 == "explanation"

    def test_classify_content_type(self, transcript_processor):
        """Test classifying content as theoretical or practical."""
        # Theoretical content
        text1 = "In theory, the concept of a limit is fundamental to calculus."
        type1 = transcript_processor._classify_content_type(text1, "explanation", "en")
        assert type1 == "theoretical"

        # Practical content
        text2 = "Let's solve this problem: find the derivative of f(x) = x^3 + 2x."
        type2 = transcript_processor._classify_content_type(text2, "problem_statement", "en")
        assert type2 == "practical"

        # Mixed content
        text3 = "This concept has both theoretical foundations and practical applications."
        type3 = transcript_processor._classify_content_type(text3, "explanation", "en")
        assert type3 in ("theoretical", "practical", "mixed")

        # Content with code (practical)
        text4 = "Here's how to implement this algorithm: ```python\ndef algo(): pass```"
        type4 = transcript_processor._classify_content_type(text4, "explanation", "en")
        assert type4 == "practical"

        # Content with formulas (could be either, but often theoretical)
        text5 = "The formula $E = mc^2$ represents the equivalence of mass and energy."
        type5 = transcript_processor._classify_content_type(text5, "explanation", "en")
        assert type5 in ("theoretical", "mixed")  # Could be either depending on implementation
