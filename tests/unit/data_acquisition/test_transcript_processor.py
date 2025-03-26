import pytest
import unittest.mock as mock
from data_acquisition.transcript_processor.python.transcript_processor import TranscriptProcessor

class TestTranscriptProcessor:
    """Tests for TranscriptProcessor class."""

    def test_process_transcript(self, mock_transcript_processor, sample_transcript_data, sample_video_data):
        """Test the complete transcript processing workflow."""
        result = mock_transcript_processor.process_transcript(
            sample_transcript_data["segments"],
            sample_video_data
        )

        # Check the structure of the result
        assert "segments" in result
        assert "sentences" in result
        assert "sections" in result
        assert "language" in result
        assert "domain" in result
        assert "video_id" in result

        # Check segments
        assert len(result["segments"]) > 0
        segment = result["segments"][0]
        assert "id" in segment
        assert "start_time" in segment
        assert "end_time" in segment
        assert "text" in segment
        assert "content_type" in segment

        # Check that all segments have content type
        for segment in result["segments"]:
            assert "content_type" in segment
            assert segment["content_type"] in ["theoretical", "practical", "mixed"]

        # Check sentences
        assert len(result["sentences"]) > 0

        # Check sections
        assert len(result["sections"]) > 0
        section = result["sections"][0]
        assert "id" in section
        assert "start_time" in section
        assert "end_time" in section
        assert "segments" in section

    def test_normalize_transcript(self, mock_transcript_processor):
        """Test transcript normalization."""
        raw_segments = [
            {
                "start": 0.0,
                "duration": 5.0,
                "text": "Welcome to the mathematics lecture.",
                "language": "en"
            },
            {
                "start": 5.0,
                "duration": 5.0,
                "text": "[Professor]: Today we'll discuss derivatives and integrals.",
                "language": "en"
            }
        ]

        normalized = mock_transcript_processor.normalize_transcript(raw_segments, "en")

        # Check normalization results
        assert len(normalized) == len(raw_segments)
        for segment in normalized:
            assert "id" in segment
            assert "start_time" in segment
            assert "end_time" in segment
            assert "text" in segment
            assert "speaker" in segment
            assert "is_section_boundary" in segment
            assert "section_id" in segment

        # Check that speaker identifier was removed
        assert "[Professor]:" not in normalized[1]["text"]

    def test_segment_into_sentences(self, mock_transcript_processor):
        """Test segmenting transcript into sentences."""
        normalized_segments = [
            {
                "id": "seg1",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "This is a test. This is another test.",
                "language": "en"
            }
        ]

        sentences = mock_transcript_processor.segment_into_sentences(normalized_segments, "en")

        # Should have two sentences
        assert len(sentences) == 2
        assert sentences[0]["text"] == "This is a test."
        assert sentences[1]["text"] == "This is another test."

        # Check time segmentation
        assert sentences[0]["start_time"] < sentences[1]["start_time"]
        assert sentences[0]["end_time"] <= sentences[1]["start_time"]
        assert sentences[1]["end_time"] <= normalized_segments[0]["end_time"]

        # Check that each sentence references its original segment
        assert "original_segment_id" in sentences[0]
        assert sentences[0]["original_segment_id"] == "seg1"

    def test_detect_sections(self, mock_transcript_processor):
        """Test detection of logical sections in transcript."""
        sentence_segments = [
            {
                "id": "sent1",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Introduction to Calculus.",
                "language": "en"
            },
            {
                "id": "sent2",
                "start_time": 5.0,
                "end_time": 10.0,
                "text": "Let's begin with basics.",
                "language": "en"
            },
            {
                "id": "sent3",
                "start_time": 15.0,  # Gap indicates section boundary
                "end_time": 20.0,
                "text": "Chapter 1: Limits.",
                "language": "en"
            }
        ]

        sections = mock_transcript_processor.detect_sections(sentence_segments, "en")

        # Should detect at least 1 section
        assert len(sections) >= 1

        # Check section structure
        for section in sections:
            assert "id" in section
            assert "start_time" in section
            assert "end_time" in section
            assert "segments" in section

        # Check that "Chapter 1: Limits" is detected as a section boundary
        found_chapter_section = False
        for section in sections:
            if any("Chapter 1: Limits" in sentence_segments[i]["text"] for i in range(len(sentence_segments))
                  if sentence_segments[i]["id"] in section["segments"]):
                found_chapter_section = True
                break

        assert found_chapter_section

    def test_enhance_with_nlp(self, mock_transcript_processor):
        """Test NLP enhancement of sentence segments."""
        sentence_segments = [
            {
                "id": "sent1",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "A derivative is defined as the limit of the difference quotient.",
                "language": "en"
            },
            {
                "id": "sent2",
                "start_time": 5.0,
                "end_time": 10.0,
                "text": "Let's solve this problem: find the derivative of f(x) = x^2.",
                "language": "en"
            },
            {
                "id": "sent3",
                "start_time": 10.0,
                "end_time": 15.0,
                "text": "Here's a code snippet: `print('Hello')`. And a code block:",
                "language": "en"
            },
            {
                "id": "sent4",
                "start_time": 15.0,
                "end_time": 20.0,
                "text": "```python\ndef add(a, b):\n    return a + b\n```",
                "language": "en"
            }
        ]

        enhanced = mock_transcript_processor.enhance_with_nlp(sentence_segments, "en")

        # Check NLP enhancements
        assert len(enhanced) == 4

        # Check that segments have content type classification
        for segment in enhanced:
            assert "content_type" in segment
            assert "nlp_data" in segment

        # Check that formulas and code snippets are extracted
        assert "formulas" in enhanced[1]["nlp_data"]
        assert "code_snippets" in enhanced[2]["nlp_data"]
        assert "code_snippets" in enhanced[3]["nlp_data"]

        # Check that code snippets are correctly identified
        code_segment = enhanced[3]
        assert len(code_segment["nlp_data"]["code_snippets"]) > 0
        assert "python" in code_segment["nlp_data"]["code_snippets"][0]["language"]

    def test_classify_domain(self, mock_transcript_processor):
        """Test domain classification from text."""
        # Mathematics text
        math_domain, math_confidence = mock_transcript_processor.classify_domain(
            "This lecture covers calculus, derivatives, and integrals.", "en"
        )
        assert math_domain == "mathematics"
        assert math_confidence > 0.0

        # Programming text
        prog_domain, prog_confidence = mock_transcript_processor.classify_domain(
            "We'll implement this algorithm in Python using functions and classes.", "en"
        )
        assert prog_domain == "programming"
        assert prog_confidence > 0.0

        # Physics text
        phys_domain, phys_confidence = mock_transcript_processor.classify_domain(
            "Newton's laws of motion describe the relationship between a body and the forces acting upon it.", "en"
        )
        assert phys_domain == "physics"
        assert phys_confidence > 0.0

    def test_classify_content_type(self, mock_transcript_processor):
        """Test content type classification (theoretical vs practical)."""
        # Theoretical content
        theoretical_text = "In theoretical calculus, we define the derivative as the limit of the difference quotient."
        theoretical_type = mock_transcript_processor._classify_content_type(
            theoretical_text, "explanation", "en"
        )
        assert theoretical_type == "theoretical"

        # Practical content
        practical_text = "Let's solve this problem: find the derivative of f(x) = x^2."
        practical_type = mock_transcript_processor._classify_content_type(
            practical_text, "problem_statement", "en"
        )
        assert practical_type == "practical"

        # Mixed content
        mixed_text = "The concept of a derivative is important, and we'll see how to apply it in real-world problems."
        mixed_type = mock_transcript_processor._classify_content_type(
            mixed_text, "explanation", "en"
        )
        assert mixed_type == "mixed"

    def test_russian_language_support(self, mock_transcript_processor):
        """Test support for Russian language transcripts."""
        ru_transcript = [
            {
                "start": 0.0,
                "duration": 5.0,
                "text": "Добро пожаловать на лекцию по математике.",
                "language": "ru"
            },
            {
                "start": 5.0,
                "duration": 5.0,
                "text": "В теоретическом исчислении мы определяем производную как предел разностного частного.",
                "language": "ru"
            },
            {
                "start": 10.0,
                "duration": 5.0,
                "text": "Давайте решим задачу: найдите производную функции f(x) = x^2.",
                "language": "ru"
            }
        ]

        ru_metadata = {
            "video_id": "test_ru",
            "title": "Лекция по математике",
            "description": "Лекция по математическому анализу.",
            "domain": "mathematics"
        }

        result = mock_transcript_processor.process_transcript(ru_transcript, ru_metadata)

        # Check Russian language handling
        assert result["language"] == "ru"
        assert len(result["segments"]) == 3

        # Check content type classification for Russian text
        assert result["segments"][0]["content_type"] in ["theoretical", "practical", "mixed"]
        assert result["segments"][1]["content_type"] == "theoretical"  # Should be identified as theoretical
        assert result["segments"][2]["content_type"] == "practical"    # Should be identified as practical
