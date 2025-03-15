"""
Transcript Processor module for the Lecture Video Content Indexer.
Handles processing of raw transcripts into normalized, structured text suitable for concept extraction.
"""

import re
import uuid
import logging
from typing import List, Dict, Tuple, Optional, Any
import nltk
import spacy
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logger = logging.getLogger(__name__)

class TranscriptProcessor:
    """
    Processes raw transcripts into normalized, structured text suitable for concept extraction.
    Provides specialized handling for mathematical notation, code snippets, and physics terminology
    in both Russian and English.
    """

    def __init__(self):
        """Initialize the Transcript Processor with required NLP models."""
        logger.info("Initializing Transcript Processor")

        # Initialize language models
        try:
            self.en_nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded English NLP model")
        except Exception as e:
            logger.warning(f"Could not load English NLP model: {e}")
            self.en_nlp = None

        try:
            self.ru_nlp = spacy.load('ru_core_news_sm')
            logger.info("Loaded Russian NLP model")
        except Exception as e:
            logger.warning(f"Could not load Russian NLP model: {e}")
            self.ru_nlp = None

    def process_transcript(self, raw_segments: List[Dict], video_metadata: Dict) -> Dict:
        """
        Process raw transcript segments into a structured format.

        Args:
            raw_segments: List of raw transcript segments
            video_metadata: Video metadata dictionary

        Returns:
            Dictionary containing processed transcript data
        """
        if not raw_segments:
            logger.warning("Empty transcript provided")
            return {
                "segments": [],
                "sentences": [],
                "sections": [],
                "language": "en",
                "domain": "unknown",
                "video_id": video_metadata.get("video_id", "")
            }

        # Determine language
        language = raw_segments[0].get("language", "en")
        if not language:
            language = "en"  # Default to English

        # Domain from metadata
        domain = video_metadata.get("domain", "unknown")

        # Process transcript
        normalized_segments = self.normalize_transcript(raw_segments, language)

        # Handle NLTK punkt error safely
        try:
            sentence_segments = self.segment_into_sentences(normalized_segments, language)
        except LookupError as e:
            logger.warning(f"NLTK data missing, fallback to simple sentence segmentation: {e}")
            # Fallback segmentation - just use segments as sentences
            sentence_segments = []
            for segment in normalized_segments:
                sentence_segment = segment.copy()
                sentence_segment["id"] = str(uuid.uuid4())
                sentence_segments.append(sentence_segment)

        sections = self.detect_sections(sentence_segments, language)
        enhanced_segments = self.enhance_with_nlp(sentence_segments, language)

        # Combine results
        result = {
            "segments": enhanced_segments,
            "sentences": sentence_segments,
            "sections": sections,
            "language": language,
            "domain": domain,
            "video_id": video_metadata.get("video_id", "")
        }

        logger.info(f"Processed transcript with {len(enhanced_segments)} segments, "
                   f"{len(sentence_segments)} sentences, {len(sections)} sections")

        return result

    def normalize_transcript(self, raw_segments: List[Dict], language: str) -> List[Dict]:
        """
        Normalize raw transcript segments.

        Args:
            raw_segments: List of raw transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            List of normalized transcript segments
        """
        logger.info(f"Normalizing transcript with {len(raw_segments)} segments")
        normalized_segments = []

        for segment in raw_segments:
            text = segment.get("text", "")

            # Skip empty segments
            if not text.strip():
                continue

            # Apply language-specific normalization
            if language == "ru":
                normalized_text = self._normalize_russian_text(text)
            else:
                normalized_text = self._normalize_english_text(text)

            # Create normalized segment
            normalized_segment = {
                "id": str(uuid.uuid4()),
                "start_time": segment.get("start", 0),
                "end_time": segment.get("start", 0) + segment.get("duration", 0),
                "text": normalized_text,
                "speaker": segment.get("speaker"),
                "section_id": None,
                "is_section_boundary": False,
                "language": language
            }

            normalized_segments.append(normalized_segment)

        return normalized_segments

    def segment_into_sentences(self, normalized_segments: List[Dict], language: str) -> List[Dict]:
        """
        Segment normalized transcript into sentences.

        Args:
            normalized_segments: List of normalized transcript segments
            language: Language code ('en' or 'ru')

        Returns:
            List of sentence segments
        """
        logger.info("Segmenting transcript into sentences")
        sentence_segments = []

        for segment in normalized_segments:
            text = segment.get("text", "")

            # Use language-specific sentence tokenization
            try:
                if language == "ru":
                    sentences = self._tokenize_russian_sentences(text)
                else:
                    sentences = sent_tokenize(text)
            except LookupError:
                # Fallback if NLTK data is missing
                sentences = [text]

            # If no sentences were detected, use the whole segment as one sentence
            if not sentences:
                sentences = [text]

            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time

            # Create sentence segments with interpolated timestamps
            for i, sentence in enumerate(sentences):
                # Estimate time position proportionally to text length
                sentence_start = start_time
                if i > 0:
                    prev_lengths = sum(len(s) for s in sentences[:i])
                    total_length = sum(len(s) for s in sentences)
                    sentence_start = start_time + (duration * prev_lengths / total_length if total_length > 0 else 0)

                sentence_end = end_time
                if i < len(sentences) - 1:
                    next_lengths = sum(len(s) for s in sentences[i+1:])
                    total_length = sum(len(s) for s in sentences)
                    sentence_end = end_time - (duration * next_lengths / total_length if total_length > 0 else 0)

                # Create sentence segment
                sentence_segment = {
                    "id": str(uuid.uuid4()),
                    "start_time": sentence_start,
                    "end_time": sentence_end,
                    "text": sentence.strip(),
                    "speaker": segment.get("speaker"),
                    "section_id": segment.get("section_id"),
                    "is_section_boundary": segment.get("is_section_boundary", False),
                    "language": language,
                    "original_segment_id": segment.get("id")
                }

                sentence_segments.append(sentence_segment)

        return sentence_segments

    def detect_sections(self, sentence_segments: List[Dict], language: str) -> List[Dict]:
        """
        Detect logical sections in the transcript.

        Args:
            sentence_segments: List of sentence segments
            language: Language code ('en' or 'ru')

        Returns:
            List of section dictionaries
        """
        logger.info("Detecting sections in transcript")
        sections = []
        current_section = None
        section_segments = []

        # Patterns that might indicate section boundaries
        section_patterns = self._get_section_patterns(language)

        for i, segment in enumerate(sentence_segments):
            text = segment.get("text", "")

            # Check if this sentence potentially starts a new section
            is_section_boundary = False
            section_title = None

            # Look for section indicators
            for pattern in section_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    is_section_boundary = True
                    # Try to extract section title
                    if match.groups():
                        section_title = match.group(1).strip()
                    break

            # Also check for pauses (gaps between segments)
            if i > 0:
                prev_end = sentence_segments[i-1].get("end_time", 0)
                current_start = segment.get("start_time", 0)
                if current_start - prev_end > 3:  # More than 3 seconds pause
                    is_section_boundary = True

            # Update segment with section boundary information
            segment["is_section_boundary"] = is_section_boundary

            # If this is a section boundary, create a new section
            if is_section_boundary and i > 0:
                # Finalize previous section
                if current_section and section_segments:
                    sections.append(self._create_section(current_section, section_segments, language))

                # Start new section
                section_id = str(uuid.uuid4())
                current_section = {
                    "id": section_id,
                    "title": section_title,
                    "start_time": segment.get("start_time", 0),
                    "end_time": None  # Will be set when section ends
                }
                section_segments = []

            # If we don't have a current section yet, create the first one
            if current_section is None:
                section_id = str(uuid.uuid4())
                current_section = {
                    "id": section_id,
                    "title": None,
                    "start_time": segment.get("start_time", 0),
                    "end_time": None  # Will be set when section ends
                }

            # Update segment with section ID
            segment["section_id"] = current_section["id"]
            section_segments.append(segment)

        # Finalize the last section
        if current_section and section_segments:
            # Set end time to last segment's end time
            current_section["end_time"] = section_segments[-1].get("end_time", 0)
            sections.append(self._create_section(current_section, section_segments, language))

        return sections

    def enhance_with_nlp(self, sentence_segments: List[Dict], language: str) -> List[Dict]:
        """
        Enhance sentence segments with NLP analysis.

        Args:
            sentence_segments: List of sentence segments
            language: Language code ('en' or 'ru')

        Returns:
            List of enhanced sentence segments
        """
        logger.info("Enhancing transcript with NLP analysis")
        enhanced_segments = []

        for segment in sentence_segments:
            text = segment.get("text", "")

            # Skip empty segments
            if not text.strip():
                continue

            # Apply NLP processing based on language
            nlp_data = {}
            if language == "ru" and self.ru_nlp:
                nlp_data = self._process_russian_nlp(text)
            elif language == "en" and self.en_nlp:
                nlp_data = self._process_english_nlp(text)

            # Extract formulas and code snippets
            nlp_data["formulas"] = self._extract_formulas(text)
            nlp_data["code_snippets"] = self._extract_code_snippets(text)

            # Determine sentence type
            sentence_type = self._classify_sentence_type(text, language)
            nlp_data["sentence_type"] = sentence_type

            # Classify content type (theoretical/practical)
            content_type = self._classify_content_type(text, sentence_type, language)

            # Create enhanced segment
            enhanced_segment = segment.copy()
            enhanced_segment["nlp_data"] = nlp_data
            enhanced_segment["content_type"] = content_type

            enhanced_segments.append(enhanced_segment)

        return enhanced_segments

    def classify_domain(self, text: str, language: str) -> Tuple[str, float]:
        """
        Classify the domain of text.

        Args:
            text: Text to classify
            language: Language code ('en' or 'ru')

        Returns:
            Tuple of (domain, confidence)
        """
        # Define domain-specific keywords with language variations
        domains = {
            "mathematics": {
                "en": [
                    r'\bmath(ematics)?\b', r'\bcalculus\b', r'\balgebra\b', r'\bgeometry\b',
                    r'\btheorem\b', r'\bproof\b', r'\bequation\b', r'\bfunction\b',
                    r'\bderivative\b', r'\bintegral\b', r'\blimit\b', r'\bvector\b',
                    r'\bmatrix\b', r'\btopology\b'
                ],
                "ru": [
                    r'\bматематик[аеиоу]\b', r'\bалгебр[аеиоу]\b', r'\bгеометри[яию]\b',
                    r'\bтеорем[аеуы]\b', r'\bдоказательств[оаеу]\b', r'\bуравнени[еяю]\b',
                    r'\bфункци[яию]\b', r'\bпроизводн[аяыеую]\b', r'\bинтеграл[аы]?\b',
                    r'\bпредел[аы]?\b', r'\bвектор[аы]?\b', r'\bматриц[аы]?\b',
                    r'\bтопологи[яию]\b'
                ]
            },
            "programming": {
                "en": [
                    r'\bprogramming\b', r'\bcode\b', r'\balgorithm\b', r'\bfunction\b',
                    r'\bvariable\b', r'\bdevelopment\b', r'\bcomputer science\b',
                    r'\bpython\b', r'\bjava\b', r'\bc\+\+\b', r'\bjavascript\b',
                    r'\bdata structure\b', r'\bclass\b', r'\bobject\b', r'\bmethod\b'
                ],
                "ru": [
                    r'\bпрограммирован[иеяю]\b', r'\bкод[аеу]?\b', r'\bалгоритм[аеыу]?\b',
                    r'\bфункци[яию]\b', r'\bпеременн[аяыеую]\b', r'\bразработк[аеиу]\b',
                    r'\bинформатик[аеиу]\b', r'\bпитон[аеу]?\b', r'\bджав[аеыу]\b',
                    r'\bси\+\+\b', r'\bджаваскрипт[аеу]?\b', r'\bструктур[аыуе]? данных\b',
                    r'\bкласс[аеыу]?\b', r'\bобъект[аыу]?\b', r'\bметод[аыу]?\b'
                ]
            },
            "physics": {
                "en": [
                    r'\bphysics\b', r'\bmechanics\b', r'\bdynamics\b', r'\bkinematics\b',
                    r'\belectromagnetism\b', r'\bthermodynamics\b', r'\bquantum\b',
                    r'\brelativity\b', r'\bforce\b', r'\bmomentum\b', r'\benergy\b',
                    r'\belectric\b', r'\bmagnetic\b', r'\bwave\b', r'\bparticle\b'
                ],
                "ru": [
                    r'\bфизик[аеиу]\b', r'\bмеханик[аеиу]\b', r'\bдинамик[аеиу]\b',
                    r'\bкинематик[аеиу]\b', r'\bэлектромагнетизм[аеу]?\b',
                    r'\bтермодинамик[аеиу]\b', r'\bквантов[аяыеую]\b',
                    r'\bотносительност[иью]\b', r'\bсил[аыу]\b', r'\bимпульс[аеу]?\b',
                    r'\bэнерги[яию]\b', r'\bэлектрическ[а-я]+\b', r'\bмагнитн[а-я]+\b',
                    r'\bволн[аыуе]\b', r'\bчастиц[аыуе]\b'
                ]
            }
        }

        # Count matches for each domain
        lang_key = "ru" if language == "ru" else "en"
        domain_counts = {}

        for domain, patterns in domains.items():
            lang_patterns = patterns.get(lang_key, patterns.get("en", []))
            count = 0

            for pattern in lang_patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))

            domain_counts[domain] = count

        # Find domain with highest count
        max_count = max(domain_counts.values())

        if max_count == 0:
            return "unknown", 0.0

        # Get domains with max count
        max_domains = [domain for domain, count in domain_counts.items() if count == max_count]

        if len(max_domains) == 1:
            domain = max_domains[0]
            total = sum(domain_counts.values())
            confidence = max_count / total if total > 0 else 0.0
            return domain, confidence
        else:
            return max_domains[0], 0.5

    def _normalize_english_text(self, text: str) -> str:
        """Normalize English text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix common OCR/caption errors
        text = text.replace(" i ", " I ")
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Add space after period

        # Remove speaker identifiers like "[Professor]:"
        text = re.sub(r'\[\w+\]:', '', text)

        # Fix ellipses
        text = re.sub(r'\.\.\.+', '...', text)

        # Remove musical notes, applause indicators, etc.
        text = re.sub(r'\[.*?\]', '', text)

        return text.strip()

    def _normalize_russian_text(self, text: str) -> str:
        """Normalize Russian text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix common OCR/caption errors with Cyrillic characters
        text = text.replace('ё', 'е')  # Standardize 'ё' to 'е'

        # Fix punctuation with Cyrillic characters
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Add space after period

        # Remove speaker identifiers like "[Профессор]:"
        text = re.sub(r'\[\w+\]:', '', text)

        # Fix ellipses
        text = re.sub(r'\.\.\.+', '...', text)

        # Remove musical notes, applause indicators, etc.
        text = re.sub(r'\[.*?\]', '', text)

        return text.strip()

    def _tokenize_russian_sentences(self, text: str) -> List[str]:
        """Tokenize Russian text into sentences."""
        # Handle Russian-specific sentence boundaries
        text = re.sub(r'([.!?])([^А-Яа-яЁё])', r'\1 \2', text)

        # Use NLTK's sent_tokenize which has Russian support
        try:
            sentences = sent_tokenize(text, language='russian')
        except LookupError:
            # Fallback if NLTK data is missing - simple splitting on punctuation
            sentences = re.split(r'(?<=[.!?]) +', text)
            if not sentences:
                sentences = [text]

        return sentences

    def _get_section_patterns(self, language: str) -> List[str]:
        """Get patterns for section detection based on language."""
        if language == "ru":
            return [
                r'^(глава\s+\d+[.:)]?\s+.+)$',
                r'^(раздел\s+\d+[.:)]?\s+.+)$',
                r'^(\d+[.:)]\s+.+)$',
                r'^(тема\s*[:]\s*(.+))$',
                r'^(введение|заключение)$',
                r'(перейдем|давайте рассмотрим|теперь|следующая тема)'
            ]
        else:
            return [
                r'^(chapter\s+\d+[.:)]?\s+.+)$',
                r'^(section\s+\d+[.:)]?\s+.+)$',
                r'^(\d+[.:)]\s+.+)$',
                r'^(topic\s*[:]\s*(.+))$',
                r'^(introduction|conclusion)$',
                r'(let\'s move on to|let\'s look at|next|the next topic)'
            ]

    def _create_section(self, section_info: Dict, segments: List[Dict], language: str) -> Dict:
        """Create a section dictionary from section info and segments."""
        # Set end time to last segment's end time
        section_info["end_time"] = segments[-1].get("end_time", 0)

        # Add segment IDs to section
        section_info["segments"] = [segment.get("id") for segment in segments]

        # Generate section summary if no title
        if not section_info.get("title"):
            # Use first segment as title if it's short enough
            if len(segments[0].get("text", "")) < 60:
                section_info["title"] = segments[0].get("text")
            else:
                # Otherwise, generate a generic title
                if language == "ru":
                    section_info["title"] = f"Раздел в {section_info['start_time']:.1f}с"
                else:
                    section_info["title"] = f"Section at {section_info['start_time']:.1f}s"

        # Determine domain for this section
        section_text = " ".join([segment.get("text", "") for segment in segments])
        domain, _ = self.classify_domain(section_text, language)
        section_info["domain"] = domain

        # Determine content type (theoretical/practical/mixed)
        theory_count = sum(1 for segment in segments if segment.get("content_type") == "theoretical")
        practice_count = sum(1 for segment in segments if segment.get("content_type") == "practical")

        if theory_count > practice_count * 2:
            content_type = "theoretical"
        elif practice_count > theory_count * 2:
            content_type = "practical"
        else:
            content_type = "mixed"

        section_info["content_type"] = content_type

        return section_info

    def _process_english_nlp(self, text: str) -> Dict:
        """Process English text with NLP."""
        if not self.en_nlp:
            return {
                "pos_tags": [],
                "entities": [],
                "discourse_markers": []
            }

        doc = self.en_nlp(text)

        # Extract POS tags
        pos_tags = [{"token": token.text, "pos": token.pos_} for token in doc]

        # Extract entities
        entities = [
            {
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            } for ent in doc.ents
        ]

        # Extract discourse markers
        discourse_markers = []
        discourse_patterns = [
            r'\b(first(ly)?|second(ly)?|third(ly)?|finally)\b',
            r'\b(however|nevertheless|therefore|thus|consequently)\b',
            r'\b(in contrast|on the other hand|in summary|to summarize)\b',
            r'\b(for example|for instance|such as|in particular)\b'
        ]

        for pattern in discourse_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                discourse_markers.append(match.group(0))

        return {
            "pos_tags": pos_tags,
            "entities": entities,
            "discourse_markers": discourse_markers
        }

    def _process_russian_nlp(self, text: str) -> Dict:
        """Process Russian text with NLP."""
        if not self.ru_nlp:
            return {
                "pos_tags": [],
                "entities": [],
                "discourse_markers": []
            }

        doc = self.ru_nlp(text)

        # Extract POS tags
        pos_tags = [{"token": token.text, "pos": token.pos_} for token in doc]

        # Extract entities
        entities = [
            {
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            } for ent in doc.ents
        ]

        # Extract discourse markers (Russian)
        discourse_markers = []
        discourse_patterns = [
            r'\b(во-первых|во-вторых|в-третьих|наконец)\b',
            r'\b(однако|тем не менее|следовательно|таким образом)\b',
            r'\b(в отличие от|с другой стороны|в итоге|подводя итог)\b',
            r'\b(например|к примеру|такие как|в частности)\b'
        ]

        for pattern in discourse_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                discourse_markers.append(match.group(0))

        return {
            "pos_tags": pos_tags,
            "entities": entities,
            "discourse_markers": discourse_markers
        }

    def _extract_formulas(self, text: str) -> List[Dict]:
        """Extract mathematical formulas from text."""
        formulas = []

        # Look for LaTeX-like formulas
        latex_patterns = [
            r'\$(.+?)\$',  # Inline math
            r'\\\((.+?)\\\)',  # Inline math alternative
            r'\\\[(.+?)\\\]',  # Display math
            r'\$\$(.+?)\$\$'   # Display math alternative
        ]

        for pattern in latex_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                formulas.append({
                    "text": match.group(0),
                    "formula": match.group(1) if match.groups() else match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })

        # Look for equation patterns
        equation_patterns = [
            r'([a-zA-Z][a-zA-Z0-9]*\s*=\s*[^.,;:]+)',  # Simple equations like y = mx + b
            r'([a-zA-Z][a-zA-Z0-9]*\([a-zA-Z0-9,\s]+\)\s*=\s*[^.,;:]+)'  # Functions like f(x) = x^2
        ]

        for pattern in equation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Avoid overlapping with already detected LaTeX formulas
                overlap = False
                for formula in formulas:
                    if match.start() >= formula["start"] and match.start() < formula["end"]:
                        overlap = True
                        break

                if not overlap:
                    formulas.append({
                        "text": match.group(0),
                        "formula": match.group(1) if match.groups() else match.group(0),
                        "start": match.start(),
                        "end": match.end()
                    })

        return formulas

    def _extract_code_snippets(self, text: str) -> List[Dict]:
        """Extract code snippets from text."""
        snippets = []

        # Look for code blocks
        code_block_patterns = [
            r'```([a-zA-Z0-9]*)\n(.*?)```',  # Markdown code blocks
            r'<code(?:\s+class="([a-zA-Z0-9]*)")?>(.+?)</code>'  # HTML code tags
        ]

        for pattern in code_block_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                language = match.group(1).strip() if match.group(1) else "unknown"
                code = match.group(2)

                snippets.append({
                    "text": match.group(0),
                    "code": code,
                    "language": language,
                    "start": match.start(),
                    "end": match.end()
                })

        # Look for inline code patterns
        inline_patterns = [
            r'`([^`]+)`'  # Inline code
        ]

        for pattern in inline_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Avoid overlapping with already detected code blocks
                overlap = False
                for snippet in snippets:
                    if match.start() >= snippet["start"] and match.start() < snippet["end"]:
                        overlap = True
                        break

                if not overlap:
                    snippets.append({
                        "text": match.group(0),
                        "code": match.group(1),
                        "language": "unknown",
                        "start": match.start(),
                        "end": match.end()
                    })

        return snippets

    def _classify_sentence_type(self, text: str, language: str) -> str:
        """Classify the type of sentence."""
        # Patterns for different sentence types
        if language == "ru":
            patterns = {
                "definition": [
                    r'([а-яА-ЯёЁ\s]+) (?:называется|определяется как|это) ',
                    r'([а-яА-ЯёЁ\s]+) — это ',
                    r'определение (?:понятия )?([а-яА-ЯёЁ\s]+)'
                ],
                "explanation": [
                    r'(?:рассмотрим|объясним|разберем)',
                    r'(?:смысл|суть|идея) (?:заключается|состоит)'
                ],
                "example": [
                    r'(?:например|к примеру|в качестве примера)',
                    r'рассмотрим пример'
                ],
                "problem_statement": [
                    r'(?:задача|проблема|вопрос|требуется)',
                    r'(?:найти|вычислить|определить|доказать)',
                    r'(?:решите|найдите|вычислите)'  # Added imperative forms
                ],
                "solution": [
                    r'(?:решение|решим|решаем)',
                    r'(?:сначала|затем|далее|наконец)'
                ],
                "proof": [
                    r'(?:доказательство|докажем|доказать)',
                    r'(?:предположим|допустим|пусть)'
                ]
            }
        else:
            patterns = {
                "definition": [
                    r'([a-zA-Z\s]+) is defined as ',
                    r'([a-zA-Z\s]+) is ',
                    r'definition of ([a-zA-Z\s]+)'
                ],
                "explanation": [
                    r'(?:let\'s|we will) (?:consider|explain|examine)',
                    r'the (?:concept|idea|meaning) (?:is|consists)'
                ],
                "example": [
                    r'(?:for example|for instance|as an example)',
                    r'let\'s (?:look at|consider) an example'
                ],
                "problem_statement": [
                    r'(?:problem|question|task|we need to)',
                    r'(?:find|calculate|determine|prove)',
                    r'(?:solve|to solve|solving)',  # Added to match test expectation
                    r'(?:compute|evaluate)'
                ],
                "solution": [
                    r'(?:solution|solved|solving)',
                    r'^(?:first|then|next|finally)'  # Starting sentence patterns
                ],
                "proof": [
                    r'(?:proof|prove|proving)',
                    r'(?:assume|suppose|let)'
                ]
            }

        # Check each pattern
        for sentence_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return sentence_type

        # Default to "explanation" if no specific type is detected
        return "explanation"

    def _classify_content_type(self, text: str, sentence_type: str, language: str) -> str:
        """Classify content as theoretical or practical."""
        # Some sentence types are inherently theoretical or practical
        if sentence_type in ["definition", "proof"]:
            return "theoretical"
        elif sentence_type in ["problem_statement", "solution"]:
            return "practical"

        # For other sentence types, check for specific markers
        if language == "ru":
            theoretical_markers = [
                r'(?:теория|концепция|понятие|определение)',
                r'(?:теоретически|в теории|концептуально)',
                r'(?:доказательство|теорема|аксиома|лемма)',
                r'(?:формализация|формализм|формальность)',
                r'(?:философия|философский)'
            ]

            practical_markers = [
                r'(?:практика|применение|использование)',
                r'(?:на практике|практически|в реальности)',
                r'(?:решение задачи|пример|задание)',
                r'(?:вычислить|рассчитать|найти)',
                r'(?:код|программа|алгоритм)'
            ]
        else:
            theoretical_markers = [
                r'(?:theory|concept|notion|definition)',
                r'(?:theoretically|in theory|conceptually)',
                r'(?:proof|theorem|axiom|lemma)',
                r'(?:formalization|formalism|formality)',
                r'(?:philosophy|philosophical)'
            ]

            practical_markers = [
                r'(?:practice|application|usage)',
                r'(?:in practice|practically|in reality)',
                r'(?:problem solving|example|exercise)',
                r'(?:calculate|compute|find)',
                r'(?:code|program|algorithm)'
            ]

        # Count theoretical and practical markers
        theoretical_count = 0
        for pattern in theoretical_markers:
            theoretical_count += len(re.findall(pattern, text, re.IGNORECASE))

        practical_count = 0
        for pattern in practical_markers:
            practical_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Classify based on markers
        if theoretical_count > practical_count:
            return "theoretical"
        elif practical_count > theoretical_count:
            return "practical"
        else:
            # If equal or no markers, check for code snippets and formulas as indicators
            if self._extract_code_snippets(text):
                return "practical"
            elif self._extract_formulas(text):
                # Formulas could be either theoretical or practical
                # Check for specific practical formula usage patterns
                if language == "ru":
                    practical_formula_patterns = [
                        r'(?:подставим|вычислим|рассчитаем)',
                        r'(?:получаем|получим|находим)'
                    ]
                else:
                    practical_formula_patterns = [
                        r'(?:substitute|calculate|compute)',
                        r'(?:we get|we find|we obtain)'
                    ]

                for pattern in practical_formula_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return "practical"

                return "theoretical"
            else:
                # Default to "mixed" if we can't determine
                return "mixed"
