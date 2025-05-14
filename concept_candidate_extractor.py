"""
Concept candidate extractor for the Video Lecture Content Indexer.
Identifies potential new concepts from video transcripts using NLP techniques
and prepares them for user review and addition to the concept repository.
"""

import re
import uuid
import logging
import json
import os
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter
import time
from datetime import datetime

# NLP libraries with graceful degradation
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, ne_chunk
    from nltk.chunk import RegexpParser
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available - using simplified candidate extraction")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - using simplified candidate extraction")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - using alternative extraction methods")

# Import project modules
from concept_repository import get_concept_repository
from data_access import get_data_access
from performance_utils import time_function

# Configure logging
logger = logging.getLogger(__name__)

class ConceptCandidateExtractor:
    """
    Extracts potential new concepts from video transcripts.

    Uses NLP techniques to identify important terms and phrases that are not
    already in the concept repository, and prepares them for user review.
    """

    def __init__(self, candidates_dir: str = "concept_candidates"):
        """
        Initialize the concept candidate extractor.

        Args:
            candidates_dir: Directory to store concept candidates
        """
        self.candidates_dir = candidates_dir
        self.concept_repository = get_concept_repository()
        self.data_access = get_data_access()
        self.candidates = {}  # candidate_id -> candidate data

        # Ensure candidates directory exists
        os.makedirs(self.candidates_dir, exist_ok=True)

        # Initialize NLP components
        self._init_nlp_components()

        # Load existing candidates
        self._load_candidates()

        logger.info(f"ConceptCandidateExtractor initialized with {len(self.candidates)} existing candidates")

    def _init_nlp_components(self):
        """Initialize NLP components for concept extraction."""
        # Initialize NLTK resources if available
        if NLTK_AVAILABLE:
            try:
                # Download necessary resources
                for resource in ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords']:
                    try:
                        nltk.data.find(f'tokenizers/{resource}')
                    except LookupError:
                        nltk.download(resource, quiet=True)

                # Initialize stopwords and lemmatizer
                self.stopwords_en = set(stopwords.words('english'))
                self.stopwords_ru = set()
                try:
                    self.stopwords_ru = set(stopwords.words('russian'))
                except:
                    pass

                self.lemmatizer = WordNetLemmatizer()

                # Define grammar for noun phrase extraction
                self.grammar = r"""
                    NP: {<DT|PP\$>?<JJ>*<NN|NNS|NNP|NNPS>+}   # Noun phrase
                    CP: {<JJ>+<NN|NNS|NNP|NNPS>}              # Adjective + Noun
                """
                self.chunk_parser = RegexpParser(self.grammar)

                logger.info("NLTK components initialized for concept extraction")
            except Exception as e:
                logger.warning(f"Error initializing NLTK components: {e}")

        # Initialize spaCy models if available
        if SPACY_AVAILABLE:
            self.nlp_models = {}
            try:
                for lang, model in [('en', 'en_core_web_sm'), ('ru', 'ru_core_news_sm')]:
                    try:
                        self.nlp_models[lang] = spacy.load(model)
                        logger.info(f"Loaded spaCy model: {model}")
                    except OSError:
                        logger.warning(f"spaCy model {model} not found")
            except Exception as e:
                logger.warning(f"Error initializing spaCy models: {e}")

        # Initialize scikit-learn components if available
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_df=0.95,
                min_df=2,
                max_features=1000,
                stop_words='english'
            )
            logger.info("scikit-learn components initialized for concept extraction")

    def _load_candidates(self):
        """Load existing concept candidates from disk."""
        if not os.path.exists(self.candidates_dir):
            return

        try:
            # Find all candidate files
            candidate_files = [f for f in os.listdir(self.candidates_dir) if f.endswith('.json')]

            for file_name in candidate_files:
                try:
                    with open(os.path.join(self.candidates_dir, file_name), 'r', encoding='utf-8') as f:
                        candidate = json.load(f)
                        candidate_id = candidate.get('candidate_id')
                        if candidate_id:
                            self.candidates[candidate_id] = candidate
                except Exception as e:
                    logger.warning(f"Error loading candidate file {file_name}: {e}")

            logger.info(f"Loaded {len(self.candidates)} concept candidates")
        except Exception as e:
            logger.error(f"Error loading candidates: {e}")

    def _save_candidate(self, candidate: Dict[str, Any]) -> bool:
        """
        Save a concept candidate to disk.

        Args:
            candidate: Candidate dictionary

        Returns:
            True if successful, False otherwise
        """
        candidate_id = candidate.get('candidate_id')
        if not candidate_id:
            logger.warning("Cannot save candidate without candidate_id")
            return False

        try:
            file_path = os.path.join(self.candidates_dir, f"{candidate_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(candidate, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved candidate: {candidate_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving candidate: {e}")
            return False

    @time_function(5000)  # Log warning if takes more than 5 seconds
    def extract_concept_candidates(
        self,
        processed_transcript: Dict[str, Any],
        video_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract potential new concept candidates from a processed transcript.

        Args:
            processed_transcript: Processed transcript from TranscriptProcessor
            video_metadata: Video metadata

        Returns:
            List of new concept candidates
        """
        video_id = video_metadata.get('video_id', 'unknown')
        language = processed_transcript.get('language', 'en')
        domain = processed_transcript.get('domain', 'unknown')
        global_analysis = processed_transcript.get('global_analysis', {})

        logger.info(f"Extracting concept candidates from video {video_id} ({language}, {domain})")

        # Get existing concepts for comparison
        existing_concepts = self._get_existing_concepts(language)

        # Get segments text
        segments = processed_transcript.get('segments', [])
        segment_texts = [segment.get('text', '') for segment in segments]

        # Combine all text for global analysis
        full_text = " ".join(segment_texts)

        # Extract candidates using available methods
        candidates = []

        # Method 1: Extract using spaCy if available (preferred for quality)
        if SPACY_AVAILABLE and language in self.nlp_models:
            spacy_candidates = self._extract_candidates_with_spacy(
                full_text, segments, language, domain
            )
            candidates.extend(spacy_candidates)

        # Method 2: Extract using NLTK if available
        elif NLTK_AVAILABLE:
            nltk_candidates = self._extract_candidates_with_nltk(
                full_text, segments, language, domain
            )
            candidates.extend(nltk_candidates)

        # Method 3: Use TF-IDF to extract important terms
        if SKLEARN_AVAILABLE:
            tfidf_candidates = self._extract_candidates_with_tfidf(
                full_text, segments, language, domain
            )
            candidates.extend(tfidf_candidates)

        # Method 4: Fallback to simple regex extraction
        if not candidates:
            regex_candidates = self._extract_candidates_with_regex(
                full_text, segments, language, domain
            )
            candidates.extend(regex_candidates)

        # Filter out candidates already in the repository
        filtered_candidates = self._filter_existing_concepts(candidates, existing_concepts)

        # Filter out duplicates within the candidates
        deduplicated_candidates = self._deduplicate_candidates(filtered_candidates)

        # Score and rank candidates
        scored_candidates = self._score_candidates(deduplicated_candidates, global_analysis)

        # Sort by score
        scored_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Take the top candidates
        top_candidates = scored_candidates[:30]  # Limit to reasonable number

        # Format candidate data and add metadata
        new_candidates = []
        for candidate in top_candidates:
            candidate_id = str(uuid.uuid4())

            # Format concept representations - ensure lowercase
            text = candidate['text'].lower()
            representations = {}
            if language in ['en', 'ru']:
                representations[language] = [text]

            # Create candidate with repository-compatible structure
            new_candidate = {
                'candidate_id': candidate_id,
                'text': text,
                'score': candidate['score'],
                'source_video_id': video_id,
                'source_video_title': video_metadata.get('title', ''),
                'source_segments': candidate.get('source_segments', []),
                'extraction_method': candidate.get('extraction_method', 'unknown'),
                'language': language,
                'domain': domain,
                'status': 'pending',  # pending, approved, rejected
                'created_at': datetime.now().isoformat(),

                # Repository-compatible structure
                'concept_data': {
                    'concept_id': '',  # Will be filled in when approved
                    'representations': representations,
                    'prerequisites': [],
                    'related': [],
                    'metadata': {
                        'source_video_id': video_id,
                        'domain': domain,
                        'created_at': datetime.now().isoformat()
                    }
                }
            }

            # Add to candidates list
            self.candidates[candidate_id] = new_candidate

            # Save to disk
            self._save_candidate(new_candidate)

            # Add to result list
            new_candidates.append(new_candidate)

        logger.info(f"Extracted {len(new_candidates)} new concept candidates from video {video_id}")
        return new_candidates

    def _extract_candidates_with_spacy(
        self,
        full_text: str,
        segments: List[Dict[str, Any]],
        language: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Extract concept candidates using spaCy NLP models.

        Args:
            full_text: Full transcript text
            segments: List of transcript segments
            language: Language code
            domain: Content domain

        Returns:
            List of candidate dictionaries
        """
        if not SPACY_AVAILABLE or language not in self.nlp_models:
            return []

        candidates = []
        nlp = self.nlp_models[language]

        try:
            # Process in chunks for very large texts
            if len(full_text) > 100000:
                # Process each segment instead of the full text
                for segment in segments:
                    text = segment.get('text', '')
                    if not text.strip():
                        continue

                    doc = nlp(text)
                    segment_candidates = self._extract_spacy_candidates(
                        doc,
                        language,
                        domain,
                        segment.get('id')
                    )
                    candidates.extend(segment_candidates)
            else:
                # Process full text at once
                doc = nlp(full_text)
                candidates = self._extract_spacy_candidates(doc, language, domain)

            return candidates
        except Exception as e:
            logger.warning(f"Error extracting candidates with spaCy: {e}")
            return []

    def _extract_spacy_candidates(
        self,
        doc: Any,
        language: str,
        domain: str,
        segment_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract candidates from a spaCy Doc object.

        Args:
            doc: spaCy Doc object
            language: Language code
            domain: Content domain
            segment_id: Optional segment ID

        Returns:
            List of candidate dictionaries
        """
        candidates = []

        # Track all noun phrases
        noun_phrases = []

        # Extract noun chunks (available for English)
        try:
            for chunk in doc.noun_chunks:
                if len(chunk.text.strip()) >= 3:  # Skip very short phrases
                    noun_phrases.append(chunk.text)
        except:
            pass

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'GPE', 'LOC', 'EVENT']:
                noun_phrases.append(ent.text)

        # For Russian specifically: get adjective+noun combinations
        if language == 'ru':
            for i, token in enumerate(doc):
                if token.pos_ == 'NOUN' and i > 0:
                    prev_token = doc[i-1]
                    if prev_token.pos_ == 'ADJ':
                        noun_phrases.append(f"{prev_token.text} {token.text}")

        # Process each noun phrase
        for text in set(noun_phrases):
            if self._is_valid_candidate(text, language, domain):
                source_segments = []
                if segment_id:
                    source_segments.append(segment_id)

                candidates.append({
                    'text': text,
                    'extraction_method': 'spacy_noun_phrase',
                    'source_segments': source_segments,
                    'score': 0.0  # Will be scored later
                })

        return candidates

    def _extract_candidates_with_nltk(
        self,
        full_text: str,
        segments: List[Dict[str, Any]],
        language: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Extract concept candidates using NLTK.

        Args:
            full_text: Full transcript text
            segments: List of transcript segments
            language: Language code
            domain: Content domain

        Returns:
            List of candidate dictionaries
        """
        if not NLTK_AVAILABLE:
            return []

        candidates = []

        try:
            # Process based on language
            if language == 'en':
                sentences = sent_tokenize(full_text)

                # Process each sentence
                for i, sentence in enumerate(sentences):
                    # Tokenize, tag and chunk
                    tokens = word_tokenize(sentence)
                    tagged = pos_tag(tokens)
                    chunked = self.chunk_parser.parse(tagged)

                    # Extract noun phrases
                    for subtree in chunked.subtrees():
                        if subtree.label() in ('NP', 'CP'):  # Noun phrase or compound
                            np_text = ' '.join([word for word, tag in subtree.leaves()])

                            # Skip very short phrases and phrases with only stopwords
                            if len(np_text) < 3 or all(word in self.stopwords_en for word in np_text.split()):
                                continue

                            # Find source segment
                            source_segments = []
                            for segment in segments:
                                if sentence in segment.get('text', ''):
                                    source_segments.append(segment.get('id'))
                                    break

                            if self._is_valid_candidate(np_text, language, domain):
                                candidates.append({
                                    'text': np_text,
                                    'extraction_method': 'nltk_noun_phrase',
                                    'source_segments': source_segments,
                                    'score': 0.0  # Will be scored later
                                })

            # For Russian, use a simplified approach (NLTK has limited Russian support)
            elif language == 'ru':
                # Fall back to regex approach for Russian with NLTK
                return self._extract_candidates_with_regex(full_text, segments, language, domain)

            return candidates
        except Exception as e:
            logger.warning(f"Error extracting candidates with NLTK: {e}")
            return []

    def _extract_candidates_with_tfidf(
        self,
        full_text: str,
        segments: List[Dict[str, Any]],
        language: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Extract concept candidates using TF-IDF analysis.

        Args:
            full_text: Full transcript text
            segments: List of transcript segments
            language: Language code
            domain: Content domain

        Returns:
            List of candidate dictionaries
        """
        if not SKLEARN_AVAILABLE:
            return []

        candidates = []

        try:
            # Configure vectorizer based on language
            stop_words = 'english' if language == 'en' else None

            # Create custom analyzer for n-grams that preserves meaningful phrases
            def custom_analyzer(text):
                # Basic tokenization
                tokens = re.findall(r'\b\w+\b', text.lower())

                # Filter stopwords
                if language == 'en':
                    filtered_tokens = [t for t in tokens if t not in self.stopwords_en]
                elif language == 'ru':
                    filtered_tokens = [t for t in tokens if t not in self.stopwords_ru]
                else:
                    filtered_tokens = tokens

                # Generate n-grams
                result = []
                # Add unigrams
                result.extend(filtered_tokens)

                # Add bigrams
                for i in range(len(tokens) - 1):
                    if tokens[i] not in self.stopwords_en and tokens[i+1] not in self.stopwords_en:
                        result.append(f"{tokens[i]} {tokens[i+1]}")

                # Add trigrams
                for i in range(len(tokens) - 2):
                    if tokens[i] not in self.stopwords_en and tokens[i+2] not in self.stopwords_en:
                        result.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

                return result

            # Create vectorizer
            vectorizer = TfidfVectorizer(
                max_df=0.95,
                min_df=2,
                max_features=1000,
                stop_words=stop_words,
                analyzer=custom_analyzer
            )

            # Prepare segment texts
            segment_texts = [segment.get('text', '') for segment in segments if segment.get('text')]

            # Skip if not enough segments
            if len(segment_texts) < 3:
                return []

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(segment_texts)

            # Get feature names
            try:
                feature_names = vectorizer.get_feature_names_out()
            except:
                # For older sklearn versions
                feature_names = vectorizer.get_feature_names()

            # Get average TF-IDF score for each term
            avg_scores = tfidf_matrix.mean(axis=0).tolist()[0]

            # Get top terms
            term_scores = list(zip(feature_names, avg_scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top 50 terms
            top_terms = term_scores[:50]

            # Create candidates from top terms
            for term, score in top_terms:
                # Skip very short terms
                if len(term) < 3:
                    continue

                # Find source segments
                source_segments = []
                for i, segment in enumerate(segments):
                    if term.lower() in segment.get('text', '').lower():
                        source_segments.append(segment.get('id'))

                if self._is_valid_candidate(term, language, domain):
                    candidates.append({
                        'text': term,
                        'extraction_method': 'tfidf',
                        'source_segments': source_segments,
                        'score': float(score * 3.0)  # Scale up TF-IDF score
                    })

            return candidates
        except Exception as e:
            logger.warning(f"Error extracting candidates with TF-IDF: {e}")
            return []

    def _extract_candidates_with_regex(
        self,
        full_text: str,
        segments: List[Dict[str, Any]],
        language: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Extract concept candidates using regex patterns.

        Args:
            full_text: Full transcript text
            segments: List of transcript segments
            language: Language code
            domain: Content domain

        Returns:
            List of candidate dictionaries
        """
        candidates = []

        try:
            # Define extraction patterns based on language
            if language == 'en':
                # Patterns for English concepts
                patterns = [
                    r'(?:the|a|an)?\s+([A-Z][a-z]+(?:\s+[a-z]+){1,3})',  # Capitalized phrases
                    r'(?:called|known as|termed)(?:\s+the|a|an)?\s+([a-z]+(?:\s+[a-z]+){1,3})',  # Definition phrases
                    r'(?:the|a|an)?\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:concept|principle|law|theory|equation|method)',  # Concept indicators
                    r'([A-Z][a-z]*(?:\'s)?\s+(?:law|principle|equation|theory|method))'  # Named concepts
                ]
            elif language == 'ru':
                # Patterns for Russian concepts
                patterns = [
                    r'(?:понятие|концепция|принцип|закон|теория|уравнение|метод)\s+([а-яА-ЯёЁ]+(?:\s+[а-яА-ЯёЁ]+){0,3})',  # Definition phrases
                    r'([а-яА-ЯёЁ]+(?:\s+[а-яА-ЯёЁ]+){0,2})\s+(?:понятие|концепция|принцип|закон|теория|уравнение|метод)',  # Concept indicators
                    r'([А-ЯЁ][а-яё]*(?:\s+[а-яёА-ЯЁ]+){1,3})'  # Capitalized phrases
                ]
            else:
                # Default to English patterns
                patterns = [
                    r'(?:the|a|an)?\s+([A-Z][a-z]+(?:\s+[a-z]+){1,3})'
                ]

            # Process each segment
            for segment in segments:
                segment_text = segment.get('text', '')
                segment_id = segment.get('id')

                if not segment_text or not segment_id:
                    continue

                # Apply each pattern
                for pattern in patterns:
                    matches = re.finditer(pattern, segment_text)

                    for match in matches:
                        # Get the matched group (actual concept text)
                        concept_text = match.group(1).strip()

                        # Skip very short concepts or stopwords only
                        if len(concept_text) < 3:
                            continue

                        if language == 'en' and all(word.lower() in self.stopwords_en for word in concept_text.split()):
                            continue

                        if language == 'ru' and all(word.lower() in self.stopwords_ru for word in concept_text.split()):
                            continue

                        # Add as candidate
                        if self._is_valid_candidate(concept_text, language, domain):
                            candidates.append({
                                'text': concept_text,
                                'extraction_method': 'regex',
                                'source_segments': [segment_id],
                                'score': 0.0  # Will be scored later
                            })

            return candidates
        except Exception as e:
            logger.warning(f"Error extracting candidates with regex: {e}")
            return []

    def _is_valid_candidate(self, text: str, language: str, domain: str) -> bool:
        """
        Check if a text is a valid concept candidate.

        Args:
            text: Candidate text
            language: Language code
            domain: Content domain

        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        if not text or len(text) < 3:
            return False

        # Skip candidates that are just numbers
        if re.match(r'^\d+$', text):
            return False

        # Check word count - concepts usually have 1-5 words
        word_count = len(text.split())
        if word_count > 5:
            return False

        # For English, check for common stopword-only phrases
        if language == 'en':
            # Skip if only consists of stopwords
            if word_count > 0 and all(word.lower() in self.stopwords_en for word in text.split()):
                return False

            # Skip generic phrases that aren't domain concepts
            generic_phrases = [
                'this case', 'this example', 'this part', 'this time',
                'this way', 'that way', 'the first', 'the second',
                'the next', 'the last', 'the following', 'the above',
                'that time', 'these cases', 'those cases'
            ]

            if text.lower() in generic_phrases:
                return False

        # For Russian, check for Russian stopword-only phrases
        if language == 'ru':
            # Skip if only consists of stopwords
            if word_count > 0 and all(word.lower() in self.stopwords_ru for word in text.split()):
                return False

            # Skip generic phrases that aren't domain concepts
            generic_phrases = [
                'этот случай', 'этот пример', 'эта часть', 'это время',
                'таким образом', 'первый', 'второй', 'следующий', 'последний'
            ]

            if text.lower() in generic_phrases:
                return False

        # Skip common phrases from math lectures that aren't concepts
        common_math_phrases = [
            'this equation', 'this formula', 'this expression',
            'left side', 'right side', 'next step', 'first step'
        ]

        if language == 'en' and domain == 'mathematics' and text.lower() in common_math_phrases:
            return False

        return True

    def _get_existing_concepts(self, language: Optional[str] = None) -> Set[str]:
        """
        Get normalized text of existing concepts for comparison.

        Args:
            language: Optional language filter

        Returns:
            Set of normalized concept texts
        """
        existing_concepts = set()

        # Get all concepts from repository
        concepts = self.concept_repository.list_concepts(language=language, limit=10000)

        # Extract all representations
        for concept in concepts:
            representations = concept.get('representations', {})

            for lang, texts in representations.items():
                if language and lang != language:
                    continue

                for text in texts:
                    # Normalize text
                    normalized = self._normalize_text(text, lang)
                    if normalized:
                        existing_concepts.add(normalized)

        # Also add pending candidates
        for candidate in self.candidates.values():
            if candidate.get('status') != 'rejected':
                text = candidate.get('text', '')
                lang = candidate.get('language', 'en')

                if language and lang != language:
                    continue

                normalized = self._normalize_text(text, lang)
                if normalized:
                    existing_concepts.add(normalized)

        return existing_concepts

    def _normalize_text(self, text: str, language: str = 'en') -> str:
        """
        Normalize text for comparison.

        Args:
            text: Text to normalize
            language: Language code

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Language-specific normalization
        if language == 'ru':
            # Russian normalization
            normalized = normalized.replace('ё', 'е')

        return normalized

    def _filter_existing_concepts(
        self,
        candidates: List[Dict[str, Any]],
        existing_concepts: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter out candidates that match existing concepts.

        Args:
            candidates: List of candidate dictionaries
            existing_concepts: Set of normalized existing concept texts

        Returns:
            Filtered list of candidates
        """
        filtered = []

        for candidate in candidates:
            text = candidate.get('text', '')
            language = candidate.get('language', 'en')

            # Normalize for comparison
            normalized = self._normalize_text(text, language)

            # Skip if already exists
            if normalized in existing_concepts:
                continue

            # Add to filtered list
            filtered.append(candidate)

        return filtered

    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate candidates.

        Args:
            candidates: List of candidate dictionaries

        Returns:
            Deduplicated list of candidates
        """
        deduplicated = []
        seen_texts = set()

        for candidate in candidates:
            text = candidate.get('text', '')
            language = candidate.get('language', 'en')

            # Normalize for comparison
            normalized = self._normalize_text(text, language)

            # Skip if already seen
            if normalized in seen_texts:
                continue

            # Add to deduplicated list
            deduplicated.append(candidate)
            seen_texts.add(normalized)

        return deduplicated

    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        global_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Score and rank candidate concepts.

        Args:
            candidates: List of candidate dictionaries
            global_analysis: Global text analysis results

        Returns:
            List of scored candidates
        """
        # Get key terms from global analysis
        key_terms = global_analysis.get('key_terms', [])

        for candidate in candidates:
            # Get base score from extraction method
            score = candidate.get('score', 0.0)
            method = candidate.get('extraction_method', '')

            # Adjust score based on extraction method
            if method == 'spacy_noun_phrase':
                score += 2.0  # spaCy is generally most accurate
            elif method == 'nltk_noun_phrase':
                score += 1.5  # NLTK is good but less accurate than spaCy
            elif method == 'tfidf':
                score += 1.0  # TF-IDF is already scored appropriately
            elif method == 'regex':
                score += 0.5  # Regex is least reliable

            # Adjust score based on text length
            text = candidate.get('text', '')
            word_count = len(text.split())

            if word_count == 1:
                score += 0.5  # Single words are often important terms
            elif word_count == 2:
                score += 1.0  # Two-word phrases are common for concepts
            elif word_count == 3:
                score += 0.8  # Three-word phrases can be good concepts
            else:
                score += 0.3  # Longer phrases are less likely to be concepts

            # Bonus for capitalized terms (often proper names of concepts)
            if text[0].isupper() and not text.isupper():
                score += 0.5

            # Check if candidate overlaps with key terms
            for term in key_terms:
                if term.lower() in text.lower() or text.lower() in term.lower():
                    score += 1.0
                    break

            # Update candidate score
            candidate['score'] = score

        return candidates

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific candidate by ID.

        Args:
            candidate_id: Candidate ID

        Returns:
            Candidate dictionary or None if not found
        """
        return self.candidates.get(candidate_id)

    def list_candidates(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List candidates with optional filtering.

        Args:
            status: Optional filter by status
            limit: Maximum number of candidates to return
            offset: Pagination offset

        Returns:
            List of candidates
        """
        # Filter by status if specified
        if status:
            filtered = [c for c in self.candidates.values() if c.get('status') == status]
        else:
            filtered = list(self.candidates.values())

        # Sort by score (higher first)
        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Apply pagination
        paginated = filtered[offset:offset+limit]

        return paginated

    def update_candidate_status(
        self,
        candidate_id: str,
        status: str,
        concept_id: Optional[str] = None
    ) -> bool:
        """
        Update a candidate's status.

        Args:
            candidate_id: Candidate ID
            status: New status ('pending', 'approved', 'rejected')
            concept_id: Concept ID to assign if approved

        Returns:
            True if successful, False otherwise
        """
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            logger.warning(f"Candidate {candidate_id} not found")
            return False

        # Update status
        candidate['status'] = status

        # If approved, update concept_id
        if status == 'approved' and concept_id:
            candidate['concept_data']['concept_id'] = concept_id

        # Save changes
        return self._save_candidate(candidate)

    def add_candidate_to_repository(self, candidate_id: str) -> Optional[str]:
        """
        Add an approved candidate to the concept repository and create occurrences.

        Args:
            candidate_id: Candidate ID

        Returns:
            New concept ID if successful, None otherwise
        """
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            logger.warning(f"Candidate {candidate_id} not found")
            return None

        # Ensure candidate is approved
        if candidate.get('status') != 'approved':
            logger.warning(f"Cannot add candidate {candidate_id} to repository - not approved")
            return None

        # Get concept data
        concept_data = candidate.get('concept_data', {})

        # If concept_id is already set, use it
        concept_id = concept_data.get('concept_id')
        if not concept_id:
            concept_id = str(uuid.uuid4())
            concept_data['concept_id'] = concept_id

        # Ensure required fields exist
        if 'representations' not in concept_data:
            concept_data['representations'] = {}

        language = candidate.get('language', 'en')
        text = candidate.get('text', '').lower()

        if language not in concept_data['representations']:
            concept_data['representations'][language] = []

        if text not in concept_data['representations'][language]:
            concept_data['representations'][language].append(text)

        # Determine domain file category
        domain = candidate.get('domain', 'unknown')
        file_category = self._domain_to_file_category(domain)

        # Add to repository
        try:
            added_id = self.concept_repository.add_concept(
                concept_id=concept_id,
                representations=concept_data.get('representations'),
                prerequisites=concept_data.get('prerequisites', []),
                related=concept_data.get('related', []),
                file_category=file_category
            )

            if added_id:
                logger.info(f"Added candidate {candidate_id} to repository as concept {added_id}")

                # Update candidate with concept_id
                candidate['concept_data']['concept_id'] = added_id
                self._save_candidate(candidate)

                # Create occurrences in the database for this concept
                self._create_occurrences_for_concept(candidate, added_id)

                return added_id
            else:
                logger.warning(f"Failed to add candidate {candidate_id} to repository")
                return None
        except Exception as e:
            logger.error(f"Error adding candidate {candidate_id} to repository: {e}")
            return None

    def _create_occurrences_for_concept(self, candidate: Dict[str, Any], concept_id: str) -> bool:
        """
        Create database occurrences for an approved concept.

        Args:
            candidate: The candidate that was approved
            concept_id: The concept ID in the repository

        Returns:
            True if occurrences were created successfully, False otherwise
        """
        try:
            # Get source video and segments
            video_id = candidate.get('source_video_id')
            source_segments = candidate.get('source_segments', [])

            if not video_id or not source_segments:
                logger.warning(f"No source information for concept {concept_id}, skipping occurrence creation")
                return False

            # First, save concept to database if it's not already there
            repository_concept = self.concept_repository.get_concept(concept_id)
            if repository_concept and self.data_access:
                self.data_access.save_repository_concept(repository_concept)

            # Create occurrences
            occurrences = []

            # Get segments from database to get their details
            for segment_id in source_segments:
                # Skip if no segment ID
                if not segment_id:
                    continue

                # Get segment details
                if self.data_access:
                    segment_query = "SELECT * FROM segments WHERE segment_id = ?"
                    segment_results = self.data_access.execute_query(segment_query, (segment_id,))

                    if segment_results:
                        segment = segment_results[0]

                        # Create occurrence
                        occurrence_id = str(uuid.uuid4())

                        # Set educational significance based on candidate score
                        score = candidate.get('score', 0)
                        educational_significance = min(score, 4.0)  # Cap at 4.0

                        # Determine occurrence type based on significance
                        occurrence_type = "comprehensive" if educational_significance >= 2.5 else "passing"

                        # Create occurrence record
                        occurrence = {
                            "occurrence_id": occurrence_id,
                            "concept_id": concept_id,
                            "video_id": video_id,
                            "segment_id": segment_id,
                            "start_time": segment.get("start_time", 0.0),
                            "educational_significance": educational_significance,
                            "occurrence_type": occurrence_type,
                            "similarity": 1.0,  # Perfect match since it's from the candidate
                            "context_text": segment.get("text", "")
                        }

                        occurrences.append(occurrence)

            # Save occurrences to database
            if occurrences and self.data_access:
                success = self.data_access.save_occurrences(occurrences)
                if success:
                    logger.info(f"Created {len(occurrences)} occurrences for concept {concept_id}")
                else:
                    logger.warning(f"Failed to save occurrences for concept {concept_id}")

                return success

            return False

        except Exception as e:
            logger.error(f"Error creating occurrences for concept {concept_id}: {e}")
            return False

    def _domain_to_file_category(self, domain: str) -> str:
        """
        Convert domain to file category.

        Args:
            domain: Domain name

        Returns:
            File category name
        """
        domain_map = {
            'mathematics': 'mathematics',
            'physics': 'physics',
            'programming': 'computer_science',
            'computer science': 'computer_science',
            'unknown': 'interdisciplinary'
        }

        return domain_map.get(domain.lower(), 'interdisciplinary')

    def edit_candidate(
        self,
        candidate_id: str,
        new_text: Optional[str] = None,
        new_domain: Optional[str] = None,
        prerequisites: Optional[List[str]] = None,
        related: Optional[List[str]] = None
    ) -> bool:
        """
        Edit a candidate's properties.

        Args:
            candidate_id: Candidate ID
            new_text: Optional new text
            new_domain: Optional new domain
            prerequisites: Optional list of prerequisite concept IDs
            related: Optional list of related concept IDs

        Returns:
            True if successful, False otherwise
        """
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            logger.warning(f"Candidate {candidate_id} not found")
            return False

        # Update text if provided (ensure lowercase)
        if new_text:
            candidate['text'] = new_text.lower()

            # Update representation in concept data
            language = candidate.get('language', 'en')
            if 'representations' in candidate['concept_data']:
                if language in candidate['concept_data']['representations']:
                    candidate['concept_data']['representations'][language] = [new_text.lower()]
                else:
                    candidate['concept_data']['representations'][language] = [new_text.lower()]

        # Update domain if provided
        if new_domain:
            candidate['domain'] = new_domain
            candidate['concept_data']['metadata']['domain'] = new_domain

        # Update prerequisites if provided
        if prerequisites is not None:
            candidate['concept_data']['prerequisites'] = prerequisites

        # Update related concepts if provided
        if related is not None:
            candidate['concept_data']['related'] = related

        # Save changes
        return self._save_candidate(candidate)

    def delete_candidate(self, candidate_id: str) -> bool:
        """
        Delete a candidate.

        Args:
            candidate_id: Candidate ID

        Returns:
            True if successful, False otherwise
        """
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            logger.warning(f"Candidate {candidate_id} not found")
            return False

        try:
            # Remove from memory
            if candidate_id in self.candidates:
                del self.candidates[candidate_id]

            # Remove from disk
            file_path = os.path.join(self.candidates_dir, f"{candidate_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)

            logger.info(f"Deleted candidate: {candidate_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting candidate: {e}")
            return False

# Singleton instance
_candidate_extractor = None

def get_concept_candidate_extractor() -> ConceptCandidateExtractor:
    """
    Get the singleton instance of ConceptCandidateExtractor.

    Returns:
        ConceptCandidateExtractor instance
    """
    global _candidate_extractor

    if _candidate_extractor is None:
        _candidate_extractor = ConceptCandidateExtractor()

    return _candidate_extractor
