"""
Enhanced data pipeline for the Lecture Video Content Indexer.
Coordinates the end-to-end process of video extraction, transcript processing,
domain classification, and theory-practice analysis with improved concept extraction.
"""

import os
import logging
import uuid
import re
import nltk
from typing import Dict, List, Set, Any, Optional, Tuple, Counter as CounterType
from collections import Counter
from datetime import datetime
import json

# Make sure NLTK resources are available
required_resources = ['punkt', 'stopwords']
for resource in required_resources:
    try:
        nltk.data.find(f"{'corpora' if resource != 'punkt' else 'tokenizers'}/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder

# Import project modules
from youtube_extractor import YouTubeExtractor
from transcript_processor import TranscriptProcessor
from performance_utils import time_function, Timer
from cache_manager import cache_get, cache_set

# Configure logging
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Coordinates the end-to-end process of video data acquisition and analysis.
    Enhanced with improved concept extraction and domain analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get("output_dir", "data/processed")

        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self._init_components()

        # Initialize NLP resources
        self._init_nlp_resources()

        logger.info("DataPipeline initialized with enhanced concept extraction")

    def _init_components(self):
        """Initialize pipeline components."""
        # Get YouTube API key from config
        youtube_api_key = self.config.get("youtube_api_key")
        if not youtube_api_key:
            logger.warning("No YouTube API key provided, using test mode")
            youtube_api_key = "test_api_key"

        # Initialize components
        self.youtube_extractor = YouTubeExtractor(youtube_api_key)
        self.transcript_processor = TranscriptProcessor()

        logger.info("Pipeline components initialized")

    def _init_nlp_resources(self):
        """Initialize NLP resources for concept extraction."""
        # Initialize stopwords for different languages
        self.stopwords_en = set(stopwords.words('english'))

        # Add common filler words not in NLTK's stopwords
        additional_stopwords = {"uh", "um", "like", "so", "well", "actually", "basically",
                                "literally", "sort", "kind", "really", "very", "quite",
                                "okay", "ok", "yeah", "yes", "no", "right", "let", "just",
                                "gonna", "going", "let's", "now", "here", "there", "this",
                                "that", "these", "those", "will", "shall", "should", "would",
                                "could", "can", "may", "might", "must"}
        self.stopwords_en.update(additional_stopwords)

        # Domain-specific keywords that are important for each domain
        # These should NOT be filtered out as stopwords
        self.domain_keywords = {
            "mathematics": {"function", "variable", "equation", "theorem", "proof",
                            "integral", "derivative", "limit", "series", "vector",
                            "matrix", "algebra", "geometry", "calculus", "topology",
                            "probability", "statistics", "set", "group", "field",
                            "differential", "discrete", "continuous", "infinite", "finite"},

            "programming": {"algorithm", "function", "class", "object", "method",
                            "variable", "array", "list", "loop", "recursion",
                            "data", "structure", "stack", "queue", "tree", "graph",
                            "hash", "sort", "search", "complexity", "database",
                            "interface", "inheritance", "polymorphism", "encapsulation"},

            "physics": {"force", "energy", "momentum", "mass", "velocity", "acceleration",
                        "gravity", "electromagnetism", "quantum", "relativity", "particle",
                        "wave", "field", "potential", "nuclear", "atomic", "thermodynamics",
                        "fluid", "mechanics", "dynamics", "kinematics", "statics"}
        }

        # Domain-specific n-gram patterns for concept extraction
        self.domain_concept_patterns = {
            "mathematics": [
                # Patterns for mathematical concepts
                r'\b(?:the|a) (\w+) (theorem|lemma|property|identity|formula|equation|inequality|principle)\b',
                r'\b(differential|partial|ordinary) (equation)\b',
                r'\b(linear|quadratic|polynomial|exponential|logarithmic|trigonometric) (function|equation|identity)\b',
                r'\b(convergent|divergent|infinite|finite) (series|sequence)\b',
                r'\b(vector|matrix|tensor) (space|field|algebra|calculus)\b',
                r'\b(probability|statistical) (distribution|model|test|analysis)\b'
            ],

            "programming": [
                # Patterns for programming concepts
                r'\b(data|abstract) (structure|type)\b',
                r'\b(sorting|search|graph|tree) (algorithm)\b',
                r'\b(time|space) (complexity)\b',
                r'\b(object[\-\s]oriented|functional|procedural|declarative) (programming|approach|paradigm)\b',
                r'\b(design|architectural) (pattern)\b',
                r'\b(binary|linear|hash) (search|table)\b',
                r'\b(linked|array|circular) (list)\b'
            ],

            "physics": [
                # Patterns for physics concepts
                r'\b(gravitational|electric|magnetic|electromagnetic) (field|force|potential)\b',
                r'\b(kinetic|potential|mechanical|thermal|nuclear) (energy)\b',
                r'\b(newton\'?s|coulomb\'?s|faraday\'?s|ohm\'?s|ampere\'?s|kepler\'?s) (law|principle)\b',
                r'\b(special|general) (relativity)\b',
                r'\b(quantum) (mechanics|field theory|chromodynamics|electrodynamics)\b',
                r'\b(wave|particle) (function|duality|theory)\b',
                r'\b(string|m|supersymmetric) (theory)\b'
            ]
        }

        # Compile the patterns for efficiency
        self.compiled_domain_patterns = {}
        for domain, patterns in self.domain_concept_patterns.items():
            self.compiled_domain_patterns[domain] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

        logger.info("NLP resources initialized for concept extraction")

    @time_function(10000)  # Log warning if takes more than 10 seconds
    def process_video(self, video_url: str, language_preference: List[str] = ['en', 'ru']) -> Dict[str, Any]:
        """
        Process a YouTube video through the entire pipeline.

        Args:
            video_url: YouTube video URL
            language_preference: List of language codes in order of preference

        Returns:
            Dictionary with processing results
        """
        # Create a timer for overall process
        timer = Timer("process_video").start()

        # Generate a unique job ID based on timestamp and UUID
        job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Starting video processing job {job_id} for URL: {video_url}")

        try:
            # Step 1: Validate URL and extract video ID
            valid, video_id = self.youtube_extractor.validate_video_url(video_url)
            if not valid or not video_id:
                error_msg = f"Invalid YouTube URL: {video_url}"
                logger.error(error_msg)
                return {
                    "job_id": job_id,
                    "status": "error",
                    "error": error_msg,
                    "video_url": video_url
                }

            logger.info(f"Validated YouTube URL, video ID: {video_id}")

            # Check cache for previously processed result
            cache_key = f"processed_video_{video_id}"
            cached_result = cache_get("video", cache_key)
            if cached_result:
                logger.info(f"Using cached processing result for video {video_id}")
                return cached_result

            # Step 2: Extract video metadata
            metadata = self.youtube_extractor.extract_video_metadata(video_id)
            logger.info(f"Extracted metadata for video: {video_id}")

            # Step 3: Extract transcript
            raw_transcript = self.youtube_extractor.extract_transcript(video_id, language_preference)
            logger.info(f"Extracted transcript with {len(raw_transcript)} segments")

            # Step 4: Process transcript
            processed_transcript = self.transcript_processor.process_transcript(raw_transcript, metadata)
            logger.info(f"Processed transcript with {len(processed_transcript['segments'])} segments")

            # Step 5: Calculate theory/practice ratio
            theory_practice_results = self._calculate_theory_practice_ratio(processed_transcript['segments'])
            logger.info(f"Calculated theory/practice ratio: {theory_practice_results['theory_practice_ratio']:.2f}")

            # Step 6: Extract key concepts
            domain_features = self._extract_domain_features(processed_transcript, metadata["domain"])
            logger.info(f"Extracted {len(domain_features['key_concepts'])} key concepts")

            # Prepare result
            processing_time = timer.stop() / 1000  # Convert from ms to seconds

            result = {
                "job_id": job_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "video_id": video_id,
                "video_url": video_url,
                "metadata": metadata,
                "transcript": processed_transcript,
                "domain_features": domain_features,
                "theory_practice_results": theory_practice_results,
                "processing_time": processing_time
            }

            # Cache the result
            cache_set("video", cache_key, result)

            # Save result to file (for backward compatibility)
            self._save_result(result)

            logger.info(f"Successfully processed video {video_id} in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")
            error_result = {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "video_url": video_url,
                "timestamp": datetime.now().isoformat()
            }

            # If we have a video_id, include it
            if 'video_id' in locals() and video_id:
                error_result["video_id"] = video_id

            # If we have metadata, include it
            if 'metadata' in locals() and metadata:
                error_result["metadata"] = metadata

            # Save error result
            self._save_result(error_result)

            return error_result

    def _calculate_theory_practice_ratio(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Calculate theory/practice ratio from segments with enhanced accuracy.

        Args:
            segments: Processed transcript segments

        Returns:
            Dictionary with theory/practice analysis
        """
        if not segments:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "theoretical_segments": 0,
                "practical_segments": 0,
                "mixed_segments": 0,
                "theory_practice_ratio": 0.5
            }

        # Count segment types with confidence weighting
        theoretical_count = 0
        practical_count = 0
        mixed_count = 0

        # Track total confidence-weighted counts
        theoretical_weighted = 0
        practical_weighted = 0
        mixed_weighted = 0

        # Track time distribution
        total_duration = 0
        theoretical_duration = 0
        practical_duration = 0
        mixed_duration = 0

        for segment in segments:
            segment_type = segment.get("content_type", "mixed")
            confidence = segment.get("classification_confidence", 0.6)  # Default confidence if not present

            # Calculate segment duration
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time
            total_duration += duration

            if segment_type == "theoretical":
                theoretical_count += 1
                theoretical_weighted += confidence
                theoretical_duration += duration
            elif segment_type == "practical":
                practical_count += 1
                practical_weighted += confidence
                practical_duration += duration
            else:  # mixed
                mixed_count += 1
                mixed_weighted += confidence
                mixed_duration += duration

        total_segments = theoretical_count + practical_count + mixed_count

        # Calculate theory/practice ratio with improved weighting
        if total_segments > 0:
            # Apply a weighted formula with confidence
            total_weighted = theoretical_weighted + practical_weighted + mixed_weighted

            if total_weighted > 0:
                # Apply confidence-weighted formula
                theory_weight = theoretical_weighted + (mixed_weighted * 0.5)
                theory_practice_ratio = theory_weight / total_weighted
            else:
                theory_practice_ratio = 0.5

            # Factor in duration-based ratio
            if total_duration > 0:
                duration_theory_ratio = (theoretical_duration + (mixed_duration * 0.5)) / total_duration

                # Final ratio is an average of count-based and duration-based ratios
                theory_practice_ratio = (theory_practice_ratio + duration_theory_ratio) / 2

        else:
            theory_practice_ratio = 0.5

        # Determine overall classification with improved confidence calculation
        if theory_practice_ratio > 0.7:
            classification = "theoretical"
            # Higher confidence if there is a strong bias towards theoretical
            if theory_practice_ratio > 0.85:
                confidence = 0.9
            elif theoretical_count > practical_count * 2:
                confidence = 0.8
            else:
                confidence = 0.7
        elif theory_practice_ratio < 0.3:
            classification = "practical"
            # Higher confidence if there is a strong bias towards practical
            if theory_practice_ratio < 0.15:
                confidence = 0.9
            elif practical_count > theoretical_count * 2:
                confidence = 0.8
            else:
                confidence = 0.7
        else:
            classification = "mixed"
            # Higher confidence when theory/practice ratio is near 0.5
            closeness_to_half = 1.0 - abs(theory_practice_ratio - 0.5) * 2  # 1.0 at 0.5, 0.0 at 0.0/1.0
            confidence = 0.6 + (closeness_to_half * 0.3)  # Maps to 0.6-0.9 range

        return {
            "classification": classification,
            "confidence": confidence,
            "theoretical_segments": theoretical_count,
            "practical_segments": practical_count,
            "mixed_segments": mixed_count,
            "theory_practice_ratio": theory_practice_ratio,
            "duration_analysis": {
                "total_duration": total_duration,
                "theoretical_duration": theoretical_duration,
                "practical_duration": practical_duration,
                "mixed_duration": mixed_duration
            }
        }

    def _extract_domain_features(self, processed_transcript: Dict, domain: str) -> Dict[str, Any]:
        """
        Extract domain-specific features from processed transcript with enhanced concept extraction.

        Args:
            processed_transcript: Processed transcript dictionary
            domain: Content domain

        Returns:
            Dictionary with domain-specific features
        """
        segments = processed_transcript.get("segments", [])
        language = processed_transcript.get("language", "en")

        # Extract combined text for analysis
        combined_text = " ".join([segment.get("text", "") for segment in segments])

        # Perform concept extraction
        key_concepts = self._extract_key_concepts(combined_text, segments, domain, language)

        # Organize concepts by segment context types
        theoretical_concepts = []
        practical_concepts = []

        for concept in key_concepts:
            # Add to appropriate list based on concept_class
            if concept["concept_class"] == "theoretical":
                theoretical_concepts.append(concept)
            else:
                practical_concepts.append(concept)

        # Find relationships between concepts
        concept_relationships = self._find_concept_relationships(key_concepts, segments)

        return {
            "domain": domain,
            "key_concepts": key_concepts,
            "theoretical_concepts": theoretical_concepts,
            "practical_concepts": practical_concepts,
            "concept_relationships": concept_relationships
        }

    def _extract_key_concepts(
        self,
        combined_text: str,
        segments: List[Dict],
        domain: str,
        language: str = "en"
    ) -> List[Dict]:
        """
        Enhanced concept extraction using NLP techniques.

        Args:
            combined_text: Combined text from all segments
            segments: List of transcript segments
            domain: Content domain
            language: Language code

        Returns:
            List of concept dictionaries
        """
        # Skip if text is empty
        if not combined_text.strip():
            return []

        # Get stopwords for the language
        stopwords_set = self.stopwords_en if language == "en" else set()

        # Remove domain keywords from stopwords to keep important terms
        domain_keywords = self.domain_keywords.get(domain, set())
        filtered_stopwords = stopwords_set - domain_keywords

        # Preprocess text
        sentences = sent_tokenize(combined_text)

        # 1. Extract n-grams (unigrams, bigrams, trigrams)
        all_words = []
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            # Filter stopwords but keep domain keywords
            filtered_tokens = [token for token in tokens
                              if token.isalnum() and (token not in filtered_stopwords or token in domain_keywords)]
            all_words.extend(filtered_tokens)

        # Handle empty token list
        if not all_words:
            return []

        # Extract and count n-grams
        unigrams = Counter(all_words)

        # Extract bigrams if enough tokens
        bigram_finder = BigramCollocationFinder.from_words(all_words)
        # Apply frequency filter to avoid rare combinations
        bigram_finder.apply_freq_filter(2)
        # Score bigrams by PMI (pointwise mutual information)
        bigram_measures = BigramAssocMeasures()
        bigram_scores = bigram_finder.score_ngrams(bigram_measures.pmi)
        # Convert to counter
        bigrams = Counter({' '.join(bigram): score for bigram, score in bigram_scores})

        # Extract trigrams if enough tokens
        if len(all_words) > 5:
            trigram_finder = TrigramCollocationFinder.from_words(all_words)
            trigram_finder.apply_freq_filter(2)
            trigram_measures = TrigramAssocMeasures()
            trigram_scores = trigram_finder.score_ngrams(trigram_measures.pmi)
            trigrams = Counter({' '.join(trigram): score for trigram, score in trigram_scores})
        else:
            trigrams = Counter()

        # 2. Extract domain-specific patterns
        pattern_matches = {}
        if domain in self.compiled_domain_patterns:
            for pattern in self.compiled_domain_patterns[domain]:
                for sentence in sentences:
                    matches = pattern.findall(sentence)
                    for match in matches:
                        if isinstance(match, tuple):
                            # Join multi-word matches
                            concept = ' '.join(match)
                        else:
                            concept = match
                        pattern_matches[concept.lower()] = pattern_matches.get(concept.lower(), 0) + 1

        # 3. Combine all concept candidates
        all_concepts = {}

        # Add unigrams (single words)
        for word, count in unigrams.most_common(50):
            if len(word) > 2:  # Skip very short words
                all_concepts[word] = {
                    "text": word,
                    "frequency": count,
                    "ngram_type": "unigram",
                    "score": count * 0.5  # Lower weight for unigrams
                }

        # Add bigrams
        for bigram, score in bigrams.most_common(30):
            all_concepts[bigram] = {
                "text": bigram,
                "frequency": bigram_finder.ngram_fd[tuple(bigram.split())],
                "ngram_type": "bigram",
                "score": score * 1.0  # Standard weight for bigrams
            }

        # Add trigrams
        for trigram, score in trigrams.most_common(20):
            all_concepts[trigram] = {
                "text": trigram,
                "frequency": trigram_finder.ngram_fd[tuple(trigram.split())],
                "ngram_type": "trigram",
                "score": score * 1.5  # Higher weight for trigrams
            }

        # Add pattern matches with the highest weight
        for pattern, count in pattern_matches.items():
            if pattern in all_concepts:
                all_concepts[pattern]["score"] += count * 2.0  # Boost score for pattern matches
                all_concepts[pattern]["pattern_match"] = True
            else:
                all_concepts[pattern] = {
                    "text": pattern,
                    "frequency": count,
                    "ngram_type": "pattern",
                    "pattern_match": True,
                    "score": count * 2.0  # Highest weight for domain patterns
                }

        # 4. Filter and rank concepts
        ranked_concepts = []
        for concept_text, concept_data in all_concepts.items():
            # Skip concepts with very low scores
            if concept_data["score"] < 1.0:
                continue

            # Determine if concept is theoretical or practical based on context
            is_theoretical = self._is_theoretical_concept(concept_text, segments)

            # Create final concept entry
            concept = {
                "text": concept_text,
                "frequency": concept_data["frequency"],
                "domain": domain,
                "theoretical": is_theoretical,
                "concept_class": "theoretical" if is_theoretical else "practical",
                "ngram_type": concept_data.get("ngram_type", "pattern"),
                "pattern_match": concept_data.get("pattern_match", False),
                "score": concept_data["score"]
            }

            ranked_concepts.append(concept)

        # Sort by score
        ranked_concepts.sort(key=lambda x: x["score"], reverse=True)

        # Take top concepts
        return ranked_concepts[:min(50, len(ranked_concepts))]

    def _is_theoretical_concept(self, term: str, segments: List[Dict]) -> bool:
        """
        Determine if a concept is theoretical based on its context with enhanced accuracy.

        Args:
            term: Concept term
            segments: Processed transcript segments

        Returns:
            True if theoretical, False if practical
        """
        # Normalize term for matching
        term_lower = term.lower()
        term_parts = term_lower.split()

        # Get segments containing this term
        theoretical_count = 0
        practical_count = 0
        theoretical_confidence_sum = 0
        practical_confidence_sum = 0

        for segment in segments:
            segment_text = segment.get("text", "").lower()
            content_type = segment.get("content_type", "mixed")
            confidence = segment.get("classification_confidence", 0.6)

            # Check if all parts of the term are in this segment
            if all(part in segment_text for part in term_parts):
                if content_type == "theoretical":
                    theoretical_count += 1
                    theoretical_confidence_sum += confidence
                elif content_type == "practical":
                    practical_count += 1
                    practical_confidence_sum += confidence

        # If no segments contain the term, check for domain-specific indicators
        if theoretical_count == 0 and practical_count == 0:
            # Default to theoretical for long compound terms (often concepts or theories)
            if len(term_parts) >= 3:
                return True

            # Check for indicator terms within the concept itself
            theoretical_indicators = {"theory", "theorem", "law", "principle", "definition", "concept"}
            practical_indicators = {"example", "application", "practice", "implementation", "method", "technique"}

            theoretical_matches = sum(1 for ind in theoretical_indicators if ind in term_parts)
            practical_matches = sum(1 for ind in practical_indicators if ind in term_parts)

            if theoretical_matches > practical_matches:
                return True
            elif practical_matches > theoretical_matches:
                return False
            else:
                # Default to theoretical for unknown cases
                return True

        # Determine by majority with confidence weighting
        if theoretical_confidence_sum > practical_confidence_sum:
            return True
        elif practical_confidence_sum > theoretical_confidence_sum:
            return False
        else:
            # Use count as tiebreaker
            return theoretical_count >= practical_count

    def _find_concept_relationships(self, concepts: List[Dict], segments: List[Dict]) -> List[Dict]:
        """
        Find relationships between concepts based on co-occurrence.

        Args:
            concepts: List of extracted concepts
            segments: List of transcript segments

        Returns:
            List of concept relationship dictionaries
        """
        # Skip if not enough concepts
        if len(concepts) < 2:
            return []

        # Create a map of concepts to their texts for easier lookup
        concept_texts = {concept["text"].lower(): concept for concept in concepts}

        # Track co-occurrences
        co_occurrences = {}

        # Analyze each segment for co-occurring concepts
        for segment in segments:
            segment_text = segment.get("text", "").lower()

            # Find all concepts in this segment
            concepts_in_segment = []
            for concept_text in concept_texts:
                concept_parts = concept_text.split()
                if all(part in segment_text for part in concept_parts):
                    concepts_in_segment.append(concept_text)

            # Record co-occurrences for each pair
            for i, concept1 in enumerate(concepts_in_segment):
                for concept2 in concepts_in_segment[i+1:]:
                    pair = tuple(sorted([concept1, concept2]))
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

        # Create relationship records
        relationships = []
        for (concept1, concept2), count in co_occurrences.items():
            # Only include significant co-occurrences
            if count >= 2:
                c1 = concept_texts[concept1]
                c2 = concept_texts[concept2]

                # Determine relationship type
                if c1["concept_class"] == c2["concept_class"]:
                    rel_type = "related_" + c1["concept_class"]
                else:
                    rel_type = "theory_practice_pair"

                relationship = {
                    "source_concept": concept1,
                    "target_concept": concept2,
                    "co_occurrence_count": count,
                    "relationship_type": rel_type,
                    "source_class": c1["concept_class"],
                    "target_class": c2["concept_class"]
                }
                relationships.append(relationship)

        # Sort by co-occurrence count
        relationships.sort(key=lambda x: x["co_occurrence_count"], reverse=True)

        return relationships

    def _save_result(self, result: Dict[str, Any]):
        """
        Save processing result to file.

        Args:
            result: Processing result dictionary
        """
        # Create filename using video_id or job_id
        video_id = result.get("video_id", "unknown")
        job_id = result.get("job_id", "unknown")
        filename = f"{video_id}_{job_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved processing result to {filepath}")
        except Exception as e:
            logger.error(f"Error saving result to file: {e}")

    def batch_process_videos(self, video_urls: List[str], language_preference: List[str] = ['en', 'ru']) -> List[Dict[str, Any]]:
        """
        Process multiple YouTube videos.

        Args:
            video_urls: List of YouTube video URLs
            language_preference: List of language codes in order of preference

        Returns:
            List of processing result dictionaries
        """
        logger.info(f"Starting batch processing for {len(video_urls)} videos")

        results = []
        for i, url in enumerate(video_urls):
            try:
                # Process video
                logger.info(f"Processing video {i+1}/{len(video_urls)}: {url}")
                result = self.process_video(url, language_preference)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing video {url}: {e}")
                results.append({
                    "video_url": url,
                    "status": "error",
                    "error": str(e)
                })

        logger.info(f"Batch processing completed for {len(video_urls)} videos")
        return results
