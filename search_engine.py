"""
Enhanced search engine for the Lecture Video Content Indexer.
Provides robust search functionality using SQLite FTS5 with improved query handling,
relevance ranking, and learning path generation.
"""

import os
import logging
import json
import time
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

# Import project modules
from data_access import get_data_access
from cache_manager import cache_get, cache_set, cached
from performance_utils import time_function

# Configure logging
logger = logging.getLogger(__name__)

try:
    from concept_signature_generator import enhance_search_engine
except ImportError:
    # Handle import error gracefully
    logger.warning("ConceptSignatureGenerator not available - enhanced learning paths will not be used")
    enhance_search_engine = lambda x: x

class SearchEngine:
    """
    Enhanced search engine for educational video content.
    Provides sophisticated search and content discovery functionality.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the search engine with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.index_dir = config.get("index_dir", "data/index")

        # Create index directory if needed
        os.makedirs(self.index_dir, exist_ok=True)

        # Get data access layer
        db_path = os.path.join(self.index_dir, "indexer.db")
        self.data_access = get_data_access(db_path)

        # Initialize ranking weights for search results
        self._init_ranking_weights()

        enhance_search_engine(self)

        logger.info("SearchEngine initialized with enhanced search capabilities")

    def _init_ranking_weights(self):
        """Initialize weights for relevance ranking algorithm."""
        # Default weights for ranking factors
        self.ranking_weights = {
            # Match quality weights
            "exact_match": 2.0,      # Exact match of search terms
            "partial_match": 1.0,    # Partial match of search terms
            "title_match": 1.5,      # Match in video title
            "context_match": 1.2,    # Match in context of segment

            # Content type weights
            "theoretical": 1.0,      # Base weight for theoretical content
            "practical": 1.0,        # Base weight for practical content

            # Item type weights
            "concept": 1.3,          # Concepts are weighted higher than segments
            "segment": 1.0,          # Base weight for segments

            # Segment classification confidence
            "high_confidence": 1.2,  # High classification confidence
            "medium_confidence": 1.0, # Medium classification confidence
            "low_confidence": 0.8,   # Low classification confidence
        }

    @time_function(2000)  # Log warning if takes more than 2 seconds
    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced search for content matching the query with improved ranking and language support.

        Args:
            query: Structured query dictionary

        Returns:
            Search results dictionary with ranked results
        """
        start_time = time.time()

        try:
            # Extract query parameters
            query_text = query.get("original_text", "").strip()
            filters = query.get("filters", {})
            theory_practice_ratio = query.get("theory_practice_ratio")
            domain = query.get("domain")
            language = query.get("language")  # Extract language parameter
            pagination = query.get("pagination", {})

            # Apply pagination parameters
            offset = pagination.get("offset", 0)
            limit = pagination.get("limit", 10)

            # Check for empty query
            if not query_text:
                return {
                    "results": [],
                    "totalResults": 0,
                    "executionTimeMs": 0,
                    "status": "error",
                    "message": "Empty search query"
                }

            # Generate cache key for this query
            cache_key = self._generate_cache_key(query_text, filters, theory_practice_ratio, domain, offset, limit, language)
            cached_results = cache_get("search", cache_key)

            if cached_results:
                logger.info(f"Using cached search results for query: {query_text}")
                return cached_results

            # Add language to the query for data access layer
            query["language"] = language

            # Execute base search through data access layer
            base_results = self.data_access.search(query)

            if not base_results.get("results"):
                # Try synonym expansion if no results
                expanded_query = self._expand_query_with_synonyms(query_text, domain, language)
                if expanded_query and expanded_query != query_text:
                    logger.info(f"No results for '{query_text}', trying with synonyms: '{expanded_query}'")
                    query["original_text"] = expanded_query
                    base_results = self.data_access.search(query)

            # Enhanced ranking and processing of results
            enhanced_results = self._enhance_search_results(
                base_results,
                query_text,
                theory_practice_ratio,
                domain,
                language
            )

            # Cache the enhanced results
            execution_time_ms = int((time.time() - start_time) * 1000)
            enhanced_results["executionTimeMs"] = execution_time_ms

            cache_set("search", cache_key, enhanced_results)

            return enhanced_results

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "totalResults": 0,
                "theoreticalResults": 0,
                "practicalResults": 0,
                "executionTimeMs": execution_time_ms,
                "status": "error",
                "message": str(e)
            }

    def _generate_cache_key(
        self,
        query_text: str,
        filters: Dict,
        theory_practice_ratio: Optional[float],
        domain: Optional[str],
        offset: int,
        limit: int,
        language: Optional[str] = None
    ) -> str:
        """Generate a consistent cache key for search queries."""
        key_parts = [f"q:{query_text}"]

        # Add filters
        if "video_id" in filters:
            key_parts.append(f"vid:{filters['video_id']}")

        if "video_ids" in filters and filters["video_ids"]:
            video_ids_str = ",".join(sorted(filters["video_ids"]))
            key_parts.append(f"vids:{video_ids_str}")

        # Add theory/practice ratio
        if theory_practice_ratio is not None:
            key_parts.append(f"tpr:{theory_practice_ratio:.2f}")

        # Add domain
        if domain:
            key_parts.append(f"dom:{domain}")

        # Add language
        if language:
            key_parts.append(f"lang:{language}")

        # Add pagination
        key_parts.append(f"off:{offset}")
        key_parts.append(f"lim:{limit}")

        # Combine parts and hash to keep key length reasonable
        combined = "_".join(key_parts)
        if len(combined) > 100:
            # Use a hash for very long keys
            return f"search_{hash(combined)}"

        return f"search_{combined}"

    def _expand_query_with_synonyms(self, query_text: str, domain: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Expand query with domain-specific synonyms to improve recall.

        Args:
            query_text: Original query text
            domain: Optional domain to use domain-specific synonyms
            language: Optional language for multilingual synonym expansion

        Returns:
            Expanded query text
        """
        # Make sure query_text is not None
        if not query_text:
            return ""

        lang = language if language in ['en', 'ru'] else 'en'

        # Domain-specific synonyms for common terms by language
        domain_synonyms = {
            "mathematics": {
                "en": {
                    # Common mathematical term synonyms
                    "derivative": ["differentiation", "differential"],
                    "integral": ["integration", "antiderivative"],
                    "equation": ["formula", "relation"],
                    "function": ["mapping", "transformation"],
                    "variable": ["parameter", "unknown"],
                    "theorem": ["proposition", "lemma", "corollary"],
                    "vector": ["direction", "array"],
                    "matrix": ["array", "grid"],
                    "set": ["collection", "group"],
                    "limit": ["convergence", "boundary"]
                },
                "ru": {
                    # Russian mathematical synonyms
                    "производная": ["дифференцирование", "дифференциал"],
                    "интеграл": ["интегрирование", "первообразная"],
                    "уравнение": ["формула", "соотношение"],
                    "функция": ["отображение", "преобразование"],
                    "переменная": ["параметр", "неизвестная"],
                    "теорема": ["утверждение", "лемма", "следствие"],
                    "вектор": ["направление", "массив"],
                    "матрица": ["массив", "таблица"],
                    "множество": ["набор", "группа"],
                    "предел": ["сходимость", "граница"]
                }
            },
            "programming": {
                "en": {
                    # Common programming term synonyms
                    "function": ["method", "procedure", "routine"],
                    "class": ["object", "type", "struct"],
                    "algorithm": ["procedure", "routine", "process"],
                    "variable": ["field", "property", "attribute"],
                    "loop": ["iteration", "repetition", "cycle"],
                    "array": ["list", "collection", "sequence"],
                    "database": ["data store", "repository"],
                    "inheritance": ["subclassing", "extension"],
                    "interface": ["contract", "protocol"],
                    "recursion": ["self-reference", "recurrence"]
                },
                "ru": {
                    # Russian programming synonyms
                    "функция": ["метод", "процедура", "подпрограмма"],
                    "класс": ["объект", "тип", "структура"],
                    "алгоритм": ["процедура", "процесс", "последовательность"],
                    "переменная": ["поле", "свойство", "атрибут"],
                    "цикл": ["итерация", "повторение"],
                    "массив": ["список", "коллекция", "последовательность"],
                    "база данных": ["хранилище данных", "репозиторий"],
                    "наследование": ["расширение", "подклассирование"],
                    "интерфейс": ["контракт", "протокол"],
                    "рекурсия": ["самовызов", "рекуррентность"]
                }
            },
            "physics": {
                "en": {
                    # Common physics term synonyms
                    "force": ["interaction", "push", "pull"],
                    "energy": ["work", "power"],
                    "velocity": ["speed", "rate"],
                    "acceleration": ["rate of change of velocity"],
                    "momentum": ["inertia", "impulse"],
                    "field": ["domain", "space"],
                    "wave": ["oscillation", "vibration"],
                    "particle": ["body", "corpuscle"],
                    "charge": ["electric charge", "electrostatic charge"],
                    "mass": ["inertia", "matter"]
                },
                "ru": {
                    # Russian physics synonyms
                    "сила": ["взаимодействие", "воздействие"],
                    "энергия": ["работа", "мощность"],
                    "скорость": ["быстрота", "темп"],
                    "ускорение": ["изменение скорости"],
                    "импульс": ["количество движения", "момент"],
                    "поле": ["область", "пространство"],
                    "волна": ["колебание", "вибрация"],
                    "частица": ["тело", "корпускула"],
                    "заряд": ["электрический заряд", "электростатический заряд"],
                    "масса": ["инерция", "вещество", "материя"]
                }
            }
        }

        # General academic synonyms by language
        general_synonyms = {
            "en": {
                "concept": ["idea", "notion", "principle"],
                "theory": ["principle", "hypothesis", "postulate"],
                "example": ["instance", "illustration", "demonstration"],
                "application": ["use case", "implementation", "usage"],
                "problem": ["challenge", "question", "issue"],
                "solution": ["answer", "resolution", "approach"],
                "method": ["technique", "approach", "procedure"]
            },
            "ru": {
                "концепция": ["идея", "понятие", "принцип"],
                "теория": ["принцип", "гипотеза", "постулат"],
                "пример": ["случай", "иллюстрация", "демонстрация"],
                "применение": ["использование", "реализация", "использование"],
                "проблема": ["задача", "вопрос", "вызов"],
                "решение": ["ответ", "подход", "метод"],
                "метод": ["техника", "подход", "процедура", "способ"]
            }
        }

        # Use domain-specific synonyms if domain is provided
        synonyms = general_synonyms.get(lang, general_synonyms['en']).copy()
        if domain and domain in domain_synonyms:
            domain_lang_synonyms = domain_synonyms[domain].get(lang, domain_synonyms[domain].get('en', {}))
            synonyms.update(domain_lang_synonyms)

        # Convert query to lowercase for consistent matching
        query_text_lower = query_text.lower()

        # Split query into tokens
        tokens = re.findall(r'\b\w+\b', query_text_lower)

        # Check if any tokens have synonyms
        has_synonyms = False
        expanded_parts = []

        for token in tokens:
            if token in synonyms:
                synonym_list = [token] + synonyms[token]
                synonym_part = f"({' OR '.join(synonym_list)})"
                expanded_parts.append(synonym_part)
                has_synonyms = True
            else:
                expanded_parts.append(token)

        # If no synonyms found, return original query
        if not has_synonyms:
            return query_text

        # Join expanded parts
        expanded_query = " ".join(expanded_parts)
        return expanded_query

    def _enhance_search_results(
        self,
        base_results: Dict[str, Any],
        query_text: str,
        theory_practice_ratio: Optional[float],
        domain: Optional[str],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance search results with improved ranking and organization.

        Args:
            base_results: Base search results from data access layer
            query_text: Original query text
            theory_practice_ratio: Theory/practice preference ratio
            domain: Optional domain filter
            language: Optional language filter

        Returns:
            Enhanced search results
        """
        results = base_results.get("results", [])

        # Apply advanced ranking to results
        ranked_results = self._rank_results(results, query_text, theory_practice_ratio, language)

        # Calculate result type statistics
        total_results = base_results.get("totalResults", 0)
        theoretical_results = sum(1 for r in ranked_results if r.get("context_type") == "theoretical")
        practical_results = sum(1 for r in ranked_results if r.get("context_type") == "practical")
        mixed_results = sum(1 for r in ranked_results if r.get("context_type") == "mixed")

        # Group results by video
        results_by_video = defaultdict(list)
        for result in ranked_results:
            video_id = result.get("video_id")
            if video_id:
                results_by_video[video_id].append(result)

        # Extract concepts and calculate domain distribution
        concepts = {}
        domain_counts = Counter()
        language_counts = Counter()  # Track languages

        for result in ranked_results:
            if result.get("result_type") == "concept":
                concept_id = result.get("concept_id")
                concepts[concept_id] = result

            result_domain = result.get("domain")
            if result_domain:
                domain_counts[result_domain] += 1

            # Track language distribution
            result_language = result.get("language")
            if result_language:
                language_counts[result_language] += 1

        enhanced_results = {
            "results": ranked_results,
            "totalResults": total_results,
            "theoreticalResults": theoretical_results,
            "practicalResults": practical_results,
            "mixedResults": mixed_results,
            "videosCount": len(results_by_video),
            "conceptsCount": len(concepts),
            "domainDistribution": [
                {"domain": domain, "count": count}
                for domain, count in domain_counts.most_common()
            ],
            "languageDistribution": [
                {"language": lang, "count": count}
                for lang, count in language_counts.most_common()
            ]
        }

        # Generate query-dependent suggestions
        suggestions = self._generate_search_suggestions(query_text, ranked_results, domain, language)
        if suggestions:
            enhanced_results["suggestions"] = suggestions

        return enhanced_results

    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        query_text: str,
        theory_practice_ratio: Optional[float],
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply advanced ranking algorithm to search results with language support.

        Args:
            results: List of search results
            query_text: Original query text
            theory_practice_ratio: Theory/practice preference ratio
            language: Optional language code for language-specific ranking

        Returns:
            Ranked list of search results
        """
        if not results:
            return []

        # Ensure query_text is not None
        if query_text is None:
            query_text = ""

        # Prepare query terms for matching
        query_terms = set(re.findall(r'\b\w+\b', query_text.lower()))

        # Score and sort results
        scored_results = []

        for result in results:
            # Initialize base score
            score = 1.0

            # Factor 1: Match quality - ensure text is not None
            result_text = (result.get("text") or "").lower()
            exact_matches = sum(1 for term in query_terms if f" {term} " in f" {result_text} ")
            partial_matches = sum(1 for term in query_terms if term in result_text) - exact_matches

            score += exact_matches * self.ranking_weights["exact_match"]
            score += partial_matches * self.ranking_weights["partial_match"]

            # Check for matches in video title - ensure video_title is not None
            video_title = (result.get("video_title") or "").lower()
            if any(term in video_title for term in query_terms):
                score += self.ranking_weights["title_match"]

            # Factor 2: Content type weights based on theory/practice ratio
            content_type = result.get("context_type", "mixed")

            if theory_practice_ratio is not None:
                if content_type == "theoretical":
                    type_weight = theory_practice_ratio * 2  # Scale to 0-2
                    score += type_weight * self.ranking_weights["theoretical"]
                elif content_type == "practical":
                    type_weight = (1 - theory_practice_ratio) * 2  # Scale to 0-2
                    score += type_weight * self.ranking_weights["practical"]
            else:
                # No preference specified, use base weights
                if content_type == "theoretical":
                    score += self.ranking_weights["theoretical"]
                elif content_type == "practical":
                    score += self.ranking_weights["practical"]

            # Factor 3: Item type weights
            item_type = result.get("result_type", "segment")
            if item_type == "concept":
                score += self.ranking_weights["concept"]
            else:
                score += self.ranking_weights["segment"]

            # Factor 4: Language match boost
            # If user specified a language and it matches the result's language
            result_language = result.get("language")
            if language and result_language and language == result_language:
                score += 0.5  # Boost score for language match

            # Store score and add to results
            result["relevance_score"] = round(score, 2)
            scored_results.append(result)

        # Sort by score, descending
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return scored_results

    def _generate_search_suggestions(
        self,
        query_text: str,
        results: List[Dict[str, Any]],
        domain: Optional[str],
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate helpful search suggestions based on current results with language support.

        Args:
            query_text: Original query text
            results: Search results
            domain: Current domain filter
            language: Optional language code

        Returns:
            List of search suggestion dictionaries
        """
        suggestions = []

        # Ensure query_text is not None and has sufficient length
        if not query_text or len(query_text) < 3:
            return []

        # Extract domains from results
        domains = set(r.get("domain") for r in results if r.get("domain"))

        # Suggest domain filters if multiple domains present
        if len(domains) > 1 and not domain:
            for result_domain in domains:
                if result_domain and result_domain != "unknown":
                    suggestions.append({
                        "type": "domain_filter",
                        "text": f'Search within "{result_domain}" domain',
                        "domain": result_domain,
                        "query": query_text,
                        "language": language  # Pass language to suggestion
                    })

        # Language-specific concept and practice terms
        concept_terms = {
            "en": ["definition", "explain", "concept", "theory", "mean"],
            "ru": ["определение", "объяснить", "концепция", "теория", "означать"]
        }

        practical_terms = {
            "en": ["example", "how to", "application", "implement", "code"],
            "ru": ["пример", "как", "применение", "реализация", "код"]
        }

        # Use appropriate language terms and ensure lowercase comparison
        lang_key = language if language in concept_terms else "en"
        current_concept_terms = concept_terms[lang_key]
        current_practical_terms = practical_terms[lang_key]

        # Convert query to lowercase safely
        query_text_lower = query_text.lower()

        # Suggest concept-focused search if general terms found
        if any(term in query_text_lower for term in current_concept_terms):
            suggestions.append({
                "type": "theory_focus",
                "text": f"Focus on theoretical explanations of {query_text}",
                "theory_practice_ratio": 0.8,
                "query": query_text,
                "language": language
            })

        # Suggest example-focused search if practical terms found
        if any(term in query_text_lower for term in current_practical_terms):
            suggestions.append({
                "type": "practice_focus",
                "text": f"Find practical examples of {query_text}",
                "theory_practice_ratio": 0.2,
                "query": query_text,
                "language": language
            })

        # Suggest learning path for complex subjects
        if len(results) > 5 and any(r.get("result_type") == "concept" for r in results):
            concept_ids = [r.get("concept_id") for r in results
                        if r.get("result_type") == "concept" and r.get("concept_id")]
            if concept_ids:
                suggestions.append({
                    "type": "learning_path",
                    "text": f"Create a learning path for {query_text}",
                    "concept_ids": concept_ids[:10],  # Limit to first 10 concepts
                    "language": language
                })

        # Language-specific suggestions
        if language != "en" and any(r.get("language") == "en" for r in results):
            suggestions.append({
                "type": "language_suggestion",
                "text": f"Search for English content on '{query_text}'",
                "language": "en",
                "query": query_text
            })
        elif language != "ru" and any(r.get("language") == "ru" for r in results):
            suggestions.append({
                "type": "language_suggestion",
                "text": f"Search for Russian content on '{query_text}'",
                "language": "ru",
                "query": query_text
            })

        return suggestions[:3]  # Limit to top 3 suggestions

    @time_function(5000)  # Log warning if takes more than 5 seconds
    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content for search with improved error handling.

        Args:
            processed_result: Processing result dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract key information for logging
            video_id = processed_result.get("video_id")
            if not video_id:
                logger.error("Missing video_id in processed result")
                return False

            # Index content using data access layer
            success = self.data_access.index_content(processed_result)

            if success:
                logger.info(f"Successfully indexed content for video {video_id}")
            else:
                logger.error(f"Failed to index content for video {video_id}")

            return success

        except Exception as e:
            logger.error(f"Error indexing content: {str(e)}")
            # Log detailed exception
            import traceback
            logger.debug(f"Indexing error details: {traceback.format_exc()}")
            return False

    @cached("concept")
    def get_concept_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept with improved related concepts.

        Args:
            concept_id: Concept ID

        Returns:
            Concept details dictionary or None if not found
        """
        try:
            # Get concept from data access layer
            concept = self.data_access.get_concept(concept_id)
            if not concept:
                return None

            # Get occurrences
            occurrences_query = """
            SELECT o.*, v.title as video_title, v.domain as video_domain,
                   s.text as segment_text
            FROM occurrences o
            JOIN videos v ON o.video_id = v.video_id
            JOIN segments s ON o.segment_id = s.segment_id
            WHERE o.concept_id = ?
            ORDER BY v.domain, o.video_id, o.start_time
            """
            occurrences = self.data_access.execute_query(occurrences_query, (concept_id,))

            # Group occurrences by video
            videos = {}
            domain_distribution = defaultdict(int)

            for occurrence in occurrences:
                video_id = occurrence["video_id"]
                domain = occurrence["video_domain"]
                domain_distribution[domain] += 1

                if video_id not in videos:
                    videos[video_id] = {
                        "video_id": video_id,
                        "title": occurrence["video_title"],
                        "domain": domain,
                        "occurrences": []
                    }

                # Add context text from segment
                context_text = occurrence["segment_text"] if "segment_text" in occurrence else occurrence["context_text"]

                videos[video_id]["occurrences"].append({
                    "occurrence_id": occurrence["occurrence_id"],
                    "segment_id": occurrence["segment_id"],
                    "start_time": occurrence["start_time"],
                    "end_time": occurrence["end_time"],
                    "context_type": occurrence["context_type"],
                    "context_text": context_text
                })

            # Find related concepts
            related_concepts = self._find_related_concepts(concept_id, concept["domain"])

            # Combine into result
            result = {
                "concept_id": concept_id,
                "text": concept["text"],
                "domain": concept["domain"],
                "concept_class": concept["concept_class"],
                "total_occurrences": concept["total_occurrences"],
                "videos": list(videos.values()),
                "domain_distribution": [
                    {"domain": domain, "count": count}
                    for domain, count in domain_distribution.items()
                ],
                "related_concepts": related_concepts
            }

            return result

        except Exception as e:
            logger.error(f"Error getting concept details for {concept_id}: {e}")
            return None

    def _find_related_concepts(self, concept_id: str, domain: str) -> List[Dict[str, Any]]:
        """
        Find concepts related to the given concept.

        Args:
            concept_id: Concept ID
            domain: Concept domain

        Returns:
            List of related concept dictionaries
        """
        try:
            # Find concepts that co-occur in the same videos
            co_occurrence_query = """
            SELECT c.concept_id, c.text, c.concept_class, c.domain,
                   COUNT(DISTINCT o1.video_id) as shared_videos
            FROM concepts c
            JOIN occurrences o1 ON c.concept_id = o1.concept_id
            JOIN occurrences o2 ON o1.video_id = o2.video_id
            WHERE o2.concept_id = ? AND c.concept_id != ?
            GROUP BY c.concept_id
            ORDER BY shared_videos DESC, c.total_occurrences DESC
            LIMIT 10
            """

            co_occurring = self.data_access.execute_query(co_occurrence_query, (concept_id, concept_id))

            # Find concepts in the same domain
            domain_query = """
            SELECT c.concept_id, c.text, c.concept_class, c.domain, c.total_occurrences
            FROM concepts c
            WHERE c.domain = ? AND c.concept_id != ?
            ORDER BY c.total_occurrences DESC
            LIMIT 10
            """

            domain_concepts = self.data_access.execute_query(domain_query, (domain, concept_id))

            # Combine and deduplicate
            related = {}

            # Add co-occurring concepts with relationship type
            for concept in co_occurring:
                concept_id = concept["concept_id"]
                related[concept_id] = {
                    "concept_id": concept_id,
                    "text": concept["text"],
                    "concept_class": concept["concept_class"],
                    "domain": concept["domain"],
                    "shared_videos": concept["shared_videos"],
                    "relationship": "co_occurrence"
                }

            # Add domain concepts not already included
            for concept in domain_concepts:
                concept_id = concept["concept_id"]
                if concept_id not in related:
                    related[concept_id] = {
                        "concept_id": concept_id,
                        "text": concept["text"],
                        "concept_class": concept["concept_class"],
                        "domain": concept["domain"],
                        "total_occurrences": concept["total_occurrences"],
                        "relationship": "same_domain"
                    }

            # Convert to list and sort
            result = list(related.values())
            result.sort(key=lambda x: x.get("shared_videos", 0), reverse=True)

            return result[:10]  # Limit to top 10

        except Exception as e:
            logger.error(f"Error finding related concepts: {e}")
            return []

    @cached("video")
    def get_video_concepts(self, video_id: str, context_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get concepts extracted from a video with improved organization.

        Args:
            video_id: YouTube video ID
            context_type: Optional context type filter

        Returns:
            Dictionary with video concepts or None if not found
        """
        try:
            # Use data access layer to get video concepts
            video_data = self.data_access.get_video_concepts(video_id)

            if not video_data:
                return None

            # Extract video and concept information
            video = video_data.get("video", {})
            concepts = video_data.get("concepts", [])

            # Filter concepts by context_type if specified
            if context_type:
                concepts = [c for c in concepts if c.get("concept_class") == context_type]

            # Group concepts by class (theoretical vs practical)
            theoretical_concepts = [c for c in concepts if c.get("concept_class") == "theoretical"]
            practical_concepts = [c for c in concepts if c.get("concept_class") == "practical"]

            # Get segments for the video
            segments_query = """
            SELECT s.*, COUNT(o.concept_id) as concept_count
            FROM segments s
            LEFT JOIN occurrences o ON s.segment_id = o.segment_id
            WHERE s.video_id = ?
            GROUP BY s.segment_id
            ORDER BY s.start_time
            """

            segments = self.data_access.execute_query(segments_query, (video_id,))

            # Create timeline data
            timeline = []
            for segment in segments:
                segment_type = segment.get("context_type", "mixed")
                if context_type and segment_type != context_type:
                    continue

                timeline.append({
                    "segment_id": segment["segment_id"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "text": segment["text"],
                    "context_type": segment_type,
                    "concept_count": segment["concept_count"]
                })

            # Build enhanced result
            result = {
                "video": video,
                "concepts": concepts,
                "theoretical_concepts": theoretical_concepts,
                "practical_concepts": practical_concepts,
                "timeline": timeline,
                "theory_practice_ratio": video.get("theory_practice_ratio", 0.5),
                "total_concepts": len(concepts),
                "theoretical_count": len(theoretical_concepts),
                "practical_count": len(practical_concepts)
            }

            return result

        except Exception as e:
            logger.error(f"Error getting video concepts for {video_id}: {e}")
            return None

    def generate_learning_path(
        self,
        concept_ids: List[str],
        theory_practice_ratio: float = 0.5,
        domain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an enhanced learning path for a set of concepts with improved sequencing.

        Args:
            concept_ids: List of concept IDs
            theory_practice_ratio: Desired ratio of theoretical to practical content
            domain: Optional domain filter

        Returns:
            Learning path dictionary or None if generation fails
        """
        try:
            if not concept_ids:
                return None

            # Get concept details for all concepts
            concepts = []
            for concept_id in concept_ids:
                concept = self.get_concept_details(concept_id)
                if concept:
                    concepts.append(concept)

            if not concepts:
                return None

            # Filter by domain if specified
            if domain:
                concepts = [c for c in concepts if c["domain"] == domain]

            # If no concepts remain after filtering, return None
            if not concepts:
                return None

            # Extract dependency graph - which concepts should precede others
            dependency_graph = self._extract_concept_dependencies(concepts)

            # Sequence concepts based on dependencies and theory/practice ratio
            sequenced_concepts = self._sequence_concepts(concepts, dependency_graph, theory_practice_ratio)

            # Organize into sections
            sections = self._organize_learning_path_sections(sequenced_concepts, theory_practice_ratio)

            # Find recommended videos for each concept
            for concept in sequenced_concepts:
                concept["recommended_videos"] = self._find_recommended_videos_for_concept(concept)

            # Calculate statistics
            theoretical_count = sum(1 for c in sequenced_concepts if c["concept_class"] == "theoretical")
            practical_count = sum(1 for c in sequenced_concepts if c["concept_class"] == "practical")
            total_count = theoretical_count + practical_count
            actual_ratio = theoretical_count / total_count if total_count > 0 else 0.5

            # Create learning path result
            result = {
                "concepts": sequenced_concepts,
                "sections": sections,
                "theory_practice_ratio": {
                    "requested": theory_practice_ratio,
                    "actual": actual_ratio
                },
                "domain": domain,
                "theoretical_concepts": theoretical_count,
                "practical_concepts": practical_count,
                "total_concepts": total_count
            }

            return result

        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return None

    def _extract_concept_dependencies(self, concepts: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """
        Extract dependency relationships between concepts.

        Args:
            concepts: List of concept dictionaries

        Returns:
            Dictionary mapping concept IDs to sets of prerequisite concept IDs
        """
        dependency_graph = defaultdict(set)

        # Get all concept IDs and texts for easy lookup
        concept_map = {c["concept_id"]: c for c in concepts}
        concept_text_to_id = {c["text"].lower(): c["concept_id"] for c in concepts}

        # Keywords that indicate dependencies
        dependency_indicators = [
            "requires", "depends on", "based on",
            "builds on", "extends", "uses"
        ]

        # Check each concept's occurrences for dependency indicators
        for concept in concepts:
            concept_id = concept["concept_id"]
            concept_text = concept["text"].lower()

            # Check each video's occurrences
            for video in concept.get("videos", []):
                for occurrence in video.get("occurrences", []):
                    context_text = occurrence.get("context_text", "").lower()

                    # Look for indicators of dependencies
                    for indicator in dependency_indicators:
                        if indicator in context_text:
                            # Look for other concepts in the same context
                            for other_concept_text, other_id in concept_text_to_id.items():
                                # Skip self-references
                                if other_id == concept_id:
                                    continue

                                if other_concept_text in context_text:
                                    # Check if the indicator appears between the concepts
                                    indicator_pos = context_text.find(indicator)
                                    other_pos = context_text.find(other_concept_text)

                                    # If other concept appears before the indicator
                                    # it might be a prerequisite
                                    if other_pos < indicator_pos:
                                        dependency_graph[concept_id].add(other_id)

        # Add theoretical foundations for practical concepts
        for concept in concepts:
            concept_id = concept["concept_id"]

            # If this is a practical concept, look for related theoretical concepts
            if concept["concept_class"] == "practical":
                for other_concept in concepts:
                    if other_concept["concept_class"] == "theoretical":
                        # Check if they share words
                        concept_words = set(concept["text"].lower().split())
                        other_words = set(other_concept["text"].lower().split())

                        # If significant overlap, consider the theoretical concept a prerequisite
                        if len(concept_words & other_words) / max(len(concept_words), 1) > 0.3:
                            dependency_graph[concept_id].add(other_concept["concept_id"])

        return dependency_graph

    def _sequence_concepts(
        self,
        concepts: List[Dict[str, Any]],
        dependency_graph: Dict[str, Set[str]],
        theory_practice_ratio: float
    ) -> List[Dict[str, Any]]:
        """
        Sequence concepts based on dependencies and theory/practice ratio.

        Args:
            concepts: List of concept dictionaries
            dependency_graph: Concept dependency graph
            theory_practice_ratio: Desired theory/practice ratio

        Returns:
            Sequenced list of concept dictionaries
        """
        # Copy concepts to avoid modifying originals
        concepts_copy = [concept.copy() for concept in concepts]

        # Sort theoretical concepts by complexity (estimated by word count and total occurrences)
        theoretical = [c for c in concepts_copy if c["concept_class"] == "theoretical"]
        theoretical.sort(key=lambda c: (len(c["text"].split()), c.get("total_occurrences", 0)))

        # Sort practical concepts similarly
        practical = [c for c in concepts_copy if c["concept_class"] == "practical"]
        practical.sort(key=lambda c: (len(c["text"].split()), c.get("total_occurrences", 0)))

        # Process based on theory/practice ratio
        if theory_practice_ratio > 0.7:
            # Theory-heavy path: Start with theoretical foundations, then practical applications
            sequenced = self._topological_sort(theoretical, dependency_graph)
            sequenced.extend(self._topological_sort(practical, dependency_graph))

        elif theory_practice_ratio < 0.3:
            # Practice-heavy path: Focus on practical concepts first
            sequenced = self._topological_sort(practical, dependency_graph)

            # Add theoretical concepts as needed
            for theoretical_concept in theoretical:
                # Check if this theoretical concept is a dependency for any practical concept
                is_dependency = False
                for practical_concept in practical:
                    practical_id = practical_concept["concept_id"]
                    if theoretical_concept["concept_id"] in dependency_graph.get(practical_id, set()):
                        is_dependency = True
                        break

                # Add dependencies or important theoretical concepts
                if is_dependency or theoretical_concept.get("total_occurrences", 0) > 5:
                    sequenced.append(theoretical_concept)

        else:
            # Balanced path: Alternate between theoretical and practical
            # While respecting dependencies
            sequenced = []
            theory_index = 0
            practice_index = 0

            # Sort each group by dependencies
            theoretical_sorted = self._topological_sort(theoretical, dependency_graph)
            practical_sorted = self._topological_sort(practical, dependency_graph)

            # Interleave while respecting dependencies
            while theory_index < len(theoretical_sorted) or practice_index < len(practical_sorted):
                # Add theoretical if available
                if theory_index < len(theoretical_sorted):
                    sequenced.append(theoretical_sorted[theory_index])
                    theory_index += 1

                # Add practical if available
                if practice_index < len(practical_sorted):
                    sequenced.append(practical_sorted[practice_index])
                    practice_index += 1

        # Add sequence order
        for i, concept in enumerate(sequenced):
            concept["sequence_order"] = i + 1

        return sequenced

    def _topological_sort(
        self,
        concepts: List[Dict[str, Any]],
        dependency_graph: Dict[str, Set[str]]
    ) -> List[Dict[str, Any]]:
        """
        Sort concepts in topological order based on dependencies.

        Args:
            concepts: List of concept dictionaries
            dependency_graph: Concept dependency graph

        Returns:
            Sorted list of concept dictionaries
        """
        # Create a map of concept_id to concept
        concept_map = {c["concept_id"]: c for c in concepts}

        # Get concept IDs in this set
        concept_ids = set(concept_map.keys())

        # Filter dependency graph to include only concepts in this set
        filtered_graph = {}
        for concept_id, deps in dependency_graph.items():
            if concept_id in concept_ids:
                filtered_deps = deps.intersection(concept_ids)
                if filtered_deps:
                    filtered_graph[concept_id] = filtered_deps

        # Perform topological sort
        visited = set()
        temp_visited = set()
        result = []

        def visit(concept_id):
            if concept_id in temp_visited:
                # Cycle detected - break the cycle
                return

            if concept_id in visited:
                return

            temp_visited.add(concept_id)

            # Visit dependencies first
            for dep_id in filtered_graph.get(concept_id, set()):
                visit(dep_id)

            temp_visited.remove(concept_id)
            visited.add(concept_id)

            if concept_id in concept_map:
                result.append(concept_map[concept_id])

        # Visit all concepts
        for concept_id in concept_ids:
            if concept_id not in visited:
                visit(concept_id)

        # Reverse the result to get correct order
        return list(reversed(result))

    def _organize_learning_path_sections(
        self,
        concepts: List[Dict[str, Any]],
        theory_practice_ratio: float
    ) -> List[Dict[str, Any]]:
        """
        Organize concepts into meaningful sections for a learning path.

        Args:
            concepts: Sequenced list of concept dictionaries
            theory_practice_ratio: Desired theory/practice ratio

        Returns:
            List of section dictionaries
        """
        if not concepts:
            return []

        # Determine the approach based on the theory/practice ratio
        if theory_practice_ratio > 0.7:
            # Theory-focused learning path
            sections = [
                {
                    "title": "Theoretical Foundations",
                    "description": "Core theoretical concepts and principles",
                    "concept_indices": []
                },
                {
                    "title": "Advanced Theory",
                    "description": "More complex theoretical concepts",
                    "concept_indices": []
                },
                {
                    "title": "Practical Applications",
                    "description": "Applying the theoretical knowledge",
                    "concept_indices": []
                }
            ]

            # Distribute concepts to sections
            for i, concept in enumerate(concepts):
                if concept["concept_class"] == "theoretical":
                    # Simple heuristic: early theoretical concepts go to foundations
                    position = i / len(concepts)
                    if position < 0.4:
                        sections[0]["concept_indices"].append(i)
                    else:
                        sections[1]["concept_indices"].append(i)
                else:
                    sections[2]["concept_indices"].append(i)

        elif theory_practice_ratio < 0.3:
            # Practice-focused learning path
            sections = [
                {
                    "title": "Getting Started",
                    "description": "Practical introduction to core concepts",
                    "concept_indices": []
                },
                {
                    "title": "Building Skills",
                    "description": "Practical skills and applications",
                    "concept_indices": []
                },
                {
                    "title": "Theoretical Background",
                    "description": "Understanding the underlying theory",
                    "concept_indices": []
                }
            ]

            # Distribute concepts to sections
            for i, concept in enumerate(concepts):
                if concept["concept_class"] == "practical":
                    # Simple heuristic: early practical concepts go to getting started
                    position = i / len(concepts)
                    if position < 0.4:
                        sections[0]["concept_indices"].append(i)
                    else:
                        sections[1]["concept_indices"].append(i)
                else:
                    sections[2]["concept_indices"].append(i)

        else:
            # Balanced learning path
            sections = [
                {
                    "title": "Core Concepts",
                    "description": "Fundamental ideas and principles",
                    "concept_indices": []
                },
                {
                    "title": "Practical Foundations",
                    "description": "Essential practical skills",
                    "concept_indices": []
                },
                {
                    "title": "Advanced Topics",
                    "description": "More complex concepts and applications",
                    "concept_indices": []
                }
            ]

            # Distribute concepts to sections
            for i, concept in enumerate(concepts):
                position = i / len(concepts)
                if position < 0.33:
                    sections[0]["concept_indices"].append(i)
                elif position < 0.67:
                    sections[1]["concept_indices"].append(i)
                else:
                    sections[2]["concept_indices"].append(i)

        # Remove empty sections
        sections = [s for s in sections if s["concept_indices"]]

        return sections

    def _find_recommended_videos_for_concept(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find the best videos for learning a particular concept.

        Args:
            concept: Concept dictionary

        Returns:
            List of recommended video dictionaries
        """
        recommended = []

        # Skip if no videos available
        if not concept.get("videos"):
            return recommended

        # Score each video based on occurrence quality
        video_scores = {}

        for video in concept["videos"]:
            video_id = video["video_id"]

            if not video.get("occurrences"):
                continue

            # Count occurrence types
            theoretical_count = 0
            practical_count = 0

            for occurrence in video["occurrences"]:
                if occurrence["context_type"] == "theoretical":
                    theoretical_count += 1
                elif occurrence["context_type"] == "practical":
                    practical_count += 1

            # Calculate score based on concept type preference
            total_occurrences = len(video["occurrences"])

            if concept["concept_class"] == "theoretical":
                # For theoretical concepts, prefer videos with more theoretical occurrences
                score = (theoretical_count * 1.5 + practical_count) / max(total_occurrences, 1)
            else:
                # For practical concepts, prefer videos with more practical occurrences
                score = (practical_count * 1.5 + theoretical_count) / max(total_occurrences, 1)

            # Bonus for multiple occurrences
            if total_occurrences > 3:
                score *= 1.2

            video_scores[video_id] = {
                "video_id": video_id,
                "title": video["title"],
                "score": score,
                "theoretical_occurrences": theoretical_count,
                "practical_occurrences": practical_count,
                "total_occurrences": total_occurrences,
                # Get first occurrence time as starting point
                "start_time": min(occ["start_time"] for occ in video["occurrences"])
            }

        # Sort by score
        ranked_videos = sorted(video_scores.values(), key=lambda x: x["score"], reverse=True)

        # Take top 3
        return ranked_videos[:3]

    def optimize_database(self) -> bool:
        """
        Optimize the search database.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Run VACUUM on SQLite database
            self.data_access.execute_update("VACUUM")

            # Run ANALYZE on tables for improved query planning
            self.data_access.execute_update("ANALYZE")

            # Optimize FTS5 tables
            self.data_access.execute_update("INSERT INTO search_index(search_index) VALUES('optimize')")

            logger.info("Database optimized successfully")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False

    def clear_search_cache(self) -> bool:
        """
        Clear the search cache to force fresh results.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear search cache
            cache_clear("search")
            logger.info("Search cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing search cache: {e}")
            return False

from collections import Counter  # Add this import at the top of the file
