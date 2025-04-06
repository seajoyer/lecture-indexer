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
from cache_manager import cache_get, cache_set, cached, cache_clear
from performance_utils import time_function
from collections import Counter

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

            # NEW: Educational content weights
            "educational": 2.0,      # Strong boost for educational content (vs passing mention)
            "educational_weight": 0.4 # Multiplier for educational weight score
        }

    @time_function(2000)  # Log warning if takes more than 2 seconds
    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for content across indexed videos with improved handling of canonical concepts.

        Args:
            query: Query parameters dictionary

        Returns:
            Search results dictionary with deduplicated concepts
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

            offset = pagination.get("offset", 0)
            limit = pagination.get("limit", 10)

            # Check for empty query
            if not query_text:
                return {
                    "results": [],
                    "totalResults": 0,
                    "executionTimeMs": 0
                }

            # Generate cache key
            cache_key = f"search_{query_text}_{domain}_{theory_practice_ratio}_{offset}_{limit}_{language}"
            if filters:
                cache_key += f"_filters:{hash(str(filters))}"

            cached_result = cache_get("search", cache_key)
            if cached_result:
                logger.info(f"Using cached search results for query: {query_text}")
                return cached_result

            # Add language to the query for data access layer
            query["language"] = language

            # Execute search through data access layer
            base_results = self.data_access.search(query)

            if not base_results.get("results"):
                # Try synonym expansion if no results
                expanded_query = self._expand_query_with_synonyms(query_text, domain, language)
                if expanded_query and expanded_query != query_text:
                    logger.info(f"No results for '{query_text}', trying with synonyms: '{expanded_query}'")
                    query["original_text"] = expanded_query
                    base_results = self.data_access.search(query)

            # Handle canonical concept relationships
            base_results = self._apply_canonical_concept_filtering(base_results)

            # Enhanced ranking and processing of results
            enhanced_results = self._enhance_search_results(
                base_results,
                query_text,
                theory_practice_ratio,
                domain,
                language
            )

            # Create a more comprehensive list of related concepts
            # for all concept results to improve exploration
            concept_results = [r for r in enhanced_results.get("results", [])
                            if r.get("result_type") == "concept"]

            if concept_results:
                self._enhance_concept_results_with_occurrences(concept_results)

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

    def _enhance_concept_results_with_occurrences(self, concept_results: List[Dict[str, Any]]) -> None:
        """
        Enhance concept search results with occurrence count across all videos.

        Args:
            concept_results: List of concept search results to enhance
        """
        try:
            for concept in concept_results:
                concept_id = concept.get("concept_id")
                if not concept_id:
                    continue

                # Get all videos where this concept appears (including as variants)
                query = """
                WITH all_concept_ids AS (
                    SELECT ? AS id
                    UNION ALL
                    SELECT concept_id FROM concepts WHERE canonical_concept_id = ?
                )
                SELECT COUNT(DISTINCT o.video_id) as video_count,
                    COUNT(o.occurrence_id) as total_occurrences
                FROM occurrences o
                JOIN all_concept_ids c ON o.concept_id = c.id
                """

                result = self.data_access.execute_query(query, (concept_id, concept_id))

                if result and result[0]:
                    # Add occurrence information to concept
                    concept["video_count"] = result[0].get("video_count", 0)
                    concept["total_occurrences"] = result[0].get("total_occurrences", 0)

                    if result[0].get("video_count", 0) > 1:
                        # Make it clear this concept appears in multiple videos
                        concept["appears_in_multiple_videos"] = True

                # Get a sample of videos where this concept appears
                video_query = """
                WITH all_concept_ids AS (
                    SELECT ? AS id
                    UNION ALL
                    SELECT concept_id FROM concepts WHERE canonical_concept_id = ?
                )
                SELECT DISTINCT v.video_id, v.title
                FROM videos v
                JOIN occurrences o ON v.video_id = o.video_id
                JOIN all_concept_ids c ON o.concept_id = c.id
                LIMIT 3
                """

                videos = self.data_access.execute_query(video_query, (concept_id, concept_id))

                if videos:
                    concept["sample_videos"] = videos

        except Exception as e:
            logger.error(f"Error enhancing concept results: {e}")

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
                    "предел": ["сходимость", "граница"],
                    "шаровая функция": ["сферическая функция"]
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
                    "mass": ["inertia", "matter"],
                    "spherical harmonics": ["spherical function"]
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
                    "масса": ["инерция", "вещество", "материя"],
                    "шаровая функция": ["сферическая гармоника", "сферическая функция"],
                    "волновая функция": ["функция состояния", "пси-функция"]
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
                "метод": ["техника", "подход", "процедура", "способ"],
                "функция": ["отображение", "зависимость"]
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
        Prioritizes educational content over passing mentions.

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

        # Apply advanced ranking to results with educational content boost
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
        educational_counts = Counter() # Track educational vs passing mention counts

        for result in ranked_results:
            if result.get("result_type") == "concept":
                concept_id = result.get("concept_id")
                concepts[concept_id] = result

                # Track if this is an educational concept or passing mention
                is_educational = result.get("is_educational", False)
                educational_type = "educational" if is_educational else "passing_mention"
                educational_counts[educational_type] += 1

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
            ],
            "educationalDistribution": [
                {"type": edu_type, "count": count}
                for edu_type, count in educational_counts.most_common()
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
        Apply advanced ranking algorithm to search results with educational content boost.

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

            # Factor 5: Educational content boost (NEW)
            # Prioritize educational content over passing mentions
            is_educational = result.get("is_educational", False)
            educational_weight = result.get("educational_weight", 0.0)

            if is_educational:
                score += 2.0  # Significant boost for educational content

            # Add weighted educational score
            score += min(educational_weight, 5.0) * 0.4

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

        # NEW: Suggest focusing on educational content
        # Check if we have a mix of educational and passing mentions
        educational_concepts = [r for r in results if r.get("result_type") == "concept" and r.get("is_educational", False)]
        passing_mentions = [r for r in results if r.get("result_type") == "concept" and not r.get("is_educational", False)]

        if educational_concepts and passing_mentions and len(educational_concepts) < len(passing_mentions):
            suggestions.append({
                "type": "educational_focus",
                "text": f"Focus on educational explanations of {query_text}",
                "filter": "educational",
                "query": query_text,
                "language": language
            })

        # Suggest learning path for complex subjects
        if len(results) > 5 and any(r.get("result_type") == "concept" for r in results):
            # Prioritize educational concepts for learning paths
            concept_ids = [r.get("concept_id") for r in results
                        if r.get("result_type") == "concept" and r.get("concept_id") and r.get("is_educational", False)]

            # Fall back to all concepts if no educational ones found
            if not concept_ids:
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


    def _apply_cross_video_dedup(self, processed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply cross-video concept deduplication to link similar concepts between videos.

        Args:
            processed_result: Video processing result dictionary

        Returns:
            Updated processing result with cross-video concept links
        """
        video_id = processed_result.get("video_id")
        if not video_id:
            return processed_result

        # Get domain features with concepts
        domain_features = processed_result.get("domain_features", {})
        video_concepts = domain_features.get("key_concepts", [])

        if not video_concepts:
            return processed_result

        # Get language from processed result
        language = processed_result.get("transcript", {}).get("language", "en")

        # For each concept in this video, find similar concepts in other videos
        updated_concepts = []

        for concept in video_concepts:
            concept_text = concept.get("text", "").lower()
            normalized_text = concept.get("normalized_text", concept_text)

            # Skip if this concept is already a variant
            if concept.get("canonical_concept_id"):
                updated_concepts.append(concept)
                continue

            # Try to find similar concepts across all videos
            query = f"""
            SELECT c.concept_id, c.text, c.normalized_text, c.domain,
                c.canonical_concept_id, v.video_id
            FROM concepts c
            JOIN occurrences o ON c.concept_id = o.concept_id
            JOIN videos v ON o.video_id = v.video_id
            WHERE
                (c.normalized_text = ? OR c.text = ?) AND
                v.video_id != ? AND
                (c.canonical_concept_id IS NULL OR c.canonical_concept_id = '')
            LIMIT 5
            """

            similar_concepts = self.data_access.execute_query(
                query, (normalized_text, concept_text, video_id)
            )

            # If we found similar concepts in other videos
            if similar_concepts:
                # Use the first one as canonical
                canonical = similar_concepts[0]
                canonical_id = canonical.get("concept_id")

                # Mark this concept as a variant
                updated_concept = concept.copy()
                updated_concept["canonical_concept_id"] = canonical_id

                logger.info(f"Linked concept '{concept_text}' to canonical concept {canonical_id} from video {canonical.get('video_id')}")

                updated_concepts.append(updated_concept)
            else:
                # No similar concepts found, keep as is (this will be a new canonical concept)
                updated_concepts.append(concept)

        # Update the domain features with the modified concepts
        domain_features["key_concepts"] = updated_concepts
        processed_result["domain_features"] = domain_features

        return processed_result

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

            # If this is a variant concept (has a canonical_concept_id), get the canonical concept instead
            if concept.get("canonical_concept_id"):
                canonical_id = concept.get("canonical_concept_id")
                logger.info(f"Redirecting to canonical concept {canonical_id} from variant {concept_id}")
                concept = self.data_access.get_concept(canonical_id)
                if not concept:
                    return None
                concept_id = canonical_id

            # Get all variant concept IDs for this canonical concept
            variant_ids = []
            if concept.get("canonical_concept_id") is None or concept.get("canonical_concept_id") == "":
                variant_query = """
                SELECT concept_id, text
                FROM concepts
                WHERE canonical_concept_id = ?
                """
                variants = self.data_access.execute_query(variant_query, (concept_id,))
                variant_ids = [v["concept_id"] for v in variants]

            # Log how many variant concepts were found
            logger.info(f"Found {len(variant_ids)} variant concepts for canonical concept {concept_id}")

            # Build query for occurrences - include both canonical and variant concepts
            all_concept_ids = [concept_id] + variant_ids
            placeholders = ",".join(["?"] * len(all_concept_ids))

            occurrences_query = f"""
            SELECT o.*, v.title as video_title, v.domain as video_domain,
                s.text as segment_text
            FROM occurrences o
            JOIN videos v ON o.video_id = v.video_id
            JOIN segments s ON o.segment_id = s.segment_id
            WHERE o.concept_id IN ({placeholders})
            ORDER BY v.domain, o.video_id, o.start_time
            """
            occurrences = self.data_access.execute_query(occurrences_query, tuple(all_concept_ids))

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

            # Log how many occurrences and videos were found
            logger.info(f"Found {len(occurrences)} occurrences across {len(videos)} videos for concept {concept_id} and its variants")

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
                "related_concepts": related_concepts,
                "variant_concept_ids": variant_ids
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
            WHERE o2.concept_id = ?
            AND c.concept_id != ?
            -- Only include canonical concepts
            AND (c.canonical_concept_id IS NULL OR c.canonical_concept_id = '')
            GROUP BY c.concept_id
            ORDER BY shared_videos DESC, c.total_occurrences DESC
            LIMIT 10
            """

            co_occurring = self.data_access.execute_query(co_occurrence_query, (concept_id, concept_id))

            # Find concepts in the same domain
            domain_query = """
            SELECT c.concept_id, c.text, c.concept_class, c.domain, c.total_occurrences
            FROM concepts c
            WHERE c.domain = ?
            AND c.concept_id != ?
            -- Only include canonical concepts
            AND (c.canonical_concept_id IS NULL OR c.canonical_concept_id = '')
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
            all_concepts = video_data.get("concepts", [])

            # Filter concepts by context_type if specified
            if context_type:
                all_concepts = [c for c in all_concepts if c.get("concept_class") == context_type]

            # Group concepts by class (theoretical vs practical)
            theoretical_concepts = [c for c in all_concepts if c.get("concept_class") == "theoretical"]
            practical_concepts = [c for c in all_concepts if c.get("concept_class") == "practical"]

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
                "concepts": all_concepts,
                "theoretical_concepts": theoretical_concepts,
                "practical_concepts": practical_concepts,
                "timeline": timeline,
                "theory_practice_ratio": video.get("theory_practice_ratio", 0.5),
                "total_concepts": len(all_concepts),
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

            # Handle potential variant concepts by mapping to their canonical concepts
            canonical_concept_ids = []
            for concept_id in concept_ids:
                concept = self.data_access.get_concept(concept_id)
                if not concept:
                    continue

                if concept.get("canonical_concept_id"):
                    # This is a variant, use its canonical concept instead
                    canonical_id = concept.get("canonical_concept_id")
                    if canonical_id not in canonical_concept_ids:
                        canonical_concept_ids.append(canonical_id)
                else:
                    # This is already a canonical concept
                    canonical_concept_ids.append(concept_id)

            # Replace original concept IDs with canonical versions
            concept_ids = canonical_concept_ids

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

    def _apply_canonical_concept_filtering(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply canonical concept filtering to search results with improved text-based deduplication.
        Ensures that only canonical concepts are included in results, with variants merged.
        Also ensures concepts with the same text are properly deduplicated.
        Preserves educational content metrics in the results.

        Args:
            search_results: Original search results

        Returns:
            Updated search results with canonical concepts only and no duplicates
        """
        if not search_results or not search_results.get("results"):
            return search_results

        results = search_results.get("results", [])

        # Process only if there are concept results
        concept_results = [r for r in results if r.get("result_type") == "concept"]

        if not concept_results:
            return search_results  # No concepts to process

        # Extract concept IDs that need checking
        concept_ids = [r.get("concept_id") for r in concept_results if r.get("concept_id")]

        if not concept_ids:
            return search_results

        # Find canonical mappings for all these concepts
        canonical_map = {}
        seen_canonical_ids = set()  # Track canonical IDs we've already included

        try:
            # First get all canonical relationships
            placeholders = ", ".join(["?"] * len(concept_ids))
            query = f"""
            SELECT concept_id, canonical_concept_id, text, normalized_text, language,
                educational_weight, is_educational
            FROM concepts
            WHERE concept_id IN ({placeholders})
            """

            canon_results = self.data_access.execute_query(query, tuple(concept_ids))

            # Build mapping from concept ID to canonical ID
            for result in canon_results:
                concept_id = result.get("concept_id")
                canonical_id = result.get("canonical_concept_id")

                if canonical_id:
                    canonical_map[concept_id] = canonical_id

            # Now fetch canonical concepts so we have their data
            canonical_ids = list(set(canonical_map.values()))
            if canonical_ids:
                placeholders = ", ".join(["?"] * len(canonical_ids))
                query = f"""
                SELECT *
                FROM concepts
                WHERE concept_id IN ({placeholders})
                """
                canonical_concepts_data = self.data_access.execute_query(query, tuple(canonical_ids))
                canonical_concepts = {c["concept_id"]: c for c in canonical_concepts_data}
            else:
                canonical_concepts = {}

        except Exception as e:
            logger.warning(f"Error checking canonical concepts: {e}")
            canonical_map = {}
            canonical_concepts = {}

        # Filter and deduplicate results
        filtered_results = []

        # Dictionary to track canonical concepts by ID
        included_canonical_concepts = {}

        # Dictionary to deduplicate by text+language
        seen_concept_texts = {}

        # First process non-concept results (keep all of them)
        for result in results:
            if result.get("result_type") != "concept":
                filtered_results.append(result)
                continue

            concept_id = result.get("concept_id")
            if not concept_id:
                filtered_results.append(result)
                continue

            # Create a unique key by text+language to avoid duplicates
            text_key = f"{result.get('text', '').lower()}_{result.get('language', '')}"

            # Skip if we've already seen this text
            if text_key in seen_concept_texts:
                continue

            # Check if this is a variant concept
            canonical_id = canonical_map.get(concept_id)

            if not canonical_id:
                # This is already a canonical concept or has no canonical relationship
                # Only include if we haven't already seen this canonical ID
                if concept_id not in seen_canonical_ids:
                    filtered_results.append(result)
                    seen_canonical_ids.add(concept_id)
                    included_canonical_concepts[concept_id] = result
                    seen_concept_texts[text_key] = True
                continue

            # Skip if we've already included this canonical concept
            if canonical_id in seen_canonical_ids:
                continue

            # Get canonical concept data
            canonical_concept = canonical_concepts.get(canonical_id)

            if canonical_concept:
                # Create a result entry for the canonical concept with merged metadata
                # Use the higher relevance score between variant and canonical
                relevance_score = result.get("relevance_score", 0)
                if canonical_id in included_canonical_concepts:
                    # If we already have this canonical concept in results from another variant,
                    # use the higher relevance score
                    existing_result = included_canonical_concepts[canonical_id]
                    if relevance_score > existing_result.get("relevance_score", 0):
                        # Update existing result with higher score
                        existing_result["relevance_score"] = relevance_score
                        # Keep track of the variant that caused this result
                        if "variant_matches" not in existing_result:
                            existing_result["variant_matches"] = []
                        existing_result["variant_matches"].append(result.get("text"))
                    continue
                else:
                    # Create new canonical result with merged metadata
                    canonical_result = {
                        "result_type": "concept",
                        "concept_id": canonical_id,
                        "text": canonical_concept.get("text", result.get("text")),
                        "domain": canonical_concept.get("domain", result.get("domain")),
                        "context_type": canonical_concept.get("concept_class", result.get("context_type")),
                        "concept_class": canonical_concept.get("concept_class", result.get("concept_class")),
                        "language": canonical_concept.get("language", result.get("language")),
                        "relevance_score": relevance_score,
                        "is_canonical": True,
                        "variant_matches": [result.get("text")],
                        # Include educational content metrics
                        "educational_weight": canonical_concept.get("educational_weight", 0.0),
                        "is_educational": bool(canonical_concept.get("is_educational", 0))
                    }

                    # Copy over other fields if available
                    if "video_id" in result:
                        canonical_result["video_id"] = result.get("video_id")
                    if "video_title" in result:
                        canonical_result["video_title"] = result.get("video_title")

                    filtered_results.append(canonical_result)
                    seen_canonical_ids.add(canonical_id)
                    included_canonical_concepts[canonical_id] = canonical_result
                    # Mark the canonical concept text as seen
                    canonical_text_key = f"{canonical_concept.get('text', '').lower()}_{canonical_concept.get('language', '')}"
                    seen_concept_texts[canonical_text_key] = True
            else:
                # Couldn't find canonical concept, use the original
                filtered_results.append(result)
                seen_canonical_ids.add(concept_id)  # Mark as seen to prevent duplicates
                seen_concept_texts[text_key] = True

        # Re-sort results by relevance score
        filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Update the result count and return
        search_results["results"] = filtered_results
        search_results["totalResults"] = len(filtered_results)

        return search_results

    @time_function(5000)  # Log warning if takes more than 5 seconds
    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content for search with improved error handling and batching.
        Adds educational content metrics to search index.

        Args:
            processed_result: Processing result dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract key fields
            video_id = processed_result.get("video_id")
            if not video_id:
                logger.error("Missing video_id for indexing")
                return False

            metadata = processed_result.get("metadata", {})
            transcript = processed_result.get("transcript", {})
            domain_features = processed_result.get("domain_features", {})

            # Get language from transcript
            language = transcript.get("language", "en")
            domain = metadata.get("domain", "unknown")

            # Clear existing content for this video to avoid duplicates
            self.data_access.execute_update("DELETE FROM segments WHERE video_id = ?", (video_id,))
            self.data_access.execute_update("DELETE FROM occurrences WHERE video_id = ?", (video_id,))
            self.data_access.execute_update(
                "DELETE FROM search_index WHERE (item_type = 'segment' OR video_id = ?) AND video_id = ?",
                (video_id, video_id)
            )

            # Save video metadata
            video_data = {
                "video_id": video_id,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "channel": metadata.get("channel", ""),
                "publication_date": metadata.get("publication_date", ""),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "language": language,
                "domain": domain,
                "domain_confidence": metadata.get("domain_confidence", 0.0),
                "theory_practice_ratio": processed_result.get("theory_practice_results", {}).get("theory_practice_ratio", 0.5),
                "theoretical_segments": processed_result.get("theory_practice_results", {}).get("theoretical_segments", 0),
                "practical_segments": processed_result.get("theory_practice_results", {}).get("practical_segments", 0),
                "processing_status": "completed",
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            video_saved = self.data_access.save_video(video_data)
            if not video_saved:
                logger.error(f"Failed to save video metadata for {video_id}")
                return False

            # Save segments
            segments = transcript.get("segments", [])

            # Add domain and language to each segment if not present
            for segment in segments:
                if "domain" not in segment:
                    segment["domain"] = domain
                if "language" not in segment:
                    segment["language"] = language

            segments_saved = self.data_access.save_segments(video_id, segments)
            if not segments_saved:
                logger.error(f"Failed to save segments for {video_id}")
                return False

            # Save concepts in batches - process theoretical and practical concepts separately
            theoretical_concepts = domain_features.get("theoretical_concepts", [])
            practical_concepts = domain_features.get("practical_concepts", [])

            all_concepts = theoretical_concepts + practical_concepts

            if not all_concepts:
                logger.warning(f"No concepts found for video {video_id}")
                # We still want to return True as the segments were saved successfully
                return True

            # Make sure all concepts have domain and language
            for concept in all_concepts:
                if "domain" not in concept:
                    concept["domain"] = domain
                if "language" not in concept:
                    concept["language"] = language
                # Ensure normalized_text is present
                if "normalized_text" not in concept and "text" in concept:
                    # Use the concept text as normalized_text if not provided
                    concept["normalized_text"] = concept["text"].lower()

            # Process concepts in batches of 20 for better performance
            batch_size = 20
            successful_concepts = 0
            concept_ids = []

            for i in range(0, len(all_concepts), batch_size):
                batch = all_concepts[i:i + batch_size]
                for concept in batch:
                    concept_data = concept.copy()
                    concept_data["video_id"] = video_id

                    # Get educational metrics if available
                    concept_data["educational_weight"] = concept.get("educational_weight", 0.0)
                    concept_data["is_educational"] = concept.get("is_educational",
                                                            concept.get("educational_weight", 0) > 2.5)

                    concept_id = self.data_access.save_concept(concept_data)
                    if concept_id:
                        successful_concepts += 1
                        concept_ids.append(concept_id)

                        # Process occurrences if available
                        occurrences = concept.get("occurrences", [])
                        if occurrences:
                            # Ensure each occurrence has the right concept_id and video_id
                            for occurrence in occurrences:
                                occurrence["concept_id"] = concept_id
                                occurrence["video_id"] = video_id

                            # Save occurrences
                            self.data_access.save_occurrences(concept_id, occurrences)

            # Log success and clear related caches
            logger.info(f"Successfully indexed {successful_concepts}/{len(all_concepts)} concepts for video {video_id}")

            # Clear caches related to this video
            self.data_access.clear_cache(f"video_{video_id}")
            self.data_access.clear_cache(f"segments_{video_id}")
            self.data_access.clear_cache(f"video_concepts_{video_id}")
            self.data_access.clear_cache(f"video_concept_data_{video_id}")

            # Clear search cache
            from cache_manager import cache_clear
            cache_clear("search")

            return True

        except Exception as e:
            logger.error(f"Error indexing content: {e}")

            # Log detailed exception but only in debug mode
            import traceback
            logger.debug(f"Indexing error details: {traceback.format_exc()}")

            return False
