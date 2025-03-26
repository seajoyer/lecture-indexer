"""
Enhanced Search Engine module for the Lecture Video Content Indexer.
Implements advanced search capabilities with repository pattern, caching, and performance monitoring.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import hashlib
import threading
import hashlib
import json

from database.db_init import get_db_context
from common.utils.performance_utils import measure_time, time_function, measure_memory

# Configure logging
logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Enhanced search engine for the Lecture Video Content Indexer.
    Implements repository pattern, caching, and performance monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Search Engine with configuration.

        Args:
            config: Configuration dictionary
        """
        with measure_time("SearchEngine.init"):
            logger.info("Initializing Enhanced Search Engine")
            self.config = config

            # Get database context
            self.db_context = get_db_context()
            if not self.db_context:
                raise RuntimeError("Database context not initialized")

            # Get repositories
            self.search_repository = self.db_context.search_repository
            self.video_repository = self.db_context.video_repository
            self.concept_repository = self.db_context.concept_repository

            # Initialize cache regions
            self.search_cache = self.db_context.get_cache_region("search")
            self.concept_cache = self.db_context.get_cache_region("concepts")
            self.video_cache = self.db_context.get_cache_region("videos")

            # Configure search settings from config
            self.use_stemming = config.get("use_stemming", True)
            self.use_fuzzy_matching = config.get("use_fuzzy_matching", True)
            self.min_ngram_size = config.get("min_ngram_size", 2)
            self.max_ngram_size = config.get("max_ngram_size", 3)
            self.fuzzy_match_threshold = config.get("fuzzy_match_threshold", 0.8)

            # Thread lock for synchronization
            self.lock = threading.RLock()

            logger.info("Enhanced Search Engine initialized")

    @time_function(threshold_ms=2000)
    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content through the search repository.

        Args:
            processed_result: Processing result dictionary

        Returns:
            True if indexing was successful, False otherwise
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
            theory_practice_results = processed_result.get("theory_practice_results", {})
            theory_practice_patterns = processed_result.get("theory_practice_patterns", {})

            # Clear any existing cache entries for this video
            self._clear_video_cache(video_id)

            # Index video metadata
            with measure_time(f"index_video_metadata_{video_id}", threshold_ms=500):
                video_saved = self.video_repository.save_video({
                    "video_id": video_id,
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "channel": metadata.get("channel", ""),
                    "publication_date": metadata.get("publication_date", ""),
                    "duration_seconds": metadata.get("duration_seconds", 0),
                    "language": metadata.get("language", ""),
                    "domain": metadata.get("domain", "unknown"),
                    "domain_confidence": metadata.get("domain_confidence", 0.0),
                    "theory_practice_ratio": theory_practice_results.get("theory_practice_ratio", 0.5),
                    "theoretical_segments": theory_practice_results.get("theoretical_segments", 0),
                    "practical_segments": theory_practice_results.get("practical_segments", 0),
                    "playlist_id": metadata.get("playlist_id")
                })

                if not video_saved:
                    logger.error(f"Failed to save video metadata for {video_id}")
                    return False

            # Index segments
            with measure_time(f"index_segments_{video_id}", threshold_ms=1000):
                segments = transcript.get("segments", [])
                segments_saved = self.video_repository.save_segments(video_id, segments)

                if not segments_saved:
                    logger.error(f"Failed to save segments for {video_id}")
                    return False

            # Index theory-practice patterns
            with measure_time(f"index_theory_practice_patterns_{video_id}", threshold_ms=500):
                patterns_saved = self.video_repository.save_theory_practice_patterns(video_id, theory_practice_patterns)

                if not patterns_saved:
                    logger.error(f"Failed to save theory-practice patterns for {video_id}")
                    return False

            # Extract and index concepts
            with measure_time(f"index_concepts_{video_id}", threshold_ms=1500):
                with measure_memory(f"index_concepts_memory_{video_id}", threshold_mb=50):
                    key_concepts = domain_features.get("key_concepts", [])

                    for concept in key_concepts:
                        # Save concept
                        concept_id = self.concept_repository.save_concept(concept)

                        if not concept_id:
                            logger.warning(f"Failed to save concept: {concept.get('text')}")
                            continue

                        # Find occurrences in segments
                        occurrences = self._find_concept_occurrences(
                            concept.get("text", ""),
                            video_id,
                            segments
                        )

                        # Save occurrences
                        if occurrences:
                            self.concept_repository.save_occurrences(concept_id, occurrences)

            # Index for search
            with measure_time(f"index_search_{video_id}", threshold_ms=1000):
                search_indexed = self.search_repository.index_video_metadata({
                    "video_id": video_id,
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "channel": metadata.get("channel", ""),
                    "domain": metadata.get("domain", "unknown")
                })

                if not search_indexed:
                    logger.error(f"Failed to index video metadata for search: {video_id}")
                    return False

                segments_indexed = self.search_repository.index_segments(video_id, segments)

                if not segments_indexed:
                    logger.error(f"Failed to index segments for search: {video_id}")
                    return False

            logger.info(f"Successfully indexed content for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return False

    def _find_concept_occurrences(
        self,
        concept_text: str,
        video_id: str,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find occurrences of a concept in video segments.

        Args:
            concept_text: Concept text
            video_id: Video ID
            segments: List of video segments

        Returns:
            List of occurrence dictionaries
        """
        occurrences = []
        concept_text_lower = concept_text.lower()

        for segment in segments:
            segment_text = segment.get("text", "").lower()
            segment_id = segment.get("id")

            if not segment_text or not segment_id:
                continue

            # Check if concept appears in segment
            if concept_text_lower in segment_text:
                # Get additional data
                start_time = segment.get("start_time", 0)
                end_time = segment.get("end_time", 0)
                context_type = segment.get("content_type", "mixed")

                # Calculate relevance score
                relevance_score = 0.7  # Default medium relevance

                # Adjust relevance based on position and frequency
                if segment_text.startswith(concept_text_lower):
                    relevance_score += 0.2  # Concept appears at the beginning

                # Count occurrences
                occurrence_count = segment_text.count(concept_text_lower)
                if occurrence_count > 1:
                    relevance_score += min(0.1, 0.02 * occurrence_count)  # Boost for multiple occurrences

                occurrences.append({
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "context_type": context_type,
                    "context_text": segment.get("text", ""),
                    "relevance_score": min(1.0, relevance_score)  # Cap at 1.0
                })

        return occurrences

    def _clear_video_cache(self, video_id: str):
        """
        Clear all cache entries related to a video.

        Args:
            video_id: Video ID
        """
        # Clear video metadata cache
        self.video_cache.delete(f"video_{video_id}")

        # Clear video segments cache
        self.video_cache.delete(f"segments_{video_id}")

        # Clear video concepts cache
        self.concept_cache.delete(f"video_concepts_{video_id}")

        # Clear search cache entries containing this video
        # This is more complex as we don't know which search queries might include this video
        # Better approach is to use time-based expiration for search results

    @time_function(threshold_ms=1000)
    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a search query with optimized caching and performance monitoring.
        """
        start_time = time.time()

        try:
            # Extract query fields
            query_text = query.get("original_text", "").strip()

            if not query_text:
                logger.warning("Empty search query")
                return {
                    "results": [],
                    "totalResults": 0,
                    "theoreticalResults": 0,
                    "practicalResults": 0,
                    "executionTimeMs": 0,
                    "query": query
                }

            # Check cache first
            if hasattr(self, 'cache'):
                # Create a cache key from the query
                cache_key = f"search_{hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Using cached search results for query: {query_text}")
                    return cached_result

            # Delegate search to repository
            with measure_time(f"search_repository_query", threshold_ms=500):
                search_results = self.search_repository.search(query)

            # Enhance results with additional data if needed
            with measure_time(f"enhance_search_results", threshold_ms=300):
                enhanced_results = self._enhance_search_results(search_results)

            # Cache the results if cache is available
            if hasattr(self, 'cache'):
                self.cache.set(cache_key, enhanced_results, ttl=300)  # Cache for 5 minutes

            logger.info(f"Search for '{query_text}' returned {enhanced_results.get('totalResults', 0)} results in {enhanced_results.get('executionTimeMs', 0)}ms")
            return enhanced_results

        except Exception as e:
            logger.error(f"Error executing search query: {e}")
            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "totalResults": 0,
                "theoreticalResults": 0,
                "practicalResults": 0,
                "executionTimeMs": execution_time_ms,
                "error": str(e),
                "query": query
            }

    def _enhance_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance search results with additional data.

        Args:
            results: Search results dictionary

        Returns:
            Enhanced search results
        """
        # No need to modify the results if there are none
        if not results.get("results"):
            return results

        # Create a copy to avoid modifying the original
        enhanced = results.copy()
        result_items = enhanced.get("results", [])

        # Get additional data for each result
        for item in result_items:
            # Add concept relationships if this is a concept result
            concept_id = item.get("concept_id")
            if concept_id and item.get("result_type") == "concept":
                # Check cache first
                cache_key = f"concept_relations_{concept_id}"
                relations = self.concept_cache.get(cache_key)

                if not relations:
                    # Get from repository
                    relations = self.concept_repository.get_concept_relationships(concept_id, limit=5)
                    # Cache for future use
                    self.concept_cache.set(cache_key, relations, ttl=3600)  # 1 hour TTL

                item["related_concepts"] = relations

        # Update the results with enhanced items
        enhanced["results"] = result_items
        return enhanced

    def get_concept_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept.

        Args:
            concept_id: Concept ID

        Returns:
            Concept details dictionary if found, None otherwise
        """
        # Check cache first
        if hasattr(self, 'video_cache'):
            cache_key = f"concept_details_{concept_id}"
            cached_result = self.video_cache.get(cache_key)
            if cached_result:
                return cached_result

        try:
            # Get basic concept information
            concept = self.concept_repository.get_concept(concept_id)
            if not concept:
                return None

            # Get occurrences
            occurrences = self.concept_repository.get_concept_occurrences(concept_id, limit=50)

            # Get related concepts
            related = self.concept_repository.get_concept_relationships(concept_id, limit=10)

            # Get videos containing this concept
            videos = self.concept_repository.get_videos_for_concept(concept_id, limit=20)

            # Compile result
            result = {
                "concept": concept,
                "occurrences": occurrences,
                "related": related,
                "videos": videos
            }

            # Cache the result if caching is available
            if hasattr(self, 'video_cache'):
                self.video_cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

            return result

        except Exception as e:
            logger.error(f"Error getting concept details: {e}")
            return None

    def get_video_concepts(self, video_id: str, context_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get concepts extracted from a video.

        Args:
            video_id: YouTube video ID
            context_type: Content type filter (theoretical, practical, mixed)

        Returns:
            Video concepts dictionary if found, None otherwise
        """
        # Check cache first
        if hasattr(self, 'video_cache'):
            cache_key = f"video_concepts_{video_id}_{context_type}"
            cached_result = self.video_cache.get(cache_key)
            if cached_result:
                logger.info(f"Using cached video concepts for video: {video_id}")
                return cached_result

        try:
            # Get video metadata
            video = self.video_repository.get_video(video_id)
            if not video:
                return None

            # Get concepts
            concepts = self.concept_repository.get_concepts_for_video(video_id, context_type=context_type)

            # Get theory-practice patterns
            patterns = self.video_repository.get_video_theory_practice_patterns(video_id)

            # Compile result
            result = {
                "video": video,
                "concepts": concepts,
                "theory_practice_patterns": patterns,
                "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
            }

            # Cache the result
            if hasattr(self, 'video_cache'):
                self.video_cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

            return result

        except Exception as e:
            logger.error(f"Error getting video concepts: {e}")
            return None

def generate_learning_path(
    self,
    concept_ids: List[str],
    theory_practice_ratio: float = 0.5,
    domain: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate a learning path for a set of concepts.

    Args:
        concept_ids: List of concept IDs
        theory_practice_ratio: Desired ratio of theoretical to practical content
        domain: Optional domain filter

    Returns:
        Learning path dictionary if successful, None otherwise
    """
    # Check cache first
    if hasattr(self, 'concept_cache'):
        # Create a cache key based on the inputs
        sorted_ids = sorted(concept_ids) # Sort for consistent cache key
        cache_key = f"learning_path_{'_'.join(sorted_ids)}_{theory_practice_ratio}_{domain}"
        cached_result = self.concept_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached learning path")
            return cached_result

    try:
        if not concept_ids:
            return None

        # Get all concepts
        concepts = []
        for concept_id in concept_ids:
            concept = self.concept_repository.get_concept(concept_id)
            if concept:
                concepts.append(concept)

        if not concepts:
            return None

        # Determine domain if not specified
        if not domain:
            domains = {}
            for concept in concepts:
                concept_domain = concept.get("domain")
                if concept_domain:
                    domains[concept_domain] = domains.get(concept_domain, 0) + 1

            if domains:
                domain = max(domains.items(), key=lambda x: x[1])[0]

        # Find related concepts
        related_concepts = []
        for concept in concepts:
            relations = self.concept_repository.get_concept_relationships(concept.get("concept_id"), limit=3)
            for relation in relations:
                related_concept = self.concept_repository.get_concept(relation.get("target_concept_id"))
                if related_concept and related_concept not in related_concepts and related_concept not in concepts:
                    related_concepts.append(related_concept)

        # Combine target and related concepts
        all_concepts = concepts + related_concepts

        # Sort concepts by theoretical/practical class based on desired ratio
        theoretical_concepts = [c for c in all_concepts if c.get("concept_class") == "theoretical"]
        practical_concepts = [c for c in all_concepts if c.get("concept_class") == "practical"]
        mixed_concepts = [c for c in all_concepts if c.get("concept_class") not in ("theoretical", "practical")]

        # Create learning path based on theory/practice ratio
        learning_path = self._create_balanced_learning_path(
            theoretical_concepts,
            practical_concepts,
            mixed_concepts,
            theory_practice_ratio
        )

        # Calculate total time
        total_time = sum(c.get("estimated_time_minutes", 10) for c in learning_path)

        # Compile result
        result = {
            "concepts": learning_path,
            "theory_practice_ratio": theory_practice_ratio,
            "total_theoretical_concepts": len(theoretical_concepts),
            "total_practical_concepts": len(practical_concepts),
            "estimated_total_time_minutes": total_time,
            "domain": domain
        }

        # Cache the result if caching is available
        if hasattr(self, 'concept_cache'):
            self.concept_cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

        return result

    except Exception as e:
        logger.error(f"Error generating learning path: {e}")
        return None

    def _create_balanced_learning_path(
        self,
        theoretical_concepts: List[Dict[str, Any]],
        practical_concepts: List[Dict[str, Any]],
        mixed_concepts: List[Dict[str, Any]],
        theory_practice_ratio: float
    ) -> List[Dict[str, Any]]:
        """
        Create a balanced learning path based on the desired theory/practice ratio.

        Args:
            theoretical_concepts: List of theoretical concepts
            practical_concepts: List of practical concepts
            mixed_concepts: List of mixed concepts
            theory_practice_ratio: Desired ratio of theoretical to practical content

        Returns:
            Balanced learning path
        """
        # Calculate target numbers based on ratio
        total_concepts = len(theoretical_concepts) + len(practical_concepts) + len(mixed_concepts)
        target_theoretical = int(total_concepts * theory_practice_ratio)
        target_practical = total_concepts - target_theoretical

        # Adjust theoretical and practical counts
        if len(theoretical_concepts) < target_theoretical and mixed_concepts:
            # Move some mixed concepts to theoretical
            mixed_to_theoretical = min(
                target_theoretical - len(theoretical_concepts),
                len(mixed_concepts)
            )
            theoretical_concepts.extend(mixed_concepts[:mixed_to_theoretical])
            mixed_concepts = mixed_concepts[mixed_to_theoretical:]

        if len(practical_concepts) < target_practical and mixed_concepts:
            # Move remaining mixed concepts to practical
            practical_concepts.extend(mixed_concepts)

        # Create alternating learning path based on ratio
        learning_path = []
        t_index = 0
        p_index = 0

        # Decide whether to start with theoretical or practical
        start_with_theoretical = theory_practice_ratio >= 0.5

        for i in range(total_concepts):
            if start_with_theoretical:
                # Add theoretical, then practical
                if i % 2 == 0 and t_index < len(theoretical_concepts):
                    concept = theoretical_concepts[t_index].copy()
                    concept["order"] = i + 1
                    learning_path.append(concept)
                    t_index += 1
                elif p_index < len(practical_concepts):
                    concept = practical_concepts[p_index].copy()
                    concept["order"] = i + 1
                    learning_path.append(concept)
                    p_index += 1
                elif t_index < len(theoretical_concepts):
                    concept = theoretical_concepts[t_index].copy()
                    concept["order"] = i + 1
                    learning_path.append(concept)
                    t_index += 1
            else:
                # Add practical, then theoretical
                if i % 2 == 0 and p_index < len(practical_concepts):
                    concept = practical_concepts[p_index].copy()
                    concept["order"] = i + 1
                    learning_path.append(concept)
                    p_index += 1
                elif t_index < len(theoretical_concepts):
                    concept = theoretical_concepts[t_index].copy()
                    concept["order"] = i + 1
                    learning_path.append(concept)
                    t_index += 1
                elif p_index < len(practical_concepts):
                    concept = practical_concepts[p_index].copy()
                    concept["order"] = i + 1
                    learning_path.append(concept)
                    p_index += 1

        # Add any remaining concepts
        remaining_theoretical = theoretical_concepts[t_index:]
        remaining_practical = practical_concepts[p_index:]

        for i, concept in enumerate(remaining_theoretical + remaining_practical):
            concept_copy = concept.copy()
            concept_copy["order"] = len(learning_path) + i + 1
            learning_path.append(concept_copy)

        return learning_path

    def batch_index_content(self, processed_results: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Index multiple processed content items in a batch for better performance.

        Args:
            processed_results: List of processing result dictionaries

        Returns:
            Dictionary mapping video IDs to indexing success status
        """
        if not processed_results:
            return {}

        results = {}

        for processed_result in processed_results:
            video_id = processed_result.get("video_id")
            if not video_id:
                continue

            # Index each video
            success = self.index_content(processed_result)
            results[video_id] = success

        return results

    def optimize_database(self) -> bool:
        """
        Optimize the database for better search performance.

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Optimize search indexes
            self.search_repository.optimize_search_indexes()

            # Optimize database
            self.db_context.optimize_database()

            logger.info("Database optimization completed")
            return True

        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False

    def rebuild_search_indexes(self) -> bool:
        """
        Rebuild all search indexes from scratch.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Rebuild search indexes
            success = self.search_repository.rebuild_search_indexes()

            if success:
                logger.info("Search indexes rebuilt successfully")

                # Clear all search-related caches
                self.search_cache.flush()
                logger.info("Search cache flushed")

            return success

        except Exception as e:
            logger.error(f"Error rebuilding search indexes: {e}")
            return False
