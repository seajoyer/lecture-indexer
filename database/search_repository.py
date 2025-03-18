"""
Search Repository module for the Lecture Video Content Indexer.
Handles maintenance and optimization of search indexes with efficient querying capabilities.
"""

import logging
import re
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from database.db_manager import DBManager

# Configure logging
logger = logging.getLogger(__name__)

class SearchRepository:
    """
    Repository for search index data with optimized querying capabilities.
    Provides methods to maintain and query search indexes for videos, segments, and concepts.
    """

    def __init__(self, db_manager: DBManager):
        """
        Initialize the search repository.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._ensure_schema()
        logger.info("SearchRepository initialized")

    def _ensure_schema(self):
        """Ensure search-related tables exist in the database."""
        schema_script = """
        -- FTS5 table for segment content
        CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
            segment_id,
            video_id,
            text,
            text_stemmed,
            domain,
            context_type,
            tokenize='porter unicode61 remove_diacritics 2'
        );

        -- FTS5 table for video metadata
        CREATE VIRTUAL TABLE IF NOT EXISTS video_metadata_fts USING fts5(
            video_id,
            title,
            title_stemmed,
            description,
            description_stemmed,
            channel,
            domain,
            tokenize='porter unicode61 remove_diacritics 2'
        );

        -- N-gram index for fuzzy matching
        CREATE TABLE IF NOT EXISTS ngram_index (
            item_id TEXT,
            item_type TEXT,  -- segment, concept, video
            ngram TEXT,
            PRIMARY KEY (item_id, item_type, ngram)
        );

        -- Create index on ngram for faster searches
        CREATE INDEX IF NOT EXISTS idx_ngram_index_ngram ON ngram_index(ngram);

        -- Search history table
        CREATE TABLE IF NOT EXISTS search_history (
            search_id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            query_params TEXT,
            result_count INTEGER,
            execution_time_ms INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            client_info TEXT
        );

        -- Search result feedback table
        CREATE TABLE IF NOT EXISTS search_feedback (
            feedback_id TEXT PRIMARY KEY,
            search_id TEXT,
            item_id TEXT,
            item_type TEXT,
            relevance_score INTEGER,  -- 1-5
            clicked BOOLEAN,
            time_spent_seconds INTEGER,
            feedback_text TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            FOREIGN KEY (search_id) REFERENCES search_history(search_id)
        );

        -- Popular searches table for search suggestions
        CREATE TABLE IF NOT EXISTS popular_searches (
            query_text TEXT PRIMARY KEY,
            count INTEGER DEFAULT 1,
            last_searched TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Search synonyms table for query expansion
        CREATE TABLE IF NOT EXISTS search_synonyms (
            term TEXT,
            synonym TEXT,
            domain TEXT,
            source TEXT,  -- 'manual', 'automatic'
            PRIMARY KEY (term, synonym, domain)
        );

        -- Create triggers to maintain FTS indexes
        CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
            INSERT INTO segments_fts(segment_id, video_id, text, text_stemmed, domain, context_type)
            VALUES (new.segment_id, new.video_id, new.text, new.text_stemmed, new.domain, new.context_type);
        END;

        CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
            DELETE FROM segments_fts WHERE segment_id = old.segment_id;
        END;

        CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
            DELETE FROM segments_fts WHERE segment_id = old.segment_id;
            INSERT INTO segments_fts(segment_id, video_id, text, text_stemmed, domain, context_type)
            VALUES (new.segment_id, new.video_id, new.text, new.text_stemmed, new.domain, new.context_type);
        END;

        -- Video metadata triggers
        CREATE TRIGGER IF NOT EXISTS video_metadata_ai AFTER INSERT ON videos BEGIN
            INSERT INTO video_metadata_fts(
                video_id, title, title_stemmed, description, description_stemmed, channel, domain
            ) VALUES (
                new.video_id, new.title, '', new.description, '', new.channel, new.domain
            );
        END;

        CREATE TRIGGER IF NOT EXISTS video_metadata_ad AFTER DELETE ON videos BEGIN
            DELETE FROM video_metadata_fts WHERE video_id = old.video_id;
        END;

        -- We don't need an update trigger for videos as we'll handle stemming elsewhere
        """

        try:
            self.db_manager.execute_script(schema_script)
            logger.info("Search repository schema initialized")
        except Exception as e:
            logger.error(f"Error initializing search repository schema: {e}")
            raise

    def index_video_metadata(self, video_data: Dict[str, Any]) -> bool:
        """
        Index video metadata for search.

        Args:
            video_data: Video metadata dictionary

        Returns:
            True if successful, False otherwise
        """
        video_id = video_data.get("video_id")
        if not video_id:
            logger.error("Cannot index video metadata without video_id")
            return False

        try:
            # Get existing FTS record
            query = "SELECT 1 FROM video_metadata_fts WHERE video_id = ?"
            result = self.db_manager.execute_query(query, (video_id,))
            exists = bool(result)

            title = video_data.get("title", "")
            description = video_data.get("description", "")
            channel = video_data.get("channel", "")
            domain = video_data.get("domain", "unknown")

            # Create stemmed versions
            title_stemmed = self._stem_text(title.lower())
            description_stemmed = self._stem_text(description.lower())

            # Update or insert FTS record
            if exists:
                query = """
                DELETE FROM video_metadata_fts WHERE video_id = ?
                """
                self.db_manager.execute_update(query, (video_id,))

            query = """
            INSERT INTO video_metadata_fts (
                video_id, title, title_stemmed, description, description_stemmed, channel, domain
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self.db_manager.execute_update(query, (
                video_id, title, title_stemmed, description, description_stemmed, channel, domain
            ))

            # Index title and description n-grams for fuzzy matching
            self._index_ngrams(video_id, "video", title + " " + description)

            logger.info(f"Indexed metadata for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing metadata for video {video_id}: {e}")
            return False

    def index_segments(self, video_id: str, segments: List[Dict[str, Any]]) -> bool:
        """
        Index video transcript segments for search.

        Args:
            video_id: Video ID
            segments: List of segment dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not video_id or not segments:
            logger.error("Cannot index segments without video_id or segments")
            return False

        try:
            # Clear existing segment FTS entries
            query = "DELETE FROM segments_fts WHERE video_id = ?"
            self.db_manager.execute_update(query, (video_id,))

            # Prepare batch insert
            query = """
            INSERT INTO segments_fts (
                segment_id, video_id, text, text_stemmed, domain, context_type
            ) VALUES (?, ?, ?, ?, ?, ?)
            """

            params_list = []
            for segment in segments:
                segment_id = segment.get("id")
                if not segment_id:
                    continue

                text = segment.get("text", "")
                domain = segment.get("domain", "unknown")
                context_type = segment.get("content_type", "mixed")

                # Create stemmed version
                text_stemmed = self._stem_text(text.lower())

                params_list.append((
                    segment_id, video_id, text, text_stemmed, domain, context_type
                ))

                # Index segment n-grams for fuzzy matching
                self._index_ngrams(segment_id, "segment", text)

            # Execute batch insert
            if params_list:
                self.db_manager.execute_many(query, params_list)

            logger.info(f"Indexed {len(params_list)} segments for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing segments for video {video_id}: {e}")
            return False

    def _index_ngrams(self, item_id: str, item_type: str, text: str, min_size: int = 2, max_size: int = 3) -> None:
        """
        Index n-grams for fuzzy matching.

        Args:
            item_id: Item ID (segment_id, concept_id, video_id)
            item_type: Type of item ('segment', 'concept', 'video')
            text: Text to generate n-grams from
            min_size: Minimum n-gram size
            max_size: Maximum n-gram size
        """
        # Clear existing n-grams
        query = "DELETE FROM ngram_index WHERE item_id = ? AND item_type = ?"
        self.db_manager.execute_update(query, (item_id, item_type))

        # Extract significant tokens (filter out short words and stopwords)
        tokens = self._extract_significant_tokens(text)

        # Generate n-grams for each token
        ngrams = []
        for token in tokens:
            if len(token) >= min_size + 1:  # Only generate n-grams for tokens of sufficient length
                token_ngrams = self._generate_ngrams(token, min_size, max_size)
                ngrams.extend([(item_id, item_type, ngram) for ngram in token_ngrams])

        # Insert n-grams in batches
        if ngrams:
            query = "INSERT OR IGNORE INTO ngram_index (item_id, item_type, ngram) VALUES (?, ?, ?)"
            self.db_manager.execute_many(query, ngrams)

    def _generate_ngrams(self, text: str, min_size: int, max_size: int) -> List[str]:
        """
        Generate character n-grams for fuzzy matching.

        Args:
            text: Text to generate n-grams from
            min_size: Minimum n-gram size
            max_size: Maximum n-gram size

        Returns:
            List of n-grams
        """
        ngrams = []
        text = text.lower()

        for n in range(min_size, min(max_size + 1, len(text) + 1)):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])

        return ngrams

    def _extract_significant_tokens(self, text: str) -> List[str]:
        """
        Extract significant tokens from text.

        Args:
            text: Text to extract tokens from

        Returns:
            List of significant tokens
        """
        # Normalize text
        text = self._normalize_text(text)

        # Extract tokens
        tokens = text.split()

        # Filter tokens
        filtered_tokens = []
        stopwords = self._get_stopwords()

        for token in tokens:
            # Filter out short tokens and stopwords
            if len(token) > 2 and token.lower() not in stopwords:
                filtered_tokens.append(token)

        return filtered_tokens

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing punctuation, etc.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation except for internal hyphens and apostrophes
        text = re.sub(r'[^\w\s\'-]|(?<=\s)[\'\\-]|[\'\\-](?=\s)', ' ', text)

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _stem_text(self, text: str) -> str:
        """
        Apply simple stemming to text.

        Args:
            text: Text to stem

        Returns:
            Stemmed text
        """
        # This is a very simple implementation
        # In a real implementation, use a proper stemmer like Porter Stemmer
        words = text.split()
        stemmed_words = []

        for word in words:
            # Simple suffix removal
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s'):
                word = word[:-1]
            elif word.endswith('ly'):
                word = word[:-2]
            elif word.endswith('ies'):
                word = word[:-3] + 'y'
            elif word.endswith('es'):
                word = word[:-2]

            stemmed_words.append(word)

        return ' '.join(stemmed_words)

    def _get_stopwords(self) -> Set[str]:
        """
        Get stopwords for filtering.

        Returns:
            Set of stopwords
        """
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now'
        }

    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a search query with advanced processing.

        Args:
            query: Structured query dictionary containing:
                - original_text: Original query text
                - filters: Query filters
                - theory_practice_ratio: Optional theory/practice ratio filter
                - domain: Optional domain filter
                - pagination: Pagination options

        Returns:
            Search results dictionary
        """
        start_time = time.time()

        try:
            # Extract query fields
            query_text = query.get("original_text", "").strip()
            filters = query.get("filters", {})
            theory_practice_ratio = query.get("theory_practice_ratio")
            domain = query.get("domain")
            pagination = query.get("pagination", {})

            offset = pagination.get("offset", 0)
            limit = pagination.get("limit", 10)

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

            # Process query text
            processed_query = self._process_query_text(query_text)

            # Log search parameters
            logger.info(f"Searching for '{query_text}' with theory/practice ratio: {theory_practice_ratio}, domain: {domain}")

            # Perform hybrid search
            hybrid_results = self._hybrid_search(
                processed_query,
                query_text,
                filters,
                theory_practice_ratio,
                domain
            )

            # Count results by context type
            theoretical_results = sum(1 for r in hybrid_results if r.get("context_type") == "theoretical")
            practical_results = sum(1 for r in hybrid_results if r.get("context_type") == "practical")
            total_results = len(hybrid_results)

            # Apply pagination
            paginated_results = hybrid_results[offset:offset+limit] if hybrid_results else []

            # Record search in history
            search_id = self._record_search_history(query_text, query, total_results, time.time() - start_time)

            # Track popular searches
            self._update_popular_searches(query_text)

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Prepare response
            response = {
                "results": paginated_results,
                "totalResults": total_results,
                "theoreticalResults": theoretical_results,
                "practicalResults": practical_results,
                "executionTimeMs": execution_time_ms,
                "searchId": search_id,
                "query": query
            }

            # Suggest similar searches
            if total_results < 5:
                response["suggestedSearches"] = self._suggest_similar_searches(query_text, domain)

            logger.info(f"Search for '{query_text}' returned {total_results} results in {execution_time_ms}ms")
            return response

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

    def _process_query_text(self, query_text: str) -> Dict[str, Any]:
        """
        Process query text for enhanced search.

        Args:
            query_text: Original query text

        Returns:
            Dictionary with processed query components
        """
        # Normalize text
        normalized_text = self._normalize_text(query_text)

        # Extract exact phrases (text within quotes)
        exact_phrases = re.findall(r'"([^"]+)"', query_text)

        # Remove exact phrases from query text for further processing
        clean_text = query_text
        for phrase in exact_phrases:
            clean_text = clean_text.replace(f'"{phrase}"', '')
        clean_text = clean_text.strip()

        # Tokenize and filter tokens
        tokens = [token.strip() for token in clean_text.split() if token.strip()]
        stopwords = self._get_stopwords()
        tokens = [token for token in tokens if len(token) > 1 and token.lower() not in stopwords]

        # Generate stemmed versions
        stemmed_tokens = [self._stem_text(token) for token in tokens]
        stemmed_phrases = [' '.join([self._stem_text(word) for word in phrase.split()])
                          for phrase in exact_phrases]

        # Generate ngrams for fuzzy matching
        ngrams = []
        for token in tokens:
            if len(token) >= 3:  # Only generate ngrams for tokens of sufficient length
                token_ngrams = self._generate_ngrams(token, 2, 3)
                ngrams.extend(token_ngrams)

        # Get synonyms for query expansion
        synonyms = self._get_query_synonyms(tokens)

        return {
            "original": query_text,
            "normalized": normalized_text,
            "tokens": tokens,
            "stemmed_tokens": stemmed_tokens,
            "exact_phrases": exact_phrases,
            "stemmed_phrases": stemmed_phrases,
            "ngrams": ngrams,
            "synonyms": synonyms
        }

    def _get_query_synonyms(self, tokens: List[str], domain: str = None) -> Dict[str, List[str]]:
        """
        Get synonyms for query terms.

        Args:
            tokens: Query tokens
            domain: Optional domain for context-specific synonyms

        Returns:
            Dictionary mapping tokens to lists of synonyms
        """
        synonyms = {}

        if not tokens:
            return synonyms

        try:
            for token in tokens:
                query = """
                SELECT synonym FROM search_synonyms
                WHERE term = ?
                """
                params = [token.lower()]

                if domain:
                    query += " AND (domain = ? OR domain IS NULL)"
                    params.append(domain)

                results = self.db_manager.execute_query(query, tuple(params))
                if results:
                    synonyms[token] = [r["synonym"] for r in results]

            return synonyms

        except Exception as e:
            logger.error(f"Error getting query synonyms: {e}")
            return {}

    def _hybrid_search(
        self,
        processed_query: Dict[str, Any],
        original_query: str,
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using multiple approaches and combine results.

        Args:
            processed_query: Processed query dictionary
            original_query: Original query text
            filters: Query filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            Combined and ranked search results
        """
        # Search in concepts (concept-based search)
        concept_results = self._search_concepts(processed_query, filters, theory_practice_ratio, domain)

        # Search in segments (full-text search)
        segment_results = self._search_segments(processed_query, filters, theory_practice_ratio, domain)

        # Search in video metadata (titles, descriptions)
        metadata_results = self._search_metadata(processed_query, filters, theory_practice_ratio, domain)

        # Fuzzy search using ngrams
        fuzzy_results = self._fuzzy_search(processed_query, filters, theory_practice_ratio, domain)

        # Combine all results
        all_results = concept_results + segment_results + metadata_results + fuzzy_results

        # Deduplicate results
        deduplicated_results = self._deduplicate_results(all_results)

        # Final ranking of results
        ranked_results = self._rank_results(deduplicated_results, processed_query, original_query, theory_practice_ratio)

        return ranked_results

    def _search_concepts(
        self,
        processed_query: Dict[str, Any],
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Search for concepts matching the query.

        Args:
            processed_query: Processed query dictionary
            filters: Query filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            List of concept search results
        """
        # Build FTS query using stemmed tokens and synonyms
        fts_clauses = []

        # Add original tokens
        for token in processed_query["stemmed_tokens"]:
            fts_clauses.append(f'"{token}"')

        # Add exact phrases
        for phrase in processed_query["stemmed_phrases"]:
            fts_clauses.append(f'"{phrase}"')

        # Add synonyms
        for token, syn_list in processed_query.get("synonyms", {}).items():
            stemmed_token = self._stem_text(token.lower())
            if stemmed_token not in processed_query["stemmed_tokens"]:
                for syn in syn_list:
                    stemmed_syn = self._stem_text(syn.lower())
                    fts_clauses.append(f'"{stemmed_syn}"')

        fts_query = " OR ".join(fts_clauses)

        # Build SQL query
        sql = """
        SELECT c.*, o.video_id, o.segment_id, o.start_time, o.end_time,
            o.context_type, o.context_text, o.relevance_score, o.occurrence_id,
            v.title as video_title
        FROM concepts c
        JOIN concepts_fts fts ON c.concept_id = fts.concept_id
        JOIN occurrences o ON c.concept_id = o.concept_id
        JOIN videos v ON o.video_id = v.video_id
        WHERE concepts_fts MATCH ?
        """
        params = [fts_query]

        # Apply domain filter
        if domain:
            sql += " AND c.domain = ?"
            params.append(domain)

        # Apply video filter
        if "video_id" in filters:
            sql += " AND o.video_id = ?"
            params.append(filters["video_id"])

        # Apply video list filter
        if "video_ids" in filters and filters["video_ids"]:
            placeholders = ", ".join(["?"] * len(filters["video_ids"]))
            sql += f" AND o.video_id IN ({placeholders})"
            params.extend(filters["video_ids"])

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.7:
                # Prioritize theoretical
                sql += " ORDER BY CASE WHEN o.context_type = 'theoretical' THEN 1 ELSE 2 END, o.relevance_score DESC"
            elif theory_practice_ratio < 0.3:
                # Prioritize practical
                sql += " ORDER BY CASE WHEN o.context_type = 'practical' THEN 1 ELSE 2 END, o.relevance_score DESC"
            else:
                # Balance between theoretical and practical
                sql += " ORDER BY o.relevance_score DESC"
        else:
            # Default ordering by relevance
            sql += " ORDER BY o.relevance_score DESC"

        # Set limit to avoid processing too many results
        sql += " LIMIT 100"

        # Execute query
        results = self.db_manager.execute_query(sql, tuple(params))

        # Add result type
        for result in results:
            result["result_type"] = "concept"

        return results

    def _search_segments(
        self,
        processed_query: Dict[str, Any],
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Search for transcript segments matching the query.

        Args:
            processed_query: Processed query dictionary
            filters: Query filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            List of segment search results
        """
        # Build FTS query using stemmed tokens and synonyms
        fts_clauses = []

        # Add original tokens
        for token in processed_query["stemmed_tokens"]:
            fts_clauses.append(f'"{token}"')

        # Add exact phrases
        for phrase in processed_query["stemmed_phrases"]:
            fts_clauses.append(f'"{phrase}"')

        # Add synonyms
        for token, syn_list in processed_query.get("synonyms", {}).items():
            stemmed_token = self._stem_text(token.lower())
            if stemmed_token not in processed_query["stemmed_tokens"]:
                for syn in syn_list:
                    stemmed_syn = self._stem_text(syn.lower())
                    fts_clauses.append(f'"{stemmed_syn}"')

        fts_query = " OR ".join(fts_clauses)

        # Build SQL query
        sql = """
        SELECT s.segment_id, s.video_id, s.text AS context_text,
            s.start_time, s.end_time, s.context_type, s.domain,
            v.title as video_title,
            0.8 as relevance_score, NULL as concept_id, NULL as occurrence_id
        FROM segments s
        JOIN segments_fts fts ON s.segment_id = fts.segment_id
        JOIN videos v ON s.video_id = v.video_id
        WHERE segments_fts MATCH ?
        """
        params = [fts_query]

        # Apply domain filter
        if domain:
            sql += " AND s.domain = ?"
            params.append(domain)

        # Apply video filter
        if "video_id" in filters:
            sql += " AND s.video_id = ?"
            params.append(filters["video_id"])

        # Apply video list filter
        if "video_ids" in filters and filters["video_ids"]:
            placeholders = ", ".join(["?"] * len(filters["video_ids"]))
            sql += f" AND s.video_id IN ({placeholders})"
            params.extend(filters["video_ids"])

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.7:
                # Prioritize theoretical
                sql += " ORDER BY CASE WHEN s.context_type = 'theoretical' THEN 1 ELSE 2 END"
            elif theory_practice_ratio < 0.3:
                # Prioritize practical
                sql += " ORDER BY CASE WHEN s.context_type = 'practical' THEN 1 ELSE 2 END"
            else:
                # Balance with natural ordering (no explicit rank)
                sql += " ORDER BY relevance_score DESC"
        else:
            # Default ranking
            sql += " ORDER BY relevance_score DESC"

        # Set limit to avoid processing too many results
        sql += " LIMIT 100"

        # Execute query
        results = self.db_manager.execute_query(sql, tuple(params))

        # Add result type
        for result in results:
            result["result_type"] = "segment"

        return results

    def _search_metadata(
        self,
        processed_query: Dict[str, Any],
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Search for videos whose metadata (title, description) matches the query.

        Args:
            processed_query: Processed query dictionary
            filters: Query filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            List of metadata search results
        """
        # Build FTS query using stemmed tokens and synonyms
        fts_clauses = []

        # Add original tokens
        for token in processed_query["stemmed_tokens"]:
            fts_clauses.append(f'"{token}"')

        # Add exact phrases
        for phrase in processed_query["stemmed_phrases"]:
            fts_clauses.append(f'"{phrase}"')

        # Add synonyms
        for token, syn_list in processed_query.get("synonyms", {}).items():
            stemmed_token = self._stem_text(token.lower())
            if stemmed_token not in processed_query["stemmed_tokens"]:
                for syn in syn_list:
                    stemmed_syn = self._stem_text(syn.lower())
                    fts_clauses.append(f'"{stemmed_syn}"')

        fts_query = " OR ".join(fts_clauses)

        # Build SQL query
        sql = """
        SELECT v.video_id, v.title as video_title, v.description as context_text,
            NULL as start_time, NULL as end_time, NULL as concept_id,
            NULL as segment_id, NULL as occurrence_id,
            CASE
                WHEN v.theory_practice_ratio > 0.7 THEN 'theoretical'
                WHEN v.theory_practice_ratio < 0.3 THEN 'practical'
                ELSE 'mixed'
            END as context_type,
            v.domain, 0.7 as relevance_score
        FROM videos v
        JOIN video_metadata_fts m ON v.video_id = m.video_id
        WHERE video_metadata_fts MATCH ?
        """
        params = [fts_query]

        # Apply domain filter
        if domain:
            sql += " AND v.domain = ?"
            params.append(domain)

        # Apply video filter
        if "video_id" in filters:
            sql += " AND v.video_id = ?"
            params.append(filters["video_id"])

        # Apply video list filter
        if "video_ids" in filters and filters["video_ids"]:
            placeholders = ", ".join(["?"] * len(filters["video_ids"]))
            sql += f" AND v.video_id IN ({placeholders})"
            params.extend(filters["video_ids"])

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.7:
                sql += " ORDER BY v.theory_practice_ratio DESC"
            elif theory_practice_ratio < 0.3:
                sql += " ORDER BY v.theory_practice_ratio ASC"
            else:
                sql += " ORDER BY relevance_score DESC"
        else:
            sql += " ORDER BY relevance_score DESC"

        # Set limit to avoid processing too many results
        sql += " LIMIT 50"

        # Execute query
        results = self.db_manager.execute_query(sql, tuple(params))

        # Add result type
        for result in results:
            result["result_type"] = "metadata"

        return results

    def _fuzzy_search(
        self,
        processed_query: Dict[str, Any],
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform fuzzy search using ngrams.

        Args:
            processed_query: Processed query dictionary
            filters: Query filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            List of fuzzy search results
        """
        # If no ngrams, return empty results
        if not processed_query["ngrams"]:
            return []

        # Build SQL query for ngram search
        ngram_placeholders = ", ".join(["?"] * len(processed_query["ngrams"]))
        sql = f"""
        SELECT item_id, item_type, COUNT(*) as match_count
        FROM ngram_index
        WHERE ngram IN ({ngram_placeholders})
        GROUP BY item_id, item_type
        """
        params = processed_query["ngrams"]

        # Execute query
        matches = self.db_manager.execute_query(sql, tuple(params))

        # Filter matches with sufficient ngram coverage
        fuzzy_results = []

        # Threshold for fuzzy matching (proportion of ngrams that must match)
        threshold = 0.6

        for match in matches:
            item_id = match["item_id"]
            item_type = match["item_type"]
            match_count = match["match_count"]

            # Calculate match score based on number of matching ngrams
            # Normalize by total ngrams in the query
            match_ratio = match_count / len(processed_query["ngrams"])

            # Only include results that meet the threshold
            if match_ratio >= threshold:
                if item_type == "segment":
                    # Get segment details
                    segment_sql = """
                    SELECT s.segment_id, s.video_id, s.text as context_text,
                        s.start_time, s.end_time, s.context_type, s.domain,
                        v.title as video_title,
                        ? as relevance_score, NULL as concept_id, NULL as occurrence_id
                    FROM segments s
                    JOIN videos v ON s.video_id = v.video_id
                    WHERE s.segment_id = ?
                    """
                    segment_results = self.db_manager.execute_query(segment_sql, (match_ratio * 0.6, item_id))

                    if segment_results:
                        result = segment_results[0]
                        result["result_type"] = "fuzzy_segment"

                        # Apply filters
                        include_result = True

                        if domain and result["domain"] != domain:
                            include_result = False

                        if "video_id" in filters and result["video_id"] != filters["video_id"]:
                            include_result = False

                        if "video_ids" in filters and filters["video_ids"] and result["video_id"] not in filters["video_ids"]:
                            include_result = False

                        if include_result:
                            fuzzy_results.append(result)

                elif item_type == "concept":
                    # Get concept details with occurrence
                    concept_sql = """
                    SELECT c.*, o.video_id, o.segment_id, o.start_time, o.end_time,
                        o.context_type, o.context_text, ? as relevance_score, o.occurrence_id,
                        v.title as video_title
                    FROM concepts c
                    JOIN occurrences o ON c.concept_id = o.concept_id
                    JOIN videos v ON o.video_id = v.video_id
                    WHERE c.concept_id = ?
                    LIMIT 10
                    """
                    concept_results = self.db_manager.execute_query(concept_sql, (match_ratio * 0.6, item_id))

                    for result in concept_results:
                        result["result_type"] = "fuzzy_concept"

                        # Apply filters
                        include_result = True

                        if domain and result["domain"] != domain:
                            include_result = False

                        if "video_id" in filters and result["video_id"] != filters["video_id"]:
                            include_result = False

                        if "video_ids" in filters and filters["video_ids"] and result["video_id"] not in filters["video_ids"]:
                            include_result = False

                        if include_result:
                            fuzzy_results.append(result)

        return fuzzy_results

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate search results.

        Args:
            results: Combined search results

        Returns:
            Deduplicated results
        """
        deduplicated = {}

        for result in results:
            # Use occurrence_id or segment_id as key for deduplication
            if result.get("occurrence_id"):
                key = f"occurrence_{result['occurrence_id']}"
            elif result.get("segment_id"):
                key = f"segment_{result['segment_id']}"
            else:
                # For metadata results, use video_id
                key = f"video_{result.get('video_id', '')}"

            # Only keep result with highest relevance score
            if key not in deduplicated or result.get("relevance_score", 0) > deduplicated[key].get("relevance_score", 0):
                deduplicated[key] = result

        return list(deduplicated.values())

    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        processed_query: Dict[str, Any],
        original_query: str,
        theory_practice_ratio: Optional[float]
    ) -> List[Dict[str, Any]]:
        """
        Perform final ranking of search results.

        Args:
            results: Search results to rank
            processed_query: Processed query information
            original_query: Original query text
            theory_practice_ratio: Theory/practice ratio preference

        Returns:
            Ranked results
        """
        if not results:
            return []

        for result in results:
            # Calculate query term matches
            context_text = result.get("context_text", "").lower()
            title = result.get("video_title", "").lower()

            # Calculate term frequency and term coverage
            query_tokens = processed_query["tokens"]
            matching_tokens = sum(1 for token in query_tokens if token.lower() in context_text)
            token_ratio = matching_tokens / max(1, len(query_tokens))

            # Calculate proximity score
            proximity_score = 0
            if len(query_tokens) > 1 and matching_tokens == len(query_tokens):
                # Check if the original query appears in the text
                if original_query.lower() in context_text:
                    proximity_score = 1.0

            # Calculate exact phrase matches
            phrase_score = 0
            for phrase in processed_query["exact_phrases"]:
                if phrase.lower() in context_text:
                    phrase_score = 1.0
                    break

            # Theory/practice preference adjustment
            theory_practice_score = 0
            if theory_practice_ratio is not None:
                context_type = result.get("context_type", "mixed")
                if theory_practice_ratio > 0.7 and context_type == "theoretical":
                    theory_practice_score = 0.2
                elif theory_practice_ratio < 0.3 and context_type == "practical":
                    theory_practice_score = 0.2

            # Title match boost
            title_score = 0
            for token in query_tokens:
                if token.lower() in title:
                    title_score = 0.15
                    break

            # Result type scoring
            type_score = 0
            result_type = result.get("result_type", "")
            if "concept" in result_type:
                type_score = 0.1  # Boost for concept matches
            elif result_type == "segment":
                type_score = 0.05  # Smaller boost for direct segment matches

            # Combine scores using weights
            base_score = result.get("relevance_score", 0.5)

            # Final score combining all factors
            final_score = (
                base_score * 0.3 +
                token_ratio * 0.2 +
                proximity_score * 0.15 +
                phrase_score * 0.15 +
                theory_practice_score * 0.1 +
                title_score * 0.05 +
                type_score * 0.05
            )

            # Update the result's relevance score
            result["relevance_score"] = final_score

        # Sort by final relevance score
        ranked_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

        return ranked_results

    def _record_search_history(
        self,
        query_text: str,
        query_params: Dict[str, Any],
        result_count: int,
        execution_time: float,
        user_id: str = None,
        client_info: str = None
    ) -> str:
        """
        Record search query in history.

        Args:
            query_text: Original query text
            query_params: Query parameters
            result_count: Number of results returned
            execution_time: Query execution time in seconds
            user_id: Optional user ID
            client_info: Optional client information

        Returns:
            Search ID
        """
        import uuid

        search_id = str(uuid.uuid4())

        try:
            # Serialize query params
            query_params_json = json.dumps(query_params)

            query = """
            INSERT INTO search_history (
                search_id, query_text, query_params, result_count,
                execution_time_ms, timestamp, user_id, client_info
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            self.db_manager.execute_update(query, (
                search_id,
                query_text,
                query_params_json,
                result_count,
                int(execution_time * 1000),
                datetime.now().isoformat(),
                user_id,
                client_info
            ))

            return search_id

        except Exception as e:
            logger.error(f"Error recording search history: {e}")
            return search_id

    def _update_popular_searches(self, query_text: str) -> None:
        """
        Update popular searches.

        Args:
            query_text: Search query text
        """
        if not query_text or len(query_text) < 3:
            return

        try:
            # Normalize query text
            normalized_query = self._normalize_text(query_text)

            # Check if query exists
            existing = self.db_manager.execute_query(
                "SELECT count FROM popular_searches WHERE query_text = ?",
                (normalized_query,)
            )

            current_time = datetime.now().isoformat()

            if existing:
                # Update count
                count = existing[0]["count"] + 1
                self.db_manager.execute_update(
                    "UPDATE popular_searches SET count = ?, last_searched = ? WHERE query_text = ?",
                    (count, current_time, normalized_query)
                )
            else:
                # Insert new entry
                self.db_manager.execute_update(
                    "INSERT INTO popular_searches (query_text, count, last_searched) VALUES (?, ?, ?)",
                    (normalized_query, 1, current_time)
                )

        except Exception as e:
            logger.error(f"Error updating popular searches: {e}")

    def _suggest_similar_searches(self, query_text: str, domain: str = None, limit: int = 5) -> List[str]:
        """
        Suggest similar searches.

        Args:
            query_text: Original query text
            domain: Optional domain context
            limit: Maximum number of suggestions to return

        Returns:
            List of suggested search queries
        """
        if not query_text or len(query_text) < 3:
            return []

        try:
            # Get word-level tokens from query
            tokens = self._extract_significant_tokens(query_text)

            if not tokens:
                return []

            # Generate partial query for SQL LIKE
            partial_query = f"%{tokens[0]}%"

            # Query popular searches
            query = """
            SELECT query_text, count FROM popular_searches
            WHERE query_text LIKE ?
            ORDER BY count DESC
            LIMIT ?
            """

            popular = self.db_manager.execute_query(query, (partial_query, limit))

            # Get suggestions
            suggestions = [p["query_text"] for p in popular
                          if p["query_text"].lower() != query_text.lower()]

            # If we don't have enough suggestions from popular searches,
            # add some from synonyms and related concepts
            if len(suggestions) < limit and tokens:
                # Get related terms from the first token
                related_query = """
                SELECT DISTINCT term
                FROM search_synonyms
                WHERE synonym = ?
                """
                params = [tokens[0].lower()]

                if domain:
                    related_query += " AND (domain = ? OR domain IS NULL)"
                    params.append(domain)

                related = self.db_manager.execute_query(related_query, tuple(params))

                for r in related:
                    term = r["term"]
                    # Replace the first token with the related term
                    suggestion = query_text.replace(tokens[0], term, 1)
                    if suggestion.lower() != query_text.lower() and suggestion not in suggestions:
                        suggestions.append(suggestion)

                        if len(suggestions) >= limit:
                            break

            return suggestions[:limit]

        except Exception as e:
            logger.error(f"Error suggesting similar searches: {e}")
            return []

    def record_search_feedback(
        self,
        search_id: str,
        item_id: str,
        item_type: str,
        relevance_score: int = None,
        clicked: bool = None,
        time_spent_seconds: int = None,
        feedback_text: str = None,
        user_id: str = None
    ) -> bool:
        """
        Record feedback for a search result.

        Args:
            search_id: Search ID
            item_id: Item ID (concept_id, segment_id, video_id)
            item_type: Item type ('concept', 'segment', 'video')
            relevance_score: Relevance score (1-5)
            clicked: Whether the result was clicked
            time_spent_seconds: Time spent viewing the result
            feedback_text: Optional feedback text
            user_id: Optional user ID

        Returns:
            True if successful, False otherwise
        """
        if not search_id or not item_id or not item_type:
            return False

        try:
            import uuid

            feedback_id = str(uuid.uuid4())
            current_time = datetime.now().isoformat()

            query = """
            INSERT INTO search_feedback (
                feedback_id, search_id, item_id, item_type,
                relevance_score, clicked, time_spent_seconds,
                feedback_text, timestamp, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            self.db_manager.execute_update(query, (
                feedback_id,
                search_id,
                item_id,
                item_type,
                relevance_score,
                1 if clicked else 0 if clicked is not None else None,
                time_spent_seconds,
                feedback_text,
                current_time,
                user_id
            ))

            return True

        except Exception as e:
            logger.error(f"Error recording search feedback: {e}")
            return False

    def add_search_synonym(
        self,
        term: str,
        synonym: str,
        domain: str = None,
        source: str = 'manual'
    ) -> bool:
        """
        Add a synonym for query expansion.

        Args:
            term: Original term
            synonym: Synonym for the term
            domain: Optional domain context
            source: Source of the synonym ('manual', 'automatic')

        Returns:
            True if successful, False otherwise
        """
        if not term or not synonym:
            return False

        try:
            # Normalize terms
            term = term.strip().lower()
            synonym = synonym.strip().lower()

            query = """
            INSERT OR REPLACE INTO search_synonyms (
                term, synonym, domain, source
            ) VALUES (?, ?, ?, ?)
            """

            self.db_manager.execute_update(query, (
                term,
                synonym,
                domain,
                source
            ))

            # Also add reverse mapping
            self.db_manager.execute_update(query, (
                synonym,
                term,
                domain,
                source
            ))

            return True

        except Exception as e:
            logger.error(f"Error adding search synonym: {e}")
            return False

    def get_popular_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get popular searches.

        Args:
            limit: Maximum number of searches to return

        Returns:
            List of popular search dictionaries
        """
        query = """
        SELECT query_text, count, last_searched
        FROM popular_searches
        ORDER BY count DESC
        LIMIT ?
        """

        return self.db_manager.execute_query(query, (limit,))

    def get_search_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get search statistics.

        Args:
            days: Number of days to include in statistics

        Returns:
            Dictionary of search statistics
        """
        try:
            stats = {}

            # Calculate date threshold
            from datetime import datetime, timedelta
            threshold = (datetime.now() - timedelta(days=days)).isoformat()

            # Total searches
            query = """
            SELECT COUNT(*) as count FROM search_history
            WHERE timestamp >= ?
            """
            results = self.db_manager.execute_query(query, (threshold,))
            stats["total_searches"] = results[0]["count"] if results else 0

            # Searches by result count
            query = """
            SELECT
                CASE
                    WHEN result_count = 0 THEN 'no_results'
                    WHEN result_count BETWEEN 1 AND 5 THEN 'few_results'
                    WHEN result_count BETWEEN 6 AND 20 THEN 'moderate_results'
                    ELSE 'many_results'
                END as result_category,
                COUNT(*) as count
            FROM search_history
            WHERE timestamp >= ?
            GROUP BY result_category
            """
            stats["searches_by_result_count"] = self.db_manager.execute_query(query, (threshold,))

            # Average execution time
            query = """
            SELECT AVG(execution_time_ms) as avg_time FROM search_history
            WHERE timestamp >= ?
            """
            results = self.db_manager.execute_query(query, (threshold,))
            stats["avg_execution_time_ms"] = results[0]["avg_time"] if results else 0

            # Top searches
            query = """
            SELECT query_text, COUNT(*) as count
            FROM search_history
            WHERE timestamp >= ?
            GROUP BY query_text
            ORDER BY count DESC
            LIMIT 10
            """
            stats["top_searches"] = self.db_manager.execute_query(query, (threshold,))

            # Feedback statistics
            query = """
            SELECT
                AVG(relevance_score) as avg_relevance,
                SUM(CASE WHEN clicked = 1 THEN 1 ELSE 0 END) as total_clicks,
                COUNT(*) as total_feedback
            FROM search_feedback sf
            JOIN search_history sh ON sf.search_id = sh.search_id
            WHERE sh.timestamp >= ?
            """
            results = self.db_manager.execute_query(query, (threshold,))
            if results:
                stats["avg_relevance"] = results[0]["avg_relevance"]
                stats["total_clicks"] = results[0]["total_clicks"]
                stats["total_feedback"] = results[0]["total_feedback"]
                stats["click_through_rate"] = (results[0]["total_clicks"] / results[0]["total_feedback"]
                                            if results[0]["total_feedback"] > 0 else 0)

            return stats

        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {"error": str(e)}

    def rebuild_search_indexes(self) -> bool:
        """
        Rebuild all search indexes.

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db_manager.transaction() as cursor:
                # Drop and recreate search tables
                cursor.executescript("""
                -- Clear segment FTS
                DELETE FROM segments_fts;

                -- Clear video metadata FTS
                DELETE FROM video_metadata_fts;

                -- Clear ngram index
                DELETE FROM ngram_index;

                -- Rebuild segment FTS
                INSERT INTO segments_fts (segment_id, video_id, text, text_stemmed, domain, context_type)
                SELECT segment_id, video_id, text, text_stemmed, domain, context_type
                FROM segments;

                -- Rebuild video metadata FTS
                INSERT INTO video_metadata_fts (video_id, title, title_stemmed, description, description_stemmed, channel, domain)
                SELECT
                    video_id,
                    title,
                    '', -- title_stemmed will be populated separately
                    description,
                    '', -- description_stemmed will be populated separately
                    channel,
                    domain
                FROM videos;
                """)

                # Rebuild ngram index (this will be done in a separate step
                # since it requires more complex processing)

            # Process each video for n-grams (more processor-intensive)
            videos = self.db_manager.execute_query("SELECT video_id, title, description FROM videos")

            for video in videos:
                # Index title and description n-grams
                self._index_ngrams(
                    video["video_id"],
                    "video",
                    f"{video['title']} {video['description']}"
                )

                # Index segments n-grams
                segments = self.db_manager.execute_query(
                    "SELECT segment_id, text FROM segments WHERE video_id = ?",
                    (video["video_id"],)
                )

                for segment in segments:
                    self._index_ngrams(segment["segment_id"], "segment", segment["text"])

            logger.info("Successfully rebuilt search indexes")
            return True

        except Exception as e:
            logger.error(f"Error rebuilding search indexes: {e}")
            return False

    def optimize_search_indexes(self) -> bool:
        """
        Optimize search indexes for better performance.

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db_manager.transaction() as cursor:
                # Optimize FTS indexes
                cursor.executescript("""
                -- Optimize segments FTS
                INSERT INTO segments_fts(segments_fts) VALUES('optimize');

                -- Optimize concepts FTS
                INSERT INTO concepts_fts(concepts_fts) VALUES('optimize');

                -- Optimize video metadata FTS
                INSERT INTO video_metadata_fts(video_metadata_fts) VALUES('optimize');
                """)

                # Run analyze on other tables
                cursor.execute("ANALYZE ngram_index")
                cursor.execute("ANALYZE search_history")
                cursor.execute("ANALYZE search_feedback")
                cursor.execute("ANALYZE popular_searches")
                cursor.execute("ANALYZE search_synonyms")

            logger.info("Successfully optimized search indexes")
            return True

        except Exception as e:
            logger.error(f"Error optimizing search indexes: {e}")
            return False

    def purge_search_history(self, days_to_keep: int = 90) -> bool:
        """
        Purge old search history records.

        Args:
            days_to_keep: Number of days of history to keep

        Returns:
            True if successful, False otherwise
        """
        try:
            from datetime import datetime, timedelta
            threshold = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

            with self.db_manager.transaction() as cursor:
                # Get search IDs to delete
                cursor.execute("""
                SELECT search_id FROM search_history
                WHERE timestamp < ?
                """, (threshold,))

                search_ids = [row[0] for row in cursor.fetchall()]

                if not search_ids:
                    logger.info(f"No search history records older than {days_to_keep} days")
                    return True

                # Delete feedback first (foreign key constraint)
                if search_ids:
                    # Delete in batches to avoid parameter limit
                    batch_size = 100
                    for i in range(0, len(search_ids), batch_size):
                        batch = search_ids[i:i+batch_size]
                        placeholders = ','.join(['?'] * len(batch))
                        cursor.execute(f"""
                        DELETE FROM search_feedback
                        WHERE search_id IN ({placeholders})
                        """, batch)

                # Delete search history
                cursor.execute("""
                DELETE FROM search_history
                WHERE timestamp < ?
                """, (threshold,))

                deleted_count = cursor.rowcount
                logger.info(f"Purged {deleted_count} search history records older than {days_to_keep} days")

            return True

        except Exception as e:
            logger.error(f"Error purging search history: {e}")
            return False
