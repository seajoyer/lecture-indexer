"""
Enhanced Search Engine module for the Lecture Video Content Indexer.
Handles search queries and retrieval with advanced text processing and relevance scoring.
"""

import os
import json
import logging
import time
import uuid
import re
import hashlib
import sqlite3
from pathlib import Path
import threading
from typing import Dict, List, Any, Optional, Tuple, Set

# Configure logging
logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Enhanced search engine for the Lecture Video Content Indexer.
    Supports theory/practice filtering, domain-specific search,
    and advanced text analysis for improved search relevance.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Search Engine with configuration.

        Args:
            config: Configuration dictionary
        """
        logger.info("Initializing Enhanced Search Engine")

        self.config = config
        self.index_dir = config.get("index_dir", "data/index")
        self.use_stemming = config.get("use_stemming", True)
        self.use_fuzzy_matching = config.get("use_fuzzy_matching", True)
        self.max_fuzzy_distance = config.get("max_fuzzy_distance", 2)  # Levenshtein distance threshold
        self.min_relevance_score = config.get("min_relevance_score", 0.1)
        self.result_context_length = config.get("result_context_length", 150)
        self.enable_query_expansion = config.get("enable_query_expansion", True)
        self.max_expanded_terms = config.get("max_expanded_terms", 3)
        self.max_cache_entries = config.get("max_cache_entries", 100)

        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        # Initialize SQLite database for indexing
        self.db_path = Path(self.index_dir) / "index.db"
        self._init_database()

        # Thread lock for database operations
        self.db_lock = threading.Lock()

        # Initialize stemmer if enabled
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                logger.info("NLTK PorterStemmer initialized for text normalization")
            except ImportError:
                logger.warning("NLTK not available, falling back to simple normalization")
                self.stemmer = None

        # Cache for frequently accessed data
        self.cache = {
            "synonyms": {},
            "related_concepts": {},
            "stemmed_terms": {},
            "recent_searches": []
        }

        logger.info("Enhanced Search Engine initialized")

    def _init_database(self):
        """Initialize SQLite database for indexing with enhanced schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create videos table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                channel TEXT,
                publication_date TEXT,
                duration_seconds INTEGER,
                language TEXT,
                domain TEXT,
                domain_confidence REAL,
                theory_practice_ratio REAL,
                theoretical_segments INTEGER,
                practical_segments INTEGER,
                indexed_at TEXT
            )
            ''')

            # Create concepts table with enhanced fields
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id TEXT PRIMARY KEY,
                text TEXT,
                normalized_text TEXT,
                stemmed_text TEXT,
                domain TEXT,
                concept_class TEXT,  -- theoretical, practical, both
                total_occurrences INTEGER,
                theoretical_occurrences INTEGER,
                practical_occurrences INTEGER,
                relevance_score REAL, -- Global relevance score based on occurrences and context
                indexed_at TEXT
            )
            ''')

            # Create occurrences table with enhanced context
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS occurrences (
                occurrence_id TEXT PRIMARY KEY,
                concept_id TEXT,
                video_id TEXT,
                segment_id TEXT,
                start_time REAL,
                end_time REAL,
                context_type TEXT,  -- theoretical, practical, mixed
                context_text TEXT,
                context_before TEXT, -- Text before the concept mention
                context_after TEXT,  -- Text after the concept mention
                relevance_score REAL,
                FOREIGN KEY (concept_id) REFERENCES concepts (concept_id),
                FOREIGN KEY (video_id) REFERENCES videos (video_id                    )
            )

    def _index_ngrams(
        self,
        cursor: sqlite3.Cursor,
        video_id: str,
        segments: List[Dict[str, Any]]
    ):
        """
        Index n-grams from text for better phrase matching.

        Args:
            cursor: Database cursor
            video_id: YouTube video ID
            segments: List of transcript segments
        """
        # Clear previous ngrams for this video
        cursor.execute("DELETE FROM ngrams WHERE video_id = ?", (video_id,))

        # Combine all segment texts
        full_text = " ".join([segment.get("text", "") for segment in segments])

        # Extract n-grams (2-4 words)
        ngrams = {}

        # Normalize text
        text = full_text.lower()

        # Extract word tokens
        words = re.findall(r'\b[a-z0-9][a-z0-9\-_\']*', text)

        # Generate n-grams
        for n in range(2, 5):  # 2, 3, and 4-grams
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])

                # Skip very short ngrams or those with very short words
                if len(ngram) < 5 or min(len(w) for w in words[i:i+n]) < 3:
                    continue

                # Count frequency
                ngrams[ngram] = ngrams.get(ngram, 0) + 1

        # Generate stemmed forms of ngrams
        for ngram, frequency in ngrams.items():
            # Only index frequent ngrams to keep the index size manageable
            if frequency < 2:
                continue

            # Generate a stemmed version
            if self.stemmer:
                stemmed_words = [self.stemmer.stem(w) for w in ngram.split()]
                stemmed_ngram = " ".join(stemmed_words)
            else:
                stemmed_words = [self._simple_stem(w) for w in ngram.split()]
                stemmed_ngram = " ".join(stemmed_words)

            # Check if this ngram is a known concept
            is_concept = 0
            cursor.execute(
                "SELECT 1 FROM concepts WHERE normalized_text = ? LIMIT 1",
                (ngram,)
            )
            if cursor.fetchone():
                is_concept = 1

            # Generate ID
            ngram_id = hashlib.md5(f"{video_id}:{ngram}".encode()).hexdigest()

            # Find segment that contains this ngram
            segment_id = None
            for segment in segments:
                if ngram in segment.get("text", "").lower():
                    segment_id = segment.get("id")
                    break

            # Insert ngram
            cursor.execute(
                """
                INSERT INTO ngrams (
                    ngram_id, ngram, stemmed_ngram, segment_id, video_id, frequency, is_concept
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ngram_id, ngram, stemmed_ngram, segment_id, video_id, frequency, is_concept)
            )

    def _index_concept_relationships(
        self,
        cursor: sqlite3.Cursor,
        video_id: str,
        key_concepts: List[Dict[str, Any]]
    ):
        """
        Extract and index relationships between concepts.

        Args:
            cursor: Database cursor
            video_id: YouTube video ID
            key_concepts: List of key concepts
        """
        # Get concept IDs for all concepts in this video
        concept_ids = []
        for concept in key_concepts:
            concept_text = concept.get("text", "")
            normalized_text = concept_text.lower().strip()
            domain = concept.get("domain", "unknown")

            if not concept_text or len(concept_text) < 2:
                continue

            # Generate concept ID
            concept_id = hashlib.md5(f"{normalized_text}:{domain}".encode()).hexdigest()
            concept_ids.append(concept_id)

        if not concept_ids:
            return

        # Get segments where each concept appears
        concept_segments = {}
        for concept_id in concept_ids:
            cursor.execute(
                """
                SELECT segment_id, start_time, end_time
                FROM occurrences
                WHERE concept_id = ? AND video_id = ?
                ORDER BY start_time
                """,
                (concept_id, video_id)
            )

            rows = cursor.fetchall()
            concept_segments[concept_id] = [(row[0], row[1], row[2]) for row in rows]

        # Find concept pairs that appear in sequence or close together
        for source_id in concept_ids:
            for target_id in concept_ids:
                if source_id == target_id:
                    continue

                source_segments = concept_segments.get(source_id, [])
                target_segments = concept_segments.get(target_id, [])

                if not source_segments or not target_segments:
                    continue

                # Relationship types:
                # - prerequisite: source appears consistently before target
                # - related: source and target appear together

                # Count how many times source appears before target
                prereq_count = 0
                for source_seg in source_segments:
                    for target_seg in target_segments:
                        if source_seg[1] < target_seg[1]:  # source starts before target
                            prereq_count += 1

                # Count how many times they appear in the same segment
                related_count = 0
                source_segment_ids = {seg[0] for seg in source_segments}
                target_segment_ids = {seg[0] for seg in target_segments}
                common_segments = source_segment_ids.intersection(target_segment_ids)
                related_count = len(common_segments)

                # Create relationships if strong evidence
                if prereq_count > 0:
                    # Create prerequisite relationship (source -> target)
                    relationship_id = hashlib.md5(f"prereq:{source_id}:{target_id}".encode()).hexdigest()
                    strength = min(1.0, prereq_count / 5.0)  # Cap at 1.0, reaches max at 5 occurrences

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO concept_relationships (
                            relationship_id, source_concept_id, target_concept_id,
                            relationship_type, strength
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (relationship_id, source_id, target_id, "prerequisite", strength)
                    )

                if related_count > 0:
                    # Create related relationship (bidirectional)
                    rel_strength = min(1.0, related_count / 3.0)  # Cap at 1.0, reaches max at 3 occurrences

                    # source -> target
                    relationship_id = hashlib.md5(f"related:{source_id}:{target_id}".encode()).hexdigest()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO concept_relationships (
                            relationship_id, source_concept_id, target_concept_id,
                            relationship_type, strength
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (relationship_id, source_id, target_id, "related", rel_strength)
                    )

                    # target -> source
                    relationship_id = hashlib.md5(f"related:{target_id}:{source_id}".encode()).hexdigest()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO concept_relationships (
                            relationship_id, source_concept_id, target_concept_id,
                            relationship_type, strength
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (relationship_id, target_id, source_id, "related", rel_strength)
                    )

        # Ensure all theoretical concepts have connections to practical concepts and vice versa
        # to improve the learning path generation
        cursor.execute(
            """
            SELECT c1.concept_id, c1.text, c1.concept_class
            FROM concepts c1
            JOIN occurrences o ON c1.concept_id = o.concept_id
            WHERE o.video_id = ?
            GROUP BY c1.concept_id
            """,
            (video_id,)
        )

        all_concepts = cursor.fetchall()
        theoretical_concepts = [c for c in all_concepts if c[2] == "theoretical"]
        practical_concepts = [c for c in all_concepts if c[2] == "practical"]

        # Connect theoretical concepts to practical concepts
        for theoretical in theoretical_concepts:
            for practical in practical_concepts:
                # Create application relationship (theoretical -> practical)
                relationship_id = hashlib.md5(f"application:{theoretical[0]}:{practical[0]}".encode()).hexdigest()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO concept_relationships (
                        relationship_id, source_concept_id, target_concept_id,
                        relationship_type, strength
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (relationship_id, theoretical[0], practical[0], "application", 0.5)
                )

                # Create foundation relationship (practical -> theoretical)
                relationship_id = hashlib.md5(f"foundation:{practical[0]}:{theoretical[0]}".encode()).hexdigest()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO concept_relationships (
                        relationship_id, source_concept_id, target_concept_id,
                        relationship_type, strength
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (relationship_id, practical[0], theoretical[0], "foundation", 0.5)
                )

    def _index_domain_synonyms(
        self,
        cursor: sqlite3.Cursor,
        key_concepts: List[Dict[str, Any]],
        domain: Optional[str] = None
    ):
        """
        Generate and index domain-specific synonyms for concepts.

        Args:
            cursor: Database cursor
            key_concepts: List of key concepts
            domain: Domain of the content
        """
        # Domain-specific synonym patterns (prefixes, suffixes, etc.)
        domain_patterns = {
            "mathematics": {
                "prefixes": ["mathematical ", "math "],
                "suffixes": [" theorem", " formula", " equation", " identity", " function"],
                "replacements": [
                    ("calculate", "compute"),
                    ("formula", "equation"),
                    ("integration", "integral"),
                    ("differentiation", "derivative")
                ]
            },
            "programming": {
                "prefixes": ["programming ", "code "],
                "suffixes": [" method", " function", " algorithm", " class", " library"],
                "replacements": [
                    ("function", "method"),
                    ("class", "object"),
                    ("algorithm", "procedure"),
                    ("variable", "parameter")
                ]
            },
            "physics": {
                "prefixes": ["physics ", "physical "],
                "suffixes": [" law", " principle", " effect", " theory", " model"],
                "replacements": [
                    ("energy", "power"),
                    ("velocity", "speed"),
                    ("acceleration", "force"),
                    ("wave", "oscillation")
                ]
            }
        }

        # Process each concept
        for concept in key_concepts:
            concept_text = concept.get("text", "")

            if not concept_text or len(concept_text) < 3:
                continue

            synonyms = []

            # Add domain-specific synonyms if domain specified
            if domain and domain in domain_patterns:
                patterns = domain_patterns[domain]

                # Add prefix synonyms
                for prefix in patterns.get("prefixes", []):
                    if not concept_text.startswith(prefix):
                        synonyms.append((prefix + concept_text, 0.8))

                # Add suffix synonyms
                for suffix in patterns.get("suffixes", []):
                    if not concept_text.endswith(suffix):
                        synonyms.append((concept_text + suffix, 0.7))

                # Add replacement synonyms
                for old, new in patterns.get("replacements", []):
                    if old in concept_text:
                        synonyms.append((concept_text.replace(old, new), 0.9))

            # Store synonyms
            for synonym, confidence in synonyms:
                # Skip if synonym is too similar to original
                if synonym.lower() == concept_text.lower():
                    continue

                # Generate synonym ID
                synonym_id = hashlib.md5(f"{concept_text}:{synonym}".encode()).hexdigest()

                # Store in database
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO synonyms (
                        synonym_id, term, synonym, domain, confidence
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (synonym_id, concept_text, synonym, domain, confidence)
                )

    def _order_concepts_by_dependencies(
        self,
        cursor: sqlite3.Cursor,
        concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Order concepts based on prerequisite relationships for optimal learning sequence.

        Args:
            cursor: Database cursor
            concepts: List of concept dictionaries

        Returns:
            Ordered list of concepts
        """
        if not concepts:
            return []

        # Get all concept IDs
        concept_ids = [c.get("concept_id") for c in concepts if c.get("concept_id")]

        if not concept_ids:
            return concepts

        # Create a dependency graph
        graph = {}
        for concept_id in concept_ids:
            graph[concept_id] = []

        # Find prerequisite relationships
        placeholders = ",".join(["?"] * len(concept_ids))
        cursor.execute(
            f"""
            SELECT source_concept_id, target_concept_id
            FROM concept_relationships
            WHERE relationship_type = 'prerequisite'
              AND source_concept_id IN ({placeholders})
              AND target_concept_id IN ({placeholders})
            """,
            concept_ids + concept_ids
        )

        relationships = cursor.fetchall()

        # Build graph of dependencies
        for source, target in relationships:
            if source in graph and target not in graph[source]:
                graph[source].append(target)

        # Use topological sort to order concepts
        ordered_ids = self._topological_sort(graph)

        # Map back to full concept dictionaries
        concept_map = {c.get("concept_id"): c for c in concepts if c.get("concept_id")}
        ordered_concepts = []

        # Add concepts in topological order
        for concept_id in ordered_ids:
            if concept_id in concept_map:
                ordered_concepts.append(concept_map[concept_id])

        # Add any remaining concepts (those without relationships)
        for concept in concepts:
            concept_id = concept.get("concept_id")
            if concept_id and concept_id not in ordered_ids and concept not in ordered_concepts:
                ordered_concepts.append(concept)

        return ordered_concepts

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        Perform topological sort on a dependency graph.

        Args:
            graph: Dependency graph as adjacency list

        Returns:
            Topologically sorted list of nodes
        """
        # Initialize data structures
        visited = set()
        temp_marks = set()
        ordered = []

        def visit(node):
            # Skip if already processed
            if node in visited:
                return

            # Check for cycles (would indicate a node that depends on itself)
            if node in temp_marks:
                return

            # Mark node as being processed
            temp_marks.add(node)

            # Process dependencies first
            for dependency in graph.get(node, []):
                visit(dependency)

            # Mark node as processed and add to result
            temp_marks.remove(node)
            visited.add(node)
            ordered.append(node)

        # Process all nodes
        for node in graph:
            if node not in visited:
                visit(node)

        # We need the reverse order (prerequisites first)
        return list(reversed(ordered))

    def _create_optimal_learning_path(
        self,
        theoretical_concepts: List[Dict[str, Any]],
        practical_concepts: List[Dict[str, Any]],
        theory_practice_ratio: float
    ) -> List[Dict[str, Any]]:
        """
        Create an optimal learning path by interleaving theoretical and practical concepts.

        Args:
            theoretical_concepts: List of theoretical concepts in dependency order
            practical_concepts: List of practical concepts in dependency order
            theory_practice_ratio: Desired ratio of theoretical to practical content

        Returns:
            Optimally ordered list of concepts
        """
        if not theoretical_concepts and not practical_concepts:
            return []

        if not theoretical_concepts:
            return practical_concepts

        if not practical_concepts:
            return theoretical_concepts

        # Determine the optimal pattern of interleaving based on the ratio
        optimal_path = []

        # If high theoretical ratio, group multiple theoretical concepts together
        if theory_practice_ratio > 0.7:
            # Theory-heavy pattern: 3 theoretical concepts, then 1 practical
            t_index = 0
            p_index = 0

            while t_index < len(theoretical_concepts) or p_index < len(practical_concepts):
                # Add up to 3 theoretical concepts
                for _ in range(3):
                    if t_index < len(theoretical_concepts):
                        optimal_path.append(theoretical_concepts[t_index])
                        t_index += 1

                # Add 1 practical concept
                if p_index < len(practical_concepts):
                    optimal_path.append(practical_concepts[p_index])
                    p_index += 1

        # If high practical ratio, group multiple practical concepts together
        elif theory_practice_ratio < 0.3:
            # Practice-heavy pattern: 1 theoretical concept, then 3 practical
            t_index = 0
            p_index = 0

            while t_index < len(theoretical_concepts) or p_index < len(practical_concepts):
                # Add 1 theoretical concept
                if t_index < len(theoretical_concepts):
                    optimal_path.append(theoretical_concepts[t_index])
                    t_index += 1

                # Add up to 3 practical concepts
                for _ in range(3):
                    if p_index < len(practical_concepts):
                        optimal_path.append(practical_concepts[p_index])
                        p_index += 1

        # Otherwise use a balanced pattern
        else:
            # Calculate the approximate number of groups needed
            total_concepts = len(theoretical_concepts) + len(practical_concepts)
            theory_count = int(total_concepts * theory_practice_ratio)
            practice_count = total_concepts - theory_count

            # Use a pattern that approximates the desired ratio
            theory_per_group = max(1, min(3, int(theory_count / practice_count) + 1))
            practice_per_group = max(1, min(3, int(practice_count / theory_count) + 1))

            t_index = 0
            p_index = 0

            while t_index < len(theoretical_concepts) or p_index < len(practical_concepts):
                # Add theoretical concepts
                for _ in range(theory_per_group):
                    if t_index < len(theoretical_concepts):
                        optimal_path.append(theoretical_concepts[t_index])
                        t_index += 1

                # Add practical concepts
                for _ in range(practice_per_group):
                    if p_index < len(practical_concepts):
                        optimal_path.append(practical_concepts[p_index])
                        p_index += 1

        return optimal_path

    def _get_recommended_videos_for_path(
        self,
        cursor: sqlite3.Cursor,
        concept_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find the best videos that cover a learning path's concepts.

        Args:
            cursor: Database cursor
            concept_ids: List of concept IDs in the learning path

        Returns:
            List of recommended video dictionaries
        """
        if not concept_ids:
            return []

        # Find videos that contain the most concepts in the learning path
        placeholders = ",".join(["?"] * len(concept_ids))
        cursor.execute(
            f"""
            SELECT
                v.*,
                COUNT(DISTINCT o.concept_id) as concept_coverage,
                COUNT(DISTINCT o.concept_id) * 1.0 / ? as coverage_ratio
            FROM videos v
            JOIN occurrences o ON v.video_id = o.video_id
            WHERE o.concept_id IN ({placeholders})
            GROUP BY v.video_id
            ORDER BY concept_coverage DESC, v.theory_practice_ratio DESC
            LIMIT 10
            """,
            [len(concept_ids)] + concept_ids
        )

        rows = cursor.fetchall()

        # Convert to dictionaries and add metadata
        videos = []
        for row in rows:
            video_dict = dict(row)

            # Get concepts covered by this video
            video_id = video_dict["video_id"]
            cursor.execute(
                f"""
                SELECT c.text
                FROM concepts c
                JOIN occurrences o ON c.concept_id = o.concept_id
                WHERE o.video_id = ? AND c.concept_id IN ({placeholders})
                GROUP BY c.concept_id
                """,
                [video_id] + concept_ids
            )

            covered_concepts = [row[0] for row in cursor.fetchall()]
            video_dict["covered_concepts"] = covered_concepts

            videos.append(video_dict)

        return videos
            ''')

            # Create theory_practice_patterns table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS theory_practice_patterns (
                pattern_id TEXT PRIMARY KEY,
                video_id TEXT,
                pattern_type TEXT,  -- theory_to_practice, practice_to_theory
                pattern_subtype TEXT,  -- domain-specific pattern type
                start_segment_id TEXT,
                end_segment_id TEXT,
                start_time REAL,
                end_time REAL,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
            ''')

            # Create FTS (Full-Text Search) virtual table for concepts with enhanced tokenizing
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
                concept_id,
                text,
                normalized_text,
                stemmed_text,
                domain,
                content='concepts',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )
            ''')

            # Create FTS (Full-Text Search) virtual table for segments with enhanced tokenizing
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                segment_id,
                video_id,
                text,
                stemmed_text,
                domain,
                context_type,
                tokenize='porter unicode61'
            )
            ''')

            # Create a segments table that includes timestamps and stemmed text
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                video_id TEXT,
                start_time REAL,
                end_time REAL,
                text TEXT,
                stemmed_text TEXT,
                domain TEXT,
                context_type TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
            ''')

            # Create concept_relationships table for storing relationships between concepts
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_concept_id TEXT,
                target_concept_id TEXT,
                relationship_type TEXT,
                strength REAL,
                FOREIGN KEY (source_concept_id) REFERENCES concepts (concept_id),
                FOREIGN KEY (target_concept_id) REFERENCES concepts (concept_id)
            )
            ''')

            # Create ngrams table for better phrase matching
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ngrams (
                ngram_id TEXT PRIMARY KEY,
                ngram TEXT,
                stemmed_ngram TEXT,
                segment_id TEXT,
                video_id TEXT,
                frequency INTEGER,
                is_concept BOOLEAN,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
            ''')

            # Create synonym table for query expansion
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonyms (
                synonym_id TEXT PRIMARY KEY,
                term TEXT,
                synonym TEXT,
                domain TEXT,
                confidence REAL
            )
            ''')

            # Create search_cache table for frequently used searches
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT,
                result_json TEXT,
                timestamp TEXT,
                count INTEGER
            )
            ''')

            # Create necessary indices for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_occurrences_concept_id ON occurrences(concept_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_occurrences_video_id ON occurrences(video_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_occurrences_context_type ON occurrences(context_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts_class ON concepts(concept_class)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts_stemmed ON concepts(stemmed_text)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_stemmed ON segments(stemmed_text)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ngrams_ngram ON ngrams(ngram)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ngrams_stemmed ON ngrams(stemmed_ngram)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_synonyms_term ON synonyms(term)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON concept_relationships(source_concept_id)')

            # Create a trigger to automatically update the FTS table for concepts
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS concepts_ai AFTER INSERT ON concepts BEGIN
                INSERT INTO concepts_fts(concept_id, text, normalized_text, stemmed_text, domain)
                VALUES (new.concept_id, new.text, new.normalized_text, new.stemmed_text, new.domain);
            END;
            ''')

            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS concepts_ad AFTER DELETE ON concepts BEGIN
                DELETE FROM concepts_fts WHERE concept_id = old.concept_id;
            END;
            ''')

            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS concepts_au AFTER UPDATE ON concepts BEGIN
                DELETE FROM concepts_fts WHERE concept_id = old.concept_id;
                INSERT INTO concepts_fts(concept_id, text, normalized_text, stemmed_text, domain)
                VALUES (new.concept_id, new.text, new.normalized_text, new.stemmed_text, new.domain);
            END;
            ''')

            # Create a trigger to automatically update the FTS table for segments
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS segments_fts_ai AFTER INSERT ON segments BEGIN
                INSERT INTO segments_fts(segment_id, video_id, text, stemmed_text, domain, context_type)
                VALUES (new.segment_id, new.video_id, new.text, new.stemmed_text, new.domain, new.context_type);
            END;
            ''')

            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS segments_fts_ad AFTER DELETE ON segments BEGIN
                DELETE FROM segments_fts WHERE segment_id = old.segment_id;
            END;
            ''')

            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS segments_fts_au AFTER UPDATE ON segments BEGIN
                DELETE FROM segments_fts WHERE segment_id = old.segment_id;
                INSERT INTO segments_fts(segment_id, video_id, text, stemmed_text, domain, context_type)
                VALUES (new.segment_id, new.video_id, new.text, new.stemmed_text, new.domain, new.context_type);
            END;
            ''')

            conn.commit()
            conn.close()

            logger.info("Enhanced database schema initialized")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def index_content(self, processed_result: Dict[str, Any]) -> bool:
        """
        Index processed content with enhanced analysis and relationship extraction.

        Args:
            processed_result: Processing result dictionary

        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            # Extract key fields
            video_id = processed_result.get("video_id")
            metadata = processed_result.get("metadata", {})
            transcript = processed_result.get("transcript", {})
            domain_features = processed_result.get("domain_features", {})
            theory_practice_results = processed_result.get("theory_practice_results", {})
            theory_practice_patterns = processed_result.get("theory_practice_patterns", {})

            if not video_id or not metadata:
                logger.error("Missing required fields for indexing")
                return False

            # Log domain features
            key_concepts = domain_features.get("key_concepts", [])
            logger.info(f"Indexing video {video_id} with {len(key_concepts)} key concepts")

            # If no key concepts were found, try to extract them from domain keywords
            if not key_concepts and metadata.get("domain"):
                logger.warning(f"No key concepts found for video {video_id}, attempting domain-based extraction")
                key_concepts = self._extract_fallback_concepts(
                    transcript.get("segments", []),
                    metadata.get("domain"),
                    metadata.get("language", "en")
                )
                domain_features["key_concepts"] = key_concepts
                logger.info(f"Extracted {len(key_concepts)} fallback concepts based on domain keywords")

            # Lock database for thread safety
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                try:
                    # Begin transaction
                    conn.execute("BEGIN TRANSACTION")

                    # Index video
                    self._index_video(cursor, video_id, metadata, theory_practice_results)

                    # Index segments with enhanced text processing
                    segments = transcript.get("segments", [])
                    self._index_segments(cursor, video_id, segments)

                    # Index concepts with enhanced analysis
                    self._index_concepts(cursor, video_id, key_concepts, segments)

                    # Index theory-practice patterns
                    self._index_theory_practice_patterns(cursor, video_id, theory_practice_patterns)

                    # Index n-grams for better phrase matching
                    self._index_ngrams(cursor, video_id, segments)

                    # Extract and index concept relationships
                    self._index_concept_relationships(cursor, video_id, key_concepts)

                    # Generate and index domain-specific synonyms
                    self._index_domain_synonyms(cursor, key_concepts, metadata.get("domain"))

                    # Commit transaction
                    conn.commit()
                    logger.info(f"Successfully indexed content for video {video_id}")
                    return True

                except Exception as e:
                    # Rollback on error
                    conn.rollback()
                    logger.error(f"Error during indexing: {e}")
                    return False

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return False

    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a search query with enhanced processing for better relevance.

        Args:
            query: Structured query dictionary

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

            # Check cache for frequently used queries
            query_hash = self._generate_query_hash(query_text, theory_practice_ratio, domain, filters)
            cached_result = self._check_search_cache(query_hash)
            if cached_result:
                logger.info(f"Using cached result for query: '{query_text}'")
                cached_result["executionTimeMs"] = int((time.time() - start_time) * 1000)
                return cached_result

            # Log search parameters
            logger.info(f"Searching for '{query_text}' with theory/practice ratio: {theory_practice_ratio}, domain: {domain}")

            # Preprocess query for better matching
            processed_query = self._preprocess_query(query_text, domain)

            # Lock database for thread safety
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row  # Use row factory for dict-like results
                cursor = conn.cursor()

                try:
                    # Try direct concept search first with processed query and expanded terms
                    sql_query, params = self._build_enhanced_concept_search_query(
                        processed_query, filters, theory_practice_ratio, domain
                    )

                    # Get total count
                    count_sql = f"SELECT COUNT(*) FROM ({sql_query})"
                    cursor.execute(count_sql, params)
                    total_results = cursor.fetchone()[0]

                    # If no direct concept matches, try segment text search
                    segment_search_used = False
                    if total_results == 0:
                        logger.info(f"No direct concept matches for '{query_text}', trying segment text search")
                        try:
                            sql_query, params = self._build_enhanced_segment_search_query(
                                processed_query, filters, theory_practice_ratio, domain
                            )
                            segment_search_used = True

                            # Get total count again
                            count_sql = f"SELECT COUNT(*) FROM ({sql_query})"
                            cursor.execute(count_sql, params)
                            total_results = cursor.fetchone()[0]
                        except Exception as e:
                            logger.error(f"Error in segment search: {e}")
                            # Fallback to fuzzy search as a last resort
                            try:
                                sql_query, params = self._build_fuzzy_search_query(
                                    query_text, filters, theory_practice_ratio, domain
                                )
                                segment_search_used = True

                                # Get total count again
                                count_sql = f"SELECT COUNT(*) FROM ({sql_query})"
                                cursor.execute(count_sql, params)
                                total_results = cursor.fetchone()[0]
                            except Exception as e2:
                                logger.error(f"Error in fuzzy search: {e2}")
                                # Ultimate fallback to empty results
                                total_results = 0
                                segment_search_used = False
                                sql_query = "SELECT 1 WHERE 0"  # Empty query

                    # Get counts by context type
                    theoretical_results = 0
                    practical_results = 0

                    if total_results > 0:
                        theoretical_sql = f"SELECT COUNT(*) FROM ({sql_query}) WHERE context_type = 'theoretical'"
                        try:
                            cursor.execute(theoretical_sql, params)
                            theoretical_results = cursor.fetchone()[0]
                        except Exception as e:
                            logger.warning(f"Error counting theoretical results: {e}")

                        practical_sql = f"SELECT COUNT(*) FROM ({sql_query}) WHERE context_type = 'practical'"
                        try:
                            cursor.execute(practical_sql, params)
                            practical_results = cursor.fetchone()[0]
                        except Exception as e:
                            logger.warning(f"Error counting practical results: {e}")

                    # Add pagination
                    if total_results > 0:
                        paginated_sql = f"{sql_query} LIMIT ? OFFSET ?"
                        paginated_params = params + [limit, offset]

                        # Execute query
                        cursor.execute(paginated_sql, paginated_params)
                        rows = cursor.fetchall()
                    else:
                        rows = []

                    # Convert to list of dicts and enhance results
                    results = []
                    for row in rows:
                        result = dict(row)

                        # Enhance with related concepts if this is a concept result
                        if not segment_search_used and result.get("concept_id"):
                            result["related_concepts"] = self._get_related_concepts(
                                cursor, result.get("concept_id")
                            )

                        # Get video title if needed
                        if "video_title" not in result and result.get("video_id"):
                            video_id = result.get("video_id")
                            try:
                                cursor.execute("SELECT title FROM videos WHERE video_id = ?", (video_id,))
                                video_row = cursor.fetchone()
                                if video_row:
                                    result["video_title"] = video_row[0]
                            except Exception as e:
                                logger.warning(f"Error getting video title: {e}")

                        # Enhance context with highlighting of search terms
                        if "context_text" in result:
                            result["context_text"] = self._highlight_search_terms(
                                result["context_text"],
                                query_text,
                                processed_query["stemmed_terms"]
                            )

                        # Add relevance explanation
                        result["relevance_explanation"] = self._generate_relevance_explanation(result)

                        results.append(result)

                    # Calculate execution time
                    execution_time_ms = int((time.time() - start_time) * 1000)

                    # Prepare response
                    response = {
                        "results": results,
                        "totalResults": total_results,
                        "theoreticalResults": theoretical_results,
                        "practicalResults": practical_results,
                        "executionTimeMs": execution_time_ms,
                        "query": query,
                        "expandedTerms": processed_query.get("expanded_terms", []),
                        "searchType": "segment" if segment_search_used else "concept"
                    }

                    logger.info(f"Search for '{query_text}' returned {total_results} results in {execution_time_ms}ms")

                    # Cache the result for future use
                    self._update_search_cache(query_hash, response)

                    return response

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error executing search query: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())

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

    def get_concept_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept with enhanced relationship data.

        Args:
            concept_id: Concept ID

        Returns:
            Concept details dictionary if found, None otherwise
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                try:
                    # Get concept details
                    cursor.execute(
                        "SELECT * FROM concepts WHERE concept_id = ?",
                        (concept_id,)
                    )
                    concept_row = cursor.fetchone()

                    if not concept_row:
                        return None

                    concept = dict(concept_row)

                    # Get occurrences with enhanced context
                    cursor.execute(
                        """
                        SELECT o.*, v.title AS video_title
                        FROM occurrences o
                        JOIN videos v ON o.video_id = v.video_id
                        WHERE o.concept_id = ?
                        ORDER BY o.relevance_score DESC
                        """,
                        (concept_id,)
                    )
                    occurrence_rows = cursor.fetchall()
                    occurrences = [dict(row) for row in occurrence_rows]

                    # Get related concepts with relationship types
                    related_concepts = self._get_related_concepts(cursor, concept_id, include_relationship_type=True)

                    # Get theoretical foundations (for practical concepts)
                    theoretical_foundations = []
                    if concept.get("concept_class") == "practical":
                        cursor.execute(
                            """
                            SELECT c.*, cr.relationship_type, cr.strength
                            FROM concepts c
                            JOIN concept_relationships cr ON c.concept_id = cr.target_concept_id
                            WHERE cr.source_concept_id = ? AND c.concept_class = 'theoretical'
                            ORDER BY cr.strength DESC
                            LIMIT 10
                            """,
                            (concept_id,)
                        )
                        foundation_rows = cursor.fetchall()
                        theoretical_foundations = [dict(row) for row in foundation_rows]

                    # Get practical applications (for theoretical concepts)
                    practical_applications = []
                    if concept.get("concept_class") == "theoretical":
                        cursor.execute(
                            """
                            SELECT c.*, cr.relationship_type, cr.strength
                            FROM concepts c
                            JOIN concept_relationships cr ON c.concept_id = cr.target_concept_id
                            WHERE cr.source_concept_id = ? AND c.concept_class = 'practical'
                            ORDER BY cr.strength DESC
                            LIMIT 10
                            """,
                            (concept_id,)
                        )
                        application_rows = cursor.fetchall()
                        practical_applications = [dict(row) for row in application_rows]

                    # Get synonyms for this concept
                    synonyms = []
                    cursor.execute(
                        """
                        SELECT synonym, confidence
                        FROM synonyms
                        WHERE term = ?
                        ORDER BY confidence DESC
                        """,
                        (concept.get("text", ""),)
                    )
                    synonym_rows = cursor.fetchall()
                    synonyms = [{"text": row[0], "confidence": row[1]} for row in synonym_rows]

                    # Compile result
                    result = {
                        "concept": concept,
                        "occurrences": occurrences,
                        "related": related_concepts,
                        "theoretical_foundations": theoretical_foundations,
                        "practical_applications": practical_applications,
                        "synonyms": synonyms
                    }

                    return result

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error getting concept details: {e}")
            return None

    def get_video_concepts(self,
        video_id: str,
        context_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get concepts extracted from a video with enhanced relationship information.

        Args:
            video_id: YouTube video ID
            context_type: Content type filter (theoretical, practical, mixed)

        Returns:
            Video concepts dictionary if found, None otherwise
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                try:
                    # Get video details
                    cursor.execute(
                        "SELECT * FROM videos WHERE video_id = ?",
                        (video_id,)
                    )
                    video_row = cursor.fetchone()

                    if not video_row:
                        return None

                    video = dict(video_row)

                    # Get concepts with enhanced ordering by relevance and occurrence count
                    sql = """
                    SELECT c.*, COUNT(o.occurrence_id) AS occurrence_count,
                           AVG(o.relevance_score) AS avg_relevance
                    FROM concepts c
                    JOIN occurrences o ON c.concept_id = o.concept_id
                    WHERE o.video_id = ?
                    """
                    params = [video_id]

                    if context_type:
                        sql += " AND o.context_type = ?"
                        params.append(context_type)

                    sql += " GROUP BY c.concept_id ORDER BY avg_relevance DESC, occurrence_count DESC"

                    cursor.execute(sql, params)
                    concept_rows = cursor.fetchall()
                    concepts = [dict(row) for row in concept_rows]

                    # Get concept relationships within this video
                    cursor.execute(
                        """
                        SELECT cr.*
                        FROM concept_relationships cr
                        JOIN occurrences o1 ON cr.source_concept_id = o1.concept_id
                        JOIN occurrences o2 ON cr.target_concept_id = o2.concept_id
                        WHERE o1.video_id = ? AND o2.video_id = ?
                        GROUP BY cr.relationship_id
                        ORDER BY cr.strength DESC
                        """,
                        (video_id, video_id)
                    )
                    relationship_rows = cursor.fetchall()
                    relationships = [dict(row) for row in relationship_rows]

                    # Get theory-practice patterns
                    cursor.execute(
                        """
                        SELECT * FROM theory_practice_patterns
                        WHERE video_id = ?
                        ORDER BY start_time
                        """,
                        (video_id,)
                    )
                    pattern_rows = cursor.fetchall()
                    patterns = [dict(row) for row in pattern_rows]

                    # Get most frequent n-grams
                    cursor.execute(
                        """
                        SELECT ngram, stemmed_ngram, frequency
                        FROM ngrams
                        WHERE video_id = ? AND length(ngram) > 3
                        ORDER BY frequency DESC, length(ngram) DESC
                        LIMIT 20
                        """,
                        (video_id,)
                    )
                    ngram_rows = cursor.fetchall()
                    ngrams = [{"text": row[0], "stemmed_text": row[1], "frequency": row[2]} for row in ngram_rows]

                    # Compile result
                    result = {
                        "video": video,
                        "concepts": concepts,
                        "relationships": relationships,
                        "theory_practice_patterns": patterns,
                        "frequent_phrases": ngrams,
                        "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
                    }

                    return result

                finally:
                    conn.close()

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
        Generate a learning path for a set of concepts with improved sequencing.

        Args:
            concept_ids: List of concept IDs
            theory_practice_ratio: Desired ratio of theoretical to practical content
            domain: Optional domain filter

        Returns:
            Learning path dictionary if successful, None otherwise
        """
        try:
            if not concept_ids:
                return None

            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                try:
                    # Get concepts
                    placeholders = ", ".join(["?"] * len(concept_ids))
                    cursor.execute(
                        f"SELECT * FROM concepts WHERE concept_id IN ({placeholders})",
                        concept_ids
                    )
                    concept_rows = cursor.fetchall()
                    target_concepts = [dict(row) for row in concept_rows]

                    if not target_concepts:
                        return None

                    # Determine the domain if not specified
                    if not domain:
                        # Use the most common domain among target concepts
                        domains = {}
                        for concept in target_concepts:
                            concept_domain = concept.get("domain")
                            if concept_domain:
                                domains[concept_domain] = domains.get(concept_domain, 0) + 1

                        if domains:
                            domain = max(domains.items(), key=lambda x: x[1])[0]

                    # Get prerequisite concepts using concept relationships
                    prerequisite_concepts = []

                    # For each target concept, find prerequisite concepts
                    for concept in target_concepts:
                        concept_id = concept.get("concept_id")

                        # Use concept relationships to find prerequisites
                        cursor.execute(
                            """
                            SELECT c.*, cr.relationship_type, cr.strength
                            FROM concepts c
                            JOIN concept_relationships cr ON c.concept_id = cr.target_concept_id
                            WHERE cr.source_concept_id = ?
                              AND cr.relationship_type = 'prerequisite'
                              AND (? IS NULL OR c.domain = ?)
                            ORDER BY cr.strength DESC
                            LIMIT 5
                            """,
                            (concept_id, domain, domain)
                        )

                        prereq_rows = cursor.fetchall()

                        for row in prereq_rows:
                            prereq = dict(row)
                            if prereq not in prerequisite_concepts and prereq.get("concept_id") not in concept_ids:
                                prerequisite_concepts.append(prereq)

                    # If we don't have enough prerequisites, find more using co-occurrence
                    if len(prerequisite_concepts) < 5:
                        for concept in target_concepts:
                            concept_id = concept.get("concept_id")

                            # Find concepts that frequently appear before this concept
                            cursor.execute(
                                """
                                SELECT c.*, COUNT(*) AS co_occurrence
                                FROM concepts c
                                JOIN occurrences o1 ON c.concept_id = o1.concept_id
                                JOIN occurrences o2 ON o1.video_id = o2.video_id
                                    AND o1.start_time < o2.start_time
                                WHERE o2.concept_id = ?
                                AND (? IS NULL OR c.domain = ?)
                                AND c.concept_id NOT IN ({})
                                GROUP BY c.concept_id
                                ORDER BY co_occurrence DESC
                                LIMIT 5
                                """.format(placeholders),
                                [concept_id, domain, domain] + concept_ids
                            )

                            cooccur_rows = cursor.fetchall()

                            for row in cooccur_rows:
                                prereq = dict(row)
                                if prereq not in prerequisite_concepts and prereq.get("concept_id") not in concept_ids:
                                    prerequisite_concepts.append(prereq)

                    # Combine target and prerequisite concepts
                    all_concepts = target_concepts + prerequisite_concepts

                    # Sort concepts by theoretical/practical class based on desired ratio
                    theoretical_concepts = [c for c in all_concepts if c.get("concept_class") == "theoretical"]
                    practical_concepts = [c for c in all_concepts if c.get("concept_class") == "practical"]
                    mixed_concepts = [c for c in all_concepts if c.get("concept_class") not in ("theoretical", "practical")]

                    # Calculate actual numbers based on ratio
                    total_concepts = len(all_concepts)
                    target_theoretical = int(total_concepts * theory_practice_ratio)
                    target_practical = total_concepts - target_theoretical

                    # Adjust for available concepts
                    available_theoretical = len(theoretical_concepts)
                    available_practical = len(practical_concepts)

                    if available_theoretical < target_theoretical:
                        # Add some mixed concepts to theoretical
                        needed = min(target_theoretical - available_theoretical, len(mixed_concepts))
                        theoretical_concepts.extend(mixed_concepts[:needed])
                        mixed_concepts = mixed_concepts[needed:]

                    if available_practical < target_practical:
                        # Add remaining mixed concepts to practical
                        needed = min(target_practical - available_practical, len(mixed_concepts))
                        practical_concepts.extend(mixed_concepts[:needed])

                    # Order concepts by prerequisite relationships and co-occurrence
                    ordered_theoretical = self._order_concepts_by_dependencies(
                        cursor, theoretical_concepts
                    )
                    ordered_practical = self._order_concepts_by_dependencies(
                        cursor, practical_concepts
                    )

                    # Create learning path by interleaving theoretical and practical concepts
                    path = self._create_optimal_learning_path(
                        ordered_theoretical,
                        ordered_practical,
                        theory_practice_ratio
                    )

                    # Estimate time for each concept
                    for i, concept in enumerate(path):
                        cursor.execute(
                            """
                            SELECT AVG(o.end_time - o.start_time) as avg_duration
                            FROM occurrences o
                            WHERE o.concept_id = ?
                            """,
                            (concept.get("concept_id"),)
                        )

                        row = cursor.fetchone()
                        avg_duration = row[0] if row and row[0] else 300  # Default to 5 minutes

                        concept["order"] = i + 1
                        concept["estimated_time_minutes"] = round(avg_duration / 60)

                    # Calculate total time
                    total_time = sum(c.get("estimated_time_minutes", 0) for c in path)

                    # Count theoretical and practical concepts
                    path_theoretical = sum(1 for c in path if c.get("concept_class") == "theoretical")
                    path_practical = sum(1 for c in path if c.get("concept_class") == "practical")

                    # Calculate actual ratio
                    actual_ratio = path_theoretical / total_concepts if total_concepts > 0 else 0.5

                    # Get best videos for this learning path
                    recommended_videos = self._get_recommended_videos_for_path(cursor, [c.get("concept_id") for c in path])

                    # Compile result
                    result = {
                        "concepts": path,
                        "theory_practice_ratio": actual_ratio,
                        "total_theoretical_concepts": path_theoretical,
                        "total_practical_concepts": path_practical,
                        "estimated_total_time_minutes": total_time,
                        "domain": domain,
                        "recommended_videos": recommended_videos[:5]  # Limit to top 5 videos
                    }

                    return result

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return None

    def _extract_fallback_concepts(
        self,
        segments: List[Dict[str, Any]],
        domain: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Extract fallback concepts from segments based on domain keywords.

        Args:
            segments: List of transcript segments
            domain: Content domain (mathematics, programming, physics)
            language: Language of content (en, ru)

        Returns:
            List of extracted concepts
        """
        # Domain-specific keywords to look for
        domain_keywords = {
            "mathematics": {
                "en": [
                    "theorem", "lemma", "proof", "equation", "function", "calculus",
                    "derivative", "integral", "algebra", "geometry", "matrix",
                    "vector", "set", "topology", "analysis", "group", "field"
                ],
                "ru": [
                    "теорема", "лемма", "доказательство", "уравнение", "функция",
                    "анализ", "производная", "интеграл", "алгебра", "геометрия",
                    "матрица", "вектор", "множество", "топология", "группа", "поле"
                ]
            },
            "programming": {
                "en": [
                    "algorithm", "function", "method", "class", "object", "variable",
                    "array", "list", "tree", "graph", "stack", "queue", "heap", "sort"
                ],
                "ru": [
                    "алгоритм", "функция", "метод", "класс", "объект", "переменная",
                    "массив", "список", "дерево", "граф", "стек", "очередь", "куча",
                    "сортировка"
                ]
            },
            "physics": {
                "en": [
                    "force", "energy", "momentum", "mass", "velocity", "acceleration",
                    "gravity", "field", "wave", "particle", "quantum", "relativity"
                ],
                "ru": [
                    "сила", "энергия", "импульс", "масса", "скорость", "ускорение",
                    "гравитация", "поле", "волна", "частица", "квант", "относительность"
                ]
            }
        }

        # Use keywords for specific domain and language
        keywords = domain_keywords.get(domain, {}).get(language, [])
        if not keywords and language == "ru":
            # Fallback to English if no Russian keywords
            keywords = domain_keywords.get(domain, {}).get("en", [])

        if not keywords:
            logger.warning(f"No keywords found for domain {domain} and language {language}")
            return []

        # Combine all segment texts
        full_text = " ".join([s.get("text", "") for s in segments])

        # Find all occurrences of each keyword
        concepts = []
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, full_text.lower())
            frequency = len(matches)

            if frequency > 0:
                # Determine if theoretical based on segments where it appears
                theoretical = self._is_concept_theoretical_in_segments(keyword, segments)

                concepts.append({
                    "text": keyword,
                    "domain": domain,
                    "frequency": frequency,
                    "theoretical": theoretical
                })

        # Return top concepts by frequency
        return sorted(concepts, key=lambda x: x["frequency"], reverse=True)[:20]

    def _is_concept_theoretical_in_segments(self, concept: str, segments: List[Dict[str, Any]]) -> bool:
        """Determine if a concept is predominantly theoretical based on the segments it appears in."""
        theoretical_count = 0
        practical_count = 0

        for segment in segments:
            text = segment.get("text", "").lower()
            if re.search(r'\b' + re.escape(concept.lower()) + r'\b', text):
                content_type = segment.get("content_type", "")
                if content_type == "theoretical":
                    theoretical_count += 1
                elif content_type == "practical":
                    practical_count += 1

        # If the concept appears more in theoretical segments, classify it as theoretical
        return theoretical_count >= practical_count

    def _preprocess_query(self, query_text: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Preprocess search query with text normalization and expansion.

        Args:
            query_text: Raw query text
            domain: Optional domain for domain-specific expansions

        Returns:
            Processed query information
        """
        # Normalize the query text
        normalized_text = query_text.lower().strip()

        # Split into terms
        terms = [t for t in re.findall(r'\b[a-z0-9]+\b', normalized_text) if len(t) > 2]

        # Apply stemming if available
        stemmed_terms = []
        if self.stemmer:
            stemmed_terms = [self.stemmer.stem(term) for term in terms]
        else:
            # Simple stemming fallback (remove common suffixes)
            stemmed_terms = []
            for term in terms:
                # Cache stemmed terms for performance
                if term in self.cache["stemmed_terms"]:
                    stemmed_term = self.cache["stemmed_terms"][term]
                else:
                    stemmed_term = self._simple_stem(term)
                    self.cache["stemmed_terms"][term] = stemmed_term
                stemmed_terms.append(stemmed_term)

        # Prepare result
        result = {
            "original": query_text,
            "normalized": normalized_text,
            "terms": terms,
            "stemmed_terms": stemmed_terms,
            "expanded_terms": []
        }

        # Add query expansion if enabled
        if self.enable_query_expansion and terms:
            expanded_terms = self._expand_query_terms(terms, stemmed_terms, domain)
            result["expanded_terms"] = expanded_terms

        return result

    def _simple_stem(self, term: str) -> str:
        """
        Simple stemming function for when NLTK is not available.

        Args:
            term: Word to stem

        Returns:
            Stemmed word
        """
        if len(term) < 4:
            return term

        # Remove common suffixes
        if term.endswith('ing'):
            return term[:-3]
        elif term.endswith('ed'):
            return term[:-2]
        elif term.endswith('s') and not term.endswith('ss'):
            return term[:-1]
        elif term.endswith('ies'):
            return term[:-3] + 'y'
        elif term.endswith('es'):
            return term[:-2]
        elif term.endswith('ly'):
            return term[:-2]
        elif term.endswith('ment'):
            return term[:-4]
        return term

    def _expand_query_terms(self, terms: List[str], stemmed_terms: List[str], domain: Optional[str] = None) -> List[str]:
        """
        Expand query terms with synonyms and related terms.

        Args:
            terms: Original query terms
            stemmed_terms: Stemmed query terms
            domain: Optional domain for domain-specific expansions

        Returns:
            List of expanded terms
        """
        expanded_terms = []

        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                # Look up synonyms in the database
                for term in terms:
                    # Check cache first
                    if term in self.cache["synonyms"]:
                        term_synonyms = self.cache["synonyms"][term]
                    else:
                        # Query for synonyms with domain filter if provided
                        if domain:
                            cursor.execute(
                                """
                                SELECT synonym FROM synonyms
                                WHERE term = ? AND (domain = ? OR domain IS NULL)
                                ORDER BY confidence DESC
                                LIMIT ?
                                """,
                                (term, domain, self.max_expanded_terms)
                            )
                        else:
                            cursor.execute(
                                """
                                SELECT synonym FROM synonyms
                                WHERE term = ?
                                ORDER BY confidence DESC
                                LIMIT ?
                                """,
                                (term, self.max_expanded_terms)
                            )

                        rows = cursor.fetchall()
                        term_synonyms = [row[0] for row in rows]

                        # Store in cache
                        self.cache["synonyms"][term] = term_synonyms

                    # Add unique synonyms to expanded terms
                    for synonym in term_synonyms:
                        if synonym not in terms and synonym not in expanded_terms:
                            expanded_terms.append(synonym)

                            # Limit total expanded terms
                            if len(expanded_terms) >= self.max_expanded_terms:
                                break

                    # Limit total expanded terms
                    if len(expanded_terms) >= self.max_expanded_terms:
                        break

                # If we still need more terms, look for stemmed synonyms
                if len(expanded_terms) < self.max_expanded_terms:
                    for stemmed_term in stemmed_terms:
                        # Look for concepts with matching stemmed form
                        cursor.execute(
                            """
                            SELECT text FROM concepts
                            WHERE stemmed_text LIKE ? AND text NOT IN ({}) AND text NOT IN ({})
                            ORDER BY total_occurrences DESC
                            LIMIT ?
                            """.format(
                                ','.join(['?'] * len(terms)),
                                ','.join(['?'] * len(expanded_terms))
                            ),
                            [f"%{stemmed_term}%"] + terms + expanded_terms + [self.max_expanded_terms - len(expanded_terms)]
                        )

                        rows = cursor.fetchall()
                        for row in rows:
                            expanded_terms.append(row[0])

                            # Limit total expanded terms
                            if len(expanded_terms) >= self.max_expanded_terms:
                                break

                        # Limit total expanded terms
                        if len(expanded_terms) >= self.max_expanded_terms:
                            break
            finally:
                conn.close()

        return expanded_terms

    def _generate_query_hash(self, query_text: str, theory_practice_ratio: Optional[float], domain: Optional[str], filters: Dict[str, Any]) -> str:
        """
        Generate a hash for a search query for caching.

        Args:
            query_text: Query text
            theory_practice_ratio: Theory/practice ratio
            domain: Domain filter
            filters: Additional filters

        Returns:
            Query hash string
        """
        # Create a dictionary of query parameters
        query_params = {
            "text": query_text.lower().strip(),
            "theory_practice_ratio": theory_practice_ratio,
            "domain": domain
        }

        # Add relevant filters
        if filters:
            for key, value in filters.items():
                if key in ["video_id", "domain"] and value:
                    query_params[f"filter_{key}"] = value

        # Convert to string and hash
        param_str = json.dumps(query_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _check_search_cache(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if a search query result is cached.

        Args:
            query_hash: Query hash

        Returns:
            Cached result or None
        """
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                # Check if query is in cache
                cursor.execute(
                    "SELECT result_json, count FROM search_cache WHERE query_hash = ?",
                    (query_hash,)
                )

                row = cursor.fetchone()
                if row:
                    # Update usage count
                    cursor.execute(
                        "UPDATE search_cache SET count = count + 1, timestamp = ? WHERE query_hash = ?",
                        (time.strftime("%Y-%m-%d %H:%M:%S"), query_hash)
                    )
                    conn.commit()

                    # Parse the cached result
                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError:
                        # Invalid JSON, remove from cache
                        cursor.execute("DELETE FROM search_cache WHERE query_hash = ?", (query_hash,))
                        conn.commit()

                return None

            except Exception as e:
                logger.warning(f"Error checking search cache: {e}")
                return None

            finally:
                conn.close()

    def _update_search_cache(self, query_hash: str, result: Dict[str, Any]) -> None:
        """
        Update the search cache with a new result.

        Args:
            query_hash: Query hash
            result: Search result
        """
        # Don't cache small result sets as they're quick to generate
        if result.get("totalResults", 0) < 3:
            return

        # Don't cache error results
        if "error" in result:
            return

        # Create a cacheable version (without pagination-specific data)
        cacheable_result = result.copy()

        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                # Convert to JSON
                result_json = json.dumps(cacheable_result)

                # Insert or replace in cache
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO search_cache
                    (query_hash, query, result_json, timestamp, count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        query_hash,
                        cacheable_result.get("query", {}).get("original_text", ""),
                        result_json,
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        1
                    )
                )

                # Remove old cache entries if too many
                cursor.execute(
                    """
                    DELETE FROM search_cache
                    WHERE query_hash NOT IN (
                        SELECT query_hash FROM search_cache
                        ORDER BY count DESC, timestamp DESC
                        LIMIT ?
                    )
                    """,
                    (self.max_cache_entries,)
                )

                conn.commit()

            except Exception as e:
                logger.warning(f"Error updating search cache: {e}")
                conn.rollback()

            finally:
                conn.close()

    def _highlight_search_terms(self, text: str, query: str, stemmed_terms: List[str]) -> str:
        """
        Highlight search terms in context text.

        Args:
            text: Context text
            query: Original query text
            stemmed_terms: Stemmed query terms

        Returns:
            Text with search terms highlighted
        """
        if not text or not query:
            return text

        # Extract terms from query
        query_terms = re.findall(r'\b\w+\b', query.lower())

        # Make a copy of the text for highlighting
        highlighted_text = text

        # First try to highlight exact query
        if len(query) > 3:  # Only if query is substantive
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            highlighted_text = pattern.sub(r'<mark>\g<0></mark>', highlighted_text)

        # Then highlight individual terms
        for term in query_terms:
            if len(term) < 3:  # Skip very short terms
                continue

            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            # Avoid highlighting terms inside already highlighted sections
            highlighted_text = re.sub(
                r'(<mark>.*?)' + re.escape(term) + r'(.*?</mark>)',
                r'\1' + term + r'\2',
                highlighted_text,
                flags=re.IGNORECASE
            )
            highlighted_text = pattern.sub(r'<mark>\g<0></mark>', highlighted_text)

        # Also try to highlight stemmed terms if they're different from original terms
        for stem in stemmed_terms:
            if stem not in query_terms and len(stem) >= 3:
                # Find words that might match this stem
                for word in re.findall(r'\b\w+\b', text):
                    if self._simple_stem(word.lower()) == stem:
                        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                        # Avoid highlighting terms inside already highlighted sections
                        highlighted_text = re.sub(
                            r'(<mark>.*?)' + re.escape(word) + r'(.*?</mark>)',
                            r'\1' + word + r'\2',
                            highlighted_text,
                            flags=re.IGNORECASE
                        )
                        highlighted_text = pattern.sub(r'<mark>\g<0></mark>', highlighted_text)

        return highlighted_text

    def _generate_relevance_explanation(self, result: Dict[str, Any]) -> str:
        """
        Generate an explanation for why a result is relevant.

        Args:
            result: Search result dictionary

        Returns:
            Explanation string
        """
        explanations = []

        # Check if this is a concept match
        if result.get("concept_id") and result.get("text"):
            explanations.append(f"Matches the concept '{result.get('text')}'")

            # Add domain information
            if result.get("domain"):
                explanations.append(f"In the {result.get('domain')} domain")

            # Add theoretical/practical information
            if result.get("concept_class") == "theoretical":
                explanations.append("Theoretical concept")
            elif result.get("concept_class") == "practical":
                explanations.append("Practical concept")

            # Add occurrence information
            if result.get("occurrence_relevance"):
                if result.get("occurrence_relevance") > 0.8:
                    explanations.append("Very relevant occurrence")
                elif result.get("occurrence_relevance") > 0.5:
                    explanations.append("Relevant occurrence")

        # For segment matches
        elif result.get("segment_id") and not result.get("concept_id"):
            explanations.append("Matched in video segment text")

            # Add domain information
            if result.get("domain"):
                explanations.append(f"In the {result.get('domain')} domain")

            # Add theoretical/practical information
            if result.get("context_type") == "theoretical":
                explanations.append("Theoretical context")
            elif result.get("context_type") == "practical":
                explanations.append("Practical context")

        # Add search score information
        if result.get("search_relevance_score"):
            score = result.get("search_relevance_score")
            if score > 0.8:
                explanations.append("Very high relevance score")
            elif score > 0.6:
                explanations.append("High relevance score")
            elif score > 0.4:
                explanations.append("Medium relevance score")

        # Join explanations
        if explanations:
            return "; ".join(explanations)

        return "Matched search terms in content"

    def _build_enhanced_concept_search_query(
        self,
        processed_query: Dict[str, Any],
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> Tuple[str, List[Any]]:
        """
        Build SQL query for searching concepts with enhanced processing.

        Args:
            processed_query: Processed query dictionary
            filters: Additional filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            Tuple of (SQL query, parameters)
        """
        terms = processed_query.get("terms", [])
        stemmed_terms = processed_query.get("stemmed_terms", [])
        expanded_terms = processed_query.get("expanded_terms", [])

        # We need at least one search term
        if not terms and not stemmed_terms:
            raise ValueError("No valid search terms in query")

        # Base query with improved ranking
        sql = """
        SELECT
            c.*,
            o.video_id,
            o.segment_id,
            o.start_time,
            o.end_time,
            o.context_type,
            o.context_text,
            o.relevance_score as occurrence_relevance,
            v.title as video_title,
            (
                -- Base relevance (between 0.5 and 1.0 based on concept score)
                (0.5 + (c.relevance_score * 0.5)) *
                -- Occurrence relevance (multiplier between 0.5 and 1.0)
                (0.5 + (o.relevance_score * 0.5)) *
                -- Exact match boost (1.5x multiplier)
                CASE WHEN c.text LIKE ? THEN 1.5 ELSE 1.0 END *
                -- Stem match boost (1.2x multiplier)
                CASE WHEN c.stemmed_text LIKE ? THEN 1.2 ELSE 1.0 END
            ) as search_relevance_score
        FROM concepts c
        JOIN concepts_fts f ON c.concept_id = f.concept_id
        JOIN occurrences o ON c.concept_id = o.concept_id
        JOIN videos v ON o.video_id = v.video_id
        """

        # Start with exact matching term
        query_str = ""
        for term in terms:
            if query_str:
                query_str += " OR "
            query_str += f"{term}"

        # Add stemmed terms
        for stem in stemmed_terms:
            if query_str:
                query_str += " OR "
            query_str += f"{stem}*"

        # Add expanded terms with lower weight
        expanded_str = ""
        for ex_term in expanded_terms:
            if expanded_str:
                expanded_str += " OR "
            expanded_str += f"{ex_term}"

        # Combine the parts
        full_query = query_str
        if expanded_str:
            full_query = f"({query_str}) OR ({expanded_str})"

        # Handle empty terms
        if not full_query:
            full_query = "dummy_term_for_empty_query"

        # Add the MATCH condition
        sql += " WHERE concepts_fts MATCH ?"
        params = [full_query]

        # Add exact match pattern for ranking
        exact_match_pattern = f"%{processed_query.get('normalized', '')}%"
        params.append(exact_match_pattern)

        # Add stemmed match pattern for ranking
        stemmed_pattern = f"%{' '.join(stemmed_terms)}%"
        params.append(stemmed_pattern)

        # Apply domain filter
        if domain:
            sql += " AND c.domain = ?"
            params.append(domain)

        # Apply video filter if specified
        if "video_id" in filters:
            sql += " AND o.video_id = ?"
            params.append(filters["video_id"])

        # Apply video list filter if specified
        if "video_ids" in filters and filters["video_ids"]:
            video_ids = filters["video_ids"]
            placeholders = ','.join(['?'] * len(video_ids))
            sql += f" AND s.video_id IN ({placeholders})"
            params.extend(video_ids)

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.8:
                # Heavily favor theoretical
                sql += " AND s.context_type = 'theoretical'"
            elif theory_practice_ratio < 0.2:
                # Heavily favor practical
                sql += " AND s.context_type = 'practical'"
            elif theory_practice_ratio < 0.5:
                # Favor practical in ordering
                sql += " ORDER BY CASE WHEN s.context_type = 'practical' THEN 1 ELSE 2 END, search_relevance_score DESC"
            else:
                # Favor theoretical in ordering
                sql += " ORDER BY CASE WHEN s.context_type = 'theoretical' THEN 1 ELSE 2 END, search_relevance_score DESC"
        else:
            # Default ordering by relevance
            sql += " ORDER BY search_relevance_score DESC"

        return sql, params

    def _build_fuzzy_search_query(
        self,
        query_text: str,
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> Tuple[str, List[Any]]:
        """
        Build a fuzzy matching query as a fallback for when exact matching fails.

        Args:
            query_text: Original query text
            filters: Additional filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            Tuple of (SQL query, parameters)
        """
        # This is a fallback query that uses LIKE with wildcards for fuzzy matching
        normalized_query = query_text.lower().strip()
        terms = [t for t in re.findall(r'\b[a-z0-9]+\b', normalized_query) if len(t) > 3]

        if not terms:
            raise ValueError("No valid search terms for fuzzy matching")

        # Base query with segments
        sql = """
        SELECT
            s.segment_id,
            s.video_id,
            s.text AS context_text,
            s.context_type,
            v.title,
            v.domain,
            NULL as concept_id,
            NULL as text,
            NULL as normalized_text,
            NULL as concept_class,
            NULL as occurrence_id,
            0.5 as search_relevance_score,
            s.start_time,
            s.end_time
        FROM segments s
        JOIN videos v ON s.video_id = v.video_id
        WHERE
        """

        # Build LIKE conditions for each term
        like_conditions = []
        params = []

        for term in terms:
            if len(term) <= 3:
                continue

            # Create wildcards for fuzzy matching
            if len(term) > 5:
                # For longer terms, allow characters in the middle to differ
                prefix = term[:len(term)//2]
                suffix = term[-(len(term)//3):]
                like_pattern = f"%{prefix}%{suffix}%"
            else:
                # For shorter terms, just use prefix matching
                like_pattern = f"%{term}%"

            like_conditions.append("s.text LIKE ?")
            params.append(like_pattern)

        # Join conditions with OR
        if not like_conditions:
            raise ValueError("No valid search terms for fuzzy matching")

        sql += "(" + " OR ".join(like_conditions) + ")"

        # Apply domain filter
        if domain:
            sql += " AND s.domain = ?"
            params.append(domain)

        # Apply video filter if specified
        if "video_id" in filters:
            sql += " AND s.video_id = ?"
            params.append(filters["video_id"])

        # Apply video list filter if specified
        if "video_ids" in filters and filters["video_ids"]:
            video_ids = filters["video_ids"]
            placeholders = ','.join(['?'] * len(video_ids))
            sql += f" AND s.video_id IN ({placeholders})"
            params.extend(video_ids)

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.8:
                # Heavily favor theoretical
                sql += " AND s.context_type = 'theoretical'"
            elif theory_practice_ratio < 0.2:
                # Heavily favor practical
                sql += " AND s.context_type = 'practical'"
            elif theory_practice_ratio < 0.5:
                # Favor practical in ordering
                sql += " ORDER BY CASE WHEN s.context_type = 'practical' THEN 1 ELSE 2 END"
            else:
                # Favor theoretical in ordering
                sql += " ORDER BY CASE WHEN s.context_type = 'theoretical' THEN 1 ELSE 2 END"
        else:
            # Default ordering by text similarity (based on string length difference as a heuristic)
            sql += " ORDER BY s.segment_id"

        return sql, params filters["video_ids"]
            placeholders = ','.join(['?'] * len(video_ids))
            sql += f" AND o.video_id IN ({placeholders})"
            params.extend(video_ids)

        # Apply theory/practice filter
        if theory_practice_ratio is not None:
            if theory_practice_ratio > 0.8:
                # Heavily favor theoretical
                sql += " AND (o.context_type = 'theoretical' OR c.concept_class = 'theoretical')"
            elif theory_practice_ratio < 0.2:
                # Heavily favor practical
                sql += " AND (o.context_type = 'practical' OR c.concept_class = 'practical')"
            elif theory_practice_ratio < 0.5:
                # Favor practical in ordering
                sql += " ORDER BY CASE WHEN o.context_type = 'practical' THEN 1 ELSE 2 END, search_relevance_score DESC"
            else:
                # Favor theoretical in ordering
                sql += " ORDER BY CASE WHEN o.context_type = 'theoretical' THEN 1 ELSE 2 END, search_relevance_score DESC"
        else:
            # Default ordering by relevance
            sql += " ORDER BY search_relevance_score DESC"

        return sql, params

    def _build_enhanced_segment_search_query(
        self,
        processed_query: Dict[str, Any],
        filters: Dict[str, Any],
        theory_practice_ratio: Optional[float],
        domain: Optional[str]
    ) -> Tuple[str, List[Any]]:
        """
        Build SQL query for searching segments with enhanced processing.

        Args:
            processed_query: Processed query dictionary
            filters: Additional filters
            theory_practice_ratio: Theory/practice ratio filter
            domain: Domain filter

        Returns:
            Tuple of (SQL query, parameters)
        """
        terms = processed_query.get("terms", [])
        stemmed_terms = processed_query.get("stemmed_terms", [])
        expanded_terms = processed_query.get("expanded_terms", [])

        # We need at least one search term
        if not terms and not stemmed_terms:
            raise ValueError("No valid search terms in query")

        # Build a query that searches segments with ranking by relevance
        sql = """
        SELECT
            s.segment_id,
            s.video_id,
            s.text AS context_text,
            s.stemmed_text,
            s.context_type,
            v.title,
            v.domain,
            NULL as concept_id,
            NULL as text,
            NULL as normalized_text,
            NULL as stemmed_text,
            NULL as concept_class,
            NULL as occurrence_id,
            s.start_time,
            s.end_time,
            (
                -- Base relevance (higher for exact matches)
                CASE WHEN s.text LIKE ? THEN 1.0 ELSE 0.7 END *
                -- Stem match boost
                CASE WHEN s.stemmed_text LIKE ? THEN 1.2 ELSE 1.0 END
            ) as search_relevance_score
        FROM segments s
        JOIN segments_fts fts ON s.segment_id = fts.segment_id
        JOIN videos v ON s.video_id = v.video_id
        """

        # Start with exact matching terms
        query_str = ""
        for term in terms:
            if query_str:
                query_str += " OR "
            query_str += f"{term}"

        # Add stemmed terms
        for stem in stemmed_terms:
            if query_str:
                query_str += " OR "
            query_str += f"{stem}*"

        # Add expanded terms with lower weight
        expanded_str = ""
        for ex_term in expanded_terms:
            if expanded_str:
                expanded_str += " OR "
            expanded_str += f"{ex_term}"

        # Combine the parts
        full_query = query_str
        if expanded_str:
            full_query = f"({query_str}) OR ({expanded_str})"

        # Handle empty terms
        if not full_query:
            full_query = "dummy_term_for_empty_query"

        # Add the MATCH condition
        sql += " WHERE segments_fts MATCH ?"
        params = [full_query]

        # Add exact match pattern for ranking
        exact_match_pattern = f"%{processed_query.get('normalized', '')}%"
        params.append(exact_match_pattern)

        # Add stemmed match pattern for ranking
        stemmed_pattern = f"%{' '.join(stemmed_terms)}%"
        params.append(stemmed_pattern)

        # Apply domain filter
        if domain:
            sql += " AND s.domain = ?"
            params.append(domain)

        # Apply video filter if specified
        if "video_id" in filters:
            sql += " AND s.video_id = ?"
            params.append(filters["video_id"])

        # Apply video list filter if specified
        if "video_ids" in filters and filters["video_ids"]:
            video_ids =
