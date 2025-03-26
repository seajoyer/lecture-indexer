"""
Enhanced Query Processor module for Lecture Video Content Indexer.
Handles query analysis, expansion, and optimization for educational content.
Integrated with database, caching, and performance monitoring.
"""

import re
import logging
import sqlite3
import random
import hashlib
import json
from typing import Dict, List, Any, Optional, Set, Tuple

# Import new components
from database.db_init import get_db_context
from common.utils.cache_manager import CacheRegion
from common.utils.performance_utils import measure_time, time_function, measure_memory

# Configure logging
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Advanced query processor for educational content search.
    Improves search quality by analyzing, expanding, and optimizing queries.
    Integrated with database persistence, caching, and performance monitoring.
    """

    def __init__(self, db_path: str = None, stemmer=None):
        """
        Initialize the Query Processor with database and caching support.

        Args:
            db_path: Path to the SQLite database (legacy support)
            stemmer: Optional stemmer instance to use
        """
        with measure_time("query_processor_init"):
            logger.info("Initializing Query Processor with database integration")

            # Initialize database connection
            self.db_context = get_db_context()
            if self.db_context:
                logger.info("Connected to database context")
                # Get cache region for caching
                self.cache = self.db_context.get_cache_region("query_processor")
            else:
                logger.warning("Database context not available, using direct db connection")
                # Use provided db_path for backward compatibility
                self.db_path = db_path
                # Create a standalone cache if DB context is not available
                from common.utils.cache_manager import CacheManager
                cache_manager = CacheManager()
                self.cache = cache_manager.region("query_processor")

            self.stemmer = stemmer

            # Load domain-specific data with caching
            self.domain_synonyms = self._load_domain_synonyms()
            self.concept_hierarchy = self._load_concept_hierarchy()
            self.educational_phrases = self._load_educational_phrases()
            self.stopwords = self._load_stopwords()

            logger.info("Query Processor initialized with database integration")

    def _load_domain_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load domain-specific synonyms for query expansion with caching.

        Returns:
            Dictionary mapping domains to term mappings
        """
        # Check cache first
        cached_data = self.cache.get("domain_synonyms")
        if cached_data:
            logger.debug("Using cached domain synonyms")
            return cached_data

        # Try to load from database if available
        if self.db_context and hasattr(self.db_context, 'search_repository'):
            try:
                # In a real implementation, this would query a synonyms table
                # For now, we use hardcoded data
                logger.info("Using hardcoded domain synonyms (would load from database in production)")
            except Exception as e:
                logger.warning(f"Error loading domain synonyms from database: {e}")

        # Fallback to hardcoded synonyms
        domain_synonyms = {
            "mathematics": {
                "derivative": ["differentiation", "rate of change", "dx/dy", "d/dx"],
                "integral": ["integration", "antiderivative", "primitive"],
                "function": ["mapping", "transformation", "operator"],
                "equation": ["formula", "identity", "relation"],
                "matrix": ["array", "table", "grid"],
                "vector": ["direction", "arrow", "magnitude and direction"],
                "theorem": ["proposition", "statement", "law"],
                "proof": ["demonstration", "verification", "derivation"],
                "calculus": ["analysis", "differentiation and integration"],
                "algebra": ["algebraic manipulation", "equation solving"],
                "geometry": ["shape", "spatial relations", "euclidean space"]
            },
            "programming": {
                "function": ["method", "procedure", "routine", "subroutine"],
                "class": ["object type", "data structure", "blueprint"],
                "variable": ["property", "field", "attribute"],
                "algorithm": ["procedure", "process", "method"],
                "array": ["list", "collection", "sequence"],
                "loop": ["iteration", "repetition", "cycle"],
                "recursion": ["recursive algorithm", "self-reference"],
                "inheritance": ["subclassing", "extension", "derivation"],
                "framework": ["library", "toolkit", "platform"],
                "api": ["interface", "service", "endpoint"],
                "python": ["python programming", "python language"]
            },
            "physics": {
                "force": ["push", "pull", "interaction"],
                "energy": ["work", "power over time", "capacity for work"],
                "momentum": ["mass times velocity", "p=mv"],
                "velocity": ["speed", "rate of position change", "displacement over time"],
                "acceleration": ["rate of velocity change", "second derivative of position"],
                "mass": ["quantity of matter", "inertia", "resistance to acceleration"],
                "gravity": ["gravitational force", "attraction between masses"],
                "quantum": ["quantized", "discrete packets", "quantum mechanics"],
                "relativity": ["relativistic", "special relativity", "general relativity"],
                "electromagnetism": ["electromagnetic force", "electricity and magnetism"]
            }
        }

        # Cache the data
        self.cache.set("domain_synonyms", domain_synonyms, ttl=86400)  # Cache for 24 hours

        return domain_synonyms

    def _load_concept_hierarchy(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load concept hierarchy for domain-specific query expansion with caching.
        Maps concepts to broader/narrower concepts.

        Returns:
            Dictionary mapping domains to concept hierarchies
        """
        # Check cache first
        cached_data = self.cache.get("concept_hierarchy")
        if cached_data:
            logger.debug("Using cached concept hierarchy")
            return cached_data

        # Try to load from database if available
        if self.db_context and hasattr(self.db_context, 'concept_repository'):
            try:
                # In a real implementation, this would load concept relationships
                # For now, we use hardcoded data
                logger.info("Using hardcoded concept hierarchy (would load from database in production)")
            except Exception as e:
                logger.warning(f"Error loading concept hierarchy from database: {e}")

        # Fallback to hardcoded concept hierarchy
        concept_hierarchy = {
            "mathematics": {
                # Broader -> Narrower concepts
                "calculus": ["differentiation", "integration", "limits", "series", "differential equations"],
                "algebra": ["linear algebra", "abstract algebra", "polynomial algebra", "boolean algebra"],
                "geometry": ["euclidean geometry", "non-euclidean geometry", "analytic geometry", "differential geometry"],
                "analysis": ["real analysis", "complex analysis", "functional analysis", "harmonic analysis"],
                "statistics": ["probability", "regression", "hypothesis testing", "bayesian statistics"],

                # Narrower -> Broader concepts
                "_reverse_mappings": {
                    "differentiation": ["calculus"],
                    "integration": ["calculus"],
                    "limits": ["calculus"],
                    "series": ["calculus"],
                    "differential equations": ["calculus"],
                    "linear algebra": ["algebra"],
                    "abstract algebra": ["algebra"],
                    "polynomial algebra": ["algebra"],
                    "boolean algebra": ["algebra"],
                    "euclidean geometry": ["geometry"],
                    "non-euclidean geometry": ["geometry"],
                    "analytic geometry": ["geometry"],
                    "differential geometry": ["geometry"],
                    "real analysis": ["analysis"],
                    "complex analysis": ["analysis"],
                    "functional analysis": ["analysis"],
                    "harmonic analysis": ["analysis"],
                    "probability": ["statistics"],
                    "regression": ["statistics"],
                    "hypothesis testing": ["statistics"],
                    "bayesian statistics": ["statistics"]
                }
            },
            "programming": {
                # Broader -> Narrower concepts
                "programming paradigms": ["object-oriented", "functional", "procedural", "declarative"],
                "data structures": ["arrays", "linked lists", "trees", "graphs", "hash tables", "stacks", "queues"],
                "algorithms": ["sorting", "searching", "graph algorithms", "dynamic programming", "recursion"],
                "web development": ["frontend", "backend", "full-stack", "api", "web services"],
                "languages": ["python", "javascript", "java", "c++", "ruby", "go", "rust"],

                # Narrower -> Broader concepts
                "_reverse_mappings": {
                    "object-oriented": ["programming paradigms"],
                    "functional": ["programming paradigms"],
                    "procedural": ["programming paradigms"],
                    "declarative": ["programming paradigms"],
                    "arrays": ["data structures"],
                    "linked lists": ["data structures"],
                    "trees": ["data structures"],
                    "graphs": ["data structures"],
                    "hash tables": ["data structures"],
                    "stacks": ["data structures"],
                    "queues": ["data structures"],
                    "sorting": ["algorithms"],
                    "searching": ["algorithms"],
                    "graph algorithms": ["algorithms"],
                    "dynamic programming": ["algorithms"],
                    "recursion": ["algorithms"],
                    "frontend": ["web development"],
                    "backend": ["web development"],
                    "full-stack": ["web development"],
                    "api": ["web development"],
                    "web services": ["web development"]
                }
            },
            "physics": {
                # Broader -> Narrower concepts
                "mechanics": ["kinematics", "dynamics", "statics", "fluid mechanics", "classical mechanics"],
                "thermodynamics": ["heat", "energy", "entropy", "statistical mechanics"],
                "electromagnetism": ["electricity", "magnetism", "electromagnetic radiation"],
                "quantum physics": ["quantum mechanics", "quantum field theory", "quantum computing"],
                "relativity": ["special relativity", "general relativity", "spacetime"],

                # Narrower -> Broader concepts
                "_reverse_mappings": {
                    "kinematics": ["mechanics"],
                    "dynamics": ["mechanics"],
                    "statics": ["mechanics"],
                    "fluid mechanics": ["mechanics"],
                    "classical mechanics": ["mechanics"],
                    "heat": ["thermodynamics"],
                    "energy": ["thermodynamics"],
                    "entropy": ["thermodynamics"],
                    "statistical mechanics": ["thermodynamics"],
                    "electricity": ["electromagnetism"],
                    "magnetism": ["electromagnetism"],
                    "electromagnetic radiation": ["electromagnetism"],
                    "quantum mechanics": ["quantum physics"],
                    "quantum field theory": ["quantum physics"],
                    "quantum computing": ["quantum physics"],
                    "special relativity": ["relativity"],
                    "general relativity": ["relativity"],
                    "spacetime": ["relativity"]
                }
            }
        }

        # Cache the data
        self.cache.set("concept_hierarchy", concept_hierarchy, ttl=86400)  # Cache for 24 hours

        return concept_hierarchy

    def _load_educational_phrases(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Load common educational phrases and their components with caching.
        This helps when users search for parts of common educational phrases.

        Returns:
            Dictionary mapping domains to lists of (phrase, weight) tuples
        """
        # Check cache first
        cached_data = self.cache.get("educational_phrases")
        if cached_data:
            logger.debug("Using cached educational phrases")
            return cached_data

        # Try to load from database if available
        if self.db_context and hasattr(self.db_context, 'search_repository'):
            try:
                # In a real implementation, this would query a phrases table
                # For now, we use hardcoded data
                logger.info("Using hardcoded educational phrases (would load from database in production)")
            except Exception as e:
                logger.warning(f"Error loading educational phrases from database: {e}")

        # Fallback to hardcoded educational phrases
        educational_phrases = {
            "mathematics": [
                ("definition of derivative", 0.9),
                ("proof by induction", 0.9),
                ("fundamental theorem of calculus", 0.9),
                ("vector space", 0.8),
                ("eigenvalues and eigenvectors", 0.8),
                ("taylor series expansion", 0.8),
                ("differential equation", 0.8),
                ("riemann sum", 0.7),
                ("integration by parts", 0.7),
                ("partial derivative", 0.7)
            ],
            "programming": [
                ("object oriented programming", 0.9),
                ("data structures and algorithms", 0.9),
                ("linked list implementation", 0.8),
                ("binary search tree", 0.8),
                ("recursive function", 0.8),
                ("database normalization", 0.7),
                ("asynchronous programming", 0.7),
                ("memory management", 0.7),
                ("inheritance and polymorphism", 0.8),
                ("design patterns", 0.7)
            ],
            "physics": [
                ("newton's laws of motion", 0.9),
                ("conservation of energy", 0.9),
                ("quantum mechanics", 0.9),
                ("theory of relativity", 0.9),
                ("electromagnetic field", 0.8),
                ("wave-particle duality", 0.8),
                ("thermodynamic equilibrium", 0.7),
                ("nuclear fusion", 0.7),
                ("gravitational potential", 0.7),
                ("doppler effect", 0.7)
            ]
        }

        # Cache the data
        self.cache.set("educational_phrases", educational_phrases, ttl=86400)  # Cache for 24 hours

        return educational_phrases

    def _load_stopwords(self) -> Set[str]:
        """
        Load stopwords for query processing with caching.

        Returns:
            Set of stopwords
        """
        # Check cache first
        cached_data = self.cache.get("stopwords")
        if cached_data:
            logger.debug("Using cached stopwords")
            return cached_data

        # Fallback to hardcoded stopwords
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "can", "will", "just", "should", "now"
        }

        # Cache the data
        self.cache.set("stopwords", stopwords, ttl=86400)  # Cache for 24 hours

        return stopwords

    @time_function(threshold_ms=500)
    def process_query(self, query_text: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Process and enhance a query for better search results with caching.

        Args:
            query_text: Original query text
            domain: Optional domain to focus query expansion

        Returns:
            Dictionary with processed query information
        """
        # Create cache key
        cache_key = f"process_query_{hashlib.md5(query_text.encode()).hexdigest()}_{domain}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.debug("Using cached processed query")
            return cached_result

        # Basic cleanup
        query_text = query_text.strip()
        if not query_text:
            return {"original": "", "tokens": [], "expanded_terms": [], "stemmed_tokens": []}

        # Extract exact phrases (text within quotes)
        exact_phrases = re.findall(r'"([^"]+)"', query_text)

        # Remove exact phrases from query text for further processing
        clean_text = query_text
        for phrase in exact_phrases:
            clean_text = clean_text.replace(f'"{phrase}"', '')
        clean_text = clean_text.strip()

        # Tokenize
        tokens = [token.strip() for token in clean_text.split() if token.strip()]

        # Filter tokens
        filtered_tokens = [token for token in tokens if len(token) > 1 and token.lower() not in self.stopwords]

        # Detect the most likely domain if not provided
        detected_domain = domain or self._detect_query_domain(filtered_tokens, exact_phrases)

        # Expand the query with domain-specific terms
        expanded_terms = self._expand_query(filtered_tokens, exact_phrases, detected_domain)

        # Create stemmed versions if a stemmer is available
        stemmed_tokens = []
        if self.stemmer:
            stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
            stemmed_phrases = [' '.join([self.stemmer.stem(word) for word in phrase.split()])
                              for phrase in exact_phrases]
        else:
            stemmed_tokens = filtered_tokens
            stemmed_phrases = exact_phrases

        # Compile result
        result = {
            "original": query_text,
            "tokens": filtered_tokens,
            "stemmed_tokens": stemmed_tokens,
            "exact_phrases": exact_phrases,
            "stemmed_phrases": stemmed_phrases,
            "expanded_terms": expanded_terms,
            "domain": detected_domain
        }

        # Cache the processed query
        self.cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

        return result

    def _detect_query_domain(self, tokens: List[str], phrases: List[str]) -> Optional[str]:
        """
        Detect the most likely domain for a query using database features.

        Args:
            tokens: Query tokens
            phrases: Exact phrases in the query

        Returns:
            Detected domain or None if uncertain
        """
        # Create cache key
        token_hash = hashlib.md5(json.dumps(tokens).encode()).hexdigest()
        phrase_hash = hashlib.md5(json.dumps(phrases).encode()).hexdigest()
        cache_key = f"domain_detection_{token_hash}_{phrase_hash}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.debug("Using cached domain detection result")
            return cached_result

        # Count domain-specific term matches
        domain_counts = {"mathematics": 0, "programming": 0, "physics": 0}

        # Check tokens against domain terms
        all_tokens = tokens + [word for phrase in phrases for word in phrase.split()]
        for token in all_tokens:
            token_lower = token.lower()

            # Check against domain synonyms
            for domain, synonym_dict in self.domain_synonyms.items():
                if token_lower in synonym_dict:
                    domain_counts[domain] += 2  # Direct match
                else:
                    # Check if token appears as a synonym
                    for term, synonyms in synonym_dict.items():
                        if token_lower in [syn.lower() for syn in synonyms]:
                            domain_counts[domain] += 1  # Synonym match

            # Check against concept hierarchy
            for domain, hierarchy in self.concept_hierarchy.items():
                if token_lower in hierarchy:
                    domain_counts[domain] += 2  # Direct match with broader concept
                if token_lower in hierarchy.get("_reverse_mappings", {}):
                    domain_counts[domain] += 1  # Match with narrower concept

        # Check phrases against educational phrases
        for phrase in phrases:
            phrase_lower = phrase.lower()
            for domain, edu_phrases in self.educational_phrases.items():
                for edu_phrase, weight in edu_phrases:
                    if phrase_lower in edu_phrase or edu_phrase in phrase_lower:
                        domain_counts[domain] += int(weight * 10)  # Weight the match

        # Use database to check for common domain concepts if available
        try:
            db_domain_scores = self._query_domain_term_frequency(all_tokens)
            for domain, score in db_domain_scores.items():
                domain_counts[domain] += score
        except Exception as e:
            logger.warning(f"Error querying domain term frequency: {e}")

        # Find the domain with the highest score
        if sum(domain_counts.values()) == 0:
            return None  # No clear domain

        max_domain = max(domain_counts.items(), key=lambda x: x[1])

        # Only return if the score is above a threshold and there's a clear winner
        if max_domain[1] > 2:
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_domains) > 1 and sorted_domains[0][1] > sorted_domains[1][1] * 1.5:
                # Cache the result
                self.cache.set(cache_key, max_domain[0], ttl=3600)  # Cache for 1 hour
                return max_domain[0]

        # Cache the result (None in this case)
        self.cache.set(cache_key, None, ttl=3600)  # Cache for 1 hour
        return None  # No clear winner

    def _expand_query(self, tokens: List[str], phrases: List[str], domain: Optional[str]) -> List[str]:
        """
        Expand the query with domain-specific related terms using database knowledge.

        Args:
            tokens: Query tokens
            phrases: Exact phrases in the query
            domain: Detected or specified domain

        Returns:
            List of expanded terms
        """
        # Create cache key
        token_hash = hashlib.md5(json.dumps(tokens).encode()).hexdigest()
        phrase_hash = hashlib.md5(json.dumps(phrases).encode()).hexdigest()
        cache_key = f"expand_query_{token_hash}_{phrase_hash}_{domain}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.debug("Using cached query expansion result")
            return cached_result

        expanded_terms = set()

        # Process all tokens and phrases
        all_tokens = tokens + [word for phrase in phrases for word in phrase.split()]

        # First try to get expansion terms from database
        if self.db_context and hasattr(self.db_context, 'search_repository'):
            try:
                # In a real implementation, this would query a synonyms table
                # For now, we use hardcoded data
                logger.debug("Would use database for query expansion in production")
            except Exception as e:
                logger.warning(f"Error getting expansion terms from database: {e}")

        # If domain is specified, use domain-specific expansion
        if domain and domain in self.domain_synonyms:
            # Add synonyms
            synonym_dict = self.domain_synonyms[domain]
            for token in all_tokens:
                token_lower = token.lower()
                # Add direct synonyms
                if token_lower in synonym_dict:
                    expanded_terms.update([term.lower() for term in synonym_dict[token_lower]])

            # Add hierarchical concepts
            hierarchy = self.concept_hierarchy.get(domain, {})
            reverse_mappings = hierarchy.get("_reverse_mappings", {})

            for token in all_tokens:
                token_lower = token.lower()
                # Add narrower concepts for broader concept
                if token_lower in hierarchy and token_lower != "_reverse_mappings":
                    expanded_terms.update([term.lower() for term in hierarchy[token_lower]])
                # Add broader concepts for narrower concept
                if token_lower in reverse_mappings:
                    expanded_terms.update([term.lower() for term in reverse_mappings[token_lower]])

            # Add terms from educational phrases
            edu_phrases = self.educational_phrases.get(domain, [])
            for token in all_tokens:
                token_lower = token.lower()
                for edu_phrase, weight in edu_phrases:
                    # If token is part of an educational phrase, add the whole phrase
                    if token_lower in edu_phrase.lower().split() and weight >= 0.7:
                        expanded_terms.add(edu_phrase.lower())
        else:
            # If domain is not specified, use more conservative expansion
            # Add synonyms from all domains (with lower priority)
            for token in all_tokens:
                token_lower = token.lower()
                for domain, synonym_dict in self.domain_synonyms.items():
                    if token_lower in synonym_dict:
                        # Add only the most relevant synonyms when domain is uncertain
                        top_synonyms = synonym_dict[token_lower][:2]  # Limit to top 2
                        expanded_terms.update([term.lower() for term in top_synonyms])

        # Remove the original tokens from expanded terms
        expanded_terms = expanded_terms - set([token.lower() for token in all_tokens])

        # Remove stopwords
        expanded_terms = set([term for term in expanded_terms if term not in self.stopwords])

        # Limit to reasonable number of expansions to avoid query explosion
        result = list(expanded_terms)
        if len(result) > 10:
            result = result[:10]

        # Cache the result
        self.cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

        return result

    def _query_domain_term_frequency(self, tokens: List[str]) -> Dict[str, int]:
        """
        Query the database to check how frequently tokens appear in different domains.

        Args:
            tokens: Query tokens

        Returns:
            Dictionary mapping domains to frequency scores
        """
        domain_scores = {"mathematics": 0, "programming": 0, "physics": 0}

        if not tokens:
            return domain_scores

        # If using DB context
        if self.db_context and hasattr(self.db_context, 'db_manager'):
            try:
                # Create cache key
                token_hash = hashlib.md5(json.dumps(tokens).encode()).hexdigest()
                cache_key = f"domain_term_freq_{token_hash}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug("Using cached domain term frequency result")
                    return cached_result

                for token in tokens:
                    token_lower = token.lower()

                    # Check concepts table
                    query = """
                    SELECT domain, COUNT(*) as term_count
                    FROM concepts
                    WHERE normalized_text LIKE ? OR text LIKE ?
                    GROUP BY domain
                    """
                    results = self.db_context.db_manager.execute_query(
                        query, (f"%{token_lower}%", f"%{token_lower}%")
                    )

                    for row in results:
                        domain, count = row["domain"], row["term_count"]
                        if domain in domain_scores:
                            domain_scores[domain] += count

                # Cache the result
                self.cache.set(cache_key, domain_scores, ttl=3600)  # Cache for 1 hour

            except Exception as e:
                logger.warning(f"Error querying domain term frequency via DB context: {e}")

        # Fallback to legacy direct DB connection
        elif hasattr(self, 'db_path') and self.db_path:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
                cursor = conn.cursor()

                for token in tokens:
                    token_lower = token.lower()

                    # Check concepts table
                    cursor.execute(
                        """
                        SELECT domain, COUNT(*) as term_count
                        FROM concepts
                        WHERE normalized_text LIKE ? OR text LIKE ?
                        GROUP BY domain
                        """,
                        (f"%{token_lower}%", f"%{token_lower}%")
                    )

                    rows = cursor.fetchall()
                    for row in rows:
                        domain, count = row["domain"], row["term_count"]
                        if domain in domain_scores:
                            domain_scores[domain] += count

                conn.close()

            except Exception as e:
                logger.warning(f"Error querying domain term frequency via direct connection: {e}")

        return domain_scores

    @time_function(threshold_ms=500)
    def suggest_related_searches(self, query_text: str, domain: Optional[str] = None) -> List[str]:
        """
        Generate suggestions for related searches based on the current query with caching.

        Args:
            query_text: Original query text
            domain: Optional domain to focus suggestions

        Returns:
            List of suggested related searches
        """
        # Create cache key
        cache_key = f"related_searches_{hashlib.md5(query_text.encode()).hexdigest()}_{domain}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.debug("Using cached related searches")
            return cached_result

        suggestions = []

        # Process the query
        processed_query = self.process_query(query_text, domain)
        detected_domain = processed_query.get("domain") or domain

        # Get tokens and expanded terms
        tokens = processed_query.get("tokens", [])
        expanded_terms = processed_query.get("expanded_terms", [])

        if not tokens:
            return suggestions

        # Generate suggestions based on expanded terms
        for expanded_term in expanded_terms[:3]:  # Use top 3 expanded terms
            # Combine with original query
            if len(tokens) <= 2:  # For short queries, add expanded terms
                suggestions.append(f"{query_text} {expanded_term}")
            else:  # For longer queries, replace one token
                for i, token in enumerate(tokens):
                    modified_tokens = tokens.copy()
                    modified_tokens[i] = expanded_term
                    suggestions.append(" ".join(modified_tokens))

        # Generate domain-specific suggestions
        if detected_domain:
            # Add domain context for specificity
            if detected_domain == "mathematics":
                context_terms = ["theorem", "proof", "definition", "formula", "example"]
            elif detected_domain == "programming":
                context_terms = ["tutorial", "example", "implementation", "syntax", "code"]
            elif detected_domain == "physics":
                context_terms = ["law", "theory", "experiment", "equation", "example"]
            else:
                context_terms = ["definition", "explanation", "example"]

            # Add a random context term
            for context in random.sample(context_terms, min(2, len(context_terms))):
                suggestions.append(f"{query_text} {context}")

        # Try to get suggestions from database
        try:
            db_suggestions = self._query_related_searches(query_text, detected_domain)
            suggestions.extend(db_suggestions)
        except Exception as e:
            logger.warning(f"Error querying related searches: {e}")

        # Remove duplicates and limit to reasonable number
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.lower() != query_text.lower() and suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)

        # Limit to 5 suggestions
        result = unique_suggestions[:5]

        # Cache the result
        self.cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

        return result

    def _query_related_searches(self, query_text: str, domain: Optional[str] = None) -> List[str]:
        """
        Query the database for related search suggestions.

        Args:
            query_text: Original query text
            domain: Optional domain to focus suggestions

        Returns:
            List of related search suggestions from database
        """
        suggestions = []

        # If using DB context
        if self.db_context and hasattr(self.db_context, 'db_manager'):
            try:
                # Find concepts similar to the query
                query_terms = re.sub(r'[^\w\s]', ' ', query_text.lower()).split()
                query_terms = [term for term in query_terms if term not in self.stopwords and len(term) > 2]

                if not query_terms:
                    return suggestions

                # Build a fuzzy match query
                like_conditions = []
                params = []

                for term in query_terms:
                    like_conditions.append("text LIKE ?")
                    params.append(f"%{term}%")

                sql = f"""
                SELECT text
                FROM concepts
                WHERE {" OR ".join(like_conditions)}
                """

                if domain:
                    sql += " AND domain = ?"
                    params.append(domain)

                sql += " LIMIT 5"

                concepts = self.db_context.db_manager.execute_query(sql, tuple(params))

                # Add concept text as suggestions
                for row in concepts:
                    concept_text = row["text"]
                    if concept_text.lower() != query_text.lower():
                        suggestions.append(concept_text)

            except Exception as e:
                logger.warning(f"Error querying related searches via DB context: {e}")

        # Fallback to legacy direct DB connection
        elif hasattr(self, 'db_path') and self.db_path:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
                cursor = conn.cursor()

                # Find concepts similar to the query
                query_terms = re.sub(r'[^\w\s]', ' ', query_text.lower()).split()
                query_terms = [term for term in query_terms if term not in self.stopwords and len(term) > 2]

                if not query_terms:
                    return suggestions

                # Build a fuzzy match query
                like_conditions = []
                params = []

                for term in query_terms:
                    like_conditions.append("text LIKE ?")
                    params.append(f"%{term}%")

                sql = f"""
                SELECT text
                FROM concepts
                WHERE {" OR ".join(like_conditions)}
                """

                if domain:
                    sql += " AND domain = ?"
                    params.append(domain)

                sql += " LIMIT 5"

                cursor.execute(sql, params)
                concepts = cursor.fetchall()

                # Add concept text as suggestions
                for row in concepts:
                    concept_text = row["text"]
                    if concept_text.lower() != query_text.lower():
                        suggestions.append(concept_text)

                conn.close()

            except Exception as e:
                logger.warning(f"Error querying related searches via direct connection: {e}")

        return suggestions

# Helper functions for SearchEngine integration

def create_optimized_fts_query(processed_query: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Create an optimized FTS query string and parameters from processed query.

    Args:
        processed_query: Processed query dictionary

    Returns:
        Tuple of (query string, parameters)
    """
    query_components = []
    params = []

    # Add original tokens with higher weight
    for token in processed_query.get("stemmed_tokens", []):
        query_components.append(f"{token}*")

    # Add exact phrases
    for phrase in processed_query.get("stemmed_phrases", []):
        if phrase:
            query_components.append(f"\"{phrase}\"")

    # Add expanded terms with lower weight
    for term in processed_query.get("expanded_terms", []):
        stemmed_term = term
        if "stemmer" in processed_query:
            stemmed_term = processed_query["stemmer"].stem(term)
        query_components.append(f"{stemmed_term}")

    # Create the final query string
    query_string = " OR ".join(query_components)

    return query_string, params

def enhance_search_params(
    query: Dict[str, Any],
    processor: QueryProcessor
) -> Dict[str, Any]:
    """
    Enhance search parameters using query processor.

    Args:
        query: Original search query dictionary
        processor: QueryProcessor instance

    Returns:
        Enhanced query dictionary
    """
    enhanced_query = query.copy()

    # Extract original text and domain
    query_text = query.get("original_text", "")
    domain = query.get("domain")

    if query_text:
        # Process the query
        processed_query = processor.process_query(query_text, domain)

        # Update the query
        enhanced_query["processed_query"] = processed_query

        # If domain was detected but not specified, add it
        if not domain and processed_query.get("domain"):
            enhanced_query["domain"] = processed_query["domain"]

        # Add related search suggestions
        enhanced_query["related_searches"] = processor.suggest_related_searches(query_text, domain)

    return enhanced_query
