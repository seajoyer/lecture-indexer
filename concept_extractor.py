"""
Concept extractor implementation for the Lecture Video Content Indexer.
This module provides high-level functions to extract, analyze, and visualize
concepts from educational videos.
"""

import os
import logging
import uuid
import json
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Import project modules
from data_pipeline import DataPipeline
from transcript_processor import TranscriptProcessor
try:
    from data_access import get_data_access
    HAS_DATA_ACCESS = True
except ImportError:
    HAS_DATA_ACCESS = False
    logging.warning("data_access module not available, some functionality will be limited")

try:
    from concept_signature_generator import RelationshipGraph, ConceptSignature, ConceptExtractor
    HAS_CONCEPT_GENERATOR = True
except ImportError:
    HAS_CONCEPT_GENERATOR = False
    logging.warning("concept_signature_generator module not available, some functionality will be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptAnalyzer:
    """
    ConceptAnalyzer provides high-level functions to extract, analyze, and visualize
    concepts from educational videos.
    """

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        """
        Initialize the concept analyzer.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_pipeline = DataPipeline(self.config)
        self.transcript_processor = TranscriptProcessor()

        # Initialize data access if available
        self.data_access = get_data_access() if HAS_DATA_ACCESS else None

        logger.info("ConceptAnalyzer initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        # Try to load configuration from file
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate and fill defaults
            if not config:
                config = {}

            # Ensure required fields
            if "youtube_api_key" not in config:
                config["youtube_api_key"] = os.environ.get("YOUTUBE_API_KEY", "")

            if "output_dir" not in config:
                config["output_dir"] = "data/processed"

            return config
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_path}: {e}")

            # Return default configuration
            return {
                "youtube_api_key": os.environ.get("YOUTUBE_API_KEY", ""),
                "output_dir": "data/processed"
            }

    def process_video(self, video_url: str, language_preference: List[str] = ['en', 'ru']) -> Dict[str, Any]:
        """
        Process a video and extract concepts.

        Args:
            video_url: YouTube video URL
            language_preference: List of language codes in order of preference

        Returns:
            Processing result dictionary
        """
        return self.data_pipeline.process_video(video_url, language_preference)

    def extract_concepts(self, video_id: str) -> Dict[str, Any]:
        """
        Extract concepts from a processed video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with extracted concepts
        """
        # Check if we have data access
        if self.data_access:
            # Get video data
            video = self.data_access.get_video(video_id)
            if not video:
                return {"status": "error", "message": f"Video {video_id} not found"}

            # Get concepts for video
            concepts = self.data_access.get_concepts_for_video(video_id)

            # Format results
            return {
                "status": "success",
                "video_id": video_id,
                "title": video.get("title", ""),
                "domain": video.get("domain", "unknown"),
                "language": video.get("language", "en"),
                "theory_practice_ratio": video.get("theory_practice_ratio", 0.5),
                "concepts": {
                    "theoretical": [c for c in concepts if c.get("concept_class", "theoretical") == "theoretical"],
                    "practical": [c for c in concepts if c.get("concept_class", "theoretical") == "practical"]
                },
                "total_concepts": len(concepts)
            }

        # If no data access, try to get from cache
        try:
            from cache_manager import cache_get

            # Get cached processed result
            cache_key = f"processed_video_{video_id}"
            result = cache_get("video", cache_key)

            if not result:
                return {"status": "error", "message": "Video data not available in cache"}

            # Extract concepts from cached data
            domain_features = result.get("domain_features", {})
            metadata = result.get("metadata", {})

            # Format results
            return {
                "status": "success",
                "video_id": video_id,
                "title": metadata.get("title", ""),
                "domain": metadata.get("domain", "unknown"),
                "language": result.get("transcript", {}).get("language", "en"),
                "theory_practice_ratio": result.get("theory_practice_results", {}).get("theory_practice_ratio", 0.5),
                "concepts": {
                    "theoretical": domain_features.get("theoretical_concepts", []),
                    "practical": domain_features.get("practical_concepts", [])
                },
                "total_concepts": len(domain_features.get("key_concepts", []))
            }
        except ImportError:
            return {"status": "error", "message": "Cache manager not available"}

    def build_concept_graph(self, video_id: str) -> Dict[str, Any]:
        """
        Build a concept graph for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with concept graph data
        """
        return self.data_pipeline.build_concept_graph_from_video(video_id)

    def get_related_concepts(self, concept_text: str, domain: str = None, language: str = None) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept.

        Args:
            concept_text: The concept text
            domain: Optional domain filter
            language: Optional language filter

        Returns:
            List of related concepts
        """
        return self.data_pipeline.extract_related_concepts(concept_text, domain, language)

    def search_concepts(self, query: str, domain: str = None, language: str = None) -> Dict[str, Any]:
        """
        Search for concepts matching a query.

        Args:
            query: Search query
            domain: Optional domain filter
            language: Optional language filter

        Returns:
            Dictionary with search results
        """
        if not self.data_access:
            return {"status": "error", "message": "Search requires data_access module"}

        # Prepare search parameters
        search_params = {
            "original_text": query,
            "filters": {},
            "pagination": {"offset": 0, "limit": 20}
        }

        # Add domain and language filters if provided
        if domain:
            search_params["domain"] = domain

        if language:
            search_params["language"] = language

        # Execute search
        results = self.data_access.search(search_params)

        # Format results
        return {
            "status": "success",
            "query": query,
            "domain": domain,
            "language": language,
            "results": results.get("results", []),
            "total_results": results.get("totalResults", 0),
            "theoretical_results": results.get("theoreticalResults", 0),
            "practical_results": results.get("practicalResults", 0)
        }

    def analyze_video_concepts(self, video_id: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of concepts in a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with video concept analysis
        """
        # Extract concepts
        concepts_data = self.extract_concepts(video_id)
        if concepts_data.get("status") != "success":
            return concepts_data

        # Build concept graph
        graph_data = self.build_concept_graph(video_id)

        # Get additional video data if available
        video_data = {}
        if self.data_access:
            video = self.data_access.get_video(video_id)
            if video:
                video_data = dict(video)

        # Perform analysis
        theoretical_concepts = concepts_data.get("concepts", {}).get("theoretical", [])
        practical_concepts = concepts_data.get("concepts", {}).get("practical", [])

        # Count concept types
        multi_word_count = sum(1 for c in theoretical_concepts + practical_concepts
                              if len(c.get("text", "").split()) > 1)

        single_word_count = concepts_data.get("total_concepts", 0) - multi_word_count

        # Identify most frequent concepts
        all_concepts = theoretical_concepts + practical_concepts
        all_concepts.sort(key=lambda x: x.get("frequency", 0), reverse=True)
        top_concepts = all_concepts[:10] if len(all_concepts) >= 10 else all_concepts

        # Calculate average concept scores
        avg_theoretical_score = sum(c.get("score", 0) for c in theoretical_concepts) / max(len(theoretical_concepts), 1)
        avg_practical_score = sum(c.get("score", 0) for c in practical_concepts) / max(len(practical_concepts), 1)

        return {
            "status": "success",
            "video_id": video_id,
            "title": video_data.get("title", concepts_data.get("title", "")),
            "domain": concepts_data.get("domain", "unknown"),
            "language": concepts_data.get("language", "en"),
            "theory_practice_ratio": concepts_data.get("theory_practice_ratio", 0.5),
            "concepts_count": {
                "total": concepts_data.get("total_concepts", 0),
                "theoretical": len(theoretical_concepts),
                "practical": len(practical_concepts),
                "multi_word": multi_word_count,
                "single_word": single_word_count
            },
            "top_concepts": top_concepts,
            "avg_scores": {
                "theoretical": avg_theoretical_score,
                "practical": avg_practical_score
            },
            "graph_stats": {
                "nodes": len(graph_data.get("graph_data", {}).get("concepts", {})),
                "edges": len(graph_data.get("graph_data", {}).get("edges", []))
            }
        }

    def compare_videos(self, video_ids: List[str]) -> Dict[str, Any]:
        """
        Compare concepts across multiple videos.

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            Dictionary with comparison results
        """
        if not video_ids:
            return {"status": "error", "message": "No video IDs provided"}

        # Extract concepts for each video
        video_concepts = {}
        for video_id in video_ids:
            concepts_data = self.extract_concepts(video_id)
            if concepts_data.get("status") == "success":
                video_concepts[video_id] = concepts_data

        if not video_concepts:
            return {"status": "error", "message": "No valid videos found"}

        # Find common concepts
        common_concepts = self._find_common_concepts(video_concepts)

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(video_concepts)

        # Get video metadata
        video_metadata = {}
        for video_id, concepts_data in video_concepts.items():
            video_metadata[video_id] = {
                "title": concepts_data.get("title", ""),
                "domain": concepts_data.get("domain", "unknown"),
                "language": concepts_data.get("language", "en"),
                "theory_practice_ratio": concepts_data.get("theory_practice_ratio", 0.5),
                "total_concepts": concepts_data.get("total_concepts", 0)
            }

        return {
            "status": "success",
            "videos": video_metadata,
            "common_concepts": common_concepts,
            "similarity_matrix": similarity_matrix
        }

    def _find_common_concepts(self, video_concepts: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find concepts common to multiple videos.

        Args:
            video_concepts: Dictionary mapping video IDs to concept data

        Returns:
            List of common concepts with metadata
        """
        # Extract all concepts by text
        concept_texts = defaultdict(list)

        for video_id, data in video_concepts.items():
            theoretical = data.get("concepts", {}).get("theoretical", [])
            practical = data.get("concepts", {}).get("practical", [])

            for concept in theoretical + practical:
                text = concept.get("text", "").lower()
                if text:
                    concept_info = concept.copy()
                    concept_info["video_id"] = video_id
                    concept_texts[text].append(concept_info)

        # Find concepts present in multiple videos
        common_concepts = []

        for text, occurrences in concept_texts.items():
            video_count = len(set(c.get("video_id") for c in occurrences))

            if video_count > 1:  # Present in at least 2 videos
                # Aggregate information
                domains = set(c.get("domain") for c in occurrences if c.get("domain"))
                languages = set(c.get("language") for c in occurrences if c.get("language"))
                videos = set(c.get("video_id") for c in occurrences)

                # Calculate average score
                avg_score = sum(c.get("score", 0) for c in occurrences) / len(occurrences)

                # Determine if theoretical or practical
                theoretical_count = sum(1 for c in occurrences if c.get("theoretical", True))
                practical_count = len(occurrences) - theoretical_count
                is_theoretical = theoretical_count >= practical_count

                common_concepts.append({
                    "text": text,
                    "video_count": video_count,
                    "videos": list(videos),
                    "domains": list(domains),
                    "languages": list(languages),
                    "avg_score": avg_score,
                    "theoretical": is_theoretical,
                    "concept_class": "theoretical" if is_theoretical else "practical"
                })

        # Sort by video count and score
        common_concepts.sort(key=lambda x: (x.get("video_count", 0), x.get("avg_score", 0)), reverse=True)

        return common_concepts

    def _calculate_similarity_matrix(self, video_concepts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate similarity matrix between videos based on shared concepts.

        Args:
            video_concepts: Dictionary mapping video IDs to concept data

        Returns:
            Dictionary with similarity scores between videos
        """
        # Extract concept sets for each video
        video_concept_sets = {}

        for video_id, data in video_concepts.items():
            theoretical = data.get("concepts", {}).get("theoretical", [])
            practical = data.get("concepts", {}).get("practical", [])

            # Extract concept texts
            concept_texts = set()
            for concept in theoretical + practical:
                text = concept.get("text", "").lower()
                if text:
                    concept_texts.add(text)

            video_concept_sets[video_id] = concept_texts

        # Calculate Jaccard similarity between all video pairs
        similarity_matrix = defaultdict(dict)

        video_ids = list(video_concept_sets.keys())
        for i, video1 in enumerate(video_ids):
            for j, video2 in enumerate(video_ids):
                if i == j:
                    # Same video, similarity = 1.0
                    similarity_matrix[video1][video2] = 1.0
                elif j > i:
                    # Calculate Jaccard similarity
                    set1 = video_concept_sets[video1]
                    set2 = video_concept_sets[video2]

                    if not set1 or not set2:
                        similarity = 0.0
                    else:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        similarity = intersection / union if union > 0 else 0.0

                    similarity_matrix[video1][video2] = similarity
                    similarity_matrix[video2][video1] = similarity

        return similarity_matrix


# Main functions for command-line usage

def process_video(video_url: str, output_path: str = None):
    """
    Process a video and save results to a file.

    Args:
        video_url: YouTube video URL
        output_path: Path to save results (optional)
    """
    analyzer = ConceptAnalyzer()
    result = analyzer.process_video(video_url)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")

    print(f"Processed video: {result.get('video_id')}")
    print(f"Title: {result.get('metadata', {}).get('title', '')}")
    print(f"Domain: {result.get('metadata', {}).get('domain', 'unknown')}")
    print(f"Theory/Practice Ratio: {result.get('theory_practice_results', {}).get('theory_practice_ratio', 0.5):.2f}")
    print(f"Extracted {len(result.get('domain_features', {}).get('key_concepts', []))} concepts")


def analyze_concepts(video_id: str, output_path: str = None):
    """
    Analyze concepts in a video and save results to a file.

    Args:
        video_id: YouTube video ID
        output_path: Path to save results (optional)
    """
    analyzer = ConceptAnalyzer()
    result = analyzer.analyze_video_concepts(video_id)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")

    print(f"Analyzed video: {video_id}")
    print(f"Title: {result.get('title', '')}")
    print(f"Domain: {result.get('domain', 'unknown')}")
    print(f"Theory/Practice Ratio: {result.get('theory_practice_ratio', 0.5):.2f}")
    print(f"Total Concepts: {result.get('concepts_count', {}).get('total', 0)}")
    print(f"Theoretical Concepts: {result.get('concepts_count', {}).get('theoretical', 0)}")
    print(f"Practical Concepts: {result.get('concepts_count', {}).get('practical', 0)}")

    print("\nTop Concepts:")
    for concept in result.get('top_concepts', [])[:5]:
        print(f"- {concept.get('text', '')}")


def compare_videos(video_ids: List[str], output_path: str = None):
    """
    Compare concepts across multiple videos and save results to a file.

    Args:
        video_ids: List of YouTube video IDs
        output_path: Path to save results (optional)
    """
    analyzer = ConceptAnalyzer()
    result = analyzer.compare_videos(video_ids)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")

    print(f"Compared {len(video_ids)} videos")
    print(f"Found {len(result.get('common_concepts', []))} common concepts")

    print("\nTop Common Concepts:")
    for concept in result.get('common_concepts', [])[:5]:
        print(f"- {concept.get('text', '')} (found in {concept.get('video_count', 0)} videos)")

    print("\nVideo Similarity Matrix:")
    similarity = result.get('similarity_matrix', {})
    for video1 in similarity:
        for video2, score in similarity.get(video1, {}).items():
            if video1 < video2:  # Only print each pair once
                print(f"- {video1} vs {video2}: {score:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Concept extractor for educational videos")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process video command
    process_parser = subparsers.add_parser("process", help="Process a video")
    process_parser.add_argument("url", help="YouTube video URL")
    process_parser.add_argument("--output", "-o", help="Output file path")

    # Analyze concepts command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze concepts in a video")
    analyze_parser.add_argument("video_id", help="YouTube video ID")
    analyze_parser.add_argument("--output", "-o", help="Output file path")

    # Compare videos command
    compare_parser = subparsers.add_parser("compare", help="Compare concepts across videos")
    compare_parser.add_argument("video_ids", nargs="+", help="YouTube video IDs")
    compare_parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    if args.command == "process":
        process_video(args.url, args.output)
    elif args.command == "analyze":
        analyze_concepts(args.video_id, args.output)
    elif args.command == "compare":
        compare_videos(args.video_ids, args.output)
    else:
        parser.print_help()
