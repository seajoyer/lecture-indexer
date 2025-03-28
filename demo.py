#!/usr/bin/env python3
"""
Enhanced demonstration script for Lecture Video Content Indexer.
Tests the core functionality with improved language handling and visualization.
"""

import os
import sys
import re
import logging
import argparse
import time
import json
import textwrap
from typing import Dict, List, Any
import colorama
from colorama import Fore, Style, Back
from collections import deque, defaultdict, Counter

# Import enhanced components
from youtube_extractor import YouTubeExtractor
from data_pipeline import DataPipeline
from search_engine import SearchEngine
from data_access import get_data_access
from cache_manager import cache_clear, get_cache_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo")

try:
    from concept_signature_generator import (
        get_concept_signature_generator,
        ConceptSignature,
        RelationshipGraph
    )
    CONCEPT_SIGNATURES_AVAILABLE = True
except ImportError:
    CONCEPT_SIGNATURES_AVAILABLE = False
    # Create dummy functions for graceful degradation
    def get_concept_signature_generator(*args, **kwargs): return None
    ConceptSignature = type('ConceptSignature', (), {})
    RelationshipGraph = type('RelationshipGraph', (), {})

# Initialize colorama
colorama.init(autoreset=True)

def main():
    """Main function for the demonstration script."""
    parser = argparse.ArgumentParser(description='Lecture Video Content Indexer Demo')

    # Source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('url', nargs='?', help='YouTube video URL or playlist URL')
    source_group.add_argument('--video', help='Show details for specific video ID (already indexed)')
    source_group.add_argument('--list-concepts', action='store_true', help='List all indexed concepts')
    source_group.add_argument('--learning-path', action='store_true', help='Generate a learning path from concepts')

    # API key handling
    parser.add_argument('--api-key', help='YouTube API key (NOT RECOMMENDED - use environment variable instead)')

    # Search options
    parser.add_argument('--search', help='Search query after processing')
    parser.add_argument('--theory-ratio', type=float, default=0.5,
                      help='Theory/practice ratio for search (0-1, 1=all theoretical)')

    # Playlist options
    parser.add_argument('--no-limit', action='store_true', help='Process all videos in a playlist without limit')
    parser.add_argument('--max-videos', type=int, default=5,
                      help='Maximum number of videos to process from a playlist (ignored with --no-limit)')

    # Filtering options
    parser.add_argument('--filter-domain', choices=['mathematics', 'programming', 'physics'],
                      help='Filter by domain')
    parser.add_argument('--filter-video', help='Filter to a specific video ID')
    parser.add_argument('--filter-playlist', help='Filter to a specific playlist ID')
    parser.add_argument('--filter-language', choices=['en', 'ru'], help='Filter by language')

    # Learning path options
    parser.add_argument('--concepts', nargs='+', help='Concept IDs for learning path generation')

    # Performance options
    parser.add_argument('--cache-stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--optimize-db', action='store_true', help='Optimize database before running')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all caches before running')

    # Language options
    parser.add_argument('--language', choices=['en', 'ru', 'auto'], default='auto',
                      help='Preferred language for processing (auto=detect)')

    # Debug option
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    # Output options
    parser.add_argument('--output-json', help='Save results to JSON file')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')

    # MLCS related
    parser.add_argument('--concept-signatures', action='store_true',
                     help='Show concept signatures for indexed concepts')
    parser.add_argument('--relationship-graph', action='store_true',
                     help='Display concept relationship graph')
    parser.add_argument('--export-graph', help='Export concept relationship graph to specified file path')
    parser.add_argument('--analyze-concepts', action='store_true',
                     help='Analyze concept relationships across all videos')

    args = parser.parse_args()

    # Handle concept signature commands
    if args.concept_signatures or args.relationship_graph or args.export_graph or args.analyze_concepts:
        if not CONCEPT_SIGNATURES_AVAILABLE:
            print(f"{Fore.RED}Concept signature functionality is not available.{Style.RESET_ALL}")
            print("Make sure concept_signature_generator.py is properly installed.")
            sys.exit(1)

        # Get YouTube API key
        api_key = os.environ.get("YOUTUBE_API_KEY", "")
        if args.api_key:
            api_key = args.api_key
            logger.warning("Using API key from command line argument (not secure)")

        # Get configuration for concept generator
        concept_config = {
            "youtube_api_key": api_key,
            "output_dir": "data/processed",
            "index_dir": "data/index"
        }

        # Get the concept signature generator
        concept_generator = get_concept_signature_generator(concept_config)

        if args.concept_signatures:
            display_concept_signatures(concept_generator,
                                      domain_filter=args.filter_domain,
                                      video_filter=args.filter_video,
                                      language=args.filter_language)
            sys.exit(0)

        if args.relationship_graph:
            display_relationship_graph(concept_generator,
                                      domain_filter=args.filter_domain)
            sys.exit(0)

        if args.export_graph:
            export_concept_graph(concept_generator, args.export_graph)
            sys.exit(0)

        if args.analyze_concepts:
            analyze_concept_relationships(concept_generator,
                                        domain_filter=args.filter_domain)
            sys.exit(0)

    # Set color output
    if args.no_color:
        disable_color()

    # Print header
    print_header()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # Create necessary directories
    create_data_directories()

    # Get YouTube API key - use secure handling
    api_key = get_api_key(args)
    if not api_key:
        print(f"{Fore.RED}No YouTube API key provided. Set YOUTUBE_API_KEY environment variable "
              f"or use --api-key option.{Style.RESET_ALL}")
        sys.exit(1)

    # Clear cache if requested
    if args.clear_cache:
        print(f"{Fore.CYAN}Clearing all caches...{Style.RESET_ALL}")
        cache_clear()

    # Load configuration
    config = {
        "youtube_api_key": api_key,
        "output_dir": "data/processed",
        "index_dir": "data/index"
    }

    # Initialize components
    youtube_extractor = YouTubeExtractor(api_key)
    data_pipeline = DataPipeline(config)
    search_engine = SearchEngine(config)
    data_access = get_data_access()

    # Show cache stats if requested
    if args.cache_stats:
        display_cache_stats()

    # Optimize database if requested
    if args.optimize_db:
        print(f"{Fore.CYAN}Optimizing database...{Style.RESET_ALL}")
        search_engine.optimize_database()

    # Determine language preference
    language_preference = determine_language_preference(args.language)

    # If we're just listing concepts
    if args.list_concepts:
        list_indexed_concepts(data_access, domain_filter=args.filter_domain,
                             video_filter=args.filter_video,
                             playlist_filter=args.filter_playlist,
                             language=args.filter_language)
        sys.exit(0)

    # If we're just showing details for a specific video
    if args.video:
        show_video_concepts(search_engine, args.video)
        sys.exit(0)

    # If we're generating a learning path
    if args.learning_path:
        if not args.concepts:
            print(f"{Fore.RED}You must specify concept IDs using --concepts when using --learning-path{Style.RESET_ALL}")
            sys.exit(1)
        generate_learning_path(search_engine, args.concepts, args.theory_ratio, args.filter_domain)
        sys.exit(0)

    # If we're just searching for a term without processing a video
    if args.search and not args.url:
        search_concepts(search_engine, args.search, args.theory_ratio,
                      domain_filter=args.filter_domain,
                      video_filter=args.filter_video,
                      language=args.filter_language)
        sys.exit(0)

    # Process video or playlist
    url = args.url
    if not url:
        # Use a good educational video as default example
        url = "https://www.youtube.com/watch?v=rfscVS0vtbw"  # Python tutorial
        print(f"{Fore.YELLOW}No URL provided, using example: {url}{Style.RESET_ALL}")

    # Check if URL is a playlist
    if is_playlist_url(url):
        # Process playlist
        process_playlist(url, youtube_extractor, data_pipeline, search_engine,
                        None if args.no_limit else args.max_videos,
                        language_preference)

        # If search is requested after playlist processing
        if args.search:
            # Extract playlist ID for filtering
            playlist_id = extract_playlist_id(url)
            search_concepts(search_engine, args.search, args.theory_ratio,
                          domain_filter=args.filter_domain,
                          video_filter=args.filter_video,
                          language=args.filter_language,
                          playlist_filter=args.filter_playlist or playlist_id)

        sys.exit(0)

    # Process single video
    print(f"{Fore.CYAN}Processing video: {url}{Style.RESET_ALL}")
    print(f"Language preference: {', '.join(language_preference)}")
    print("This may take a few minutes...")
    start_time = time.time()

    try:
        # Extract video ID
        is_valid, video_id = youtube_extractor.validate_video_url(url)
        if not is_valid:
            print(f"{Fore.RED}Invalid YouTube URL: {url}{Style.RESET_ALL}")
            sys.exit(1)

        result = data_pipeline.process_video(url, language_preference)
        process_time = time.time() - start_time

        if result.get("status") == "completed":
            display_results(result, process_time)

            # Index the results
            print(f"\n{Fore.CYAN}Indexing video content...{Style.RESET_ALL}")
            index_success = search_engine.index_content(result)

            if index_success:
                print(f"{Fore.GREEN}Successfully indexed video content{Style.RESET_ALL}")

                # Save to JSON if requested
                if args.output_json:
                    save_results_to_json(result, args.output_json)

                # Search if requested
                if args.search:
                    search_concepts(search_engine, args.search, args.theory_ratio,
                                  domain_filter=args.filter_domain,
                                  video_filter=args.filter_video or video_id,
                                  language=args.filter_language)
                else:
                    # Show a default search example
                    domain = result['metadata']['domain']
                    if domain == "physics":
                        search_term = "quantum"
                    elif domain == "mathematics":
                        search_term = "calculus"
                    elif domain == "programming":
                        search_term = "function"
                    else:
                        search_term = domain

                    print(f"\n{Fore.MAGENTA}===== Search Examples ====={Style.RESET_ALL}")
                    print(f"{Fore.CYAN}1. Searching for theoretical concepts:{Style.RESET_ALL}")
                    search_concepts(search_engine, search_term, 0.8, video_filter=video_id)

                    print(f"\n{Fore.CYAN}2. Searching for practical examples:{Style.RESET_ALL}")
                    search_concepts(search_engine, "example", 0.2, video_filter=video_id)
            else:
                print(f"{Fore.RED}Failed to index video content{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Error processing video: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error running demo: {e}{Style.RESET_ALL}")
        if args.debug:
            import traceback
            traceback.print_exc()

# Add these new functions for concept signature capabilities
def display_concept_signatures(
    concept_generator,
    domain_filter=None,
    video_filter=None,
    language=None
):
    """Display concept signatures for indexed concepts."""
    print(f"{Fore.MAGENTA}===== Concept Signatures ====={Style.RESET_ALL}")

    # Apply filters for display
    filter_str = []
    if domain_filter:
        filter_str.append(f"domain={domain_filter}")
    if video_filter:
        filter_str.append(f"video={video_filter}")
    if language:
        filter_str.append(f"language={language}")

    if filter_str:
        print(f"Filters: {', '.join(filter_str)}")

    # Get relationship graph
    graph = concept_generator.relationship_graph

    # Filter concepts based on criteria
    filtered_concepts = {}
    for concept_id, concept in graph.concepts.items():
        # Apply domain filter
        if domain_filter and concept.domain != domain_filter:
            continue

        # Apply language filter
        if language and concept.language != language:
            continue

        # Apply video filter (requires parsing video_id from concept_id if stored that way)
        if video_filter and video_filter not in concept_id:
            continue

        filtered_concepts[concept_id] = concept

    if not filtered_concepts:
        print(f"{Fore.YELLOW}No concept signatures found matching the filters.{Style.RESET_ALL}")
        return

    # Group concepts by domain
    by_domain = defaultdict(list)
    for concept in filtered_concepts.values():
        by_domain[concept.domain].append(concept)

    # Display counts
    total = len(filtered_concepts)
    print(f"Found {total} concept signatures")

    # Display theoretical vs practical counts
    theoretical = sum(1 for c in filtered_concepts.values() if c.concept_class == "theoretical")
    practical = sum(1 for c in filtered_concepts.values() if c.concept_class == "practical")

    theory_percent = theoretical / total * 100 if total > 0 else 0
    practice_percent = practical / total * 100 if total > 0 else 0

    # Create visual bars
    theory_bar = get_progress_bar(theory_percent, 40, Fore.BLUE)
    practice_bar = get_progress_bar(practice_percent, 40, Fore.GREEN)

    print(f"{Fore.BLUE}Theoretical: {theory_bar}{Style.RESET_ALL} ({theoretical}, {theory_percent:.1f}%)")
    print(f"{Fore.GREEN}Practical:   {practice_bar}{Style.RESET_ALL} ({practical}, {practice_percent:.1f}%)")

    # Display by domain
    for domain, domain_concepts in by_domain.items():
        domain_color = (Fore.BLUE if domain == 'mathematics' else
                       Fore.GREEN if domain == 'programming' else
                       Fore.YELLOW if domain == 'physics' else
                       Fore.WHITE)

        print(f"\n{domain_color}Domain: {domain.upper()}{Style.RESET_ALL}")
        print(f"Found {len(domain_concepts)} concepts")

        # Sort by hierarchy score (most fundamental first)
        sorted_concepts = sorted(domain_concepts,
                                key=lambda c: c.hierarchy_score,
                                reverse=True)

        # Display top concepts
        for i, concept in enumerate(sorted_concepts[:20], 1):
            # Format confidence as stars
            confidence_stars = "★" * int(concept.confidence * 5)
            confidence_stars = f"{confidence_stars:<5}"

            # Format relationships
            rel_count = len(concept.related_concepts)

            # Format class with color
            if concept.concept_class == "theoretical":
                class_str = f"{Fore.BLUE}theoretical{Style.RESET_ALL}"
            else:
                class_str = f"{Fore.GREEN}practical{Style.RESET_ALL}"

            # Print concept info
            print(f"{i:2}. {concept.text} [{class_str}]")
            print(f"    Confidence: {confidence_stars} ({concept.confidence:.2f})  " +
                  f"Hierarchy: {concept.hierarchy_score:.2f}  " +
                  f"Related: {rel_count}")

            # Show signature pattern if available
            if concept.signature_pattern:
                pattern_str = " ".join(concept.signature_pattern)
                print(f"    Signature: {Fore.CYAN}{pattern_str[:80]}{Style.RESET_ALL}")

            # Show some related concepts if available
            if concept.related_concepts and i <= 10:  # Only for top 10
                related = list(concept.related_concepts.items())[:3]
                rel_strs = []

                for rel_id, rel_data in related:
                    if rel_id in graph.concepts:
                        rel_text = graph.concepts[rel_id].text
                        rel_str = f"{rel_text} ({rel_data['type']}, {rel_data['strength']:.2f})"
                        rel_strs.append(rel_str)

                if rel_strs:
                    print(f"    Related: {', '.join(rel_strs)}")

            print()

        if len(domain_concepts) > 20:
            print(f"... and {len(domain_concepts) - 20} more concepts")

def display_relationship_graph(concept_generator, domain_filter=None):
    """Display concept relationship graph visualization."""
    print(f"{Fore.MAGENTA}===== Concept Relationship Graph ====={Style.RESET_ALL}")

    # Get relationship graph
    graph = concept_generator.relationship_graph

    if not graph.concepts:
        print(f"{Fore.YELLOW}No concepts found in relationship graph.{Style.RESET_ALL}")
        return

    # Apply domain filter
    if domain_filter:
        domain_concepts = graph.domain_clusters.get(domain_filter, set())
        print(f"Filtering to domain: {domain_filter} ({len(domain_concepts)} concepts)")
    else:
        # Use all concepts
        domain_concepts = set(graph.concepts.keys())
        print(f"Showing all domains ({len(domain_concepts)} concepts)")

    # Print statistics
    edge_count = sum(1 for source_id in graph.adjacency_list if source_id in domain_concepts
                    for target_id in graph.adjacency_list[source_id] if target_id in domain_concepts)

    print(f"Graph contains {len(domain_concepts)} concepts and {edge_count} relationships")

    # Count relationship types
    relationship_types = Counter()
    for (source_id, target_id), attrs in graph.edge_attributes.items():
        if source_id in domain_concepts and target_id in domain_concepts:
            rel_type = attrs.get("type", "unknown")
            relationship_types[rel_type] += 1

    print("\nRelationship types:")
    for rel_type, count in relationship_types.most_common():
        print(f"  - {rel_type}: {count}")

    # Calculate connectivity metrics
    if domain_concepts:
        avg_connections = edge_count / len(domain_concepts) if domain_concepts else 0
        print(f"\nAverage connections per concept: {avg_connections:.2f}")

    # Display most central concepts (highest hierarchy score and most connections)
    central_concepts = []
    for concept_id in domain_concepts:
        concept = graph.concepts.get(concept_id)
        if concept:
            # Count outgoing and incoming connections
            outgoing = len(graph.adjacency_list.get(concept_id, set()))
            incoming = sum(1 for src, targets in graph.adjacency_list.items()
                          if concept_id in targets and src in domain_concepts)

            central_concepts.append((
                concept,
                concept.hierarchy_score,
                outgoing + incoming
            ))

    # Sort by combined score
    central_concepts.sort(key=lambda x: (x[1] + x[2]/10), reverse=True)

    print(f"\n{Fore.CYAN}Most central concepts:{Style.RESET_ALL}")
    for i, (concept, hierarchy, connections) in enumerate(central_concepts[:10], 1):
        print(f"{i}. {concept.text} (hierarchy: {hierarchy:.2f}, connections: {connections})")

    # Show some example paths between concepts
    if len(central_concepts) >= 2:
        print(f"\n{Fore.CYAN}Example learning paths between concepts:{Style.RESET_ALL}")

        # Get a few important concepts
        important_concepts = [c[0].concept_id for c in central_concepts[:min(5, len(central_concepts))]]

        # Generate some paths between them
        for i, start_id in enumerate(important_concepts):
            for end_id in important_concepts[i+1:i+2]:  # Just get the next one
                if start_id != end_id:
                    path = find_path_between_concepts(graph, start_id, end_id)

                    if path:
                        start_concept = graph.concepts[start_id].text
                        end_concept = graph.concepts[end_id].text

                        print(f"\nPath from '{start_concept}' to '{end_concept}':")

                        for j, concept_id in enumerate(path, 1):
                            concept = graph.concepts[concept_id]
                            print(f"  {j}. {concept.text}")

def find_path_between_concepts(graph, start_id, end_id, max_depth=5):
    """Find a path between two concepts in the graph using BFS."""
    if start_id not in graph.concepts or end_id not in graph.concepts:
        return None

    # BFS to find path
    queue = deque([(start_id, [start_id])])
    visited = set([start_id])

    while queue:
        (node, path) = queue.popleft()

        # Check path length
        if len(path) > max_depth:
            continue

        # Check if we found the target
        if node == end_id:
            return path

        # Add neighbors to queue
        for neighbor in graph.adjacency_list.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None

def export_concept_graph(concept_generator, file_path):
    """Export concept relationship graph to a file."""
    print(f"{Fore.MAGENTA}===== Exporting Concept Graph ====={Style.RESET_ALL}")

    try:
        # Get relationship graph
        graph = concept_generator.relationship_graph

        # Save to specified path
        graph.save_to_json(file_path)

        print(f"{Fore.GREEN}Successfully exported graph with {len(graph.concepts)} concepts to {file_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error exporting graph: {e}{Style.RESET_ALL}")

def analyze_concept_relationships(concept_generator, domain_filter=None):
    """Analyze concept relationships across all videos."""
    print(f"{Fore.MAGENTA}===== Analyzing Concept Relationships ====={Style.RESET_ALL}")

    # Get relationship graph
    graph = concept_generator.relationship_graph

    # Filter by domain if specified
    if domain_filter:
        domain_concepts = graph.domain_clusters.get(domain_filter, set())
        concepts = {cid: graph.concepts[cid] for cid in domain_concepts if cid in graph.concepts}
        print(f"Analyzing domain: {domain_filter} ({len(concepts)} concepts)")
    else:
        concepts = graph.concepts
        print(f"Analyzing all domains ({len(concepts)} concepts)")

    if not concepts:
        print(f"{Fore.YELLOW}No concepts found for analysis.{Style.RESET_ALL}")
        return

    # Calculate hierarchy scores
    print(f"{Fore.CYAN}Calculating concept hierarchy scores...{Style.RESET_ALL}")
    graph.calculate_all_hierarchy_scores()

    # Find clusters of related concepts
    print(f"{Fore.CYAN}Finding concept clusters...{Style.RESET_ALL}")

    # Simple clustering based on connections
    clusters = find_concept_clusters(graph, concepts)

    # Print cluster information
    print(f"\nFound {len(clusters)} concept clusters:")

    for i, cluster in enumerate(clusters[:10], 1):  # Show top 10 clusters
        # Get central concept (highest hierarchy score)
        central_id = max(cluster, key=lambda cid: concepts[cid].hierarchy_score if cid in concepts else 0)
        central = concepts[central_id]

        # Count theoretical vs practical
        theoretical = sum(1 for cid in cluster if cid in concepts and concepts[cid].concept_class == "theoretical")
        practical = sum(1 for cid in cluster if cid in concepts and concepts[cid].concept_class == "practical")

        print(f"\n{Fore.YELLOW}Cluster {i}: {central.text} (size: {len(cluster)}){Style.RESET_ALL}")
        print(f"  Theoretical: {theoretical}, Practical: {practical}")

        # List top concepts in the cluster
        top_concepts = sorted(
            [concepts[cid] for cid in cluster if cid in concepts],
            key=lambda c: c.hierarchy_score,
            reverse=True
        )[:5]  # Top 5

        print("  Top concepts:")
        for j, concept in enumerate(top_concepts, 1):
            print(f"    {j}. {concept.text} (hierarchy: {concept.hierarchy_score:.2f})")

    if len(clusters) > 10:
        print(f"\n... and {len(clusters) - 10} more clusters")

    # Suggest learning paths through the clusters
    print(f"\n{Fore.CYAN}Suggested learning paths:{Style.RESET_ALL}")

    # Get top clusters
    top_clusters = clusters[:min(3, len(clusters))]

    for i, cluster in enumerate(top_clusters, 1):
        # Get top concept in cluster
        central_id = max(cluster, key=lambda cid: concepts[cid].hierarchy_score if cid in concepts else 0)
        central = concepts[central_id]

        print(f"\nPath {i}: Learning about {central.text}")

        # Generate learning path for this cluster
        path_concepts = graph.generate_learning_path(list(cluster), max_concepts=8)

        for j, concept_id in enumerate(path_concepts, 1):
            if concept_id in concepts:
                concept = concepts[concept_id]
                class_color = Fore.BLUE if concept.concept_class == "theoretical" else Fore.GREEN
                print(f"  {j}. {class_color}{concept.text}{Style.RESET_ALL}")

def find_concept_clusters(graph, concepts, min_connection_strength=0.3):
    """Find clusters of related concepts in the graph."""
    # Create a simplified graph with only strong connections
    simplified_graph = defaultdict(set)

    for src_id, tgt_ids in graph.adjacency_list.items():
        if src_id not in concepts:
            continue

        for tgt_id in tgt_ids:
            if tgt_id not in concepts:
                continue

            # Check connection strength
            edge_attrs = graph.edge_attributes.get((src_id, tgt_id), {})
            strength = edge_attrs.get("strength", 0)

            if strength >= min_connection_strength:
                simplified_graph[src_id].add(tgt_id)
                simplified_graph[tgt_id].add(src_id)  # Make it undirected

    # Find connected components (clusters)
    clusters = []
    visited = set()

    for concept_id in concepts:
        if concept_id in visited:
            continue

        # BFS to find connected component
        cluster = set()
        queue = deque([concept_id])

        while queue:
            node = queue.popleft()

            if node in visited:
                continue

            visited.add(node)
            cluster.add(node)

            # Add neighbors
            for neighbor in simplified_graph.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        if cluster:
            clusters.append(cluster)

    # Sort clusters by size
    clusters.sort(key=len, reverse=True)

    return clusters

def determine_language_preference(language_arg: str) -> List[str]:
    """
    Determine language preference list based on argument.

    Args:
        language_arg: Language argument ('en', 'ru', or 'auto')

    Returns:
        List of language codes in preference order
    """
    if language_arg == 'auto':
        # Default - auto-detect with English and Russian support
        return ['en', 'ru']
    elif language_arg == 'en':
        return ['en', 'ru']  # Prefer English, fall back to Russian
    elif language_arg == 'ru':
        return ['ru', 'en']  # Prefer Russian, fall back to English
    else:
        return ['en', 'ru']  # Default

def disable_color():
    """Disable colored output."""
    global Fore, Back, Style
    # Create empty class for no colors
    class NoColor:
        def __getattr__(self, name):
            return ''

    Fore = NoColor()
    Back = NoColor()
    Style = NoColor()

def display_cache_stats():
    """Display cache statistics."""
    try:
        stats = get_cache_stats()

        print(f"{Fore.MAGENTA}===== Cache Statistics ====={Style.RESET_ALL}")

        for cache_type, cache_stats in stats.items():
            hit_rate = cache_stats.get('hit_rate_percent', 0)
            size = cache_stats.get('size', 0)
            max_size = cache_stats.get('max_size', 0)

            # Color based on hit rate
            rate_color = (Fore.GREEN if hit_rate > 80 else
                         Fore.YELLOW if hit_rate > 50 else
                         Fore.RED)

            # Usage bar
            usage_percent = (size / max_size * 100) if max_size > 0 else 0
            usage_bar = get_progress_bar(usage_percent, 20)

            print(f"{Fore.CYAN}{cache_type.capitalize()} cache:{Style.RESET_ALL}")
            print(f"  Size: {size}/{max_size} items {usage_bar}")
            print(f"  Hit rate: {rate_color}{hit_rate:.1f}%{Style.RESET_ALL}")
            print(f"  Hits: {cache_stats.get('hits', 0)}, Misses: {cache_stats.get('misses', 0)}")
            print()

    except Exception as e:
        print(f"{Fore.RED}Error displaying cache stats: {e}{Style.RESET_ALL}")

def save_results_to_json(result: Dict[str, Any], filename: str):
    """Save processing results to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}Results saved to {filename}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving results to JSON: {e}{Style.RESET_ALL}")

def is_playlist_url(url):
    """Check if URL is a YouTube playlist."""
    return "list=" in url

def extract_playlist_id(url):
    """Extract playlist ID from URL."""
    match = re.search(r'list=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def process_playlist(playlist_url, youtube_extractor, data_pipeline, search_engine, max_videos=None, language_preference=None):
    """
    Process all videos in a YouTube playlist.

    Args:
        playlist_url: URL of the YouTube playlist
        youtube_extractor: YouTubeExtractor instance
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        max_videos: Maximum number of videos to process (None for unlimited)
        language_preference: List of language codes in order of preference
    """
    if language_preference is None:
        language_preference = ['en', 'ru']

    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        print(f"{Fore.RED}Invalid playlist URL: {playlist_url}{Style.RESET_ALL}")
        return

    print(f"{Fore.MAGENTA}Processing videos from playlist ID: {playlist_id}{Style.RESET_ALL}")
    print(f"Language preference: {', '.join(language_preference)}")

    # Get video URLs from playlist
    video_urls = extract_playlist_videos(youtube_extractor, playlist_id, max_videos)

    if not video_urls:
        print(f"{Fore.RED}No videos found in playlist or error accessing playlist{Style.RESET_ALL}")
        return

    limit_str = f" (limited to {max_videos})" if max_videos else " (no limit)"
    print(f"{Fore.GREEN}Found {len(video_urls)} videos in playlist{limit_str}{Style.RESET_ALL}")

    # Save playlist info to database
    save_playlist_mapping(get_data_access(), playlist_id, [extract_video_id(url) for url in video_urls])

    # Process each video
    successful = 0
    for i, url in enumerate(video_urls, 1):
        print(f"\n{Fore.MAGENTA}[{i}/{len(video_urls)}] Processing playlist video: {url}{Style.RESET_ALL}")
        start_time = time.time()

        try:
            # Process video
            result = data_pipeline.process_video(url, language_preference)
            process_time = time.time() - start_time

            if result.get("status") == "completed":
                print(f"{Fore.GREEN}Successfully processed video in {process_time:.2f} seconds{Style.RESET_ALL}")
                print(f"Title: {result['metadata']['title']}")
                print(f"Language: {result['transcript']['language']}")
                print(f"Domain: {result['metadata']['domain']}")

                # Index the video
                print(f"{Fore.CYAN}Indexing video...{Style.RESET_ALL}")
                index_success = search_engine.index_content(result)

                if index_success:
                    print(f"{Fore.GREEN}Successfully indexed video{Style.RESET_ALL}")
                    successful += 1
                else:
                    print(f"{Fore.RED}Failed to index video{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Error processing video: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error processing {url}: {e}{Style.RESET_ALL}")
            continue

    print(f"\n{Fore.GREEN}Playlist processing completed: {successful}/{len(video_urls)} videos processed successfully{Style.RESET_ALL}")
    print(f"{Fore.CYAN}You can search within this playlist using: --search \"query\" --filter-playlist {playlist_id}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}You can list all concepts from this playlist using: --list-concepts --filter-playlist {playlist_id}{Style.RESET_ALL}")

def extract_playlist_videos(youtube_extractor, playlist_id, max_videos=None):
    """Extract video URLs from a playlist."""
    try:
        # Use the YouTube API to get playlist items
        # This is a simplified version that gets a limited number of videos
        youtube = youtube_extractor.youtube

        if not youtube:
            print(f"{Fore.RED}YouTube API client not available. Check your API key.{Style.RESET_ALL}")
            return []

        videos = []
        page_token = None

        while True:
            # Request parameters
            request_params = {
                "part": "snippet",
                "maxResults": 50,  # Maximum allowed by API
                "playlistId": playlist_id
            }

            if page_token:
                request_params["pageToken"] = page_token

            request = youtube.playlistItems().list(**request_params)
            response = request.execute()

            # Extract video IDs and create URLs
            for item in response.get("items", []):
                video_id = item["snippet"]["resourceId"]["videoId"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                videos.append(video_url)

                # Check if we've reached the limit
                if max_videos and len(videos) >= max_videos:
                    return videos

            # Check for more pages
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return videos

    except Exception as e:
        logger.error(f"Error extracting playlist videos: {e}")
        return []

def extract_video_id(url):
    """Extract video ID from URL."""
    # Check if URL is valid
    valid, video_id = YouTubeExtractor("").validate_video_url(url)
    if valid:
        return video_id
    return None

def save_playlist_mapping(data_access, playlist_id, video_ids):
    """Save mapping from playlist ID to video IDs in database."""
    try:
        # Create a playlists table if it doesn't exist
        data_access.execute_update("""
        CREATE TABLE IF NOT EXISTS playlists (
            playlist_id TEXT PRIMARY KEY,
            video_ids TEXT,
            created_at TEXT
        )
        """)

        # Insert or update playlist mapping
        data_access.execute_update("""
        INSERT OR REPLACE INTO playlists (playlist_id, video_ids, created_at)
        VALUES (?, ?, datetime('now'))
        """, (playlist_id, ",".join(video_ids)))

        logger.info(f"Saved mapping for playlist {playlist_id} with {len(video_ids)} videos")
    except Exception as e:
        logger.error(f"Error saving playlist mapping: {e}")

def create_data_directories():
    """Create necessary directories for the application data."""
    directories = [
        "data/processed",
        "data/index"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def get_api_key(args):
    """
    Get YouTube API key from environment variable or command-line arguments.

    Args:
        args: Command-line arguments

    Returns:
        YouTube API key
    """
    # Check for environment variable first (recommended approach)
    api_key = os.environ.get("YOUTUBE_API_KEY", "")

    if api_key:
        logger.info("Using YouTube API key from environment variable")
        return api_key

    # Last resort: use the argument directly (not recommended)
    if args.api_key:
        logger.warning("Using API key from command line argument (not secure)")
        return args.api_key

    return ""

def list_indexed_concepts(data_access, domain_filter=None, video_filter=None, playlist_filter=None, language=None):
    """
    List all concepts in the index with optional filtering.

    Args:
        data_access: DataAccess instance
        domain_filter: Optional domain filter
        video_filter: Optional video ID filter
        playlist_filter: Optional playlist ID filter
        language: Optional language filter
    """
    print(f"{Fore.MAGENTA}===== Indexed Concepts ====={Style.RESET_ALL}")

    # Build filter description
    filter_desc = []
    if domain_filter:
        filter_desc.append(f"domain: {domain_filter}")
    if video_filter:
        filter_desc.append(f"video: {video_filter}")
    if playlist_filter:
        filter_desc.append(f"playlist: {playlist_filter}")
    if language:
        filter_desc.append(f"language: {language}")

    if filter_desc:
        print(f"Filters: {', '.join(filter_desc)}")

    try:
        # Use optimized list_concepts method
        concepts = data_access.list_concepts(
            domain_filter=domain_filter,
            video_filter=video_filter,
            playlist_filter=playlist_filter,
            language=language
        )

        if not concepts:
            print(f"{Fore.YELLOW}No concepts found matching filters.{Style.RESET_ALL}")
            return

        # Group concepts by domain and classify as theoretical or practical
        concepts_by_domain = {}
        theoretical_count = 0
        practical_count = 0

        for concept in concepts:
            domain = concept['domain']
            if domain not in concepts_by_domain:
                concepts_by_domain[domain] = {'theoretical': [], 'practical': []}

            if concept['concept_class'] == 'theoretical':
                concepts_by_domain[domain]['theoretical'].append(concept)
                theoretical_count += 1
            else:
                concepts_by_domain[domain]['practical'].append(concept)
                practical_count += 1

        # Display stats
        total_concepts = theoretical_count + practical_count
        if total_concepts > 0:
            print(f"Total concepts: {total_concepts}")
            print(f"{Fore.BLUE}Theoretical concepts: {theoretical_count} ({theoretical_count / total_concepts * 100:.1f}%){Style.RESET_ALL}")
            print(f"{Fore.GREEN}Practical concepts: {practical_count} ({practical_count / total_concepts * 100:.1f}%){Style.RESET_ALL}")

            # Add visual representation
            theory_bar = get_progress_bar(theoretical_count / total_concepts * 100, 40, Fore.BLUE)
            practical_bar = get_progress_bar(practical_count / total_concepts * 100, 40, Fore.GREEN)

            print(f"Distribution: ")
            print(f"  Theory: {theory_bar}")
            print(f"Practice: {practical_bar}")
        else:
            print("No concepts found.")
            return

        # Display by domain
        for domain, concepts_dict in concepts_by_domain.items():
            domain_color = (Fore.BLUE if domain == 'mathematics' else
                           Fore.GREEN if domain == 'programming' else
                           Fore.YELLOW if domain == 'physics' else
                           Fore.WHITE)

            print(f"\n{domain_color}Domain: {domain.upper()}{Style.RESET_ALL}")

            # Display theoretical concepts
            if concepts_dict['theoretical']:
                print(f"\n{Fore.BLUE}  Theoretical concepts ({len(concepts_dict['theoretical'])}):{Style.RESET_ALL}")
                for i, concept in enumerate(concepts_dict['theoretical'][:30], 1):  # Limit to 30 per category
                    videos = concept['video_count']
                    occurrences = concept['occurrence_count']
                    related = concept.get('related_concepts_count', 0)

                    # Format with related concepts info
                    if related > 0:
                        print(f"  {i}. {concept['text']} (videos: {videos}, occurrences: {occurrences}, related: {related})")
                    else:
                        print(f"  {i}. {concept['text']} (videos: {videos}, occurrences: {occurrences})")

                if len(concepts_dict['theoretical']) > 30:
                    print(f"  ... and {len(concepts_dict['theoretical']) - 30} more")

            # Display practical concepts
            if concepts_dict['practical']:
                print(f"\n{Fore.GREEN}  Practical concepts ({len(concepts_dict['practical'])}):{Style.RESET_ALL}")
                for i, concept in enumerate(concepts_dict['practical'][:30], 1):  # Limit to 30 per category
                    videos = concept['video_count']
                    occurrences = concept['occurrence_count']
                    related = concept.get('related_concepts_count', 0)

                    # Format with related concepts info
                    if related > 0:
                        print(f"  {i}. {concept['text']} (videos: {videos}, occurrences: {occurrences}, related: {related})")
                    else:
                        print(f"  {i}. {concept['text']} (videos: {videos}, occurrences: {occurrences})")

                if len(concepts_dict['practical']) > 30:
                    print(f"  ... and {len(concepts_dict['practical']) - 30} more")

        # Provide hints for refining search
        print(f"\n{Fore.CYAN}Use '--search \"concept name\"' to find specific concepts.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Use '--learning-path --concepts <concept_ids>' to create a learning path.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error listing concepts: {e}{Style.RESET_ALL}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()

def show_video_concepts(search_engine, video_id):
    """Show concepts extracted from a specific video with enhanced visualization."""
    print(f"{Fore.CYAN}Getting concepts for video ID: {video_id}{Style.RESET_ALL}")

    video_concepts = search_engine.get_video_concepts(video_id)

    if not video_concepts:
        print(f"{Fore.RED}No concepts found for video ID: {video_id}{Style.RESET_ALL}")
        print("Make sure the video has been processed and indexed.")
        return

    video = video_concepts.get('video', {})
    concepts = video_concepts.get('concepts', [])
    theoretical_concepts = video_concepts.get('theoretical_concepts', [])
    practical_concepts = video_concepts.get('practical_concepts', [])
    timeline = video_concepts.get('timeline', [])
    theory_practice_ratio = video.get('theory_practice_ratio', 0)

    print(f"\n{Fore.MAGENTA}===== Video Information ====={Style.RESET_ALL}")
    print(f"Title: {Fore.YELLOW}{video.get('title', 'Unknown')}{Style.RESET_ALL}")
    print(f"Channel: {video.get('channel', 'Unknown')}")
    print(f"Domain: {Fore.GREEN}{video.get('domain', 'Unknown')}{Style.RESET_ALL}")
    print(f"Theory/Practice Ratio: {theory_practice_ratio:.2f}")
    print(f"Video Link: {Fore.CYAN}https://www.youtube.com/watch?v={video_id}{Style.RESET_ALL}")

    # Visual representation of theory/practice ratio
    theory_percent = theory_practice_ratio * 100
    practice_percent = (1 - theory_practice_ratio) * 100

    theory_bar = get_progress_bar(theory_percent, 40, Fore.BLUE)
    practice_bar = get_progress_bar(practice_percent, 40, Fore.GREEN)

    print(f"{Fore.BLUE}Theory:   {theory_bar}{Style.RESET_ALL} ({theory_percent:.1f}%)")
    print(f"{Fore.GREEN}Practice: {practice_bar}{Style.RESET_ALL} ({practice_percent:.1f}%)")

    # Display concept distribution
    total_concepts = len(theoretical_concepts) + len(practical_concepts)
    if total_concepts > 0:
        theo_concept_percent = len(theoretical_concepts) / total_concepts * 100
        prac_concept_percent = len(practical_concepts) / total_concepts * 100

        print(f"\n{Fore.MAGENTA}===== Concept Distribution ====={Style.RESET_ALL}")
        print(f"Total: {total_concepts} concepts extracted")

        theo_concept_bar = get_progress_bar(theo_concept_percent, 40, Fore.BLUE)
        prac_concept_bar = get_progress_bar(prac_concept_percent, 40, Fore.GREEN)

        print(f"{Fore.BLUE}Theoretical: {theo_concept_bar}{Style.RESET_ALL} ({len(theoretical_concepts)} concepts, {theo_concept_percent:.1f}%)")
        print(f"{Fore.GREEN}Practical:   {prac_concept_bar}{Style.RESET_ALL} ({len(practical_concepts)} concepts, {prac_concept_percent:.1f}%)")

    # Display theoretical concepts
    print(f"\n{Fore.BLUE}Theoretical concepts ({len(theoretical_concepts)}):{Style.RESET_ALL}")
    for i, concept in enumerate(theoretical_concepts[:15], 1):
        occurrences = concept.get('occurrence_count', concept.get('total_occurrences', 0))
        print(f"{i}. {concept.get('text', 'Unknown')} (occurrences: {occurrences})")

    # Display practical concepts
    print(f"\n{Fore.GREEN}Practical concepts ({len(practical_concepts)}):{Style.RESET_ALL}")
    for i, concept in enumerate(practical_concepts[:15], 1):
        occurrences = concept.get('occurrence_count', concept.get('total_occurrences', 0))
        print(f"{i}. {concept.get('text', 'Unknown')} (occurrences: {occurrences})")

    # Display timeline visualization if available
    if timeline:
        print(f"\n{Fore.MAGENTA}===== Content Timeline ====={Style.RESET_ALL}")
        print("Timeline shows theory vs. practice distribution throughout the video:")

        # Simplified timeline visualization
        duration = video.get('duration_seconds', 0)
        if duration > 0:
            timeline_width = 60  # Characters wide

            # Create buckets for the timeline
            bucket_count = min(timeline_width, len(timeline))
            bucket_duration = duration / bucket_count

            # Initialize buckets
            buckets = [{"theoretical": 0, "practical": 0, "mixed": 0, "total": 0} for _ in range(bucket_count)]

            # Fill buckets
            for segment in timeline:
                start_time = segment.get('start_time', 0)
                bucket_index = min(int(start_time / bucket_duration), bucket_count - 1)

                content_type = segment.get('context_type', 'mixed')
                buckets[bucket_index][content_type] += 1
                buckets[bucket_index]["total"] += 1

            # Display timeline
            timeline_bar = ""
            for bucket in buckets:
                if bucket["total"] == 0:
                    timeline_bar += " "
                elif bucket["theoretical"] > bucket["practical"]:
                    timeline_bar += Fore.BLUE + "T" + Style.RESET_ALL
                elif bucket["practical"] > bucket["theoretical"]:
                    timeline_bar += Fore.GREEN + "P" + Style.RESET_ALL
                else:
                    timeline_bar += Fore.YELLOW + "M" + Style.RESET_ALL

            print(f"Timeline: [{timeline_bar}]")
            print(f"Legend: {Fore.BLUE}T{Style.RESET_ALL}=Theory, {Fore.GREEN}P{Style.RESET_ALL}=Practice, {Fore.YELLOW}M{Style.RESET_ALL}=Mixed")

            # Display timescale
            start_marker = "0:00"
            middle_marker = format_timecode(duration / 2)
            end_marker = format_timecode(duration)

            # Padding to align with timeline markers
            padding = " " * (timeline_width // 2 - len(middle_marker) // 2 - len(start_marker))

            print(f"{start_marker}{padding}{middle_marker}{padding}{end_marker}")

    print(f"\n{Fore.CYAN}Run the demo script with '{os.path.basename(__file__)} --search \"query\" --filter-video {video_id}' to search within this video.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}You can also create a learning path using '--learning-path --concepts <concept_ids>'.{Style.RESET_ALL}")

def find_concept_ids(search_engine, concept_terms, domain=None, language=None):
    """
    Find concept IDs by name/text rather than requiring exact IDs.

    Args:
        search_engine: SearchEngine instance
        concept_terms: List of concept names/search terms
        domain: Optional domain filter
        language: Optional language filter

    Returns:
        Dictionary mapping provided terms to found concept IDs
    """
    found_concepts = {}
    not_found = []

    print(f"{Fore.CYAN}Looking up concepts...{Style.RESET_ALL}")

    for term in concept_terms:
        # Perform a search for this term
        query = {
            "original_text": term,
            "filters": {},
            "domain": domain,
            "language": language,
            "pagination": {"offset": 0, "limit": 5}
        }

        results = search_engine.search(query)

        # Filter for concept results only
        concept_results = [r for r in results.get("results", [])
                          if r.get("result_type") == "concept"]

        if concept_results:
            # Find the best match - favor exact matches
            best_match = None
            for result in concept_results:
                if result.get("text", "").lower() == term.lower():
                    # Exact match
                    best_match = result
                    break

            # If no exact match, use the first result
            if not best_match and concept_results:
                best_match = concept_results[0]

            if best_match:
                concept_id = best_match.get("concept_id")
                found_concepts[term] = concept_id
                concept_text = best_match.get("text")
                print(f"  • Found concept: {concept_text} (ID: {concept_id})")
                continue

        # If we get here, the concept wasn't found
        not_found.append(term)
        print(f"  • {Fore.YELLOW}Concept not found: '{term}'{Style.RESET_ALL}")

    # Show warning if some weren't found
    if not_found:
        print(f"\n{Fore.YELLOW}Warning: {len(not_found)} concept(s) were not found in the database.{Style.RESET_ALL}")
        print(f"Run '{os.path.basename(__file__)} --list-concepts' to see available concepts.")

    # Show error if none were found
    if not found_concepts:
        print(f"\n{Fore.RED}Error: No matching concepts were found.{Style.RESET_ALL}")
        print(f"Try using different terms or run with --list-concepts to see available concepts.")

    return found_concepts

def generate_learning_path(search_engine, concept_terms, theory_practice_ratio=0.5, domain=None):
    """
    Generate and display a learning path from a set of concepts.

    Args:
        search_engine: SearchEngine instance
        concept_terms: List of concept terms or IDs
        theory_practice_ratio: Theory/practice ratio preference
        domain: Optional domain filter
    """
    print(f"{Fore.MAGENTA}===== Generating Learning Path ====={Style.RESET_ALL}")
    print(f"Using {len(concept_terms)} concepts with theory ratio: {theory_practice_ratio}")

    if domain:
        print(f"Filtering to domain: {domain}")

    # Try to find concepts by name first
    concept_map = find_concept_ids(search_engine, concept_terms, domain)

    # If no concepts were found, exit
    if not concept_map:
        return

    concept_ids = list(concept_map.values())

    # Get relationship graph directly for debugging
    try:
        generator = get_concept_signature_generator(search_engine.config)
        graph = generator.relationship_graph

        print(f"\n{Fore.CYAN}Debug info:{Style.RESET_ALL}")
        print(f"Total concepts in relationship graph: {len(graph.concepts)}")

        for concept_id in concept_ids:
            if concept_id in graph.concepts:
                print(f"Concept '{concept_id}' found in relationship graph")
            else:
                print(f"{Fore.YELLOW}Concept '{concept_id}' not found in relationship graph{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Could not access relationship graph: {e}{Style.RESET_ALL}")

    # Generate learning path
    learning_path = search_engine.generate_learning_path(concept_ids, theory_practice_ratio, domain)

    if not learning_path:
        print(f"{Fore.RED}Could not generate learning path with the provided concepts.{Style.RESET_ALL}")
        return

    # Display basic information
    concepts = learning_path.get("concepts", [])
    sections = learning_path.get("sections", [])

    print(f"\n{Fore.GREEN}Successfully generated learning path with {len(concepts)} concepts{Style.RESET_ALL}")

    # Display theory/practice ratio
    theory_ratio = learning_path.get("theory_practice_ratio", {})

    # Handle both dictionary and float values for theory_ratio
    if isinstance(theory_ratio, dict):
        requested_ratio = theory_ratio.get("requested", theory_practice_ratio)
        actual_ratio = theory_ratio.get("actual", 0.5)
    else:
        # If theory_ratio is a float or other type, use it directly
        requested_ratio = theory_practice_ratio
        actual_ratio = theory_ratio if isinstance(theory_ratio, (int, float)) else 0.5

    theoretical_count = learning_path.get("theoretical_concepts", 0)
    practical_count = learning_path.get("practical_concepts", 0)

    print(f"\n{Fore.CYAN}Theory/Practice Distribution:{Style.RESET_ALL}")
    print(f"Requested ratio: {requested_ratio:.2f}")
    print(f"Actual ratio: {actual_ratio:.2f}")

    # Display visual representation
    theory_percent = actual_ratio * 100
    practice_percent = 100 - theory_percent

    theory_bar = get_progress_bar(theory_percent, 40, Fore.BLUE)
    practice_bar = get_progress_bar(practice_percent, 40, Fore.GREEN)

    print(f"{Fore.BLUE}Theory:   {theory_bar}{Style.RESET_ALL} ({theoretical_count} concepts, {theory_percent:.1f}%)")
    print(f"{Fore.GREEN}Practice: {practice_bar}{Style.RESET_ALL} ({practical_count} concepts, {practice_percent:.1f}%)")

    # Display sections
    print(f"\n{Fore.MAGENTA}===== Learning Path Sections ====={Style.RESET_ALL}")

    for i, section in enumerate(sections, 1):
        title = section.get("title", f"Section {i}")
        description = section.get("description", "")
        concept_indices = section.get("concept_indices", [])

        section_concepts = [concepts[idx] for idx in concept_indices if idx < len(concepts)]

        print(f"\n{Fore.YELLOW}{i}. {title}{Style.RESET_ALL}")
        print(f"   {description}")
        print(f"   ({len(section_concepts)} concepts)")

        # Show concepts in this section
        for j, concept in enumerate(section_concepts, 1):
            concept_class = concept.get("concept_class", "unknown")
            color = Fore.BLUE if concept_class == "theoretical" else Fore.GREEN

            # Include sequence order if available
            sequence = concept.get("sequence_order", j)

            print(f"   {color}{sequence}. {concept.get('text', 'Unknown')}{Style.RESET_ALL}")

            # Show recommended videos if available
            recommended_videos = concept.get("recommended_videos", [])
            if recommended_videos and j <= 3:  # Only show for first 3 concepts to avoid clutter
                print(f"      {Fore.CYAN}Recommended videos:{Style.RESET_ALL}")
                for k, video in enumerate(recommended_videos[:2], 1):  # Limit to 2 videos
                    video_id = video.get("video_id", "")
                    title = video.get("title", "Unknown")
                    print(f"      {k}. {title} - https://www.youtube.com/watch?v={video_id}")

    print(f"\n{Fore.CYAN}Learning path successfully generated. Follow the sections in order for optimal learning.{Style.RESET_ALL}")

def search_concepts(search_engine, query: str, theory_practice_ratio: float = None, domain_filter: str = None,
                   video_filter: str = None, playlist_filter: str = None, language: str = None):
    """
    Search for concepts in the indexed content with optional filtering.

    Args:
        search_engine: SearchEngine instance
        query: Search query text
        theory_practice_ratio: Theory/practice ratio filter (0-1)
        domain_filter: Optional domain filter
        video_filter: Optional video ID filter
        playlist_filter: Optional playlist ID filter
        language: Optional language filter
    """
    print(f"\n{Fore.MAGENTA}===== Searching for: '{query}' ====={Style.RESET_ALL}")

    # Build filter description
    filter_desc = []
    if domain_filter:
        filter_desc.append(f"domain: {domain_filter}")
    if video_filter:
        filter_desc.append(f"video: {video_filter}")
    if playlist_filter:
        filter_desc.append(f"playlist: {playlist_filter}")
    if language:
        filter_desc.append(f"language: {language}")

    if filter_desc:
        print(f"Filters: {', '.join(filter_desc)}")

    ratio_desc = "balanced"
    theory_color = Fore.BLUE
    practice_color = Fore.GREEN

    if theory_practice_ratio is not None:
        if theory_practice_ratio > 0.7:
            ratio_desc = f"{theory_color}theoretical{Style.RESET_ALL}"
        elif theory_practice_ratio < 0.3:
            ratio_desc = f"{practice_color}practical{Style.RESET_ALL}"
        else:
            ratio_desc = f"{Fore.YELLOW}balanced{Style.RESET_ALL}"

    print(f"Theory/Practice preference: {ratio_desc}")

    # Create structured query
    structured_query = {
        "original_text": query,
        "filters": {},
        "theory_practice_ratio": theory_practice_ratio,
        "domain": domain_filter,
        "language": language,  # Add language to query
        "pagination": {
            "offset": 0,
            "limit": 15
        }
    }

    # Add video filter if specified
    if video_filter:
        structured_query["filters"]["video_id"] = video_filter

    # Get video IDs from playlist if specified
    if playlist_filter:
        video_ids = get_playlist_video_ids(get_data_access(), playlist_filter)
        if video_ids:
            structured_query["filters"]["video_ids"] = video_ids
        else:
            print(f"{Fore.YELLOW}No videos found for playlist {playlist_filter}{Style.RESET_ALL}")
            return None

    # Execute search
    try:
        results = search_engine.search(structured_query)

        total = results.get('totalResults', 0)
        theoretical = results.get('theoreticalResults', 0)
        practical = results.get('practicalResults', 0)
        mixed = results.get('mixedResults', 0)

        if total > 0:
            print(f"Total results: {total}")

            # Show result type distribution
            if total > 0:
                theory_percent = (theoretical / total) * 100
                practice_percent = (practical / total) * 100
                mixed_percent = (mixed / total) * 100 if mixed > 0 else 0

                theory_bar = get_progress_bar(theory_percent, 30, Fore.BLUE)
                practice_bar = get_progress_bar(practice_percent, 30, Fore.GREEN)
                mixed_bar = get_progress_bar(mixed_percent, 30, Fore.YELLOW) if mixed > 0 else ""

                print(f"{theory_color}Theoretical: {theory_bar}{Style.RESET_ALL} ({theoretical}, {theory_percent:.1f}%)")
                print(f"{practice_color}Practical: {practice_bar}{Style.RESET_ALL} ({practical}, {practice_percent:.1f}%)")

                if mixed > 0:
                    print(f"{Fore.YELLOW}Mixed: {mixed_bar}{Style.RESET_ALL} ({mixed}, {mixed_percent:.1f}%)")

            # Show domain distribution if available
            domains = results.get('domainDistribution', [])
            if domains:
                print(f"\n{Fore.MAGENTA}Domain Distribution:{Style.RESET_ALL}")
                for domain_info in domains:
                    domain_name = domain_info.get('domain', 'unknown')
                    count = domain_info.get('count', 0)
                    domain_percent = (count / total) * 100

                    # Choose color based on domain
                    domain_color = (Fore.BLUE if domain_name == 'mathematics' else
                                  Fore.GREEN if domain_name == 'programming' else
                                  Fore.YELLOW if domain_name == 'physics' else
                                  Fore.WHITE)

                    domain_bar = get_progress_bar(domain_percent, 20, domain_color)
                    print(f"{domain_color}{domain_name}: {domain_bar}{Style.RESET_ALL} ({count}, {domain_percent:.1f}%)")

        search_results = results.get('results', [])
        if search_results:
            print("\n{0} Top Results {0}".format("="*20))

            # Group results by video
            videos_seen = set()

            for i, result in enumerate(search_results, 1):
                context_type = result.get('context_type', 'unknown')
                result_type = result.get('result_type', 'unknown')
                video_id = result.get('video_id')
                result_language = result.get('language', '') # Get result language

                # Choose emoji and color based on content type
                emoji = "🧠" if context_type == 'theoretical' else "🛠️" if context_type == 'practical' else "📝"
                color = theory_color if context_type == 'theoretical' else practice_color if context_type == 'practical' else Style.RESET_ALL

                # Different styling for concepts vs segments
                if result_type == 'concept':
                    concept_id = result.get('concept_id')
                    # Add language indicator if available
                    lang_indicator = f" [{result_language}]" if result_language and result_language != 'en' else ""
                    print(f"\n{i}. {emoji} {color}Concept: {result.get('text')}{lang_indicator}{Style.RESET_ALL} [ID: {concept_id}]")

                    # Show concept's relevance score if available
                    relevance = result.get('relevance_score')
                    if relevance:
                        relevance_bar = get_progress_bar(min(relevance * 20, 100), 20)
                        print(f"   Relevance: {relevance_bar} ({relevance:.2f})")
                else:
                    print(f"\n{i}. {emoji} {color}Segment match in video{Style.RESET_ALL}")

                # Add video information (avoid repeating video title when showing multiple segments from same video)
                video_title = result.get('video_title', 'Unknown')
                is_new_video = video_id not in videos_seen
                videos_seen.add(video_id)

                if is_new_video:
                    print(f"   {Fore.YELLOW}Video: {video_title}{Style.RESET_ALL}")
                    print(f"   {Fore.CYAN}https://www.youtube.com/watch?v={video_id}{Style.RESET_ALL}")
                else:
                    print(f"   {Fore.YELLOW}From: {video_title}{Style.RESET_ALL}")

                # Show the context with proper wrapping
                context_text = result.get('context_text', result.get('text', ''))
                if context_text:
                    # Wrap text at reasonable width
                    wrapped_text = textwrap.fill(context_text, width=80, initial_indent="   ", subsequent_indent="   ")
                    print(f"{wrapped_text}")

                # Add timecode and video link information for segments
                start_time = result.get('start_time')
                if start_time is not None:
                    timecode = format_timecode(start_time)
                    video_url = get_video_url_with_timecode(video_id, start_time)
                    print(f"   Timecode: {timecode} - {video_url}")

            # Show suggestions if available
            suggestions = results.get('suggestions', [])
            if suggestions:
                print(f"\n{Fore.MAGENTA}Suggestions to improve your search:{Style.RESET_ALL}")
                for i, suggestion in enumerate(suggestions, 1):
                    suggestion_text = suggestion.get('text', '')
                    print(f"{i}. {suggestion_text}")

        else:
            print(f"{Fore.YELLOW}No results found.{Style.RESET_ALL}")
            print("Try another search query or check if content has been indexed.")

        return results
    except Exception as e:
        print(f"{Fore.RED}Error performing search: {e}{Style.RESET_ALL}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        return None

def display_results(result: Dict[str, Any], process_time: float):
    """Display the results of video processing in a readable format."""
    # Video information
    video_id = result['metadata']['video_id']
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    print(f"\n{Fore.MAGENTA}===== Video Information ====={Style.RESET_ALL}")
    print(f"Title: {Fore.YELLOW}{result['metadata']['title']}{Style.RESET_ALL}")
    print(f"Channel: {result['metadata']['channel']}")
    print(f"Domain: {Fore.GREEN}{result['metadata']['domain']}{Style.RESET_ALL} (confidence: {result['metadata']['domain_confidence']:.2f})")
    print(f"Duration: {format_duration(result['metadata'].get('duration_seconds', 0))}")
    print(f"Video URL: {Fore.CYAN}{video_url}{Style.RESET_ALL}")
    print(f"Processing time: {process_time:.2f} seconds")

    # Theory/Practice analysis
    tp_results = result['theory_practice_results']
    print(f"\n{Fore.MAGENTA}===== Theory/Practice Analysis ====={Style.RESET_ALL}")

    classification = tp_results['classification']
    classification_color = (Fore.BLUE if classification == "theoretical" else
                           Fore.GREEN if classification == "practical" else
                           Fore.YELLOW)

    print(f"Classification: {classification_color}{classification}{Style.RESET_ALL} (confidence: {tp_results['confidence']:.2f})")
    print(f"Theoretical segments: {Fore.BLUE}{tp_results['theoretical_segments']}{Style.RESET_ALL}")
    print(f"Practical segments: {Fore.GREEN}{tp_results['practical_segments']}{Style.RESET_ALL}")
    print(f"Mixed segments: {Fore.YELLOW}{tp_results['mixed_segments']}{Style.RESET_ALL}")
    print(f"Theory/Practice ratio: {tp_results['theory_practice_ratio']:.2f}")

    # Visual representation of theory/practice ratio
    theory_percent = tp_results['theory_practice_ratio'] * 100
    practice_percent = (1 - tp_results['theory_practice_ratio']) * 100

    theory_bar = get_progress_bar(theory_percent, 40, Fore.BLUE)
    practice_bar = get_progress_bar(practice_percent, 40, Fore.GREEN)

    print(f"{Fore.BLUE}Theory:   {theory_bar}{Style.RESET_ALL} ({theory_percent:.1f}%)")
    print(f"{Fore.GREEN}Practice: {practice_bar}{Style.RESET_ALL} ({practice_percent:.1f}%)")

    # Show duration analysis if available
    duration_analysis = tp_results.get('duration_analysis', {})
    if duration_analysis:
        total_duration = duration_analysis.get('total_duration', 0)
        if total_duration > 0:
            theoretical_duration = duration_analysis.get('theoretical_duration', 0)
            practical_duration = duration_analysis.get('practical_duration', 0)

            theoretical_time_percent = (theoretical_duration / total_duration) * 100
            practical_time_percent = (practical_duration / total_duration) * 100

            print(f"\n{Fore.MAGENTA}Time Distribution:{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Theory:   {format_duration(theoretical_duration)} ({theoretical_time_percent:.1f}%){Style.RESET_ALL}")
            print(f"{Fore.GREEN}Practice: {format_duration(practical_duration)} ({practical_time_percent:.1f}%){Style.RESET_ALL}")

    # Display key concepts
    domain_features = result['domain_features']
    print(f"\n{Fore.MAGENTA}===== Key Concepts ====={Style.RESET_ALL}")

    key_concepts = domain_features.get('key_concepts', [])
    if not key_concepts:
        print(f"{Fore.YELLOW}No key concepts extracted. This may indicate a content analysis issue.{Style.RESET_ALL}")
    else:
        # Use the improved concept organization if available
        theoretical_concepts = domain_features.get('theoretical_concepts',
                                                [c for c in key_concepts if c.get('theoretical', False)])
        practical_concepts = domain_features.get('practical_concepts',
                                               [c for c in key_concepts if not c.get('theoretical', False)])

        print(f"{Fore.BLUE}Theoretical concepts ({len(theoretical_concepts)}):{Style.RESET_ALL}")
        for i, concept in enumerate(theoretical_concepts[:10], 1):
            # Add score or pattern match info if available
            score = concept.get('score', concept.get('frequency', 0))
            pattern_match = concept.get('pattern_match', False)

            if pattern_match:
                print(f"{i}. {concept['text']} (score: {score:.1f}) [pattern match]")
            else:
                print(f"{i}. {concept['text']} (score: {score:.1f})")

        print(f"\n{Fore.GREEN}Practical concepts ({len(practical_concepts)}):{Style.RESET_ALL}")
        for i, concept in enumerate(practical_concepts[:10], 1):
            # Add score or pattern match info if available
            score = concept.get('score', concept.get('frequency', 0))
            pattern_match = concept.get('pattern_match', False)

            if pattern_match:
                print(f"{i}. {concept['text']} (score: {score:.1f}) [pattern match]")
            else:
                print(f"{i}. {concept['text']} (score: {score:.1f})")

        # Show concept relationships if available
        concept_relationships = domain_features.get('concept_relationships', [])
        if concept_relationships:
            print(f"\n{Fore.MAGENTA}Concept Relationships:{Style.RESET_ALL}")
            for i, rel in enumerate(concept_relationships[:5], 1):
                source = rel.get('source_concept', '')
                target = rel.get('target_concept', '')
                rel_type = rel.get('relationship_type', 'related')
                count = rel.get('co_occurrence_count', 0)

                # Format relationship type
                type_color = Fore.YELLOW
                if "theoretical" in rel_type:
                    type_color = Fore.BLUE
                elif "practical" in rel_type:
                    type_color = Fore.GREEN

                print(f"{i}. {source} {type_color}{rel_type}{Style.RESET_ALL} {target} ({count} co-occurrences)")

    # Display transcript sample
    transcript = result.get('transcript', {})
    segments = transcript.get('segments', [])
    print(f"\n{Fore.MAGENTA}===== Transcript Summary ====={Style.RESET_ALL}")
    print(f"Total segments: {len(segments)}")

    if segments:
        # Show a small sample of segments with their classification
        print("Sample segments:")
        for i, segment in enumerate(segments[:5], 1):
            content_type = segment.get('content_type', 'unknown')
            confidence = segment.get('classification_confidence', 0.6)

            color = Fore.BLUE if content_type == 'theoretical' else Fore.GREEN if content_type == 'practical' else Fore.WHITE
            emoji = "🧠" if content_type == 'theoretical' else "🛠️" if content_type == 'practical' else "📝"
            text = segment.get('text', '')
            start_time = segment.get('start_time', 0)
            timecode = format_timecode(start_time)

            # Limit text length while being smart about not cutting in the middle of a word
            if len(text) > 100:
                text = text[:97] + "..."

            # Add confidence info
            conf_str = f" ({confidence:.2f} confidence)" if confidence != 0.6 else ""

            print(f"{i}. {emoji} {color}[{content_type}{conf_str}] {text}{Style.RESET_ALL}")

            # Display timecode and link
            video_url = get_video_url_with_timecode(video_id, start_time)
            print(f"   {Fore.YELLOW}Timecode: {timecode}{Style.RESET_ALL}")
            print(f"   {Fore.CYAN}Link: {video_url}{Style.RESET_ALL}")

def format_timecode(seconds):
    """Format seconds into a human-readable timecode (MM:SS)."""
    if seconds is None or seconds < 0:
        return "00:00"

    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def get_video_url_with_timecode(video_id, start_time):
    """Create a YouTube URL with timecode."""
    if video_id is None:
        return "N/A"

    # Round down to nearest integer for YouTube URL
    start_seconds = int(start_time) if start_time is not None else 0
    return f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}"

def format_duration(seconds):
    """Format seconds into a human-readable duration string."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def print_header():
    """Print a fancy header for the demo script."""
    width = 80
    print(f"{Fore.CYAN}" + "=" * width)
    print(" LECTURE VIDEO CONTENT INDEXER ".center(width, "="))
    print(" Enhanced Multilingual Theory vs. Practice Classification Demo ".center(width, "="))
    print("=" * width + f"{Style.RESET_ALL}")
    print()
    print("This script demonstrates the enhanced Lecture Video Content Indexer with the following capabilities:")
    print(f" - {Fore.BLUE}Processes{Style.RESET_ALL} educational videos from YouTube with multilingual support")
    print(f" - {Fore.BLUE}Detects{Style.RESET_ALL} language automatically or by preference (English/Russian support)")
    print(f" - {Fore.BLUE}Classifies{Style.RESET_ALL} content as theoretical or practical with improved accuracy")
    print(f" - {Fore.BLUE}Extracts{Style.RESET_ALL} domain-specific concepts using ML-based techniques")
    print(f" - {Fore.BLUE}Indexes{Style.RESET_ALL} educational concepts for advanced multilingual search")
    print(f" - {Fore.BLUE}Generates{Style.RESET_ALL} learning paths for educational content")
    print()
    print(f"{Fore.YELLOW}Common commands:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Process a video:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/watch?v=VIDEO_ID")
    print(f"  {Fore.GREEN}Process with language preference:{Style.RESET_ALL} {os.path.basename(__file__)} --language ru https://www.youtube.com/watch?v=VIDEO_ID")
    print(f"  {Fore.GREEN}List all concepts:{Style.RESET_ALL} {os.path.basename(__file__)} --list-concepts")
    print(f"  {Fore.GREEN}List concepts by language:{Style.RESET_ALL} {os.path.basename(__file__)} --list-concepts --filter-language ru")
    print(f"  {Fore.GREEN}Search concepts:{Style.RESET_ALL} {os.path.basename(__file__)} --search \"query\"")
    print()
    print(f"{Fore.CYAN}API Key Setup:{Style.RESET_ALL}")
    print("  Set the YOUTUBE_API_KEY environment variable (recommended):")
    print(f"  {Fore.YELLOW}export YOUTUBE_API_KEY='your_api_key'{Style.RESET_ALL}")
    print()

def get_progress_bar(percent: float, width: int, color: str = "") -> str:
    """Create a progress bar visualization."""
    filled_width = int(width * percent / 100)
    empty_width = width - filled_width

    bar = color + "█" * filled_width + Style.RESET_ALL + "░" * empty_width
    return bar

if __name__ == "__main__":
    main()
