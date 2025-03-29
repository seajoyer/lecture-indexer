def extract_playlist_id(playlist_url_or_id):
    """
    Extract playlist ID from a URL or return the ID directly.

    Args:
        playlist_url_or_id: Playlist URL or ID

    Returns:
        Playlist ID
    """
    # Check if it's a URL
    if "youtube.com" in playlist_url_or_id or "youtu.be" in playlist_url_or_id:
        # Extract playlist ID from URL
        import re
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/playlist\?list=([^&\s]+)',  # Standard playlist URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?.*[\?&]list=([^&\s]+)'  # Video URL with playlist
        ]

        for pattern in patterns:
            match = re.match(pattern, playlist_url_or_id)
            if match:
                return match.group(1)

        print(f"Warning: Could not extract playlist ID from URL: {playlist_url_or_id}")
        return None
    else:
        # Assume it's already a playlist ID
        return playlist_url_or_id

def process_playlist(data_pipeline, search_engine, playlist_url_or_id, args):
    """
    Process all videos in a playlist.

    Args:
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        playlist_url_or_id: Playlist URL or ID
        args: Command-line arguments
    """
    # Extract playlist ID
    playlist_id = extract_playlist_id(playlist_url_or_id)
    if not playlist_id:
        print(f"Error: Invalid playlist URL or ID: {playlist_url_or_id}")
        return

    print(f"Processing playlist: {playlist_id}")

    # Get videos in playlist
    videos = get_playlist_videos(playlist_id)
    if not videos:
        print(f"Error: No videos found in playlist: {playlist_id}")
        return

    print(f"Found {len(videos)} videos in playlist")

    # Apply max videos limit if specified
    if args.max_videos and not args.no_limit:
        print(f"Limiting to {args.max_videos} videos")
        videos = videos[:args.max_videos]

    # Convert video IDs to URLs
    video_urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in videos]

    # Process videos
    processed_videos = process_videos_batch(data_pipeline, search_engine, video_urls, args)

    print(f"\nSummary: Processed {len(processed_videos)} out of {len(videos)} videos from playlist")

    # Show processed video IDs
    if processed_videos:
        print("\nProcessed Videos:")
        for i, video_id in enumerate(processed_videos):
            print(f"  {i+1}. {video_id}")
#!/usr/bin/env python3
"""
Demo application for the Lecture Video Content Indexer.
Provides command-line access to key functionality including video processing,
concept exploration, and search.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional
import time
import json

# Import project modules
from data_pipeline import DataPipeline
from search_engine import SearchEngine
from data_access import get_data_access
from cache_manager import get_cache_stats, cache_clear
try:
    from concept_signature_generator import get_concept_signature_generator, RelationshipGraph
    HAS_CONCEPT_GENERATOR = True
except ImportError:
    HAS_CONCEPT_GENERATOR = False
    print("Warning: concept_signature_generator module not available, related features will be disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Demo for Lecture Video Content Indexer")

    # Video processing options
    parser.add_argument("--video", help="Show details for specific video ID (already indexed)")
    parser.add_argument("--playlist", help="Process all videos in a YouTube playlist URL or ID")
    parser.add_argument("--list-concepts", action="store_true", help="List all indexed concepts")
    parser.add_argument("--learning-path", action="store_true", help="Generate a learning path from concepts")
    parser.add_argument("--api-key", help="YouTube API key (NOT RECOMMENDED - use environment variable instead)")

    # Search options
    parser.add_argument("--search", help="Search query after processing")
    parser.add_argument("--theory-ratio", type=float, help="Theory/practice ratio for search (0-1, 1=all theoretical)")

    # Playlist processing options
    parser.add_argument("--no-limit", action="store_true", help="Process all videos in a playlist without limit (default)")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process from a playlist (ignored with --no-limit)")

    # Filtering options
    parser.add_argument("--filter-domain", choices=["mathematics", "programming", "physics"], help="Filter by domain")
    parser.add_argument("--filter-video", help="Filter to a specific video ID")
    parser.add_argument("--filter-playlist", help="Filter to a specific playlist ID")
    parser.add_argument("--filter-language", choices=["en", "ru"], help="Filter by language")

    # Learning path options
    parser.add_argument("--concepts", nargs="+", help="Concept IDs for learning path generation")

    # System options
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--optimize-db", action="store_true", help="Optimize database before running")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all caches before running")
    parser.add_argument("--language", choices=["en", "ru", "auto"], default="auto", help="Preferred language for processing (auto=detect)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Advanced concept analysis
    parser.add_argument("--concept-signatures", action="store_true", help="Show concept signatures for indexed concepts")
    parser.add_argument("--relationship-graph", action="store_true", help="Display concept relationship graph")
    parser.add_argument("--analyze-concepts", action="store_true", help="Analyze concept relationships across all videos")

    return parser.parse_args()

def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables and arguments.

    Returns:
        Configuration dictionary
    """
    config = {
        "youtube_api_key": os.environ.get("YOUTUBE_API_KEY", ""),
        "output_dir": os.environ.get("OUTPUT_DIR", "data/processed"),
        "index_dir": os.environ.get("INDEX_DIR", "data/index"),
    }

    # Create necessary directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["index_dir"], exist_ok=True)

    return config

def init_components(args):
    """
    Initialize all required components.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (data_pipeline, search_engine, data_access)
    """
    # Load configuration
    config = load_config()

    # Override API key if provided
    if args.api_key:
        config["youtube_api_key"] = args.api_key
        os.environ["YOUTUBE_API_KEY"] = args.api_key

    # Initialize components
    data_access = get_data_access()
    data_pipeline = DataPipeline(config)
    search_engine = SearchEngine(config)

    logger.info("Components initialized")

    return data_pipeline, search_engine, data_access

def process_or_show_video(data_pipeline, search_engine, video_id_or_url, language_preference=None):
    """
    Process a video if it's a URL, show details if it's a video ID.

    Args:
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        video_id_or_url: Video ID or YouTube URL
        language_preference: Optional language preference list
    """
    # Check if it's a URL
    if "youtube.com" in video_id_or_url or "youtu.be" in video_id_or_url:
        # It's a URL, process it
        print(f"Processing video URL: {video_id_or_url}")

        # Set default language preference if not provided
        if language_preference is None:
            language_preference = ["en", "ru"]

        result = data_pipeline.process_video(video_id_or_url, language_preference)

        if result.get("status") == "completed":
            video_id = result.get("video_id")
            print(f"Successfully processed video: {video_id}")

            # Index the content
            success = search_engine.index_content(result)
            if success:
                print(f"Successfully indexed video content")

                # Now show the details
                show_video_details(search_engine, video_id)
            else:
                print(f"Failed to index video content")
        else:
            print(f"Failed to process video: {result.get('error', 'Unknown error')}")
    else:
        # It's a video ID, show details
        show_video_details(search_engine, video_id_or_url)

def extract_playlist_id(playlist_url_or_id):
    """
    Extract playlist ID from a URL or return the ID directly.

    Args:
        playlist_url_or_id: Playlist URL or ID

    Returns:
        Playlist ID
    """
    # Check if it's a URL
    if "youtube.com" in playlist_url_or_id or "youtu.be" in playlist_url_or_id:
        # Extract playlist ID from URL
        import re
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/playlist\?list=([^&\s]+)',  # Standard playlist URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?.*[\?&]list=([^&\s]+)'  # Video URL with playlist
        ]

        for pattern in patterns:
            match = re.match(pattern, playlist_url_or_id)
            if match:
                return match.group(1)

        print(f"Warning: Could not extract playlist ID from URL: {playlist_url_or_id}")
        return None
    else:
        # Assume it's already a playlist ID
        return playlist_url_or_id

def process_playlist(data_pipeline, search_engine, playlist_url_or_id, args):
    """
    Process all videos in a playlist.

    Args:
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        playlist_url_or_id: Playlist URL or ID
        args: Command-line arguments
    """
    # Extract playlist ID
    playlist_id = extract_playlist_id(playlist_url_or_id)
    if not playlist_id:
        print(f"Error: Invalid playlist URL or ID: {playlist_url_or_id}")
        return

    print(f"Processing playlist: {playlist_id}")

    # Get videos in playlist
    videos = get_playlist_videos(playlist_id)
    if not videos:
        print(f"Error: No videos found in playlist: {playlist_id}")
        return

    print(f"Found {len(videos)} videos in playlist")

    # Apply max videos limit if specified
    if args.max_videos and not args.no_limit:
        print(f"Limiting to {args.max_videos} videos")
        videos = videos[:args.max_videos]

    # Convert video IDs to URLs
    video_urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in videos]

    # Process videos
    processed_videos = process_videos_batch(data_pipeline, search_engine, video_urls, args)

    print(f"\nSummary: Processed {len(processed_videos)} out of {len(videos)} videos from playlist")

    # Show processed video IDs
    if processed_videos:
        print("\nProcessed Videos:")
        for i, video_id in enumerate(processed_videos):
            print(f"  {i+1}. {video_id}")

def process_videos_batch(data_pipeline, search_engine, videos, args):
    """
    Process a batch of videos.

    Args:
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        videos: List of video URLs or IDs
        args: Command-line arguments

    Returns:
        List of processed video IDs
    """
    # Set language preference
    if args.language == "auto":
        language_preference = ["en", "ru"]
    else:
        language_preference = [args.language]

    processed_videos = []

    for i, video in enumerate(videos):
        # Process the video
        print(f"\nProcessing video {i+1}/{len(videos)}: {video}")

        try:
            result = data_pipeline.process_video(video, language_preference)

            if result.get("status") == "completed":
                video_id = result.get("video_id")
                print(f"Successfully processed video: {video_id}")

                # Index the content
                try:
                    success = search_engine.index_content(result)
                    if success:
                        print(f"Successfully indexed video content")
                        processed_videos.append(video_id)
                    else:
                        print(f"Failed to index video content")
                except Exception as e:
                    print(f"Error indexing video content: {e}")
            else:
                print(f"Failed to process video: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error processing video {video}: {e}")
            print("Continuing with next video...")

        # Add a small delay between videos to avoid rate limiting
        time.sleep(1)

    return processed_videos

def show_video_details(search_engine, video_id):
    """
    Show details for a specific video.

    Args:
        search_engine: SearchEngine instance
        video_id: YouTube video ID
    """
    # Get video concepts
    video_concepts = search_engine.get_video_concepts(video_id)
    if not video_concepts:
        print(f"Video not found or not processed: {video_id}")
        return

    # Extract video details
    video = video_concepts.get("video", {})
    concepts = video_concepts.get("concepts", [])
    theory_practice_ratio = video.get("theory_practice_ratio", 0.5)

    # Print video information
    print(f"\n=== Video Details: {video_id} ===")
    print(f"Title: {video.get('title', 'N/A')}")
    print(f"Channel: {video.get('channel', 'N/A')}")
    print(f"Domain: {video.get('domain', 'N/A')}")
    print(f"Theory/Practice Ratio: {theory_practice_ratio:.2f}")
    print(f"Total Concepts: {len(concepts)}")

    # Print theoretical concepts
    theoretical = [c for c in concepts if c.get("concept_class") == "theoretical"]
    print(f"\nTheoretical Concepts ({len(theoretical)}):")
    for i, concept in enumerate(theoretical[:10]):  # Show top 10
        print(f"  {i+1}. {concept.get('text', 'N/A')}")
    if len(theoretical) > 10:
        print(f"  ... and {len(theoretical) - 10} more")

    # Print practical concepts
    practical = [c for c in concepts if c.get("concept_class") == "practical"]
    print(f"\nPractical Concepts ({len(practical)}):")
    for i, concept in enumerate(practical[:10]):  # Show top 10
        print(f"  {i+1}. {concept.get('text', 'N/A')}")
    if len(practical) > 10:
        print(f"  ... and {len(practical) - 10} more")

    print("\nVideo URL: https://www.youtube.com/watch?v=" + video_id)

def list_concepts(data_access, args):
    """
    List all indexed concepts with optional filtering.

    Args:
        data_access: DataAccess instance
        args: Command-line arguments
    """
    # Apply filters
    domain_filter = args.filter_domain
    video_filter = args.filter_video
    playlist_filter = args.filter_playlist
    language_filter = args.filter_language

    # Query concepts
    concepts = data_access.list_concepts(
        domain_filter=domain_filter,
        video_filter=video_filter,
        playlist_filter=playlist_filter,
        language=language_filter
    )

    if not concepts:
        print("No concepts found matching the criteria")
        return

    # Separate theoretical and practical concepts
    theoretical = [c for c in concepts if c.get("concept_class") == "theoretical"]
    practical = [c for c in concepts if c.get("concept_class") == "practical"]

    # Print summary
    print(f"\n=== Concepts Summary ===")
    print(f"Total Concepts: {len(concepts)}")
    print(f"Theoretical Concepts: {len(theoretical)}")
    print(f"Practical Concepts: {len(practical)}")

    if domain_filter:
        print(f"Filter: Domain = {domain_filter}")
    if video_filter:
        print(f"Filter: Video ID = {video_filter}")
    if playlist_filter:
        print(f"Filter: Playlist ID = {playlist_filter}")
    if language_filter:
        print(f"Filter: Language = {language_filter}")

    # Print theoretical concepts
    print(f"\nTheoretical Concepts:")
    for i, concept in enumerate(theoretical[:20]):  # Show top 20
        print(f"  {i+1}. {concept.get('text', 'N/A')} (ID: {concept.get('concept_id', 'N/A')})")
    if len(theoretical) > 20:
        print(f"  ... and {len(theoretical) - 20} more")

    # Print practical concepts
    print(f"\nPractical Concepts:")
    for i, concept in enumerate(practical[:20]):  # Show top 20
        print(f"  {i+1}. {concept.get('text', 'N/A')} (ID: {concept.get('concept_id', 'N/A')})")
    if len(practical) > 20:
        print(f"  ... and {len(practical) - 20} more")

def generate_learning_path(search_engine, args):
    """
    Generate a learning path from concepts.

    Args:
        search_engine: SearchEngine instance
        args: Command-line arguments
    """
    if not args.concepts:
        print("Error: No concept IDs provided. Use --concepts to specify concept IDs.")
        return

    # Set theory ratio
    theory_ratio = args.theory_ratio if args.theory_ratio is not None else 0.5

    # Generate learning path
    learning_path = search_engine.generate_learning_path(
        args.concepts,
        theory_practice_ratio=theory_ratio,
        domain=args.filter_domain
    )

    if not learning_path:
        print("Error: Could not generate learning path. Check that concept IDs are valid.")
        return

    # Print learning path
    print(f"\n=== Learning Path ===")
    print(f"Total Concepts: {learning_path.get('total_concepts', 0)}")
    print(f"Theoretical Concepts: {learning_path.get('theoretical_concepts', 0)}")
    print(f"Practical Concepts: {learning_path.get('practical_concepts', 0)}")
    print(f"Theory/Practice Ratio: {learning_path.get('theory_practice_ratio', {}).get('actual', 0.5):.2f}")

    if args.filter_domain:
        print(f"Domain Filter: {args.filter_domain}")

    # Print concepts in order
    concepts = learning_path.get("concepts", [])
    print(f"\nLearning Sequence:")
    for i, concept in enumerate(concepts):
        concept_class = concept.get("concept_class", "unknown")
        print(f"  {i+1}. {concept.get('text', 'N/A')} ({concept_class})")

        # Print recommended videos if available
        recommended_videos = concept.get("recommended_videos", [])
        if recommended_videos:
            print(f"     Recommended Video: {recommended_videos[0].get('title', 'N/A')}")
            print(f"     Video URL: https://www.youtube.com/watch?v={recommended_videos[0].get('video_id', '')}")

    # Print sections if available
    sections = learning_path.get("sections", [])
    if sections:
        print(f"\nLearning Path Sections:")
        for i, section in enumerate(sections):
            print(f"  {i+1}. {section.get('title', 'N/A')}")
            print(f"     {section.get('description', '')}")
            print(f"     Concepts: {len(section.get('concept_indices', []))}")

def search_content(search_engine, args):
    """
    Search for content matching a query.

    Args:
        search_engine: SearchEngine instance
        args: Command-line arguments
    """
    if not args.search:
        print("Error: No search query provided. Use --search to specify a query.")
        return

    # Prepare search query
    filters = {}
    if args.filter_domain:
        filters["domain"] = args.filter_domain
    if args.filter_video:
        filters["video_id"] = args.filter_video
    if args.filter_playlist:
        # Get video IDs from playlist
        try:
            playlist_videos = get_playlist_videos(args.filter_playlist)
            if playlist_videos:
                filters["video_ids"] = playlist_videos
            else:
                print(f"Warning: No videos found for playlist {args.filter_playlist}")
        except Exception as e:
            print(f"Error getting playlist videos: {e}")

    # Create structured query
    structured_query = {
        "original_text": args.search,
        "filters": filters,
        "theory_practice_ratio": args.theory_ratio,
        "domain": args.filter_domain,
        "language": args.filter_language,
        "pagination": {"offset": 0, "limit": 20}
    }

    # Execute search
    results = search_engine.search(structured_query)

    # Print results
    total_results = results.get("totalResults", 0)
    theoretical_results = results.get("theoreticalResults", 0)
    practical_results = results.get("practicalResults", 0)

    print(f"\n=== Search Results: '{args.search}' ===")
    print(f"Total Results: {total_results}")
    print(f"Theoretical Results: {theoretical_results}")
    print(f"Practical Results: {practical_results}")

    if args.filter_domain:
        print(f"Domain Filter: {args.filter_domain}")
    if args.filter_video:
        print(f"Video Filter: {args.filter_video}")
    if args.filter_language:
        print(f"Language Filter: {args.filter_language}")
    if args.theory_ratio is not None:
        print(f"Theory/Practice Ratio: {args.theory_ratio:.2f}")

    # Print result items
    result_items = results.get("results", [])
    print(f"\nTop Results:")
    for i, result in enumerate(result_items[:15]):  # Show top 15
        result_type = result.get("result_type", "unknown")
        context_type = result.get("context_type", "unknown")
        text = result.get("text", "N/A")
        video_id = result.get("video_id", "")
        video_title = result.get("video_title", "")

        print(f"  {i+1}. {text}")
        print(f"     Type: {result_type} ({context_type})")
        print(f"     Video: {video_title}")
        if video_id:
            print(f"     URL: https://www.youtube.com/watch?v={video_id}")
        if result.get("start_time") is not None:
            start_time = result.get("start_time", 0)
            time_str = f"{int(start_time // 60)}:{int(start_time % 60):02d}"
            print(f"     Time: {time_str}")
        print()

    if len(result_items) > 15:
        print(f"  ... and {len(result_items) - 15} more results")

def get_playlist_videos(playlist_id):
    """
    Get video IDs in a playlist.

    Args:
        playlist_id: YouTube playlist ID

    Returns:
        List of video IDs
    """
    # This is a simplified implementation
    # In a real implementation, you would use the YouTube API
    # to get the videos in the playlist

    # For now, just check if we have this playlist in the database
    data_access = get_data_access()
    playlist_query = "SELECT video_ids FROM playlists WHERE playlist_id = ?"
    playlists = data_access.execute_query(playlist_query, (playlist_id,))

    if playlists and playlists[0].get("video_ids"):
        # Split comma-separated video IDs
        return playlists[0].get("video_ids").split(",")

    # If not in database, try to get it from YouTube API if available
    try:
        # Import and check for YouTube API key
        from youtube_extractor import YouTubeExtractor
        api_key = os.environ.get("YOUTUBE_API_KEY")

        if api_key:
            # Get extractor
            extractor = YouTubeExtractor(api_key)

            # Try to get playlist items through the YouTube API
            youtube = extractor.youtube

            # Request playlist items
            request = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50
            )
            response = request.execute()

            # Extract video IDs
            video_ids = []
            for item in response.get("items", []):
                video_id = item.get("snippet", {}).get("resourceId", {}).get("videoId")
                if video_id:
                    video_ids.append(video_id)

            # Save to database for future use
            if video_ids:
                data_access.execute_update(
                    "INSERT OR REPLACE INTO playlists (playlist_id, video_ids, video_count, created_at) VALUES (?, ?, ?, datetime('now'))",
                    (playlist_id, ",".join(video_ids), len(video_ids))
                )

            return video_ids
    except Exception as e:
        print(f"Error getting playlist from YouTube API: {e}")

    return []

def show_concept_signatures(args):
    """
    Show concept signatures for indexed concepts.

    Args:
        args: Command-line arguments
    """
    if not HAS_CONCEPT_GENERATOR:
        print("Error: concept_signature_generator module not available")
        return

    # Get the concept signature generator
    generator = get_concept_signature_generator()

    # Get relationship graph
    graph = generator.relationship_graph

    # Apply filters
    domain = args.filter_domain

    # Print summary
    print(f"\n=== Concept Signatures ===")
    print(f"Total Concepts: {len(graph.concepts)}")

    if domain:
        # Filter concepts by domain
        domain_concepts = {cid: c for cid, c in graph.concepts.items() if c.domain == domain}
        print(f"Domain Filter: {domain}")
        print(f"Matching Concepts: {len(domain_concepts)}")
        concepts = domain_concepts
    else:
        concepts = graph.concepts

    # Print signatures
    for i, (concept_id, concept) in enumerate(list(concepts.items())[:20]):  # Show top 20
        print(f"\n{i+1}. {concept.text} (ID: {concept_id})")
        print(f"   Domain: {concept.domain}")
        print(f"   Class: {concept.concept_class}")
        print(f"   Hierarchy Score: {concept.hierarchy_score:.2f}")
        print(f"   Signature Pattern: {', '.join(concept.signature_pattern[:10])}")
        if concept.definition:
            print(f"   Definition: {concept.definition}")

        # Print related concepts
        if concept.related_concepts:
            print(f"   Related Concepts:")
            for rel_id, rel_data in list(concept.related_concepts.items())[:5]:
                rel_concept = graph.concepts.get(rel_id)
                if rel_concept:
                    print(f"     - {rel_concept.text} ({rel_data.get('type', 'related')})")

    if len(concepts) > 20:
        print(f"\n... and {len(concepts) - 20} more concepts")

def show_relationship_graph(args):
    """
    Display concept relationship graph.

    Args:
        args: Command-line arguments
    """
    if not HAS_CONCEPT_GENERATOR:
        print("Error: concept_signature_generator module not available")
        return

    # Get the concept signature generator
    generator = get_concept_signature_generator()

    # Get relationship graph
    graph = generator.relationship_graph

    # Apply filters
    domain = args.filter_domain

    # Print summary
    print(f"\n=== Concept Relationship Graph ===")
    print(f"Total Concepts: {len(graph.concepts)}")
    print(f"Total Relationships: {sum(len(targets) for targets in graph.adjacency_list.values())}")

    if domain:
        print(f"Domain Filter: {domain}")

    # Print domains
    domains = {}
    for concept in graph.concepts.values():
        domains[concept.domain] = domains.get(concept.domain, 0) + 1

    print(f"\nDomain Distribution:")
    for domain_name, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain_name}: {count} concepts")

    # Print top concepts by hierarchy score
    print(f"\nTop Fundamental Concepts (by hierarchy score):")
    top_concepts = sorted(
        graph.concepts.values(),
        key=lambda c: c.hierarchy_score,
        reverse=True
    )

    if domain:
        top_concepts = [c for c in top_concepts if c.domain == domain]

    for i, concept in enumerate(top_concepts[:15]):
        print(f"  {i+1}. {concept.text}")
        print(f"     Hierarchy Score: {concept.hierarchy_score:.2f}")
        print(f"     Domain: {concept.domain}")
        print(f"     Related Concepts: {len(concept.related_concepts)}")

    # Print most connected concepts
    print(f"\nMost Connected Concepts:")
    connected_concepts = sorted(
        graph.concepts.values(),
        key=lambda c: len(c.related_concepts),
        reverse=True
    )

    if domain:
        connected_concepts = [c for c in connected_concepts if c.domain == domain]

    for i, concept in enumerate(connected_concepts[:10]):
        print(f"  {i+1}. {concept.text}")
        print(f"     Related Concepts: {len(concept.related_concepts)}")
        print(f"     Domain: {concept.domain}")

        # Show top relationships for this concept
        if concept.related_concepts:
            print(f"     Top Related Concepts:")
            for rel_id, rel_data in list(sorted(
                concept.related_concepts.items(),
                key=lambda x: x[1].get("strength", 0),
                reverse=True
            ))[:5]:
                rel_concept = graph.concepts.get(rel_id)
                if rel_concept:
                    print(f"       - {rel_concept.text} ({rel_data.get('type', 'related')}, " +
                          f"strength: {rel_data.get('strength', 0):.2f})")


def main():
    """Main function to run the demo."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # Initialize components
    data_pipeline, search_engine, data_access = init_components(args)

    # Clear cache if requested
    if args.clear_cache:
        cache_clear()
        print("All caches cleared")

    # Optimize database if requested
    if args.optimize_db:
        optimize_database(data_access, search_engine)

    # Show cache statistics if requested
    if args.cache_stats:
        show_cache_stats()

    # Process specific actions
    if args.video:
        # Set language preference
        if args.language == "auto":
            language_preference = ["en", "ru"]
        else:
            language_preference = [args.language]

        # Process or show video details
        process_or_show_video(data_pipeline, search_engine, args.video, language_preference)

    elif args.playlist:
        # Process a playlist
        process_playlist(data_pipeline, search_engine, args.playlist, args)

    elif args.list_concepts:
        # List all indexed concepts
        list_concepts(data_access, args)

    elif args.learning_path:
        # Generate learning path
        generate_learning_path(search_engine, args)

    elif args.search:
        # Search for content
        search_content(search_engine, args)

    elif args.concept_signatures:
        # Show concept signatures
        show_concept_signatures(args)

    elif args.relationship_graph:
        # Display concept relationship graph
        show_relationship_graph(args)

    elif args.analyze_concepts:
        # Analyze concept relationships
        analyze_concepts(search_engine, data_access, args)

    else:
        # No specific action requested, show help
        print("No action specified. Use --help to see available options.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

def show_cache_stats():
    """Show cache statistics."""
    stats = get_cache_stats()

    print("\n=== Cache Statistics ===")
    for cache_type, cache_stats in stats.items():
        print(f"\n{cache_type.capitalize()} Cache:")
        print(f"  Size: {cache_stats.get('size', 0)} / {cache_stats.get('max_size', 0)}")
        print(f"  Hits: {cache_stats.get('hits', 0)}")
        print(f"  Misses: {cache_stats.get('misses', 0)}")
        print(f"  Hit Rate: {cache_stats.get('hit_rate_percent', 0):.2f}%")
        print(f"  TTL: {cache_stats.get('default_ttl', 0)} seconds")
        print(f"  Valid Entries: {cache_stats.get('valid_entries', 0)}")
        print(f"  Expired Entries: {cache_stats.get('expired_entries', 0)}")

def optimize_database(data_access, search_engine):
    """
    Optimize the database.

    Args:
        data_access: DataAccess instance
        search_engine: SearchEngine instance
    """
    print("\n=== Optimizing Database ===")

    try:
        # Execute VACUUM
        print("Running VACUUM...")
        data_access.execute_update("VACUUM")

        # Execute ANALYZE
        print("Running ANALYZE...")
        data_access.execute_update("ANALYZE")

        # Optimize search engine
        print("Optimizing search index...")
        success = search_engine.optimize_database()

        if success:
            print("Database optimization completed successfully")
        else:
            print("Database optimization completed with warnings")

    except Exception as e:
        print(f"Error optimizing database: {e}")

def extract_playlist_id(playlist_url_or_id):
    """
    Extract playlist ID from a URL or return the ID directly.

    Args:
        playlist_url_or_id: Playlist URL or ID

    Returns:
        Playlist ID
    """
    # Check if it's a URL
    if "youtube.com" in playlist_url_or_id or "youtu.be" in playlist_url_or_id:
        # Extract playlist ID from URL
        import re
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/playlist\?list=([^&\s]+)',  # Standard playlist URL
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?.*[\?&]list=([^&\s]+)'  # Video URL with playlist
        ]

        for pattern in patterns:
            match = re.match(pattern, playlist_url_or_id)
            if match:
                return match.group(1)

        print(f"Warning: Could not extract playlist ID from URL: {playlist_url_or_id}")
        return None
    else:
        # Assume it's already a playlist ID
        return playlist_url_or_id

def process_playlist(data_pipeline, search_engine, playlist_url_or_id, args):
    """
    Process all videos in a playlist.

    Args:
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        playlist_url_or_id: Playlist URL or ID
        args: Command-line arguments
    """
    # Extract playlist ID
    playlist_id = extract_playlist_id(playlist_url_or_id)
    if not playlist_id:
        print(f"Error: Invalid playlist URL or ID: {playlist_url_or_id}")
        return

    print(f"Processing playlist: {playlist_id}")

    # Get videos in playlist
    videos = get_playlist_videos(playlist_id)
    if not videos:
        print(f"Error: No videos found in playlist: {playlist_id}")
        return

    print(f"Found {len(videos)} videos in playlist")

    # Apply max videos limit if specified
    if args.max_videos and not args.no_limit:
        print(f"Limiting to {args.max_videos} videos")
        videos = videos[:args.max_videos]

    # Convert video IDs to URLs
    video_urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in videos]

    # Process videos
    processed_videos = process_videos_batch(data_pipeline, search_engine, video_urls, args)

    print(f"\nSummary: Processed {len(processed_videos)} out of {len(videos)} videos from playlist")

    # Show processed video IDs
    if processed_videos:
        print("\nProcessed Videos:")
        for i, video_id in enumerate(processed_videos):
            print(f"  {i+1}. {video_id}")

def process_videos_batch(data_pipeline, search_engine, videos, args):
    """
    Process a batch of videos.

    Args:
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        videos: List of video URLs or IDs
        args: Command-line arguments

    Returns:
        List of processed video IDs
    """
    # Set language preference
    if args.language == "auto":
        language_preference = ["en", "ru"]
    else:
        language_preference = [args.language]

    processed_videos = []

    for video in videos:
        # Process the video
        print(f"\nProcessing video: {video}")

        result = data_pipeline.process_video(video, language_preference)

        if result.get("status") == "completed":
            video_id = result.get("video_id")
            print(f"Successfully processed video: {video_id}")

            # Index the content
            success = search_engine.index_content(result)
            if success:
                print(f"Successfully indexed video content")
                processed_videos.append(video_id)
            else:
                print(f"Failed to index video content")
        else:
            print(f"Failed to process video: {result.get('error', 'Unknown error')}")

    return processed_videos

def analyze_concepts(search_engine, data_access, args):
    """
    Analyze concept relationships across all videos.

    Args:
        search_engine: SearchEngine instance
        data_access: DataAccess instance
        args: Command-line arguments
    """
    # Get all indexed videos
    videos_query = "SELECT video_id, title, domain, theory_practice_ratio FROM videos ORDER BY domain, title"
    videos = data_access.execute_query(videos_query)

    if not videos:
        print("No indexed videos found")
        return

    print(f"\n=== Concept Analysis Across {len(videos)} Videos ===")

    # Apply domain filter
    if args.filter_domain:
        videos = [v for v in videos if v.get("domain") == args.filter_domain]
        print(f"Domain Filter: {args.filter_domain}")
        print(f"Matching Videos: {len(videos)}")

    if not videos:
        print("No videos match the filter criteria")
        return

    # Apply playlist filter
    if args.filter_playlist:
        playlist_videos = get_playlist_videos(args.filter_playlist)
        if playlist_videos:
            videos = [v for v in videos if v.get("video_id") in playlist_videos]
            print(f"Playlist Filter: {args.filter_playlist}")
            print(f"Matching Videos: {len(videos)}")
        else:
            print(f"No videos found for playlist {args.filter_playlist}")
            return

    # Apply max videos limit if specified
    if args.max_videos and not args.no_limit:
        videos = videos[:args.max_videos]
        print(f"Limited to {args.max_videos} videos")

    # Collect all concepts
    all_concepts = {}
    video_concept_map = {}

    for video in videos:
        video_id = video.get("video_id")
        video_concepts = search_engine.get_video_concepts(video_id)

        if video_concepts:
            concepts = video_concepts.get("concepts", [])
            video_concept_map[video_id] = concepts

            for concept in concepts:
                concept_id = concept.get("concept_id")
                concept_text = concept.get("text")

                if concept_id and concept_text:
                    if concept_id in all_concepts:
                        all_concepts[concept_id]["video_count"] += 1
                        all_concepts[concept_id]["videos"].append(video_id)
                    else:
                        all_concepts[concept_id] = {
                            "id": concept_id,
                            "text": concept_text,
                            "domain": concept.get("domain", "unknown"),
                            "concept_class": concept.get("concept_class", "unknown"),
                            "video_count": 1,
                            "videos": [video_id]
                        }

    # Print common concepts
    common_concepts = sorted(
        all_concepts.values(),
        key=lambda c: c["video_count"],
        reverse=True
    )

    print(f"\nTotal Unique Concepts: {len(all_concepts)}")
    print(f"Most Common Concepts (appearing in multiple videos):")
    for i, concept in enumerate(common_concepts[:20]):
        if concept["video_count"] > 1:
            print(f"  {i+1}. {concept['text']} (in {concept['video_count']} videos)")
            print(f"     Class: {concept['concept_class']}")
            print(f"     Domain: {concept['domain']}")

    # Analyze domain distribution
    domain_counts = {}
    for concept in all_concepts.values():
        domain = concept.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"\nConcept Domain Distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count} concepts")

    # Analyze concept class distribution
    theoretical_count = sum(1 for c in all_concepts.values() if c.get("concept_class") == "theoretical")
    practical_count = sum(1 for c in all_concepts.values() if c.get("concept_class") == "practical")

    print(f"\nConcept Class Distribution:")
    print(f"  Theoretical: {theoretical_count} concepts")
    print(f"  Practical: {practical_count} concepts")

    # Calculate video similarity based on shared concepts
    if len(videos) > 1:
        print(f"\nVideo Similarity Analysis (based on shared concepts):")

        # For each pair of videos, calculate Jaccard similarity
        video_ids = list(video_concept_map.keys())

        # Show only the top 10 most similar pairs
        similarities = []

        for i in range(len(video_ids)):
            for j in range(i + 1, len(video_ids)):
                video1 = video_ids[i]
                video2 = video_ids[j]

                concepts1 = set(c.get("concept_id") for c in video_concept_map.get(video1, []))
                concepts2 = set(c.get("concept_id") for c in video_concept_map.get(video2, []))

                # Calculate Jaccard similarity
                intersection = len(concepts1.intersection(concepts2))
                union = len(concepts1.union(concepts2))

                if union > 0:
                    similarity = intersection / union

                    # Find video titles
                    title1 = next((v.get("title") for v in videos if v.get("video_id") == video1), "Unknown")
                    title2 = next((v.get("title") for v in videos if v.get("video_id") == video2), "Unknown")

                    similarities.append({
                        "video1": video1,
                        "video2": video2,
                        "title1": title1,
                        "title2": title2,
                        "similarity": similarity,
                        "shared_concepts": intersection
                    })

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Show top 10 most similar pairs
        for i, sim in enumerate(similarities[:10]):
            print(f"  {i+1}. {sim['title1']} & {sim['title2']}")
            print(f"     Similarity: {sim['similarity']:.2f}")
            print(f"     Shared Concepts: {sim['shared_concepts']}")
            print(f"     Video IDs: {sim['video1']} & {sim['video2']}")
