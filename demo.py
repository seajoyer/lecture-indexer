"""
Update to the demo.py file for improved concept listing.
Enhances list_concepts functionality to use the unified concept extractor approach.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Set, Any, Optional
import time
import re

# Import project modules
from data_pipeline import DataPipeline
from search_engine import SearchEngine
from data_access import get_data_access
from cache_manager import get_cache_stats, cache_clear

# New imports for enhanced concept extraction
try:
    from unified_concept_extractor import UnifiedConceptExtractor
    UNIFIED_EXTRACTOR_AVAILABLE = True
except ImportError:
    UNIFIED_EXTRACTOR_AVAILABLE = False
    print("Warning: UnifiedConceptExtractor not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length while preserving whole words.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    # Find the last space before max_length
    last_space = text[:max_length].rfind(' ')
    if last_space > 0:
        return text[:last_space] + "..."
    else:
        # If no space found, just cut at max_length
        return text[:max_length] + "..."

def extract_playlist_id(playlist_url_or_id: str) -> Optional[str]:
    """
    Extract playlist ID from a URL or return the ID directly.

    Args:
        playlist_url_or_id: Playlist URL or ID

    Returns:
        Playlist ID or None if invalid
    """
    # Check if it's a URL
    if "youtube.com" in playlist_url_or_id or "youtu.be" in playlist_url_or_id:
        # Extract playlist ID from URL
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

def extract_video_id(video_url_or_id: str) -> Optional[str]:
    """
    Extract video ID from a URL or return the ID directly.

    Args:
        video_url_or_id: Video URL or ID

    Returns:
        Video ID or None if invalid
    """
    # Check if it's a URL
    if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
        # Extract video ID from URL
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',  # Standard YouTube URL
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^&\s]+)'  # Short YouTube URL
        ]

        for pattern in patterns:
            match = re.match(pattern, video_url_or_id)
            if match:
                return match.group(1)

        print(f"Warning: Could not extract video ID from URL: {video_url_or_id}")
        return None
    else:
        # Assume it's already a video ID
        return video_url_or_id

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

    # New option for direct extraction
    parser.add_argument("--use-unified-extractor", action="store_true", help="Use the unified concept extractor directly")

    # Learning path options
    parser.add_argument("--concepts", nargs="+", help="Concept IDs for learning path generation")

    # System options
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all caches before running")
    parser.add_argument("--language", choices=["en", "ru", "auto"], default="auto", help="Preferred language for processing (auto=detect)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Limit option
    parser.add_argument("--limit", type=int, default=0, help="Limit number of concepts shown")

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

    # Initialize unified concept extractor if requested and available
    unified_extractor = None
    if args.use_unified_extractor and UNIFIED_EXTRACTOR_AVAILABLE:
        language = args.filter_language or args.language
        if language == "auto":
            language = "en"
        unified_extractor = UnifiedConceptExtractor(language)
        print(f"Initialized UnifiedConceptExtractor with language: {language}")

    logger.info("Components initialized")

    return data_pipeline, search_engine, data_access, unified_extractor

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

def get_playlist_videos(playlist_id):
    """
    Get video IDs in a playlist.

    Args:
        playlist_id: YouTube playlist ID

    Returns:
        List of video IDs
    """
    # Check if we have this playlist in the database
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

def process_concepts_unified_extractor(unified_extractor, segments, domain="physics", language="en"):
    """
    Process concepts using the unified concept extractor directly.
    This function bypasses the database and generates concepts directly.

    Args:
        unified_extractor: UnifiedConceptExtractor instance
        segments: List of transcript segments
        domain: Content domain
        language: Language code

    Returns:
        List of concepts
    """
    if not unified_extractor:
        return []

    # Extract concepts using the unified extractor
    print(f"Extracting concepts directly using UnifiedConceptExtractor ({domain}, {language})...")

    # If segments is empty, return empty list
    if not segments:
        print("No segments provided")
        return []

    # Extract concepts
    concepts = unified_extractor.extract_concepts_from_segments(segments, domain, language)

    print(f"Extracted {len(concepts)} concepts directly")
    return concepts

def list_concepts(data_access, args, unified_extractor=None):
    """
    List all indexed concepts with optional filtering.

    Args:
        data_access: DataAccess instance
        args: Command-line arguments
        unified_extractor: Optional UnifiedConceptExtractor instance
    """
    # Apply filters
    domain_filter = args.filter_domain
    video_filter = args.filter_video
    playlist_filter = args.filter_playlist
    language_filter = args.filter_language

    # If using unified extractor directly, get segments and process them
    if args.use_unified_extractor and unified_extractor and video_filter:
        # Get video details and segments
        video_query = "SELECT * FROM videos WHERE video_id = ?"
        video_result = data_access.execute_query(video_query, (video_filter,))

        if not video_result:
            print(f"Video {video_filter} not found in database")
            return

        video = video_result[0]
        domain = video.get("domain", "physics")
        language = video.get("language", "en")

        # Get segments
        segments_query = "SELECT * FROM segments WHERE video_id = ? ORDER BY start_time"
        segments = data_access.execute_query(segments_query, (video_filter,))

        if not segments:
            print(f"No segments found for video {video_filter}")
            return

        # Extract concepts using unified extractor
        concepts = process_concepts_unified_extractor(
            unified_extractor,
            segments,
            domain=domain,
            language=language
        )
    else:
        # Use database query to get concepts
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

    # Limit the number of concepts if specified
    if args.limit > 0:
        theoretical = theoretical[:args.limit]
        practical = practical[:args.limit]

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
    if args.use_unified_extractor:
        print("Using: UnifiedConceptExtractor (direct extraction)")

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

    # Parse concepts from comma-separated list if needed
    concept_ids = []
    for concept_arg in args.concepts:
        # Check if it's a comma-separated list
        if ',' in concept_arg:
            concept_ids.extend([c.strip() for c in concept_arg.split(',')])
        else:
            concept_ids.append(concept_arg)

    # Set theory ratio
    theory_ratio = args.theory_ratio if args.theory_ratio is not None else 0.5

    # Generate learning path
    learning_path = search_engine.generate_learning_path(
        concept_ids,
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

    # Handle different theory practice ratio formats
    theory_practice_ratio = learning_path.get('theory_practice_ratio', 0.5)
    if isinstance(theory_practice_ratio, dict):
        actual_ratio = theory_practice_ratio.get('actual', 0.5)
    else:
        actual_ratio = theory_practice_ratio

    print(f"Theory/Practice Ratio: {actual_ratio:.2f}")

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
        if len(text) > 100:
            text = truncate_text(text)
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

def show_cache_stats():
    """Show cache statistics."""
    stats = get_cache_stats()

    print("\n=== Cache Statistics ===")
    for cache_type, cache_stats in stats.items():
        print(f"\n{cache_type.capitalize()} Cache:")
        print(f"  Size: {cache_stats.get('size', 0)} items")
        print(f"  Hits: {cache_stats.get('hits', 0)}")
        print(f"  Misses: {cache_stats.get('misses', 0)}")
        hit_rate = cache_stats.get('hit_rate', 0) * 100
        print(f"  Hit Rate: {hit_rate:.2f}%")
        print(f"  TTL: {cache_stats.get('ttl_seconds', 0)} seconds")

def main():
    """Main function to run the demo."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # Initialize components
    data_pipeline, search_engine, data_access, unified_extractor = init_components(args)

    # Clear cache if requested
    if args.clear_cache:
        cache_clear()
        print("All caches cleared")

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
        list_concepts(data_access, args, unified_extractor)

    elif args.learning_path:
        # Generate learning path
        generate_learning_path(search_engine, args)

    elif args.search:
        # Search for content
        search_content(search_engine, args)

    else:
        # No specific action requested, show help
        print("No action specified. Use --help to see available options.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
