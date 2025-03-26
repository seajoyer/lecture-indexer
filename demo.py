#!/usr/bin/env python3
"""
Enhanced demonstration script for Lecture Video Content Indexer.
Shows the system's capabilities with actual YouTube videos,
with improved user experience and powerful filtering options.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Set
import time
import re
from datetime import datetime
import traceback
import getpass
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo")

# Try to import colorama for colored terminal output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    # Create dummy color objects
    class DummyFore:
        def __getattr__(self, name):
            return ""
    class DummyStyle:
        def __getattr__(self, name):
            return ""
    Fore = DummyFore()
    Style = DummyStyle()
    HAS_COLOR = False

    # Print warning about installing colorama
    print("Note: For colored output, install colorama: pip install colorama")

def main():

    # Add this code after imports, before main()
    class MockSearchEngine:
        """A fallback search engine implementation when database is not available."""

        def __init__(self, config):
            self.config = config
            logger.info("Initialized MockSearchEngine (fallback implementation)")

        def search(self, query):
            """Return empty search results."""
            return {
                "results": [],
                "totalResults": 0,
                "theoreticalResults": 0,
                "practicalResults": 0,
                "executionTimeMs": 0,
                "message": "Search is disabled when database is not initialized."
            }

        def index_content(self, processed_result):
            """Mock indexing."""
            logger.info(f"Mock indexing content for video {processed_result.get('video_id', 'unknown')}")
            return True

        def get_video_concepts(self, video_id):
            """Return empty concepts."""
            return None

        def get_concept_details(self, concept_id):
            """Return empty concept details."""
            return None

        def generate_learning_path(self, concept_ids, theory_practice_ratio=0.5, domain=None):
            """Return empty learning path."""
            return None

        def optimize_database(self):
            """Mock database optimization."""
            return True

        def rebuild_search_indexes(self):
            """Mock index rebuilding."""
            return True


    def initialize_search_engine(config):
        """Initialize search engine with fallback to mock implementation."""
        try:
            from search_retrieval.python.search_engine import SearchEngine
            search_engine = SearchEngine(config)
            return search_engine
        except RuntimeError as e:
            if "Database context not initialized" in str(e):
                print(f"{Fore.YELLOW}Warning: Database not initialized. Search functionality will be limited.{Style.RESET_ALL}")
                return MockSearchEngine(config)
            else:
                # Re-raise other RuntimeErrors
                raise

    """Main function for the demonstration script."""
    parser = argparse.ArgumentParser(description='Lecture Video Content Indexer Demo')

    # Source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('url', nargs='?', help='YouTube video URL, playlist URL, or .txt file with URLs (optional)')
    source_group.add_argument('--video', help='Show details for specific video ID (already indexed)')
    source_group.add_argument('--batch', help='File with YouTube URLs (one per line)')
    source_group.add_argument('--list-concepts', action='store_true', help='List all indexed concepts')

    # API key handling options
    api_key_group = parser.add_mutually_exclusive_group()
    api_key_group.add_argument('--api-key', help='YouTube API key (NOT RECOMMENDED - use environment variable instead)')
    api_key_group.add_argument('--prompt-api-key', action='store_true', help='Prompt for YouTube API key (safer)')

    # Search options
    parser.add_argument('--search', help='Search query after processing')
    parser.add_argument('--theory-ratio', type=float, default=0.5,
                        help='Theory/practice ratio for search (0-1, 1=all theoretical)')

    # Playlist options
    parser.add_argument('--no-limit', action='store_true', help='Process all videos in a playlist without limit')
    parser.add_argument('--max-videos', type=int, default=10,
                        help='Maximum number of videos to process from a playlist (ignored with --no-limit)')

    # Filtering options
    parser.add_argument('--filter-domain', choices=['mathematics', 'programming', 'physics'],
                        help='Filter by domain')
    parser.add_argument('--filter-video', help='Filter to a specific video ID')
    parser.add_argument('--filter-playlist', help='Filter to a specific playlist ID')

    # Debug option
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Print header for better UX
    print_header()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # Check for API key - use secure handling
    api_key = get_api_key(args)
    if not api_key:
        print(f"{Fore.RED}No YouTube API key provided. Set YOUTUBE_API_KEY environment variable "
              f"or use --prompt-api-key option.{Style.RESET_ALL}")
        sys.exit(1)

    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/index", exist_ok=True)
    os.makedirs("data/playlists", exist_ok=True)

    # Import components (do this here to show clear error if dependencies are missing)
    try:
        from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor
        from data_acquisition.youtube_api.python.data_pipeline import DataPipeline
        from search_retrieval.python.search_engine import SearchEngine
    except ImportError as e:
        print(f"{Fore.RED}Error importing required modules: {e}{Style.RESET_ALL}")
        print("Make sure you have installed the project and activated the correct Python environment.")
        sys.exit(1)

    # Initialize components
    pipeline_config = {"youtube_api_key": api_key, "output_dir": "data/processed"}
    search_config = {"index_dir": "data/index"}

    youtube_extractor = YouTubeDataExtractor(api_key)
    data_pipeline = DataPipeline(pipeline_config)

    # Use our new initialization function to handle database unavailability
    search_engine = initialize_search_engine(search_config)

    # If we're just listing concepts
    if args.list_concepts:
        list_indexed_concepts(search_engine, domain_filter=args.filter_domain,
                              video_filter=args.filter_video,
                              playlist_filter=args.filter_playlist)
        sys.exit(0)

    # If we're just showing details for a specific video
    if args.video:
        show_video_concepts(search_engine, args.video)
        sys.exit(0)

    # If we're processing a batch of videos
    if args.batch:
        process_batch(args.batch, data_pipeline, search_engine)

        # If search is requested after batch processing
        if args.search:
            search_concepts(search_engine, args.search, args.theory_ratio,
                           domain_filter=args.filter_domain,
                           video_filter=args.filter_video,
                           playlist_filter=args.filter_playlist)
        sys.exit(0)

    # If we're just searching for a term without processing a video
    if args.search and not args.url:
        search_concepts(search_engine, args.search, args.theory_ratio,
                       domain_filter=args.filter_domain,
                       video_filter=args.filter_video,
                       playlist_filter=args.filter_playlist)
        sys.exit(0)

    # Process single video or a list of videos
    url = args.url
    if not url:
        # Use a good educational video as default example
        url = "https://www.youtube.com/watch?v=rfscVS0vtbw"  # Python tutorial
        print(f"{Fore.YELLOW}No URL provided, using example: {url}{Style.RESET_ALL}")

        # Process single video
        print(f"{Fore.CYAN}Processing video: {url}{Style.RESET_ALL}")
        print("This may take a few minutes...")
        start_time = time.time()
    elif url.lower().endswith('.txt'):
        # Treat the URL as a text file containing a list of videos
        print(f"{Fore.CYAN}Processing videos from file: {url}{Style.RESET_ALL}")
        try:
            # Call the batch processing function
            process_batch(url, data_pipeline, search_engine)

            # If search is requested after batch processing
            if args.search:
                search_concepts(search_engine, args.search, args.theory_ratio,
                               domain_filter=args.filter_domain,
                               video_filter=args.filter_video,
                               playlist_filter=args.filter_playlist)
            sys.exit(0)
        except Exception as e:
            print(f"{Fore.RED}Error processing video list from file: {e}{Style.RESET_ALL}")
            sys.exit(1)
    elif "list=" in url:
        # Treat the URL as a YouTube playlist
        print(f"{Fore.CYAN}Detected YouTube playlist: {url}{Style.RESET_ALL}")
        try:
            # Get playlist ID for possible filtering later
            playlist_id = extract_playlist_id(url)

            # Process the playlist - remove limit if --no-limit is specified
            max_videos = None if args.no_limit else args.max_videos
            process_playlist(url, youtube_extractor, data_pipeline, search_engine, max_videos)

            # If search is requested after playlist processing
            if args.search:
                # If no specific filter is set, use the playlist ID as filter
                playlist_filter = args.filter_playlist or playlist_id
                search_concepts(search_engine, args.search, args.theory_ratio,
                               domain_filter=args.filter_domain,
                               video_filter=args.filter_video,
                               playlist_filter=playlist_filter)
            sys.exit(0)
        except Exception as e:
            print(f"{Fore.RED}Error processing playlist: {e}{Style.RESET_ALL}")
            if args.debug:
                traceback.print_exc()
            sys.exit(1)
    else:
        # Process single video
        print(f"{Fore.CYAN}Processing video: {url}{Style.RESET_ALL}")
        print("This may take a few minutes...")
        start_time = time.time()

    try:
        # Extract video ID for possible filtering later
        is_valid, video_id = youtube_extractor.validate_video_url(url)
        if not is_valid:
            print(f"{Fore.RED}Invalid YouTube URL: {url}{Style.RESET_ALL}")
            sys.exit(1)

        result = data_pipeline.process_video(url)
        process_time = time.time() - start_time

        if result.get("status") == "completed":
            display_results(result, process_time)

            # Index the results
            print(f"\n{Fore.CYAN}Indexing video content...{Style.RESET_ALL}")
            index_success = search_engine.index_content(result)

            if index_success:
                print(f"{Fore.GREEN}Successfully indexed video content{Style.RESET_ALL}")

                # Search if requested
                if args.search:
                    # If no specific filter is set, use the video ID as filter
                    video_filter = args.filter_video or video_id
                    search_concepts(search_engine, args.search, args.theory_ratio,
                                   domain_filter=args.filter_domain,
                                   video_filter=video_filter,
                                   playlist_filter=args.filter_playlist)
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

                    # Show how to list concepts for this video
                    print(f"\n{Fore.YELLOW}List all concepts in this video:{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Run '{os.path.basename(__file__)} --list-concepts --filter-video {video_id}'{Style.RESET_ALL}")

                    print(f"\n{Fore.YELLOW}Run '{os.path.basename(__file__)} --search \"your query\" --theory-ratio 0.7 --filter-video {video_id}' to search within this video{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to index video content{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Error processing video: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error running demo: {e}{Style.RESET_ALL}")
        if args.debug:
            traceback.print_exc()

def list_indexed_concepts(search_engine, domain_filter=None, video_filter=None, playlist_filter=None):
    """
    List all concepts in the index with optional filtering.

    Args:
        search_engine: SearchEngine instance
        domain_filter: Optional domain filter ("mathematics", "programming", "physics")
        video_filter: Optional video ID filter
        playlist_filter: Optional playlist ID filter
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

    if filter_desc:
        print(f"Filters: {', '.join(filter_desc)}")

    try:
        # Get all concepts from the database
        db_path = Path(search_engine.index_dir) / "index.db"

        if not os.path.exists(db_path):
            print(f"{Fore.RED}No index database found at {db_path}. Process some videos first.{Style.RESET_ALL}")
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Handle playlist filter by getting video IDs
        video_list = None
        if playlist_filter:
            video_list = get_playlist_video_ids(playlist_filter)
            if not video_list:
                print(f"{Fore.YELLOW}Playlist {playlist_filter} not found or contains no videos{Style.RESET_ALL}")
                conn.close()
                return

        # Start building the query
        query = """
        SELECT c.*, COUNT(DISTINCT o.video_id) as video_count
        FROM concepts c
        JOIN occurrences o ON c.concept_id = o.concept_id
        JOIN videos v ON o.video_id = v.video_id
        """

        # Add WHERE clause if we have filters
        where_clauses = []
        params = []

        if domain_filter:
            where_clauses.append("c.domain = ?")
            params.append(domain_filter)

        if video_filter:
            where_clauses.append("o.video_id = ?")
            params.append(video_filter)

        if video_list:
            placeholders = ','.join(['?'] * len(video_list))
            where_clauses.append(f"o.video_id IN ({placeholders})")
            params.extend(video_list)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        # Group and order
        query += """
        GROUP BY c.concept_id
        ORDER BY c.domain, c.concept_class, video_count DESC, c.total_occurrences DESC
        """

        cursor.execute(query, params)
        concepts = cursor.fetchall()

        if not concepts:
            print(f"{Fore.YELLOW}No concepts found matching filters.{Style.RESET_ALL}")
            conn.close()
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
        print(f"Total concepts: {total_concepts}")
        print(f"{Fore.BLUE}Theoretical concepts: {theoretical_count} ({theoretical_count / total_concepts * 100:.1f}%){Style.RESET_ALL}")
        print(f"{Fore.GREEN}Practical concepts: {practical_count} ({practical_count / total_concepts * 100:.1f}%){Style.RESET_ALL}")

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
                    occurrences = concept['total_occurrences']
                    print(f"  {i}. {concept['text']} (videos: {videos}, occurrences: {occurrences})")

                if len(concepts_dict['theoretical']) > 30:
                    print(f"  ... and {len(concepts_dict['theoretical']) - 30} more")

            # Display practical concepts
            if concepts_dict['practical']:
                print(f"\n{Fore.GREEN}  Practical concepts ({len(concepts_dict['practical'])}):{Style.RESET_ALL}")
                for i, concept in enumerate(concepts_dict['practical'][:30], 1):  # Limit to 30 per category
                    videos = concept['video_count']
                    occurrences = concept['total_occurrences']
                    print(f"  {i}. {concept['text']} (videos: {videos}, occurrences: {occurrences})")

                if len(concepts_dict['practical']) > 30:
                    print(f"  ... and {len(concepts_dict['practical']) - 30} more")

        # Provide hints for refining search
        print(f"\n{Fore.CYAN}Use '--search \"concept name\"' to find specific concepts.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error listing concepts: {e}{Style.RESET_ALL}")
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()

    finally:
        if 'conn' in locals() and conn:
            conn.close()

def get_api_key(args):
    """
    Get YouTube API key from environment variable or command-line arguments.
    Uses a secure method to get the API key.

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

    # If --prompt-api-key, securely prompt for the key
    if args.prompt_api_key:
        print(f"{Fore.YELLOW}Enter your YouTube API key (input will be hidden):{Style.RESET_ALL}")
        api_key = getpass.getpass("API Key: ")
        if api_key:
            # Set as environment variable for this session
            os.environ["YOUTUBE_API_KEY"] = api_key
            return api_key

    # Last resort: use the argument directly (not recommended)
    if args.api_key:
        logger.warning("Using API key from command line argument (not secure)")
        return args.api_key

    return ""

def process_playlist(playlist_url, youtube_extractor, data_pipeline, search_engine, max_videos=None):
    """
    Process all videos in a YouTube playlist.

    Args:
        playlist_url: URL of the YouTube playlist
        youtube_extractor: YouTubeDataExtractor instance
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        max_videos: Maximum number of videos to process (None for unlimited)
    """
    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        print(f"{Fore.RED}Invalid playlist URL: {playlist_url}{Style.RESET_ALL}")
        return

    print(f"{Fore.MAGENTA}Extracting videos from playlist ID: {playlist_id}{Style.RESET_ALL}")

    # Get video URLs from playlist
    video_urls = get_playlist_videos(youtube_extractor, playlist_id, max_videos)

    if not video_urls:
        print(f"{Fore.RED}No videos found in playlist or error accessing playlist{Style.RESET_ALL}")
        return

    limit_str = f" (limited to {max_videos})" if max_videos else " (no limit)"
    print(f"{Fore.GREEN}Found {len(video_urls)} videos in playlist{limit_str}{Style.RESET_ALL}")

    # Save playlist info for filtering
    save_playlist_mapping(playlist_id, [extract_video_id(url) for url in video_urls])

    # Process each video
    successful = 0
    for i, url in enumerate(video_urls, 1):
        print(f"\n{Fore.MAGENTA}[{i}/{len(video_urls)}] Processing playlist video: {url}{Style.RESET_ALL}")
        start_time = time.time()

        try:
            result = data_pipeline.process_video(url)
            process_time = time.time() - start_time

            if result.get("status") == "completed":
                print(f"{Fore.GREEN}Successfully processed video in {process_time:.2f} seconds{Style.RESET_ALL}")
                print(f"Title: {result['metadata']['title']}")
                print(f"Domain: {result['metadata']['domain']}")
                print(f"Classification: {result['theory_practice_results']['classification']}")

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

def save_playlist_mapping(playlist_id, video_ids):
    """
    Save mapping of playlist ID to video IDs for filtering purposes.

    Args:
        playlist_id: YouTube playlist ID
        video_ids: List of video IDs in the playlist
    """
    # Create playlists directory if it doesn't exist
    os.makedirs("data/playlists", exist_ok=True)

    # Save mapping
    with open(f"data/playlists/{playlist_id}.json", 'w') as f:
        json.dump({
            "playlist_id": playlist_id,
            "video_ids": video_ids,
            "updated_at": datetime.now().isoformat()
        }, f)

def get_playlist_video_ids(playlist_id):
    """
    Get video IDs for a playlist from saved mapping.

    Args:
        playlist_id: YouTube playlist ID

    Returns:
        List of video IDs or None if mapping not found
    """
    try:
        filepath = f"data/playlists/{playlist_id}.json"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                mapping = json.load(f)
                return mapping.get("video_ids", [])
        return None
    except Exception as e:
        logger.error(f"Error loading playlist mapping: {e}")
        return None

def extract_playlist_id(url):
    """Extract playlist ID from a YouTube URL."""
    # Pattern to match playlist IDs
    playlist_pattern = r'(?:list=)([a-zA-Z0-9_-]+)'
    match = re.search(playlist_pattern, url)
    if match:
        return match.group(1)
    return None

def extract_video_id(url):
    """
    Extract video ID from a YouTube URL.

    Args:
        url: YouTube video URL

    Returns:
        Video ID or None if not found
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',  # Standard URL
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^\?\s]+)',  # Shortened URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\?\s]+)',  # Embedded URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^\?\s]+)',  # Old embed URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([^\?\s]+)'  # YouTube shorts URL
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_playlist_videos(youtube_extractor, playlist_id, max_videos=None):
    """
    Get video URLs from a YouTube playlist.

    Args:
        youtube_extractor: YouTubeDataExtractor instance
        playlist_id: YouTube playlist ID
        max_videos: Maximum number of videos to return (None for unlimited)

    Returns:
        List of video URLs
    """
    try:
        # Get a reference to the YouTube API client
        youtube = youtube_extractor.youtube

        # Track videos retrieved
        video_urls = []
        next_page_token = None
        page_size = 50  # Maximum allowed by API

        while True:
            # Build request
            request_params = {
                "part": "snippet",
                "maxResults": page_size,
                "playlistId": playlist_id
            }

            if next_page_token:
                request_params["pageToken"] = next_page_token

            playlist_request = youtube.playlistItems().list(**request_params)
            playlist_response = playlist_request.execute()

            # Extract video IDs from this page
            items = playlist_response.get("items", [])
            for item in items:
                video_id = item["snippet"]["resourceId"]["videoId"]
                video_urls.append(f"https://www.youtube.com/watch?v={video_id}")

                # Check if we've reached the limit
                if max_videos is not None and len(video_urls) >= max_videos:
                    print(f"{Fore.YELLOW}Reached maximum of {max_videos} videos.{Style.RESET_ALL}")
                    return video_urls

            # Get next page token
            next_page_token = playlist_response.get("nextPageToken")

            # If no more pages or all videos retrieved, break
            if not next_page_token:
                break

            # Status update for large playlists
            print(f"{Fore.CYAN}Retrieved {len(video_urls)} videos, getting more...{Style.RESET_ALL}")

        return video_urls

    except Exception as e:
        logger.error(f"Error fetching playlist videos: {e}")
        return []

def process_batch(batch_file, data_pipeline, search_engine):
    """
    Process multiple videos from a batch file.

    Args:
        batch_file: Path to file containing URLs (one per line)
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
    """
    try:
        if not os.path.exists(batch_file):
            print(f"{Fore.RED}File not found: {batch_file}{Style.RESET_ALL}")
            return

        with open(batch_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not urls:
            print(f"{Fore.RED}No valid URLs found in batch file{Style.RESET_ALL}")
            return

        print(f"{Fore.CYAN}Processing {len(urls)} videos from batch file...{Style.RESET_ALL}")

        batch_id = f"batch_{int(time.time())}"
        video_ids = []

        successful = 0
        for i, url in enumerate(urls, 1):
            print(f"\n{Fore.MAGENTA}[{i}/{len(urls)}] Processing: {url}{Style.RESET_ALL}")
            start_time = time.time()

            try:
                result = data_pipeline.process_video(url)
                process_time = time.time() - start_time

                if result.get("status") == "completed":
                    print(f"{Fore.GREEN}Successfully processed video in {process_time:.2f} seconds{Style.RESET_ALL}")
                    print(f"Title: {result['metadata']['title']}")
                    print(f"Domain: {result['metadata']['domain']}")
                    print(f"Classification: {result['theory_practice_results']['classification']}")

                    video_id = result.get("video_id")
                    if video_id:
                        video_ids.append(video_id)

                    # Display key concepts
                    concepts = result['domain_features'].get('key_concepts', [])
                    theoretical_concepts = [c for c in concepts if c.get('theoretical', False)]
                    practical_concepts = [c for c in concepts if not c.get('theoretical', False)]

                    print(f"{Fore.BLUE}Theoretical concepts: {len(theoretical_concepts)}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}Practical concepts: {len(practical_concepts)}{Style.RESET_ALL}")

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

        # Save batch info for filtering (similar to playlists)
        if video_ids:
            save_playlist_mapping(batch_id, video_ids)

        print(f"\n{Fore.GREEN}Batch processing completed: {successful}/{len(urls)} videos processed successfully{Style.RESET_ALL}")
        if video_ids:
            print(f"{Fore.CYAN}You can search within this batch using: --search \"query\" --filter-playlist {batch_id}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}You can list all concepts from this batch using: --list-concepts --filter-playlist {batch_id}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error reading batch file: {e}{Style.RESET_ALL}")

def show_video_concepts(search_engine, video_id):
    """Show concepts extracted from a specific video."""
    print(f"{Fore.CYAN}Getting concepts for video ID: {video_id}{Style.RESET_ALL}")

    video_concepts = search_engine.get_video_concepts(video_id)

    if not video_concepts:
        print(f"{Fore.RED}No concepts found for video ID: {video_id}{Style.RESET_ALL}")
        print("Make sure the video has been processed and indexed.")
        return

    video = video_concepts.get('video', {})
    concepts = video_concepts.get('concepts', [])
    theory_practice_ratio = video.get('theory_practice_ratio', 0)

    print(f"\n{Fore.MAGENTA}===== Video Information ====={Style.RESET_ALL}")
    print(f"Title: {Fore.YELLOW}{video.get('title', 'Unknown')}{Style.RESET_ALL}")
    print(f"Domain: {Fore.GREEN}{video.get('domain', 'Unknown')}{Style.RESET_ALL}")
    print(f"Theory/Practice Ratio: {theory_practice_ratio:.2f}")
    print(f"Video Link: {Fore.CYAN}https://www.youtube.com/watch?v={video_id}{Style.RESET_ALL}")

    # Visual representation of theory/practice ratio
    theory_percent = theory_practice_ratio * 100
    practice_percent = (1 - theory_practice_ratio) * 100

    theory_bar = "█" * int(theory_percent // 5)
    practice_bar = "█" * int(practice_percent // 5)

    print(f"{Fore.BLUE}Theory:   {theory_bar}{Style.RESET_ALL} ({theory_percent:.1f}%)")
    print(f"{Fore.GREEN}Practice: {practice_bar}{Style.RESET_ALL} ({practice_percent:.1f}%)")

    # Group concepts by type
    theoretical_concepts = [c for c in concepts if c.get('concept_class') == 'theoretical']
    practical_concepts = [c for c in concepts if c.get('concept_class') == 'practical']
    other_concepts = [c for c in concepts if c.get('concept_class') not in ('theoretical', 'practical')]

    print(f"\n{Fore.BLUE}Theoretical concepts ({len(theoretical_concepts)}):{Style.RESET_ALL}")
    for i, concept in enumerate(theoretical_concepts[:15], 1):
        print(f"{i}. {concept.get('text', 'Unknown')} (occurrences: {concept.get('total_occurrences', 0)})")

    print(f"\n{Fore.GREEN}Practical concepts ({len(practical_concepts)}):{Style.RESET_ALL}")
    for i, concept in enumerate(practical_concepts[:15], 1):
        print(f"{i}. {concept.get('text', 'Unknown')} (occurrences: {concept.get('total_occurrences', 0)})")

    # Get theory-practice patterns
    patterns = video_concepts.get('theory_practice_patterns', [])

    if patterns:
        print(f"\n{Fore.MAGENTA}===== Theory/Practice Patterns ====={Style.RESET_ALL}")
        for i, pattern in enumerate(patterns[:5], 1):
            pattern_type = pattern.get('pattern_type', 'Unknown')
            pattern_subtype = pattern.get('pattern_subtype', 'Unknown')
            start_time = pattern.get('start_time', 0)
            timecode = format_timecode(start_time)

            print(f"{i}. {Fore.YELLOW}{pattern_type} ({pattern_subtype}){Style.RESET_ALL} at {timecode}")
            video_url = get_video_url_with_timecode(video_id, start_time)
            print(f"   Link: {Fore.CYAN}{video_url}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}Run the demo script with '{os.path.basename(__file__)} --search \"query\" --filter-video {video_id}' to search within this video.{Style.RESET_ALL}")

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
    print(f"Theory/Practice ratio: {tp_results['theory_practice_ratio']:.2f}")

    # Visual representation of theory/practice ratio
    theory_percent = tp_results['theory_practice_ratio'] * 100
    practice_percent = (1 - tp_results['theory_practice_ratio']) * 100

    theory_bar = "█" * int(theory_percent // 5)
    practice_bar = "█" * int(practice_percent // 5)

    print(f"{Fore.BLUE}Theory:   {theory_bar}{Style.RESET_ALL} ({theory_percent:.1f}%)")
    print(f"{Fore.GREEN}Practice: {practice_bar}{Style.RESET_ALL} ({practice_percent:.1f}%)")

    # Display key concepts
    print(f"\n{Fore.MAGENTA}===== Key Concepts ====={Style.RESET_ALL}")

    key_concepts = result['domain_features'].get('key_concepts', [])
    if not key_concepts:
        print(f"{Fore.YELLOW}No key concepts extracted. This may indicate a content analysis issue.{Style.RESET_ALL}")
    else:
        theoretical_concepts = [c for c in key_concepts if c.get('theoretical', False)]
        practical_concepts = [c for c in key_concepts if not c.get('theoretical', False)]

        print(f"{Fore.BLUE}Theoretical concepts ({len(theoretical_concepts)}):{Style.RESET_ALL}")
        for i, concept in enumerate(theoretical_concepts[:10], 1):
            print(f"{i}. {concept['text']} (frequency: {concept['frequency']})")

        print(f"\n{Fore.GREEN}Practical concepts ({len(practical_concepts)}):{Style.RESET_ALL}")
        for i, concept in enumerate(practical_concepts[:10], 1):
            print(f"{i}. {concept['text']} (frequency: {concept['frequency']})")

    # Display theory to practice transitions
    tp_patterns = result['theory_practice_patterns']
    t2p_sequences = tp_patterns.get('theory_to_practice_sequences', [])
    if t2p_sequences:
        print(f"\n{Fore.MAGENTA}===== Theory to Practice Transitions ====={Style.RESET_ALL}")
        for i, seq in enumerate(t2p_sequences[:3], 1):
            print(f"{i}. {Fore.YELLOW}{seq.get('pattern_type', 'Unknown')} transition:{Style.RESET_ALL}")
            start_time = None
            for segment in seq.get('segments', [])[:2]:
                content_type = segment.get('content_type', 'unknown')
                color = Fore.BLUE if content_type == 'theoretical' else Fore.GREEN if content_type == 'practical' else Fore.WHITE
                text = segment.get('text', '')
                # Get the start time of the first segment for the timecode link
                if start_time is None:
                    start_time = segment.get('start_time', 0)
                # Limit text length while being smart about not cutting in the middle of a word
                if len(text) > 100:
                    text = text[:97] + "..."
                print(f"   {color}- [{content_type}] {text}{Style.RESET_ALL}")

            # Add timecode and link
            if start_time is not None:
                timecode = format_timecode(start_time)
                video_url = get_video_url_with_timecode(video_id, start_time)
                print(f"   {Fore.YELLOW}Timecode: {timecode}{Style.RESET_ALL}")
                print(f"   {Fore.CYAN}Link: {video_url}{Style.RESET_ALL}")

    # Display transcript summary
    transcript = result.get('transcript', {})
    segments = transcript.get('segments', [])
    print(f"\n{Fore.MAGENTA}===== Transcript Summary ====={Style.RESET_ALL}")
    print(f"Total segments: {len(segments)}")

    if segments:
        # Show a small sample of segments with their classification
        print("Sample segments:")
        for i, segment in enumerate(segments[:5], 1):
            content_type = segment.get('content_type', 'unknown')
            color = Fore.BLUE if content_type == 'theoretical' else Fore.GREEN if content_type == 'practical' else Fore.WHITE
            emoji = "🧠" if content_type == 'theoretical' else "🛠️" if content_type == 'practical' else "📝"
            text = segment.get('text', '')
            start_time = segment.get('start_time', 0)
            timecode = format_timecode(start_time)

            # Limit text length while being smart about not cutting in the middle of a word
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"{i}. {emoji} {color}[{content_type}] {text}{Style.RESET_ALL}")

            # Display timecode and link
            video_url = get_video_url_with_timecode(video_id, start_time)
            print(f"   {Fore.YELLOW}Timecode: {timecode}{Style.RESET_ALL}")
            print(f"   {Fore.CYAN}Link: {video_url}{Style.RESET_ALL}")

def search_concepts(search_engine, query: str, theory_practice_ratio: float = None, domain_filter: str = None,
                   video_filter: str = None, playlist_filter: str = None):
    """
    Search for concepts in the indexed content with optional filtering.

    Args:
        search_engine: SearchEngine instance
        query: Search query text
        theory_practice_ratio: Theory/practice ratio filter (0-1)
        domain_filter: Optional domain filter
        video_filter: Optional video ID filter
        playlist_filter: Optional playlist ID filter
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
        "pagination": {
            "offset": 0,
            "limit": 10
        }
    }

    # Handle playlist filter by getting video IDs
    video_list = None
    if playlist_filter:
        video_list = get_playlist_video_ids(playlist_filter)
        if not video_list:
            print(f"{Fore.YELLOW}Playlist {playlist_filter} not found or contains no videos{Style.RESET_ALL}")
            return None

        # Add video list to filters
        structured_query["filters"]["video_ids"] = video_list

    # Add video filter if specified
    if video_filter:
        structured_query["filters"]["video_id"] = video_filter

    # Execute search
    try:
        results = search_engine.search(structured_query)

        total = results.get('totalResults', 0)
        theoretical = results.get('theoreticalResults', 0)
        practical = results.get('practicalResults', 0)

        print(f"Total results: {total}")

        if total > 0:
            theory_percent = (theoretical / total) * 100 if total > 0 else 0
            practice_percent = (practical / total) * 100 if total > 0 else 0

            print(f"{theory_color}Theoretical: {theoretical} ({theory_percent:.1f}%){Style.RESET_ALL}")
            print(f"{practice_color}Practical: {practical} ({practice_percent:.1f}%){Style.RESET_ALL}")

            # Visual representation of theory/practice ratio
            theory_bar = "█" * int(theory_percent // 5)
            practice_bar = "█" * int(practice_percent // 5)

            print(f"{theory_color}Theory:   {theory_bar}{Style.RESET_ALL} ({theory_percent:.1f}%)")
            print(f"{practice_color}Practice: {practice_bar}{Style.RESET_ALL} ({practice_percent:.1f}%)")

        search_results = results.get('results', [])
        if search_results:
            print("\nTop results:")
            for i, result in enumerate(search_results[:15], 1):
                context_type = result.get('context_type', 'unknown')
                emoji = "🧠" if context_type == 'theoretical' else "🛠️" if context_type == 'practical' else "📝"
                color = theory_color if context_type == 'theoretical' else practice_color if context_type == 'practical' else Style.RESET_ALL

                # Get the concept text (might be None for segment-based results)
                concept_text = result.get('text')
                if concept_text:
                    print(f"{i}. {emoji} {color}Concept: {concept_text}{Style.RESET_ALL}")
                else:
                    print(f"{i}. {emoji} {color}Segment match{Style.RESET_ALL}")

                print(f"   Video: {result.get('video_title', 'Unknown')}")

                # Show the context
                context_text = result.get('context_text', '')
                if len(context_text) > 120:
                    context_text = context_text[:117] + "..."
                print(f"   Context: {context_text}")

                # Add timecode and video link information
                start_time = result.get('start_time', 0)
                video_id = result.get('video_id')
                timecode = format_timecode(start_time)
                video_url = get_video_url_with_timecode(video_id, start_time)

                print(f"   {Fore.YELLOW}Timecode: {timecode}{Style.RESET_ALL}")
                print(f"   {Fore.CYAN}Video Link: {video_url}{Style.RESET_ALL}")
                print()
        else:
            print(f"{Fore.YELLOW}No results found.{Style.RESET_ALL}")
            print("Try another search query or check if content has been indexed.")

        return results
    except Exception as e:
        print(f"{Fore.RED}Error performing search: {e}{Style.RESET_ALL}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(traceback.format_exc())
        return None

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
    print(" Theory vs. Practice Classification Demo ".center(width, "="))
    print("=" * width + f"{Style.RESET_ALL}")
    print()
    print("This script demonstrates the Lecture Video Content Indexer with the following capabilities:")
    print(f" - {Fore.BLUE}Processes{Style.RESET_ALL} educational videos from YouTube")
    print(f" - {Fore.BLUE}Classifies{Style.RESET_ALL} content as theoretical or practical")
    print(f" - {Fore.BLUE}Indexes{Style.RESET_ALL} educational concepts for advanced search")
    print(f" - {Fore.BLUE}Extracts{Style.RESET_ALL} theory-to-practice transitions")
    print()
    print(f"{Fore.YELLOW}Common commands:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Process a video:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/watch?v=VIDEO_ID")
    print(f"  {Fore.GREEN}Process a playlist:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/playlist?list=PLAYLIST_ID")
    print(f"  {Fore.GREEN}Process with no limit:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/playlist?list=PLAYLIST_ID --no-limit")
    print(f"  {Fore.GREEN}Search all concepts:{Style.RESET_ALL} {os.path.basename(__file__)} --search \"query\"")
    print(f"  {Fore.GREEN}Search by domain:{Style.RESET_ALL} {os.path.basename(__file__)} --search \"query\" --filter-domain programming")
    print(f"  {Fore.GREEN}List all concepts:{Style.RESET_ALL} {os.path.basename(__file__)} --list-concepts")
    print(f"  {Fore.GREEN}List domain concepts:{Style.RESET_ALL} {os.path.basename(__file__)} --list-concepts --filter-domain mathematics")
    print(f"  {Fore.GREEN}Show video details:{Style.RESET_ALL} {os.path.basename(__file__)} --video VIDEO_ID")
    print()
    print(f"{Fore.CYAN}API Key Setup:{Style.RESET_ALL}")
    print("  1. Set the YOUTUBE_API_KEY environment variable (recommended):")
    print(f"     {Fore.YELLOW}export YOUTUBE_API_KEY='your_api_key'{Style.RESET_ALL}")
    print("  2. Or use the --prompt-api-key flag for secure input")
    print()

if __name__ == "__main__":
    main()
