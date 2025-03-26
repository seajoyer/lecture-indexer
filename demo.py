#!/usr/bin/env python3
"""
Simplified demonstration script for Lecture Video Content Indexer.
Tests the core functionality of the simplified architecture.
"""

import os
import sys
import re
import logging
import argparse
import time
from typing import Dict, List, Any, Optional
import colorama
from colorama import Fore, Style

# Import simplified components
from youtube_extractor import YouTubeExtractor
from transcript_processor import TranscriptProcessor
from data_pipeline import DataPipeline
from search_engine import SearchEngine
from data_access import get_data_access
from cache_manager import cache_clear

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo")

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

    # Debug option
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

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

    # If we're just listing concepts
    if args.list_concepts:
        list_indexed_concepts(data_access, domain_filter=args.filter_domain,
                             video_filter=args.filter_video,
                             playlist_filter=args.filter_playlist)
        sys.exit(0)

    # If we're just showing details for a specific video
    if args.video:
        show_video_concepts(search_engine, args.video)
        sys.exit(0)

    # If we're just searching for a term without processing a video
    if args.search and not args.url:
        search_concepts(search_engine, args.search, args.theory_ratio,
                      domain_filter=args.filter_domain,
                      video_filter=args.filter_video)
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
                        None if args.no_limit else args.max_videos)

        # If search is requested after playlist processing
        if args.search:
            # Extract playlist ID for filtering
            playlist_id = extract_playlist_id(url)
            search_concepts(search_engine, args.search, args.theory_ratio,
                          domain_filter=args.filter_domain,
                          video_filter=args.filter_video,
                          playlist_filter=args.filter_playlist or playlist_id)

        sys.exit(0)

    # Process single video
    print(f"{Fore.CYAN}Processing video: {url}{Style.RESET_ALL}")
    print("This may take a few minutes...")
    start_time = time.time()

    try:
        # Extract video ID
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
                    search_concepts(search_engine, args.search, args.theory_ratio,
                                  domain_filter=args.filter_domain,
                                  video_filter=args.filter_video or video_id)
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

def is_playlist_url(url):
    """Check if URL is a YouTube playlist."""
    return "list=" in url

def extract_playlist_id(url):
    """Extract playlist ID from URL."""
    match = re.search(r'list=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def process_playlist(playlist_url, youtube_extractor, data_pipeline, search_engine, max_videos=None):
    """
    Process all videos in a YouTube playlist.

    Args:
        playlist_url: URL of the YouTube playlist
        youtube_extractor: YouTubeExtractor instance
        data_pipeline: DataPipeline instance
        search_engine: SearchEngine instance
        max_videos: Maximum number of videos to process (None for unlimited)
    """
    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        print(f"{Fore.RED}Invalid playlist URL: {playlist_url}{Style.RESET_ALL}")
        return

    print(f"{Fore.MAGENTA}Processing videos from playlist ID: {playlist_id}{Style.RESET_ALL}")

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
            result = data_pipeline.process_video(url)
            process_time = time.time() - start_time

            if result.get("status") == "completed":
                print(f"{Fore.GREEN}Successfully processed video in {process_time:.2f} seconds{Style.RESET_ALL}")
                print(f"Title: {result['metadata']['title']}")
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

def get_playlist_video_ids(data_access, playlist_id):
    """Get video IDs for a playlist from database."""
    try:
        result = data_access.execute_query(
            "SELECT video_ids FROM playlists WHERE playlist_id = ?",
            (playlist_id,)
        )

        if result and result[0]["video_ids"]:
            return result[0]["video_ids"].split(",")

        return []
    except Exception as e:
        logger.error(f"Error getting playlist video IDs: {e}")
        return []

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

def list_indexed_concepts(data_access, domain_filter=None, video_filter=None, playlist_filter=None):
    """
    List all concepts in the index with optional filtering.

    Args:
        data_access: DataAccess instance
        domain_filter: Optional domain filter
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
        # Get video IDs from playlist if specified
        video_ids = []
        if playlist_filter:
            video_ids = get_playlist_video_ids(data_access, playlist_filter)
            if not video_ids:
                print(f"{Fore.YELLOW}No videos found for playlist {playlist_filter}{Style.RESET_ALL}")
                return

        # Build query for concepts
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

        if video_ids:
            placeholders = ",".join(['?'] * len(video_ids))
            where_clauses.append(f"o.video_id IN ({placeholders})")
            params.extend(video_ids)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        # Group and order
        query += """
        GROUP BY c.concept_id
        ORDER BY c.domain, c.concept_class, video_count DESC, c.total_occurrences DESC
        """

        concepts = data_access.execute_query(query, tuple(params))

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
            import traceback
            traceback.print_exc()

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

    print(f"\n{Fore.BLUE}Theoretical concepts ({len(theoretical_concepts)}):{Style.RESET_ALL}")
    for i, concept in enumerate(theoretical_concepts[:15], 1):
        print(f"{i}. {concept.get('text', 'Unknown')} (occurrences: {concept.get('total_occurrences', 0)})")

    print(f"\n{Fore.GREEN}Practical concepts ({len(practical_concepts)}):{Style.RESET_ALL}")
    for i, concept in enumerate(practical_concepts[:15], 1):
        print(f"{i}. {concept.get('text', 'Unknown')} (occurrences: {concept.get('total_occurrences', 0)})")

    print(f"\n{Fore.CYAN}Run the demo script with '{os.path.basename(__file__)} --search \"query\" --filter-video {video_id}' to search within this video.{Style.RESET_ALL}")

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

                # Show the context - FIXED: Use text field as fallback if context_text is missing
                context_text = result.get('context_text', result.get('text', ''))
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
    print("This script demonstrates the simplified Lecture Video Content Indexer with the following capabilities:")
    print(f" - {Fore.BLUE}Processes{Style.RESET_ALL} educational videos from YouTube")
    print(f" - {Fore.BLUE}Processes{Style.RESET_ALL} entire YouTube playlists")
    print(f" - {Fore.BLUE}Classifies{Style.RESET_ALL} content as theoretical or practical")
    print(f" - {Fore.BLUE}Indexes{Style.RESET_ALL} educational concepts for advanced search")
    print()
    print(f"{Fore.YELLOW}Common commands:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Process a video:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/watch?v=VIDEO_ID")
    print(f"  {Fore.GREEN}Process a playlist:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/playlist?list=PLAYLIST_ID")
    print(f"  {Fore.GREEN}Process playlist with limit:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/playlist?list=PLAYLIST_ID --max-videos 10")
    print(f"  {Fore.GREEN}Process all videos in playlist:{Style.RESET_ALL} {os.path.basename(__file__)} https://www.youtube.com/playlist?list=PLAYLIST_ID --no-limit")
    print(f"  {Fore.GREEN}Search all concepts:{Style.RESET_ALL} {os.path.basename(__file__)} --search \"query\"")
    print(f"  {Fore.GREEN}Search by domain:{Style.RESET_ALL} {os.path.basename(__file__)} --search \"query\" --filter-domain programming")
    print(f"  {Fore.GREEN}List all concepts:{Style.RESET_ALL} {os.path.basename(__file__)} --list-concepts")
    print(f"  {Fore.GREEN}Show video details:{Style.RESET_ALL} {os.path.basename(__file__)} --video VIDEO_ID")
    print()
    print(f"{Fore.CYAN}API Key Setup:{Style.RESET_ALL}")
    print("  Set the YOUTUBE_API_KEY environment variable (recommended):")
    print(f"  {Fore.YELLOW}export YOUTUBE_API_KEY='your_api_key'{Style.RESET_ALL}")
    print()

if __name__ == "__main__":
    main()
