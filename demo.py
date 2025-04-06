#!/usr/bin/env python3
"""
Enhanced demo script for the Lecture Video Content Indexer.
Provides various commands to demonstrate and test the system.
Added support for processing video playlists.
"""

import argparse
import sys
import os
import json
import pprint
import logging
from typing import Dict, List, Any, Optional
import time
import textwrap
from tabulate import tabulate
import concurrent.futures
from tqdm import tqdm  # Progress bar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use try-except blocks to handle import errors gracefully
try:
    from data_access import get_data_access
    from youtube_extractor import YouTubeExtractor
    from data_pipeline import DataPipeline
    from search_engine import SearchEngine
    from unified_concept_extractor import UnifiedConceptExtractor
    from concept_dedup import ConceptDedupExtension, apply_concept_deduplication
    from concept_signature_generator import ConceptSignatureGenerator, enhance_search_engine
    from mlcs_algorithm import MLCSAlgorithm
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    print(f"Error: Could not import required modules: {e}")
    print("Make sure you've installed all dependencies and are in the correct directory.")
    sys.exit(1)

class Demo:
    """Enhanced demo class for showcasing the Lecture Video Content Indexer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the demo with configuration.

        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = config or {
            "youtube_api_key": os.environ.get("YOUTUBE_API_KEY", "test_api_key"),
            "output_dir": "data/processed",
            "index_dir": "data/index",
            "db_path": "data/index/indexer.db"
        }

        # Create necessary directories
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["index_dir"], exist_ok=True)

        # Initialize components
        try:
            self.data_access = get_data_access(self.config["db_path"])
            self.youtube_extractor = YouTubeExtractor(self.config["youtube_api_key"])
            self.data_pipeline = DataPipeline(self.config)
            self.search_engine = SearchEngine(self.config)
            self.concept_extractor = UnifiedConceptExtractor()
            self.concept_dedup = ConceptDedupExtension()
            self.signature_generator = ConceptSignatureGenerator(self.config)

            # Enhance search engine with improved learning paths
            enhance_search_engine(self.search_engine)

            print("Demo initialized successfully.")
            print(f"Using database at: {self.config['db_path']}")

        except Exception as e:
            logger.error(f"Error initializing demo: {e}")
            raise e

    def process_video(self, url: str, language_preference: List[str] = None) -> None:
        """
        Process a YouTube video through the pipeline.

        Args:
            url: YouTube video URL
            language_preference: List of language preferences
        """
        if language_preference is None:
            language_preference = ['en', 'ru']

        print(f"Processing video: {url}")
        print(f"Language preference: {', '.join(language_preference)}")

        try:
            # Process video
            start_time = time.time()
            result = self.data_pipeline.process_video(url, language_preference)
            processing_time = time.time() - start_time

            # Print results
            if result.get("status") == "completed":
                metadata = result.get("metadata", {})
                transcript = result.get("transcript", {})
                domain_features = result.get("domain_features", {})

                print("\n=== Video Processed Successfully ===")
                print(f"Video ID: {result.get('video_id')}")
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"Channel: {metadata.get('channel', 'N/A')}")
                print(f"Domain: {metadata.get('domain', 'unknown')}")
                print(f"Language: {transcript.get('language', 'unknown')}")
                print(f"Segments: {len(transcript.get('segments', []))}")
                print(f"Key Concepts: {len(domain_features.get('key_concepts', []))}")
                print(f"Theoretical Concepts: {len(domain_features.get('theoretical_concepts', []))}")
                print(f"Practical Concepts: {len(domain_features.get('practical_concepts', []))}")
                print(f"Processing Time: {processing_time:.2f} seconds")

                # Print key concepts
                self.print_concepts(domain_features.get("key_concepts", []), limit=50)

                return result
            else:
                print("\n=== Video Processing Failed ===")
                print(f"Status: {result.get('status')}")
                print(f"Error: {result.get('error')}")
                return result

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            print(f"Error: {e}")
            return {"status": "error", "error": str(e), "video_url": url}

    def process_playlist(self, url: str, language_preference: List[str] = None,
                        max_videos: int = 10,
                        parallel: bool = False) -> None:
        """
        Process a YouTube playlist through the pipeline.

        Args:
            url: YouTube playlist URL
            language_preference: List of language preferences
            max_videos: Maximum number of videos to process
            parallel: Whether to process videos in parallel
        """
        if language_preference is None:
            language_preference = ['en', 'ru']

        print(f"Processing playlist: {url}")
        print(f"Language preference: {', '.join(language_preference)}")
        print(f"Maximum videos: {max_videos}")
        print(f"Parallel processing: {'Yes' if parallel else 'No'}")

        try:
            # Validate and extract playlist ID
            valid, playlist_id = self.youtube_extractor.validate_playlist_url(url)
            if not valid or not playlist_id:
                print(f"Invalid YouTube playlist URL: {url}")
                return

            # Extract playlist metadata
            playlist_metadata = self.youtube_extractor.extract_playlist_metadata(playlist_id)
            print(f"\n=== Playlist Information ===")
            print(f"Playlist ID: {playlist_id}")
            print(f"Title: {playlist_metadata.get('title', 'N/A')}")
            print(f"Channel: {playlist_metadata.get('channel', 'N/A')}")
            print(f"Total videos: {playlist_metadata.get('item_count', 0)}")

            # Extract video information from playlist
            playlist_videos = self.youtube_extractor.extract_playlist_videos(
                playlist_id, max_results=max_videos
            )

            if not playlist_videos:
                print("No videos found in playlist")
                return

            print(f"Found {len(playlist_videos)} videos in playlist")

            # Display video list
            print("\n=== Videos in Playlist ===")
            for i, video in enumerate(playlist_videos):
                print(f"{i+1}. {video.get('title', 'Unknown')} (ID: {video.get('video_id', 'N/A')})")

            # Process videos
            results = []
            successful = 0
            failed = 0

            if parallel and len(playlist_videos) > 1:
                print("\nProcessing videos in parallel...")
                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(playlist_videos))) as executor:
                    # Create a list of futures
                    futures = []
                    for video in playlist_videos:
                        video_id = video.get('video_id')
                        if not video_id:
                            continue
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        futures.append(executor.submit(self.process_video, video_url, language_preference))

                    # Process results as they complete with progress bar
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing videos"):
                        try:
                            result = future.result()
                            results.append(result)
                            if result.get("status") == "completed":
                                successful += 1
                            else:
                                failed += 1
                        except Exception as e:
                            logger.error(f"Error processing video: {e}")
                            failed += 1
            else:
                print("\nProcessing videos sequentially...")
                # Process videos sequentially with progress bar
                for video in tqdm(playlist_videos, desc="Processing videos"):
                    video_id = video.get('video_id')
                    if not video_id:
                        continue
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    try:
                        result = self.process_video(video_url, language_preference)
                        results.append(result)
                        if result.get("status") == "completed":
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Error processing video: {e}")
                        failed += 1

            # Print summary
            print("\n=== Playlist Processing Summary ===")
            print(f"Total videos: {len(playlist_videos)}")
            print(f"Successfully processed: {successful}")
            print(f"Failed: {failed}")

            # If we have data access, save playlist information
            if self.data_access:
                # Build playlist data
                playlist_data = {
                    "playlist_id": playlist_id,
                    "title": playlist_metadata.get('title', ''),
                    "channel": playlist_metadata.get('channel', ''),
                    "video_ids": ",".join([v.get('video_id', '') for v in playlist_videos if v.get('video_id')]),
                    "video_count": len(playlist_videos),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # Save to database
                try:
                    query = """
                    INSERT OR REPLACE INTO playlists (
                        playlist_id, title, channel, video_ids, video_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    self.data_access.execute_update(query, (
                        playlist_data["playlist_id"],
                        playlist_data["title"],
                        playlist_data["channel"],
                        playlist_data["video_ids"],
                        playlist_data["video_count"],
                        playlist_data["created_at"],
                        playlist_data["updated_at"]
                    ))
                    print("Playlist information saved to database")
                except Exception as e:
                    logger.error(f"Error saving playlist to database: {e}")

            return results

        except Exception as e:
            logger.error(f"Error processing playlist: {e}")
            print(f"Error: {e}")
            return []

    def index_content(self, video_id: str) -> None:
        """
        Index video content for search.

        Args:
            video_id: YouTube video ID
        """
        try:
            # First load the processed content
            filepath = None

            # Look for the processed file
            for filename in os.listdir(self.config["output_dir"]):
                if filename.startswith(f"{video_id}_"):
                    filepath = os.path.join(self.config["output_dir"], filename)
                    break

            if not filepath:
                print(f"No processed file found for video ID: {video_id}")
                return

            # Load the processed content
            with open(filepath, 'r', encoding='utf-8') as f:
                processed_result = json.load(f)

            # Index the content
            print(f"Indexing content for video: {video_id}")
            success = self.search_engine.index_content(processed_result)

            if success:
                print(f"Successfully indexed content for video: {video_id}")
            else:
                print(f"Failed to index content for video: {video_id}")

        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            print(f"Error: {e}")

    def list_concepts(self, domain: str = None, limit: int = 20) -> None:
        """
        List concepts in the database with domain filter.

        Args:
            domain: Optional domain filter
            limit: Maximum number of concepts to display per category
        """
        try:
            # Get concepts from database
            concepts = self.data_access.list_concepts(domain_filter=domain)

            # Group concepts
            theoretical = [c for c in concepts if c.get("concept_class") == "theoretical"]
            practical = [c for c in concepts if c.get("concept_class") == "practical"]

            print("\n=== Concepts Summary ===")
            print(f"Total Concepts: {len(concepts)}")
            print(f"Theoretical Concepts: {len(theoretical)}")
            print(f"Practical Concepts: {len(practical)}")

            # Print theoretical concepts
            print("\nTheoretical Concepts:")
            self.print_concept_list(theoretical, limit)

            # Print practical concepts
            print("\nPractical Concepts:")
            self.print_concept_list(practical, limit)

        except Exception as e:
            logger.error(f"Error listing concepts: {e}")
            print(f"Error: {e}")

    def list_playlists(self, limit: int = 10) -> None:
        """
        List playlists in the database.

        Args:
            limit: Maximum number of playlists to display
        """
        try:
            # Query playlists
            query = """
            SELECT * FROM playlists
            ORDER BY updated_at DESC
            LIMIT ?
            """
            playlists = self.data_access.execute_query(query, (limit,))

            if not playlists:
                print("No playlists found in the database")
                return

            print("\n=== Playlists ===")
            headers = ["#", "ID", "Title", "Channel", "Videos", "Updated"]
            rows = []

            for i, playlist in enumerate(playlists):
                # Build the row
                row = [
                    i+1,
                    playlist.get("playlist_id", "N/A"),
                    playlist.get("title", "N/A"),
                    playlist.get("channel", "N/A"),
                    playlist.get("video_count", 0),
                    playlist.get("updated_at", "N/A")
                ]
                rows.append(row)

            # Print table
            print(tabulate(rows, headers=headers, tablefmt="pretty"))

        except Exception as e:
            logger.error(f"Error listing playlists: {e}")
            print(f"Error: {e}")

    def print_concept_list(self, concepts: List[Dict[str, Any]], limit: int = 20) -> None:
        """
        Print a formatted list of concepts.

        Args:
            concepts: List of concept dictionaries
            limit: Maximum number of concepts to display
        """
        if not concepts:
            print("  No concepts found.")
            return

        # Sort by frequency and score
        concepts = sorted(concepts, key=lambda x: (
            x.get("frequency", 0) * 0.6 + x.get("score", 0) * 0.4
        ), reverse=True)

        # Limit the number of concepts to display
        display_concepts = concepts[:limit]

        # Print in a formatted way
        for i, concept in enumerate(display_concepts):
            print(f"  {i+1}. {concept.get('text', 'N/A')} (ID: {concept.get('concept_id', 'N/A')})")

        if len(concepts) > limit:
            print(f"  ... and {len(concepts) - limit} more")

    def print_concepts(self, concepts: List[Dict[str, Any]], limit: int = 20) -> None:
        """
        Print concepts in a detailed tabular format.

        Args:
            concepts: List of concept dictionaries
            limit: Maximum number of concepts to display
        """
        if not concepts:
            print("No concepts found.")
            return

        # Sort by score and frequency
        concepts = sorted(concepts, key=lambda x: (
            x.get("score", 0) * 0.6 + x.get("frequency", 0) * 0.4
        ), reverse=True)

        # Limit the number of concepts to display
        display_concepts = concepts[:limit]

        # Prepare table data
        headers = ["#", "Concept", "Class", "Score", "Frequency", "Definition"]
        rows = []

        for i, concept in enumerate(display_concepts):
            # Truncate long definitions for display
            definition = concept.get("definition", "")
            if definition and len(definition) > 50:
                definition = definition[:47] + "..."

            # Build the row
            row = [
                i+1,
                concept.get("text", "N/A"),
                concept.get("concept_class", "N/A"),
                f"{concept.get('score', 0):.2f}",
                concept.get("frequency", 0),
                definition
            ]
            rows.append(row)

        # Print table
        print("\n=== Top Concepts ===")
        print(tabulate(rows, headers=headers, tablefmt="pretty"))

        if len(concepts) > limit:
            print(f"...and {len(concepts) - limit} more concepts")

    def search(self, query: str, domain: str = None) -> None:
        """
        Search for content with improved results formatting.

        Args:
            query: Search query text
            domain: Optional domain filter
        """
        try:
            # Build search query
            search_query = {
                "original_text": query,
                "filters": {},
                "pagination": {"page": 1, "limit": 10}
            }

            if domain:
                search_query["filters"]["domain"] = domain

            # Execute search
            print(f"\nSearching for: {query}" + (f" in domain: {domain}" if domain else ""))
            results = self.search_engine.search(search_query)

            # Print results
            if not results or not results.get("results"):
                print("No results found.")
                return

            print(f"\n=== Search Results ({results.get('totalResults', 0)} total) ===")

            # Format and print each result
            for i, result in enumerate(results.get("results", [])):
                result_type = result.get("result_type", "unknown")
                print(f"\n{i+1}. [{result_type.upper()}] {result.get('text', 'N/A')}")

                if "video_title" in result:
                    print(f"   Video: {result.get('video_title', 'N/A')}")

                if "context_text" in result:
                    # Format and truncate context text
                    context = result.get("context_text", "")
                    if len(context) > 100:
                        context = context[:97] + "..."
                    wrapped = textwrap.fill(context, width=80, initial_indent="   ", subsequent_indent="     ")
                    print(wrapped)

                if "start_time" in result and result["start_time"] is not None:
                    start_time = result.get("start_time", 0)
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    print(f"   Time: {minutes}:{seconds:02d}")

                if "concept_id" in result:
                    print(f"   Concept ID: {result.get('concept_id', 'N/A')}")

        except Exception as e:
            logger.error(f"Error executing search: {e}")
            print(f"Error: {e}")

    def extract_concepts_from_text(self, text: str, domain: str = "physics", language: str = "en") -> None:
        """
        Extract concepts from provided text using the UnifiedConceptExtractor.

        Args:
            text: Input text
            domain: Content domain
            language: Language code
        """
        try:
            print(f"\nExtracting concepts from text (domain: {domain}, language: {language})")
            concepts = self.concept_extractor.extract_concepts(text, domain, language)

            # Print extracted concepts
            print(f"\nExtracted {len(concepts)} concepts:")
            self.print_concepts(concepts)

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            print(f"Error: {e}")

    def generate_learning_path(self, concept_ids: List[str], theory_practice_ratio: float = 0.5) -> None:
        """
        Generate a learning path from specified concepts.

        Args:
            concept_ids: List of concept IDs to include
            theory_practice_ratio: Desired theory/practice ratio (0.0-1.0)
        """
        try:
            print(f"\nGenerating learning path with {len(concept_ids)} concepts")
            print(f"Theory/Practice Ratio: {theory_practice_ratio}")

            # Generate learning path
            learning_path = self.search_engine.generate_learning_path(
                concept_ids, theory_practice_ratio
            )

            if not learning_path or not learning_path.get("concepts"):
                print("Could not generate learning path with the provided concepts.")
                return

            # Print learning path
            print("\n=== Learning Path ===")
            print(f"Total Concepts: {learning_path.get('total_concepts', 0)}")
            print(f"Theoretical: {learning_path.get('theoretical_concepts', 0)}")
            print(f"Practical: {learning_path.get('practical_concepts', 0)}")

            # Print concepts in order
            print("\nConcepts in Order:")
            for i, concept in enumerate(learning_path.get("concepts", [])):
                print(f"\n{i+1}. {concept.get('text', 'N/A')} "
                      f"({concept.get('concept_class', 'unknown')})")

                # Print recommended videos
                if "recommended_videos" in concept and concept["recommended_videos"]:
                    for j, video in enumerate(concept["recommended_videos"][:2]):
                        print(f"   Video {j+1}: {video.get('title', 'N/A')}")

                # Print relationships
                if "prerequisites" in concept and concept["prerequisites"]:
                    prereq_texts = []
                    for prereq_id in concept["prerequisites"][:3]:
                        # Find the prerequisite concept in the path
                        prereq = next((c for c in learning_path.get("concepts", [])
                                     if c.get("concept_id") == prereq_id), None)
                        if prereq:
                            prereq_texts.append(prereq.get("text", prereq_id))

                    if prereq_texts:
                        print(f"   Prerequisites: {', '.join(prereq_texts)}")

        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            print(f"Error: {e}")

    def deduplicate_concepts_demo(self, video_id: str = None) -> None:
        """
        Demonstrate concept deduplication on a video's concepts.

        Args:
            video_id: Optional video ID to use for demonstration
        """
        try:
            if not video_id:
                # Use a sample of concepts from the database
                concepts = self.data_access.list_concepts()[:50]
                video_id = "sample"
            else:
                # Get concepts for the specified video
                video_concepts = self.search_engine.get_video_concepts(video_id)
                if not video_concepts:
                    print(f"No concepts found for video: {video_id}")
                    return
                concepts = video_concepts.get("concepts", [])

            print(f"\nDemonstrating concept deduplication on {len(concepts)} concepts")

            # Print before deduplication
            print("\n=== Before Deduplication ===")
            self.print_concept_list(concepts, limit=20)

            # Deduplicate concepts
            deduplicated = self.concept_dedup.deduplicate_concepts(concepts)

            # Print after deduplication
            print("\n=== After Deduplication ===")
            self.print_concept_list(deduplicated, limit=20)

            # Print reduction statistics
            reduction = len(concepts) - len(deduplicated)
            reduction_percent = (reduction / len(concepts) * 100) if concepts else 0
            print(f"\nReduction: {reduction} concepts ({reduction_percent:.1f}%)")

        except Exception as e:
            logger.error(f"Error demonstrating concept deduplication: {e}")
            print(f"Error: {e}")

    def generate_concept_signatures(self, video_id: str) -> None:
        """
        Generate and display concept signatures for a video.

        Args:
            video_id: Video ID
        """
        try:
            # Get video concepts
            video_concepts = self.search_engine.get_video_concepts(video_id)
            if not video_concepts:
                print(f"No concepts found for video: {video_id}")
                return

            concepts = video_concepts.get("concepts", [])
            print(f"\nGenerating concept signatures for {len(concepts)} concepts in video {video_id}")

            # Prepare a simplified result structure for the signature generator
            processed_result = {
                "video_id": video_id,
                "metadata": {"domain": video_concepts.get("video", {}).get("domain", "unknown")},
                "transcript": {"segments": video_concepts.get("timeline", [])},
                "domain_features": {"key_concepts": concepts}
            }

            # Generate signatures
            result = self.signature_generator.process_video_concepts(processed_result)

            # Extract signatures
            signatures = result.get("domain_features", {}).get("concept_signatures", [])

            if not signatures:
                print("No signatures generated.")
                return

            # Print signatures
            print(f"\n=== Generated {len(signatures)} Concept Signatures ===")

            for i, signature in enumerate(signatures[:10]):  # Limit to top 10
                print(f"\n{i+1}. {signature.get('text', 'N/A')}")
                print(f"   Signature Pattern: {signature.get('signature_pattern', [])}")
                print(f"   Hierarchy Score: {signature.get('hierarchy_score', 0):.3f}")
                print(f"   Confidence: {signature.get('confidence', 0):.3f}")

                # Print definition if available
                if "definition" in signature and signature["definition"]:
                    print(f"   Definition: {signature['definition']}")

                # Print related concepts
                related = signature.get("related_concepts", {})
                if related:
                    # Take top 3 related concepts
                    top_related = list(related.items())[:3]
                    related_texts = [f"{rel_id} ({rel_data.get('type', 'related')})"
                                    for rel_id, rel_data in top_related]
                    print(f"   Related: {', '.join(related_texts)}")

            if len(signatures) > 10:
                print(f"\n...and {len(signatures) - 10} more signatures")

        except Exception as e:
            logger.error(f"Error generating concept signatures: {e}")
            print(f"Error: {e}")


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Lecture Video Content Indexer Demo")

    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process video command
    process_parser = subparsers.add_parser("process", help="Process a YouTube video")
    process_parser.add_argument("url", help="YouTube video URL")
    process_parser.add_argument("--language", "-l", nargs="+", default=["en", "ru"],
                               help="Language preference (e.g., en ru)")

    # Process playlist command
    playlist_parser = subparsers.add_parser("process-playlist", help="Process a YouTube playlist")
    playlist_parser.add_argument("url", help="YouTube playlist URL")
    playlist_parser.add_argument("--language", "-l", nargs="+", default=["en", "ru"],
                               help="Language preference (e.g., en ru)")
    playlist_parser.add_argument("--max-videos", "-m", type=int, default=10,
                               help="Maximum number of videos to process")
    playlist_parser.add_argument("--parallel", "-p", action="store_true",
                               help="Process videos in parallel")

    # List playlists command
    list_playlists_parser = subparsers.add_parser("list-playlists", help="List processed playlists")
    list_playlists_parser.add_argument("--limit", "-l", type=int, default=10,
                                     help="Maximum number of playlists to display")

    # Index content command
    index_parser = subparsers.add_parser("index", help="Index video content for search")
    index_parser.add_argument("video_id", help="YouTube video ID")

    # List concepts command
    list_parser = subparsers.add_parser("list-concepts", help="List concepts in the database")
    list_parser.add_argument("--domain", "-d", help="Filter concepts by domain")
    list_parser.add_argument("--limit", "-l", type=int, default=20,
                            help="Maximum number of concepts to display per category")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for content")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--domain", "-d", help="Filter results by domain")

    # Extract concepts command
    extract_parser = subparsers.add_parser("extract", help="Extract concepts from text")
    extract_parser.add_argument("text", help="Text to extract concepts from")
    extract_parser.add_argument("--domain", "-d", default="physics",
                               help="Domain for concept extraction")
    extract_parser.add_argument("--language", "-l", default="en",
                               help="Language code for text")

    # Generate learning path command
    path_parser = subparsers.add_parser("path", help="Generate a learning path")
    path_parser.add_argument("concept_ids", nargs="+", help="Concept IDs to include in the path")
    path_parser.add_argument("--ratio", "-r", type=float, default=0.5,
                            help="Desired theory/practice ratio (0.0-1.0)")

    # Deduplicate concepts command
    dedup_parser = subparsers.add_parser("dedup", help="Demonstrate concept deduplication")
    dedup_parser.add_argument("--video_id", "-v", help="Optional video ID for demonstration")

    # Generate concept signatures command
    signature_parser = subparsers.add_parser("signatures", help="Generate concept signatures")
    signature_parser.add_argument("video_id", help="Video ID to generate signatures for")

    # Parse arguments
    args = parser.parse_args()

    # Initialize demo
    try:
        demo = Demo()
    except Exception as e:
        print(f"Error initializing demo: {e}")
        return 1

    # Execute command
    try:
        if args.command == "process":
            demo.process_video(args.url, args.language)
        elif args.command == "process-playlist":
            demo.process_playlist(args.url, args.language, args.max_videos, args.parallel)
        elif args.command == "list-playlists":
            demo.list_playlists(args.limit)
        elif args.command == "index":
            demo.index_content(args.video_id)
        elif args.command == "list-concepts":
            demo.list_concepts(args.domain, args.limit)
        elif args.command == "search":
            demo.search(args.query, args.domain)
        elif args.command == "extract":
            demo.extract_concepts_from_text(args.text, args.domain, args.language)
        elif args.command == "path":
            demo.generate_learning_path(args.concept_ids, args.ratio)
        elif args.command == "dedup":
            demo.deduplicate_concepts_demo(args.video_id)
        elif args.command == "signatures":
            demo.generate_concept_signatures(args.video_id)
        else:
            # If no command or invalid command, print help
            parser.print_help()
            return 0
    except Exception as e:
        print(f"Error executing command: {e}")
        logger.error(f"Error executing command: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
