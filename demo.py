#!/usr/bin/env python3
"""
Video Lecture Content Indexer - Demo CLI.

This script provides a command-line interface for the Video Lecture Content Indexer,
allowing users to process videos, search for concepts, generate learning paths,
manage the concept repository, and review concept candidates.
"""

import os
import sys
import argparse
import logging
import json
import time
import uuid
from typing import List, Dict, Any, Optional, TextIO, Tuple
from datetime import datetime

# Import system components
from concept_repository import get_concept_repository
from learning_path_generator import get_learning_path_generator

# Import concept candidate extractor
from concept_candidate_extractor import get_concept_candidate_extractor

# Try to import optional components with graceful fallbacks
try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLOR_AVAILABLE = True
except ImportError:
    COLOR_AVAILABLE = False
    logging.warning("colorama not available - running without color support")

# Try to import video processing components
try:
    from youtube_extractor import YouTubeExtractor
    from transcript_processor import TranscriptProcessor
    from unified_concept_extractor import UnifiedConceptExtractor
    from data_pipeline import DataPipeline
    from data_access import get_data_access
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False
    logging.warning("Video processing components not available - some features will be disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConceptIndexerCLI:
    """
    Command-line interface for the Video Lecture Content Indexer.

    This class provides commands for processing videos, searching for concepts,
    generating learning paths, managing the concept repository, and reviewing concept candidates.
    """

    def __init__(self):
        """Initialize the CLI with necessary components."""
        self.concept_repository = get_concept_repository()
        self.learning_path_generator = get_learning_path_generator()

        # Initialize concept candidate extractor
        self.candidate_extractor = get_concept_candidate_extractor()

        # Initialize video processing components if available
        self.data_pipeline = None
        self.data_access = None

        if VIDEO_PROCESSING_AVAILABLE:
            try:
                # Get the data access instance
                self.data_access = get_data_access()

                # Initialize data pipeline
                config = {
                    "output_dir": "data/processed",
                    "youtube_api_key": os.environ.get("YOUTUBE_API_KEY", "")
                }
                self.data_pipeline = DataPipeline(config)
                logger.info("Video processing components initialized")
            except Exception as e:
                logger.error(f"Error initializing video processing components: {e}")

        # Print banner
        self._print_banner()

    def _print_banner(self):
        """Print application banner."""
        banner = f"""
{'='*80}
Video Lecture Content Indexer - CLI Demo
{'='*80}
Version 1.0.0
Concepts loaded: {len(self.concept_repository.concepts)}
Concept candidates: {len(self.candidate_extractor.candidates)}
{'='*80}
"""
        self._print_colorized(banner, Fore.CYAN)

    def _print_colorized(self, text: str, color=None, file: TextIO = sys.stdout):
        """
        Print text with optional color.

        Args:
            text: Text to print
            color: Optional color (from colorama.Fore)
            file: Output file (default: stdout)
        """
        if COLOR_AVAILABLE and color:
            print(f"{color}{text}{Style.RESET_ALL}", file=file)
        else:
            print(text, file=file)

    def _print_error(self, text: str):
        """
        Print error message.

        Args:
            text: Error message
        """
        self._print_colorized(f"ERROR: {text}", Fore.RED, file=sys.stderr)

    def _print_success(self, text: str):
        """
        Print success message.

        Args:
            text: Success message
        """
        self._print_colorized(f"SUCCESS: {text}", Fore.GREEN)

    def _print_warning(self, text: str):
        """
        Print warning message.

        Args:
            text: Warning message
        """
        self._print_colorized(f"WARNING: {text}", Fore.YELLOW)

    def _print_info(self, text: str):
        """
        Print info message.

        Args:
            text: Info message
        """
        self._print_colorized(text, Fore.CYAN)

    def _print_concept(self, concept: Dict[str, Any], detailed: bool = False):
        """
        Print concept information.

        Args:
            concept: Concept dictionary
            detailed: Whether to print detailed information
        """
        concept_id = concept.get('concept_id', 'unknown')

        # Print concept ID and representations
        self._print_colorized(f"Concept: {concept_id}", Fore.GREEN)

        # Print representations
        representations = concept.get('representations', {})
        if representations:
            self._print_info("Representations:")
            for language, texts in representations.items():
                texts_str = ", ".join(texts)
                self._print_colorized(f"  {language.upper()}: {texts_str}", Fore.YELLOW)

        # Print prerequisites
        prerequisites = concept.get('prerequisites', [])
        if prerequisites:
            self._print_info(f"Prerequisites ({len(prerequisites)}):")
            for prereq_id in prerequisites:
                prereq = self.concept_repository.get_concept(prereq_id)
                if prereq:
                    # Try to get an English representation if available
                    prereq_text = "unknown"
                    if 'en' in prereq.get('representations', {}):
                        prereq_text = prereq['representations']['en'][0]
                    elif next(iter(prereq.get('representations', {}).values()), []):
                        # Fallback to first representation in any language
                        first_lang = next(iter(prereq['representations']))
                        prereq_text = prereq['representations'][first_lang][0]

                    self._print_colorized(f"  {prereq_id} ({prereq_text})", Fore.BLUE)
                else:
                    self._print_colorized(f"  {prereq_id} (not found)", Fore.RED)

        # Print related concepts
        related = concept.get('related', [])
        if related:
            self._print_info(f"Related Concepts ({len(related)}):")
            for related_id in related:
                related_concept = self.concept_repository.get_concept(related_id)
                if related_concept:
                    # Try to get an English representation if available
                    related_text = "unknown"
                    if 'en' in related_concept.get('representations', {}):
                        related_text = related_concept['representations']['en'][0]
                    elif next(iter(related_concept.get('representations', {}).values()), []):
                        # Fallback to first representation in any language
                        first_lang = next(iter(related_concept['representations']))
                        related_text = related_concept['representations'][first_lang][0]

                    self._print_colorized(f"  {related_id} ({related_text})", Fore.BLUE)
                else:
                    self._print_colorized(f"  {related_id} (not found)", Fore.RED)

        # Print metadata
        if detailed:
            metadata = concept.get('metadata', {})
            if metadata:
                self._print_info("Metadata:")
                for key, value in metadata.items():
                    self._print_colorized(f"  {key}: {value}", Fore.WHITE)

        print()  # Empty line

    def _print_candidate(self, candidate: Dict[str, Any], detailed: bool = False):
        """
        Print concept candidate information.

        Args:
            candidate: Candidate dictionary
            detailed: Whether to print detailed information
        """
        candidate_id = candidate.get('candidate_id', 'unknown')
        status = candidate.get('status', 'pending')

        # Choose color based on status
        if status == 'approved':
            status_color = Fore.GREEN
        elif status == 'rejected':
            status_color = Fore.RED
        else:  # pending
            status_color = Fore.YELLOW

        # Print candidate ID and status
        self._print_colorized(f"Candidate: {candidate_id} [{status.upper()}]", status_color)

        # Print text
        text = candidate.get('text', '')
        language = candidate.get('language', 'en')
        domain = candidate.get('domain', 'unknown')
        score = candidate.get('score', 0.0)

        self._print_colorized(f"  Text: {text}", Fore.WHITE)
        self._print_colorized(f"  Language: {language.upper()}", Fore.WHITE)
        self._print_colorized(f"  Domain: {domain}", Fore.WHITE)
        self._print_colorized(f"  Score: {score:.2f}", Fore.WHITE)

        # Print source information
        source_video_id = candidate.get('source_video_id', '')
        source_video_title = candidate.get('source_video_title', '')

        if source_video_id:
            self._print_colorized(f"  Source: {source_video_title} ({source_video_id})", Fore.BLUE)

        # Print detailed information if requested
        if detailed:
            # Print extraction method
            extraction_method = candidate.get('extraction_method', '')
            if extraction_method:
                self._print_colorized(f"  Extraction Method: {extraction_method}", Fore.WHITE)

            # Print created timestamp
            created_at = candidate.get('created_at', '')
            if created_at:
                self._print_colorized(f"  Created: {created_at}", Fore.WHITE)

            # Print source segments if available
            source_segments = candidate.get('source_segments', [])
            if source_segments:
                self._print_colorized(f"  Source Segments: {', '.join(source_segments[:3])}"
                                    f"{' and more...' if len(source_segments) > 3 else ''}", Fore.WHITE)

            # If concept_data is available and has prerequisites/related, show them
            concept_data = candidate.get('concept_data', {})

            # Show prerequisites
            prerequisites = concept_data.get('prerequisites', [])
            if prerequisites:
                self._print_info(f"  Prerequisites ({len(prerequisites)}):")
                for prereq_id in prerequisites:
                    prereq = self.concept_repository.get_concept(prereq_id)
                    if prereq:
                        # Try to get an English representation if available
                        prereq_text = "unknown"
                        if 'en' in prereq.get('representations', {}):
                            prereq_text = prereq['representations']['en'][0]
                        elif next(iter(prereq.get('representations', {}).values()), []):
                            # Fallback to first representation in any language
                            first_lang = next(iter(prereq['representations']))
                            prereq_text = prereq['representations'][first_lang][0]

                        self._print_colorized(f"    {prereq_id} ({prereq_text})", Fore.BLUE)
                    else:
                        self._print_colorized(f"    {prereq_id} (not found)", Fore.RED)

            # Show related concepts
            related = concept_data.get('related', [])
            if related:
                self._print_info(f"  Related Concepts ({len(related)}):")
                for related_id in related:
                    related_concept = self.concept_repository.get_concept(related_id)
                    if related_concept:
                        related_text = "unknown"
                        if 'en' in related_concept.get('representations', {}):
                            related_text = related_concept['representations']['en'][0]
                        elif next(iter(related_concept.get('representations', {}).values()), []):
                            # Fallback to first representation in any language
                            first_lang = next(iter(related_concept['representations']))
                            related_text = related_concept['representations'][first_lang][0]

                        self._print_colorized(f"    {related_id} ({related_text})", Fore.BLUE)
                    else:
                        self._print_colorized(f"    {related_id} (not found)", Fore.RED)

        print()  # Empty line

    def _print_learning_path(self, path_result: Dict[str, Any]):
        """
        Print learning path information.

        Args:
            path_result: Learning path result dictionary
        """
        status = path_result.get('status', 'unknown')
        if status != 'success':
            self._print_error(f"Learning path generation failed: {path_result.get('message', 'Unknown error')}")
            return

        path = path_result.get('path', [])
        target_concepts = path_result.get('target_concepts', [])

        self._print_colorized(f"Learning Path ({len(path)} concepts)", Fore.GREEN)
        self._print_info(f"Target concepts: {', '.join(target_concepts)}")

        # Print path steps
        print()
        self._print_colorized("Learning Path Steps:", Fore.CYAN)
        for i, concept_entry in enumerate(path, 1):
            concept_id = concept_entry.get('concept_id', 'unknown')
            representations = concept_entry.get('representations', {})
            is_target = concept_entry.get('is_target', False)

            # Try to get best representation for display
            concept_text = "unknown"
            if 'en' in representations:
                concept_text = representations['en'][0]
            elif next(iter(representations.values()), []):
                # Fallback to first representation in any language
                first_lang = next(iter(representations))
                concept_text = representations[first_lang][0]

            # Mark target concepts
            target_marker = "*" if is_target else " "

            # Print concept with appropriate color
            if is_target:
                self._print_colorized(f"{i:2d}. {target_marker} {concept_text} ({concept_id})", Fore.GREEN)
            else:
                self._print_colorized(f"{i:2d}. {target_marker} {concept_text} ({concept_id})", Fore.WHITE)

        print()
        self._print_info("* Indicates target concepts")

        # Print generation metadata
        metadata = path_result.get('metadata', {})
        if metadata:
            self._print_info(f"Generation time: {metadata.get('generation_time_ms', 0)} ms")

    def process_video(self, args):
        """
        Process a YouTube video or playlist.

        Args:
            args: Command-line arguments
        """
        url = args.youtube_url
        language_preference = args.language
        max_videos = args.max_videos

        if not VIDEO_PROCESSING_AVAILABLE:
            self._print_error("Video processing components are not available")
            return

        if not self.data_pipeline:
            self._print_error("Data pipeline not initialized")
            return

        # Determine if URL is a video or playlist
        is_playlist = False
        is_video = False

        # Check for playlist URL
        valid_playlist, playlist_id = self.data_pipeline.youtube_extractor.validate_playlist_url(url)
        if valid_playlist and playlist_id:
            is_playlist = True
            self._print_info(f"Detected a YouTube playlist: {url}")
        else:
            # Check for video URL
            valid_video, video_id = self.data_pipeline.youtube_extractor.validate_video_url(url)
            if valid_video and video_id:
                is_video = True
                self._print_info(f"Detected a YouTube video: {url}")
            else:
                self._print_error(f"Invalid YouTube URL: {url}")
                return

        try:
            start_time = time.time()

            if is_playlist:
                # Process playlist
                self._print_info(f"Processing playlist with up to {max_videos} videos...")
                self._process_playlist(url, language_preference, max_videos)
            elif is_video:
                # Process single video
                self._print_info(f"Processing video: {url}")
                self._process_single_video(url, language_preference)

            total_time = time.time() - start_time
            self._print_info(f"Total processing time: {total_time:.2f} seconds")

        except Exception as e:
            self._print_error(f"Error during processing: {e}")

    def _process_single_video(self, url: str, language_preference: List[str]):
        """
        Process a single YouTube video.

        Args:
            url: YouTube video URL
            language_preference: List of language codes in order of preference
        """
        try:
            start_time = time.time()
            result = self.data_pipeline.process_video(url, language_preference)
            processing_time = time.time() - start_time

            if result.get('status') == 'completed':
                video_id = result.get('video_id', 'unknown')
                title = result.get('metadata', {}).get('title', 'Unknown Title')

                self._print_success(f"Successfully processed video:")
                self._print_info(f"Video ID: {video_id}")
                self._print_info(f"Title: {title}")
                self._print_info(f"Processing time: {processing_time:.2f} seconds")

                # Print concept statistics
                total_concepts = result.get('concepts', {}).get('total', 0)
                educational_concepts = result.get('concepts', {}).get('educational', 0)
                passing_concepts = result.get('concepts', {}).get('passing', 0)

                self._print_info(f"Extracted {total_concepts} known concepts:")
                self._print_colorized(f"  Educational concepts: {educational_concepts}", Fore.GREEN)
                self._print_colorized(f"  Passing mentions: {passing_concepts}", Fore.YELLOW)

                # Print concept candidate statistics
                total_candidates = result.get('concept_candidates', {}).get('total', 0)
                candidate_ids = result.get('concept_candidates', {}).get('candidate_ids', [])

                self._print_info(f"\nExtracted {total_candidates} new concept candidates")
                if candidate_ids:
                    self._print_info("Use 'candidates' command to review them")
                    # Print a few candidate IDs as examples if available
                    if len(candidate_ids) > 0:
                        self._print_colorized(f"  Example candidate ID: {candidate_ids[0]}", Fore.YELLOW)

                # Get detailed concept information if available
                video_concepts = self.data_access.get_video_concepts(video_id)
                comprehensive_concepts = video_concepts.get('comprehensive_concepts', [])

                if comprehensive_concepts:
                    self._print_info(f"\nTop educational concepts:")
                    for i, concept_data in enumerate(comprehensive_concepts[:5], 1):
                        concept = concept_data.get('data', {})
                        representations = concept.get('representations', {})

                        # Get best representation
                        concept_text = "unknown"
                        if 'en' in representations:
                            concept_text = representations['en'][0]
                        elif next(iter(representations.values()), []):
                            first_lang = next(iter(representations))
                            concept_text = representations[first_lang][0]

                        # Print with max significance
                        occurrences = concept_data.get('occurrences', [])
                        max_significance = max([o.get('educational_significance', 0) for o in occurrences]) if occurrences else 0

                        self._print_colorized(
                            f"{i:2d}. {concept_text} "
                            f"[significance: {max_significance:.1f}]",
                            Fore.GREEN
                        )
            else:
                self._print_error(f"Processing failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            self._print_error(f"Error processing video: {e}")

    def _process_playlist(self, url: str, language_preference: List[str], max_videos: int):
        """
        Process a YouTube playlist.

        Args:
            url: YouTube playlist URL
            language_preference: List of language codes in order of preference
            max_videos: Maximum number of videos to process
        """
        try:
            self._print_info(f"Processing playlist: {url}")
            start_time = time.time()
            result = self.data_pipeline.process_youtube_playlist(url, language_preference, max_videos)
            processing_time = time.time() - start_time

            if result.get('status') == 'completed':
                playlist_id = result.get('playlist_id', 'unknown')
                title = result.get('playlist_title', 'Unknown Playlist')

                self._print_success(f"Successfully processed playlist:")
                self._print_info(f"Playlist ID: {playlist_id}")
                self._print_info(f"Title: {title}")
                self._print_info(f"Channel: {result.get('playlist_channel', 'Unknown Channel')}")

                # Print video statistics
                total_videos = result.get('video_count', 0)
                processed_videos = result.get('processed_count', 0)

                self._print_info(f"Videos: {processed_videos} processed out of {total_videos}")

                # Print concept statistics
                total_concepts = result.get('concepts', {}).get('total', 0)
                educational_concepts = result.get('concepts', {}).get('educational', 0)

                # Print candidate statistics
                total_candidates = result.get('concept_candidates', {}).get('total', 0)

                self._print_info(f"Extracted {total_concepts} known concepts:")
                self._print_colorized(f"  Educational concepts: {educational_concepts}", Fore.GREEN)
                self._print_colorized(f"  Passing mentions: {total_concepts - educational_concepts}", Fore.YELLOW)
                self._print_info(f"Extracted {total_candidates} new concept candidates")

                # Print processed video details
                video_results = result.get('videos', [])
                successful_videos = [v for v in video_results if v.get('status') == 'completed']

                if successful_videos:
                    self._print_info(f"\nProcessed videos:")
                    for i, video in enumerate(successful_videos, 1):
                        video_id = video.get('video_id', 'unknown')
                        title = video.get('metadata', {}).get('title', 'Unknown Title')
                        concepts = video.get('concepts', {}).get('total', 0)
                        candidates = video.get('concept_candidates', {}).get('total', 0)

                        self._print_colorized(
                            f"{i:2d}. {title} ({video_id}) - {concepts} concepts, {candidates} candidates",
                            Fore.CYAN
                        )

                self._print_info(f"\nProcessing time: {processing_time:.2f} seconds")
            else:
                self._print_error(f"Processing failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            self._print_error(f"Error processing playlist: {e}")

    def search(self, args):
        """
        Search for concepts and segments in video transcripts.

        Args:
            args: Command-line arguments
        """
        query = args.query_text
        language = args.language
        threshold = args.threshold
        max_results = args.max_results

        # Search in both concept repository and video transcripts
        self._print_info(f"Searching for: {query}")
        try:
            # Step 1: First search in the concept repository
            start_time = time.time()
            concept_results = self.concept_repository.find_concepts_by_text(
                query,
                language=language,
                threshold=threshold,
                max_results=max_results
            )
            concept_search_time = time.time() - start_time

            # Step 2: Then search in video transcripts if data_access is available
            transcript_results = []
            transcript_search_time = 0
            if self.data_access:
                start_time = time.time()
                search_response = self.data_access.search(
                    query,
                    language=language,
                    min_educational_significance=None,  # Include all results
                    item_types=["segment", "occurrence"],  # Only transcript-related items
                    limit=max_results
                )
                transcript_results = search_response.get("results", [])
                transcript_search_time = time.time() - start_time

            # Step 3: Display results with combined counts
            total_results = len(concept_results) + len(transcript_results)

            if total_results > 0:
                self._print_success(f"Found {total_results} results in {concept_search_time + transcript_search_time:.3f} seconds:")

                # Display concept results with related videos and timecodes
                if concept_results:
                    self._print_colorized(f"\nCONCEPT MATCHES ({len(concept_results)}):", Fore.CYAN)
                    for i, result in enumerate(concept_results, 1):
                        self._print_concept_search_result(i, result)

                # Display transcript segment results with video and timecode
                if transcript_results:
                    self._print_colorized(f"\nTRANSCRIPT MATCHES ({len(transcript_results)}):", Fore.CYAN)
                    for i, result in enumerate(transcript_results, 1):
                        self._print_transcript_search_result(i, result)
            else:
                self._print_warning(f"No results found matching: {query}")

        except Exception as e:
            self._print_error(f"Error searching: {e}")

    def _print_concept_search_result(self, index: int, result: Dict[str, Any]):
        """
        Print a concept search result with related videos and timecodes.

        Args:
            index: Result index
            result: Concept search result
        """
        concept = result.get("concept", {})
        similarity = result.get("similarity", 0.0)
        match_type = result.get("match_type", "unknown")
        concept_id = concept.get("concept_id", "unknown")

        # Get best representation for display
        representations = concept.get("representations", {})
        concept_text = "unknown"
        if representations:
            if "en" in representations and representations["en"]:
                concept_text = representations["en"][0]
            else:
                # Use first available language
                first_lang = next(iter(representations))
                if representations[first_lang]:
                    concept_text = representations[first_lang][0]

        # Print concept information with appropriate color
        if match_type == "exact":
            header_color = Fore.GREEN
        else:
            header_color = Fore.YELLOW

        self._print_colorized(
            f"{index:2d}. {concept_text} ({concept_id}) "
            f"[{match_type}, {similarity:.2f}]",
            header_color
        )

        # Get related videos and timecodes if data access is available
        if self.data_access:
            try:
                # First check if concept exists in database
                check_query = "SELECT COUNT(*) as count FROM repository_concepts WHERE concept_id = ?"
                check_result = self.data_access.execute_query(check_query, (concept_id,))

                if not check_result or check_result[0]["count"] == 0:
                    self._print_colorized(f"  Concept {concept_id} not found in database, saving now...", Fore.YELLOW)
                    # Save concept to database first
                    self.data_access.save_repository_concept(concept)

                # Get occurrences of this concept in videos - try with no filtering first
                occurrences = self.data_access.get_concept_occurrences(
                    concept_id,
                    min_educational_significance=None,  # No minimum threshold
                    limit=10  # Get more occurrences for better chances
                )

                if occurrences:
                    self._print_colorized(f"  📹 Related videos ({len(occurrences)} occurrences):", Fore.BLUE)

                    # Group by video
                    occurrences_by_video = {}
                    for occ in occurrences:
                        video_id = occ.get("video_id")
                        if video_id not in occurrences_by_video:
                            occurrences_by_video[video_id] = {
                                "title": occ.get("video_title", "Unknown Video"),
                                "occurrences": []
                            }
                        occurrences_by_video[video_id]["occurrences"].append(occ)

                    # Print each video with its occurrences
                    for video_id, video_data in occurrences_by_video.items():
                        video_title = video_data["title"]
                        video_occurrences = video_data["occurrences"]

                        self._print_colorized(f"     • {video_title} ({video_id}):", Fore.WHITE)

                        # Print each occurrence with timecode
                        for occ in video_occurrences:
                            # Format timecode (seconds to MM:SS)
                            start_time = occ.get("start_time", 0)
                            timecode = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"

                            # Truncate context text if too long
                            context = occ.get("context_text", "")
                            if len(context) > 70:
                                context = context[:67] + "..."

                            educational_significance = occ.get("educational_significance", 0)

                            # Print with color based on educational significance
                            if educational_significance >= 3.5:
                                sig_color = Fore.GREEN
                            elif educational_significance >= 2.5:
                                sig_color = Fore.YELLOW
                            else:
                                sig_color = Fore.WHITE

                            self._print_colorized(
                                f"        @ {timecode} - \"{context}\" [sig: {educational_significance:.1f}]",
                                sig_color
                            )

                else:
                    # Try to check occurrences table directly to debug
                    check_occ_query = "SELECT COUNT(*) as count FROM occurrences WHERE concept_id = ?"
                    check_occ_result = self.data_access.execute_query(check_occ_query, (concept_id,))
                    occ_count = check_occ_result[0]["count"] if check_occ_result else 0

                    if occ_count > 0:
                        self._print_colorized(f"  Found {occ_count} occurrences in database, but couldn't retrieve them with joins. Check segment and video linkage.", Fore.YELLOW)
                    else:
                        self._print_colorized("  No video occurrences found for this concept - try processing videos with this concept", Fore.RED)
            except Exception as e:
                self._print_colorized(f"  Error retrieving video occurrences: {e}", Fore.RED)

        # Print a few prerequisites if available
        prerequisites = concept.get("prerequisites", [])
        if prerequisites:
            prereq_str = ", ".join(prerequisites[:3])
            if len(prerequisites) > 3:
                prereq_str += f" and {len(prerequisites) - 3} more"
            self._print_colorized(f"  Prerequisites: {prereq_str}", Fore.BLUE)

        print()  # Empty line for readability

    def _print_transcript_search_result(self, index: int, result: Dict[str, Any]):
        """
        Print a transcript search result with video and timecode.

        Args:
            index: Result index
            result: Transcript search result
        """
        item_type = result.get("item_type", "unknown")
        text = result.get("text", "")
        video_id = result.get("video_id", "unknown")
        video_title = result.get("video_title", "Unknown Video")
        educational_significance = result.get("educational_significance", 0)

        # Format timecode depending on the item type
        timecode = "00:00"
        if item_type == "segment":
            segment = result.get("segment", {})
            if segment:
                # Convert seconds to MM:SS format
                start_time = segment.get("start_time", 0)
                timecode = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
        elif item_type == "occurrence":
            occurrence = result.get("occurrence", {})
            if occurrence:
                start_time = occurrence.get("start_time", 0)
                timecode = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"

        # Truncate text if too long
        if len(text) > 100:
            text = text[:97] + "..."

        # Determine color based on educational significance
        if educational_significance >= 3.5:
            color = Fore.GREEN
        elif educational_significance >= 2.5:
            color = Fore.YELLOW
        else:
            color = Fore.WHITE

        # Print result with video info and timecode
        self._print_colorized(f"{index:2d}. {text}", color)
        self._print_colorized(f"    📹 {video_title} ({video_id}) @ {timecode}", Fore.BLUE)

        # If it's an occurrence, show the related concept
        if item_type == "occurrence":
            concept = result.get("concept", {})
            if concept and isinstance(concept, dict):
                concept_id = concept.get("concept_id", "")
                representations = concept.get("representations", {})

                # Get best representation
                concept_text = "unknown concept"
                if representations:
                    if "en" in representations and representations["en"]:
                        concept_text = representations["en"][0]
                    else:
                        # Use first available language
                        first_lang = next(iter(representations))
                        if representations[first_lang]:
                            concept_text = representations[first_lang][0]

                self._print_colorized(f"    🔄 Related to concept: {concept_text} ({concept_id})", Fore.MAGENTA)

        print()  # Empty line for readability

    def view_concept(self, args):
        """
        View detailed information about a concept.

        Args:
            args: Command-line arguments
        """
        concept_id = args.concept_id

        # Get the concept
        concept = self.concept_repository.get_concept(concept_id)
        if not concept:
            self._print_error(f"Concept not found: {concept_id}")
            return

        # Print detailed concept information
        self._print_concept(concept, detailed=True)

    def generate_path(self, args):
        """
        Generate a learning path.

        Args:
            args: Command-line arguments
        """
        # Parse concept IDs
        concept_ids = args.concept_ids.split(',')
        max_concepts = args.max_concepts

        # Generate learning path
        self._print_info(f"Generating learning path for concepts: {', '.join(concept_ids)}")
        try:
            start_time = time.time()
            path_result = self.learning_path_generator.generate_path(
                concept_ids,
                max_concepts=max_concepts
            )
            generation_time = time.time() - start_time

            # Print path
            self._print_learning_path(path_result)

        except Exception as e:
            self._print_error(f"Error generating learning path: {e}")

    def list_concepts(self, args):
        """
        List available concepts.

        Args:
            args: Command-line arguments
        """
        language = args.language
        limit = args.limit
        offset = args.offset

        # List concepts
        try:
            concepts = self.concept_repository.list_concepts(
                language=language,
                limit=limit,
                offset=offset
            )

            if concepts:
                total_concepts = len(self.concept_repository.concepts)
                self._print_success(f"Found {len(concepts)} concepts (offset {offset}, showing up to {limit}):")
                self._print_info(f"Total concepts in repository: {total_concepts}")
                print()

                for i, concept in enumerate(concepts, offset + 1):
                    concept_id = concept.get('concept_id', 'unknown')

                    # Get best representation for display
                    representations = concept.get('representations', {})
                    concept_text = "unknown"
                    if language and language in representations:
                        concept_text = representations[language][0]
                    elif 'en' in representations:
                        concept_text = representations['en'][0]
                    elif next(iter(representations.values()), []):
                        first_lang = next(iter(representations))
                        concept_text = representations[first_lang][0]

                    # Count relationships
                    prereq_count = len(concept.get('prerequisites', []))
                    related_count = len(concept.get('related', []))

                    self._print_colorized(f"{i:4d}. {concept_text} ({concept_id})", Fore.GREEN)

                    # Print language representations
                    for lang, texts in representations.items():
                        text_str = ", ".join(texts)
                        self._print_colorized(f"       {lang.upper()}: {text_str}", Fore.YELLOW)

                    # Print relationship counts
                    self._print_colorized(
                        f"       Relationships: {prereq_count} prereqs, {related_count} related",
                        Fore.BLUE
                    )

                    print()  # Empty line
            else:
                self._print_warning("No concepts found")

        except Exception as e:
            self._print_error(f"Error listing concepts: {e}")

    def add_concept(self, args):
        """
        Add a new concept.

        Args:
            args: Command-line arguments
        """
        concept_id = args.concept_id
        en_representation = args.en
        ru_representation = args.ru
        category = args.category

        # Prepare representations
        representations = {}
        if en_representation:
            representations['en'] = [en_representation.lower()]
        if ru_representation:
            representations['ru'] = [ru_representation.lower()]

        if not representations:
            self._print_error("At least one representation (--en or --ru) must be provided")
            return

        # Add the concept
        try:
            new_id = self.concept_repository.add_concept(
                concept_id=concept_id,
                representations=representations,
                file_category=category
            )

            if new_id:
                self._print_success(f"Added new concept: {new_id}")
                # Show the new concept
                new_concept = self.concept_repository.get_concept(new_id)
                if new_concept:
                    self._print_concept(new_concept)
            else:
                self._print_error("Failed to add concept")

        except Exception as e:
            self._print_error(f"Error adding concept: {e}")

    def add_representation(self, args):
        """
        Add a representation to a concept.

        Args:
            args: Command-line arguments
        """
        concept_id = args.concept_id
        language = args.lang
        text = args.text.lower()  # Convert to lowercase

        if not language or not text:
            self._print_error("Both language (--lang) and text (--text) must be provided")
            return

        # Add the representation
        try:
            success = self.concept_repository.add_representation(
                concept_id=concept_id,
                text=text,
                language=language
            )

            if success:
                self._print_success(f"Added '{text}' as {language} representation to concept {concept_id}")
                # Show the updated concept
                updated_concept = self.concept_repository.get_concept(concept_id)
                if updated_concept:
                    self._print_concept(updated_concept)
            else:
                self._print_error(f"Failed to add representation to concept {concept_id}")

        except Exception as e:
            self._print_error(f"Error adding representation: {e}")

    def add_relationship(self, args):
        """
        Add a relationship between concepts.

        Args:
            args: Command-line arguments
        """
        concept_id = args.concept_id
        prereq_id = args.prereq
        related_id = args.related

        if prereq_id and related_id:
            self._print_error("Only one of --prereq or --related can be specified")
            return

        if not prereq_id and not related_id:
            self._print_error("Either --prereq or --related must be specified")
            return

        # Determine relationship type and target
        relationship_type = "prerequisite" if prereq_id else "related"
        target_id = prereq_id if prereq_id else related_id

        # Add the relationship
        try:
            success = self.concept_repository.add_relationship(
                source_id=concept_id,
                target_id=target_id,
                relationship_type=relationship_type
            )

            if success:
                self._print_success(
                    f"Added {relationship_type} relationship from {concept_id} to {target_id}"
                )
                # Show the updated concept
                updated_concept = self.concept_repository.get_concept(concept_id)
                if updated_concept:
                    self._print_concept(updated_concept)
            else:
                self._print_error(
                    f"Failed to add {relationship_type} relationship from {concept_id} to {target_id}"
                )

        except Exception as e:
            self._print_error(f"Error adding relationship: {e}")

    def edit_concept(self, args):
        """
        Edit a concept with comprehensive modifications.

        Args:
            args: Command-line arguments
        """
        concept_id = args.concept_id
        new_concept_id = args.new_id
        category = args.category

        # Get add representations
        add_representations = {}
        if args.add_en:
            add_representations['en'] = [rep.strip().lower() for rep in args.add_en.split(',')]
        if args.add_ru:
            add_representations['ru'] = [rep.strip().lower() for rep in args.add_ru.split(',')]

        # Get remove representations
        remove_representations = {}
        if args.remove_en:
            remove_representations['en'] = [rep.strip().lower() for rep in args.remove_en.split(',')]
        if args.remove_ru:
            remove_representations['ru'] = [rep.strip().lower() for rep in args.remove_ru.split(',')]

        # Get add/remove prerequisites
        add_prerequisites = args.add_prereqs.split(',') if args.add_prereqs else None
        remove_prerequisites = args.remove_prereqs.split(',') if args.remove_prereqs else None

        # Get add/remove related concepts
        add_related = args.add_related.split(',') if args.add_related else None
        remove_related = args.remove_related.split(',') if args.remove_related else None

        # Check if concept exists
        concept = self.concept_repository.get_concept(concept_id)
        if not concept:
            self._print_error(f"Concept not found: {concept_id}")
            return

        # Show current concept
        self._print_info("Current concept:")
        self._print_concept(concept)

        # Confirm if nothing to change
        if not any([new_concept_id, category, add_representations, remove_representations,
                   add_prerequisites, remove_prerequisites, add_related, remove_related]):
            self._print_warning("No changes specified. Use --help to see available options.")
            return

        # Show changes that will be made
        self._print_info("\nChanges to be applied:")

        if new_concept_id:
            self._print_colorized(f"- Rename concept ID to: {new_concept_id}", Fore.YELLOW)

        if category:
            self._print_colorized(f"- Move to category: {category}", Fore.YELLOW)

        if add_representations:
            self._print_colorized("- Add representations:", Fore.YELLOW)
            for lang, reps in add_representations.items():
                self._print_colorized(f"  {lang.upper()}: {', '.join(reps)}", Fore.WHITE)

        if remove_representations:
            self._print_colorized("- Remove representations:", Fore.YELLOW)
            for lang, reps in remove_representations.items():
                self._print_colorized(f"  {lang.upper()}: {', '.join(reps)}", Fore.WHITE)

        if add_prerequisites:
            prereq_str = ", ".join(add_prerequisites)
            self._print_colorized(f"- Add prerequisites: {prereq_str}", Fore.YELLOW)

        if remove_prerequisites:
            prereq_str = ", ".join(remove_prerequisites)
            self._print_colorized(f"- Remove prerequisites: {prereq_str}", Fore.YELLOW)

        if add_related:
            related_str = ", ".join(add_related)
            self._print_colorized(f"- Add related concepts: {related_str}", Fore.YELLOW)

        if remove_related:
            related_str = ", ".join(remove_related)
            self._print_colorized(f"- Remove related concepts: {related_str}", Fore.YELLOW)

        # Confirm changes
        user_input = input("\nApply these changes? (y/n): ")
        if user_input.lower() != 'y':
            self._print_info("Edit cancelled.")
            return

        # Apply changes
        try:
            # Call the repository edit method
            success = self.concept_repository.edit_concept(
                concept_id=concept_id,
                new_concept_id=new_concept_id,
                add_representations=add_representations,
                remove_representations=remove_representations,
                add_prerequisites=add_prerequisites,
                remove_prerequisites=remove_prerequisites,
                add_related=add_related,
                remove_related=remove_related,
                file_category=category
            )

            if success:
                # Get the updated concept ID (might have changed)
                updated_id = new_concept_id if new_concept_id else concept_id

                self._print_success(f"Successfully edited concept: {updated_id}")

                # Show the updated concept
                updated_concept = self.concept_repository.get_concept(updated_id)
                if updated_concept:
                    self._print_info("\nUpdated concept:")
                    self._print_concept(updated_concept)

                    # If this concept is in the database, update it there too
                    if self.data_access:
                        try:
                            # Save to database
                            self.data_access.save_repository_concept(updated_concept)
                            self._print_info("Updated concept in database as well")
                        except Exception as db_err:
                            self._print_warning(f"Note: The concept was updated in the repository but an error occurred updating it in the database: {db_err}")
            else:
                self._print_error(f"Failed to edit concept {concept_id}")

        except Exception as e:
            self._print_error(f"Error editing concept: {e}")

    def find_candidates(self, args):
        """
        Find potential new concept candidates.

        Args:
            args: Command-line arguments
        """
        threshold = args.threshold

        # Find candidates
        try:
            candidates = self.concept_repository.find_concept_candidates(threshold=threshold)

            if candidates:
                self._print_success(f"Found {len(candidates)} potential new concept candidates:")
                for i, candidate in enumerate(candidates, 1):
                    text = candidate.get('text', 'unknown')
                    score = candidate.get('score', 0.0)
                    source = candidate.get('source', 'unknown')

                    self._print_colorized(
                        f"{i:2d}. {text} [score: {score:.2f}, source: {source}]",
                        Fore.YELLOW
                    )
            else:
                self._print_warning("No concept candidates found")

        except Exception as e:
            self._print_error(f"Error finding concept candidates: {e}")

    def get_statistics(self, args):
        """
        Get statistics about the concept repository.

        Args:
            args: Command-line arguments
        """
        # Get statistics
        try:
            stats = self.concept_repository.get_concept_statistics()

            if stats:
                self._print_colorized("Concept Repository Statistics", Fore.GREEN)
                self._print_info(f"Total concepts: {stats.get('total_concepts', 0)}")

                # Print language statistics
                languages = stats.get('languages', {})
                if languages:
                    self._print_info("Languages:")
                    for language, count in languages.items():
                        self._print_colorized(f"  {language}: {count} concepts", Fore.YELLOW)

                # Print file statistics
                files = stats.get('files', {})
                if files:
                    self._print_info("Files:")
                    for file_name, count in files.items():
                        self._print_colorized(f"  {file_name}: {count} concepts", Fore.YELLOW)

                # Print relationship statistics
                relationships = stats.get('relationships', {})
                if relationships:
                    self._print_info("Relationships:")
                    self._print_colorized(
                        f"  Prerequisites: {relationships.get('prerequisites_count', 0)}",
                        Fore.YELLOW
                    )
                    self._print_colorized(
                        f"  Related: {relationships.get('related_count', 0)}",
                        Fore.YELLOW
                    )

                # Print candidate statistics
                candidate_stats = self.get_candidate_statistics()
                if candidate_stats:
                    self._print_info("Concept Candidates:")
                    self._print_colorized(
                        f"  Total candidates: {candidate_stats.get('total', 0)}",
                        Fore.YELLOW
                    )
                    self._print_colorized(
                        f"  Pending: {candidate_stats.get('pending', 0)}",
                        Fore.YELLOW
                    )
                    self._print_colorized(
                        f"  Approved: {candidate_stats.get('approved', 0)}",
                        Fore.YELLOW
                    )
                    self._print_colorized(
                        f"  Rejected: {candidate_stats.get('rejected', 0)}",
                        Fore.YELLOW
                    )
            else:
                self._print_warning("No statistics available")

        except Exception as e:
            self._print_error(f"Error getting statistics: {e}")

    def get_candidate_statistics(self) -> Dict[str, int]:
        """
        Get statistics about concept candidates.

        Returns:
            Dictionary with candidate statistics
        """
        try:
            # Get candidates
            all_candidates = list(self.candidate_extractor.candidates.values())

            # Count by status
            pending = sum(1 for c in all_candidates if c.get('status') == 'pending')
            approved = sum(1 for c in all_candidates if c.get('status') == 'approved')
            rejected = sum(1 for c in all_candidates if c.get('status') == 'rejected')

            return {
                'total': len(all_candidates),
                'pending': pending,
                'approved': approved,
                'rejected': rejected
            }
        except Exception as e:
            logger.error(f"Error getting candidate statistics: {e}")
            return {}

    def find_path(self, args):
        """
        Find a path between two concepts.

        Args:
            args: Command-line arguments
        """
        source_id = args.source_id
        target_id = args.target_id

        # Find the path
        try:
            path = self.learning_path_generator.find_concept_path(source_id, target_id)

            if path:
                self._print_success(f"Found path from {source_id} to {target_id} with {len(path)} steps:")
                for i, concept_id in enumerate(path, 1):
                    concept = self.concept_repository.get_concept(concept_id)
                    if concept:
                        # Get best representation for display
                        representations = concept.get('representations', {})
                        concept_text = "unknown"
                        if 'en' in representations:
                            concept_text = representations['en'][0]
                        elif next(iter(representations.values()), []):
                            first_lang = next(iter(representations))
                            concept_text = representations[first_lang][0]

                        self._print_colorized(f"{i:2d}. {concept_text} ({concept_id})", Fore.YELLOW)
                    else:
                        self._print_colorized(f"{i:2d}. {concept_id} (not found)", Fore.RED)
            else:
                self._print_warning(f"No path found from {source_id} to {target_id}")

        except Exception as e:
            self._print_error(f"Error finding path: {e}")

    def list_candidates(self, args):
        """
        List concept candidates.

        Args:
            args: Command-line arguments
        """
        status = args.status
        limit = args.limit
        offset = args.offset

        # List candidates
        try:
            candidates = self.candidate_extractor.list_candidates(
                status=status,
                limit=limit,
                offset=offset
            )

            if candidates:
                status_str = f" with status '{status}'" if status else ""
                self._print_success(f"Found {len(candidates)} concept candidates{status_str}:")
                self._print_info(f"Showing results {offset+1}-{offset+len(candidates)}")
                print()

                for i, candidate in enumerate(candidates, offset + 1):
                    candidate_id = candidate.get('candidate_id', 'unknown')
                    text = candidate.get('text', '')
                    status = candidate.get('status', 'pending')
                    score = candidate.get('score', 0.0)
                    language = candidate.get('language', '')
                    domain = candidate.get('domain', '')

                    # Choose color based on status
                    if status == 'approved':
                        status_color = Fore.GREEN
                    elif status == 'rejected':
                        status_color = Fore.RED
                    else:  # pending
                        status_color = Fore.YELLOW

                    self._print_colorized(f"{i:4d}. {text} ({candidate_id})", status_color)
                    self._print_colorized(f"       Status: {status.upper()}", status_color)
                    self._print_colorized(f"       Score: {score:.2f} | Language: {language.upper()} | Domain: {domain}", Fore.WHITE)

                    # Print source video if available
                    source_video_id = candidate.get('source_video_id', '')
                    source_video_title = candidate.get('source_video_title', '')
                    if source_video_id:
                        self._print_colorized(f"       Source: {source_video_title} ({source_video_id})", Fore.BLUE)

                    print()  # Empty line
            else:
                filter_str = f" with status '{status}'" if status else ""
                self._print_warning(f"No concept candidates found{filter_str}")

        except Exception as e:
            self._print_error(f"Error listing candidates: {e}")

    def view_candidate(self, args):
        """
        View detailed information about a concept candidate.

        Args:
            args: Command-line arguments
        """
        candidate_id = args.candidate_id

        # Get the candidate
        candidate = self.candidate_extractor.get_candidate(candidate_id)
        if not candidate:
            self._print_error(f"Candidate not found: {candidate_id}")
            return

        # Print detailed candidate information
        self._print_candidate(candidate, detailed=True)

    def approve_candidate(self, args):
        """
        Approve a concept candidate and add it to the repository.

        Args:
            args: Command-line arguments
        """
        candidate_id = args.candidate_id
        concept_id = args.concept_id

        # Get the candidate
        candidate = self.candidate_extractor.get_candidate(candidate_id)
        if not candidate:
            self._print_error(f"Candidate not found: {candidate_id}")
            return

        # Check if already approved
        if candidate.get('status') == 'approved':
            self._print_warning(f"Candidate {candidate_id} is already approved")

            # Check if it's been added to repository
            concept_data = candidate.get('concept_data', {})
            existing_concept_id = concept_data.get('concept_id', '')

            if existing_concept_id:
                self._print_info(f"Concept ID: {existing_concept_id}")

                # Check if concept exists in repository
                concept = self.concept_repository.get_concept(existing_concept_id)
                if concept:
                    self._print_info("Concept exists in repository:")
                    self._print_concept(concept)
                else:
                    self._print_warning(f"Concept {existing_concept_id} not found in repository")

                    # Offer to add it
                    user_input = input("Would you like to add it to the repository now? (y/n): ")
                    if user_input.lower() == 'y':
                        add_result = self.candidate_extractor.add_candidate_to_repository(candidate_id)
                        if add_result:
                            self._print_success(f"Added concept {add_result} to repository")
                        else:
                            self._print_error("Failed to add concept to repository")
            return

        # Update status to approved
        success = self.candidate_extractor.update_candidate_status(
            candidate_id,
            'approved',
            concept_id
        )

        if not success:
            self._print_error(f"Failed to update candidate {candidate_id} status")
            return

        self._print_success(f"Approved candidate: {candidate_id}")

        # Add to repository
        add_result = self.candidate_extractor.add_candidate_to_repository(candidate_id)
        if add_result:
            self._print_success(f"Added concept {add_result} to repository")

            # Show the new concept
            new_concept = self.concept_repository.get_concept(add_result)
            if new_concept:
                self._print_concept(new_concept)
        else:
            self._print_error("Failed to add concept to repository")

    def reject_candidate(self, args):
        """
        Reject a concept candidate.

        Args:
            args: Command-line arguments
        """
        candidate_id = args.candidate_id
        reason = args.reason

        # Get the candidate
        candidate = self.candidate_extractor.get_candidate(candidate_id)
        if not candidate:
            self._print_error(f"Candidate not found: {candidate_id}")
            return

        # Check if already rejected
        if candidate.get('status') == 'rejected':
            self._print_warning(f"Candidate {candidate_id} is already rejected")
            return

        # Update status to rejected
        success = self.candidate_extractor.update_candidate_status(
            candidate_id,
            'rejected'
        )

        if not success:
            self._print_error(f"Failed to update candidate {candidate_id} status")
            return

        # Add rejection reason if provided
        if reason:
            # Add reason to candidate metadata
            candidate = self.candidate_extractor.get_candidate(candidate_id)
            if candidate:
                if 'concept_data' not in candidate:
                    candidate['concept_data'] = {}

                if 'metadata' not in candidate['concept_data']:
                    candidate['concept_data']['metadata'] = {}

                candidate['concept_data']['metadata']['rejection_reason'] = reason

                # Save updated candidate
                self.candidate_extractor._save_candidate(candidate)

        self._print_success(f"Rejected candidate: {candidate_id}")
        if reason:
            self._print_info(f"Rejection reason: {reason}")

    def edit_candidate(self, args):
        """
        Edit a concept candidate.

        Args:
            args: Command-line arguments
        """
        candidate_id = args.candidate_id
        new_text = args.text
        new_domain = args.domain
        prereq_ids = args.prereqs.split(',') if args.prereqs else None
        related_ids = args.related.split(',') if args.related else None

        # Get the candidate
        candidate = self.candidate_extractor.get_candidate(candidate_id)
        if not candidate:
            self._print_error(f"Candidate not found: {candidate_id}")
            return

        # Edit the candidate
        success = self.candidate_extractor.edit_candidate(
            candidate_id,
            new_text=new_text,
            new_domain=new_domain,
            prerequisites=prereq_ids,
            related=related_ids
        )

        if success:
            self._print_success(f"Updated candidate: {candidate_id}")

            # Show the updated candidate
            updated_candidate = self.candidate_extractor.get_candidate(candidate_id)
            if updated_candidate:
                self._print_candidate(updated_candidate, detailed=True)
        else:
            self._print_error(f"Failed to update candidate {candidate_id}")

    def delete_candidate(self, args):
        """
        Delete a concept candidate.

        Args:
            args: Command-line arguments
        """
        candidate_id = args.candidate_id

        # Confirm deletion
        user_input = input(f"Are you sure you want to delete candidate {candidate_id}? (y/n): ")
        if user_input.lower() != 'y':
            self._print_info("Deletion cancelled")
            return

        # Delete the candidate
        success = self.candidate_extractor.delete_candidate(candidate_id)

        if success:
            self._print_success(f"Deleted candidate: {candidate_id}")
        else:
            self._print_error(f"Failed to delete candidate {candidate_id}")


def main():
    """Main entry point for the CLI application."""
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Video Lecture Content Indexer - CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py process https://www.youtube.com/watch?v=dQw4w9WgXcQ
  python demo.py search "quantum mechanics"
  python demo.py concept quantum_mechanics
  python demo.py path quantum_mechanics,wave_function
  python demo.py list-concepts --language en
  python demo.py candidates
  python demo.py candidate <candidate_id>
  python demo.py approve-candidate <candidate_id>
  python demo.py add-concept new_concept --en "New Concept" --ru "Новая концепция"
  python demo.py add-representation quantum_mechanics --lang ru --text "квантовая механика"
  python demo.py add-relationship quantum_mechanics --prereq wave_function
  python demo.py edit-concept quantum_mechanics --add-en "quantum physics,quantum theory"
  python demo.py stats
"""
    )

    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process video command
    process_parser = subparsers.add_parser(
        "process", help="Process a YouTube video or playlist"
    )
    process_parser.add_argument("youtube_url", help="YouTube video or playlist URL")
    process_parser.add_argument(
        "--max-videos", type=int, default=10,
        help="Maximum number of videos to process from a playlist"
    )
    process_parser.add_argument(
        "--language", nargs="+", default=["en", "ru"],
        help="Language preference order (e.g., 'en ru')"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search for concepts and segments"
    )
    search_parser.add_argument("query_text", help="Text to search for")
    search_parser.add_argument("--language", help="Language filter (e.g., 'en', 'ru')")
    search_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Minimum similarity threshold (0.0-1.0)"
    )
    search_parser.add_argument(
        "--max-results", type=int, default=10, help="Maximum number of results to return"
    )

    # View concept command
    concept_parser = subparsers.add_parser(
        "concept", help="View detailed information about a concept"
    )
    concept_parser.add_argument("concept_id", help="Concept ID")

    # Generate learning path command
    path_parser = subparsers.add_parser(
        "path", help="Generate a learning path"
    )
    path_parser.add_argument(
        "concept_ids", help="Comma-separated list of target concept IDs"
    )
    path_parser.add_argument(
        "--max-concepts", type=int, default=15, help="Maximum number of concepts in the path"
    )

    # List concepts command
    list_parser = subparsers.add_parser(
        "list-concepts", help="List available concepts"
    )
    list_parser.add_argument("--language", help="Language filter (e.g., 'en', 'ru')")
    list_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of concepts to list"
    )
    list_parser.add_argument(
        "--offset", type=int, default=0, help="Pagination offset"
    )

    # Add concept command
    add_concept_parser = subparsers.add_parser(
        "add-concept", help="Add a new concept"
    )
    add_concept_parser.add_argument(
        "concept_id", nargs="?", default=None, help="Concept ID (optional, will be generated if not provided)"
    )
    add_concept_parser.add_argument(
        "--en", help="English representation"
    )
    add_concept_parser.add_argument(
        "--ru", help="Russian representation"
    )
    add_concept_parser.add_argument(
        "--category", default="interdisciplinary",
        help="Category file to save to (e.g., 'mathematics', 'physics', 'computer_science', 'interdisciplinary')"
    )

    # Add representation command
    add_representation_parser = subparsers.add_parser(
        "add-representation", help="Add a representation to a concept"
    )
    add_representation_parser.add_argument("concept_id", help="Concept ID")
    add_representation_parser.add_argument("--lang", help="Language code (e.g., 'en', 'ru')")
    add_representation_parser.add_argument("--text", help="Representation text")

    # Add relationship command
    add_relationship_parser = subparsers.add_parser(
        "add-relationship", help="Add a relationship between concepts"
    )
    add_relationship_parser.add_argument("concept_id", help="Source concept ID")
    add_relationship_parser.add_argument("--prereq", help="Prerequisite concept ID")
    add_relationship_parser.add_argument("--related", help="Related concept ID")

    # Edit concept command
    edit_concept_parser = subparsers.add_parser(
        "edit-concept", help="Edit a concept with comprehensive modifications"
    )
    edit_concept_parser.add_argument("concept_id", help="Concept ID to edit")
    edit_concept_parser.add_argument("--new-id", help="New concept ID (to rename)")
    edit_concept_parser.add_argument(
        "--category", help="Move concept to new category file"
    )
    edit_concept_parser.add_argument(
        "--add-en", help="Add English representations (comma-separated)"
    )
    edit_concept_parser.add_argument(
        "--add-ru", help="Add Russian representations (comma-separated)"
    )
    edit_concept_parser.add_argument(
        "--remove-en", help="Remove English representations (comma-separated)"
    )
    edit_concept_parser.add_argument(
        "--remove-ru", help="Remove Russian representations (comma-separated)"
    )
    edit_concept_parser.add_argument(
        "--add-prereqs", help="Add prerequisite concept IDs (comma-separated)"
    )
    edit_concept_parser.add_argument(
        "--remove-prereqs", help="Remove prerequisite concept IDs (comma-separated)"
    )
    edit_concept_parser.add_argument(
        "--add-related", help="Add related concept IDs (comma-separated)"
    )
    edit_concept_parser.add_argument(
        "--remove-related", help="Remove related concept IDs (comma-separated)"
    )

    # Find candidates command
    candidates_parser = subparsers.add_parser(
        "candidates", help="Find potential new concept candidates"
    )
    candidates_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Similarity threshold (0.0-1.0)"
    )

    # Statistics command
    stats_parser = subparsers.add_parser(
        "stats", help="Get statistics about the concept repository"
    )

    # Find path command
    find_path_parser = subparsers.add_parser(
        "find-path", help="Find a path between two concepts"
    )
    find_path_parser.add_argument("source_id", help="Source concept ID")
    find_path_parser.add_argument("target_id", help="Target concept ID")

    # List candidates command
    list_candidates_parser = subparsers.add_parser(
        "list-candidates", help="List concept candidates"
    )
    list_candidates_parser.add_argument(
        "--status", choices=["pending", "approved", "rejected"],
        help="Filter by status (pending, approved, rejected)"
    )
    list_candidates_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of candidates to list"
    )
    list_candidates_parser.add_argument(
        "--offset", type=int, default=0, help="Pagination offset"
    )

    # View candidate command
    view_candidate_parser = subparsers.add_parser(
        "candidate", help="View detailed information about a concept candidate"
    )
    view_candidate_parser.add_argument("candidate_id", help="Candidate ID")

    # Approve candidate command
    approve_candidate_parser = subparsers.add_parser(
        "approve-candidate", help="Approve a concept candidate and add it to the repository"
    )
    approve_candidate_parser.add_argument("candidate_id", help="Candidate ID")
    approve_candidate_parser.add_argument(
        "--concept-id", help="Optional concept ID to assign (generated if not provided)"
    )

    # Reject candidate command
    reject_candidate_parser = subparsers.add_parser(
        "reject-candidate", help="Reject a concept candidate"
    )
    reject_candidate_parser.add_argument("candidate_id", help="Candidate ID")
    reject_candidate_parser.add_argument(
        "--reason", help="Optional reason for rejection"
    )

    # Edit candidate command
    edit_candidate_parser = subparsers.add_parser(
        "edit-candidate", help="Edit a concept candidate"
    )
    edit_candidate_parser.add_argument("candidate_id", help="Candidate ID")
    edit_candidate_parser.add_argument(
        "--text", help="New text for the candidate"
    )
    edit_candidate_parser.add_argument(
        "--domain", help="New domain for the candidate"
    )
    edit_candidate_parser.add_argument(
        "--prereqs", help="Comma-separated list of prerequisite concept IDs"
    )
    edit_candidate_parser.add_argument(
        "--related", help="Comma-separated list of related concept IDs"
    )

    # Delete candidate command
    delete_candidate_parser = subparsers.add_parser(
        "delete-candidate", help="Delete a concept candidate"
    )
    delete_candidate_parser.add_argument("candidate_id", help="Candidate ID")

    # Parse arguments
    args = parser.parse_args()

    # Create CLI instance
    cli = ConceptIndexerCLI()

    # Execute command
    if args.command == "process":
        cli.process_video(args)
    elif args.command == "search":
        cli.search(args)
    elif args.command == "concept":
        cli.view_concept(args)
    elif args.command == "path":
        cli.generate_path(args)
    elif args.command == "list-concepts":
        cli.list_concepts(args)
    elif args.command == "add-concept":
        cli.add_concept(args)
    elif args.command == "add-representation":
        cli.add_representation(args)
    elif args.command == "add-relationship":
        cli.add_relationship(args)
    elif args.command == "edit-concept":
        cli.edit_concept(args)
    elif args.command == "candidates":
        cli.find_candidates(args)
    elif args.command == "stats":
        cli.get_statistics(args)
    elif args.command == "find-path":
        cli.find_path(args)
    elif args.command == "list-candidates":
        cli.list_candidates(args)
    elif args.command == "candidate":
        cli.view_candidate(args)
    elif args.command == "approve-candidate":
        cli.approve_candidate(args)
    elif args.command == "reject-candidate":
        cli.reject_candidate(args)
    elif args.command == "edit-candidate":
        cli.edit_candidate(args)
    elif args.command == "delete-candidate":
        cli.delete_candidate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
