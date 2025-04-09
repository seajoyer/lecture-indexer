#!/usr/bin/env python3
"""
Diagnostic script for the Lecture Video Content Indexer.
Checks and repairs database integrity, specifically focusing on search index issues.
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("diagnostic")

try:
    from data_access import get_data_access
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    print(f"Error: Could not import required modules: {e}")
    print("Make sure you're running this script from the correct directory.")
    sys.exit(1)

def check_database_integrity():
    """Check database integrity and attempt to fix any issues."""
    print("Running diagnostic checks on the database...")

    # Get data access
    data_access = get_data_access()

    # Check if tables exist
    print("\nChecking database tables...")
    tables = ["videos", "segments", "concepts", "occurrences", "search_index"]
    table_counts = {}

    for table in tables:
        try:
            if table == "search_index":
                # Special handling for virtual table
                count_query = f"SELECT COUNT(*) as count FROM {table}"
            else:
                count_query = f"SELECT COUNT(*) as count FROM {table}"

            result = data_access.execute_query(count_query)
            if result and len(result) > 0:
                count = result[0]["count"]
                table_counts[table] = count
                print(f"  ✓ Table '{table}' exists with {count} rows")
            else:
                print(f"  ✗ Couldn't get count for table '{table}'")
        except Exception as e:
            print(f"  ✗ Error checking table '{table}': {e}")

    # Check search index status
    print("\nChecking search index...")
    if "search_index" in table_counts and table_counts["search_index"] == 0:
        print("  ✗ Search index is empty")
        if table_counts.get("concepts", 0) > 0:
            print("  → Attempting to rebuild search index from existing data...")
            rebuild_search_index(data_access)
    elif "search_index" in table_counts:
        print(f"  ✓ Search index has {table_counts['search_index']} entries")
    else:
        print("  ✗ Search index table not found")
        print("  → Attempting to recreate search index...")
        recreate_search_index(data_access)

    # Check if concepts exist without search index entries
    print("\nChecking for unindexed concepts...")
    if table_counts.get("concepts", 0) > 0:
        # Sample some concepts to see if they appear in search index
        sample_query = "SELECT concept_id, text FROM concepts LIMIT 5"
        concepts = data_access.execute_query(sample_query)

        if concepts:
            for concept in concepts:
                concept_id = concept["concept_id"]
                check_query = "SELECT COUNT(*) as count FROM search_index WHERE id = ?"
                result = data_access.execute_query(check_query, (concept_id,))
                count = result[0]["count"] if result else 0

                if count == 0:
                    print(f"  ✗ Concept '{concept['text']}' ({concept_id}) not found in search index")
                else:
                    print(f"  ✓ Concept '{concept['text']}' found in search index")

    # Check database integrity
    print("\nChecking database integrity...")
    integrity_query = "PRAGMA integrity_check"
    integrity_result = data_access.execute_query(integrity_query)

    if integrity_result and integrity_result[0].get("integrity_check") == "ok":
        print("  ✓ Database integrity check passed")
    else:
        print("  ✗ Database integrity check failed")
        print("  → Consider backing up and rebuilding the database")

    # Check for concepts without video_id in search index
    print("\nChecking for search index entries with missing video_id...")
    null_video_query = "SELECT COUNT(*) as count FROM search_index WHERE video_id IS NULL"
    null_video_result = data_access.execute_query(null_video_query)
    null_video_count = null_video_result[0]["count"] if null_video_result else 0

    if null_video_count > 0:
        print(f"  ✗ Found {null_video_count} search index entries with NULL video_id")
        print("  → Attempting to fix these entries...")
        fix_null_video_ids(data_access)
    else:
        print("  ✓ All search index entries have a video_id")

    # Perform a test search
    print("\nPerforming test searches...")
    test_searches = ["test", "function", "функция"]

    for query in test_searches:
        search_query = f"SELECT COUNT(*) as count FROM search_index WHERE search_index MATCH '\"{query}\"'"
        search_result = data_access.execute_query(search_query)
        search_count = search_result[0]["count"] if search_result else 0

        fallback_query = f"SELECT COUNT(*) as count FROM search_index WHERE lower(text) LIKE '%{query}%'"
        fallback_result = data_access.execute_query(fallback_query)
        fallback_count = fallback_result[0]["count"] if fallback_result else 0

        print(f"  Search for '{query}':")
        print(f"    - FTS match: {search_count} results")
        print(f"    - LIKE match: {fallback_count} results")

    print("\nDiagnostic checks complete!")

def rebuild_search_index(data_access):
    """Rebuild the search index from existing data."""
    try:
        # Clear existing search index
        data_access.execute_update("DELETE FROM search_index")

        # Get all concepts
        concepts_query = """
        SELECT c.concept_id, c.text, c.domain, c.language, c.educational_weight, o.video_id
        FROM concepts c
        JOIN occurrences o ON c.concept_id = o.concept_id
        GROUP BY c.concept_id, o.video_id
        """
        concepts = data_access.execute_query(concepts_query)

        # Insert concepts into search index
        for concept in concepts:
            index_query = """
            INSERT INTO search_index (id, text, domain, item_type, video_id, language, educational_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            data_access.execute_update(
                index_query,
                (
                    concept["concept_id"],
                    concept["text"],
                    concept["domain"],
                    "concept",
                    concept["video_id"],
                    concept["language"],
                    concept["educational_weight"]
                )
            )

        # Also index segments
        segments_query = """
        SELECT segment_id, text, video_id,
               (SELECT domain FROM videos WHERE videos.video_id = segments.video_id) as domain,
               language, educational_value
        FROM segments
        """
        segments = data_access.execute_query(segments_query)

        # Insert segments into search index
        for segment in segments:
            index_query = """
            INSERT INTO search_index (id, text, domain, item_type, video_id, language, educational_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            data_access.execute_update(
                index_query,
                (
                    segment["segment_id"],
                    segment["text"],
                    segment.get("domain", "unknown"),
                    "segment",
                    segment["video_id"],
                    segment.get("language", "en"),
                    segment.get("educational_value", 0)
                )
            )

        print(f"  ✓ Rebuilt search index with {len(concepts)} concepts and {len(segments)} segments")

        # Verify the index was populated
        count_query = "SELECT COUNT(*) as count FROM search_index"
        count_result = data_access.execute_query(count_query)
        count = count_result[0]["count"] if count_result else 0

        if count > 0:
            print(f"  ✓ Search index now has {count} entries")
        else:
            print(f"  ✗ Search index is still empty after rebuild")

    except Exception as e:
        print(f"  ✗ Error rebuilding search index: {e}")

def recreate_search_index(data_access):
    """Recreate the search index table and rebuild it."""
    try:
        # Drop and recreate the search index table
        data_access.execute_update("DROP TABLE IF EXISTS search_index")

        # Create with default tokenizer (better for multilingual)
        data_access.execute_update("""
        CREATE VIRTUAL TABLE search_index USING fts5(
            id,
            text,
            domain,
            item_type,
            video_id,
            language,
            educational_weight
        )
        """)

        print("  ✓ Successfully recreated search index table")

        # Rebuild the index
        rebuild_search_index(data_access)

    except Exception as e:
        print(f"  ✗ Error recreating search index: {e}")

def fix_null_video_ids(data_access):
    """Fix search index entries with NULL video_ids."""
    try:
        # Get entries with NULL video_id
        null_entries_query = """
        SELECT id, item_type FROM search_index WHERE video_id IS NULL
        """
        null_entries = data_access.execute_query(null_entries_query)

        fixed_count = 0

        for entry in null_entries:
            item_type = entry["item_type"]
            item_id = entry["id"]

            if item_type == "concept":
                # Find occurrences for this concept to get video_id
                occurrence_query = """
                SELECT video_id FROM occurrences WHERE concept_id = ? LIMIT 1
                """
                occurrence = data_access.execute_query(occurrence_query, (item_id,))

                if occurrence and occurrence[0].get("video_id"):
                    video_id = occurrence[0]["video_id"]

                    # Update the search index entry
                    update_query = """
                    UPDATE search_index SET video_id = ? WHERE id = ?
                    """
                    data_access.execute_update(update_query, (video_id, item_id))
                    fixed_count += 1

            elif item_type == "segment":
                # Get video_id for this segment
                segment_query = """
                SELECT video_id FROM segments WHERE segment_id = ?
                """
                segment = data_access.execute_query(segment_query, (item_id,))

                if segment and segment[0].get("video_id"):
                    video_id = segment[0]["video_id"]

                    # Update the search index entry
                    update_query = """
                    UPDATE search_index SET video_id = ? WHERE id = ?
                    """
                    data_access.execute_update(update_query, (video_id, item_id))
                    fixed_count += 1

        print(f"  ✓ Fixed {fixed_count}/{len(null_entries)} entries with NULL video_id")

    except Exception as e:
        print(f"  ✗ Error fixing NULL video_ids: {e}")

if __name__ == "__main__":
    check_database_integrity()
