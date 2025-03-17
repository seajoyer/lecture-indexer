#!/usr/bin/env python
"""
Test script to verify that search results include proper timestamps.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-timestamps")

def main():
    """Main function for testing the search timestamp fix."""
    print("Testing search engine timestamp fix...")

    # Import search engine components
    try:
        from search_retrieval.search_engine.python.search_engine import SearchEngine
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        sys.exit(1)

    # Initialize search engine
    search_config = {"index_dir": "data/index"}
    search_engine = SearchEngine(search_config)

    # Create structured query that should test segment search
    structured_query = {
        "original_text": "example",  # A generic term likely to be in many videos
        "filters": {},
        "theory_practice_ratio": None,
        "pagination": {
            "offset": 0,
            "limit": 5
        }
    }

    # Execute search
    try:
        results = search_engine.search(structured_query)

        # Print search stats
        print(f"Total results: {results.get('totalResults', 0)}")
        print(f"Theoretical results: {results.get('theoreticalResults', 0)}")
        print(f"Practical results: {results.get('practicalResults', 0)}")

        # Examine the search results for timestamps
        search_results = results.get('results', [])

        if not search_results:
            print("No search results found. Please ensure you have indexed some videos first.")
            sys.exit(0)

        print("\nExamining search results for timestamps:")
        for i, result in enumerate(search_results, 1):
            video_id = result.get('video_id', 'unknown')
            start_time = result.get('start_time', 'MISSING')
            end_time = result.get('end_time', 'MISSING')

            print(f"Result {i}:")
            print(f"  Video ID: {video_id}")
            print(f"  Start time: {start_time}")
            print(f"  End time: {end_time}")

            # Check if timestamps look valid
            if start_time == 0 and end_time == 0:
                print("  WARNING: Both start_time and end_time are 0, which might indicate missing timestamps")
            elif start_time == 'MISSING' or end_time == 'MISSING':
                print("  ERROR: Timestamp fields are missing from the result")
            else:
                print("  SUCCESS: Timestamps appear to be present")

        print("\nTest completed. If you see 'SUCCESS' messages above, the fix is working correctly.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
