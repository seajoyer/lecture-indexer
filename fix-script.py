#!/usr/bin/env python3

"""
Fix script for measure_memory decorator error in TranscriptProcessor.
"""

import os
import re
import sys

def fix_transcript_processor():
    # Path to transcript_processor.py
    transcript_processor_path = "data_acquisition/transcript_processor/python/transcript_processor.py"

    # Check if file exists
    if not os.path.exists(transcript_processor_path):
        print(f"Error: File not found at {transcript_processor_path}")
        print("Make sure you're running this script from the project root directory.")
        return False

    # Read the file content
    with open(transcript_processor_path, 'r') as f:
        content = f.read()

    # Check if the problematic decorator is present
    if "@measure_memory(threshold_mb=200)" not in content:
        print("The problematic decorator wasn't found. The file might have been fixed already.")
        return False

    # Replace @measure_memory with @memory_function
    # First make sure the import is correct
    modified_content = re.sub(
        r'from common\.utils\.performance_utils import measure_time, time_function, measure_memory',
        r'from common.utils.performance_utils import measure_time, time_function, memory_function',
        content
    )

    # Then replace the decorator
    modified_content = re.sub(
        r'@measure_memory\(threshold_mb=200\)',
        r'@memory_function(threshold_mb=200)',
        modified_content
    )

    # Write the changes back to the file
    with open(transcript_processor_path, 'w') as f:
        f.write(modified_content)

    print(f"Successfully fixed {transcript_processor_path}")
    print("Changed @measure_memory(threshold_mb=200) to @memory_function(threshold_mb=200)")
    print("Updated imports to use memory_function instead of measure_memory")
    return True

if __name__ == "__main__":
    print("Running fix for measure_memory decorator error...")
    if fix_transcript_processor():
        print("Fix applied successfully. You can now run demo.py without errors.")
    else:
        print("Fix was not applied. Please check the error messages above.")
