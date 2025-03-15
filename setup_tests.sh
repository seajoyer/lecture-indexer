#!/usr/bin/env sh

# setup_tests.sh
# Script to create the test directory structure for Lecture Video Content Indexer
# Usage: ./setup_tests.sh [base_directory]
# If base_directory is not provided, assumes current directory

set -e  # Exit on error

# Determine base directory
BASE_DIR="${1:-.}"
echo "Setting up test structure in: $BASE_DIR"

# Create directory structure
create_dirs() {
    echo "Creating directory structure..."

    # Common utils directory
    mkdir -p "$BASE_DIR/common/utils"

    # Test directory structure
    mkdir -p "$BASE_DIR/tests/unit/data_acquisition"
    mkdir -p "$BASE_DIR/tests/unit/concept_analysis"
    mkdir -p "$BASE_DIR/tests/unit/search_retrieval"
    mkdir -p "$BASE_DIR/tests/integration"

    echo "Directory structure created."
}

# Create common utility files placeholders
create_utils() {
    echo "Creating common utility files..."

    # config_loader.py
    CONFIG_LOADER="$BASE_DIR/common/utils/config_loader.py"
    if [ ! -f "$CONFIG_LOADER" ]; then
        echo "Creating $CONFIG_LOADER"
        touch "$CONFIG_LOADER"
    else
        echo "$CONFIG_LOADER already exists, skipping"
    fi

    # error_handling.py
    ERROR_HANDLING="$BASE_DIR/common/utils/error_handling.py"
    if [ ! -f "$ERROR_HANDLING" ]; then
        echo "Creating $ERROR_HANDLING"
        touch "$ERROR_HANDLING"
    else
        echo "$ERROR_HANDLING already exists, skipping"
    fi

    # Create __init__.py files
    touch "$BASE_DIR/common/__init__.py"
    touch "$BASE_DIR/common/utils/__init__.py"

    echo "Common utility files created."
}

# Create unit test files
create_unit_tests() {
    echo "Creating unit test files..."

    # YouTube Data Extractor Test
    YOUTUBE_TEST="$BASE_DIR/tests/unit/data_acquisition/test_youtube_data_extractor.py"
    if [ ! -f "$YOUTUBE_TEST" ]; then
        echo "Creating $YOUTUBE_TEST"
        touch "$YOUTUBE_TEST"
    else
        echo "$YOUTUBE_TEST already exists, skipping"
    fi

    # Transcript Processor Test
    TRANSCRIPT_TEST="$BASE_DIR/tests/unit/data_acquisition/test_transcript_processor.py"
    if [ ! -f "$TRANSCRIPT_TEST" ]; then
        echo "Creating $TRANSCRIPT_TEST"
        touch "$TRANSCRIPT_TEST"
    else
        echo "$TRANSCRIPT_TEST already exists, skipping"
    fi

    # Domain Classifier Test
    DOMAIN_TEST="$BASE_DIR/tests/unit/concept_analysis/test_domain_classifier.py"
    if [ ! -f "$DOMAIN_TEST" ]; then
        echo "Creating $DOMAIN_TEST"
        touch "$DOMAIN_TEST"
    else
        echo "$DOMAIN_TEST already exists, skipping"
    fi

    # Theory Practice Classifier Test
    THEORY_PRACTICE_TEST="$BASE_DIR/tests/unit/concept_analysis/test_theory_practice_classifier.py"
    if [ ! -f "$THEORY_PRACTICE_TEST" ]; then
        echo "Creating $THEORY_PRACTICE_TEST"
        touch "$THEORY_PRACTICE_TEST"
    else
        echo "$THEORY_PRACTICE_TEST already exists, skipping"
    fi

    # Data Pipeline Test
    PIPELINE_TEST="$BASE_DIR/tests/unit/data_acquisition/test_data_pipeline.py"
    if [ ! -f "$PIPELINE_TEST" ]; then
        echo "Creating $PIPELINE_TEST"
        touch "$PIPELINE_TEST"
    else
        echo "$PIPELINE_TEST already exists, skipping"
    fi

    # Search Engine Test
    SEARCH_TEST="$BASE_DIR/tests/unit/search_retrieval/test_search_engine.py"
    if [ ! -f "$SEARCH_TEST" ]; then
        echo "Creating $SEARCH_TEST"
        touch "$SEARCH_TEST"
    else
        echo "$SEARCH_TEST already exists, skipping"
    fi

    echo "Unit test files created."
}

# Create integration test files
create_integration_tests() {
    echo "Creating integration test files..."

    # API and Task Manager Integration Test
    API_TEST="$BASE_DIR/tests/integration/test_api_task_manager.py"
    if [ ! -f "$API_TEST" ]; then
        echo "Creating $API_TEST"
        touch "$API_TEST"
    else
        echo "$API_TEST already exists, skipping"
    fi

    # Full Workflow Integration Test
    WORKFLOW_TEST="$BASE_DIR/tests/integration/test_full_workflow.py"
    if [ ! -f "$WORKFLOW_TEST" ]; then
        echo "Creating $WORKFLOW_TEST"
        touch "$WORKFLOW_TEST"
    else
        echo "$WORKFLOW_TEST already exists, skipping"
    fi

    echo "Integration test files created."
}

# Create test configuration files
create_test_config() {
    echo "Creating test configuration files..."

    # conftest.py
    CONFTEST="$BASE_DIR/tests/conftest.py"
    if [ ! -f "$CONFTEST" ]; then
        echo "Creating $CONFTEST"
        touch "$CONFTEST"
    else
        echo "$CONFTEST already exists, skipping"
    fi

    # README.md
    README="$BASE_DIR/tests/README.md"
    if [ ! -f "$README" ]; then
        echo "Creating $README"
        touch "$README"
    else
        echo "$README already exists, skipping"
    fi

    echo "Test configuration files created."
}

# Initialize empty __init__.py files for proper imports
create_init_files() {
    echo "Creating __init__.py files..."

    # Test directories
    touch "$BASE_DIR/tests/__init__.py"
    touch "$BASE_DIR/tests/unit/__init__.py"
    touch "$BASE_DIR/tests/unit/data_acquisition/__init__.py"
    touch "$BASE_DIR/tests/unit/concept_analysis/__init__.py"
    touch "$BASE_DIR/tests/unit/search_retrieval/__init__.py"
    touch "$BASE_DIR/tests/integration/__init__.py"

    echo "__init__.py files created."
}

# Main execution
main() {
    create_dirs
    create_utils
    create_unit_tests
    create_integration_tests
    create_test_config
    create_init_files

    echo "Test structure setup complete!"
    echo "You can now add your test implementations to the created files."
}

main
