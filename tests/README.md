# Running Tests for Lecture Video Content Indexer

This document provides instructions for running the test suite for the Lecture Video Content Indexer.

## Test Structure

The test suite is organized as follows:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing how components work together
- **End-to-End Tests**: Testing the full workflow

## Prerequisites

Before running tests, make sure you have the following installed:

- Python 3.8 or higher
- pytest
- pytest-asyncio (for async tests)
- All dependencies listed in requirements.txt

## Running Tests

### Installing Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov
```

### Running All Unit Tests

```bash
pytest tests/unit/
```

### Running Specific Component Tests

```bash
# YouTube Data Extractor tests
pytest tests/unit/data_acquisition/test_youtube_data_extractor.py

# Transcript Processor tests
pytest tests/unit/data_acquisition/test_transcript_processor.py

# Domain Classifier tests
pytest tests/unit/concept_analysis/test_domain_classifier.py

# Theory Practice Classifier tests
pytest tests/unit/concept_analysis/test_theory_practice_classifier.py

# Search Engine tests
pytest tests/unit/search_retrieval/test_search_engine.py
```

### Running Integration Tests

Integration tests are marked with the `@pytest.mark.integration` decorator and are skipped by default. To run them:

```bash
pytest tests/integration/ --integration
```

### Running with Coverage

To generate a coverage report:

```bash
pytest --cov=lecture_indexer tests/
```

To generate an HTML coverage report:

```bash
pytest --cov=lecture_indexer --cov-report=html tests/
```

## Writing Tests

### Best Practices

1. **Create Isolated Tests**: Each test should be independent and not rely on the state of other tests.
2. **Use Fixtures**: Use pytest fixtures to set up and tear down test environments.
3. **Mock External Dependencies**: Use unittest.mock to mock external dependencies.
4. **Test Edge Cases**: Include tests for error conditions and edge cases.
5. **Descriptive Names**: Use descriptive names for test functions to clearly indicate what they test.

### Example Test Structure

```python
import pytest
from unittest.mock import patch, MagicMock

# Import the component to test
from lecture_indexer.component import Component

@pytest.fixture
def component():
    """Create a Component instance for testing."""
    return Component()

class TestComponent:
    """Tests for the Component class."""
    
    def test_normal_operation(self, component):
        """Test that the component works correctly under normal conditions."""
        result = component.process("input")
        assert result == "expected output"
    
    def test_error_handling(self, component):
        """Test that the component handles errors correctly."""
        with pytest.raises(ValueError):
            component.process(None)
    
    @patch('external.dependency.function')
    def test_with_mock(self, mock_function, component):
        """Test with a mocked dependency."""
        mock_function.return_value = "mocked result"
        result = component.process_with_dependency("input")
        assert result == "expected with mocked result"
        mock_function.assert_called_once_with("input")
```

## Test Directory Structure

```
tests/
├── unit/                    # Unit tests
│   ├── data_acquisition/    # Tests for data acquisition components
│   ├── concept_analysis/    # Tests for concept analysis components
│   ├── indexing/            # Tests for indexing components
│   └── search_retrieval/    # Tests for search components
│
├── integration/             # Integration tests
│   ├── test_pipeline.py     # Tests for data processing pipeline
│   ├── test_api.py          # Tests for API functionality
│   └── test_search.py       # Tests for search functionality
│
└── conftest.py              # Shared pytest fixtures and configuration
```

## Mocking External Services

For testing components that interact with external services (e.g., YouTube API), use the `unittest.mock` module to mock the external service:

```python
@patch('googleapiclient.discovery.build')
def test_youtube_api(mock_build, component):
    # Set up the mock
    mock_youtube = MagicMock()
    mock_build.return_value = mock_youtube
    
    # Test the component
    result = component.process("youtube_url")
    
    # Assert that the mock was called correctly
    mock_build.assert_called_once_with(
        "youtube", "v3", developerKey="api_key"
    )
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure that the `conftest.py` file adds the project root to the Python path.
2. **Missing Dependencies**: Make sure all required packages are installed.
3. **Permission Issues**: Ensure you have write permissions for test output directories.
