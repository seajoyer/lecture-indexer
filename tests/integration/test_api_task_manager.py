"""
API and Task Manager tests for the Lecture Video Content Indexer.
Tests the integration between API service and task manager components.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from integration.api_service.python.api_service import app
from integration.task_manager.python.task_manager import TaskManager

# Create a test client for the FastAPI app
client = TestClient(app)

@pytest.fixture
def mock_task_manager():
    """Create a mock task manager for testing."""
    with patch("integration.api_service.python.api_service.TaskManager", autospec=True) as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # Mock async methods
        mock_instance.create_video_processing_task = AsyncMock(return_value="test_job_id")
        mock_instance.get_video_processing_status = AsyncMock(return_value={
            "status": "processing",
            "progress": 0.5,
            "domain": "mathematics"
        })
        mock_instance.search_concepts = AsyncMock(return_value={
            "results": [{"id": "concept1", "text": "Calculus"}],
            "totalResults": 1
        })
        mock_instance.get_concept_details = AsyncMock(return_value={
            "concept": {"id": "concept1", "text": "Calculus"},
            "occurrences": [{"video_id": "test123"}]
        })
        mock_instance.get_video_concepts = AsyncMock(return_value={
            "video": {"video_id": "test123", "title": "Test Video"},
            "concepts": [{"id": "concept1", "text": "Calculus"}]
        })
        mock_instance.generate_learning_path = AsyncMock(return_value={
            "concepts": [{"id": "concept1", "text": "Calculus"}],
            "total_concepts": 1
        })
        mock_instance.process_video_task = AsyncMock()

        # Simulate task_queue
        mock_instance.task_queue = MagicMock()
        mock_instance.task_queue.put = AsyncMock()

        yield mock_instance

@pytest.fixture
def mock_oauth_token():
    """Mock OAuth token verification."""
    with patch("integration.api_service.python.api_service.verify_token", return_value="test_token"):
        yield

@pytest.mark.asyncio
async def test_api_initialization(test_db_context):
    """Test API service initialization."""
    # Test startup event
    with patch("integration.api_service.python.api_service.load_config") as mock_load_config, \
         patch("integration.api_service.python.api_service.init_database") as mock_init_db, \
         patch("integration.api_service.python.api_service.TaskManager") as mock_task_manager:

        # Configure mocks
        mock_load_config.return_value = {"youtube_api_key": "test_key"}
        mock_init_db.return_value = test_db_context
        mock_task_manager.return_value = MagicMock()

        # Call startup event
        from integration.api_service.python.api_service import startup_event
        await startup_event()

        # Verify initialization
        mock_load_config.assert_called_once()
        mock_init_db.assert_called_once()
        mock_task_manager.assert_called_once()

        # Check global variables were set
        from integration.api_service.python.api_service import config, task_manager, db_context
        assert config is not None
        assert task_manager is not None
        assert db_context is not None

@pytest.mark.asyncio
async def test_submit_video_endpoint(mock_task_manager, mock_oauth_token):
    """Test the /api/v1/videos endpoint for submitting videos."""
    with patch("integration.api_service.python.api_service.task_manager", mock_task_manager):
        response = client.post(
            "/api/v1/videos",
            json={
                "url": "https://www.youtube.com/watch?v=test123",
                "metadata": {"course": "Test Course"},
                "priority": 2,
                "language": "en"
            },
            headers={"Authorization": "Bearer test_token"}
        )

    # Verify response
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "test_job_id",
        "video_id": "test123",
        "status": "submitted"
    }

    # Verify task manager was called correctly
    mock_task_manager.create_video_processing_task.assert_called_once()
    # We can check that the video_url parameter was passed
    args, kwargs = mock_task_manager.create_video_processing_task.call_args
    assert kwargs.get("video_url") == "https://www.youtube.com/watch?v=test123"

@pytest.mark.asyncio
async def test_check_video_status_endpoint(mock_task_manager, mock_oauth_token):
    """Test the /api/v1/videos/{video_id}/status endpoint."""
    with patch("integration.api_service.python.api_service.task_manager", mock_task_manager):
        response = client.get(
            "/api/v1/videos/test123/status",
            headers={"Authorization": "Bearer test_token"}
        )

    # Verify response
    assert response.status_code == 200
    assert response.json() == {
        "video_id": "test123",
        "status": "processing",
        "progress": 0.5,
        "domain": "mathematics"
    }

    # Verify task manager was called
    mock_task_manager.get_video_processing_status.assert_called_once_with("test123")

@pytest.mark.asyncio
async def test_search_concepts_endpoint(mock_task_manager, mock_oauth_token):
    """Test the /api/v1/search endpoint."""
    with patch("integration.api_service.python.api_service.task_manager", mock_task_manager):
        response = client.get(
            "/api/v1/search?q=calculus&domain=mathematics&page=1&limit=10",
            headers={"Authorization": "Bearer test_token"}
        )

    # Verify response
    assert response.status_code == 200
    assert "results" in response.json()
    assert response.json()["results"] == [{"id": "concept1", "text": "Calculus"}]
    assert response.json()["totalResults"] == 1

    # Verify task manager was called with correct parameters
    mock_task_manager.search_concepts.assert_called_once()
    call_args = mock_task_manager.search_concepts.call_args[1]
    assert call_args["query"] == "calculus"
    assert call_args["domain"] == "mathematics"
    assert call_args["page"] == 1
    assert call_args["limit"] == 10

@pytest.mark.asyncio
async def test_get_concept_details_endpoint(mock_task_manager, mock_oauth_token):
    """Test the /api/v1/concepts/{concept_id} endpoint."""
    with patch("integration.api_service.python.api_service.task_manager", mock_task_manager):
        response = client.get(
            "/api/v1/concepts/concept1",
            headers={"Authorization": "Bearer test_token"}
        )

    # Verify response
    assert response.status_code == 200
    assert "concept" in response.json()
    assert response.json()["concept"] == {"id": "concept1", "text": "Calculus"}
    assert "occurrences" in response.json()
    assert response.json()["occurrences"] == [{"video_id": "test123"}]

    # Verify task manager was called
    mock_task_manager.get_concept_details.assert_called_once_with("concept1")

@pytest.mark.asyncio
async def test_get_video_concepts_endpoint(mock_task_manager, mock_oauth_token):
    """Test the /api/v1/videos/{video_id}/concepts endpoint."""
    with patch("integration.api_service.python.api_service.task_manager", mock_task_manager):
        response = client.get(
            "/api/v1/videos/test123/concepts?context_type=theoretical",
            headers={"Authorization": "Bearer test_token"}
        )

    # Verify response
    assert response.status_code == 200
    assert "video" in response.json()
    assert response.json()["video"] == {"video_id": "test123", "title": "Test Video"}
    assert "concepts" in response.json()
    assert response.json()["concepts"] == [{"id": "concept1", "text": "Calculus"}]

    # Verify task manager was called
    mock_task_manager.get_video_concepts.assert_called_once_with(
        video_id="test123",
        context_type="theoretical"
    )

@pytest.mark.asyncio
async def test_generate_learning_path_endpoint(mock_task_manager, mock_oauth_token):
    """Test the /api/v1/learning-paths endpoint."""
    with patch("integration.api_service.python.api_service.task_manager", mock_task_manager):
        response = client.post(
            "/api/v1/learning-paths",
            json={
                "concept_ids": ["concept1", "concept2"],
                "theory_practice_ratio": 0.7,
                "domain": "mathematics"
            },
            headers={"Authorization": "Bearer test_token"}
        )

    # Verify response
    assert response.status_code == 200
    assert "concepts" in response.json()
    assert response.json()["concepts"] == [{"id": "concept1", "text": "Calculus"}]
    assert "total_concepts" in response.json()
    assert response.json()["total_concepts"] == 1

    # Verify task manager was called
    mock_task_manager.generate_learning_path.assert_called_once_with(
        concept_ids=["concept1", "concept2"],
        theory_practice_ratio=0.7,
        domain="mathematics"
    )

@pytest.mark.asyncio
async def test_health_check_endpoint(test_db_context):
    """Test the /api/v1/health endpoint."""
    # Mock the database connection
    with patch("integration.api_service.python.api_service.db_context", test_db_context):
        response = client.get("/api/v1/health")

    # Verify response
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"
    assert "database" in response.json()

@pytest.mark.asyncio
async def test_submit_video_error_handling(mock_task_manager, mock_oauth_token):
    """Test error handling in the submit video endpoint."""
    # Configure mock to raise an exception
    mock_task_manager.create_video_processing_task.side_effect = ValueError("Invalid video URL")

    with patch("integration.api_service.python.api_service.task_manager", mock_task_manager):
        response = client.post(
            "/api/v1/videos",
            json={
                "url": "invalid_url",
                "metadata": {},
                "priority": 0,
                "language": "en"
            },
            headers={"Authorization": "Bearer test_token"}
        )

    # Verify response indicates error
    assert response.status_code != 200
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_api_authentication():
    """Test API authentication."""
    # Test without token
    response = client.post(
        "/api/v1/videos",
        json={
            "url": "https://www.youtube.com/watch?v=test123",
            "metadata": {},
            "priority": 0,
            "language": "en"
        }
    )

    # Should fail due to missing token
    assert response.status_code == 401

    # Test with invalid token
    with patch("integration.api_service.python.api_service.verify_token",
              side_effect=HTTPException(status_code=401, detail="Invalid token")):
        response = client.post(
            "/api/v1/videos",
            json={
                "url": "https://www.youtube.com/watch?v=test123",
                "metadata": {},
                "priority": 0,
                "language": "en"
            },
            headers={"Authorization": "Bearer invalid_token"}
        )

    # Should fail due to invalid token
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_task_manager_initialization(test_db_context):
    """Test TaskManager initialization."""
    config = {"youtube_api_key": "test_key", "max_workers": 2}

    # Mock components to avoid actual initialization
    with patch("integration.task_manager.python.task_manager.YouTubeDataExtractor") as mock_extractor, \
         patch("integration.task_manager.python.task_manager.TranscriptProcessor") as mock_processor, \
         patch("integration.task_manager.python.task_manager.DomainClassifier") as mock_classifier, \
         patch("integration.task_manager.python.task_manager.TheoryPracticeClassifier") as mock_tp_classifier, \
         patch("integration.task_manager.python.task_manager.SearchEngine") as mock_search_engine:

        # Initialize task manager
        task_manager = TaskManager(config, test_db_context)

        # Verify components were initialized
        mock_extractor.assert_called_once_with("test_key")
        mock_processor.assert_called_once()
        mock_classifier.assert_called_once_with(config)
        mock_tp_classifier.assert_called_once()
        mock_search_engine.assert_called_once_with(config)

        # Verify task manager properties
        assert task_manager.config == config
        assert task_manager.max_workers == 2
        assert task_manager.db_context == test_db_context
        assert hasattr(task_manager, "worker_semaphore")
        assert hasattr(task_manager, "shutdown_event")
        assert hasattr(task_manager, "task_queue")

@pytest.mark.asyncio
async def test_task_manager_create_video_task(test_db_context):
    """Test TaskManager create_video_processing_task."""
    # Create task manager with mocked components
    config = {"youtube_api_key": "test_key"}
    task_manager = TaskManager(config, test_db_context)

    # Mock YouTube extractor for URL validation
    mock_extractor = MagicMock()
    mock_extractor.validate_video_url.return_value = (True, "test123")
    task_manager.youtube_extractor = mock_extractor

    # Mock db_context.video_repository
    mock_video_repo = MagicMock()
    mock_video_repo.add_to_processing_queue.return_value = "queue123"
    test_db_context.video_repository = mock_video_repo

    # Mock task_queue
    task_manager.task_queue = MagicMock()
    task_manager.task_queue.put = AsyncMock()

    # Create a video processing task
    task_id = await task_manager.create_video_processing_task(
        video_id=None,
        video_url="https://www.youtube.com/watch?v=test123",
        metadata={"course": "Test Course"},
        priority=2,
        language="en"
    )

    # Verify task was created
    assert task_id == "queue123"
    mock_extractor.validate_video_url.assert_called_once_with("https://www.youtube.com/watch?v=test123")
    mock_video_repo.add_to_processing_queue.assert_called_once()
    task_manager.task_queue.put.assert_awaited_once()

@pytest.mark.asyncio
async def test_task_manager_get_video_status(test_db_context):
    """Test TaskManager get_video_processing_status."""
    # Create task manager
    config = {"youtube_api_key": "test_key"}
    task_manager = TaskManager(config, test_db_context)

    # Mock cache
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    task_manager.cache = mock_cache

    # Mock active tasks
    task_manager.active_tasks = {
        "task123": {
            "video_id": "test123",
            "status": "processing",
            "progress": 0.5,
            "domain": "mathematics"
        }
    }

    # Get status for active task
    status = await task_manager.get_video_processing_status("test123")

    # Verify status
    assert status is not None
    assert status["status"] == "processing"
    assert status["progress"] == 0.5
    assert status["domain"] == "mathematics"

    # Test status for non-existent video
    # First mock the database query
    mock_get_video_data = AsyncMock(return_value=None)
    task_manager._get_video_data = mock_get_video_data
    mock_get_queue_item = AsyncMock(return_value=None)
    task_manager._get_queue_item = mock_get_queue_item

    # Get status
    status = await task_manager.get_video_processing_status("nonexistent")

    # Verify status is None
    assert status is None
    mock_get_video_data.assert_awaited_once_with("nonexistent")
    mock_get_queue_item.assert_awaited_once_with("nonexistent")

@pytest.mark.asyncio
async def test_task_manager_search_concepts(test_db_context):
    """Test TaskManager search_concepts."""
    # Create task manager
    config = {"youtube_api_key": "test_key"}
    task_manager = TaskManager(config, test_db_context)

    # Mock cache
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    task_manager.cache = mock_cache

    # Mock search engine
    mock_search_engine = MagicMock()
    expected_results = {
        "results": [{"id": "concept1", "text": "Calculus"}],
        "totalResults": 1
    }
    mock_search_engine.search.return_value = expected_results
    task_manager.search_engine = mock_search_engine

    # Perform search
    results = await task_manager.search_concepts(
        query="calculus",
        filters={"domain": "mathematics"},
        theory_practice_ratio=0.7,
        domain="mathematics",
        page=1,
        limit=10
    )

    # Verify results
    assert results == expected_results
    mock_search_engine.search.assert_called_once()
    search_query = mock_search_engine.search.call_args[0][0]
    assert search_query["original_text"] == "calculus"
    assert search_query["filters"] == {"domain": "mathematics"}
    assert search_query["theory_practice_ratio"] == 0.7
    assert search_query["domain"] == "mathematics"
    assert search_query["pagination"]["page"] == 1
    assert search_query["pagination"]["limit"] == 10

@pytest.mark.asyncio
async def test_process_video_task_with_error(test_db_context):
    """Test error handling in process_video_task."""
    # Create task manager
    config = {"youtube_api_key": "test_key"}
    task_manager = TaskManager(config, test_db_context)

    # Mock components to raise an exception
    mock_extractor = MagicMock()
    mock_extractor.extract_video_metadata.side_effect = ValueError("API error")
    task_manager.youtube_extractor = mock_extractor

    # Create a task
    task = {
        "id": "task123",
        "type": "video_processing",
        "video_id": "test123",
        "video_url": "https://www.youtube.com/watch?v=test123",
        "status": "pending",
        "progress": 0.0
    }

    # Process the task (should catch the error)
    await task_manager.process_video_task(task)

    # Verify task status is updated with error
    assert task["status"] == "error"
    assert "error" in task
    assert task["progress"] == 0.0
