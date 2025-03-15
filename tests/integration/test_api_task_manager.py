"""
Integration tests for the API Service and Task Manager components.
"""

import pytest
import json
import os
import shutil
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from integration.api_service.python.api_service import app
from integration.task_manager.python.task_manager import TaskManager

# Test configuration
TEST_CONFIG = {
    "youtube_api_key": "test_api_key",
    "task_dir": "test_tasks",
    "result_dir": "test_results",
    "max_workers": 2
}

TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TEST_VIDEO_ID = "dQw4w9WgXcQ"

# Mock task data
MOCK_TASK = {
    "task_id": "test-task-id",
    "video_id": TEST_VIDEO_ID,
    "video_url": TEST_VIDEO_URL,
    "metadata": {},
    "priority": 0,
    "language": "en",
    "status": "pending",
    "progress": 0,
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-01T00:00:00Z"
}

MOCK_COMPLETED_TASK = {
    "task_id": "test-task-id",
    "video_id": TEST_VIDEO_ID,
    "video_url": TEST_VIDEO_URL,
    "metadata": {},
    "priority": 0,
    "language": "en",
    "status": "completed",
    "progress": 100,
    "domain": "mathematics",
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-01T01:00:00Z"
}

# Mock search results
MOCK_SEARCH_RESULTS = {
    "results": [
        {
            "concept_id": "concept1",
            "text": "derivative",
            "domain": "mathematics",
            "context_type": "theoretical",
            "video_id": TEST_VIDEO_ID,
            "video_title": "Test Video",
            "relevance_score": 0.9
        }
    ],
    "total": 1,
    "page": 1,
    "limit": 10,
    "theoretical_count": 1,
    "practical_count": 0
}

# Mock concept details
MOCK_CONCEPT_DETAILS = {
    "concept": {
        "concept_id": "concept1",
        "text": "derivative",
        "domain": "mathematics",
        "concept_class": "theoretical"
    },
    "occurrences": [
        {
            "video_id": TEST_VIDEO_ID,
            "video_title": "Test Video",
            "context_type": "theoretical",
            "relevance_score": 0.9
        }
    ],
    "related": []
}

# Mock video concepts
MOCK_VIDEO_CONCEPTS = {
    "video": {
        "video_id": TEST_VIDEO_ID,
        "title": "Test Video",
        "domain": "mathematics",
        "theory_practice_ratio": 0.8
    },
    "concepts": [
        {
            "concept_id": "concept1",
            "text": "derivative",
            "domain": "mathematics",
            "concept_class": "theoretical"
        }
    ],
    "theory_practice_ratio": 0.8
}

# Mock learning path
MOCK_LEARNING_PATH = {
    "concepts": [
        {
            "concept_id": "concept1",
            "text": "derivative",
            "domain": "mathematics",
            "concept_class": "theoretical",
            "order": 1
        }
    ],
    "theory_practice_ratio": 0.8,
    "total_theoretical_concepts": 1,
    "total_practical_concepts": 0,
    "estimated_total_time_minutes": 30,
    "domain": "mathematics"
}

@pytest.fixture
def mock_config_loader():
    """Mock the config loader."""
    with patch('common.utils.config_loader.load_config') as mock_loader:
        mock_loader.return_value = TEST_CONFIG
        yield mock_loader

@pytest.fixture
def mock_task_manager():
    """Mock the task manager."""
    with patch('integration.task_manager.python.task_manager.TaskManager') as mock_manager:
        # Set up AsyncMock for async methods
        mock_instance = AsyncMock()
        mock_instance.create_video_processing_task = AsyncMock(return_value="test-task-id")
        mock_instance.process_video_task = AsyncMock()
        mock_instance.get_video_processing_status = AsyncMock(return_value=MOCK_COMPLETED_TASK)
        mock_instance.search_concepts = AsyncMock(return_value=MOCK_SEARCH_RESULTS)
        mock_instance.get_concept_details = AsyncMock(return_value=MOCK_CONCEPT_DETAILS)
        mock_instance.get_video_concepts = AsyncMock(return_value=MOCK_VIDEO_CONCEPTS)
        mock_instance.generate_learning_path = AsyncMock(return_value=MOCK_LEARNING_PATH)

        mock_manager.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_youtube_extractor():
    """Mock the YouTube data extractor."""
    with patch('data_acquisition.youtube_api.python.youtube_data_extractor.YouTubeDataExtractor') as mock_extractor:
        mock_instance = MagicMock()
        mock_instance.validate_video_url.return_value = (True, TEST_VIDEO_ID)

        mock_extractor.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def api_client(mock_config_loader, mock_task_manager, mock_youtube_extractor):
    """Create a test client for the FastAPI app."""
    # Mock verify_token dependency
    app.dependency_overrides = {}

    def mock_verify_token():
        return "test-token"

    app.dependency_overrides[app.dependencies[0].dependency] = mock_verify_token

    # Set mock task_manager as global variable in app
    app.app.state.task_manager = mock_task_manager
    app.app.state.youtube_extractor = mock_youtube_extractor

    with TestClient(app) as client:
        yield client

@pytest.fixture
def task_manager():
    """Create a Task Manager instance with mocked components."""
    # Create test directories
    os.makedirs(TEST_CONFIG["task_dir"], exist_ok=True)
    os.makedirs(TEST_CONFIG["result_dir"], exist_ok=True)

    # Create task manager with mocks
    with patch('data_acquisition.youtube_api.python.data_pipeline.DataPipeline') as mock_pipeline, \
         patch('search_retrieval.search_engine.python.search_engine.SearchEngine') as mock_search, \
         patch('common.utils.config_loader.load_config') as mock_config:

        # Set up mock returns
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_search_instance = MagicMock()
        mock_search.return_value = mock_search_instance

        mock_config.return_value = {}

        # Create task manager
        task_manager = TaskManager(TEST_CONFIG)

        # Store mocks for assertions
        task_manager.mock_pipeline = mock_pipeline_instance
        task_manager.mock_search = mock_search_instance

        yield task_manager

        # Clean up test directories
        shutil.rmtree(TEST_CONFIG["task_dir"], ignore_errors=True)
        shutil.rmtree(TEST_CONFIG["result_dir"], ignore_errors=True)

# API Service Tests
class TestAPIService:
    """Test the API Service component."""

    def test_submit_video(self, api_client, mock_task_manager):
        """Test submitting a video for processing."""
        response = api_client.post(
            "/api/v1/videos",
            json={
                "url": TEST_VIDEO_URL,
                "metadata": {"course": "Test Course"},
                "priority": 1,
                "language": "en"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-task-id"
        assert data["video_id"] == TEST_VIDEO_ID
        assert data["status"] == "submitted"

        mock_task_manager.create_video_processing_task.assert_called_once()

    def test_check_video_status(self, api_client, mock_task_manager):
        """Test checking video processing status."""
        response = api_client.get(f"/api/v1/videos/{TEST_VIDEO_ID}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == TEST_VIDEO_ID
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["domain"] == "mathematics"

        mock_task_manager.get_video_processing_status.assert_called_once_with(TEST_VIDEO_ID)

    def test_search_concepts(self, api_client, mock_task_manager):
        """Test searching for concepts."""
        response = api_client.get(
            "/api/v1/search",
            params={
                "q": "derivative",
                "domain": "mathematics",
                "theory_practice_ratio": 0.7,
                "page": 1,
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "derivative"
        assert data["results"][0]["domain"] == "mathematics"
        assert data["theoretical_count"] == 1
        assert data["practical_count"] == 0

        mock_task_manager.search_concepts.assert_called_once()

    def test_get_concept(self, api_client, mock_task_manager):
        """Test getting concept details."""
        response = api_client.get("/api/v1/concepts/concept1")

        assert response.status_code == 200
        data = response.json()
        assert data["concept"]["concept_id"] == "concept1"
        assert data["concept"]["text"] == "derivative"
        assert len(data["occurrences"]) == 1

        mock_task_manager.get_concept_details.assert_called_once_with("concept1")

    def test_get_video_concepts(self, api_client, mock_task_manager):
        """Test getting concepts from a video."""
        response = api_client.get(
            f"/api/v1/videos/{TEST_VIDEO_ID}/concepts",
            params={"context_type": "theoretical"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["video"]["video_id"] == TEST_VIDEO_ID
        assert len(data["concepts"]) == 1
        assert data["concepts"][0]["concept_id"] == "concept1"

        mock_task_manager.get_video_concepts.assert_called_once_with(
            video_id=TEST_VIDEO_ID,
            context_type="theoretical"
        )

    def test_generate_learning_path(self, api_client, mock_task_manager):
        """Test generating a learning path."""
        response = api_client.post(
            "/api/v1/learning-paths",
            json={
                "concept_ids": ["concept1"],
                "theory_practice_ratio": 0.7,
                "domain": "mathematics"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["concepts"]) == 1
        assert data["concepts"][0]["concept_id"] == "concept1"
        assert data["theory_practice_ratio"] == 0.8
        assert data["domain"] == "mathematics"

        mock_task_manager.generate_learning_path.assert_called_once_with(
            concept_ids=["concept1"],
            theory_practice_ratio=0.7,
            domain="mathematics"
        )

# Task Manager Tests
class TestTaskManager:
    """Test the Task Manager component."""

    @pytest.mark.asyncio
    async def test_create_video_processing_task(self, task_manager):
        """Test creating a video processing task."""
        task_id = await task_manager.create_video_processing_task(
            video_id=TEST_VIDEO_ID,
            video_url=TEST_VIDEO_URL,
            metadata={"course": "Test Course"},
            priority=1,
            language="en"
        )

        assert isinstance(task_id, str)
        assert len(task_id) > 0

        # Check that task was saved to file
        task_file = os.path.join(TEST_CONFIG["task_dir"], f"{task_id}.json")
        assert os.path.exists(task_file)

        # Verify task content
        with open(task_file, 'r') as f:
            task = json.load(f)
            assert task["task_id"] == task_id
            assert task["video_id"] == TEST_VIDEO_ID
            assert task["video_url"] == TEST_VIDEO_URL
            assert task["metadata"]["course"] == "Test Course"
            assert task["priority"] == 1
            assert task["language"] == "en"
            assert task["status"] == "pending"
            assert task["progress"] == 0

    @pytest.mark.asyncio
    async def test_process_video_task(self, task_manager):
        """Test processing a video task."""
        # Create a test task
        task_id = await task_manager.create_video_processing_task(
            video_id=TEST_VIDEO_ID,
            video_url=TEST_VIDEO_URL
        )

        # Set up pipeline mock to return success
        mock_result = {
            "job_id": "test-job-id",
            "video_id": TEST_VIDEO_ID,
            "metadata": {"domain": "mathematics"},
            "status": "completed"
        }
        task_manager.mock_pipeline.process_video.return_value = mock_result

        # Process the task
        await task_manager.process_video_task(task_id)

        # Check that pipeline was called
        task_manager.mock_pipeline.process_video.assert_called_once_with(
            TEST_VIDEO_URL, ['en', 'ru']
        )

        # Check that task was updated
        task_file = os.path.join(TEST_CONFIG["task_dir"], f"{task_id}.json")
        with open(task_file, 'r') as f:
            task = json.load(f)
            assert task["status"] == "completed"
            assert task["progress"] == 100
            assert task["domain"] == "mathematics"

    @pytest.mark.asyncio
    async def test_process_video_task_failure(self, task_manager):
        """Test handling failures in video processing."""
        # Create a test task
        task_id = await task_manager.create_video_processing_task(
            video_id=TEST_VIDEO_ID,
            video_url=TEST_VIDEO_URL
        )

        # Set up pipeline mock to raise exception
        task_manager.mock_pipeline.process_video.side_effect = Exception("Processing error")

        # Process the task
        await task_manager.process_video_task(task_id)

        # Check that task was updated with error
        task_file = os.path.join(TEST_CONFIG["task_dir"], f"{task_id}.json")
        with open(task_file, 'r') as f:
            task = json.load(f)
            assert task["status"] == "failed"
            assert "error" in task
            assert "Processing error" in task["error"]

    @pytest.mark.asyncio
    async def test_get_video_processing_status(self, task_manager):
        """Test getting video processing status."""
        # Create a test task
        task_id = await task_manager.create_video_processing_task(
            video_id=TEST_VIDEO_ID,
            video_url=TEST_VIDEO_URL
        )

        # Get status
        status = await task_manager.get_video_processing_status(TEST_VIDEO_ID)

        assert status is not None
        assert status["video_id"] == TEST_VIDEO_ID
        assert status["status"] == "pending"
        assert status["progress"] == 0

    @pytest.mark.asyncio
    async def test_search_concepts(self, task_manager):
        """Test searching for concepts."""
        # Set up search engine mock to return results
        task_manager.mock_search.search.return_value = MOCK_SEARCH_RESULTS

        # Execute search
        results = await task_manager.search_concepts(
            query="derivative",
            domain="mathematics",
            theory_practice_ratio=0.7
        )

        assert results == MOCK_SEARCH_RESULTS
        task_manager.mock_search.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_concept_details(self, task_manager):
        """Test getting concept details."""
        # Set up search engine mock to return concept details
        task_manager.mock_search.get_concept_details.return_value = MOCK_CONCEPT_DETAILS

        # Get concept details
        details = await task_manager.get_concept_details("concept1")

        assert details == MOCK_CONCEPT_DETAILS
        task_manager.mock_search.get_concept_details.assert_called_once_with("concept1")

    @pytest.mark.asyncio
    async def test_get_video_concepts(self, task_manager):
        """Test getting concepts from a video."""
        # Set up search engine mock to return video concepts
        task_manager.mock_search.get_video_concepts.return_value = MOCK_VIDEO_CONCEPTS

        # Get video concepts
        concepts = await task_manager.get_video_concepts(
            video_id=TEST_VIDEO_ID,
            context_type="theoretical"
        )

        assert concepts == MOCK_VIDEO_CONCEPTS
        task_manager.mock_search.get_video_concepts.assert_called_once_with(
            TEST_VIDEO_ID, "theoretical"
        )

    @pytest.mark.asyncio
    async def test_generate_learning_path(self, task_manager):
        """Test generating a learning path."""
        # Set up search engine mock to return learning path
        task_manager.mock_search.generate_learning_path.return_value = MOCK_LEARNING_PATH

        # Generate learning path
        path = await task_manager.generate_learning_path(
            concept_ids=["concept1"],
            theory_practice_ratio=0.7,
            domain="mathematics"
        )

        assert path == MOCK_LEARNING_PATH
        task_manager.mock_search.generate_learning_path.assert_called_once_with(
            ["concept1"], 0.7, "mathematics"
        )
