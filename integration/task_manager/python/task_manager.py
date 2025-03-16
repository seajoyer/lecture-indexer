"""
Task Manager module for the Lecture Video Content Indexer.
Manages asynchronous processing tasks and job status tracking.
"""

import os
import json
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from data_acquisition.youtube_api.python.data_pipeline import DataPipeline
from search_retrieval.search_engine.python.search_engine import SearchEngine
from common.utils.config_loader import load_config

# Configure logging
logger = logging.getLogger(__name__)

class TaskManager:
    """
    Manages asynchronous processing tasks and job status tracking.
    Provides methods for video processing, status retrieval, and search operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Task Manager with configuration.

        Args:
            config: Configuration dictionary
        """
        logger.info("Initializing Task Manager")

        self.config = config
        self.task_dir = config.get("task_dir", "data/tasks")
        self.result_dir = config.get("result_dir", "data/results")

        # Create directories if they don't exist
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        # Log the current working directory to help with debugging
        logger.info(f"Current working directory: {os.getcwd()}")

        # Check if config files exist
        pipeline_config_path = "config/pipeline.yaml"
        search_config_path = "config/search.yaml"

        logger.info(f"Checking for pipeline config at: {pipeline_config_path}")
        if os.path.exists(pipeline_config_path):
            logger.info(f"Pipeline config file found at: {pipeline_config_path}")
        else:
            logger.warning(f"Pipeline config file NOT FOUND at: {pipeline_config_path}")

        logger.info(f"Checking for search config at: {search_config_path}")
        if os.path.exists(search_config_path):
            logger.info(f"Search config file found at: {search_config_path}")
        else:
            logger.warning(f"Search config file NOT FOUND at: {search_config_path}")

        # Initialize components
        self._init_components()

        # Initialize thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))

        # Active tasks dictionary
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

        logger.info("Task Manager initialized")

    def _init_components(self):
        """Initialize pipeline and search components."""
        try:
            # Load pipeline configuration
            pipeline_config = load_config("config/pipeline.yaml")

            # Log pipeline configuration for debugging
            logger.info(f"Loaded pipeline config: {pipeline_config}")

            # Ensure YouTube API key is set
            if "youtube_api_key" not in pipeline_config or not pipeline_config["youtube_api_key"]:
                logger.warning("YouTube API key not found in pipeline config, using from main config")
                # Use API key from main config if available
                if "youtube_api_key" in self.config and self.config["youtube_api_key"]:
                    pipeline_config["youtube_api_key"] = self.config["youtube_api_key"]
                    logger.info("Using YouTube API key from main config")
                else:
                    logger.warning("No YouTube API key found in any config, using test key")

            # Initialize data pipeline
            self.data_pipeline = DataPipeline(pipeline_config)
            logger.info("Initialized Data Pipeline")

            # Initialize search engine
            search_config = load_config("config/search.yaml")
            self.search_engine = SearchEngine(search_config)
            logger.info("Initialized Search Engine")

        except Exception as e:
            logger.error(f"Error initializing Task Manager components: {e}")
            raise

    async def create_video_processing_task(
        self,
        video_id: str,
        video_url: str,
        metadata: Dict[str, Any] = None,
        priority: int = 0,
        language: str = "en"
    ) -> str:
        """
        Create a new video processing task.

        Args:
            video_id: YouTube video ID
            video_url: YouTube video URL
            metadata: Optional video metadata
            priority: Processing priority (0-10)
            language: Preferred language for transcript

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        # Create task metadata
        task = {
            "task_id": task_id,
            "video_id": video_id,
            "video_url": video_url,
            "metadata": metadata or {},
            "priority": priority,
            "language": language,
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Save task metadata to file
        await self._save_task(task)

        logger.info(f"Created video processing task {task_id} for video {video_id}")
        return task_id

    async def process_video_task(self, task_id: str):
        """
        Process a video task asynchronously.

        Args:
            task_id: Task ID
        """
        # Load task
        task = await self._load_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        # Add to active tasks
        self.active_tasks[task_id] = task

        try:
            # Update task status
            task["status"] = "processing"
            task["progress"] = 10
            task["updated_at"] = datetime.now().isoformat()
            await self._save_task(task)

            # Extract parameters
            video_url = task["video_url"]
            language_preference = [task["language"], "en"] if task["language"] != "en" else ["en", "ru"]

            # Process video in a thread to avoid blocking
            logger.info(f"Starting video processing for task {task_id}, video URL: {video_url}")

            # Use a thread for CPU-bound processing
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.data_pipeline.process_video(video_url, language_preference)
            )

            # Update task with domain information
            domain = result.get("metadata", {}).get("domain", "unknown")
            task["domain"] = domain

            # Update task status
            task["status"] = "completed"
            task["progress"] = 100
            task["result_id"] = result.get("job_id")
            task["updated_at"] = datetime.now().isoformat()
            await self._save_task(task)

            # Index the processed content
            await self._index_processed_content(result)

            logger.info(f"Completed video processing task {task_id}")

        except Exception as e:
            logger.error(f"Error processing video task {task_id}: {e}")

            # Update task status
            task["status"] = "failed"
            task["error"] = str(e)
            task["updated_at"] = datetime.now().isoformat()
            await self._save_task(task)

        finally:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    async def get_video_processing_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a video processing task.

        Args:
            video_id: YouTube video ID

        Returns:
            Task status dictionary if found, None otherwise
        """
        # First check active tasks
        for task_id, task in self.active_tasks.items():
            if task.get("video_id") == video_id:
                return {
                    "task_id": task_id,
                    "status": task.get("status", "unknown"),
                    "progress": task.get("progress", 0),
                    "domain": task.get("domain"),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at")
                }

        # Then check saved tasks
        try:
            tasks = []
            task_files = [f for f in os.listdir(self.task_dir) if f.endswith(".json")]

            for filename in task_files:
                try:
                    async with aiofiles.open(os.path.join(self.task_dir, filename), 'r') as f:
                        task_data = json.loads(await f.read())
                        if task_data.get("video_id") == video_id:
                            tasks.append(task_data)
                except Exception as e:
                    logger.error(f"Error reading task file {filename}: {e}")

            if not tasks:
                return None

            # Return the most recently updated task
            tasks.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            task = tasks[0]

            return {
                "task_id": task.get("task_id"),
                "status": task.get("status", "unknown"),
                "progress": task.get("progress", 0),
                "domain": task.get("domain"),
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at")
            }

        except Exception as e:
            logger.error(f"Error getting video processing status for {video_id}: {e}")
            return None

    async def search_concepts(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        theory_practice_ratio: Optional[float] = None,
        domain: Optional[str] = None,
        page: int = 1,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for concepts using the search engine.

        Args:
            query: Search query text
            filters: Additional filters
            theory_practice_ratio: Theory/practice ratio filter (0=all practical, 1=all theoretical)
            domain: Domain filter (mathematics, programming, physics)
            page: Page number
            limit: Results per page

        Returns:
            Search results dictionary
        """
        try:
            # Create structured query
            structured_query = {
                "original_text": query,
                "filters": filters or {},
                "theory_practice_ratio": theory_practice_ratio,
                "domain": domain,
                "pagination": {
                    "offset": (page - 1) * limit,
                    "limit": limit
                }
            }

            # Execute search in a thread to avoid blocking
            search_results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.search_engine.search(structured_query)
            )

            # Process results
            result = {
                "results": search_results.get("results", []),
                "total": search_results.get("totalResults", 0),
                "page": page,
                "limit": limit,
                "theoretical_count": search_results.get("theoreticalResults", 0),
                "practical_count": search_results.get("practicalResults", 0),
                "execution_time_ms": search_results.get("executionTimeMs", 0)
            }

            return result

        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            raise

    async def get_concept_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept.

        Args:
            concept_id: Concept ID

        Returns:
            Concept details dictionary if found, None otherwise
        """
        try:
            # Get concept details from search engine
            concept_details = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.search_engine.get_concept_details(concept_id)
            )

            if not concept_details:
                return None

            return concept_details

        except Exception as e:
            logger.error(f"Error getting concept details for {concept_id}: {e}")
            return None

    async def get_video_concepts(
        self,
        video_id: str,
        context_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get concepts extracted from a video.

        Args:
            video_id: YouTube video ID
            context_type: Content type filter (theoretical, practical, mixed)

        Returns:
            Video concepts dictionary if found, None otherwise
        """
        try:
            # Get video concepts from search engine
            video_concepts = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.search_engine.get_video_concepts(video_id, context_type)
            )

            if not video_concepts:
                return None

            return video_concepts

        except Exception as e:
            logger.error(f"Error getting video concepts for {video_id}: {e}")
            return None

    async def generate_learning_path(
        self,
        concept_ids: List[str],
        theory_practice_ratio: float = 0.5,
        domain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a learning path for a set of concepts.

        Args:
            concept_ids: List of concept IDs
            theory_practice_ratio: Desired ratio of theoretical to practical content
            domain: Optional domain filter

        Returns:
            Learning path dictionary if successful, None otherwise
        """
        try:
            # Generate learning path from search engine
            learning_path = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.search_engine.generate_learning_path(
                    concept_ids, theory_practice_ratio, domain
                )
            )

            if not learning_path:
                return None

            return learning_path

        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return None

    async def _save_task(self, task: Dict[str, Any]):
        """
        Save task metadata to file.

        Args:
            task: Task metadata dictionary
        """
        task_id = task.get("task_id")
        if not task_id:
            logger.error("Task ID missing in task metadata")
            return

        filepath = os.path.join(self.task_dir, f"{task_id}.json")

        try:
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(task, ensure_ascii=False, indent=2))

            logger.debug(f"Saved task metadata to {filepath}")

        except Exception as e:
            logger.error(f"Error saving task metadata: {e}")

    async def _load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Load task metadata from file.

        Args:
            task_id: Task ID

        Returns:
            Task metadata dictionary if found, None otherwise
        """
        filepath = os.path.join(self.task_dir, f"{task_id}.json")

        try:
            if not os.path.exists(filepath):
                logger.warning(f"Task file not found: {filepath}")
                return None

            async with aiofiles.open(filepath, 'r') as f:
                task_data = json.loads(await f.read())

            logger.debug(f"Loaded task metadata from {filepath}")
            return task_data

        except Exception as e:
            logger.error(f"Error loading task metadata: {e}")
            return None

    async def _index_processed_content(self, result: Dict[str, Any]):
        """
        Index processed content in the search engine.

        Args:
            result: Processing result dictionary
        """
        try:
            # Index the content in a thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.search_engine.index_content(result)
            )

            logger.info(f"Indexed content for video {result.get('video_id')}")

        except Exception as e:
            logger.error(f"Error indexing content: {e}")

    async def shutdown(self):
        """Shut down the Task Manager and clean up resources."""
        logger.info("Shutting down Task Manager")

        # Shut down executor
        self.executor.shutdown(wait=True)
        logger.info("Thread pool executor shut down")
