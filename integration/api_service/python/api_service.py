"""
API Service module for the Lecture Video Content Indexer.
Provides RESTful API endpoints for video processing and search functionality.
Integrated with database, caching, and performance monitoring.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, HttpUrl
import uvicorn

from database.db_init import init_database, get_db_context
from integration.task_manager.python.task_manager import TaskManager
from common.utils.config_loader import load_config
from common.utils.performance_utils import measure_time, time_function, measure_memory
from common.utils.error_handling import handle_api_error
from data_acquisition.youtube_api.python.youtube_data_extractor import YouTubeDataExtractor

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Lecture Video Content Indexer API",
    description="API for processing and searching educational lecture videos with theory/practice classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models for API
class VideoSubmission(BaseModel):
    url: HttpUrl
    metadata: Dict[str, Any] = {}
    priority: int = 0
    language: str = "en"

class VideoStatus(BaseModel):
    video_id: str
    status: str
    progress: float = 0
    domain: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    filters: Dict[str, Any] = {}
    theory_practice_ratio: Optional[float] = None
    domain: Optional[str] = None
    page: int = 1
    limit: int = 10

class ConceptRequest(BaseModel):
    concept_ids: List[str]
    theory_practice_ratio: float = 0.5
    domain: Optional[str] = None

# Global state
config = None
task_manager = None
youtube_extractor = None
db_context = None

@app.on_event("startup")
async def startup_event():
    """Initialize API service on startup."""
    global config, task_manager, youtube_extractor, db_context

    try:
        # Log the current working directory to help with debugging
        logger.info(f"Current working directory: {os.getcwd()}")

        # Check if config files exist
        api_config_path = "config/api.yaml"
        logger.info(f"Checking for API config at: {api_config_path}")
        if os.path.exists(api_config_path):
            logger.info(f"API config file found at: {api_config_path}")
        else:
            logger.warning(f"API config file NOT FOUND at: {api_config_path}")
            # Try parent directory
            api_config_path = "../config/api.yaml"
            logger.info(f"Trying parent directory: {api_config_path}")
            if os.path.exists(api_config_path):
                logger.info(f"API config file found at: {api_config_path}")

        # Load configuration
        config = load_config(api_config_path)
        logger.info(f"Loaded API configuration: {list(config.keys())}")

        # Initialize database connection
        db_config_path = "config/db_config.yaml"
        db_context = init_database(db_config_path)
        logger.info("Database initialized successfully")

        # Initialize performance monitoring
        from common.utils.performance_utils import initialize as init_performance
        init_performance(
            enable_monitoring=True,
            log_interval=300,
            memory_profiling=False,
            auto_optimize=True
        )
        logger.info("Performance monitoring initialized")

        # Extract YouTube API key (first few and last few chars for security)
        api_key = config.get("youtube_api_key", "")
        if api_key:
            key_prefix = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "[redacted]"
            logger.info(f"Using YouTube API key: {key_prefix}")
        else:
            logger.warning("No YouTube API key found in config")

        # Initialize task manager
        task_manager = TaskManager(config, db_context)
        logger.info("Initialized Task Manager")

        # Initialize YouTube extractor (for URL validation)
        youtube_api_key = config.get("youtube_api_key")
        if not youtube_api_key:
            logger.error("YouTube API key not provided in configuration")
        else:
            youtube_extractor = YouTubeDataExtractor(youtube_api_key)
            logger.info("Initialized YouTube Data Extractor")

        logger.info("API Service started successfully")

    except Exception as e:
        logger.critical(f"Failed to initialize API service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global task_manager, db_context

    if task_manager:
        await task_manager.shutdown()
        logger.info("Task Manager shut down")

    if db_context:
        db_context.close()
        logger.info("Database connection closed")

    logger.info("API Service shut down")

async def verify_token(token: str = Depends(oauth2_scheme)):
    """Verify authentication token."""
    # In a real implementation, validate the token against your auth system
    # For now, we're just checking if a token was provided
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

@time_function(threshold_ms=500)
async def validate_video_url(url: str) -> str:
    """
    Validate a YouTube URL and return the video ID.

    Args:
        url: YouTube URL

    Returns:
        video_id: YouTube video ID

    Raises:
        HTTPException: If URL is invalid
    """
    global youtube_extractor

    if not youtube_extractor:
        raise HTTPException(status_code=500, detail="YouTube extractor not initialized")

    valid, video_id = youtube_extractor.validate_video_url(url)
    if not valid or not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

    return video_id

# API endpoints
@app.post("/api/v1/videos", response_model=Dict[str, Any])
@handle_api_error
async def submit_video(
    submission: VideoSubmission,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Submit a new video for processing.

    - **url**: YouTube video URL
    - **metadata**: Optional metadata for the video
    - **priority**: Processing priority (0-10, higher = higher priority)
    - **language**: Preferred language for transcript ("en" or "ru")
    """
    with measure_time("submit_video_api"):
        try:
            # Validate video URL
            video_id = await validate_video_url(str(submission.url))

            # Create processing job
            job_id = await task_manager.create_video_processing_task(
                video_id=video_id,
                video_url=str(submission.url),
                metadata=submission.metadata,
                priority=submission.priority,
                language=submission.language
            )

            # Start processing in background
            background_tasks.add_task(task_manager.process_video_task, job_id)

            return {
                "job_id": job_id,
                "video_id": video_id,
                "status": "submitted"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error submitting video: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/videos/{video_id}/status", response_model=VideoStatus)
@handle_api_error
async def check_video_status(
    video_id: str,
    token: str = Depends(verify_token)
):
    """
    Check the processing status of a video.

    - **video_id**: YouTube video ID
    """
    with measure_time("check_video_status_api"):
        try:
            # Get status from task manager
            status = await task_manager.get_video_processing_status(video_id)

            if not status:
                raise HTTPException(status_code=404, detail=f"No processing job found for video ID: {video_id}")

            return VideoStatus(
                video_id=video_id,
                status=status.get("status", "unknown"),
                progress=status.get("progress", 0),
                domain=status.get("domain")
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking video status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/search", response_model=Dict[str, Any])
@handle_api_error
async def search_concepts(
    q: str = Query(..., description="Search query text"),
    filters: str = Query("{}", description="JSON string of filter criteria"),
    theory_practice_ratio: Optional[float] = Query(
        None, ge=0, le=1, description="Ratio of theoretical to practical content (0=all practical, 1=all theoretical)"
    ),
    domain: Optional[str] = Query(None, description="Domain filter (mathematics, programming, physics)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Results per page"),
    token: str = Depends(verify_token)
):
    """
    Search for concepts across indexed videos.

    - **q**: Search query text
    - **filters**: JSON string of filter criteria
    - **theory_practice_ratio**: Ratio of theoretical to practical content (0=all practical, 1=all theoretical)
    - **domain**: Domain filter (mathematics, programming, physics)
    - **page**: Page number
    - **limit**: Results per page
    """
    with measure_time("search_concepts_api"):
        try:
            # Parse filters
            filter_dict = {}
            if filters:
                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid filters JSON format")

            # Execute search query
            search_results = await task_manager.search_concepts(
                query=q,
                filters=filter_dict,
                theory_practice_ratio=theory_practice_ratio,
                domain=domain,
                page=page,
                limit=limit
            )

            return search_results

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/concepts/{concept_id}", response_model=Dict[str, Any])
@handle_api_error
async def get_concept(
    concept_id: str,
    token: str = Depends(verify_token)
):
    """
    Get detailed information about a specific concept.

    - **concept_id**: Concept ID
    """
    with measure_time("get_concept_api"):
        try:
            # Get concept details
            concept = await task_manager.get_concept_details(concept_id)

            if not concept:
                raise HTTPException(status_code=404, detail=f"Concept not found: {concept_id}")

            return concept

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting concept: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/videos/{video_id}/concepts", response_model=Dict[str, Any])
@handle_api_error
async def get_video_concepts(
    video_id: str,
    context_type: Optional[str] = Query(
        None, description="Content type filter (theoretical, practical, mixed)"
    ),
    token: str = Depends(verify_token)
):
    """
    Get concepts extracted from a specific video.

    - **video_id**: YouTube video ID
    - **context_type**: Content type filter (theoretical, practical, mixed)
    """
    with measure_time("get_video_concepts_api"):
        try:
            # Get video concepts
            concepts = await task_manager.get_video_concepts(
                video_id=video_id,
                context_type=context_type
            )

            if concepts is None:
                raise HTTPException(status_code=404, detail=f"Video not found or not processed: {video_id}")

            return concepts

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting video concepts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/learning-paths", response_model=Dict[str, Any])
@handle_api_error
async def generate_learning_path(
    concept_request: ConceptRequest,
    token: str = Depends(verify_token)
):
    """
    Generate a learning path for a set of concepts.

    - **concept_ids**: List of concept IDs to include in the learning path
    - **theory_practice_ratio**: Desired ratio of theoretical to practical content
    - **domain**: Optional domain filter
    """
    with measure_time("generate_learning_path_api"):
        try:
            # Generate learning path
            learning_path = await task_manager.generate_learning_path(
                concept_ids=concept_request.concept_ids,
                theory_practice_ratio=concept_request.theory_practice_ratio,
                domain=concept_request.domain
            )

            if not learning_path:
                raise HTTPException(status_code=404, detail="Could not generate learning path for the specified concepts")

            return learning_path

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint to verify service status.

    Returns system health status and key metrics.
    """
    try:
        # Check database connection
        db_status = "healthy"
        if db_context:
            try:
                # Execute a simple query to verify database connection
                db_context.db_manager.execute_query("SELECT 1")
            except Exception as e:
                db_status = f"unhealthy: {str(e)}"
        else:
            db_status = "not initialized"

        # Get basic performance metrics
        from common.utils.performance_utils import get_system_metrics
        system_metrics = get_system_metrics()

        return {
            "status": "healthy",
            "timestamp": system_metrics.get("timestamp"),
            "database": db_status,
            "memory_usage_mb": system_metrics.get("process_memory_rss_mb", 0),
            "cpu_percent": system_metrics.get("process_cpu_percent", 0),
            "uptime": "unknown"  # Would require tracking start time
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def start_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Start the API server."""
    uvicorn.run("api_service:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    # Run the API server directly if this file is executed
    start_server()
