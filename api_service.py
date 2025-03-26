"""
Simplified API Service for the Lecture Video Content Indexer.
Provides a minimal REST API for video processing and search.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn

# Import simplified modules
from data_access import get_data_access
from youtube_extractor import YouTubeExtractor
from data_pipeline import DataPipeline
from search_engine import SearchEngine
from cache_manager import cache_clear

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Lecture Video Content Indexer API",
    description="API for processing and searching educational lecture videos",
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

# Pydantic models for API
class VideoSubmission(BaseModel):
    url: HttpUrl
    language: str = "en"

class SearchQuery(BaseModel):
    query: str
    filters: Dict[str, Any] = {}
    theory_practice_ratio: Optional[float] = None
    domain: Optional[str] = None
    page: int = 1
    limit: int = 10

# Global components
config = {}
youtube_extractor = None
data_pipeline = None
search_engine = None
data_access = None

@app.on_event("startup")
async def startup_event():
    """Initialize API service on startup."""
    global config, youtube_extractor, data_pipeline, search_engine, data_access

    try:
        # Load configuration
        config = load_config()
        logger.info(f"Loaded configuration: {list(config.keys())}")

        # Initialize components
        youtube_api_key = config.get("youtube_api_key", "")
        if youtube_api_key:
            logger.info("Using YouTube API key from configuration")
        else:
            logger.warning("No YouTube API key provided")

        # Initialize data access layer
        data_access = get_data_access()
        logger.info("Data access layer initialized")

        # Initialize components
        youtube_extractor = YouTubeExtractor(youtube_api_key)
        data_pipeline = DataPipeline(config)
        search_engine = SearchEngine(config)
        logger.info("All components initialized")

    except Exception as e:
        logger.critical(f"Failed to initialize API service: {e}")
        raise

def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables and config files.

    Returns:
        Configuration dictionary
    """
    config = {
        "youtube_api_key": os.environ.get("YOUTUBE_API_KEY", ""),
        "output_dir": "data/processed",
        "index_dir": "data/index"
    }

    # Create necessary directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["index_dir"], exist_ok=True)

    return config

@app.post("/api/v1/videos", response_model=Dict[str, Any])
async def submit_video(
    submission: VideoSubmission,
    background_tasks: BackgroundTasks
):
    """
    Submit a new video for processing.

    - **url**: YouTube video URL
    - **language**: Preferred language for transcript ("en" or "ru")
    """
    try:
        # Validate video URL
        valid, video_id = youtube_extractor.validate_video_url(str(submission.url))
        if not valid or not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

        # Process video in background
        background_tasks.add_task(process_video, str(submission.url), submission.language)

        return {
            "status": "submitted",
            "video_id": video_id,
            "video_url": str(submission.url)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video(url: str, language: str):
    """
    Process a video in the background.

    Args:
        url: YouTube video URL
        language: Preferred language
    """
    try:
        # Process the video
        language_preference = [language] if language else ["en", "ru"]
        result = data_pipeline.process_video(url, language_preference)

        # Index the content if processing was successful
        if result.get("status") == "completed":
            search_engine.index_content(result)
            logger.info(f"Successfully processed and indexed video: {result.get('video_id')}")
        else:
            logger.error(f"Failed to process video: {result.get('error')}")

    except Exception as e:
        logger.error(f"Error processing video {url}: {e}")

@app.get("/api/v1/videos/{video_id}", response_model=Dict[str, Any])
async def get_video_status(video_id: str):
    """
    Check the processing status of a video.

    - **video_id**: YouTube video ID
    """
    try:
        # Get video from data access layer
        video = data_access.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

        return {
            "video_id": video_id,
            "status": video.get("processing_status", "unknown"),
            "title": video.get("title", ""),
            "domain": video.get("domain", "unknown"),
            "theory_practice_ratio": video.get("theory_practice_ratio", 0.5)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search", response_model=Dict[str, Any])
async def search_content(query: SearchQuery):
    """
    Search for content across indexed videos.

    - **query**: Search query text
    - **filters**: Filter criteria
    - **theory_practice_ratio**: Ratio of theoretical to practical content
    - **domain**: Domain filter
    - **page**: Page number
    - **limit**: Results per page
    """
    try:
        # Create structured query
        structured_query = {
            "original_text": query.query,
            "filters": query.filters,
            "theory_practice_ratio": query.theory_practice_ratio,
            "domain": query.domain,
            "pagination": {
                "page": query.page,
                "limit": query.limit,
                "offset": (query.page - 1) * query.limit
            }
        }

        # Execute search
        results = search_engine.search(structured_query)

        return results

    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/videos/{video_id}/concepts", response_model=Dict[str, Any])
async def get_video_concepts(
    video_id: str,
    context_type: Optional[str] = Query(None, description="Content type filter (theoretical, practical, mixed)")
):
    """
    Get concepts extracted from a specific video.

    - **video_id**: YouTube video ID
    - **context_type**: Content type filter
    """
    try:
        # Get video concepts
        concepts = search_engine.get_video_concepts(video_id, context_type)
        if not concepts:
            raise HTTPException(status_code=404, detail=f"Video not found or not processed: {video_id}")

        return concepts

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/concepts/{concept_id}", response_model=Dict[str, Any])
async def get_concept(concept_id: str):
    """
    Get detailed information about a specific concept.

    - **concept_id**: Concept ID
    """
    try:
        # Get concept details
        concept = search_engine.get_concept_details(concept_id)
        if not concept:
            raise HTTPException(status_code=404, detail=f"Concept not found: {concept_id}")

        return concept

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting concept: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    return {
        "status": "healthy",
        "components": {
            "youtube_extractor": youtube_extractor is not None,
            "data_pipeline": data_pipeline is not None,
            "search_engine": search_engine is not None,
            "data_access": data_access is not None
        }
    }

@app.post("/api/v1/clear-cache", response_model=Dict[str, Any])
async def clear_cache():
    """
    Clear all caches.
    """
    try:
        cache_clear()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the API server."""
    uvicorn.run("api_service:app", host=host, port=port)

if __name__ == "__main__":
    # Run the API server directly if this file is executed
    start_server()
