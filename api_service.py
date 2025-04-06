"""
Enhanced API Service for the Lecture Video Content Indexer.
Provides a robust REST API with improved error handling, validation,
rate limiting, and documentation.
"""

import os
import json
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Annotated
from datetime import datetime, timedelta

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl, Field, validator, constr
import uvicorn

# Import project modules
from data_access import get_data_access
from youtube_extractor import YouTubeExtractor
from data_pipeline import DataPipeline
from search_engine import SearchEngine
from cache_manager import cache_clear, get_cache_stats
from concept_dedup import ConceptDedupExtension, apply_concept_deduplication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with customized documentation
app = FastAPI(
    title="Lecture Video Content Indexer API",
    description="""
    # Video Lecture Content Indexer API

    This API provides access to the Video Lecture Content Indexer system, which processes
    educational video lectures from YouTube and creates a searchable index of academic concepts.

    ## Key Features

    - **Video Processing**: Submit YouTube videos for processing and indexing
    - **Search**: Find educational content across indexed videos
    - **Concept Exploration**: Explore concepts and their relationships
    - **Theory vs. Practice**: Distinguish between theoretical and practical content

    ## Authentication

    API requests require an API key that should be included in the `X-API-Key` header.
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# API Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Configure API keys (in production, use a more secure approach)
API_KEYS = set(os.environ.get("API_KEYS", "demo_key,test_key").split(","))

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # 1 minute window
RATE_LIMIT_MAX_REQUESTS = 60  # 60 requests per minute
rate_limit_data = {}  # client_id -> {window_start, request_count}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models for API
class VideoSubmission(BaseModel):
    url: HttpUrl = Field(..., description="YouTube video URL")
    language: constr(min_length=2, max_length=5) = Field("en", description="Preferred language code (e.g., 'en', 'ru')")

    @validator('url')
    def validate_youtube_url(cls, v):
        """Validate that the URL is from YouTube."""
        url_str = str(v)
        if "youtube.com" not in url_str and "youtu.be" not in url_str:
            raise ValueError("URL must be from YouTube")
        return v

class FilterOptions(BaseModel):
    video_id: Optional[str] = Field(None, description="Filter by specific video ID")
    video_ids: Optional[List[str]] = Field(None, description="Filter by list of video IDs")
    domain: Optional[str] = Field(None, description="Filter by domain")
    min_theory_ratio: Optional[float] = Field(None, ge=0, le=1, description="Minimum theory/practice ratio")
    max_theory_ratio: Optional[float] = Field(None, ge=0, le=1, description="Maximum theory/practice ratio")

class Pagination(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(10, ge=1, le=100, description="Results per page")

    @property
    def offset(self) -> int:
        """Calculate offset based on page and limit."""
        return (self.page - 1) * self.limit

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    filters: FilterOptions = Field(default_factory=FilterOptions, description="Filter options")
    theory_practice_ratio: Optional[float] = Field(None, ge=0, le=1,
                                                  description="Desired theory/practice ratio (0=practical, 1=theoretical)")
    pagination: Pagination = Field(default_factory=Pagination, description="Pagination options")

class LearningPathRequest(BaseModel):
    concept_ids: List[str] = Field(..., min_items=1, description="List of concept IDs to include in the learning path")
    theory_practice_ratio: float = Field(0.5, ge=0, le=1,
                                        description="Desired theory/practice ratio (0=practical, 1=theoretical)")
    domain: Optional[str] = Field(None, description="Optional domain filter")

class APIResponse(BaseModel):
    """Base response model with common fields."""
    status: str = "success"
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str = Field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])

# Global components
youtube_extractor = None
data_pipeline = None
search_engine = None
data_access = None

# Startup event to initialize components
@app.on_event("startup")
async def startup_event():
    """Initialize API service on startup."""
    global youtube_extractor, data_pipeline, search_engine, data_access

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
        "output_dir": os.environ.get("OUTPUT_DIR", "data/processed"),
        "index_dir": os.environ.get("INDEX_DIR", "data/index"),
        "enable_rate_limiting": os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true",
        "debug_mode": os.environ.get("DEBUG_MODE", "false").lower() == "true"
    }

    # Create necessary directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["index_dir"], exist_ok=True)

    return config

# Authentication dependency
async def get_api_key(
    api_key: str = Depends(api_key_header),
    request: Request = None
) -> str:
    """
    Validate API key and handle rate limiting.

    Args:
        api_key: API key from header
        request: FastAPI request object

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is missing, invalid, or rate limit exceeded
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )

    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

    # Check rate limits if request is provided and rate limiting is enabled
    if request and os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true":
        client_id = f"{api_key}_{request.client.host}"

        # Check if client has rate limit data
        now = time.time()
        client_data = rate_limit_data.get(client_id)

        if not client_data or now - client_data["window_start"] > RATE_LIMIT_WINDOW:
            # New window
            rate_limit_data[client_id] = {
                "window_start": now,
                "request_count": 1
            }
        else:
            # Existing window
            client_data["request_count"] += 1

            # Check if limit exceeded
            if client_data["request_count"] > RATE_LIMIT_MAX_REQUESTS:
                # Calculate reset time
                reset_time = client_data["window_start"] + RATE_LIMIT_WINDOW
                retry_after = int(reset_time - now)

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
                    headers={"Retry-After": str(retry_after)}
                )

    return api_key

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Middleware for consistent error handling."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")

        # Prepare error response
        error_response = {
            "status": "error",
            "detail": str(e),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }

        # Include stack trace in debug mode
        if os.environ.get("DEBUG_MODE", "false").lower() == "true":
            import traceback
            error_response["debug_info"] = {
                "exception_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )

# API endpoints
@app.post("/api/v1/videos", response_model=Dict[str, Any])
async def submit_video(
    submission: VideoSubmission,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
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

        # Check if video already exists and is processed
        existing_video = data_access.get_video(video_id)
        if existing_video and existing_video.get("processing_status") == "completed":
            return {
                "status": "already_processed",
                "video_id": video_id,
                "video_url": str(submission.url),
                "title": existing_video.get("title", ""),
                "processing_time": existing_video.get("indexed_at", "")
            }

        # Process video in background
        background_tasks.add_task(process_video, str(submission.url), submission.language)

        return {
            "status": "submitted",
            "video_id": video_id,
            "video_url": str(submission.url),
            "estimated_time": "30-120 seconds"
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
            success = search_engine.index_content(result)
            if success:
                logger.info(f"Successfully processed and indexed video: {result.get('video_id')}")
            else:
                logger.error(f"Failed to index video: {result.get('video_id')}")
        else:
            logger.error(f"Failed to process video: {result.get('error')}")

    except Exception as e:
        logger.error(f"Error processing video {url}: {e}")

@app.get("/api/v1/videos/{video_id}", response_model=Dict[str, Any])
async def get_video_status(
    video_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Check the processing status of a video.

    - **video_id**: YouTube video ID
    """
    try:
        # Get video from data access layer
        video = data_access.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

        # Build response with video details
        return {
            "video_id": video_id,
            "status": video.get("processing_status", "unknown"),
            "title": video.get("title", ""),
            "channel": video.get("channel", ""),
            "domain": video.get("domain", "unknown"),
            "domain_confidence": video.get("domain_confidence", 0),
            "theory_practice_ratio": video.get("theory_practice_ratio", 0.5),
            "theoretical_segments": video.get("theoretical_segments", 0),
            "practical_segments": video.get("practical_segments", 0),
            "indexed_at": video.get("indexed_at", ""),
            "video_url": f"https://www.youtube.com/watch?v={video_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search", response_model=Dict[str, Any])
async def search_content(
    query: SearchQuery,
    api_key: str = Depends(get_api_key)
):
    """
    Search for content across indexed videos.

    - **query**: Search query text
    - **filters**: Filter criteria
    - **theory_practice_ratio**: Ratio of theoretical to practical content
    - **pagination**: Pagination options
    """
    try:
        # Create structured query
        structured_query = {
            "original_text": query.query,
            "filters": query.filters.dict(exclude_none=True),
            "theory_practice_ratio": query.theory_practice_ratio,
            "domain": query.filters.domain,
            "pagination": {
                "page": query.pagination.page,
                "limit": query.pagination.limit,
                "offset": query.pagination.offset
            }
        }

        # Execute search
        results = search_engine.search(structured_query)

        # Add pagination metadata
        total_results = results.get("totalResults", 0)
        total_pages = (total_results + query.pagination.limit - 1) // query.pagination.limit

        results["pagination"] = {
            "page": query.pagination.page,
            "limit": query.pagination.limit,
            "total_results": total_results,
            "total_pages": total_pages,
            "has_next": query.pagination.page < total_pages,
            "has_prev": query.pagination.page > 1
        }

        # Add query information
        results["query_info"] = {
            "original_text": query.query,
            "theory_practice_ratio": query.theory_practice_ratio,
            "domain": query.filters.domain
        }

        return results

    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/videos/{video_id}/concepts", response_model=Dict[str, Any])
async def get_video_concepts(
    video_id: str,
    context_type: Optional[str] = Query(None, description="Content type filter (theoretical, practical, mixed)"),
    api_key: str = Depends(get_api_key)
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
async def get_concept(
    concept_id: str,
    api_key: str = Depends(get_api_key)
):
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

@app.post("/api/v1/learning-paths", response_model=Dict[str, Any])
async def create_learning_path(
    request: LearningPathRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Create a learning path from a set of concepts.

    - **concept_ids**: List of concept IDs to include
    - **theory_practice_ratio**: Desired ratio of theoretical to practical content
    - **domain**: Optional domain filter
    """
    try:
        # Generate learning path
        learning_path = search_engine.generate_learning_path(
            request.concept_ids,
            request.theory_practice_ratio,
            request.domain
        )

        if not learning_path:
            raise HTTPException(
                status_code=404,
                detail="Could not generate learning path with the provided concepts"
            )

        return learning_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating learning path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/domains", response_model=Dict[str, Any])
async def list_domains(
    api_key: str = Depends(get_api_key)
):
    """
    Get a list of available domains and their statistics.
    """
    try:
        # Query domains from database
        domains_query = """
        SELECT domain, COUNT(DISTINCT video_id) as video_count,
               SUM(CASE WHEN concept_class = 'theoretical' THEN 1 ELSE 0 END) as theoretical_concepts,
               SUM(CASE WHEN concept_class = 'practical' THEN 1 ELSE 0 END) as practical_concepts
        FROM concepts
        GROUP BY domain
        ORDER BY video_count DESC
        """

        domains = data_access.execute_query(domains_query)

        return {
            "domains": domains,
            "total_domains": len(domains)
        }

    except Exception as e:
        logger.error(f"Error listing domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    # Check component health
    components_healthy = (
        youtube_extractor is not None and
        data_pipeline is not None and
        search_engine is not None and
        data_access is not None
    )

    # Get database stats
    try:
        db_stats = data_access.get_stats() if components_healthy else {}
        cache_stats = get_cache_stats()

        return {
            "status": "healthy" if components_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "youtube_extractor": youtube_extractor is not None,
                "data_pipeline": data_pipeline is not None,
                "search_engine": search_engine is not None,
                "data_access": data_access is not None
            },
            "database": db_stats,
            "cache": cache_stats
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/api/v1/clear-cache", response_model=Dict[str, Any])
async def clear_cache_endpoint(
    cache_type: Optional[str] = Query(None, description="Specific cache to clear"),
    api_key: str = Depends(get_api_key)
):
    """
    Clear all caches or a specific cache.

    - **cache_type**: Optional specific cache to clear (video, transcript, search, concept)
    """
    try:
        if cache_type and cache_type not in ["video", "transcript", "search", "concept"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cache type: {cache_type}. Valid types are: video, transcript, search, concept"
            )

        cache_clear(cache_type)

        return {
            "status": "success",
            "message": f"Cache {'for ' + cache_type if cache_type else 'all caches'} cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ConceptLinkRequest(BaseModel):
    """Request model for linking concepts."""
    canonical_concept_id: str = Field(..., description="ID of the canonical concept")
    variant_concept_ids: List[str] = Field(..., description="List of variant concept IDs to link to the canonical concept")

@app.post("/api/v1/concepts/link", response_model=Dict[str, Any])
async def link_concepts(
    request: ConceptLinkRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Link variant concepts to a canonical concept.
    This allows manual correction of concept relationships.

    - **canonical_concept_id**: ID of the canonical concept
    - **variant_concept_ids**: List of concept IDs to mark as variants of the canonical concept
    """
    try:
        # Verify canonical concept exists
        canonical_concept = data_access.get_concept(request.canonical_concept_id)
        if not canonical_concept:
            raise HTTPException(
                status_code=404,
                detail=f"Canonical concept not found: {request.canonical_concept_id}"
            )

        # Ensure canonical concept doesn't have its own canonical (it must be a root concept)
        if canonical_concept.get("canonical_concept_id"):
            raise HTTPException(
                status_code=400,
                detail=f"Specified canonical concept is itself a variant of another concept"
            )

        # Update each variant concept
        updated_count = 0
        skipped_count = 0
        errors = []

        for variant_id in request.variant_concept_ids:
            # Skip if variant is the same as canonical
            if variant_id == request.canonical_concept_id:
                skipped_count += 1
                continue

            # Get variant concept
            variant = data_access.get_concept(variant_id)
            if not variant:
                errors.append(f"Variant concept not found: {variant_id}")
                continue

            # Update variant to point to canonical
            try:
                update_query = """
                UPDATE concepts
                SET canonical_concept_id = ?
                WHERE concept_id = ?
                """
                data_access.execute_update(update_query, (request.canonical_concept_id, variant_id))
                updated_count += 1

                # Clear cache for this concept
                data_access.clear_cache(f"concept_{variant_id}")

            except Exception as e:
                errors.append(f"Error updating concept {variant_id}: {str(e)}")

        # Clear search cache to reflect changes
        cache_clear("search")

        # Clear cache for canonical concept
        data_access.clear_cache(f"concept_{request.canonical_concept_id}")

        return {
            "status": "success",
            "canonical_concept_id": request.canonical_concept_id,
            "canonical_concept_text": canonical_concept.get("text", ""),
            "updated_variants": updated_count,
            "skipped": skipped_count,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error linking concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/concepts/deduplicate", response_model=Dict[str, Any])
async def deduplicate_concepts(
    domain: Optional[str] = Query(None, description="Optional domain to limit deduplication"),
    language: Optional[str] = Query(None, description="Optional language to limit deduplication"),
    threshold: float = Query(0.80, description="Similarity threshold (0.0-1.0)"),
    api_key: str = Depends(get_api_key)
):
    """
    Run automatic concept deduplication process.

    - **domain**: Optional domain to limit deduplication scope
    - **language**: Optional language to limit deduplication scope
    - **threshold**: Similarity threshold for merging concepts (0.0-1.0)
    """
    try:
        # Import concept deduplication extension
        try:
            from concept_dedup import ConceptDedupExtension
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Concept deduplication module not available"
            )

        # Instantiate deduplicator
        deduplicator = ConceptDedupExtension(data_access, language)

        # Get concepts to deduplicate
        query = """
        SELECT *
        FROM concepts
        WHERE canonical_concept_id IS NULL OR canonical_concept_id = ''
        """

        params = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        if language:
            query += " AND language = ?"
            params.append(language)

        # Add limit to prevent processing too many concepts at once
        query += " LIMIT 1000"

        concepts = data_access.execute_query(query, tuple(params))

        if not concepts:
            return {
                "status": "success",
                "message": "No concepts found for deduplication",
                "processed": 0,
                "linked": 0
            }

        # Process in batches for better performance
        processed_count = 0
        linked_count = 0
        canonical_map = {}  # concept_id -> canonical_concept_id

        # First pass: identify similar concepts and establish canonical relationships
        for concept in concepts:
            concept_id = concept.get("concept_id")

            # Skip if already processed
            if concept_id in canonical_map:
                continue

            # Find similar concepts
            similar = deduplicator.find_similar_concepts(
                concept,
                [c for c in concepts if c.get("concept_id") != concept_id],
                threshold=threshold,
                language=language
            )

            # Process similar concepts
            if similar:
                # Determine canonical concept
                candidates = [concept] + similar

                # Sort candidates by quality (score, frequency, word count)
                candidates.sort(key=lambda c: (
                    c.get("score", 0) * 0.4 +
                    c.get("frequency", 0) * 0.3 +
                    len(c.get("text", "").split()) * 0.3
                ), reverse=True)

                # Use the best concept as canonical
                canonical = candidates[0]
                canonical_id = canonical.get("concept_id")

                # Map all other candidates to this canonical
                for candidate in candidates[1:]:
                    candidate_id = candidate.get("concept_id")
                    if candidate_id and candidate_id != canonical_id:
                        canonical_map[candidate_id] = canonical_id
                        linked_count += 1

            processed_count += 1

        # Second pass: update the database with canonical relationships
        update_query = """
        UPDATE concepts
        SET canonical_concept_id = ?
        WHERE concept_id = ?
        """

        for variant_id, canonical_id in canonical_map.items():
            data_access.execute_update(update_query, (canonical_id, variant_id))
            # Clear cache for this concept
            data_access.clear_cache(f"concept_{variant_id}")

        # Clear search cache
        cache_clear("search")

        return {
            "status": "success",
            "message": "Concept deduplication completed",
            "processed": processed_count,
            "linked": linked_count,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in concept deduplication: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configure custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME
        }
    }

    # Apply security to all operations
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if "security" not in operation:
                operation["security"] = [{"APIKeyHeader": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

def start_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the API server."""
    uvicorn.run("api_service:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    # Run the API server directly if this file is executed
    start_server()
