"""
Error handling utilities for the Lecture Video Content Indexer.
"""

import logging
import traceback
import time
import random
from typing import Dict, Any, Callable, TypeVar, Awaitable, Optional, Union
from functools import wraps
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic function return types
T = TypeVar('T')

def handle_api_error(func: Callable) -> Callable:
    """
    Decorator for handling API errors.

    Args:
        func: The function to decorate

    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Log the error
            logger.error(f"API error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())

            # Return error response
            error_detail = str(e)
            status_code = getattr(e, 'status_code', 500)

            from fastapi import HTTPException
            raise HTTPException(
                status_code=status_code,
                detail=error_detail
            )

    return wrapper

def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format an error response.

    Args:
        error: The exception

    Returns:
        Formatted error response
    """
    return {
        "error": str(error),
        "error_type": error.__class__.__name__,
        "status": "error"
    }

def youtube_api_retry(max_retries: int = 3, base_delay: float = 1.0) -> Callable:
    """
    Decorator for handling YouTube API errors with retry logic.

    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay between retries (will be exponentially increased)

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{max_retries} for {func.__name__}")

                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if this is a retriable error
                    if isinstance(e, (
                        # Google API specific errors
                        getattr(googleapiclient.errors, 'HttpError', type(None)),
                        # Network errors
                        ConnectionError,
                        TimeoutError,
                        # General I/O and request errors
                        IOError,
                        # YouTube Transcript API specific errors
                        getattr(youtube_transcript_api, 'CouldNotRetrieveTranscript', type(None)),
                        getattr(youtube_transcript_api, 'TranscriptsDisabled', type(None)),
                    )):
                        # Don't retry on 403 (forbidden) or 404 (not found)
                        if hasattr(e, 'resp') and hasattr(e.resp, 'status') and e.resp.status in (403, 404):
                            logger.warning(f"Non-retriable error {e.resp.status} from YouTube API: {e}")
                            break

                        # If we've reached max retries, give up
                        if attempt >= max_retries:
                            logger.error(f"Max retries ({max_retries}) reached for {func.__name__}: {e}")
                            break

                        # Calculate delay with exponential backoff and jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(f"YouTube API error in {func.__name__}: {e}. Retrying in {delay:.2f} seconds...")

                        # Wait before retrying
                        time.sleep(delay)
                    else:
                        # Non-retriable error
                        logger.error(f"Non-retriable error in {func.__name__}: {e}")
                        break

            # If we get here, all retries failed
            if last_exception:
                raise last_exception

        return wrapper

    return decorator

async def async_youtube_api_retry(
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for handling YouTube API errors with retry logic for async functions.

    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay between retries (will be exponentially increased)

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{max_retries} for {func.__name__}")

                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if this is a retriable error
                    if isinstance(e, (
                        # Google API specific errors
                        getattr(googleapiclient.errors, 'HttpError', type(None)),
                        # Network errors
                        ConnectionError,
                        TimeoutError,
                        # General I/O and request errors
                        IOError,
                        # YouTube Transcript API specific errors
                        getattr(youtube_transcript_api, 'CouldNotRetrieveTranscript', type(None)),
                        getattr(youtube_transcript_api, 'TranscriptsDisabled', type(None)),
                    )):
                        # Don't retry on 403 (forbidden) or 404 (not found)
                        if hasattr(e, 'resp') and hasattr(e.resp, 'status') and e.resp.status in (403, 404):
                            logger.warning(f"Non-retriable error {e.resp.status} from YouTube API: {e}")
                            break

                        # If we've reached max retries, give up
                        if attempt >= max_retries:
                            logger.error(f"Max retries ({max_retries}) reached for {func.__name__}: {e}")
                            break

                        # Calculate delay with exponential backoff and jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(f"YouTube API error in {func.__name__}: {e}. Retrying in {delay:.2f} seconds...")

                        # Wait before retrying (using asyncio.sleep for async functions)
                        await asyncio.sleep(delay)
                    else:
                        # Non-retriable error
                        logger.error(f"Non-retriable error in {func.__name__}: {e}")
                        break

            # If we get here, all retries failed
            if last_exception:
                raise last_exception

        return wrapper

    return decorator

class YouTubeAPIError(Exception):
    """Exception raised for YouTube API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class TranscriptExtractionError(Exception):
    """Exception raised for errors in transcript extraction."""

    def __init__(self, message: str, video_id: str):
        self.message = message
        self.video_id = video_id
        super().__init__(f"{message} (video_id: {video_id})")

class DomainClassificationError(Exception):
    """Exception raised for errors in domain classification."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class TheoryPracticeClassificationError(Exception):
    """Exception raised for errors in theory/practice classification."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class SearchError(Exception):
    """Exception raised for errors in search operations."""

    def __init__(self, message: str, query: Optional[str] = None):
        self.message = message
        self.query = query
        super().__init__(f"{message}" + (f" (query: {query})" if query else ""))
