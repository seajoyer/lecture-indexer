"""
Error handling utilities for the Lecture Video Content Indexer.
"""

import logging
import traceback
from typing import Dict, Any, Callable
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

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
