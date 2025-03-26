"""
Simple caching utilities for the Lecture Video Content Indexer.
Replaces the complex cache_manager.py with basic dictionary caches and LRU cache decorator.
"""

import time
import logging
import functools
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Simple module-level caches for different types of data
video_cache: Dict[str, Any] = {}
transcript_cache: Dict[str, Any] = {}
search_cache: Dict[str, Any] = {}
concept_cache: Dict[str, Any] = {}

def get_cache(cache_type: str) -> Dict[str, Any]:
    """
    Get a specific cache dictionary.

    Args:
        cache_type: Type of cache to get ("video", "transcript", "search", "concept")

    Returns:
        Cache dictionary
    """
    if cache_type == "video":
        return video_cache
    elif cache_type == "transcript":
        return transcript_cache
    elif cache_type == "search":
        return search_cache
    elif cache_type == "concept":
        return concept_cache
    else:
        # Default to video cache
        return video_cache

def cache_set(cache_type: str, key: str, value: Any) -> None:
    """
    Store a value in a specific cache.

    Args:
        cache_type: Type of cache to use
        key: Cache key
        value: Value to cache
    """
    cache = get_cache(cache_type)
    cache[key] = {
        "value": value,
        "timestamp": time.time()
    }

def cache_get(cache_type: str, key: str) -> Optional[Any]:
    """
    Get a value from a specific cache.

    Args:
        cache_type: Type of cache to use
        key: Cache key

    Returns:
        Cached value or None if not found
    """
    cache = get_cache(cache_type)
    entry = cache.get(key)

    if entry:
        return entry["value"]

    return None

def cache_clear(cache_type: str = None) -> None:
    """
    Clear a specific cache or all caches.

    Args:
        cache_type: Type of cache to clear (None for all)
    """
    if cache_type:
        cache = get_cache(cache_type)
        cache.clear()
    else:
        video_cache.clear()
        transcript_cache.clear()
        search_cache.clear()
        concept_cache.clear()

    logger.info(f"Cleared {'all caches' if cache_type is None else f'{cache_type} cache'}")

def cached(cache_type: str):
    """
    Decorator for caching function results.

    Args:
        cache_type: Type of cache to use

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a simple key from function name and arguments
            key_parts = [func.__name__]

            # Add positional arguments
            for arg in args:
                if isinstance(arg, (str, int, float, bool, type(None))):
                    key_parts.append(str(arg))

            # Add keyword arguments (sorted for consistency)
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                if isinstance(v, (str, int, float, bool, type(None))):
                    key_parts.append(f"{k}={v}")

            # Create a combined key
            key = "_".join(key_parts)

            # Check cache
            cached_value = cache_get(cache_type, key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_set(cache_type, key, result)

            return result

        return wrapper

    return decorator

# Examples of using the lru_cache for specific functions
video_metadata_cache = functools.lru_cache(maxsize=100)
search_results_cache = functools.lru_cache(maxsize=50)
