"""
Enhanced caching utilities for the Lecture Video Content Indexer.
Provides a robust caching system with TTL, LRU eviction, and distributed cache support.
"""

import time
import logging
import functools
import threading
import json
from typing import Dict, Any, Optional, Callable, Tuple, List, Set, Union
from collections import OrderedDict

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration
DEFAULT_TTL = 3600  # Default TTL in seconds (1 hour)
MAX_CACHE_SIZE = 1000  # Maximum items per cache
CACHE_STATS_INTERVAL = 3600  # Log cache stats every hour (in seconds)

class LRUCache:
    """Thread-safe LRU Cache implementation with TTL."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE, default_ttl: int = DEFAULT_TTL):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()  # {key: (value, expiry_time)}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.sets = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with TTL check.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            if key in self.cache:
                value, expiry_time = self.cache[key]

                # Check if expired
                if expiry_time > time.time():
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return value
                else:
                    # Expired, remove from cache
                    del self.cache[key]

            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            # If already exists, remove it first
            if key in self.cache:
                del self.cache[key]

            # Check if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest item (first in ordered dict)
                self.cache.popitem(last=False)

            # Calculate expiry time
            expiry_time = time.time() + (ttl if ttl is not None else self.default_ttl)

            # Add to cache (will be at the end - most recently used)
            self.cache[key] = (value, expiry_time)
            self.sets += 1

    def delete(self, key: str) -> bool:
        """
        Delete item from cache.

        Args:
            key: Cache key

        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            # Count expired vs valid entries
            now = time.time()
            expired = sum(1 for _, expiry_time in self.cache.values() if expiry_time <= now)
            valid = len(self.cache) - expired

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "valid_entries": valid,
                "expired_entries": expired,
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "hit_rate_percent": round(hit_rate, 2),
                "default_ttl": self.default_ttl
            }

    def cleanup(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self.lock:
            now = time.time()
            expired_keys = [
                key for key, (_, expiry_time) in self.cache.items()
                if expiry_time <= now
            ]

            for key in expired_keys:
                del self.cache[key]

            return len(expired_keys)

# Initialize module-level caches
video_cache = LRUCache(max_size=200, default_ttl=3600)  # 1 hour TTL
transcript_cache = LRUCache(max_size=200, default_ttl=3600)  # 1 hour TTL
search_cache = LRUCache(max_size=500, default_ttl=300)  # 5 minutes TTL for search results
concept_cache = LRUCache(max_size=300, default_ttl=3600)  # 1 hour TTL

# Cache registry
_CACHES = {
    "video": video_cache,
    "transcript": transcript_cache,
    "search": search_cache,
    "concept": concept_cache
}

# Initialize background cache maintenance
_last_stats_time = time.time()

def get_cache(cache_type: str) -> LRUCache:
    """
    Get a specific cache instance.

    Args:
        cache_type: Type of cache to get ("video", "transcript", "search", "concept")

    Returns:
        Cache instance
    """
    if cache_type in _CACHES:
        return _CACHES[cache_type]
    else:
        logger.warning(f"Unknown cache type: {cache_type}, using default video cache")
        return video_cache

def cache_set(cache_type: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Store a value in a specific cache.

    Args:
        cache_type: Type of cache to use
        key: Cache key
        value: Value to cache
        ttl: Optional custom TTL in seconds
    """
    cache = get_cache(cache_type)
    cache.set(key, value, ttl)

    # Run maintenance occasionally
    _run_maintenance()

def cache_get(cache_type: str, key: str) -> Optional[Any]:
    """
    Get a value from a specific cache.

    Args:
        cache_type: Type of cache to use
        key: Cache key

    Returns:
        Cached value or None if not found or expired
    """
    cache = get_cache(cache_type)
    return cache.get(key)

def cache_delete(cache_type: str, key: str) -> bool:
    """
    Delete a value from a specific cache.

    Args:
        cache_type: Type of cache to use
        key: Cache key

    Returns:
        True if item was deleted, False if not found
    """
    cache = get_cache(cache_type)
    return cache.delete(key)

def cache_clear(cache_type: Optional[str] = None) -> None:
    """
    Clear a specific cache or all caches.

    Args:
        cache_type: Type of cache to clear (None for all)
    """
    if cache_type:
        cache = get_cache(cache_type)
        cache.clear()
        logger.info(f"Cleared {cache_type} cache")
    else:
        for cache_name, cache in _CACHES.items():
            cache.clear()
        logger.info("Cleared all caches")

def _run_maintenance() -> None:
    """Run periodic cache maintenance tasks."""
    global _last_stats_time
    now = time.time()

    # Clean up expired entries
    for cache_type, cache in _CACHES.items():
        removed = cache.cleanup()
        if removed > 0:
            logger.debug(f"Removed {removed} expired entries from {cache_type} cache")

    # Log cache stats periodically
    if now - _last_stats_time > CACHE_STATS_INTERVAL:
        _last_stats_time = now
        _log_cache_stats()

def _log_cache_stats() -> None:
    """Log statistics about cache usage."""
    stats = get_cache_stats()

    # Log overall stats
    total_size = sum(cache_stats["size"] for cache_stats in stats.values())
    total_hits = sum(cache_stats["hits"] for cache_stats in stats.values())
    total_misses = sum(cache_stats["misses"] for cache_stats in stats.values())

    if total_hits + total_misses > 0:
        hit_rate = total_hits / (total_hits + total_misses) * 100
        logger.info(f"Cache stats: {total_size} items, {hit_rate:.1f}% hit rate")

    # Log per-cache stats
    for cache_type, cache_stats in stats.items():
        logger.debug(f"{cache_type.capitalize()} cache: {cache_stats['size']} items, "
                    f"{cache_stats['hit_rate_percent']}% hit rate")

def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all caches.

    Returns:
        Dictionary mapping cache types to their statistics
    """
    return {
        cache_type: cache.get_stats()
        for cache_type, cache in _CACHES.items()
    }

def configure_cache(
    cache_type: str,
    max_size: Optional[int] = None,
    default_ttl: Optional[int] = None
) -> None:
    """
    Configure cache parameters.

    Args:
        cache_type: Type of cache to configure
        max_size: Maximum cache size
        default_ttl: Default TTL in seconds
    """
    if cache_type not in _CACHES:
        logger.warning(f"Cannot configure unknown cache type: {cache_type}")
        return

    cache = _CACHES[cache_type]

    if max_size is not None:
        cache.max_size = max_size

    if default_ttl is not None:
        cache.default_ttl = default_ttl

    logger.info(f"Configured {cache_type} cache: max_size={cache.max_size}, "
               f"default_ttl={cache.default_ttl}s")

def cached(cache_type: str, ttl: Optional[int] = None):
    """
    Enhanced decorator for caching function results.

    Args:
        cache_type: Type of cache to use
        ttl: Optional custom TTL in seconds

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
                else:
                    # For complex types, use their string representation
                    key_parts.append(str(hash(str(arg))))

            # Add keyword arguments (sorted for consistency)
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                if isinstance(v, (str, int, float, bool, type(None))):
                    key_parts.append(f"{k}={v}")
                else:
                    # For complex types, use their string representation
                    key_parts.append(f"{k}={hash(str(v))}")

            # Create a combined key
            key = "_".join(key_parts)

            # Check cache
            cached_value = cache_get(cache_type, key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_set(cache_type, key, result, ttl)

            return result

        return wrapper

    return decorator

# Export commonly used LRU cache decorators from functools for direct usage
video_metadata_cache = functools.lru_cache(maxsize=100)
search_results_cache = functools.lru_cache(maxsize=50)
