"""
Cache Manager module for the Lecture Video Content Indexer.
Provides a memory-efficient caching system with TTL support and automatic pruning.
"""

import time
import logging
import threading
import json
from typing import Dict, Any, Optional, Callable, List, Tuple
from functools import wraps
from datetime import datetime, timedelta
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

class CacheEntry:
    """
    Cache entry container with expiration tracking.
    """
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        """
        Initialize a cache entry.

        Args:
            key: Cache key
            value: Cached value
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0

    def is_expired(self) -> bool:
        """
        Check if the cache entry has expired.

        Returns:
            True if expired, False otherwise
        """
        return (time.time() - self.created_at) > self.ttl

    def access(self) -> None:
        """
        Update last accessed time and access count.
        """
        self.last_accessed = time.time()
        self.access_count += 1

    def __repr__(self) -> str:
        """String representation of cache entry."""
        age = time.time() - self.created_at
        return f"CacheEntry(key={self.key}, age={age:.1f}s, ttl={self.ttl}s, access_count={self.access_count})"


class CacheManager:
    """
    Memory-efficient caching system with TTL support and automatic pruning.
    Uses LRU (Least Recently Used) and LFU (Least Frequently Used) algorithms
    to manage cache eviction.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        cleanup_interval: int = 300,
        strategy: str = 'lru'
    ):
        """
        Initialize the cache manager.

        Args:
            max_size: Maximum number of entries in the cache
            default_ttl: Default time to live in seconds
            cleanup_interval: Automatic cleanup interval in seconds
            strategy: Cache eviction strategy ('lru', 'lfu', 'hybrid')
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.strategy = strategy
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.memory_limit_mb = 200  # Default memory limit (200MB)

        # Dictionary to store computed cache key for each function call args/kwargs
        self.key_cache = {}
        self.key_cache_lock = threading.RLock()

        # Start cleanup thread
        self._start_cleanup_thread()

        logger.info(f"CacheManager initialized with strategy={strategy}, max_size={max_size}, ttl={default_ttl}s")

    def _start_cleanup_thread(self):
        """Start the automatic cleanup thread."""
        def cleanup_task():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup()
                except Exception as e:
                    logger.error(f"Error in cache cleanup thread: {e}")

        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: use instance default)
        """
        if ttl is None:
            ttl = self.default_ttl

        # Skip caching None values
        if value is None:
            return

        # Check if value is too large (rough estimation)
        try:
            if self._estimate_size(value) > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"Value for key '{key}' is too large for caching (>10MB)")
                return
        except Exception:
            # If we can't estimate size, still try to cache
            pass

        with self.lock:
            # Check if we need to make room
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_entries()

            # Add or update entry
            entry = CacheEntry(key, value, ttl)
            self.cache[key] = entry

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                self.misses += 1
                return None

            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None

            # Update access stats
            entry.access()
            self.hits += 1

            return entry.value

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def flush(self) -> None:
        """
        Clear the entire cache.
        """
        with self.lock:
            self.cache.clear()
        logger.info("Cache flushed")

    def cleanup(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        removed = 0

        with self.lock:
            # Find expired entries
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]

            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
                removed += 1

            # Check if we're approaching memory limit
            if self._estimate_cache_size_mb() > self.memory_limit_mb * 0.8:
                logger.warning(f"Cache approaching memory limit ({self._estimate_cache_size_mb():.1f}MB)")

                # Evict additional entries to reduce memory usage
                extra_evictions = self._evict_entries(int(len(self.cache) * 0.1))  # Evict 10% of entries
                removed += extra_evictions

        if removed > 0:
            logger.info(f"Removed {removed} expired/evicted entries from cache")

        return removed

    def _evict_entries(self, count: int = 1) -> int:
        """
        Evict entries according to the configured strategy.

        Args:
            count: Number of entries to evict

        Returns:
            Number of entries actually evicted
        """
        evicted = 0

        if not self.cache:
            return 0

        if self.strategy == 'lru':
            # Least Recently Used strategy
            entries = sorted(self.cache.values(), key=lambda e: e.last_accessed)
        elif self.strategy == 'lfu':
            # Least Frequently Used strategy
            entries = sorted(self.cache.values(), key=lambda e: e.access_count)
        else:
            # Hybrid strategy - combine recency and frequency
            entries = sorted(self.cache.values(),
                        key=lambda e: (e.access_count * 0.4) +
                                      (time.time() - e.last_accessed) * 0.6)

        # Remove the entries with lowest score
        for entry in entries[:count]:
            if entry.key in self.cache:
                del self.cache[entry.key]
                evicted += 1

        return evicted

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0

            # Count expired entries without removing them
            expired = sum(1 for entry in self.cache.values() if entry.is_expired())

            # Calculate memory usage (approximate)
            memory_usage = self._estimate_cache_size_mb()

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "expired_entries": expired,
                "estimated_memory_mb": memory_usage,
                "memory_limit_mb": self.memory_limit_mb,
                "strategy": self.strategy
            }

    def set_memory_limit(self, limit_mb: int) -> None:
        """
        Set maximum memory usage for the cache.

        Args:
            limit_mb: Memory limit in megabytes
        """
        self.memory_limit_mb = limit_mb
        logger.info(f"Cache memory limit set to {limit_mb}MB")

        # Run cleanup if we're already over the new limit
        if self._estimate_cache_size_mb() > limit_mb:
            self.cleanup()

    def _estimate_cache_size_mb(self) -> float:
        """
        Estimate current cache size in megabytes.

        Returns:
            Estimated cache size in MB
        """
        total_size = 0

        # Sample a subset of entries to estimate average size
        if len(self.cache) > 100:
            # Sample 10% of entries
            sample_size = max(int(len(self.cache) * 0.1), 10)
            sample_keys = list(self.cache.keys())[:sample_size]

            # Estimate size of sampled entries
            sample_size_bytes = 0
            for key in sample_keys:
                entry = self.cache[key]
                sample_size_bytes += self._estimate_size(entry.key) + self._estimate_size(entry.value)

            # Extrapolate to full cache
            avg_entry_size = sample_size_bytes / len(sample_keys)
            total_size = avg_entry_size * len(self.cache)
        else:
            # For small caches, measure all entries
            for entry in self.cache.values():
                total_size += self._estimate_size(entry.key) + self._estimate_size(entry.value)

        # Convert to MB
        return total_size / (1024 * 1024)

    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate memory size of an object in bytes.
        This is a rough approximation.

        Args:
            obj: Object to measure

        Returns:
            Estimated size in bytes
        """
        if obj is None:
            return 8

        if isinstance(obj, (int, float, bool)):
            return 8

        if isinstance(obj, str):
            return len(obj) * 2 + 24  # Unicode characters + overhead

        if isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj) + 64

        if isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items()) + 64

        # For complex objects, use a rough estimate based on its JSON representation
        try:
            return len(json.dumps(obj)) * 2
        except:
            # If we can't serialize, use a conservative default
            return 1024  # 1KB default for unknown objects

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from function arguments.

        Args:
            prefix: Key prefix
            *args: Function args
            **kwargs: Function kwargs

        Returns:
            Generated cache key
        """
        # Create a key based on the arguments
        key_parts = [prefix]

        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                key_parts.append(str(arg))
            else:
                # For complex objects, use repr or try json
                try:
                    key_parts.append(json.dumps(arg, sort_keys=True))
                except:
                    key_parts.append(repr(arg))

        # Add keyword arguments (sorted for consistency)
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, (str, int, float, bool, type(None))):
                key_parts.append(f"{k}={v}")
            else:
                # For complex objects, use repr or try json
                try:
                    key_parts.append(f"{k}={json.dumps(v, sort_keys=True)}")
                except:
                    key_parts.append(f"{k}={repr(v)}")

        # Create a hash to keep key size reasonable
        key_string = "_".join(key_parts)
        hashed_key = hashlib.md5(key_string.encode()).hexdigest()

        # Return prefix + hashed args for readability and debugging
        return f"{prefix}_{hashed_key}"

    def cached(self, ttl: Optional[int] = None, prefix: Optional[str] = None):
        """
        Decorator for caching function results.

        Args:
            ttl: Time to live in seconds (default: use instance default)
            prefix: Key prefix (default: function name)

        Returns:
            Decorated function
        """
        def decorator(func):
            if ttl is None:
                _ttl = self.default_ttl
            else:
                _ttl = ttl

            # Use function name as prefix if not specified
            _prefix = prefix or func.__name__

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get key
                key = self._generate_key(_prefix, *args, **kwargs)

                # Check cache
                result = self.get(key)
                if result is not None:
                    return result

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                self.set(key, result, _ttl)

                return result

            return wrapper

        return decorator

    def region(self, name: str) -> 'CacheRegion':
        """
        Create a cache region with a specific key prefix.

        Args:
            name: Region name (used as key prefix)

        Returns:
            Cache region
        """
        return CacheRegion(self, name)


class CacheRegion:
    """
    A sub-region of the cache with a specific key prefix.
    Allows for easier namespace management and bulk operations.
    """

    def __init__(self, cache: CacheManager, name: str):
        """
        Initialize a cache region.

        Args:
            cache: Parent cache manager
            name: Region name (used as key prefix)
        """
        self.cache = cache
        self.name = name
        self.key_prefix = f"region:{name}:"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache region.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: use parent default)
        """
        full_key = self.key_prefix + key
        self.cache.set(full_key, value, ttl)

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache region.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        full_key = self.key_prefix + key
        return self.cache.get(full_key)

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache region.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        full_key = self.key_prefix + key
        return self.cache.delete(full_key)

    def flush(self) -> int:
        """
        Clear all entries in this cache region.

        Returns:
            Number of entries removed
        """
        removed = 0
        with self.cache.lock:
            keys_to_remove = []

            # Find keys that belong to this region
            for key in self.cache.cache.keys():
                if key.startswith(self.key_prefix):
                    keys_to_remove.append(key)

            # Remove them
            for key in keys_to_remove:
                del self.cache.cache[key]
                removed += 1

        logger.info(f"Flushed {removed} entries from cache region '{self.name}'")
        return removed

    def cached(self, key_func: Optional[Callable] = None, ttl: Optional[int] = None):
        """
        Decorator for caching function results in this region.

        Args:
            key_func: Function to generate the cache key (default: use args)
            ttl: Time to live in seconds (default: use parent default)

        Returns:
            Decorated function
        """
        def decorator(func):
            if ttl is None:
                _ttl = self.cache.default_ttl
            else:
                _ttl = ttl

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    # Use a hash of the arguments as the key
                    key = self.cache._generate_key(func.__name__, *args, **kwargs)

                # Add region prefix
                full_key = self.key_prefix + key

                # Check cache
                result = self.cache.get(full_key)
                if result is not None:
                    return result

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                self.cache.set(full_key, result, _ttl)

                return result

            return wrapper

        return decorator

    def keys(self) -> List[str]:
        """
        Get all keys in this cache region.

        Returns:
            List of keys in this region (without the region prefix)
        """
        with self.cache.lock:
            return [
                k[len(self.key_prefix):] for k in self.cache.cache.keys()
                if k.startswith(self.key_prefix)
            ]

    def items(self) -> List[Tuple[str, Any]]:
        """
        Get all items in this cache region.

        Returns:
            List of (key, value) tuples in this region (keys without the region prefix)
        """
        with self.cache.lock:
            return [
                (k[len(self.key_prefix):], v.value)
                for k, v in self.cache.cache.items()
                if k.startswith(self.key_prefix) and not v.is_expired()
            ]


# Create a default cache manager instance
default_cache = CacheManager()

def cached(ttl: Optional[int] = None, prefix: Optional[str] = None):
    """
    Decorator for caching function results using the default cache manager.

    Args:
        ttl: Time to live in seconds (default: use default cache's default)
        prefix: Key prefix (default: function name)

    Returns:
        Decorated function
    """
    return default_cache.cached(ttl, prefix)
