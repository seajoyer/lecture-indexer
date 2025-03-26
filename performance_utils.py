"""
Simple timing utilities for the Lecture Video Content Indexer.
Replaces the complex performance_utils.py with basic timing decorators and context managers.
"""

import time
import logging
import functools
from contextlib import contextmanager
from typing import Callable, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Global flag to enable/disable timing
timing_enabled = True

def enable_timing():
    """Enable operation timing."""
    global timing_enabled
    timing_enabled = True
    logger.info("Timing enabled")

def disable_timing():
    """Disable operation timing."""
    global timing_enabled
    timing_enabled = False
    logger.info("Timing disabled")

@contextmanager
def time_operation(name: str, log_threshold_ms: Optional[int] = None):
    """
    Context manager for timing operations.

    Args:
        name: Operation name
        log_threshold_ms: Threshold in milliseconds for logging warnings

    Yields:
        None
    """
    if not timing_enabled:
        yield
        return

    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000

        # Always log at debug level
        logger.debug(f"Operation {name} took {duration_ms:.2f}ms")

        # Log warning if threshold is exceeded
        if log_threshold_ms and duration_ms > log_threshold_ms:
            logger.warning(f"Slow operation: {name} took {duration_ms:.2f}ms (threshold: {log_threshold_ms}ms)")

def time_function(log_threshold_ms: Optional[int] = None):
    """
    Decorator for timing function execution.

    Args:
        log_threshold_ms: Threshold in milliseconds for logging warnings

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with time_operation(func.__name__, log_threshold_ms):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class Timer:
    """Simple timer for manual timing."""

    def __init__(self, name: str):
        """
        Initialize timer.

        Args:
            name: Timer name
        """
        self.name = name
        self.start_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def stop(self):
        """Stop the timer and log elapsed time."""
        if not self.start_time:
            logger.warning(f"Timer {self.name} stopped without being started")
            return 0

        elapsed_time = time.time() - self.start_time
        elapsed_ms = elapsed_time * 1000

        logger.debug(f"Timer {self.name} elapsed time: {elapsed_ms:.2f}ms")
        return elapsed_ms

    def __enter__(self):
        """Start timer when used as context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer when exiting context manager."""
        self.stop()
