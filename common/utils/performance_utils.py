"""
Performance Monitoring and Optimization Utilities for Lecture Video Content Indexer.
Provides tools for measuring performance, detecting bottlenecks, and optimizing resource usage.
"""

import time
import logging
import functools
import threading
import os
import resource
import gc
import sys
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime
from contextlib import contextmanager
import json
import psutil

# Configure logging
logger = logging.getLogger(__name__)

# Global performance metrics storage
_performance_metrics = {}
_metrics_lock = threading.RLock()
_enabled = True


def is_enabled() -> bool:
    """
    Check if performance monitoring is enabled.

    Returns:
        True if enabled, False otherwise
    """
    return _enabled


def enable():
    """Enable performance monitoring."""
    global _enabled
    _enabled = True
    logger.info("Performance monitoring enabled")


def disable():
    """Disable performance monitoring to reduce overhead."""
    global _enabled
    _enabled = False
    logger.info("Performance monitoring disabled")


@contextmanager
def measure_time(name: str, threshold_ms: Optional[int] = None):
    """
    Context manager for measuring execution time.

    Args:
        name: Operation name
        threshold_ms: Log warning if execution time exceeds threshold

    Yields:
        None
    """
    if not _enabled:
        yield
        return

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        with _metrics_lock:
            if name not in _performance_metrics:
                _performance_metrics[name] = {
                    'count': 0,
                    'total_ms': 0,
                    'min_ms': float('inf'),
                    'max_ms': 0,
                    'last_ms': 0
                }

            metrics = _performance_metrics[name]
            metrics['count'] += 1
            metrics['total_ms'] += duration_ms
            metrics['min_ms'] = min(metrics['min_ms'], duration_ms)
            metrics['max_ms'] = max(metrics['max_ms'], duration_ms)
            metrics['last_ms'] = duration_ms

        # Log slow operations
        if threshold_ms and duration_ms > threshold_ms:
            logger.warning(f"Slow operation: {name} took {duration_ms:.2f}ms (threshold: {threshold_ms}ms)")


def time_function(threshold_ms: Optional[int] = None):
    """
    Decorator for measuring function execution time.

    Args:
        threshold_ms: Log warning if execution time exceeds threshold

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"function:{func.__name__}"
            with measure_time(operation_name, threshold_ms):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def async_time_function(threshold_ms: Optional[int] = None):
    """
    Decorator for measuring async function execution time.

    Args:
        threshold_ms: Log warning if execution time exceeds threshold

    Returns:
        Decorated async function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            operation_name = f"async_function:{func.__name__}"
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                with _metrics_lock:
                    if operation_name not in _performance_metrics:
                        _performance_metrics[operation_name] = {
                            'count': 0,
                            'total_ms': 0,
                            'min_ms': float('inf'),
                            'max_ms': 0,
                            'last_ms': 0
                        }

                    metrics = _performance_metrics[operation_name]
                    metrics['count'] += 1
                    metrics['total_ms'] += duration_ms
                    metrics['min_ms'] = min(metrics['min_ms'], duration_ms)
                    metrics['max_ms'] = max(metrics['max_ms'], duration_ms)
                    metrics['last_ms'] = duration_ms

                # Log slow operations
                if threshold_ms and duration_ms > threshold_ms:
                    logger.warning(f"Slow operation: {operation_name} took {duration_ms:.2f}ms (threshold: {threshold_ms}ms)")

        return wrapper
    return decorator


@contextmanager
def measure_memory(name: str, threshold_mb: Optional[int] = None):
    """
    Context manager for measuring memory usage.

    Args:
        name: Operation name
        threshold_mb: Log warning if memory usage exceeds threshold

    Yields:
        None
    """
    if not _enabled:
        yield
        return

    # Get current process
    process = psutil.Process(os.getpid())

    # Measure memory before
    gc.collect()  # Force garbage collection
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB

    try:
        yield
    finally:
        # Measure memory after
        gc.collect()  # Force garbage collection
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = memory_after - memory_before

        operation_name = f"memory:{name}"
        with _metrics_lock:
            if operation_name not in _performance_metrics:
                _performance_metrics[operation_name] = {
                    'count': 0,
                    'total_mb': 0,
                    'max_mb': 0,
                    'last_mb': 0
                }

            metrics = _performance_metrics[operation_name]
            metrics['count'] += 1
            metrics['total_mb'] += memory_diff
            metrics['max_mb'] = max(metrics['max_mb'], memory_diff)
            metrics['last_mb'] = memory_diff

        # Log high memory usage
        if threshold_mb and memory_diff > threshold_mb:
            logger.warning(f"High memory usage: {name} used {memory_diff:.2f}MB (threshold: {threshold_mb}MB)")


def memory_function(threshold_mb: Optional[int] = None):
    """
    Decorator for measuring function memory usage.

    Args:
        threshold_mb: Log warning if memory usage exceeds threshold

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"function:{func.__name__}"
            with measure_memory(operation_name, threshold_mb):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def start_memory_profiling():
    """
    Start detailed memory profiling with tracemalloc.
    """
    if not _enabled:
        return

    if not tracemalloc.is_tracing():
        tracemalloc.start()
        logger.info("Memory profiling started")


def stop_memory_profiling():
    """
    Stop detailed memory profiling with tracemalloc.
    """
    if tracemalloc.is_tracing():
        tracemalloc.stop()
        logger.info("Memory profiling stopped")


def get_memory_snapshot():
    """
    Get current memory usage snapshot.

    Returns:
        Dictionary with memory usage statistics
    """
    if not _enabled or not tracemalloc.is_tracing():
        return {'error': 'Memory profiling not enabled'}

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    # Prepare statistics
    stats = []
    for stat in top_stats[:20]:  # Top 20 memory consumers
        stats.append({
            'file': str(stat.traceback.frame.filename),
            'line': stat.traceback.frame.lineno,
            'size': stat.size / (1024 * 1024),  # MB
            'count': stat.count
        })

    return {
        'top_memory_consumers': stats,
        'total_traced_memory': sum(stat.size for stat in top_stats) / (1024 * 1024)  # MB
    }


def get_performance_metrics():
    """
    Get all collected performance metrics.

    Returns:
        Dictionary with all performance metrics
    """
    with _metrics_lock:
        # Create a copy to avoid modification during iteration
        metrics_copy = json.loads(json.dumps(_performance_metrics))

    # Add system metrics
    system_metrics = get_system_metrics()
    metrics_copy['system'] = system_metrics

    # Add memory profiling data if available
    if tracemalloc.is_tracing():
        memory_snapshot = get_memory_snapshot()
        metrics_copy['memory_profile'] = memory_snapshot

    return metrics_copy


def get_system_metrics():
    """
    Get current system metrics (CPU, memory, etc.).

    Returns:
        Dictionary with system metrics
    """
    # Get current process
    process = psutil.Process(os.getpid())

    # Get system-wide information
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    # Get process-specific information
    process_cpu = process.cpu_percent(interval=0.1)
    process_memory = process.memory_info()
    process_threads = process.num_threads()

    # Get open files count (may fail on some platforms)
    try:
        open_files = len(process.open_files())
    except:
        open_files = -1

    # Get open connections count (may fail on some platforms)
    try:
        connections = len(process.connections())
    except:
        connections = -1

    return {
        'timestamp': datetime.now().isoformat(),
        'system_cpu_percent': cpu_percent,
        'system_memory_percent': memory.percent,
        'system_memory_available_mb': memory.available / (1024 * 1024),
        'process_cpu_percent': process_cpu,
        'process_memory_rss_mb': process_memory.rss / (1024 * 1024),
        'process_memory_vms_mb': process_memory.vms / (1024 * 1024),
        'process_threads': process_threads,
        'process_open_files': open_files,
        'process_connections': connections
    }


def reset_metrics():
    """
    Reset all collected performance metrics.
    """
    with _metrics_lock:
        _performance_metrics.clear()
    logger.info("Performance metrics reset")


def log_performance_summary(interval_seconds: int = 300):
    """
    Start a background thread to periodically log performance summaries.

    Args:
        interval_seconds: Interval between logs in seconds
    """
    def logging_task():
        while True:
            try:
                time.sleep(interval_seconds)
                if not _enabled:
                    continue

                metrics = get_performance_metrics()

                # Log system metrics
                sys_metrics = metrics.get('system', {})
                logger.info(
                    f"System metrics: CPU {sys_metrics.get('system_cpu_percent', 0):.1f}%, "
                    f"Memory {sys_metrics.get('system_memory_percent', 0):.1f}%, "
                    f"Process RSS {sys_metrics.get('process_memory_rss_mb', 0):.1f}MB"
                )

                # Log slow operations
                slow_ops = []
                for name, stats in metrics.items():
                    if name == 'system' or name.startswith('memory:'):
                        continue

                    if isinstance(stats, dict) and 'avg_ms' not in stats and 'count' in stats and 'total_ms' in stats:
                        # Calculate average time
                        stats['avg_ms'] = stats['total_ms'] / stats['count'] if stats['count'] > 0 else 0

                    if isinstance(stats, dict) and stats.get('avg_ms', 0) > 100:  # Over 100ms average
                        slow_ops.append((name, stats))

                if slow_ops:
                    slow_ops.sort(key=lambda x: x[1]['avg_ms'], reverse=True)
                    top_slow = slow_ops[:5]  # Top 5 slowest

                    logger.warning("Slow operations detected:")
                    for name, stats in top_slow:
                        logger.warning(
                            f"  {name}: avg={stats['avg_ms']:.1f}ms, "
                            f"calls={stats['count']}, "
                            f"max={stats['max_ms']:.1f}ms"
                        )

                # Log high memory usage operations
                high_mem_ops = []
                for name, stats in metrics.items():
                    if not name.startswith('memory:'):
                        continue

                    if stats.get('max_mb', 0) > 10:  # Over 10MB
                        high_mem_ops.append((name, stats))

                if high_mem_ops:
                    high_mem_ops.sort(key=lambda x: x[1]['max_mb'], reverse=True)
                    top_mem = high_mem_ops[:5]  # Top 5 memory users

                    logger.warning("High memory usage operations detected:")
                    for name, stats in top_mem:
                        logger.warning(
                            f"  {name}: max={stats['max_mb']:.1f}MB, "
                            f"calls={stats['count']}"
                        )

            except Exception as e:
                logger.error(f"Error in performance logging task: {e}")

    # Start logging thread
    logging_thread = threading.Thread(target=logging_task, daemon=True)
    logging_thread.start()
    logger.info(f"Performance summary logging started (interval: {interval_seconds}s)")


class ResourceLimit:
    """
    Set and manage resource limits for the process.
    """

    @staticmethod
    def set_memory_limit(max_memory_mb: int):
        """
        Set maximum memory limit for the process.

        Args:
            max_memory_mb: Maximum memory in megabytes
        """
        try:
            # Convert to bytes
            max_memory_bytes = max_memory_mb * 1024 * 1024

            # Set soft limit
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            logger.info(f"Set memory limit to {max_memory_mb}MB")
        except Exception as e:
            logger.error(f"Failed to set memory limit: {e}")

    @staticmethod
    def set_cpu_limit(max_cpu_seconds: int):
        """
        Set maximum CPU time limit for the process.

        Args:
            max_cpu_seconds: Maximum CPU time in seconds
        """
        try:
            # Set soft and hard limits
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds))
            logger.info(f"Set CPU time limit to {max_cpu_seconds}s")
        except Exception as e:
            logger.error(f"Failed to set CPU time limit: {e}")

    @staticmethod
    def get_current_limits():
        """
        Get current resource limits.

        Returns:
            Dictionary with current resource limits
        """
        limits = {}

        try:
            # Get memory limit
            mem_limit = resource.getrlimit(resource.RLIMIT_AS)
            if mem_limit[0] != resource.RLIM_INFINITY:
                limits['memory_mb'] = mem_limit[0] / (1024 * 1024)
            else:
                limits['memory_mb'] = "unlimited"

            # Get CPU time limit
            cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
            if cpu_limit[0] != resource.RLIM_INFINITY:
                limits['cpu_seconds'] = cpu_limit[0]
            else:
                limits['cpu_seconds'] = "unlimited"

            # Get file descriptor limit
            fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            limits['max_file_descriptors'] = fd_limit[0]

        except Exception as e:
            logger.error(f"Failed to get resource limits: {e}")
            limits['error'] = str(e)

        return limits


class PerformanceOptimizer:
    """
    Automatic performance optimizer that dynamically adjusts settings
    based on observed system metrics.
    """

    def __init__(
        self,
        target_memory_percent: float = 70.0,
        target_cpu_percent: float = 80.0,
        check_interval: int = 60,
        aggressive: bool = False
    ):
        """
        Initialize the performance optimizer.

        Args:
            target_memory_percent: Target memory usage percentage
            target_cpu_percent: Target CPU usage percentage
            check_interval: Interval between checks in seconds
            aggressive: Whether to use aggressive optimization
        """
        self.target_memory_percent = target_memory_percent
        self.target_cpu_percent = target_cpu_percent
        self.check_interval = check_interval
        self.aggressive = aggressive
        self.running = False
        self.thread = None

    def start(self):
        """
        Start the optimizer in a background thread.
        """
        if self.running:
            return

        self.running = True

        def optimizer_task():
            while self.running:
                try:
                    time.sleep(self.check_interval)
                    self._optimize_performance()
                except Exception as e:
                    logger.error(f"Error in performance optimizer: {e}")

        self.thread = threading.Thread(target=optimizer_task, daemon=True)
        self.thread.start()
        logger.info(f"Performance optimizer started (interval: {self.check_interval}s)")

    def stop(self):
        """
        Stop the optimizer.
        """
        self.running = False
        logger.info("Performance optimizer stopped")

    def _optimize_performance(self):
        """
        Check current performance metrics and apply optimizations.
        """
        if not _enabled:
            return

        # Get current metrics
        metrics = get_system_metrics()

        # Check memory usage
        system_memory_percent = metrics.get('system_memory_percent', 0)
        process_memory_mb = metrics.get('process_memory_rss_mb', 0)

        # Check CPU usage
        system_cpu_percent = metrics.get('system_cpu_percent', 0)
        process_cpu_percent = metrics.get('process_cpu_percent', 0)

        # Log current status
        logger.debug(
            f"Performance status: Memory {system_memory_percent:.1f}% ({process_memory_mb:.1f}MB), "
            f"CPU {system_cpu_percent:.1f}% (process: {process_cpu_percent:.1f}%)"
        )

        # Check if we need to optimize
        memory_pressure = system_memory_percent > self.target_memory_percent
        cpu_pressure = system_cpu_percent > self.target_cpu_percent

        if memory_pressure:
            logger.warning(f"Memory pressure detected: {system_memory_percent:.1f}% used")
            self._optimize_memory(system_memory_percent, process_memory_mb)

        if cpu_pressure:
            logger.warning(f"CPU pressure detected: {system_cpu_percent:.1f}% used")
            self._optimize_cpu(system_cpu_percent, process_cpu_percent)

    def _optimize_memory(self, system_memory_percent: float, process_memory_mb: float):
        """
        Apply memory optimizations.

        Args:
            system_memory_percent: System memory usage percentage
            process_memory_mb: Process memory usage in MB
        """
        # Force garbage collection
        gc.collect()

        # Get memory after garbage collection
        process = psutil.Process(os.getpid())
        process_memory_after_gc = process.memory_info().rss / (1024 * 1024)
        memory_freed = process_memory_mb - process_memory_after_gc

        if memory_freed > 5:  # Freed more than 5MB
            logger.info(f"Garbage collection freed {memory_freed:.1f}MB")

        # Clear all metrics if in aggressive mode
        if self.aggressive and system_memory_percent > 90:
            reset_metrics()
            logger.warning("Aggressive memory optimization: cleared all metrics")

        # If using tracemalloc and memory is very high, stop it
        if tracemalloc.is_tracing() and system_memory_percent > 95:
            stop_memory_profiling()
            logger.warning("Stopped memory profiling due to high memory usage")

    def _optimize_cpu(self, system_cpu_percent: float, process_cpu_percent: float):
        """
        Apply CPU optimizations.

        Args:
            system_cpu_percent: System CPU usage percentage
            process_cpu_percent: Process CPU usage percentage
        """
        # If our process is consuming too much CPU, take action
        if process_cpu_percent > 50:  # More than 50% of a core
            # Disable performance monitoring temporarily if in aggressive mode
            if self.aggressive and system_cpu_percent > 95:
                disable()
                logger.warning("Disabled performance monitoring due to high CPU usage")
                # Re-enable after a delay
                threading.Timer(30, enable).start()


def inspect_object_size(obj) -> Dict[str, Any]:
    """
    Inspect the memory usage of a Python object.

    Args:
        obj: Object to inspect

    Returns:
        Dictionary with size information
    """
    import sys
    import types
    from pympler import asizeof

    # Basic size
    size = sys.getsizeof(obj)

    # Deep size
    try:
        deep_size = asizeof.asizeof(obj)
    except:
        deep_size = -1

    # Count number of objects
    if isinstance(obj, (list, tuple, set, frozenset)):
        count = len(obj)
        item_type = str(type(obj).__name__)
    elif isinstance(obj, dict):
        count = len(obj)
        item_type = "dict"
    elif isinstance(obj, str):
        count = len(obj)
        item_type = "str"
    else:
        count = 1
        item_type = str(type(obj).__name__)

    # Check if object has custom attributes
    attributes = {}
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):  # Skip private attributes
                try:
                    attributes[key] = sys.getsizeof(value)
                except:
                    attributes[key] = -1

    # Report
    return {
        "type": str(type(obj)),
        "size_bytes": size,
        "deep_size_bytes": deep_size,
        "item_count": count,
        "item_type": item_type,
        "attributes": attributes
    }


# Initialize performance monitoring
def initialize(
    enable_monitoring: bool = True,
    log_interval: int = 300,
    memory_profiling: bool = False,
    auto_optimize: bool = False
):
    """
    Initialize performance monitoring system.

    Args:
        enable_monitoring: Whether to enable monitoring
        log_interval: Interval between summary logs in seconds
        memory_profiling: Whether to enable detailed memory profiling
        auto_optimize: Whether to enable automatic performance optimization
    """
    global _enabled
    _enabled = enable_monitoring

    if enable_monitoring:
        # Start logging thread
        log_performance_summary(log_interval)

        # Start memory profiling if requested
        if memory_profiling:
            start_memory_profiling()

        # Start optimizer if requested
        if auto_optimize:
            optimizer = PerformanceOptimizer()
            optimizer.start()

        logger.info(
            f"Performance monitoring initialized (logging: {log_interval}s, "
            f"memory profiling: {memory_profiling}, auto optimize: {auto_optimize})"
        )
    else:
        logger.info("Performance monitoring disabled")
