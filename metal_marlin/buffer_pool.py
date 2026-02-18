"""Metal buffer pool with size-bucketed reuse.

Provides a thread-safe pool for reusable Metal buffers organized by size buckets
(powers of 2). Buffers are pre-allocated and reused to minimize allocation overhead.

Usage:
    pool = MetalBufferPool(device, initial_sizes=[1024, 2048, 4096])
    
    # Acquire a buffer (auto-creates if needed)
    buf = pool.acquire(1500)  # Returns buffer >= 1500 bytes
    
    # Use buffer...
    
    # Release back to pool
    pool.release(buf)
    
    # Get statistics
    print(pool.stats())
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import Metal


class _Buffer:
    """Internal wrapper for pooled buffers."""

    __slots__ = ("buffer", "size", "acquire_count")

    def __init__(self, buffer: Any, size: int) -> None:
        self.buffer = buffer
        self.size = size
        self.acquire_count = 0


class MetalBufferPool:
    """Thread-safe Metal buffer pool with power-of-2 size buckets.

    Buffers are organized into buckets by size (powers of 2):
    - Bucket 0: 1-128 bytes
    - Bucket 1: 129-256 bytes
    - Bucket 2: 257-512 bytes
    - etc.

    The pool auto-grows when a bucket is exhausted, creating new buffers
    on demand. Released buffers are returned to their appropriate bucket
    for reuse.

    Thread-safety: All operations are protected by a single lock.

    Example:
        device = Metal.MTLCreateSystemDefaultDevice()
        pool = MetalBufferPool(device, initial_sizes=[1024, 2048])

        buf = pool.acquire(1500)  # Gets buffer from 2048 bucket
        # ... use buffer ...
        pool.release(buf)  # Returns to pool
    """

    # Minimum buffer size (128 bytes for cache line alignment)
    MIN_BUFFER_SIZE = 128

    def __init__(
        self,
        device: Any,
        initial_sizes: list[int] | None = None,
    ) -> None:
        """Initialize the buffer pool.

        Args:
            device: Metal device for buffer allocation.
            initial_sizes: List of sizes to pre-allocate. Each size will
                have one buffer created in its corresponding bucket.
                If None, no buffers are pre-allocated.
        """
        self._device = device
        self._lock = threading.Lock()

        # Buckets: size_bucket -> deque of _Buffer
        # Size bucket is the power of 2 ceiling of buffer size
        self._buckets: dict[int, deque[_Buffer]] = {}

        # Track all buffers for stats
        self._total_buffers = 0
        self._in_use_buffers = 0
        self._total_allocations = 0
        self._total_releases = 0
        self._bucket_misses: dict[int, int] = {}  # Bucket -> miss count

        # Pre-allocate initial buffers
        if initial_sizes:
            for size in initial_sizes:
                self._ensure_bucket_exists(self._size_to_bucket(size))

    def _size_to_bucket(self, size: int) -> int:
        """Convert size to bucket index (power of 2 ceiling).

        Args:
            size: Requested buffer size in bytes.

        Returns:
            Bucket index (0 = 128 bytes, 1 = 256 bytes, etc.)
        """
        if size <= self.MIN_BUFFER_SIZE:
            return 0
        # Calculate ceil(log2(size)) - 6, since 2^7 = 128
        import math

        return max(0, int(math.ceil(math.log2(size))) - 7)

    def _bucket_to_size(self, bucket: int) -> int:
        """Convert bucket index to actual buffer size.

        Args:
            bucket: Bucket index.

        Returns:
            Buffer size for this bucket (power of 2).
        """
        return self.MIN_BUFFER_SIZE << bucket

    def _create_buffer(self, bucket: int) -> _Buffer:
        """Create a new buffer for the given bucket.

        Args:
            bucket: Bucket index.

        Returns:
            New _Buffer instance.
        """
        size = self._bucket_to_size(bucket)

        # Allocate with shared storage for CPU/GPU access
        buffer = self._device.newBufferWithLength_options_(
            size, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of {size} bytes")

        return _Buffer(buffer=buffer, size=size)

    def _ensure_bucket_exists(self, bucket: int) -> None:
        """Ensure a bucket exists, creating it if necessary.

        Args:
            bucket: Bucket index to ensure exists.
        """
        if bucket not in self._buckets:
            self._buckets[bucket] = deque()
            self._bucket_misses[bucket] = 0

    def acquire(self, size: int) -> Any:
        """Get a buffer of at least the requested size.

        If a buffer is available in the appropriate bucket, it is returned.
        If the bucket is empty, a new buffer is created (auto-grow).

        Args:
            size: Minimum buffer size in bytes.

        Returns:
            MTLBuffer of at least the requested size.
        """
        bucket = self._size_to_bucket(size)

        with self._lock:
            self._ensure_bucket_exists(bucket)

            # Try to get from pool
            if self._buckets[bucket]:
                buf_wrapper = self._buckets[bucket].popleft()
                buf_wrapper.acquire_count += 1
                self._in_use_buffers += 1
                self._total_allocations += 1
                return buf_wrapper.buffer

            # Bucket empty - auto-grow by creating new buffer
            self._bucket_misses[bucket] = self._bucket_misses.get(bucket, 0) + 1
            buf_wrapper = self._create_buffer(bucket)
            buf_wrapper.acquire_count = 1
            self._total_buffers += 1
            self._in_use_buffers += 1
            self._total_allocations += 1
            return buf_wrapper.buffer

    def release(self, buffer: Any) -> None:
        """Return a buffer to the pool for reuse.

        Args:
            buffer: MTLBuffer to return to the pool.
        """
        if buffer is None:
            return

        # Determine bucket from buffer length
        buf_length = buffer.length()
        bucket = self._size_to_bucket(buf_length)

        with self._lock:
            self._ensure_bucket_exists(bucket)

            # Create wrapper (we don't track individual buffer identity)
            # Just create a new wrapper for release
            buf_wrapper = _Buffer(buffer=buffer, size=buf_length)
            self._buckets[bucket].append(buf_wrapper)
            self._in_use_buffers = max(0, self._in_use_buffers - 1)
            self._total_releases += 1

    def stats(self) -> dict[str, int]:
        """Return pool statistics.

        Returns:
            Dictionary with:
            - total_buffers: Total buffers created (in pool + in use)
            - in_use_buffers: Currently acquired buffers
            - available_buffers: Buffers available for reuse
            - total_allocations: Total acquire calls
            - total_releases: Total release calls
            - bucket_count: Number of size buckets
            - bucket_misses: Number of times buckets were empty (auto-grew)
        """
        with self._lock:
            available = sum(len(bucket) for bucket in self._buckets.values())
            total_misses = sum(self._bucket_misses.values())

            return {
                "total_buffers": self._total_buffers,
                "in_use_buffers": self._in_use_buffers,
                "available_buffers": available,
                "total_allocations": self._total_allocations,
                "total_releases": self._total_releases,
                "bucket_count": len(self._buckets),
                "bucket_misses": total_misses,
            }

    def clear(self) -> None:
        """Clear all buffers from the pool.

        This releases all pooled buffers (not in-use buffers).
        Call this to free memory when the pool is no longer needed.
        """
        with self._lock:
            self._buckets.clear()
            # Note: We don't decrement _total_buffers because
            # the buffers still exist - they're just not tracked

    def __enter__(self) -> MetalBufferPool:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - clears the pool."""
        self.clear()


# Backwards compatibility: re-export from internal buffer pool
# These are available for advanced use cases
from metal_marlin._buffer_pool import (
    BufferPoolMetrics,
    BufferPriority,
    MetalBufferPool as _InternalMetalBufferPool,
    TrackedBuffer,
    TransientRingBuffer,
)

__all__ = [
    "MetalBufferPool",
    # Backwards compatibility exports
    "BufferPriority",
    "TrackedBuffer",
    "BufferPoolMetrics",
    "TransientRingBuffer",
]
