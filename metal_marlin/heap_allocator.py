"""Metal heap allocator for efficient sub-allocation.

Provides MTLHeap-based allocation for reducing memory allocation overhead:
- Large heap allocated once, sub-allocated into buffers
- Reduces allocation overhead vs individual MTLBuffer allocations
- Better memory locality and cache behavior
- Automatic buffer pooling with heap-aware tracking

Usage:
    allocator = MetalHeapAllocator(device, heap_size=256_000_000)

    # Allocate buffer from heap (auto-aligned)
    buf, offset = allocator.alloc(1024)

    # Get buffer without offset for direct access
    buf = allocator.alloc_buffer(1024)

    # Release back to pool
    allocator.release(buf)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._compat import HAS_PYOBJC_METAL

if TYPE_CHECKING:
    import Metal

if HAS_PYOBJC_METAL:
    import Metal


@dataclass(slots=True)
class HeapAllocation:
    """Single allocation from heap with metadata."""

    buffer: Any  # MTLBuffer
    size: int
    offset: int
    heap_offset: int  # Offset within the heap
    created_at: float = 0.0
    last_used_at: float = 0.0
    use_count: int = 0


@dataclass
class HeapAllocatorMetrics:
    """Metrics for heap allocator performance tracking."""

    allocations: int = 0  # Total allocations
    deallocations: int = 0  # Total deallocations
    reuse_count: int = 0  # Allocations that reused freed buffers
    peak_allocated: int = 0  # Peak bytes allocated
    current_allocated: int = 0  # Currently allocated bytes
    fragmentation_waste: int = 0  # Wasted bytes due to fragmentation

    allocation_size_histogram: dict[int, int] = field(default_factory=dict)
    allocation_times: list[float] = field(default_factory=list)

    @property
    def reuse_rate(self) -> float:
        """Ratio of allocations that reused freed buffers."""
        if self.allocations == 0:
            return 0.0
        return self.reuse_count / self.allocations

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocations": self.allocations,
            "deallocations": self.deallocations,
            "reuse_count": self.reuse_count,
            "reuse_rate": self.reuse_rate,
            "peak_allocated_bytes": self.peak_allocated,
            "current_allocated_bytes": self.current_allocated,
            "fragmentation_waste_bytes": self.fragmentation_waste,
            "avg_allocation_time_ms": (
                sum(self.allocation_times) / len(self.allocation_times) * 1000
                if self.allocation_times
                else 0.0
            ),
        }


class MetalHeapAllocator:
    """Heap-based allocator using MTLHeap for efficient sub-allocation.

    MTLHeap allows creating multiple buffers backed by a single heap allocation,
    reducing the overhead of individual buffer allocations and improving memory
    locality.

    Features:
    - Large heap allocated once at initialization
    - Sub-allocation via MTLHeap.makeBuffer_length_options_
    - Buffer pooling for reuse of freed allocations
    - Alignment to cache line/page boundaries
    - Metrics tracking for performance analysis

    Usage:
        allocator = MetalHeapAllocator(device, heap_size=256_000_000)

        # Allocate with offset (for dispatch)
        buf, offset = allocator.alloc(4096)

        # Allocate buffer only (offset available via allocator.get_offset(buf))
        buf = allocator.alloc_buffer(4096)

        # Release back to pool for reuse
        allocator.release(buf)
    """

    CACHE_LINE_BYTES = 128
    PAGE_SIZE_BYTES = 16384
    LARGE_BUFFER_THRESHOLD = 65536
    MIN_HEAP_SIZE = 16_777_216  # 16MB minimum

    def __init__(
        self,
        device: Any,
        heap_size: int = 256_000_000,
        *,
        storage_mode: int = 0,
    ) -> None:
        """Initialize heap allocator.

        Args:
            device: MTLDevice for heap allocation
            heap_size: Total heap size in bytes (default 256MB)
            storage_mode: MTLResourceStorageMode for buffers
        """
        if not HAS_PYOBJC_METAL:
            raise RuntimeError("PyObjC Metal is required for MetalHeapAllocator")

        self._device = device
        self._heap_size = max(heap_size, self.MIN_HEAP_SIZE)
        self._storage_mode = storage_mode

        # Create MTLHeap
        heap_desc = Metal.MTLHeapDescriptor.alloc().init()
        heap_desc.setSize_(self._heap_size)
        heap_desc.setType_(Metal.MTLHeapTypeAutomatic)
        heap_desc.setStorageMode_(storage_mode)
        heap_desc.setCpuCacheMode_(Metal.MTLCPUCacheModeDefaultCache)

        self._heap = device.newHeapWithDescriptor_(heap_desc)
        if self._heap is None:
            raise RuntimeError(f"Failed to create MTLHeap of {self._heap_size} bytes")

        # Pool for freed buffers: size -> list of HeapAllocation
        self._pool: dict[int, list[HeapAllocation]] = {}

        # Track active allocations: buffer_id -> HeapAllocation
        self._allocations: dict[int, HeapAllocation] = {}

        # Metrics
        self._metrics = HeapAllocatorMetrics()

    @property
    def heap(self) -> Any:
        """Get the underlying MTLHeap."""
        return self._heap

    @property
    def heap_size(self) -> int:
        """Total heap size in bytes."""
        return self._heap_size

    @property
    def allocated_bytes(self) -> int:
        """Currently allocated bytes."""
        return self._metrics.current_allocated

    @property
    def available_bytes(self) -> int:
        """Available bytes in heap (estimated)."""
        # MTLHeap doesn't expose available size directly in Python bindings easily
        # We rely on max_available_size if we could, but here we estimate
        return self._heap_size - self._metrics.current_allocated

    @property
    def metrics(self) -> HeapAllocatorMetrics:
        """Get allocator metrics."""
        return self._metrics

    def _align_size(self, size: int) -> int:
        """Align size to appropriate boundary."""
        if size >= self.LARGE_BUFFER_THRESHOLD:
            return (
                (size + self.PAGE_SIZE_BYTES - 1) // self.PAGE_SIZE_BYTES
            ) * self.PAGE_SIZE_BYTES
        return ((size + self.CACHE_LINE_BYTES - 1) // self.CACHE_LINE_BYTES) * self.CACHE_LINE_BYTES

    def alloc(self, size: int) -> tuple[Any, int]:
        """Allocate buffer from heap.

        Args:
            size: Requested size in bytes

        Returns:
            Tuple of (MTLBuffer, byte_offset) for dispatching kernels

        Raises:
            RuntimeError: If allocation exceeds heap capacity
        """
        if not HAS_PYOBJC_METAL:
            raise RuntimeError("PyObjC Metal is required")

        start_time = time.monotonic()
        aligned_size = self._align_size(size)

        # Try to reuse from pool first
        for pool_size, pool_list in sorted(self._pool.items()):
            if pool_size >= aligned_size and pool_list:
                alloc = pool_list.pop()
                alloc.last_used_at = time.monotonic()
                alloc.use_count += 1
                self._allocations[id(alloc.buffer)] = alloc

                # Update metrics
                self._metrics.allocations += 1
                self._metrics.reuse_count += 1
                self._metrics.current_allocated += alloc.size
                self._metrics.peak_allocated = max(
                    self._metrics.peak_allocated, self._metrics.current_allocated
                )

                # Clean up empty pools
                if not pool_list:
                    del self._pool[pool_size]

                allocation_time = time.monotonic() - start_time
                self._metrics.allocation_times.append(allocation_time)

                return alloc.buffer, alloc.heap_offset

        # Allocate new buffer from heap
        # Try allocation
        buffer = self._heap.newBufferWithLength_options_(aligned_size, self._storage_mode)

        # If failed, try to compact pool and retry
        if buffer is None:
            self._compact_pool()
            buffer = self._heap.newBufferWithLength_options_(aligned_size, self._storage_mode)

        if buffer is None:
            raise RuntimeError(
                f"Heap allocation failed: need {aligned_size} bytes. "
                f"Heap size: {self._heap_size}, Allocated: {self._metrics.current_allocated}"
            )

        # Get actual offset in heap
        heap_offset = buffer.heapOffset()

        alloc = HeapAllocation(
            buffer=buffer,
            size=aligned_size,
            offset=heap_offset,
            heap_offset=heap_offset,
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
            use_count=1,
        )

        self._allocations[id(buffer)] = alloc

        # Update metrics
        self._metrics.allocations += 1
        self._metrics.current_allocated += aligned_size
        self._metrics.peak_allocated = max(
            self._metrics.peak_allocated, self._metrics.current_allocated
        )
        self._metrics.allocation_size_histogram[aligned_size] = (
            self._metrics.allocation_size_histogram.get(aligned_size, 0) + 1
        )

        allocation_time = time.monotonic() - start_time
        self._metrics.allocation_times.append(allocation_time)

        return buffer, heap_offset

    def alloc_buffer(self, size: int) -> Any:
        """Allocate buffer from heap (returns buffer only).

        Use get_offset() to retrieve the byte offset if needed.

        Args:
            size: Requested size in bytes

        Returns:
            MTLBuffer allocated from heap
        """
        buffer, _ = self.alloc(size)
        return buffer

    def get_offset(self, buffer: Any) -> int | None:
        """Get the byte offset for a buffer allocated from this heap.

        Args:
            buffer: MTLBuffer to query

        Returns:
            Byte offset within the heap, or None if buffer not tracked
        """
        alloc = self._allocations.get(id(buffer))
        if alloc:
            return alloc.heap_offset
        return None

    def release(self, buffer: Any) -> None:
        """Release buffer back to pool for reuse.

        Args:
            buffer: MTLBuffer to release
        """
        buf_id = id(buffer)
        alloc = self._allocations.pop(buf_id, None)
        if alloc is None:
            return

        # Return to pool for reuse
        self._pool.setdefault(alloc.size, []).append(alloc)

        # Update metrics
        self._metrics.deallocations += 1
        self._metrics.current_allocated -= alloc.size

    def _compact_pool(self) -> int:
        """Release all pooled buffers to free heap space.

        Returns:
            Number of buffers released
        """
        released = 0
        for pool_list in self._pool.values():
            for alloc in pool_list:
                released += 1
        self._pool.clear()
        return released

    def clear_pool(self) -> int:
        """Clear the buffer pool without affecting active allocations.

        Returns:
            Number of buffers removed from pool
        """
        count = self._compact_pool()
        return count

    def reset(self) -> None:
        """Reset allocator, freeing all allocations.

        This invalidates all previously allocated buffers.
        """
        self._allocations.clear()
        self._pool.clear()
        self._metrics = HeapAllocatorMetrics()

    def stats(self) -> dict[str, Any]:
        """Get allocator statistics.

        Returns:
            Dictionary with allocator statistics
        """
        pool_count = sum(len(pool) for pool in self._pool.values())
        size_distribution = {size: len(pool) for size, pool in self._pool.items() if pool}

        return {
            "heap_size_bytes": self._heap_size,
            "allocated_bytes": self._metrics.current_allocated,
            "available_bytes": self.available_bytes,
            "pooled_buffer_count": pool_count,
            "active_allocation_count": len(self._allocations),
            "size_distribution": size_distribution,
            "utilization": self._metrics.current_allocated / self._heap_size,
        } | self._metrics.to_dict()


class HeapBufferPool:
    """Buffer pool using MetalHeapAllocator for efficient allocations.

    Provides a convenient pool interface on top of MTLHeap for cases
    where you need many small buffers with low allocation overhead.

    Usage:
        pool = HeapBufferPool(device, heap_size=128_000_000)

        # Get buffer
        buf = pool.get(1024)

        # Release back to pool
        pool.release(buf)
    """

    def __init__(
        self,
        device: Any,
        heap_size: int = 128_000_000,
        *,
        storage_mode: int = 0,
    ) -> None:
        """Initialize heap buffer pool.

        Args:
            device: MTLDevice
            heap_size: Total heap size in bytes
            storage_mode: MTLResourceStorageMode
        """
        self._allocator = MetalHeapAllocator(device, heap_size, storage_mode=storage_mode)

    def get(self, size: int) -> Any:
        """Get buffer from pool.

        Args:
            size: Minimum buffer size in bytes

        Returns:
            MTLBuffer
        """
        return self._allocator.alloc_buffer(size)

    def release(self, buffer: Any) -> None:
        """Release buffer back to pool.

        Args:
            buffer: MTLBuffer to release
        """
        self._allocator.release(buffer)

    @property
    def allocator(self) -> MetalHeapAllocator:
        """Get the underlying heap allocator."""
        return self._allocator

    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return self._allocator.stats()

    def clear_pool(self) -> int:
        """Clear pooled buffers."""
        return self._allocator.clear_pool()
