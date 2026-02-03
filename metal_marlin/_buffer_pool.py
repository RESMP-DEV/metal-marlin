"""Metal buffer pooling for shared CPU/GPU buffers.

Provides buffer pooling with priority-based eviction control:
- PINNED: Weight buffers, never evicted
- HIGH: Activation buffers, evicted only under memory pressure
- NORMAL: General buffers, standard eviction
- LOW: Output buffers, evicted first

Also provides TransientRingBuffer for per-forward-pass allocations:
- Allocates large ring buffer once
- Hands out sequential slices during forward
- Resets pointer at end of forward (zero-cost reuse)

Memory defragmentation:
- Merge pools with similar sizes to reduce external fragmentation
- Release redundant small buffers when larger ones can serve requests
- Auto-trigger on allocation failure, or call defragment() manually

Cache line alignment:
- M3 Max cache line: 128 bytes
- All buffer sizes rounded to 128-byte multiples
- Large buffers (>64KB) aligned to 16KB page boundaries

Metrics and Tuning:
- Hit rate tracking (reuse vs new allocation)
- Fragmentation level monitoring
- Peak memory usage tracking
- Average buffer lifetime statistics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import Metal


class BufferPriority(IntEnum):
    """Buffer eviction priority. Lower values = evicted first."""

    LOW = 0  # Output/temporary buffers, evict first
    NORMAL = 1  # General purpose buffers
    HIGH = 2  # Activation buffers, hot path
    PINNED = 3  # Weight buffers, never evict


@dataclass(slots=True)
class TrackedBuffer:
    """Buffer with metadata for priority-aware management."""

    buffer: Any  # MTLBuffer
    size: int
    priority: BufferPriority
    ref_count: int = 0  # Track active references for eviction safety
    created_at: float = 0.0  # Timestamp when buffer was allocated
    last_acquired_at: float = 0.0  # Timestamp when last acquired from pool
    acquisition_count: int = 0  # Number of times this buffer was reused
    total_time_in_pool: float = 0.0  # Cumulative time spent in pool (seconds)
    last_release_at: float = 0.0  # Timestamp when last released to pool


@dataclass
class BufferPoolMetrics:
    """Comprehensive metrics for buffer pool efficiency tracking.
    
    These metrics help tune pool parameters for optimal performance:
    - High hit rate (>80%) indicates good pool sizing
    - Low fragmentation (<30%) indicates healthy memory layout
    - Short average lifetime suggests aggressive eviction or small pool
    - Peak usage helps size the pool appropriately
    """

    # Hit rate tracking (reuse vs new allocation)
    cache_hits: int = 0  # Buffers successfully reused from pool
    cache_misses: int = 0  # New buffers allocated
    hit_rate_smoothed: float = 0.0  # Exponentially smoothed hit rate (0-1)

    # Fragmentation tracking
    fragmentation_ratio: float = 0.0  # 0 = perfect, 1 = highly fragmented
    external_fragmentation: float = 0.0  # Free space that can't be used
    internal_fragmentation: float = 0.0  # Wasted space within allocated buffers

    # Memory usage tracking
    current_allocated: int = 0  # Currently allocated bytes
    current_pooled: int = 0  # Bytes available in pool for reuse
    peak_allocated: int = 0  # Peak memory allocated
    peak_pooled: int = 0  # Peak pooled memory
    peak_total: int = 0  # Peak (allocated + pooled)

    # Buffer lifetime tracking (seconds)
    total_buffer_lifetime_sum: float = 0.0  # Sum of all buffer lifetimes
    buffer_lifetime_count: int = 0  # Number of buffers tracked for lifetime
    avg_buffer_lifetime: float = 0.0  # Average time buffers spend in pool
    min_buffer_lifetime: float = float('inf')
    max_buffer_lifetime: float = 0.0

    # Operation counters
    allocations: int = 0  # Total allocation requests
    releases: int = 0  # Total releases back to pool
    evictions: int = 0  # Buffers evicted from pool
    defragmentations: int = 0  # Defragmentation operations
    bytes_evicted: int = 0  # Total bytes evicted

    # Size distribution (aligned_size -> count)
    request_size_histogram: dict[int, int] = field(default_factory=dict)
    allocation_size_histogram: dict[int, int] = field(default_factory=dict)

    # Timing
    first_allocation_at: float = 0.0
    last_allocation_at: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as ratio of cache hits to total requests."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def total_requests(self) -> int:
        """Total allocation requests."""
        return self.cache_hits + self.cache_misses

    @property
    def reuse_ratio(self) -> float:
        """Ratio of releases that were later reused."""
        if self.releases == 0:
            return 0.0
        return self.cache_hits / max(self.releases, 1)

    @property
    def pool_efficiency(self) -> float:
        """Overall pool efficiency score (0-1).
        
        Combines hit rate and fragmentation into a single metric.
        Higher is better.
        """
        if self.total_requests < 10:
            return 0.0  # Not enough data

        # Weight hit rate more heavily than fragmentation
        hit_score = self.hit_rate
        frag_score = 1.0 - self.fragmentation_ratio

        return (hit_score * 0.6) + (frag_score * 0.4)

    def update_smoothed_hit_rate(self, new_hit: bool, alpha: float = 0.1) -> None:
        """Update exponentially smoothed hit rate.
        
        Args:
            new_hit: True if this was a cache hit
            alpha: Smoothing factor (higher = more responsive)
        """
        self.hit_rate_smoothed = (alpha * (1.0 if new_hit else 0.0) +
                                  (1 - alpha) * self.hit_rate_smoothed)

    def record_buffer_lifetime(self, lifetime_seconds: float) -> None:
        """Record a buffer's lifetime in the pool."""
        self.total_buffer_lifetime_sum += lifetime_seconds
        self.buffer_lifetime_count += 1
        self.avg_buffer_lifetime = self.total_buffer_lifetime_sum / self.buffer_lifetime_count
        self.min_buffer_lifetime = min(self.min_buffer_lifetime, lifetime_seconds)
        self.max_buffer_lifetime = max(self.max_buffer_lifetime, lifetime_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            # Hit rate
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "hit_rate_smoothed": self.hit_rate_smoothed,
            "reuse_ratio": self.reuse_ratio,

            # Fragmentation
            "fragmentation_ratio": self.fragmentation_ratio,
            "external_fragmentation": self.external_fragmentation,
            "internal_fragmentation": self.internal_fragmentation,

            # Memory
            "current_allocated_bytes": self.current_allocated,
            "current_pooled_bytes": self.current_pooled,
            "peak_allocated_bytes": self.peak_allocated,
            "peak_pooled_bytes": self.peak_pooled,
            "peak_total_bytes": self.peak_total,

            # Lifetime
            "avg_buffer_lifetime_sec": self.avg_buffer_lifetime,
            "min_buffer_lifetime_sec": self.min_buffer_lifetime if self.min_buffer_lifetime != float('inf') else 0.0,
            "max_buffer_lifetime_sec": self.max_buffer_lifetime,

            # Operations
            "total_allocations": self.allocations,
            "total_releases": self.releases,
            "total_evictions": self.evictions,
            "total_bytes_evicted": self.bytes_evicted,
            "defragmentation_count": self.defragmentations,

            # Efficiency
            "pool_efficiency": self.pool_efficiency,
            "uptime_sec": (self.last_allocation_at - self.first_allocation_at)
                          if self.first_allocation_at > 0 else 0.0,
        }

    def get_tuning_recommendations(self) -> list[str]:
        """Generate tuning recommendations based on metrics."""
        recommendations: list[str] = []

        if self.total_requests < 100:
            recommendations.append("Insufficient data for tuning recommendations (need 100+ requests)")
            return recommendations

        # Hit rate analysis
        if self.hit_rate < 0.5:
            recommendations.append(
                f"Low hit rate ({self.hit_rate:.1%}). Consider increasing pool size or max_pool_size."
            )
        elif self.hit_rate > 0.95:
            recommendations.append(
                f"Very high hit rate ({self.hit_rate:.1%}). Pool may be oversized - consider reducing max_pool_size."
            )

        # Fragmentation analysis
        if self.fragmentation_ratio > 0.5:
            recommendations.append(
                f"High fragmentation ({self.fragmentation_ratio:.1%}). Consider more frequent defragmentation."
            )

        # Lifetime analysis
        if self.avg_buffer_lifetime < 0.001 and self.cache_hits > 0:  # Less than 1ms
            recommendations.append(
                "Buffers reused very quickly. Consider using TransientRingBuffer for short-lived allocations."
            )

        # Eviction analysis
        if self.evictions > self.cache_hits * 0.1:  # More than 10% of hits result in evictions
            recommendations.append(
                f"High eviction rate ({self.evictions} evictions). Pool may be undersized."
            )

        return recommendations


# M3 Max cache line size
CACHE_LINE_BYTES = 128
# Page size for large buffer alignment (avoids TLB misses)
PAGE_SIZE_BYTES = 16384
# Threshold for page-aligned allocation
LARGE_BUFFER_THRESHOLD = 65536


def _round_up(value: int, alignment: int = CACHE_LINE_BYTES) -> int:
    """Round size up to alignment boundary.

    Default alignment is 128 bytes (M3 Max cache line).
    For large buffers (>64KB), use page alignment (16KB) instead.
    """
    return ((value + alignment - 1) // alignment) * alignment


def _align_buffer_size(size: int) -> int:
    """Align buffer size for optimal M3 Max cache behavior.

    - Small buffers: 128-byte cache line alignment
    - Large buffers (>64KB): 16KB page alignment to reduce TLB misses
    """
    if size >= LARGE_BUFFER_THRESHOLD:
        return _round_up(size, PAGE_SIZE_BYTES)
    return _round_up(size, CACHE_LINE_BYTES)


class MetalBufferPool:
    """Metal buffer pool with priority-based eviction.

    Buffers are tracked with priorities that control eviction order:
    - PINNED: Never evicted (weight buffers)
    - HIGH: Evicted only under memory pressure (activation buffers)
    - NORMAL: Standard eviction policy
    - LOW: Evicted first (output/temporary buffers)

    Usage:
        pool = MetalBufferPool(device)

        # Get buffer with default priority
        buf = pool.get(1024)

        # Get buffer with specific priority
        weight_buf = pool.get_weight(1024)      # PINNED, never evicted
        act_buf = pool.get_activation(1024)    # HIGH priority
        out_buf = pool.get_output(1024)        # LOW priority, recycled first

        # Pin existing buffer to prevent eviction
        pool.pin(buf)

        # Release buffer back to pool
        pool.release(buf)

        # Evict low-priority buffers to free memory
        freed = pool.evict(target_bytes=1024 * 1024)
    """

    # Defragmentation settings
    _MERGE_RATIO = 2.0  # Merge pools where larger/smaller <= this ratio
    _DEFRAG_INTERVAL = 1000  # Auto-defrag after this many allocations
    _MIN_DEFRAG_INTERVAL_SEC = 5.0  # Minimum seconds between auto-defrags

    def __init__(
        self,
        device: Any,
        *,
        storage_mode: int = Metal.MTLResourceStorageModeShared,
        defrag_interval: int | None = None,
        max_pool_size: int = 0,
    ) -> None:
        """Initialize buffer pool.

        Args:
            device: Metal device for buffer allocation.
            storage_mode: MTLResourceStorageMode for buffers.
            defrag_interval: Allocations between auto-defrag (0 to disable).
            max_pool_size: Maximum total pooled bytes (0 = unlimited).
                          When exceeded, low-priority buffers are evicted.
        """
        self._device = device
        self._storage_mode = storage_mode
        self._max_pool_size = max_pool_size

        # Pool organization: size -> list of TrackedBuffer
        self._pools: dict[int, list[TrackedBuffer]] = {}

        # Backing storage for shared buffers (keeps bytearray alive)
        self._backing: dict[int, bytearray] = {}

        # Track buffer metadata by buffer id
        self._tracked: dict[int, TrackedBuffer] = {}

        # Defragmentation state
        self._alloc_count = 0
        self._defrag_interval = (
            defrag_interval if defrag_interval is not None else self._DEFRAG_INTERVAL
        )
        self._last_defrag_time = 0.0

        # Comprehensive metrics tracking
        self._metrics = BufferPoolMetrics()

    @property
    def total_pooled_bytes(self) -> int:
        """Total bytes in pool available for reuse."""
        return self._metrics.current_pooled

    @property
    def total_allocated_bytes(self) -> int:
        """Total bytes allocated (in use + pooled)."""
        return self._metrics.current_allocated

    @property
    def eviction_count(self) -> int:
        """Number of buffers evicted."""
        return self._metrics.evictions

    @property
    def metrics(self) -> BufferPoolMetrics:
        """Get the current metrics snapshot."""
        # Update fragmentation before returning
        self._metrics.fragmentation_ratio = self._fragmentation_ratio()
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self._metrics = BufferPoolMetrics()

    def get(
        self,
        size: int,
        priority: BufferPriority = BufferPriority.NORMAL,
    ) -> Any:
        """Get or create buffer of at least `size` bytes.

        Args:
            size: Minimum buffer size in bytes.
            priority: Eviction priority for this buffer.

        Returns:
            MTLBuffer of at least the requested size.
        """
        # Check for periodic defragmentation
        self._alloc_count += 1
        if self._defrag_interval > 0 and self._alloc_count >= self._defrag_interval:
            now = time.monotonic()
            if now - self._last_defrag_time >= self._MIN_DEFRAG_INTERVAL_SEC:
                self.defragment()

        alloc_size = _align_buffer_size(size)

        # Record request in histogram
        self._metrics.request_size_histogram[alloc_size] = (
            self._metrics.request_size_histogram.get(alloc_size, 0) + 1
        )

        # Update timing
        now = time.monotonic()
        if self._metrics.first_allocation_at == 0:
            self._metrics.first_allocation_at = now
        self._metrics.last_allocation_at = now
        self._metrics.allocations += 1

        # Try to find existing buffer of sufficient size
        for pool_size in sorted(self._pools.keys()):
            if pool_size >= alloc_size and self._pools[pool_size]:
                tracked = self._pools[pool_size].pop()
                tracked.priority = priority
                tracked.ref_count += 1

                # Track buffer lifetime
                if tracked.last_release_at > 0:
                    lifetime = now - tracked.last_release_at
                    tracked.total_time_in_pool += lifetime
                    self._metrics.record_buffer_lifetime(lifetime)
                tracked.last_acquired_at = now
                tracked.acquisition_count += 1

                # Update metrics
                self._metrics.cache_hits += 1
                self._metrics.update_smoothed_hit_rate(True)
                self._metrics.current_pooled -= tracked.size

                if not self._pools[pool_size]:
                    del self._pools[pool_size]
                return tracked.buffer

        # Check if we need to evict to stay under max_pool_size
        if self._max_pool_size > 0:
            while (
                self._metrics.current_allocated + alloc_size > self._max_pool_size
                and self._metrics.current_pooled > 0
            ):
                evicted = self._evict_one()
                if evicted is None:
                    break

        # Allocate new buffer, with retry after defragmentation on failure
        buffer = self._allocate_buffer(alloc_size)
        if buffer is None:
            self.defragment()
            buffer = self._allocate_buffer(alloc_size)
        if buffer is None:
            if alloc_size == 0:
                # Handle zero-size allocation gracefully
                alloc_size = 128  # Minimum cache line aligned size
                buffer = self._allocate_buffer(alloc_size)
            if buffer is None:
                raise RuntimeError(f"Failed to allocate Metal buffer of {alloc_size} bytes")

        tracked = TrackedBuffer(
            buffer=buffer,
            size=alloc_size,
            priority=priority,
            ref_count=1,
            created_at=now,
            last_acquired_at=now,
        )
        self._tracked[id(buffer)] = tracked

        # Update metrics for cache miss
        self._metrics.cache_misses += 1
        self._metrics.update_smoothed_hit_rate(False)
        self._metrics.current_allocated += alloc_size
        self._metrics.allocation_size_histogram[alloc_size] = (
            self._metrics.allocation_size_histogram.get(alloc_size, 0) + 1
        )

        # Update peaks
        self._metrics.peak_allocated = max(self._metrics.peak_allocated,
                                           self._metrics.current_allocated)
        self._metrics.peak_total = max(self._metrics.peak_total,
                                       self._metrics.current_allocated + self._metrics.current_pooled)

        return buffer

    def get_weight(self, size: int) -> Any:
        """Get buffer for weight storage (PINNED, never evicted)."""
        return self.get(size, BufferPriority.PINNED)

    def get_activation(self, size: int) -> Any:
        """Get buffer for activations (HIGH priority, hot path)."""
        return self.get(size, BufferPriority.HIGH)

    def get_output(self, size: int) -> Any:
        """Get buffer for outputs (LOW priority, recycled first)."""
        return self.get(size, BufferPriority.LOW)

    def _allocate_buffer(self, alloc_size: int) -> Any | None:
        """Allocate a new Metal buffer. Returns None on failure."""
        if self._storage_mode == Metal.MTLResourceStorageModeShared:
            backing = bytearray(alloc_size)
            buffer = self._device.newBufferWithBytesNoCopy_length_options_deallocator_(
                backing,
                alloc_size,
                Metal.MTLResourceStorageModeShared,
                None,
            )
            if buffer is not None:
                self._backing[id(buffer)] = backing
        else:
            buffer = self._device.newBufferWithLength_options_(
                alloc_size, self._storage_mode
            )
        return buffer

    def pin(self, buf: Any) -> None:
        """Pin a buffer to prevent eviction (sets PINNED priority)."""
        buf_id = id(buf)
        tracked = self._tracked.get(buf_id)
        if tracked is not None:
            tracked.priority = BufferPriority.PINNED

    def unpin(
        self, buf: Any, new_priority: BufferPriority = BufferPriority.NORMAL
    ) -> None:
        """Unpin buffer, restoring it to specified eviction priority."""
        buf_id = id(buf)
        tracked = self._tracked.get(buf_id)
        if tracked is not None and tracked.priority == BufferPriority.PINNED:
            tracked.priority = new_priority

    def set_priority(self, buf: Any, priority: BufferPriority) -> None:
        """Set buffer eviction priority."""
        buf_id = id(buf)
        tracked = self._tracked.get(buf_id)
        if tracked is not None:
            tracked.priority = priority

    def get_priority(self, buf: Any) -> BufferPriority | None:
        """Get buffer eviction priority, or None if not tracked."""
        buf_id = id(buf)
        tracked = self._tracked.get(buf_id)
        return tracked.priority if tracked is not None else None

    def release(self, buf: Any) -> None:
        """Return buffer to pool for reuse."""
        buf_id = id(buf)
        tracked = self._tracked.get(buf_id)
        if tracked is None:
            return
        tracked.ref_count = max(0, tracked.ref_count - 1)
        if tracked.ref_count == 0:
            now = time.monotonic()
            tracked.last_release_at = now
            self._pools.setdefault(tracked.size, []).append(tracked)

            # Update metrics
            self._metrics.releases += 1
            self._metrics.current_pooled += tracked.size
            self._metrics.current_allocated -= tracked.size
            self._metrics.peak_pooled = max(self._metrics.peak_pooled,
                                            self._metrics.current_pooled)

    def _evict_one(self) -> TrackedBuffer | None:
        """Evict one low-priority buffer. Returns evicted buffer or None."""
        # Find lowest priority non-empty pool
        for priority in BufferPriority:
            if priority == BufferPriority.PINNED:
                continue  # Never evict pinned buffers

            # Find any buffer with this priority
            for size, pool in list(self._pools.items()):
                for i, tracked in enumerate(pool):
                    if tracked.priority == priority and tracked.ref_count == 0:
                        # Evict this buffer
                        pool.pop(i)
                        if not pool:
                            del self._pools[size]
                        del self._tracked[id(tracked.buffer)]
                        self._backing.pop(id(tracked.buffer), None)

                        # Update metrics
                        self._metrics.current_pooled -= tracked.size
                        self._metrics.current_allocated -= tracked.size
                        self._metrics.evictions += 1
                        self._metrics.bytes_evicted += tracked.size

                        return tracked

        return None

    def evict(self, target_bytes: int) -> int:
        """Evict buffers to free at least target_bytes.

        Args:
            target_bytes: Minimum bytes to free.

        Returns:
            Actual bytes freed.
        """
        freed = 0
        while freed < target_bytes and self._metrics.current_pooled > 0:
            evicted_buf = self._evict_one()
            if evicted_buf is None:
                break
            freed += evicted_buf.size
            self._metrics.bytes_evicted += evicted_buf.size
        return freed

    def defragment(self) -> int:
        """Defragment the buffer pool by merging similar-sized pools.

        Returns the number of bytes freed.

        Strategy:
        1. Merge pools where the size ratio is <= _MERGE_RATIO
           (smaller buffers are released, larger ones kept)
        2. Clean up empty pools
        """
        self._alloc_count = 0
        self._last_defrag_time = time.monotonic()

        if not self._pools:
            return 0

        # Sort pools by size
        sorted_sizes = sorted(self._pools.keys())
        if len(sorted_sizes) < 2:
            return 0

        bytes_freed = 0
        pools_to_remove: list[int] = []

        # Merge smaller pools into larger compatible pools
        i = 0
        while i < len(sorted_sizes) - 1:
            small_size = sorted_sizes[i]
            small_pool = self._pools.get(small_size, [])

            if not small_pool:
                i += 1
                continue

            # Find the next larger pool that can absorb this one
            for j in range(i + 1, len(sorted_sizes)):
                large_size = sorted_sizes[j]
                if large_size / small_size > self._MERGE_RATIO:
                    break  # Too different in size

                large_pool = self._pools.get(large_size, [])
                if not large_pool:
                    continue

                # Merge: release all small buffers, they'll be replaced by large ones
                for tracked in small_pool:
                    buf_id = id(tracked.buffer)
                    self._tracked.pop(buf_id, None)
                    self._backing.pop(buf_id, None)
                    bytes_freed += tracked.size
                    self._metrics.current_allocated -= tracked.size
                    self._metrics.current_pooled -= tracked.size

                pools_to_remove.append(small_size)
                break

            i += 1

        # Remove merged pools
        for size in pools_to_remove:
            self._pools.pop(size, None)

        # Clean up empty pools
        empty_pools = [size for size, pool in self._pools.items() if not pool]
        for size in empty_pools:
            del self._pools[size]

        # Update metrics
        if bytes_freed > 0:
            self._metrics.defragmentations += 1

        return bytes_freed

    def compact(self) -> int:
        """More aggressive defragmentation: consolidate all free buffers.

        Releases all pooled buffers except the largest in each size class.
        This is useful when memory pressure is high.

        Returns the number of bytes freed.
        """
        bytes_freed = 0

        for size, pool in list(self._pools.items()):
            if len(pool) <= 1:
                continue

            # Keep only one buffer per size class
            buffers_to_release = pool[:-1]
            self._pools[size] = pool[-1:]

            for tracked in buffers_to_release:
                buf_id = id(tracked.buffer)
                self._tracked.pop(buf_id, None)
                self._backing.pop(buf_id, None)
                bytes_freed += tracked.size
                self._metrics.current_allocated -= tracked.size
                self._metrics.current_pooled -= tracked.size

        # Also run regular defragmentation
        bytes_freed += self.defragment()
        return bytes_freed

    def stats(self) -> dict[str, Any]:
        """Return pool statistics for monitoring.

        Returns dict with:
        - pool_count: Number of size buckets
        - buffer_count: Total buffers in pool (available for reuse)
        - total_allocated_bytes: All bytes allocated
        - total_pooled_bytes: Bytes available in pool
        - allocation_count: Allocations since last defrag
        - eviction_count: Total evictions
        - size_distribution: {size: count} for each pool bucket
        - priority_counts: {priority_name: total_buffer_count}
        - priority_bytes: {priority_name: total_bytes}
        - pooled_priority_counts: {priority_name: pooled_buffer_count}
        - fragmentation_ratio: 0.0 (optimal) to 1.0 (fragmented)
        - hit_rate: Cache hit rate (0-1)
        - peak_memory_bytes: Peak memory usage
        - avg_buffer_lifetime_sec: Average time buffers spend in pool
        """
        pool_count = len(self._pools)
        buffer_count = sum(len(pool) for pool in self._pools.values())
        size_distribution = {size: len(pool) for size, pool in self._pools.items() if pool}

        # Priority breakdown for all tracked buffers
        priority_counts: dict[str, int] = {p.name: 0 for p in BufferPriority}
        priority_bytes: dict[str, int] = {p.name: 0 for p in BufferPriority}
        for tracked in self._tracked.values():
            priority_counts[tracked.priority.name] += 1
            priority_bytes[tracked.priority.name] += tracked.size

        # Priority breakdown for pooled (available) buffers only
        pooled_priority_counts: dict[str, int] = {p.name: 0 for p in BufferPriority}
        for pool in self._pools.values():
            for tracked in pool:
                pooled_priority_counts[tracked.priority.name] += 1

        # Update metrics
        self._metrics.fragmentation_ratio = self._fragmentation_ratio()

        return {
            "pool_count": pool_count,
            "buffer_count": buffer_count,
            "total_allocated_bytes": self._metrics.current_allocated,
            "total_pooled_bytes": self._metrics.current_pooled,
            "allocation_count": self._alloc_count,
            "eviction_count": self._metrics.evictions,
            "size_distribution": size_distribution,
            "priority_counts": priority_counts,
            "priority_bytes": priority_bytes,
            "pooled_priority_counts": pooled_priority_counts,
            "fragmentation_ratio": self._metrics.fragmentation_ratio,
            "hit_rate": self._metrics.hit_rate,
            "hit_rate_smoothed": self._metrics.hit_rate_smoothed,
            "peak_memory_bytes": self._metrics.peak_total,
            "peak_allocated_bytes": self._metrics.peak_allocated,
            "peak_pooled_bytes": self._metrics.peak_pooled,
            "avg_buffer_lifetime_sec": self._metrics.avg_buffer_lifetime,
            "pool_efficiency": self._metrics.pool_efficiency,
            "total_requests": self._metrics.total_requests,
            "tuning_recommendations": self._metrics.get_tuning_recommendations(),
        }

    def _fragmentation_ratio(self) -> float:
        """Calculate fragmentation ratio (0 = perfect, 1 = highly fragmented).

        Based on the variance in pool sizes relative to a uniform distribution.
        """
        if not self._pools:
            return 0.0

        sizes = list(self._pools.keys())
        if len(sizes) < 2:
            return 0.0

        # Count buffers per pool
        counts = [len(self._pools[s]) for s in sizes]
        total = sum(counts)
        if total == 0:
            return 0.0

        # Ideal: all buffers in one pool
        # Fragmented: buffers spread across many pools
        # Use normalized entropy as fragmentation measure
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * (p if p > 0 else 1)  # Avoid log for simplicity

        # Normalize: max entropy is when evenly distributed
        max_entropy = -len(sizes) * (1 / len(sizes)) ** 2 if len(sizes) > 0 else 1
        if max_entropy == 0:
            return 0.0

        return 1.0 - (entropy / max_entropy) if max_entropy != 0 else 0.0


class TransientRingBuffer:
    """Ring buffer for per-forward-pass transient allocations.

    Pre-allocates a large contiguous Metal buffer and hands out sequential
    slices during a forward pass. Calling reset() at the start of each forward
    pass returns the offset to zero, enabling zero-cost buffer reuse without
    any per-allocation overhead.

    Usage:
        ring = TransientRingBuffer(device, capacity=100_000_000)  # 100MB

        def forward_pass(...):
            ring.reset()  # Start of forward - O(1) reset

            # Get slices for transient allocations
            buf1, offset1 = ring.alloc(batch * hidden * 2)  # fp16
            buf2, offset2 = ring.alloc(batch * hidden * 4)  # fp32

            # Use buffers...
            # No cleanup needed - reset on next forward

    The ring buffer returns the same underlying Metal buffer for all allocations,
    with different byte offsets. Callers must use the offset when dispatching
    Metal kernels to access the correct region.
    """

    def __init__(
        self,
        device: Any,
        capacity: int,
        *,
        storage_mode: int | None = None,
    ) -> None:
        """Initialize ring buffer with given capacity.

        Args:
            device: MTLDevice for buffer allocation
            capacity: Total capacity in bytes
            storage_mode: Metal storage mode (default: MTLResourceStorageModeShared)
        """
        if storage_mode is None:
            storage_mode = Metal.MTLResourceStorageModeShared

        self._device = device
        # Use page alignment for the large ring buffer
        self._capacity = _align_buffer_size(capacity)
        self._storage_mode = storage_mode
        self._offset = 0

        # Allocate backing buffer - shared mode for CPU/GPU access
        self._backing = bytearray(self._capacity)
        self._buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            self._backing,
            self._capacity,
            storage_mode,
            None,
        )
        if self._buffer is None:
            raise RuntimeError(
                f"Failed to allocate ring buffer of {self._capacity} bytes"
            )

        # Track high water mark for monitoring
        self._high_water_mark = 0
        self._allocation_count = 0

    @property
    def capacity(self) -> int:
        """Total buffer capacity in bytes."""
        return self._capacity

    @property
    def used(self) -> int:
        """Currently used bytes since last reset."""
        return self._offset

    @property
    def available(self) -> int:
        """Available bytes for allocation."""
        return self._capacity - self._offset

    @property
    def high_water_mark(self) -> int:
        """Peak usage since creation (bytes)."""
        return self._high_water_mark

    @property
    def buffer(self) -> Any:
        """Underlying Metal buffer for direct access."""
        return self._buffer

    def reset(self) -> None:
        """Reset allocation pointer to start. Call at beginning of forward pass."""
        self._offset = 0
        self._allocation_count = 0

    def can_alloc(self, size: int) -> bool:
        """Check if allocation of given size would succeed."""
        aligned_size = _round_up(size, CACHE_LINE_BYTES)
        return self._offset + aligned_size <= self._capacity

    def alloc(self, size: int) -> tuple[Any, int]:
        """Allocate a region from the ring buffer.

        Args:
            size: Requested allocation size in bytes

        Returns:
            Tuple of (metal_buffer, byte_offset) for dispatching kernels

        Raises:
            RuntimeError: If allocation exceeds capacity
        """
        aligned_size = _round_up(size, CACHE_LINE_BYTES)

        if self._offset + aligned_size > self._capacity:
            raise RuntimeError(
                f"Ring buffer overflow: requested {size} bytes "
                f"(aligned to {aligned_size}), "
                f"only {self._capacity - self._offset} available "
                f"of {self._capacity} total capacity. "
                f"Consider increasing ring buffer size or calling reset()."
            )

        offset = self._offset
        self._offset += aligned_size
        self._allocation_count += 1

        # Update high water mark
        if self._offset > self._high_water_mark:
            self._high_water_mark = self._offset

        return self._buffer, offset

    def alloc_bytes(self, size: int) -> tuple[memoryview, int]:
        """Allocate a region and return a memoryview for CPU access.

        Args:
            size: Requested allocation size in bytes

        Returns:
            Tuple of (memoryview into backing storage, byte_offset)
        """
        aligned_size = _round_up(size, CACHE_LINE_BYTES)

        if self._offset + aligned_size > self._capacity:
            raise RuntimeError(
                f"Ring buffer overflow: requested {size} bytes "
                f"(aligned to {aligned_size}), "
                f"only {self._capacity - self._offset} available"
            )

        offset = self._offset
        self._offset += aligned_size
        self._allocation_count += 1

        if self._offset > self._high_water_mark:
            self._high_water_mark = self._offset

        # Return memoryview into the requested region
        view = memoryview(self._backing)[offset : offset + size]
        return view, offset

    def stats(self) -> dict[str, Any]:
        """Return ring buffer statistics."""
        return {
            "capacity_bytes": self._capacity,
            "used_bytes": self._offset,
            "available_bytes": self._capacity - self._offset,
            "high_water_mark_bytes": self._high_water_mark,
            "allocation_count": self._allocation_count,
            "utilization": self._offset / self._capacity if self._capacity > 0 else 0,
            "peak_utilization": (
                self._high_water_mark / self._capacity if self._capacity > 0 else 0
            ),
        }


# Module-level ring buffer instance for dispatch functions
_transient_ring: TransientRingBuffer | None = None
_transient_ring_device_id: int | None = None


def get_transient_ring(
    device: Any,
    capacity: int = 100_000_000,  # 100MB default
) -> TransientRingBuffer:
    """Get or create the module-level transient ring buffer.

    The ring buffer is lazily created on first access and reused for all
    subsequent calls. If the device changes, a new ring buffer is created.

    Args:
        device: MTLDevice for buffer allocation
        capacity: Buffer capacity in bytes (only used on first creation)

    Returns:
        TransientRingBuffer instance for the device
    """
    global _transient_ring, _transient_ring_device_id

    device_id = id(device)
    if _transient_ring is None or _transient_ring_device_id != device_id:
        _transient_ring = TransientRingBuffer(device, capacity)
        _transient_ring_device_id = device_id

    return _transient_ring


def reset_transient_ring() -> None:
    """Reset the transient ring buffer offset.

    Call at the start of each forward pass to reclaim all transient allocations.
    This is O(1) and does not deallocate or reallocate any memory.
    """
    global _transient_ring
    if _transient_ring is not None:
        _transient_ring.reset()


def transient_ring_stats() -> dict[str, Any] | None:
    """Get statistics for the transient ring buffer, if initialized."""
    global _transient_ring
    if _transient_ring is not None:
        return _transient_ring.stats()
    return None
