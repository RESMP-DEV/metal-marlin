"""Buffer pooling for efficient transient buffer reuse.

Implements a tiered buffer pool that minimizes allocation overhead by reusing
buffers across inference iterations. Features:
- Size-tiered pooling for efficient lookup
- Age-based eviction to prevent memory bloat
- Watermark-based memory management
- Thread-safe operations
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


@dataclass
class PooledBuffer:
    """A buffer in the pool with metadata for management."""
    tensor: "torch.Tensor"
    size_bytes: int
    added_time: float = field(default_factory=time.time)
    use_count: int = 0
    
    def touch(self) -> None:
        """Mark buffer as recently used."""
        self.added_time = time.time()
        self.use_count += 1


@dataclass
class BufferPoolStats:
    """Statistics for buffer pool performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_allocated: int = 0
    total_pooled: int = 0
    peak_pooled: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def current_utilization(self) -> float:
        """Calculate current pool utilization ratio."""
        return self.total_pooled / self.total_allocated if self.total_allocated > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "total_allocated_bytes": self.total_allocated,
            "total_pooled_bytes": self.total_pooled,
            "peak_pooled_bytes": self.peak_pooled,
            "current_utilization": self.current_utilization,
        }


class BufferPool:
    """Tiered buffer pool for efficient transient buffer reuse.
    
    Organizes buffers into size tiers for efficient allocation:
    - Tiny: < 64KB
    - Small: 64KB - 1MB  
    - Medium: 1MB - 16MB
    - Large: > 16MB
    
    Each tier maintains its own LRU queue for age-based eviction.
    Watermark-based management prevents unbounded memory growth.
    
    Args:
        max_pool_size_bytes: Maximum total size of pooled buffers
        high_watermark: Fraction of max size to trigger eviction (default 0.9)
        low_watermark: Target fraction after eviction (default 0.7)
        max_age_seconds: Maximum age for pooled buffers (default 300)
        max_buffers_per_tier: Maximum buffers per tier (default 32)
    """
    
    # Size tier boundaries
    TINY_MAX = 64 * 1024          # 64KB
    SMALL_MAX = 1024 * 1024       # 1MB
    MEDIUM_MAX = 16 * 1024 * 1024 # 16MB
    
    def __init__(
        self,
        max_pool_size_bytes: int = 512 * 1024 * 1024,  # 512MB default
        high_watermark: float = 0.9,
        low_watermark: float = 0.7,
        max_age_seconds: float = 300.0,
        max_buffers_per_tier: int = 32,
    ) -> None:
        self.max_pool_size = max_pool_size_bytes
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.max_age_seconds = max_age_seconds
        self.max_buffers_per_tier = max_buffers_per_tier
        
        # Tiered pools - each tier is a deque for O(1) popleft/append
        self._tiny_pool: deque[PooledBuffer] = deque()      # < 64KB
        self._small_pool: deque[PooledBuffer] = deque()     # 64KB - 1MB
        self._medium_pool: deque[PooledBuffer] = deque()    # 1MB - 16MB
        self._large_pool: deque[PooledBuffer] = deque()     # > 16MB
        
        # Tracking
        self._current_size = 0
        self._stats = BufferPoolStats()
        self._lock = threading.RLock()
        
        # Pool selection cache for fast lookup
        self._tier_map: dict[int, deque] = {}
    
    def _get_tier(self, size_bytes: int) -> tuple[deque[PooledBuffer], str]:
        """Get the appropriate tier for a buffer size."""
        if size_bytes <= self.TINY_MAX:
            return self._tiny_pool, "tiny"
        elif size_bytes <= self.SMALL_MAX:
            return self._small_pool, "small"
        elif size_bytes <= self.MEDIUM_MAX:
            return self._medium_pool, "medium"
        else:
            return self._large_pool, "large"
    
    def _get_best_fit(
        self, 
        size_bytes: int, 
        device: str
    ) -> PooledBuffer | None:
        """Find best-fit buffer from appropriate tier."""
        tier, tier_name = self._get_tier(size_bytes)
        
        # Try exact size match first
        for i, buf in enumerate(tier):
            if buf.tensor.numel() * buf.tensor.element_size() == size_bytes:
                if str(buf.tensor.device) == device:
                    del tier[i]
                    self._current_size -= buf.size_bytes
                    buf.touch()
                    return buf
        
        # Try larger buffer from same tier (up to 25% larger)
        max_acceptable = int(size_bytes * 1.25)
        for i, buf in enumerate(tier):
            buf_size = buf.tensor.numel() * buf.tensor.element_size()
            if buf_size <= max_acceptable and str(buf.tensor.device) == device:
                del tier[i]
                self._current_size -= buf.size_bytes
                buf.touch()
                return buf
        
        return None
    
    def acquire(
        self, 
        size_bytes: int, 
        device: str,
        dtype: "torch.dtype" | None = None,
    ) -> "torch.Tensor":
        """Acquire a buffer from the pool or create new one.
        
        Args:
            size_bytes: Required buffer size in bytes
            device: Target device
            dtype: Target dtype (default uint8)
            
        Returns:
            Tensor buffer
        """
        import torch
        
        if dtype is None:
            dtype = torch.uint8
        
        with self._lock:
            # Try to get from pool
            buf = self._get_best_fit(size_bytes, device)
            if buf is not None:
                self._stats.hits += 1
                return buf.tensor
            
            self._stats.misses += 1
        
        # Create new buffer (outside lock)
        # Calculate number of elements for the dtype
        element_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
        num_elements = (size_bytes + element_size - 1) // element_size
        
        return torch.empty(num_elements, dtype=dtype, device=device)
    
    def release(
        self, 
        tensor: "torch.Tensor",
        skip_pool: bool = False,
    ) -> None:
        """Release a buffer back to the pool.
        
        Args:
            tensor: Buffer to return to pool
            skip_pool: If True, don't pool and let GC collect
        """
        if skip_pool or tensor is None:
            return
        
        size_bytes = tensor.numel() * tensor.element_size()
        
        with self._lock:
            # Check if we should accept this buffer
            if self._current_size + size_bytes > self.max_pool_size:
                # Trigger eviction if over high watermark
                if self._current_size > self.max_pool_size * self.high_watermark:
                    self._evict_to_target(self.max_pool_size * self.low_watermark)
                
                # Still over limit? Don't pool
                if self._current_size + size_bytes > self.max_pool_size:
                    return
            
            tier, tier_name = self._get_tier(size_bytes)
            
            # Limit buffers per tier
            if len(tier) >= self.max_buffers_per_tier:
                # Evict oldest from this tier
                old_buf = tier.popleft()
                self._current_size -= old_buf.size_bytes
                self._stats.evictions += 1
            
            # Add to pool
            pooled = PooledBuffer(
                tensor=tensor,
                size_bytes=size_bytes,
            )
            tier.append(pooled)
            self._current_size += size_bytes
            self._stats.total_pooled += size_bytes
            
            if self._current_size > self._stats.peak_pooled:
                self._stats.peak_pooled = self._current_size
    
    def _evict_to_target(self, target_size: float) -> None:
        """Evict oldest buffers until target size reached."""
        current_time = time.time()
        
        # Collect all buffers with their ages
        all_buffers: list[tuple[float, str, int]] = []  # (age, tier_name, index)
        
        tiers = [
            ("tiny", self._tiny_pool),
            ("small", self._small_pool),
            ("medium", self._medium_pool),
            ("large", self._large_pool),
        ]
        
        for tier_name, tier in tiers:
            for i, buf in enumerate(tier):
                age = current_time - buf.added_time
                all_buffers.append((age, tier_name, i))
        
        # Sort by age (oldest first)
        all_buffers.sort(reverse=True)
        
        # Evict oldest until under target
        evicted = 0
        for age, tier_name, idx in all_buffers:
            if self._current_size <= target_size:
                break
            
            tier_map = {
                "tiny": self._tiny_pool,
                "small": self._small_pool,
                "medium": self._medium_pool,
                "large": self._large_pool,
            }
            tier = tier_map[tier_name]
            
            # Remove buffer at index (adjusting for previous removals)
            if idx < len(tier):
                buf = tier[idx]
                self._current_size -= buf.size_bytes
                del tier[idx]
                evicted += 1
        
        self._stats.evictions += evicted
    
    def cleanup_old_buffers(self) -> int:
        """Remove buffers older than max_age_seconds.
        
        Returns:
            Number of buffers removed
        """
        current_time = time.time()
        removed = 0
        
        with self._lock:
            for tier in [self._tiny_pool, self._small_pool, self._medium_pool, self._large_pool]:
                to_remove = [
                    i for i, buf in enumerate(tier)
                    if current_time - buf.added_time > self.max_age_seconds
                ]
                # Remove in reverse order to maintain indices
                for i in reversed(to_remove):
                    self._current_size -= tier[i].size_bytes
                    del tier[i]
                    removed += 1
        
        return removed
    
    def clear(self) -> None:
        """Clear all pooled buffers."""
        with self._lock:
            self._tiny_pool.clear()
            self._small_pool.clear()
            self._medium_pool.clear()
            self._large_pool.clear()
            self._current_size = 0
    
    def get_stats(self) -> dict:
        """Get current pool statistics."""
        with self._lock:
            stats = self._stats.to_dict()
            stats.update({
                "current_size_bytes": self._current_size,
                "max_size_bytes": self.max_pool_size,
                "utilization": self._current_size / self.max_pool_size if self.max_pool_size > 0 else 0,
                "tier_sizes": {
                    "tiny": sum(b.size_bytes for b in self._tiny_pool),
                    "small": sum(b.size_bytes for b in self._small_pool),
                    "medium": sum(b.size_bytes for b in self._medium_pool),
                    "large": sum(b.size_bytes for b in self._large_pool),
                },
                "tier_counts": {
                    "tiny": len(self._tiny_pool),
                    "small": len(self._small_pool),
                    "medium": len(self._medium_pool),
                    "large": len(self._large_pool),
                },
            })
            return stats
    
    def get_size(self) -> int:
        """Get current pooled memory size in bytes."""
        with self._lock:
            return self._current_size
