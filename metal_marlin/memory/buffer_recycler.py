import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

class BufferRecycler:
    """Recycles byte buffers to reduce allocation overhead.
    
    Implements a size-tiered recycling pool for bytearray buffers used in
    memory-mapped I/O and tensor loading operations. Reduces GC pressure
    and allocation overhead during repeated weight loading.
    
    Features:
    - Size-tiered pooling for efficient lookup
    - Memory limits with LRU eviction
    - Thread-safe operations
    - Hit/miss statistics tracking
    """
    
    def __init__(self, max_pool_size_mb: int = 1024) -> None:
        self.max_pool_size = max_pool_size_mb * 1024 * 1024
        self.current_size = 0
        self.pools: dict[int, list[bytearray]] = {}
        self._lock = threading.RLock()
        
        # Statistics tracking
        self._hits = 0
        self._misses = 0
        self._returns = 0
        self._evictions = 0
        
    def get_buffer(self, size_bytes: int) -> bytearray | None:
        """Get a buffer of at least the requested size.
        
        Args:
            size_bytes: Minimum buffer size needed
            
        Returns:
            A recycled buffer if available, None otherwise
        """
        with self._lock:
            # Try exact size match first
            if size_bytes in self.pools and self.pools[size_bytes]:
                self.current_size -= size_bytes
                self._hits += 1
                return self.pools[size_bytes].pop()
            
            # Try to find a buffer that's close in size (within 25%)
            max_acceptable = int(size_bytes * 1.25)
            for pool_size in sorted(self.pools.keys()):
                if pool_size >= size_bytes and pool_size <= max_acceptable:
                    if self.pools[pool_size]:
                        self.current_size -= pool_size
                        self._hits += 1
                        return self.pools[pool_size].pop()
            
            self._misses += 1
        return None
        
    def return_buffer(self, buffer: bytearray) -> None:
        """Return a buffer to the pool for recycling.
        
        Args:
            buffer: The buffer to recycle
        """
        size_bytes = len(buffer)
        with self._lock:
            self._returns += 1
            
            # Simple policy: reject if would exceed max
            if self.current_size + size_bytes > self.max_pool_size:
                # Try to evict oldest buffers to make room
                self._evict_to_make_room(size_bytes)
                
                # Still over limit? Don't pool
                if self.current_size + size_bytes > self.max_pool_size:
                    return
            
            if size_bytes not in self.pools:
                self.pools[size_bytes] = []
            
            self.pools[size_bytes].append(buffer)
            self.current_size += size_bytes
    
    def _evict_to_make_room(self, needed_bytes: int) -> None:
        """Evict buffers to make room for new entry."""
        target_size = self.max_pool_size - needed_bytes
        
        # Evict from largest pools first to free space quickly
        while self.current_size > target_size and self.pools:
            largest_pool_size = max(
                (size for size, buffers in self.pools.items() if buffers),
                default=0
            )
            if largest_pool_size == 0:
                break
            
            if self.pools[largest_pool_size]:
                self.pools[largest_pool_size].pop()
                self.current_size -= largest_pool_size
                self._evictions += 1
            
            # Remove empty pools
            if not self.pools[largest_pool_size]:
                del self.pools[largest_pool_size]
            
    def get_stats(self) -> dict:
        """Get recycling statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total_requests if total_requests > 0 else 0.0,
                "returns": self._returns,
                "evictions": self._evictions,
                "current_size_bytes": self.current_size,
                "max_size_bytes": self.max_pool_size,
                "utilization": self.current_size / self.max_pool_size if self.max_pool_size > 0 else 0.0,
                "pool_count": sum(len(buffers) for buffers in self.pools.values()),
                "unique_sizes": len(self.pools),
            }
            
    def clear(self) -> None:
        """Clear all pooled buffers."""
        with self._lock:
            self.pools.clear()
            self.current_size = 0

# Global buffer recycler for bytearray reuse across memory operations
_global_buffer_recycler: "BufferRecycler | None" = None
_global_recycler_lock = threading.Lock()


def get_global_buffer_recycler(max_pool_size_mb: int = 1024) -> "BufferRecycler":
    """Get or create the global buffer recycler.
    
    Args:
        max_pool_size_mb: Maximum pool size in MB
        
    Returns:
        Global BufferRecycler instance
    """
    global _global_buffer_recycler
    with _global_recycler_lock:
        if _global_buffer_recycler is None:
            _global_buffer_recycler = BufferRecycler(max_pool_size_mb=max_pool_size_mb)
        return _global_buffer_recycler


def reset_global_buffer_recycler() -> None:
    """Reset the global buffer recycler."""
    global _global_buffer_recycler
    with _global_recycler_lock:
        if _global_buffer_recycler is not None:
            _global_buffer_recycler.clear()
            _global_buffer_recycler = None
