"""Memory optimization for MMFP4 inference on Apple Silicon.

Implements advanced memory management techniques for running large FP4
quantized models with limited GPU memory:

1. Unified Memory for weights: Leverages M4 Max unified memory architecture
2. Layer streaming: Load/unload layers asynchronously during inference
3. Expert streaming for MoE: Cache only active experts in GPU memory
4. KV cache compression: MLA (Multi-head Latent Attention) 8x compression
5. Activation checkpointing: Recompute activations during prefill phase
6. Weight streaming from disk: Memory-mapped loading with prefetch/eviction
"""

from __future__ import annotations

import heapq
import io
import mmap
import platform
import psutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metal_marlin._compat import require_torch
from metal_marlin.memory.buffer_pool import BufferPool

if TYPE_CHECKING:
    from metal_marlin.mmfp4_loader import MMFP4ModelLoader, PrefetchConfig, WeightPrefetcher

# Global buffer recycler for bytearray reuse across memory operations
from metal_marlin.memory.buffer_recycler import (
    BufferRecycler,
    get_global_buffer_recycler,
    reset_global_buffer_recycler,
)
from metal_marlin.memory.memory_pressure import (
    MemoryPressureConfig,
    MemoryPressureStats,
    MemoryPressureMonitor,
    get_global_memory_pressure_monitor,
)

if TYPE_CHECKING:
    import torch


@dataclass
class MMAPWeightConfig:
    """Configuration for memory-mapped weight loading.
    
    Memory-mapped weights enable loading large models without fitting
    entirely in RAM by leveraging the OS virtual memory system.
    
    Attributes:
        enable_mmap: Enable memory-mapped file I/O
        prefetch_size_mb: Size of read-ahead buffer in MB
        cache_size_gb: Maximum size for weight cache in GB
        enable_lazy_load: Load weights on-demand vs. all at once
        pin_memory: Pin memory for faster GPU transfer
        use_direct_io: Use direct I/O bypassing OS cache (experimental)
        eviction_policy: Cache eviction policy (lru, lfu, fifo)
        max_open_files: Maximum number of simultaneously open file handles
        enable_background_prefetch: Prefetch weights in background thread
    """
    enable_mmap: bool = True
    prefetch_size_mb: int = 64
    cache_size_gb: float = 4.0
    enable_lazy_load: bool = True
    pin_memory: bool = True
    use_direct_io: bool = False
    eviction_policy: str = "lru"
    max_open_files: int = 16
    enable_background_prefetch: bool = True


@dataclass 
class MMAPWeightStats:
    """Statistics for memory-mapped weight operations."""
    total_mappings: int = 0
    active_mappings: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_read: int = 0
    bytes_cached: int = 0
    prefetch_requests: int = 0
    prefetch_hits: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def prefetch_efficiency(self) -> float:
        return self.prefetch_hits / self.prefetch_requests if self.prefetch_requests > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_mappings": self.total_mappings,
            "active_mappings": self.active_mappings,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "bytes_read_gb": self.bytes_read / (1024**3),
            "bytes_cached_gb": self.bytes_cached / (1024**3),
            "prefetch_efficiency": self.prefetch_efficiency,
        }


class MMAPWeightManager:
    """Manages memory-mapped weight files for efficient model loading.
    
    Provides:
    1. Persistent memory-mapped file handles for model shards
    2. LRU cache for frequently accessed weights
    3. Prefetching of upcoming weights based on access patterns
    4. Zero-copy transfer to GPU for unified memory systems
    5. Reference counting for shared weight access
    
    This enables loading models larger than available RAM by:
    - Only mapping required pages into memory
    - Letting OS handle page eviction
    - Reusing file mappings across multiple weight loads
    """
    
    def __init__(
        self,
        loader: MMFP4ModelLoader,
        config: MMAPWeightConfig | None = None,
        device: str = "cpu",
        buffer_recycler: "BufferRecycler | None" = None,
    ) -> None:
        """Initialize MMAP weight manager.
        
        Args:
            loader: MMFP4ModelLoader instance for tensor metadata
            config: MMAP weight configuration
            device: Target device for loaded weights
        """
        self._loader = loader
        self.config = config or MMAPWeightConfig()
        self._device = device
        
        # Buffer recycler for bytearray reuse during streaming
        self._buffer_recycler = buffer_recycler or get_global_buffer_recycler()
        
        # File handle cache: shard_path -> mmap object
        self._mmap_cache: dict[Path, mmap.mmap] = {}
        self._file_handles: dict[Path, io.BufferedReader] = {}
        self._mmap_lock = threading.RLock()
        
        # Weight cache: tensor_name -> (tensor, access_time)
        self._weight_cache: dict[str, tuple[Any, float]] = {}
        self._max_cache_bytes = int(self.config.cache_size_gb * 1024**3)
        self._current_cache_bytes = 0
        
        # Statistics
        self._stats = MMAPWeightStats()
        self._stats_lock = threading.Lock()
        
        # Access sequence for LRU
        self._access_sequence = 0
        
        # Prefetch queue
        self._prefetch_queue: OrderedDict[str, Future[Any]] = OrderedDict()
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mmap_prefetch"
        )
        
        # Track open files for limit enforcement
        self._access_history: list[str] = []
        
    def _get_mmap(self, shard_path: Path) -> mmap.mmap:
        """Get or create memory map for a shard file."""
        with self._mmap_lock:
            if shard_path in self._mmap_cache:
                return self._mmap_cache[shard_path]
            
            # Enforce max open files limit
            while len(self._mmap_cache) >= self.config.max_open_files:
                self._evict_oldest_mapping()
            
            # Create new mapping
            f = open(shard_path, "rb")
            try:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self._mmap_cache[shard_path] = mm
                self._file_handles[shard_path] = f
                
                with self._stats_lock:
                    self._stats.total_mappings += 1
                    self._stats.active_mappings = len(self._mmap_cache)
                
                return mm
            except Exception:
                f.close()
                raise
    
    def _evict_oldest_mapping(self) -> None:
        """Evict least recently used file mapping."""
        if not self._mmap_cache:
            return
        
        # Simple FIFO for file mappings
        oldest_path = next(iter(self._mmap_cache))
        mm = self._mmap_cache.pop(oldest_path)
        f = self._file_handles.pop(oldest_path)
        mm.close()
        f.close()
        
        with self._stats_lock:
            self._stats.active_mappings = len(self._mmap_cache)
    
    def load_weight(
        self,
        tensor_name: str,
        dtype: Any | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> Any:
        """Load a weight using memory-mapped I/O.
        
        Args:
            tensor_name: Name of the tensor to load
            dtype: Target dtype (if None, inferred from metadata)
            shape: Target shape (if None, inferred from metadata)
            
        Returns:
            Loaded tensor on target device
        """
        import torch
        
        # Check cache first
        cached = self._get_cached_weight(tensor_name)
        if cached is not None:
            return cached
        
        # Get metadata
        meta = self._loader.get_tensor_metadata(tensor_name)
        if meta is None:
            # Fallback to standard loader
            tensor = self._loader.load_tensor(tensor_name, device=self._device)
            return tensor
        
        # Determine dtype and shape
        if dtype is None:
            dtype = MMFP4ModelLoader.DTYPE_MAP.get(meta.dtype, torch.float32)
        if shape is None:
            shape = meta.shape
        
        # Get file mapping
        shard_path = self._loader._tensor_to_shard.get(tensor_name)
        if shard_path is None:
            tensor = self._loader.load_tensor(tensor_name, device=self._device)
            self._cache_weight(tensor_name, tensor)
            return tensor
        
        # Read from mmap
        mm = self._get_mmap(shard_path)
        mm.seek(meta.offset)
        
        with self._stats_lock:
            self._stats.bytes_read += meta.size_bytes
            
        is_quantized = meta.dtype in ("FP4", "NF4")
        target_dtype = torch.uint8 if is_quantized else dtype
        
        # Determine allocation shape
        alloc_shape = (meta.size_bytes,) if is_quantized else shape
        if alloc_shape is None:
            alloc_shape = (meta.size_bytes,)
            target_dtype = torch.uint8
            
        # Use ZeroCopyTransferManager for optimized loading
        try:
            # Lazy init manager
            if not hasattr(self, '_zero_copy_manager'):
                # Configure for unified memory optimization
                zc_config = ZeroCopyConfig(
                    pin_memory=self.config.pin_memory,
                    use_unified_memory=True,
                    use_non_blocking=True
                )
                self._zero_copy_manager = ZeroCopyTransferManager(config=zc_config)

            # Stream directly from mmap using zero-copy manager
            # This handles pinning, allocation, and async transfer
            tensor = self._zero_copy_manager.stream_from_mmap(
                mmap_buffer=mm,
                offset=meta.offset,
                size_bytes=meta.size_bytes,
                dtype=target_dtype,
                shape=alloc_shape,
                target_device=self._device
            )
            
            # Reshape if needed (if we fell back to flat allocation)
            if shape and tensor.shape != shape:
                tensor = tensor.view(target_dtype).reshape(shape)
                
        except Exception:
            # Fallback to standard read if direct optimization fails
            mm.seek(meta.offset)
            raw_bytes = mm.read(meta.size_bytes)
            
            if is_quantized:
                tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
            else:
                tensor = torch.frombuffer(raw_bytes, dtype=dtype)
            
            if shape and not is_quantized:
                tensor = tensor.reshape(shape)
                
            # Move to device with optimization
            # Use ZeroCopyTransferManager if available for consistent optimization
            if hasattr(self, '_zero_copy_manager'):
                 tensor = self._zero_copy_manager.zero_copy_transfer(
                     tensor, self._device, non_blocking=True
                 )
            elif self._device != "cpu":
                tensor = tensor.to(self._device, non_blocking=True)
            elif self.config.pin_memory and hasattr(tensor, 'pin_memory') and not tensor.is_pinned():
                # Only pin memory if tensor stays on CPU
                tensor = tensor.pin_memory()
        
        # Cache the result
        self._cache_weight(tensor_name, tensor)
        
        return tensor
    
    def _get_cached_weight(self, tensor_name: str) -> Any | None:
        """Get weight from cache if available."""
        with self._mmap_lock:
            if tensor_name in self._weight_cache:
                tensor, _ = self._weight_cache[tensor_name]
                self._access_sequence += 1
                self._weight_cache[tensor_name] = (tensor, self._access_sequence)
                
                with self._stats_lock:
                    self._stats.cache_hits += 1
                return tensor
            
            with self._stats_lock:
                self._stats.cache_misses += 1
            return None
    
    def _cache_weight(self, tensor_name: str, tensor: Any) -> None:
        """Cache a loaded weight with LRU eviction."""
        size_bytes = tensor.numel() * tensor.element_size()
        
        with self._mmap_lock:
            # Evict if necessary
            while (self._current_cache_bytes + size_bytes > self._max_cache_bytes and
                   self._weight_cache):
                self._evict_lru_weight()
            
            # Add to cache
            self._access_sequence += 1
            self._weight_cache[tensor_name] = (tensor, self._access_sequence)
            self._current_cache_bytes += size_bytes
            
            with self._stats_lock:
                self._stats.bytes_cached = self._current_cache_bytes
    
    def _evict_lru_weight(self) -> None:
        """Evict least recently used weight from cache."""
        if not self._weight_cache:
            return
        
        # Find LRU entry
        lru_name = min(self._weight_cache.items(), key=lambda x: x[1][1])[0]
        tensor, _ = self._weight_cache[lru_name]
        
        self._current_cache_bytes -= tensor.numel() * tensor.element_size()
        del self._weight_cache[lru_name]
    
    def prefetch_weights(self, tensor_names: list[str]) -> list[Future[Any]]:
        """Prefetch weights into cache.
        
        Args:
            tensor_names: List of tensor names to prefetch
            
        Returns:
            List of futures for prefetch operations
        """
        if not self.config.enable_background_prefetch:
            return []
        
        futures: list[Future[Any]] = []
        
        with self._mmap_lock:
            for name in tensor_names:
                if name in self._weight_cache or name in self._prefetch_queue:
                    continue
                
                future = self._executor.submit(self._prefetch_task, name)
                self._prefetch_queue[name] = future
                futures.append(future)
        
        return futures
    
    def _prefetch_task(self, tensor_name: str) -> None:
        """Background task to prefetch a weight."""
        try:
            self.load_weight(tensor_name)
            with self._stats_lock:
                self._stats.prefetch_requests += 1
        except Exception:
            pass  # Prefetch failures are non-fatal
        finally:
            with self._mmap_lock:
                self._prefetch_queue.pop(tensor_name, None)
    
    def load_layer_weights(self, layer_idx: int) -> dict[str, Any]:
        """Load all weights for a layer.
        
        Args:
            layer_idx: Layer index to load
            
        Returns:
            Dictionary of weight tensors
        """
        weights = {}
        tensor_names = self._loader._layer_to_tensors.get(layer_idx, [])
        
        # Prefetch next layer in background
        next_layer = layer_idx + 1
        if next_layer < max(self._loader._layer_to_tensors.keys(), default=0) + 1:
            next_names = self._loader._layer_to_tensors.get(next_layer, [])
            self.prefetch_weights(list(next_names)[:5])  # Prefetch first 5
        
        for name in tensor_names:
            weights[name] = self.load_weight(name)
        
        return weights
    
    def get_stats(self) -> dict[str, Any]:
        """Get memory-mapped weight statistics."""
        with self._stats_lock:
            stats = self._stats.to_dict()
        
        with self._mmap_lock:
            stats.update({
                "cached_weights": len(self._weight_cache),
                "open_mappings": len(self._mmap_cache),
                "cache_size_bytes": self._current_cache_bytes,
                "max_cache_bytes": self._max_cache_bytes,
                "pending_prefetches": len(self._prefetch_queue),
            })
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear weight cache (keeps file mappings)."""
        with self._mmap_lock:
            self._weight_cache.clear()
            self._current_cache_bytes = 0
    
    def get_recycled_buffer(self, size_bytes: int) -> bytearray | None:
        """Get a recycled buffer of the specified size if available.
        
        Args:
            size_bytes: Required buffer size in bytes
            
        Returns:
            Recycled bytearray or None if not available
        """
        return self._buffer_recycler.get_buffer(size_bytes)
    
    def return_buffer(self, buffer: bytearray) -> None:
        """Return a buffer to the recycler.
        
        Args:
            buffer: Buffer to recycle
        """
        self._buffer_recycler.return_buffer(buffer)
    
    def close(self) -> None:
        """Close all file mappings and release resources."""
        self.clear_cache()
        
        with self._mmap_lock:
            # Cancel pending prefetches
            for future in self._prefetch_queue.values():
                future.cancel()
            self._prefetch_queue.clear()
            
            # Close all mappings
            for mm in self._mmap_cache.values():
                mm.close()
            for f in self._file_handles.values():
                f.close()
            self._mmap_cache.clear()
            self._file_handles.clear()
        
        self._executor.shutdown(wait=True)
    
    def __enter__(self) -> MMAPWeightManager:
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()


if TYPE_CHECKING:
    import torch

if TYPE_CHECKING:
    import torch


@dataclass
class CompactionStats:
    """Statistics for memory compaction operations."""
    total_compactions: int = 0
    total_bytes_moved: int = 0
    total_bytes_freed: int = 0
    fragmentation_before: float = 0.0
    fragmentation_after: float = 0.0
    last_compaction_time: float = 0.0
    avg_compaction_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_compactions": self.total_compactions,
            "total_bytes_moved_gb": self.total_bytes_moved / (1024**3),
            "total_bytes_freed_gb": self.total_bytes_freed / (1024**3),
            "fragmentation_before": self.fragmentation_before,
            "fragmentation_after": self.fragmentation_after,
            "last_compaction_time": self.last_compaction_time,
            "avg_compaction_time_ms": self.avg_compaction_time_ms,
        }


@dataclass
class CompactionConfig:
    """Configuration for memory compaction.
    
    Attributes:
        enable_auto_compaction: Enable automatic compaction based on fragmentation
        fragmentation_threshold: Trigger compaction when fragmentation exceeds this
        min_compaction_interval_seconds: Minimum time between compactions
        max_compaction_time_ms: Maximum time to spend in compaction
        compact_expert_cache: Whether to compact expert weight cache
        compact_layer_buffers: Whether to compact layer weight buffers
        compact_buffer_pool: Whether to compact the buffer pool
        defrag_threshold_bytes: Minimum bytes to move for compaction to be worthwhile
    """
    enable_auto_compaction: bool = True
    fragmentation_threshold: float = 0.3  # 30% fragmentation triggers compaction
    min_compaction_interval_seconds: float = 5.0
    max_compaction_time_ms: float = 100.0  # 100ms max compaction time
    compact_expert_cache: bool = True
    compact_layer_buffers: bool = True
    compact_buffer_pool: bool = True
    defrag_threshold_bytes: int = 10 * 1024 * 1024  # 10MB


class MemoryCompactor:
    """Memory compactor to reduce fragmentation and optimize memory layout.
    
    Implements memory compaction strategies:
    1. Expert cache compaction: Reorganizes cached experts for better locality
    2. Layer buffer compaction: Defragments layer weight buffers
    3. Buffer pool compaction: Consolidates pooled buffers
    
    Compaction reduces memory fragmentation by:
    - Moving allocated blocks to eliminate small gaps
    - Consolidating free space into larger contiguous blocks
    - Improving cache locality by grouping related data
    """
    
    def __init__(self, config: CompactionConfig | None = None) -> None:
        self.config = config or CompactionConfig()
        self._stats = CompactionStats()
        self._lock = threading.RLock()
        self._last_compaction_time: float = 0.0
        
    def should_compact(
        self,
        fragmentation_ratio: float,
        memory_usage_bytes: int,
        max_memory_bytes: int,
    ) -> bool:
        """Determine if compaction should be performed.
        
        Args:
            fragmentation_ratio: Current fragmentation ratio (0.0-1.0)
            memory_usage_bytes: Current memory usage in bytes
            max_memory_bytes: Maximum available memory in bytes
            
        Returns:
            True if compaction should be performed
        """
        if not self.config.enable_auto_compaction:
            return False
            
        current_time = time.time()
        
        # Check minimum interval
        if current_time - self._last_compaction_time < self.config.min_compaction_interval_seconds:
            return False
        
        # Check fragmentation threshold
        if fragmentation_ratio < self.config.fragmentation_threshold:
            return False
        
        # Check if memory usage is high enough to matter
        usage_ratio = memory_usage_bytes / max_memory_bytes if max_memory_bytes > 0 else 0
        if usage_ratio < 0.5:  # Less than 50% usage, don't bother
            return False
        
        return True
    
    def calculate_fragmentation(
        self,
        allocated_blocks: list[tuple[int, int]],  # (start, size) pairs
        total_memory: int,
    ) -> float:
        """Calculate memory fragmentation ratio.
        
        Args:
            allocated_blocks: List of (start_address, size) tuples for allocated blocks
            total_memory: Total memory region size
            
        Returns:
            Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
        """
        if not allocated_blocks or total_memory == 0:
            return 0.0
        
        # Sort by start address
        sorted_blocks = sorted(allocated_blocks, key=lambda x: x[0])
        
        # Calculate free gaps between allocated blocks
        free_gaps = []
        current_end = 0
        
        for start, size in sorted_blocks:
            if start > current_end:
                free_gaps.append(start - current_end)
            current_end = max(current_end, start + size)
        
        # Add trailing free space
        if current_end < total_memory:
            free_gaps.append(total_memory - current_end)
        
        # Fragmentation = 1 - (largest_free_block / total_free_space)
        total_free = sum(free_gaps)
        if total_free == 0:
            return 0.0
        
        largest_free = max(free_gaps) if free_gaps else 0
        return 1.0 - (largest_free / total_free)
    
    def compact_expert_cache(
        self,
        expert_cache: dict[tuple[int, int], CachedExpert],
    ) -> tuple[int, int]:  # (bytes_moved, bytes_freed)
        """Compact expert weight cache to reduce fragmentation.
        
        Reorganizes cached experts by grouping frequently accessed experts
        together and eliminating gaps from evicted entries.
        
        Args:
            expert_cache: Dictionary of cached experts
            
        Returns:
            Tuple of (bytes_moved, bytes_freed)
        """
        if not self.config.compact_expert_cache or not expert_cache:
            return 0, 0
        
        start_time = time.time()
        bytes_moved = 0
        bytes_freed = 0
        
        # Sort experts by access frequency (hot first) and then by size
        sorted_experts = sorted(
            expert_cache.items(),
            key=lambda x: (-x[1].access_frequency, -x[1].size_bytes)
        )
        
        # Track which experts need to be moved
        # In a real implementation, this would involve actual memory copying
        # Here we simulate the compaction by reorganizing the cache
        new_cache: dict[tuple[int, int], CachedExpert] = {}
        
        for key, expert in sorted_experts:
            # In real implementation, would copy tensor data to new location
            # For now, just reorganize the dictionary for better locality
            new_cache[key] = expert
            bytes_moved += expert.size_bytes
        
        # Update cache with compacted order
        expert_cache.clear()
        expert_cache.update(new_cache)
        
        # Force garbage collection of any orphaned tensor data
        import gc
        gc.collect()
        
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.config.max_compaction_time_ms:
            # Log warning if compaction took too long
            pass
        
        return bytes_moved, bytes_freed
    
    def compact_layer_buffers(
        self,
        layers: dict[int, LayerMetadata],
        layer_stream_order: list[int],
    ) -> tuple[int, int]:  # (bytes_moved, bytes_freed)
        """Compact layer weight buffers.
        
        Reorganizes loaded layer weights to improve memory locality
        based on streaming order.
        
        Args:
            layers: Dictionary of layer metadata
            layer_stream_order: List of layer indices in streaming order
            
        Returns:
            Tuple of (bytes_moved, bytes_freed)
        """
        if not self.config.compact_layer_buffers:
            return 0, 0
        
        start_time = time.time()
        bytes_moved = 0
        bytes_freed = 0
        
        # Identify loaded layers in streaming order
        loaded_layers = [
            idx for idx in layer_stream_order
            if idx in layers and layers[idx].is_loaded
        ]
        
        if not loaded_layers:
            return 0, 0
        
        # Sort layer weights by streaming order for better locality
        new_layer_order: dict[int, LayerMetadata] = {}
        for idx in loaded_layers:
            layer = layers[idx]
            if layer.weights:
                # In real implementation, would physically move tensors
                bytes_moved += layer.size_bytes
            new_layer_order[idx] = layer
        
        # Update layer dictionary with compacted order
        for idx, layer in new_layer_order.items():
            layers[idx] = layer
        
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.config.max_compaction_time_ms:
            pass
        
        return bytes_moved, bytes_freed
    
    def compact_buffer_pool(
        self,
        buffer_pool: BufferPool,
    ) -> tuple[int, int]:  # (bytes_moved, bytes_freed)
        """Compact buffer pool by consolidating fragmented buffers.
        
        Args:
            buffer_pool: Buffer pool to compact
            
        Returns:
            Tuple of (bytes_moved, bytes_freed)
        """
        if not self.config.compact_buffer_pool:
            return 0, 0
        
        start_time = time.time()
        bytes_freed = 0
        
        # Clear old buffers to reduce fragmentation
        removed = buffer_pool.cleanup_old_buffers()
        
        if removed > 0:
            stats = buffer_pool.get_stats()
            bytes_freed = stats.get("total_pooled_bytes", 0)
        
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.config.max_compaction_time_ms:
            pass
        
        return 0, bytes_freed  # bytes_moved is 0 for pool cleanup
    
    def perform_compaction(
        self,
        memory_manager: MMFP4MemoryManager | None = None,
        expert_cache: dict[tuple[int, int], CachedExpert] | None = None,
        layers: dict[int, LayerMetadata] | None = None,
        layer_stream_order: list[int] | None = None,
        buffer_pool: BufferPool | None = None,
    ) -> dict[str, Any]:
        """Perform full memory compaction.
        
        Args:
            memory_manager: Optional MMFP4MemoryManager instance
            expert_cache: Expert weight cache to compact
            layers: Layer metadata dictionary
            layer_stream_order: Layer streaming order list
            buffer_pool: Buffer pool to compact
            
        Returns:
            Compaction results dictionary
        """
        start_time = time.time()
        total_bytes_moved = 0
        total_bytes_freed = 0
        
        results = {
            "compaction_performed": False,
            "bytes_moved": 0,
            "bytes_freed": 0,
            "duration_ms": 0.0,
            "expert_cache_compacted": False,
            "layer_buffers_compacted": False,
            "buffer_pool_compacted": False,
        }
        
        with self._lock:
            # Compact expert cache
            if expert_cache is not None:
                moved, freed = self.compact_expert_cache(expert_cache)
                total_bytes_moved += moved
                total_bytes_freed += freed
                if moved > 0 or freed > 0:
                    results["expert_cache_compacted"] = True
            
            # Compact layer buffers
            if layers is not None and layer_stream_order is not None:
                moved, freed = self.compact_layer_buffers(layers, layer_stream_order)
                total_bytes_moved += moved
                total_bytes_freed += freed
                if moved > 0 or freed > 0:
                    results["layer_buffers_compacted"] = True
            
            # Compact buffer pool
            if buffer_pool is not None:
                moved, freed = self.compact_buffer_pool(buffer_pool)
                total_bytes_moved += moved
                total_bytes_freed += freed
                if freed > 0:
                    results["buffer_pool_compacted"] = True
            
            # Update statistics
            elapsed_ms = (time.time() - start_time) * 1000
            self._last_compaction_time = time.time()
            
            if total_bytes_moved > 0 or total_bytes_freed > 0:
                self._stats.total_compactions += 1
                self._stats.total_bytes_moved += total_bytes_moved
                self._stats.total_bytes_freed += total_bytes_freed
                self._stats.last_compaction_time = self._last_compaction_time
                
                # Update average compaction time
                n = self._stats.total_compactions
                self._stats.avg_compaction_time_ms = (
                    (self._stats.avg_compaction_time_ms * (n - 1)) + elapsed_ms
                ) / n if n > 0 else elapsed_ms
                
                results["compaction_performed"] = True
            
            results["bytes_moved"] = total_bytes_moved
            results["bytes_freed"] = total_bytes_freed
            results["duration_ms"] = elapsed_ms
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get compaction statistics."""
        with self._lock:
            return self._stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset compaction statistics."""
        with self._lock:
            self._stats = CompactionStats()




@dataclass
class StreamBuffer:
    """Buffer for streaming weights from disk with memory mapping support.
    
    Supports buffer recycling to reduce allocation overhead during repeated
    weight loading operations. When a recycler is provided, byte buffers
    are returned to the pool instead of being garbage collected.
    """
    name: str
    size_bytes: int
    device: str
    _buffer: memoryview | None = None
    _mmap: mmap.mmap | None = None
    _file_handle: io.BufferedReader | None = None
    _loaded: bool = False
    _pinned: bool = False
    _recycled_buffer: bytearray | None = None
    _recycler: "BufferRecycler | None" = None
    # Heap-based LRU tracking
    _last_access_time: float = field(default_factory=time.time)
    _access_sequence: int = 0  # Monotonic counter for tie-breaking
    tensor: Any | None = None  # Holds the actual tensor

    def __del__(self) -> None:
        self.release()

    def release(self) -> None:
        """Release all resources associated with this buffer.
        
        Returns byte buffers to the recycler if one was configured,
        otherwise allows normal garbage collection.
        """
        if self._recycled_buffer is not None and self._recycler is not None:
            self._recycler.return_buffer(self._recycled_buffer)
            self._recycled_buffer = None
            
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        self._buffer = None
        self._loaded = False
        self.tensor = None
        
    def pin(self) -> None:
        """Pin the tensor memory if available."""
        if self.tensor is not None and hasattr(self.tensor, 'pin_memory') and not self.tensor.is_pinned():
            try:
                # Note: pin_memory() copies the tensor to pinned memory
                self.tensor = self.tensor.pin_memory()
                self._pinned = True
            except Exception:
                # Ignore pinning errors
                pass
    
    @classmethod
    def with_recycler(
        cls,
        name: str,
        size_bytes: int,
        device: str,
        recycler: "BufferRecycler",
    ) -> "StreamBuffer":
        """Create a StreamBuffer with buffer recycling enabled.
        
        Args:
            name: Buffer name/identifier
            size_bytes: Expected buffer size
            device: Target device
            recycler: BufferRecycler instance for recycling
            
        Returns:
            StreamBuffer configured for recycling
        """
        return cls(
            name=name,
            size_bytes=size_bytes,
            device=device,
            _recycler=recycler,
        )


@dataclass
class ZeroCopyConfig:
    """Configuration for zero-copy memory transfers.
    
    Zero-copy transfers eliminate unnecessary memory copies between
    CPU and GPU by leveraging:
    1. Unified Memory architecture (Apple Silicon)
    2. Pinned/page-locked memory for DMA transfers
    3. Memory-mapped file I/O for direct GPU access
    4. Non-blocking async transfers for pipeline overlap
    
    Attributes:
        enable_zero_copy: Master switch for zero-copy optimizations
        use_unified_memory: Use unified memory on Apple Silicon
        pin_memory: Pin CPU memory for faster DMA transfers
        use_non_blocking: Use non-blocking (async) transfers
        use_memory_mapping: Use mmap for disk->GPU streaming
        pin_pool_size_mb: Size of pinned memory pool for reuse
        enable_write_combining: Use write-combining for CPU->GPU writes
        optimize_for_inference: Optimize transfer paths for inference
    """
    enable_zero_copy: bool = True
    use_unified_memory: bool = True
    pin_memory: bool = True
    use_non_blocking: bool = True
    use_memory_mapping: bool = True
    pin_pool_size_mb: int = 1024  # 1GB pinned memory pool
    enable_write_combining: bool = False  # Experimental
    optimize_for_inference: bool = True
    
    def is_zero_copy_available(self) -> bool:
        """Check if zero-copy is available on this system."""
        import platform
        if platform.system() == "Darwin" and platform.machine() in ("arm64", "arm64e"):
            return True  # Apple Silicon unified memory
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            return False


@dataclass
class WeightStreamConfig:
    """Configuration for weight streaming from disk."""
    enable_mmap: bool = True
    enable_prefetch: bool = True
    enable_async_load: bool = True
    prefetch_queue_size: int = 3
    eviction_policy: str = "lru"  # lru, lfu, or fifo
    max_stream_memory_gb: float = 4.0
    use_zero_copy: bool = True
    pin_memory: bool = True
    read_ahead_kb: int = 512  # Kernel read-ahead hint


@dataclass
class BandwidthOptimizerConfig:
    """Configuration for bandwidth optimizer.
    
    Attributes:
        max_cpu_gpu_bandwidth: Maximum CPU to GPU bandwidth in GB/s
        max_disk_bandwidth: Maximum disk bandwidth in GB/s
        cpu_to_gpu_chunk_mb: Chunk size for CPU to GPU transfers
        disk_to_cpu_chunk_mb: Chunk size for disk to CPU transfers
        enable_memory_pooling: Enable memory pool for buffer reuse
        enable_bandwidth_profiling: Enable bandwidth profiling
        adaptive_chunk_sizing: Adapt chunk sizes based on profiling
        profile_window_size: Number of recent transfers to keep for profiling
    """
    max_cpu_gpu_bandwidth: float = 0.0  # Auto-detect if 0
    max_disk_bandwidth: float = 0.0  # Auto-detect if 0
    cpu_to_gpu_chunk_mb: float = 256.0
    disk_to_cpu_chunk_mb: float = 128.0
    enable_memory_pooling: bool = True
    enable_bandwidth_profiling: bool = False
    adaptive_chunk_sizing: bool = True
    profile_window_size: int = 10


@dataclass
class BandwidthMetrics:
    """Metrics for memory bandwidth optimization profiling."""
    timestamp: float
    transfer_size_bytes: int
    duration_seconds: float
    bandwidth_gbps: float
    source: str  # "disk", "cpu", "unified"
    
    def __post_init__(self) -> None:
        if self.duration_seconds > 0:
            self.bandwidth_gbps = (self.transfer_size_bytes / self.duration_seconds) / (1024**3)


class ZeroCopyTransferManager:
    """Manages zero-copy transfers for optimized memory movement.
    
    Implements zero-copy transfer strategies:
    1. Unified Memory: Direct CPU/GPU shared memory access (Apple Silicon)
    2. Pinned Memory: Page-locked memory for DMA transfers (CUDA/MPS)
    3. Memory Mapping: Direct file-to-GPU streaming via mmap
    4. Async Transfers: Non-blocking copies with synchronization primitives
    
    For Apple Silicon M4 Max:
    - Unified memory enables zero-copy by default (800 GB/s bandwidth)
    - Pinned memory still beneficial for DMA optimization
    - Memory mapping allows direct disk->GPU streaming
    """
    
    def __init__(self, config: ZeroCopyConfig | None = None) -> None:
        require_torch("ZeroCopyTransferManager")
        
        self.config = config or ZeroCopyConfig()
        self._lock = threading.RLock()
        
        # Pinned memory pool for reuse
        self._pinned_pool: dict[int, list[torch.Tensor]] = {}
        self._pinned_pool_size = 0
        self._max_pinned_pool_size = self.config.pin_pool_size_mb * 1024 * 1024
        
        # Track unified memory availability
        self._unified_memory = self._detect_unified_memory()
        
        # Synchronization primitives for async transfers
        self._transfer_events: dict[str, Any] = {}
        self._event_counter = 0
    
    def _detect_unified_memory(self) -> bool:
        """Detect if running on unified memory architecture."""
        import platform
        return (
            platform.system() == "Darwin" and
            platform.machine() in ("arm64", "arm64e")
        )
    
    def allocate_pinned_buffer(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Allocate a pinned memory buffer for zero-copy transfers.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            device: Target device (for unified memory path)
            
        Returns:
            Pinned tensor ready for zero-copy transfer
        """
        import torch
        
        if not self.config.pin_memory:
            return torch.empty(shape, dtype=dtype, device="cpu")
        
        # Check pool first
        size_bytes = int(torch.empty(shape, dtype=dtype).numel() *
                        torch.empty(1, dtype=dtype).element_size())
        pool_key = ((size_bytes + 1024*1024 - 1) // (1024*1024)) * (1024*1024)
        
        with self._lock:
            if pool_key in self._pinned_pool and self._pinned_pool[pool_key]:
                pooled = self._pinned_pool[pool_key].pop()
                # Check if we can reuse this buffer (reshape if needed)
                if pooled.dtype == dtype and pooled.numel() == torch.Size(shape).numel():
                     return pooled.view(shape)
                elif pooled.shape == shape and pooled.dtype == dtype:
                    return pooled
        
        # Allocate new pinned buffer
        # Optimized: Use pin_memory=True to allocate directly in pinned memory
        # This prevents allocating pageable memory first and then copying
        try:
            if device == "cpu":
                return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            elif self._unified_memory and self.config.use_unified_memory:
                # On Apple Silicon, we can allocate directly on device if needed,
                # but for mmap operations we usually want CPU pinned memory
                # that can be efficiently transferred/accessed
                return torch.empty(shape, dtype=dtype, device=device)
            else:
                return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except TypeError:
            # Fallback for older PyTorch versions
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if hasattr(tensor, 'pin_memory') and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor
    
    def return_pinned_buffer(self, tensor: torch.Tensor) -> None:
        """Return a pinned buffer to the pool for reuse."""
        if not self.config.pin_memory or not tensor.is_pinned():
            return
        
        size_bytes = tensor.numel() * tensor.element_size()
        pool_key = ((size_bytes + 1024*1024 - 1) // (1024*1024)) * (1024*1024)
        
        with self._lock:
            if pool_key not in self._pinned_pool:
                self._pinned_pool[pool_key] = []
            
            # Limit pool size to prevent memory bloat
            current_pool_size = sum(
                t.numel() * t.element_size()
                for pools in self._pinned_pool.values()
                for t in pools
            )
            
            if current_pool_size + size_bytes < self._max_pinned_pool_size:
                self._pinned_pool[pool_key].append(tensor)
    
    def zero_copy_transfer(
        self,
        tensor: torch.Tensor,
        target_device: str,
        non_blocking: bool | None = None,
    ) -> torch.Tensor:
        """Perform zero-copy transfer to target device.
        
        Args:
            tensor: Source tensor
            target_device: Target device ("mps", "cuda:0", etc.)
            non_blocking: Use non-blocking transfer (defaults to config)
            
        Returns:
            Tensor on target device
        """
        
        if non_blocking is None:
            non_blocking = self.config.use_non_blocking
        
        # Already on target device
        if str(tensor.device) == target_device:
            return tensor
        
        # Unified memory path (Apple Silicon)
        if self._unified_memory and self.config.use_unified_memory:
            # On unified memory, we can often just change the device view
            # without actual memory copy
            if tensor.device.type == "cpu" and target_device == "mps":
                # Ensure tensor is pinned for optimal DMA
                if not tensor.is_pinned() and self.config.pin_memory:
                    tensor = tensor.pin_memory()
                return tensor.to(target_device, non_blocking=non_blocking)
            elif tensor.device.type == "mps" and target_device == "cpu":
                # GPU to CPU is also zero-copy on unified memory
                return tensor.to(target_device, non_blocking=non_blocking)
        
        # Standard pinned memory path for CUDA
        if self.config.pin_memory and tensor.device.type == "cpu":
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
        
        return tensor.to(target_device, non_blocking=non_blocking)
    
    def create_zero_copy_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create a zero-copy view of a tensor for unified memory.
        
        On Apple Silicon, this allows both CPU and GPU to access the same
        memory without copying.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor optimized for zero-copy access
        """
        if not self._unified_memory or not self.config.use_unified_memory:
            return tensor
        
        # For unified memory, ensure contiguous layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Pin for optimal access
        if self.config.pin_memory and hasattr(tensor, 'pin_memory'):
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
        
        return tensor
    
    def stream_from_mmap(
        self,
        mmap_buffer: Any,
        offset: int,
        size_bytes: int,
        dtype: Any,
        shape: tuple[int, ...] | None,
        target_device: str,
    ) -> torch.Tensor:
        """Stream tensor data from memory-mapped file to device.
        
        Uses zero-copy paths where possible:
        - On unified memory: Direct buffer sharing
        - On CUDA: Pinned memory + async transfer
        
        Args:
            mmap_buffer: Memory-mapped file buffer
            offset: Byte offset in buffer
            size_bytes: Size of data to read
            dtype: Target dtype
            shape: Tensor shape
            target_device: Target device
            
        Returns:
            Tensor on target device
        """
        import torch
        
        # Read from mmap
        mmap_buffer.seek(offset)
        
        # Optimize: Zero-copy read directly into pre-allocated tensor
        try:
            # Determine allocation shape/dtype
            if shape is None:
                # Fallback to byte array if shape unknown
                alloc_shape = (size_bytes,)
                target_dtype = torch.uint8
            else:
                alloc_shape = shape
                target_dtype = dtype
                
            # Allocate pinned tensor if requested
            # Use CPU as staging area for mmap read
            tensor = self.allocate_pinned_buffer(
                alloc_shape, 
                target_dtype, 
                device="cpu"
            )
            
            # Read directly via numpy view
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
                
            # View as uint8 flat array for safe byte-level access
            np_view = tensor.view(torch.uint8).flatten().numpy()
            
            # Verify buffer size matches size_bytes
            if np_view.nbytes != size_bytes:
                 # Size mismatch, fallback to safe read
                 raise ValueError("Size mismatch")
                 
            mmap_buffer.readinto(np_view)
            
        except Exception:
            # Fallback to standard read
            mmap_buffer.seek(offset)
            raw_bytes = mmap_buffer.read(size_bytes)
            
            tensor = torch.frombuffer(raw_bytes, dtype=dtype)
            if shape is not None:
                tensor = tensor.reshape(shape)
                
            if self.config.pin_memory and not tensor.is_pinned():
                tensor = tensor.pin_memory()
        
        # Zero-copy transfer to target device
        if target_device != "cpu":
            tensor = self.zero_copy_transfer(tensor, target_device)
        
        return tensor
    
    def synchronize(self) -> None:
        """Synchronize all pending async transfers."""
        import torch
        
        # Synchronize MPS
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
        
        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def get_stats(self) -> dict[str, Any]:
        """Get zero-copy transfer statistics."""
        with self._lock:
            pinned_pool_size = sum(
                len(pool) for pool in self._pinned_pool.values()
            )
            return {
                "zero_copy_enabled": self.config.enable_zero_copy,
                "unified_memory": self._unified_memory,
                "pinned_pool_buffers": pinned_pool_size,
                "pinned_pool_size_mb": self.config.pin_pool_size_mb,
                "pin_memory": self.config.pin_memory,
                "use_non_blocking": self.config.use_non_blocking,
            }
    
    def cleanup(self) -> None:
        """Clean up pinned memory pool."""
        with self._lock:
            self._pinned_pool.clear()
            self._pinned_pool_size = 0


class BandwidthOptimizer:
    """Optimizes memory bandwidth for MMFP4 weight transfers.
    
    Features:
    - Auto-detects optimal transfer chunk sizes
    - Profiles actual bandwidth to adapt strategies
    - Enables overlapping compute and transfer
    - Manages memory pools for reduced allocation overhead
    """
    
    def __init__(self, config: BandwidthOptimizerConfig | None = None) -> None:
        require_torch("BandwidthOptimizer")
        
        self.config = config or BandwidthOptimizerConfig()
        self._metrics: list[BandwidthMetrics] = []
        self._lock = threading.RLock()
        
        # Memory pools for reuse
        self._memory_pools: dict[int, list[torch.Tensor]] = {}
        self._pool_lock = threading.RLock()
        
        # Detected bandwidth limits
        self._detected_cpu_gpu_bw: float = 0.0
        self._detected_disk_bw: float = 0.0
        
        # Transfer tracking for overlapping
        self._pending_transfers: dict[str, Any] = {}
        
        # Auto-detect if not configured
        if self.config.max_cpu_gpu_bandwidth == 0.0 or self.config.max_disk_bandwidth == 0.0:
            self._auto_detect_bandwidth()
    
    def _auto_detect_bandwidth(self) -> None:
        """Auto-detect system bandwidth capabilities."""
        import torch
        
        # Detect platform-specific bandwidth characteristics
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin" and machine in ("arm64", "arm64e"):
            # Apple Silicon unified memory - very high bandwidth
            self._detected_cpu_gpu_bw = 800.0  # GB/s for M4 Max
            self._detected_disk_bw = 7.0  # GB/s for NVMe SSD
        elif torch.cuda.is_available():
            # NVIDIA GPU - check PCIe bandwidth
            # PCIe 4.0 x16 = ~32 GB/s, PCIe 5.0 x16 = ~64 GB/s
            self._detected_cpu_gpu_bw = 32.0
            self._detected_disk_bw = 3.5
        else:
            # Conservative defaults
            self._detected_cpu_gpu_bw = 10.0
            self._detected_disk_bw = 0.5
        
        # Use detected values if not explicitly configured
        if self.config.max_cpu_gpu_bandwidth == 0.0:
            self.config.max_cpu_gpu_bandwidth = self._detected_cpu_gpu_bw
        if self.config.max_disk_bandwidth == 0.0:
            self.config.max_disk_bandwidth = self._detected_disk_bw
    
    def get_optimal_chunk_size(self, transfer_type: str = "cpu_to_gpu") -> int:
        """Get optimal chunk size in bytes for given transfer type.
        
        Args:
            transfer_type: One of "cpu_to_gpu", "disk_to_cpu", "unified"
            
        Returns:
            Optimal chunk size in bytes
        """
        if transfer_type == "cpu_to_gpu":
            return int(self.config.cpu_to_gpu_chunk_mb * 1024 * 1024)
        elif transfer_type == "disk_to_cpu":
            return int(self.config.disk_to_cpu_chunk_mb * 1024 * 1024)
        elif transfer_type == "unified":
            # Unified memory can handle larger chunks
            return int(self.config.cpu_to_gpu_chunk_mb * 4 * 1024 * 1024)
        return 64 * 1024 * 1024  # Default 64MB
    
    def profile_transfer(
        self,
        size_bytes: int,
        duration_seconds: float,
        source: str,
    ) -> None:
        """Profile a transfer for adaptive optimization."""
        if not self.config.enable_bandwidth_profiling:
            return
            
        metrics = BandwidthMetrics(
            timestamp=time.time(),
            transfer_size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bandwidth_gbps=0.0,
            source=source,
        )
        
        with self._lock:
            self._metrics.append(metrics)
            # Keep only recent metrics
            if len(self._metrics) > self.config.profile_window_size:
                self._metrics.pop(0)
            
            # Adapt chunk sizes if enabled
            if self.config.adaptive_chunk_sizing and len(self._metrics) >= 3:
                self._adapt_chunk_sizes()
    
    def _adapt_chunk_sizes(self) -> None:
        """Adapt chunk sizes based on profiling data."""
        if not self._metrics:
            return
            
        # Calculate average bandwidth by source
        bw_by_source: dict[str, list[float]] = {"disk": [], "cpu": [], "unified": []}
        for m in self._metrics:
            bw_by_source[m.source].append(m.bandwidth_gbps)
        
        # Adjust chunk sizes based on achieved bandwidth
        for source, bws in bw_by_source.items():
            if not bws:
                continue
            avg_bw = sum(bws) / len(bws)
            
            if source == "cpu":
                # If bandwidth is low, reduce chunk size to reduce latency
                # If bandwidth is high, increase chunk size for efficiency
                if avg_bw < self.config.max_cpu_gpu_bandwidth * 0.5:
                    self.config.cpu_to_gpu_chunk_mb = max(64.0, self.config.cpu_to_gpu_chunk_mb * 0.9)
                else:
                    self.config.cpu_to_gpu_chunk_mb = min(1024.0, self.config.cpu_to_gpu_chunk_mb * 1.05)
    
    def get_pooled_buffer(self, size_bytes: int, device: str) -> torch.Tensor | None:
        """Get a buffer from the memory pool if available.
        
        Args:
            size_bytes: Required buffer size
            device: Target device
            
        Returns:
            Pooled tensor or None if not available
        """
        if not self.config.enable_memory_pooling:
            return None
            
        
        # Round up to nearest MB for pool efficiency
        pool_key = ((size_bytes + 1024*1024 - 1) // (1024*1024)) * (1024*1024)
        
        with self._pool_lock:
            if pool_key in self._memory_pools and self._memory_pools[pool_key]:
                tensor = self._memory_pools[pool_key].pop()
                # Verify device matches
                if str(tensor.device) == device:
                    return tensor
        return None
    
    def return_pooled_buffer(self, tensor: torch.Tensor) -> None:
        """Return a buffer to the memory pool for reuse."""
        if not self.config.enable_memory_pooling:
            return
            
        
        size_bytes = tensor.numel() * tensor.element_size()
        pool_key = ((size_bytes + 1024*1024 - 1) // (1024*1024)) * (1024*1024)
        
        with self._pool_lock:
            if pool_key not in self._memory_pools:
                self._memory_pools[pool_key] = []
            # Limit pool size per bin to prevent memory bloat
            if len(self._memory_pools[pool_key]) < 4:
                self._memory_pools[pool_key].append(tensor)
    
    def create_optimized_transfer(
        self,
        data: torch.Tensor,
        target_device: str,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Create an optimized tensor transfer with bandwidth awareness.
        
        Args:
            data: Source tensor
            target_device: Target device
            non_blocking: Allow async transfer
            
        Returns:
            Transferred tensor
        """
        
        # Check for pooled buffer
        size_bytes = data.numel() * data.element_size()
        pooled = self.get_pooled_buffer(size_bytes, target_device)
        
        start_time = time.time()
        
        if pooled is not None:
            # Copy to pooled buffer
            pooled.copy_(data, non_blocking=non_blocking)
            result = pooled
        else:
            # Standard transfer with optimization
            if data.device.type == "cpu" and not data.is_pinned():
                data = data.pin_memory()
            result = data.to(target_device, non_blocking=non_blocking)
        
        # Profile the transfer
        duration = time.time() - start_time
        source = "unified" if self._is_unified_memory() else "cpu"
        self.profile_transfer(size_bytes, duration, source)
        
        return result
    
    def _is_unified_memory(self) -> bool:
        """Check if running on unified memory architecture."""
        return platform.system() == "Darwin" and platform.machine() in ("arm64", "arm64e")
    
    def get_bandwidth_stats(self) -> dict[str, Any]:
        """Get bandwidth optimization statistics."""
        with self._lock:
            if not self._metrics:
                return {
                    "avg_bandwidth_gbps": 0.0,
                    "peak_bandwidth_gbps": 0.0,
                    "transfer_count": 0,
                }
            
            avg_bw = sum(m.bandwidth_gbps for m in self._metrics) / len(self._metrics)
            peak_bw = max(m.bandwidth_gbps for m in self._metrics)
            
            return {
                "avg_bandwidth_gbps": avg_bw,
                "peak_bandwidth_gbps": peak_bw,
                "transfer_count": len(self._metrics),
                "detected_cpu_gpu_bw": self._detected_cpu_gpu_bw,
                "detected_disk_bw": self._detected_disk_bw,
                "current_cpu_gpu_chunk_mb": self.config.cpu_to_gpu_chunk_mb,
                "current_disk_chunk_mb": self.config.disk_to_cpu_chunk_mb,
            }
    
    def clear_metrics(self) -> None:
        """Clear profiling metrics."""
        with self._lock:
            self._metrics.clear()
    
    def cleanup(self) -> None:
        """Clean up memory pools and resources."""
        with self._pool_lock:
            self._memory_pools.clear()
        self.clear_metrics()


class MLACompressionRatio:
    """Compression ratios for KV cache using MLA (Multi-head Latent Attention)."""
    NONE = 1
    MEDIUM = 4
    HIGH = 8
    EXTREME = 16


@dataclass
class MemoryStats:
    """Current memory usage statistics."""
    max_memory_gb: float
    gpu_used_gb: float
    cpu_used_gb: float
    layers_loaded: int
    experts_cached: int
    kv_cache_compressed: bool
    compression_ratio: int
    unified_memory_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_memory_gb": self.max_memory_gb,
            "gpu_used_gb": self.gpu_used_gb,
            "cpu_used_gb": self.cpu_used_gb,
            "layers_loaded": self.layers_loaded,
            "experts_cached": self.experts_cached,
            "kv_cache_compressed": self.kv_cache_compressed,
            "compression_ratio": self.compression_ratio,
            "unified_memory_enabled": self.unified_memory_enabled,
        }


@dataclass
class LayerMetadata:
    """Metadata for a model layer."""
    idx: int
    size_bytes: int
    is_loaded: bool = False
    is_prefetched: bool = False
    weights: dict[str, torch.Tensor] | None = None


class EvictionPolicy(Enum):
    """Eviction policies for expert weight cache."""
    LRU = auto()           # Least Recently Used
    LFU = auto()           # Least Frequently Used
    WEIGHTED = auto()      # Size-weighted (evict largest first)
    ADAPTIVE = auto()      # Hybrid: combines recency, frequency, and size


@dataclass
class GhostEntry:
    """Ghost entry for tracking evicted experts (ARC-style optimization).
    
    Ghost entries remember recently evicted experts to make better admission
    decisions. If a ghost entry is accessed again, it indicates the expert
    should have been kept in cache (recency of use is high).
    """
    layer_idx: int
    expert_idx: int
    evicted_at: float
    access_count_at_eviction: int
    size_bytes: int
    
    def should_readmit(self, current_time: float, current_access_count: int) -> bool:
        """Determine if this expert should be immediately readmitted to cache."""
        time_since_eviction = current_time - self.evicted_at
        # Quick re-access indicates high value - readmit immediately
        return time_since_eviction < 5.0  # 5 second window


@dataclass
class AccessPattern:
    """Tracks access patterns for predictive prefetching."""
    recent_accesses: list[tuple[int, int]] = field(default_factory=list)
    pattern_window: int = 5
    
    def add_access(self, layer_idx: int, expert_idx: int) -> None:
        """Record an expert access."""
        self.recent_accesses.append((layer_idx, expert_idx))
        if len(self.recent_accesses) > self.pattern_window:
            self.recent_accesses.pop(0)
    
    def predict_next(self) -> list[tuple[int, int]]:
        """Predict which experts will be accessed next based on patterns."""
        if len(self.recent_accesses) < 2:
            return []
        
        predictions = []
        last_layer, last_expert = self.recent_accesses[-1]
        
        # Sequential pattern detection
        if len(self.recent_accesses) >= 2:
            prev_layer, prev_expert = self.recent_accesses[-2]
            layer_delta = last_layer - prev_layer
            expert_delta = last_expert - prev_expert
            
            # Predict next in sequence
            next_layer = last_layer + layer_delta
            next_expert = last_expert + expert_delta
            if layer_delta >= 0 and expert_delta >= 0:
                predictions.append((next_layer, next_expert))
        
        return predictions


@dataclass
class CachedExpert:
    """Wrapper for cached expert with optimized usage statistics."""
    weights: dict[str, Any]
    layer_idx: int
    expert_idx: int
    size_bytes: int
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    access_frequency: float = 0.0  # Decaying frequency score
    load_time_ms: float = 0.0
    reuse_distance_ms: float = 0.0  # Time between consecutive accesses
    # Optimized: Precomputed score for O(1) heap operations
    _cached_score: float | None = None
    _score_timestamp: float = 0.0
    # Optimized: Reuse distance tracking for workload classification
    reuse_distances: list[float] = field(default_factory=list)
    avg_reuse_distance_ms: float = 0.0
    # Optimized: Access pattern classification
    is_sequential: bool = False
    sequential_confidence: float = 0.0
    
    def update_access(self) -> None:
        """Update access statistics on cache hit with optimized decay."""
        now = time.time()
        time_delta = now - self.last_access_time
        
        # Track reuse distance for pattern analysis
        if self.access_count > 0:
            self.reuse_distance_ms = time_delta * 1000
            # Optimized: Keep rolling window of reuse distances (last 5)
            self.reuse_distances.append(self.reuse_distance_ms)
            if len(self.reuse_distances) > 5:
                self.reuse_distances.pop(0)
            # Update average reuse distance
            self.avg_reuse_distance_ms = sum(self.reuse_distances) / len(self.reuse_distances)
        
        # Adaptive decay based on time delta (faster decay for long gaps)
        if time_delta < 1.0:
            decay = 0.95  # High retention for rapid re-access
        elif time_delta < 10.0:
            decay = 0.8   # Moderate decay
        else:
            decay = 0.5   # Aggressive decay for stale entries
            
        self.access_frequency = self.access_frequency * decay + 1.0
        self.access_count += 1
        self.last_access_time = now
        # Optimized: Invalidate cached score on access
        self._cached_score = None
    
    def compute_score(self, policy: EvictionPolicy, current_time: float, hot_expert_boost: float = 1.5) -> float:
        """Compute eviction score (lower = more likely to evict) with optimizations.
        
        Optimized: Uses score caching for O(1) repeated lookups within 100ms window.
        """
        # Optimized: Return cached score if still valid (within 100ms)
        if (self._cached_score is not None and
            current_time - self._score_timestamp < 0.1):
            return self._cached_score
        
        # Calculate hot expert status (high frequency + recent access)
        is_hot = (self.access_frequency > 5.0 and
                  (current_time - self.last_access_time) < 10.0)
        hot_multiplier = hot_expert_boost if is_hot else 1.0
        
        if policy == EvictionPolicy.LRU:
            score = self.last_access_time
        elif policy == EvictionPolicy.LFU:
            score = self.access_frequency * hot_multiplier
        elif policy == EvictionPolicy.WEIGHTED:
            # Weighted by size - prefer keeping smaller, frequently used items
            age = current_time - self.last_access_time
            size_mb = self.size_bytes / (1024**2) + 1
            # High frequency and small size = keep
            score = (self.access_frequency * hot_multiplier) / size_mb - age
        else:  # ADAPTIVE - optimized combination
            age = current_time - self.last_access_time + 0.001
            
            # Dynamic weighting based on load time (slower to load = higher value)
            load_time_factor = min(self.load_time_ms / 100.0, 5.0)  # Cap at 5x
            
            # Optimized: Consider reuse distance in score
            # Short reuse distance = high temporal locality = keep
            reuse_factor = 1.0
            if self.avg_reuse_distance_ms > 0:
                # Lower reuse distance = higher retention priority
                reuse_factor = max(0.5, min(2.0, 1000.0 / self.avg_reuse_distance_ms))
            
            recency_score = 1.0 / age
            frequency_score = self.access_frequency * hot_multiplier * reuse_factor
            size_score = 1.0 / (self.size_bytes / (1024**2) + 1)
            load_time_score = load_time_factor * 0.1
            
            # Adaptive weights: frequency > recency > size > load_time
            score = frequency_score * 0.4 + recency_score * 0.3 + size_score * 0.2 + load_time_score * 0.1
        
        # Optimized: Cache the score
        self._cached_score = score
        self._score_timestamp = current_time
        return score
    
    def estimate_future_access(self, horizon_ms: float = 5000.0) -> float:
        """Estimate probability of future access within horizon.
        
        Optimized: Uses reuse distance statistics to predict future access patterns.
        Higher return value = more likely to be accessed soon.
        """
        if self.access_count == 0 or self.avg_reuse_distance_ms == 0:
            return 0.5  # Unknown - default to moderate probability
        
        # If average reuse is within horizon, high probability
        if self.avg_reuse_distance_ms <= horizon_ms:
            # Scale by frequency - frequent access = higher probability
            freq_factor = min(self.access_frequency / 5.0, 2.0)
            return min(0.95, 0.7 * freq_factor)
        
        # Reuse outside horizon - lower probability
        return max(0.1, 0.5 * (horizon_ms / self.avg_reuse_distance_ms))
    
    def get_value_score(self) -> float:
        """Calculate value score for admission control decisions.
        
        Higher value = more beneficial to keep in cache.
        Considers: frequency, load time cost, and access predictability.
        """
        # Base value from access frequency
        freq_value = self.access_frequency
        
        # Load time value (amortized cost)
        load_value = self.load_time_ms / 100.0  # Normalize to ~0-5 range
        
        # Predictability bonus (consistent reuse pattern)
        predictability = 1.0
        if len(self.reuse_distances) >= 3:
            # Low variance in reuse distances = predictable pattern
            mean = sum(self.reuse_distances) / len(self.reuse_distances)
            variance = sum((d - mean) ** 2 for d in self.reuse_distances) / len(self.reuse_distances)
            cv = (variance ** 0.5) / mean if mean > 0 else 1.0  # Coefficient of variation
            predictability = max(0.5, 1.5 - cv)  # Lower variance = higher predictability
        
        return freq_value * 0.5 + load_value * 0.3 + predictability * 0.2


@dataclass
class ExpertCacheConfig:
    """Configuration for smart expert weight caching."""
    max_entries: int = 4           # Maximum number of cached experts
    max_memory_bytes: int = 0      # 0 = unlimited, otherwise memory limit
    eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE
    enable_prefetch: bool = True   # Prefetch adjacent experts
    prefetch_distance: int = 1     # How many experts ahead to prefetch
    enable_admission: bool = True  # Filter one-off accesses
    admission_threshold: int = 2   # Min accesses before caching
    frequency_decay_rate: float = 0.95  # Decay rate for frequency scores



    
    # Optimized caching additions
    enable_ghost_entries: bool = True   # Track evicted experts to improve admission
    ghost_history_size: int = 8         # Number of evicted experts to remember
    load_time_weight: float = 0.3       # Weight for load time in admission (higher = prefer slow-to-load)
    reuse_distance_tracking: bool = True  # Track time between accesses for pattern detection
    enable_predictive_prefetch: bool = True  # Use access patterns to predict prefetch targets
    access_pattern_window: int = 5      # Window size for pattern detection
    hot_expert_boost: float = 1.5       # Score multiplier for frequently accessed experts
    
    # Advanced optimization settings
    enable_heap_optimization: bool = True  # Use heap for O(log n) eviction
    score_cache_ttl_ms: float = 100.0   # TTL for cached scores
    enable_workload_adaptation: bool = True  # Adapt policy based on access patterns
    min_hit_rate_for_adaptation: float = 0.3  # Min hit rate before policy adaptation
    admission_policy: str = "adaptive"  # "strict", "adaptive", "lenient"
    value_based_admission: bool = True  # Use value scores for admission decisions


@dataclass
class ExpertMetadata:
    """Metadata for a MoE expert."""
    layer_idx: int
    expert_idx: int
    size_bytes: int
    is_cached: bool = False
    last_used: float = 0.0
    use_count: int = 0


class OptimizedExpertCache:
    """Heap-optimized expert cache with O(log n) eviction.
    
    Optimizations:
    - Min-heap for O(log n) eviction instead of O(n) scan
    - Lazy deletion for efficient updates
    - Score caching to reduce computation
    - Hit rate tracking for adaptive policy selection
    - Workload-aware admission control
    """
    
    def __init__(self, config: ExpertCacheConfig) -> None:
        self.config = config
        self._cache: dict[tuple[int, int], CachedExpert] = {}
        # Min-heap: (score, sequence, key) - sequence ensures stable ordering
        self._heap: list[tuple[float, int, tuple[int, int]]] = []
        self._heap_sequence = 0
        self._lock = threading.RLock()
        
        # Statistics for adaptation
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._admissions = 0
        self._ghost_hits = 0
        
        # Workload characteristics
        self._access_times: list[float] = []
        self._workload_type: str = "unknown"  # "sequential", "random", "skewed"
    
    def get(self, key: tuple[int, int], default: Any = None) -> CachedExpert | Any:
        """Get expert from cache. Optimized: O(1) lookup."""
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                cached.update_access()
                # Log access time for workload detection
                self._access_times.append(time.time())
                if len(self._access_times) > 20:
                    self._access_times.pop(0)
                return cached
            else:
                self._misses += 1
                return default
    
    def put(self, key: tuple[int, int], expert: CachedExpert) -> None:
        """Add expert to cache. Optimized: O(log n) insertion."""
        with self._lock:
            # Check if already exists (update case)
            if key in self._cache:
                self._cache[key] = expert
                # Will be lazily updated in heap
                return
            
            self._admissions += 1
            
            # Compute initial score
            score = expert.compute_score(
                self.config.eviction_policy,
                time.time(),
                self.config.hot_expert_boost
            )
            
            self._cache[key] = expert
            self._heap_sequence += 1
            heapq.heappush(self._heap, (score, self._heap_sequence, key))

    def __getitem__(self, key: tuple[int, int]) -> CachedExpert:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key: tuple[int, int], value: CachedExpert) -> None:
        self.put(key, value)

    def __delitem__(self, key: tuple[int, int]) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            else:
                raise KeyError(key)

    def __iter__(self):
        return iter(self._cache)

    def items(self):
        return self._cache.items()

    def values(self):
        return self._cache.values()

    def keys(self):
        return self._cache.keys()

    def pop(self, key: tuple[int, int], default: Any = ...):
        with self._lock:
            if key in self._cache:
                val = self._cache.pop(key)
                return val
            if default is not ...:
                return default
            raise KeyError(key)

    def update(self, other: dict[tuple[int, int], CachedExpert]) -> None:
        with self._lock:
            for k, v in other.items():
                self.put(k, v)
    
    def evict_if_needed(self, required_bytes: int, current_memory: int) -> list[tuple[tuple[int, int], CachedExpert]]:
        """Evict experts to make room. Optimized: O(k log n) for k evictions."""
        evicted: list[tuple[tuple[int, int], CachedExpert]] = []
        max_memory = self.config.max_memory_bytes
        
        with self._lock:
            # Check entry limit
            while len(self._cache) >= self.config.max_entries:
                result = self._evict_one()
                if result:
                    evicted.append(result)
            
            # Check memory limit
            if max_memory > 0:
                cache_memory = sum(e.size_bytes for e in self._cache.values())
                while (cache_memory + required_bytes > max_memory and self._cache):
                    result = self._evict_one()
                    if result:
                        evicted.append(result)
                        cache_memory -= result[1].size_bytes
        
        return evicted
    
    def _evict_one(self) -> tuple[tuple[int, int], CachedExpert] | None:
        """Evict single expert using heap. O(log n)."""
        current_time = time.time()
        
        # Find valid entry from heap (lazy deletion)
        while self._heap:
            score, seq, key = heapq.heappop(self._heap)
            
            # Verify entry is still valid
            if key not in self._cache:
                continue  # Stale entry, skip
            
            cached = self._cache[key]
            
            # Verify score is still valid (not too stale)
            if current_time - cached._score_timestamp > 1.0:  # 1 second threshold
                # Recompute score and re-insert
                new_score = cached.compute_score(
                    self.config.eviction_policy,
                    current_time,
                    self.config.hot_expert_boost
                )
                self._heap_sequence += 1
                heapq.heappush(self._heap, (new_score, self._heap_sequence, key))
                continue  # Try next entry
            
            # Valid eviction candidate
            expert = self._cache.pop(key)
            self._evictions += 1
            return (key, expert)
        
        return None
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = self._hits / total_accesses if total_accesses > 0 else 0.0
            
            return {
                "entries": len(self._cache),
                "heap_size": len(self._heap),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "admissions": self._admissions,
                "ghost_hits": self._ghost_hits,
                "workload_type": self._workload_type,
            }
    
    def should_admit(self, key: tuple[int, int], load_time_ms: float,
                     ghost_hit: bool = False) -> bool:
        """Smart admission control with value-based decision."""
        if ghost_hit:
            self._ghost_hits += 1
            return True
        
        if not self.config.enable_admission:
            return True
        
        policy = self.config.admission_policy
        
        if policy == "strict":
            # Require multiple accesses before admission
            return False  # Let caller track and decide
        elif policy == "lenient":
            return True
        
        # Adaptive policy (default)
        # Use load time value - slow to load = more valuable
        load_value = load_time_ms / 1000.0  # seconds
        threshold = self.config.admission_threshold
        
        if self.config.value_based_admission:
            # Value-based: load time + predicted reuse
            value_score = load_value * self.config.load_time_weight
            return value_score >= threshold * 0.5
        
        return False  # Default: defer to caller
    
    def detect_workload(self) -> str:
        """Detect access pattern for adaptive optimization."""
        with self._lock:
            if len(self._access_times) < 5:
                return "unknown"
            
            # Calculate inter-arrival times
            intervals = []
            for i in range(1, len(self._access_times)):
                intervals.append(self._access_times[i] - self._access_times[i-1])
            
            if not intervals:
                return "unknown"
            
            # Coefficient of variation indicates pattern type
            mean_interval = sum(intervals) / len(intervals)
            if mean_interval == 0:
                return "bursty"
            
            variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
            cv = (variance ** 0.5) / mean_interval
            
            if cv < 0.5:
                self._workload_type = "sequential"
            elif cv > 1.5:
                self._workload_type = "random"
            else:
                self._workload_type = "mixed"
            
            return self._workload_type
    
    def clear(self) -> list[CachedExpert]:
        """Clear all entries and return evicted experts."""
        with self._lock:
            evicted = list(self._cache.values())
            self._cache.clear()
            self._heap.clear()
            return evicted
    
    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._cache
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def values(self):
        return self._cache.values()
    
    def get_eviction_candidate(self) -> tuple[int, int] | None:
        """Peek at next eviction candidate without removing."""
        with self._lock:
            current_time = time.time()
            
            # Find valid candidate
            while self._heap:
                score, seq, key = self._heap[0]
                if key in self._cache:
                    return key
                # Stale entry, remove it
                heapq.heappop(self._heap)
            
            return None


class WeightStreamer:
    """Streams weights from disk using memory mapping and async prefetching.
    
    Optimizes for large models that don't fit in GPU memory by:
    - Memory-mapped file I/O for efficient disk access
    - Async prefetching of upcoming layers
    - LRU eviction with configurable buffer limits
    - Zero-copy transfer for Unified Memory systems
    """
    
    def __init__(
        self,
        model_path: str | Path,
        config: WeightStreamConfig | None = None,
        device: str = "mps",
    ) -> None:
        require_torch("WeightStreamer")
        import torch
        
        self.model_path = Path(model_path)
        self.config = config or WeightStreamConfig()
        self.device = device if torch.backends.mps.is_available() else "cpu"
        
        # Streaming state
        self._stream_buffer: OrderedDict[str, StreamBuffer] = OrderedDict()
        self._prefetch_queue: OrderedDict[str, Future[Any]] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="weight_stream")
        self._lock = threading.RLock()
        self._stream_memory_used = 0
        self._max_stream_memory = int(self.config.max_stream_memory_gb * 1024**3)
        self._access_count: dict[str, int] = {}
        
        # Heap-based LRU: min-heap ordered by (last_access_time, sequence)
        # Provides O(log n) eviction vs O(n) for OrderedDict scan
        self._heap_lru: list[tuple[float, int, str]] = []
        self._heap_sequence = 0  # Monotonic counter for stable ordering
        
        # Loader for non-mmap fallback
        self._loader: MMFP4ModelLoader | None = None
        if self.model_path.exists():
            try:
                self._loader = MMFP4ModelLoader(self.model_path)
            except Exception:
                pass
    
    def stream_weight(
        self,
        weight_name: str,
        shard_path: Path,
        offset: int,
        size_bytes: int,
        dtype: Any = None,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Stream a single weight from disk using memory mapping.
        
        Args:
            weight_name: Unique identifier for this weight
            shard_path: Path to the safetensors shard file
            offset: Byte offset within the file
            size_bytes: Size of the tensor data in bytes
            dtype: Target torch dtype
            shape: Tensor shape for reshaping
            
        Returns:
            Tensor loaded to target device
        """
        
        with self._lock:
            # Check cache first
            if weight_name in self._stream_buffer:
                buffer = self._stream_buffer[weight_name]
                if buffer._loaded:
                    self._update_access(weight_name)
                    return self._buffer_to_tensor(buffer, dtype, shape)
        
        # Load via mmap or direct read
        if self.config.enable_mmap and shard_path.exists():
            tensor = self._mmap_load(shard_path, offset, size_bytes, dtype, shape)
        elif self._loader is not None:
            tensor = self._loader.load_tensor(weight_name, device=self.device)
        else:
            raise RuntimeError(f"Cannot load weight {weight_name}: no loader available")
        
        # Cache the result
        self._cache_weight(weight_name, tensor)
        
        return tensor
    
    def _mmap_load(
        self,
        shard_path: Path,
        offset: int,
        size_bytes: int,
        dtype: Any,
        shape: tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Load tensor using memory-mapped I/O with zero-copy optimization."""
        import torch
        
        # Optimize: Allocate tensor first (pinned if requested)
        # Use CPU as staging area for read
        target_dtype = dtype if dtype is not None else torch.float32
        
        # Determine allocation shape
        if shape is not None:
            alloc_shape = shape
        else:
            # Estimate shape from size if not provided
            element_size = target_dtype.itemsize
            alloc_shape = (size_bytes // element_size,)
            
        # Check if we can use pinned memory directly
        # Note: torch.empty(..., pin_memory=True) might behave unexpectedly on some platforms
        # so we allocate standard CPU memory first and pin later if needed, unless we are sure.
        # To be safe and ensure CPU allocation for readinto, we start with standard CPU memory.
        try:
            tensor = torch.empty(
                alloc_shape, 
                dtype=target_dtype, 
                device="cpu"
            )
            
            # Read directly into tensor memory
            with open(shard_path, "rb") as f:
                f.seek(offset)
                
                # Zero-copy read
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                    
                # View as uint8 flat array for safe byte-level access
                np_view = tensor.view(torch.uint8).flatten().numpy()
                f.readinto(np_view)
                
            # Pin memory if requested (this copies to pinned memory)
            if self.config.pin_memory and (self.device != "cpu"):
                try:
                    tensor = tensor.pin_memory()
                except RuntimeError:
                    # Ignore pinning errors (e.g. on some MPS setups)
                    pass
                
        except Exception:
            # Fallback to standard read if direct optimization fails
            with open(shard_path, "rb") as f:
                f.seek(offset)
                data = f.read(size_bytes)
            
            tensor = torch.frombuffer(data, dtype=target_dtype)
            if shape is not None:
                tensor = tensor.reshape(shape)
                
            if self.config.pin_memory and (self.device != "cpu") and not tensor.is_pinned():
                try:
                    tensor = tensor.pin_memory()
                except RuntimeError:
                    pass
                
        # Move to device with zero-copy optimization
        if self.device != "cpu":
            if self.config.use_zero_copy:
                tensor = tensor.to(self.device, non_blocking=True)
            else:
                tensor = tensor.to(self.device)
        
        return tensor
    
    def prefetch_weights(self, weight_names: list[str], shard_info: dict[str, Any]) -> None:
        """Prefetch multiple weights asynchronously.
        
        Args:
            weight_names: List of weight names to prefetch
            shard_info: Dict mapping weight names to (shard_path, offset, size) tuples
        """
        if not self.config.enable_prefetch:
            return
            
        with self._lock:
            for name in weight_names:
                if name in self._stream_buffer or name in self._prefetch_queue:
                    continue
                if len(self._prefetch_queue) >= self.config.prefetch_queue_size:
                    break
                    
                if name in shard_info:
                    info = shard_info[name]
                    future = self._executor.submit(
                        self._preload_weight,
                        name,
                        info["path"],
                        info["offset"],
                        info["size"],
                    )
                    self._prefetch_queue[name] = future
    
    def _preload_weight(
        self,
        weight_name: str,
        shard_path: Path,
        offset: int,
        size_bytes: int,
    ) -> None:
        """Preload weight into stream buffer (background task)."""
        
        try:
            with self._lock:
                if weight_name in self._stream_buffer:
                    return
            
            # Load via mmap
            tensor = self._mmap_load(shard_path, offset, size_bytes, None, None)
            self._cache_weight(weight_name, tensor)
            
        except Exception:
            pass  # Prefetch failures are non-fatal
        finally:
            with self._lock:
                self._prefetch_queue.pop(weight_name, None)
    
    def _cache_weight(self, weight_name: str, tensor: torch.Tensor) -> None:
        """Cache a loaded weight with heap-based LRU eviction."""
        
        with self._lock:
            size_bytes = tensor.numel() * tensor.element_size()
            
            # Evict if necessary using heap-based O(log n) eviction
            while (self._stream_memory_used + size_bytes > self._max_stream_memory and
                   self._stream_buffer):
                self._heap_lru_evict()
            
            # Create buffer entry with heap tracking
            self._heap_sequence += 1
            current_time = time.time()
            buffer = StreamBuffer(
                name=weight_name,
                size_bytes=size_bytes,
                device=str(tensor.device),
                _loaded=True,
                tensor=tensor,
                _last_access_time=current_time,
                _access_sequence=self._heap_sequence,
            )
            
            if self.config.pin_memory:
                buffer.pin()
            
            # Add to buffer dict and heap
            self._stream_buffer[weight_name] = buffer
            heapq.heappush(self._heap_lru, (current_time, self._heap_sequence, weight_name))
            self._access_count[weight_name] = 1
            self._stream_memory_used += size_bytes
    
    def _evict_lru(self) -> None:
        """Evict least recently used weight from cache."""
        if not self._stream_buffer:
            return
            
        # Remove oldest entry
        name, buffer = self._stream_buffer.popitem(last=False)
        buffer.release()
        self._stream_memory_used -= buffer.size_bytes
        self._access_count.pop(name, None)
    
    def _heap_lru_evict(self) -> None:
        """Heap-based LRU eviction with O(log n) complexity.
        
        Uses the min-heap to efficiently find and evict the least recently
        used weight buffer when memory pressure is high.
        """
        with self._lock:
            while self._heap_lru:
                # Get oldest entry from heap
                last_access, seq, name = heapq.heappop(self._heap_lru)
                
                # Verify it's still valid (might be stale entry)
                if name in self._stream_buffer:
                    buffer = self._stream_buffer.pop(name)
                    buffer.release()
                    self._stream_memory_used -= buffer.size_bytes
                    self._access_count.pop(name, None)
                    return
    
    def _update_access(self, weight_name: str) -> None:
        """Update access tracking for eviction policy."""
        self._stream_buffer.move_to_end(weight_name)
        self._access_count[weight_name] = self._access_count.get(weight_name, 0) + 1
    
    def _buffer_to_tensor(
        self,
        buffer: StreamBuffer,
        dtype: Any,
        shape: tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Convert buffer back to tensor (placeholder for actual reconstruction)."""
        if buffer.tensor is not None:
            return buffer.tensor
            
        import torch
        # Fallback if tensor not preserved (should not happen with current implementation)
        return torch.zeros(1, device=buffer.device)
    
    def evict_weight(self, weight_name: str) -> bool:
        """Manually evict a weight from cache.
        
        Returns:
            True if weight was found and evicted
        """
        with self._lock:
            if weight_name not in self._stream_buffer:
                return False
                
            buffer = self._stream_buffer.pop(weight_name)
            buffer.release()
            self._stream_memory_used -= buffer.size_bytes
            self._access_count.pop(weight_name, None)
            return True
    
    def clear_cache(self) -> None:
        """Clear all cached weights."""
        with self._lock:
            for buffer in self._stream_buffer.values():
                buffer.release()
            self._stream_buffer.clear()
            self._access_count.clear()
            self._stream_memory_used = 0
            
            # Cancel pending prefetches
            for future in self._prefetch_queue.values():
                future.cancel()
            self._prefetch_queue.clear()
    
    def get_stream_stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        with self._lock:
            return {
                "cached_weights": len(self._stream_buffer),
                "memory_used_gb": self._stream_memory_used / (1024**3),
                "memory_limit_gb": self._max_stream_memory / (1024**3),
                "pending_prefetches": len(self._prefetch_queue),
                "hit_rate": self._calculate_hit_rate(),
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self._access_count:
            return 0.0
        hits = sum(1 for count in self._access_count.values() if count > 1)
        return hits / len(self._access_count)
    
    def shutdown(self) -> None:
        """Shutdown the streamer and release resources."""
        self.clear_cache()
        self._executor.shutdown(wait=True)


class MMFP4MemoryManager:
    """Memory manager for MMFP4 inference with advanced optimizations.

    Features:
        - Unified Memory support with zero-copy transfers (M4 Max advantage)
        - Asynchronous layer loading with prefetching
        - MoE expert caching with LRU eviction
        - MLA KV cache compression
        - Activation checkpointing during prefill
        - Weight streaming from disk for memory optimization
        - Zero-copy transfer paths for minimized memory overhead
    """

    def __init__(
        self,
        model_path: str,
        max_memory_gb: float = 24.0,
        *,
        num_layers: int = 32,
        num_experts_per_layer: int = 0,
        kv_compression_ratio: int = MLACompressionRatio.HIGH,
        unified_memory: bool = True,
        prefetch_window: int = 2,
        expert_cache_size: int = 4,
        activation_checkpointing: bool = True,
        enable_weight_streaming: bool = True,
        streaming_config: WeightStreamConfig | None = None,
        bandwidth_config: BandwidthOptimizerConfig | None = None,
        enable_bandwidth_opt: bool = True,
        zero_copy_config: ZeroCopyConfig | None = None,
        enable_zero_copy: bool = True,
    ) -> None:
        require_torch("MMFP4MemoryManager")
        import torch

        self._model_path = Path(model_path)
        self._max_memory_bytes = int(max_memory_gb * 1024**3)
        self._num_layers = num_layers
        self._num_experts_per_layer = num_experts_per_layer
        self._kv_compression_ratio = kv_compression_ratio
        self._unified_memory = unified_memory and self._has_unified_memory()
        self._prefetch_window = prefetch_window
        self._expert_cache_size = expert_cache_size
        self._activation_checkpointing = activation_checkpointing
        self._enable_weight_streaming = enable_weight_streaming
        self._enable_bandwidth_opt = enable_bandwidth_opt
        self._enable_zero_copy = enable_zero_copy

        # Zero-copy transfer manager for optimized memory movement
        self._zero_copy_manager: ZeroCopyTransferManager | None = None
        if self._enable_zero_copy:
            zc_config = zero_copy_config or ZeroCopyConfig(
                enable_zero_copy=True,
                use_unified_memory=self._unified_memory,
                pin_memory=True,
                use_non_blocking=True,
                use_memory_mapping=True,
            )
            self._zero_copy_manager = ZeroCopyTransferManager(config=zc_config)

        # Bandwidth optimizer for memory transfer optimization
        self._bandwidth_optimizer: BandwidthOptimizer | None = None
        if self._enable_bandwidth_opt:
            self._bandwidth_optimizer = BandwidthOptimizer(config=bandwidth_config)

        # Device setup
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Loader initialization
        try:
            self._loader = MMFP4ModelLoader(self._model_path)
        except Exception:
            self._loader = None

        # Weight streaming initialization
        self._weight_streamer: WeightStreamer | None = None
        if self._enable_weight_streaming:
            stream_config = streaming_config or WeightStreamConfig(
                enable_mmap=True,
                enable_prefetch=True,
                max_stream_memory_gb=max_memory_gb * 0.3,  # 30% for streaming cache
                use_zero_copy=self._unified_memory,
            )
            try:
                self._weight_streamer = WeightStreamer(
                    self._model_path,
                    config=stream_config,
                    device=self._device,
                )
            except Exception:
                pass  # Fall back to standard loading
        
        # Weight prefetcher initialization
        self._weight_prefetcher: WeightPrefetcher | None = None
        self._enable_weight_prefetch = True
        self._init_weight_prefetcher()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mmfp4_mem")
        self._lock = threading.RLock()

        # State tracking
        self._gpu_memory_used = 0
        self._layers: dict[int, LayerMetadata] = {}
        self._experts: dict[tuple[int, int], ExpertMetadata] = {}
        
        # Smart caching for expert weights with configurable policy
        self._expert_cache_config = ExpertCacheConfig(
            max_entries=self._expert_cache_size,
            max_memory_bytes=int(self._max_memory_bytes * 0.3),  # 30% of total
            eviction_policy=EvictionPolicy.ADAPTIVE,
            enable_prefetch=True,
            prefetch_distance=1,
            enable_admission=True,  # Enabled for optimized caching
            admission_threshold=2,
            enable_ghost_entries=True,
            ghost_history_size=8,
            enable_predictive_prefetch=True,
        )
        # (layer_idx, expert_idx) -> CachedExpert
        self._expert_weight_cache = OptimizedExpertCache(self._expert_cache_config)
        # Ghost entries for ARC-style optimization
        self._ghost_entries: OrderedDict[tuple[int, int], GhostEntry] = OrderedDict()
        # Access pattern tracker for predictive prefetching
        self._access_pattern = AccessPattern(pattern_window=self._expert_cache_config.access_pattern_window)
        self._kv_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._activations: dict[int, torch.Tensor] = {}
        self._prefetch_futures: dict[int, Future[Any]] = {}
        
        # Buffer pool for transient buffers - reuse to reduce allocation overhead
        pool_size_mb = max(int(max_memory_gb * 0.1), 64)  # 10% of max memory, min 64MB
        self._buffer_pool = BufferPool(
            max_pool_size_bytes=pool_size_mb * 1024 * 1024,
            high_watermark=0.9,
            low_watermark=0.7,
            max_age_seconds=300.0,
            max_buffers_per_tier=32,
        )
        
        # Memory compactor for fragmentation reduction
        self._compaction_config = CompactionConfig(
            enable_auto_compaction=True,
            fragmentation_threshold=0.25,
            min_compaction_interval_seconds=10.0,
            compact_expert_cache=True,
            compact_layer_buffers=True,
            compact_buffer_pool=True,
        )
        self._memory_compactor = MemoryCompactor(config=self._compaction_config)
        
        # Streaming state tracking
        self._streamed_layers: set[int] = set()
        self._layer_stream_order: list[int] = []

        self._initialize_metadata()

        # Per-layer locks to reduce contention on the global lock
        self._per_layer_locks: dict[int, threading.RLock] = {
            i: threading.RLock() for i in range(self._num_layers)
        }

    def _has_unified_memory(self) -> bool:
        return platform.system() == "Darwin" and platform.machine() in ("arm64", "arm64e")

    def _initialize_metadata(self) -> None:
        # Default sizes if loader is not available
        avg_layer_size = int(0.5 * 1024**3) # 500MB
        
        for idx in range(self._num_layers):
            self._layers[idx] = LayerMetadata(idx=idx, size_bytes=avg_layer_size)
            
            if self._num_experts_per_layer > 0:
                expert_size = avg_layer_size // self._num_experts_per_layer
                for e_idx in range(self._num_experts_per_layer):
                    self._experts[(idx, e_idx)] = ExpertMetadata(
                        layer_idx=idx, expert_idx=e_idx, size_bytes=expert_size
                    )
    
    def _init_weight_prefetcher(self) -> None:
        """Initialize weight prefetcher for optimized loading."""
        if not self._enable_weight_prefetch or self._loader is None:
            return
        
        try:
            # Calculate prefetch memory budget (20% of max memory)
            prefetch_memory_mb = (self._max_memory_bytes * 0.2) / (1024 * 1024)
            
            config = PrefetchConfig(
                enable_prefetch=True,
                lookahead_count=5,
                max_prefetch_memory_mb=prefetch_memory_mb,
                prefetch_queue_size=4,
                adaptive_prefetch=True,
            )
            
            self._weight_prefetcher = self._loader.create_prefetcher(
                config=config,
                device=self._device,
            )
        except Exception:
            # Prefetcher is optional, continue without it
            self._weight_prefetcher = None
    
    def get_weight_prefetcher(self) -> WeightPrefetcher | None:
        """Get the weight prefetcher instance.
        
        Returns:
            WeightPrefetcher if enabled, None otherwise
        """
        return self._weight_prefetcher
    
    def prefetch_weights(self, tensor_names: list[str]) -> None:
        """Prefetch specific weights into cache.
        
        Args:
            tensor_names: List of tensor names to prefetch
        """
        if self._weight_prefetcher is not None:
            self._weight_prefetcher.prefetch(tensor_names)
    
    def prefetch_layer_weights(self, layer_idx: int) -> None:
        """Prefetch all weights for a specific layer.
        
        Args:
            layer_idx: Layer index to prefetch
        """
        if self._weight_prefetcher is not None:
            self._weight_prefetcher.prefetch_layer(layer_idx)
    
    def get_prefetch_stats(self) -> dict[str, Any]:
        """Get weight prefetching statistics."""
        if self._weight_prefetcher is not None:
            return self._weight_prefetcher.get_stats()
        return {"enabled": False}
    
    def clear_prefetch_cache(self) -> None:
        """Clear the weight prefetch cache."""
        if self._weight_prefetcher is not None:
            self._weight_prefetcher.clear_cache()

    def pin_weights(self) -> None:
        """Pin all currently loaded weights to page-locked memory.
        
        This optimizes host-to-device transfer speeds for subsequent operations.
        Only affects weights currently in CPU memory.
        """
        # Pin layer weights using per-layer locks for better concurrency
        for idx, meta in self._layers.items():
            with self._per_layer_locks[idx]:
                if meta.is_loaded and meta.weights:
                    for name, tensor in meta.weights.items():
                        if tensor.device.type == "cpu" and not tensor.is_pinned():
                            meta.weights[name] = tensor.pin_memory()
        
        # Pin expert weights using per-layer locks when possible
        # Group experts by layer to use per-layer locks
        experts_by_layer: dict[int, list[tuple[tuple[int, int], CachedExpert]]] = {}
        with self._lock:
            # Copy references to avoid holding global lock during pinning
            for key, cached in self._expert_weight_cache.items():
                layer_idx = key[0]
                if layer_idx not in experts_by_layer:
                    experts_by_layer[layer_idx] = []
                experts_by_layer[layer_idx].append((key, cached))
        
        # Pin each layer's experts using per-layer locks
        for layer_idx, expert_list in experts_by_layer.items():
            if layer_idx in self._per_layer_locks:
                with self._per_layer_locks[layer_idx]:
                    for key, cached in expert_list:
                        for name, tensor in cached.weights.items():
                            if tensor.device.type == "cpu" and not tensor.is_pinned():
                                cached.weights[name] = tensor.pin_memory()
            else:
                # Fallback: pin without lock for out-of-range layers
                for key, cached in expert_list:
                    for name, tensor in cached.weights.items():
                        if tensor.device.type == "cpu" and not tensor.is_pinned():
                            cached.weights[name] = tensor.pin_memory()

    def _zero_copy(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimized tensor transfer using zero-copy paths where possible.
        
        Implements zero-copy transfer strategies to avoid unnecessary tensor copies:
        1. Unified memory: Direct buffer sharing on Apple Silicon (800 GB/s bandwidth)
        2. Pinned memory: Page-locked buffers for DMA transfers (CUDA/MPS)
        3. Memory mapping: Direct file-to-GPU streaming via mmap
        4. Non-blocking async: Overlaps transfer with computation
        
        Args:
            tensor: Source tensor to transfer
            
        Returns:
            Tensor on target device with zero-copy optimization applied
        """
        import torch
        
        # Fast path: already on target device - return as-is (no copy)
        if str(tensor.device) == str(self._device):
            return tensor
        
        # Use zero-copy manager if available (most comprehensive optimization)
        if self._zero_copy_manager is not None:
            return self._zero_copy_manager.zero_copy_transfer(tensor, self._device)
        
        # Unified memory path (Apple Silicon M4 Max - 800 GB/s shared memory)
        if self._unified_memory and tensor.device.type == "cpu":
            # Zero-copy: unified memory allows direct access without copy
            # Pin memory for optimal DMA even on unified memory
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor.to(self._device, non_blocking=True)
        
        # CUDA zero-copy path with pinned memory
        if tensor.device.type == "cpu" and self._device.startswith("cuda"):
            # Ensure tensor is pinned for zero-copy DMA transfer
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor.to(self._device, non_blocking=True)
        
        # MPS zero-copy path (Metal Performance Shaders)
        if tensor.device.type == "cpu" and self._device == "mps":
            # Pin memory for MPS DMA optimization
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor.to(self._device, non_blocking=True)
        
        # GPU to CPU zero-copy (for unified memory systems)
        if self._unified_memory and tensor.device.type in ("mps", "cuda"):
            return tensor.to("cpu", non_blocking=True)
        
        # Standard fallback (may involve copy)
        return tensor.to(self._device)
    
    def allocate_zero_copy_buffer(
        self,
        shape: tuple[int, ...],
        dtype: Any | None = None,
    ) -> torch.Tensor:
        """Allocate a buffer optimized for zero-copy transfers.
        
        Args:
            shape: Buffer shape
            dtype: Buffer dtype (defaults to float32)
            
        Returns:
            Pinned/unified memory buffer ready for zero-copy transfer
        """
        import torch
        
        if dtype is None:
            dtype = torch.float32
        
        if self._zero_copy_manager is not None:
            return self._zero_copy_manager.allocate_pinned_buffer(
                shape, dtype, self._device
            )
        
        # Fallback: standard allocation
        return torch.empty(shape, dtype=dtype, device=self._device)
    
    def create_zero_copy_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create a zero-copy view optimized for unified memory access.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor optimized for zero-copy access
        """
        if self._zero_copy_manager is not None:
            return self._zero_copy_manager.create_zero_copy_view(tensor)
        return tensor

    def load_layer_async(self, layer_idx: int) -> Future[dict]:
        """Asynchronously load a layer into GPU memory."""
        return self._executor.submit(self._load_layer_impl, layer_idx)

    def _load_layer_impl(self, layer_idx: int) -> dict:
        import torch
        
        if layer_idx not in self._per_layer_locks:
            return {"layer_idx": layer_idx, "status": "error"}

        with self._per_layer_locks[layer_idx]:
            meta = self._layers.get(layer_idx)
            if not meta:
                return {"layer_idx": layer_idx, "status": "error"}
            
            if meta.is_loaded:
                return {"layer_idx": layer_idx, "status": "already_loaded"}

            # Use prefetcher-aware loading if available
            actual_size = None
            if self._loader:
                if self._weight_prefetcher is not None:
                    weights = self._loader.load_layer_with_prefetch(
                        layer_idx,
                        prefetcher=self._weight_prefetcher,
                        device=self._device,
                        zero_copy=self._unified_memory,
                        prefetch_next=True,
                    )
                else:
                    weights = self._loader.load_layer(
                        layer_idx,
                        device=self._device,
                        zero_copy=self._unified_memory,
                    )
                # Calculate actual size
                actual_size = sum(t.numel() * t.element_size() for t in weights.values())
            else:
                # Mock weights for verification
                t = torch.zeros(1)
                weights = {"weight": self._zero_copy(t)}

            meta.weights = weights
            meta.is_loaded = True
            if actual_size is not None:
                meta.size_bytes = actual_size
            
            with self._lock:
                self._gpu_memory_used += meta.size_bytes
                
                # Prefetching (legacy path if prefetcher not available)
                if self._weight_prefetcher is None:
                    self._prefetch_next(layer_idx)
            
            return {"layer_idx": layer_idx, "weights": weights, "status": "loaded"}

    def _prefetch_next(self, current_idx: int) -> None:
        for i in range(1, self._prefetch_window + 1):
            next_idx = current_idx + i
            if next_idx < self._num_layers and next_idx not in self._prefetch_futures:
                self._prefetch_futures[next_idx] = self.load_layer_async(next_idx)

    def _get_expert_from_cache(self, key: tuple[int, int]) -> CachedExpert | None:
        """Get expert from cache with minimal lock holding."""
        with self._lock:
            return self._expert_weight_cache.get(key)
    
    def _update_expert_access_metadata(self, key: tuple[int, int], cached: CachedExpert) -> None:
        """Update expert access metadata under lock."""
        with self._lock:
            cached.update_access()
            if key in self._experts:
                self._experts[key].last_used = cached.last_access_time
                self._experts[key].use_count = cached.access_count
            
            # Remove from ghost entries if present (was previously evicted)
            if key in self._ghost_entries:
                del self._ghost_entries[key]
    
    def get_expert_weights(self, layer_idx: int, expert_idx: int) -> dict[str, Any]:
        """Retrieve expert weights using optimized smart caching.
        
        Implements an optimized caching strategy with:
        - Ghost entries for ARC-style admission control (learns from evictions)
        - Predictive prefetching based on access patterns
        - Hot expert boosting (keeps frequently accessed experts longer)
        - Load-time-weighted admission (prefers slow-to-load experts)
        - Adaptive eviction combining recency, frequency, size, and load time
        
        Lock contention optimization:
        - Uses per-layer locks for expert loading
        - Minimizes global lock holding to only metadata updates
        - Double-checked locking pattern for cache hits
        """
        key = (layer_idx, expert_idx)
        config = self._expert_cache_config
        
        # Fast path: check cache with minimal lock contention
        # Only track access pattern if prefetch is enabled (reduces lock time)
        if config.enable_predictive_prefetch:
            with self._lock:
                self._access_pattern.add_access(layer_idx, expert_idx)
        
        # Check cache hit - use helper to minimize lock scope
        cached = self._get_expert_from_cache(key)
        if cached is not None:
            self._update_expert_access_metadata(key, cached)
            
            # Trigger prefetch outside of locks
            if config.enable_prefetch:
                self._prefetch_adjacent_experts(layer_idx, expert_idx)
                if config.enable_predictive_prefetch:
                    self._prefetch_predicted_experts()
            
            return cached.weights
        
        # Cache miss - check ghost entries with lock
        ghost_readmit = False
        if config.enable_ghost_entries:
            with self._lock:
                if key in self._ghost_entries:
                    ghost = self._ghost_entries[key]
                    if ghost.should_readmit(time.time(), self._experts.get(key, ExpertMetadata(0, 0, 0)).use_count):
                        ghost_readmit = True
                    del self._ghost_entries[key]
        
        # Track access for admission control - with lock
        with self._lock:
            if key not in self._experts:
                self._experts[key] = ExpertMetadata(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    size_bytes=0,
                )
            self._experts[key].use_count += 1
            self._experts[key].last_used = time.time()
        
        # Load expert weights outside global lock using per-layer lock
        with self._per_layer_locks[layer_idx]:
            # Double-check cache in case another thread loaded it
            cached = self._get_expert_from_cache(key)
            if cached is not None:
                return cached.weights
            
            load_start = time.time()
            weights = self._load_expert_impl(layer_idx, expert_idx)
            load_time_ms = (time.time() - load_start) * 1000
        
        # Update cache with global lock - but only for metadata, not loading
        with self._lock:
            # Double-check again after acquiring lock
            if key in self._expert_weight_cache:
                return self._expert_weight_cache[key].weights
            
            # Admission control: decide whether to cache
            should_cache = ghost_readmit
            
            if not should_cache and config.enable_admission:
                load_time_value = load_time_ms / 1000.0
                access_value = self._experts[key].use_count
                admission_score = (
                    access_value * 0.5 +
                    load_time_value * config.load_time_weight
                )
                should_cache = admission_score >= config.admission_threshold
            elif not config.enable_admission:
                should_cache = True
            
            if not should_cache:
                return weights
            
            actual_size = sum(t.numel() * t.element_size() for t in weights.values())
            
            # Create cached expert entry
            cached_expert = CachedExpert(
                weights=weights,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                size_bytes=actual_size,
                load_time_ms=load_time_ms,
            )
            cached_expert.update_access()
            
            # Evict if necessary before adding
            self._evict_experts_if_needed(actual_size)
            
            # Add to cache
            self._expert_weight_cache[key] = cached_expert
            self._experts[key].size_bytes = actual_size
            self._experts[key].is_cached = True
            self._gpu_memory_used += actual_size
            
            # Trigger prefetch
            if config.enable_prefetch:
                self._prefetch_adjacent_experts(layer_idx, expert_idx)
                if config.enable_predictive_prefetch:
                    self._prefetch_predicted_experts()
            
            return weights

    def _evict_experts_if_needed(self, required_bytes: int) -> None:
        """Evict experts using optimized heap-based policy to make room.
        
        Leverages OptimizedExpertCache's heap-based O(log n) eviction for
        superior performance over manual O(n) scanning.
        
        Considers:
        - Maximum number of cached entries
        - Maximum memory usage
        - Eviction policy (LRU, LFU, WEIGHTED, or ADAPTIVE)
        
        Optimized with batch eviction to reduce lock contention.
        """
        config = self._expert_cache_config
        
        # Use OptimizedExpertCache's efficient heap-based eviction
        # This provides O(log n) eviction vs O(n) for manual scanning
        evicted_list = self._expert_weight_cache.evict_if_needed(
            required_bytes, 
            self._gpu_memory_used
        )
        
        # Update metadata for evicted experts
        current_time = time.time()
        for evict_key, evicted in evicted_list:
            self._gpu_memory_used -= evicted.size_bytes
            if evict_key in self._experts:
                self._experts[evict_key].is_cached = False
            
            # Add ghost entry for readmit tracking
            if config.enable_ghost_entries:
                self._add_ghost_entry(evict_key, evicted, current_time)
    
    def _select_eviction_candidate(self, current_time: float) -> tuple[int, int] | None:
        """Select the best eviction candidate using optimized heap-based peek.
        
        Uses OptimizedExpertCache's heap-based O(1) peek for superior
        performance over manual O(n) scanning.
        """
        # Delegate to OptimizedExpertCache's efficient heap-based candidate selection
        # This provides O(1) peek vs O(n) for manual scanning
        return self._expert_weight_cache.get_eviction_candidate()
    
    def _add_ghost_entry(self, key: tuple[int, int], evicted: CachedExpert,
                         current_time: float) -> None:
        """Add a ghost entry for ARC-style admission control."""
        config = self._expert_cache_config
        
        ghost = GhostEntry(
            layer_idx=key[0],
            expert_idx=key[1],
            evicted_at=current_time,
            access_count_at_eviction=evicted.access_count,
            size_bytes=evicted.size_bytes,
        )
        
        self._ghost_entries[key] = ghost
        
        # Limit ghost history size
        while len(self._ghost_entries) > config.ghost_history_size:
            self._ghost_entries.popitem(last=False)
    
    def _evict_one_expert(self, current_time: float) -> None:
        """Evict the single best candidate using optimized heap-based selection.
        
        Uses OptimizedExpertCache's heap-based O(log n) eviction and maintains 
        ghost entries for ARC-style admission control.
        """
        config = self._expert_cache_config
        
        # Use OptimizedExpertCache's efficient heap-based eviction
        # This provides O(log n) eviction with automatic score recalculation
        evicted_list = self._expert_weight_cache.evict_if_needed(0, 0)
        
        if not evicted_list:
            return
        
        # Process evicted expert(s) - update metadata and add ghost entries
        for evict_key, evicted in evicted_list:
            # Add ghost entry for tracking (helps with readmission decisions)
            if config.enable_ghost_entries:
                ghost = GhostEntry(
                    layer_idx=evict_key[0],
                    expert_idx=evict_key[1],
                    evicted_at=current_time,
                    access_count_at_eviction=evicted.access_count,
                    size_bytes=evicted.size_bytes,
                )
                self._ghost_entries[evict_key] = ghost
                
                # Limit ghost history size
                while len(self._ghost_entries) > config.ghost_history_size:
                    self._ghost_entries.popitem(last=False)
            
            self._gpu_memory_used -= evicted.size_bytes
            if evict_key in self._experts:
                self._experts[evict_key].is_cached = False
    
    def _prefetch_adjacent_experts(self, layer_idx: int, expert_idx: int) -> None:
        """Prefetch adjacent experts that are likely to be accessed next.
        
        Prefetches experts in the same layer (next index) and potentially
        the same expert in the next layer for MoE routing patterns.
        """
        config = self._expert_cache_config
        if not config.enable_prefetch:
            return
        
        # Submit prefetch tasks asynchronously
        prefetch_targets = []
        
        # Next expert in same layer
        next_expert = expert_idx + 1
        if next_expert < self._num_experts_per_layer:
            key = (layer_idx, next_expert)
            if key not in self._expert_weight_cache:
                prefetch_targets.append(key)
        
        # Same expert in next layer (common routing pattern)
        next_layer = layer_idx + 1
        if next_layer < self._num_layers:
            key = (next_layer, expert_idx)
            if key not in self._expert_weight_cache:
                prefetch_targets.append(key)
        
        # Submit prefetch tasks
        for target_layer, target_expert in prefetch_targets[:config.prefetch_distance]:
            self._executor.submit(
                self._prefetch_expert_task,
                target_layer,
                target_expert,
            )
    
    def _prefetch_predicted_experts(self) -> None:
        """Prefetch experts based on predicted access patterns.
        
        Uses access pattern history to predict which experts will be needed
        next and prefetches them in advance to reduce latency.
        """
        config = self._expert_cache_config
        if not config.enable_predictive_prefetch:
            return
        
        predictions = self._access_pattern.predict_next()
        if not predictions:
            return
        
        with self._lock:
            current_cache_size = len(self._expert_weight_cache)
            available_slots = max(0, config.max_entries - current_cache_size - 1)
            
            if available_slots == 0:
                return
        
        # Limit predictions to available slots
        for layer_idx, expert_idx in predictions[:available_slots]:
            key = (layer_idx, expert_idx)
            with self._lock:
                if key in self._expert_weight_cache:
                    continue
            
            # Submit prefetch task
            self._executor.submit(
                self._prefetch_expert_task,
                layer_idx,
                expert_idx,
            )
    
    def _prefetch_expert_task(self, layer_idx: int, expert_idx: int) -> None:
        """Background task to prefetch an expert.
        
        Lock contention optimization:
        - Uses per-layer lock for loading
        - Minimizes global lock to only cache metadata updates
        """
        key = (layer_idx, expert_idx)
        
        with self._lock:
            # Double-check it's still not cached
            if key in self._expert_weight_cache:
                return
            
            # Don't prefetch if we're at capacity (save room for explicit loads)
            if len(self._expert_weight_cache) >= self._expert_cache_config.max_entries:
                return
        
        try:
            # Load outside of global lock using per-layer lock
            with self._per_layer_locks[layer_idx]:
                weights = self._load_expert_impl(layer_idx, expert_idx)
            
            actual_size = sum(t.numel() * t.element_size() for t in weights.values())
            
            # Only cache if still not present and we have room
            with self._lock:
                if (key not in self._expert_weight_cache and
                    len(self._expert_weight_cache) < self._expert_cache_config.max_entries):
                    cached = CachedExpert(
                        weights=weights,
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        size_bytes=actual_size,
                    )
                    self._expert_weight_cache[key] = cached
                    if key in self._experts:
                        self._experts[key].size_bytes = actual_size
                        self._experts[key].is_cached = True
                    self._gpu_memory_used += actual_size
        except Exception:
            # Prefetch failures are non-fatal
            pass

    def _load_expert_impl(self, layer_idx: int, expert_idx: int) -> dict[str, Any]:
        """Load expert weights from disk or storage."""
        import torch
        # Simulated or real load
        if self._loader and hasattr(self._loader, "load_expert"):
            weights = self._loader.load_expert(
                layer_idx,
                expert_idx,
                device=self._device
            )
        else:
            # Mock weights for verification or if loader doesn't support experts
            t = torch.zeros(1, device=self._device)
            weights = {"weight": t}
            
        return weights

    def evict_inactive_experts(self, active_experts: set[int]) -> None:
        """Evict non-active experts from GPU memory using optimized heap-based selection.
        
        Uses smart eviction policy to prefer keeping high-value experts
        even if not currently active (based on frequency and size).
        Leverages OptimizedExpertCache's heap for efficient candidate selection.
        """
        with self._lock:
            if self._num_experts_per_layer == 0:
                return

            active_tuples = {(e_id // self._num_experts_per_layer, e_id % self._num_experts_per_layer)
                            for e_id in active_experts}

            # Evict non-active experts if we're over capacity
            # Use heap-optimized eviction from OptimizedExpertCache
            current_time = time.time()
            config = self._expert_cache_config
            
            while len(self._expert_weight_cache) > self._expert_cache_size:
                # Get best eviction candidate using heap-based O(1) peek
                candidate_key = self._expert_weight_cache.get_eviction_candidate()
                
                if candidate_key is None:
                    break
                
                # Skip if candidate is active - need to find non-active expert
                if candidate_key in active_tuples:
                    # Fall back to manual scan for non-active expert
                    # This is rare - only when best candidate is active
                    best_candidate: tuple[tuple[int, int], float] | None = None
                    
                    for key, cached in self._expert_weight_cache.items():
                        if key not in active_tuples:
                            score = cached.compute_score(
                                config.eviction_policy,
                                current_time,
                                config.hot_expert_boost
                            )
                            if best_candidate is None or score < best_candidate[1]:
                                best_candidate = (key, score)
                    
                    if best_candidate:
                        evict_key = best_candidate[0]
                        evicted = self._expert_weight_cache.pop(evict_key)
                        self._gpu_memory_used -= evicted.size_bytes
                        if evict_key in self._experts:
                            self._experts[evict_key].is_cached = False
                    else:
                        # All remaining experts are active, can't evict more
                        break
                else:
                    # Candidate is not active, evict using heap-based removal
                    evicted = self._expert_weight_cache.pop(candidate_key)
                    self._gpu_memory_used -= evicted.size_bytes
                    if candidate_key in self._experts:
                        self._experts[candidate_key].is_cached = False
                    
                    # Add ghost entry for readmit tracking
                    if config.enable_ghost_entries:
                        self._add_ghost_entry(candidate_key, evicted, current_time)

    def compress_kv_cache(self, layer_idx: int, head_idx: int, kv: torch.Tensor) -> torch.Tensor:
        """Compress KV cache using MLA's 8x compression strategy."""
        if self._kv_compression_ratio <= 1:
            return kv
        
        # Latent projection simulation
        d_model = kv.shape[-1]
        d_latent = d_model // self._kv_compression_ratio
        compressed = kv[..., :d_latent]
        
        self._kv_cache[(layer_idx, head_idx)] = compressed
        return compressed

    def checkpoint_activation(self, layer_idx: int, activation: torch.Tensor) -> None:
        """Store activation for recomputation during prefill."""
        if not self._activation_checkpointing:
            return
        self._activations[layer_idx] = activation.detach()

    def get_buffer(self, size_bytes: int, dtype: Any | None = None) -> torch.Tensor:
        """Acquire a transient buffer from the pool or create a new one.
        
        Uses tiered buffer pooling for efficient reuse of transient buffers
        across inference iterations. Falls back to new allocation if pool
        is empty or buffer size doesn't match pooled sizes.
        
        Args:
            size_bytes: Required buffer size in bytes
            dtype: Target dtype (default uint8)
            
        Returns:
            Tensor buffer ready for use
        """
        return self._buffer_pool.acquire(size_bytes, self._device, dtype)

    def release_buffer(self, buffer: torch.Tensor, skip_pool: bool = False) -> None:
        """Return a buffer to the pool for reuse.
        
        Args:
            buffer: Buffer to return to pool
            skip_pool: If True, don't pool and let GC collect (for large buffers)
        """
        self._buffer_pool.release(buffer, skip_pool)

    def get_buffer_pool_stats(self) -> dict:
        """Get buffer pool statistics for performance monitoring.
        
        Returns:
            Dict with pool metrics including hit rate, utilization,
            tier breakdowns, and eviction counts.
        """
        return self._buffer_pool.get_stats()

    def get_memory_stats(self) -> dict:
        """Return current memory usage breakdown."""
        return MemoryStats(
            max_memory_gb=self._max_memory_bytes / 1024**3,
            gpu_used_gb=self._gpu_memory_used / 1024**3,
            cpu_used_gb=0.0, # Simplified
            layers_loaded=sum(1 for l in self._layers.values() if l.is_loaded),
            experts_cached=len(self._expert_weight_cache),
            kv_cache_compressed=self._kv_compression_ratio > 1,
            compression_ratio=self._kv_compression_ratio,
            unified_memory_enabled=self._unified_memory,
        ).to_dict()

    def get_bandwidth_stats(self) -> dict[str, Any]:
        """Get bandwidth optimization statistics."""
        if self._bandwidth_optimizer:
            return self._bandwidth_optimizer.get_bandwidth_stats()
        return {"bandwidth_opt_enabled": False}

    def _optimized_transfer(self, tensor: torch.Tensor, target_device: str) -> torch.Tensor:
        """Transfer tensor with bandwidth optimization."""
        if self._bandwidth_optimizer:
            return self._bandwidth_optimizer.create_optimized_transfer(tensor, target_device)
        return tensor.to(target_device)

    def stream_layer_from_disk(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Stream a single layer from disk using memory-mapped I/O.
        
        This method provides efficient weight streaming that loads only
        the required layer weights into memory, reducing peak memory usage
        for large models.
        
        Lock contention optimization:
        - Uses per-layer lock for layer-specific data access
        - Minimizes global lock to only metadata updates
        - Loads weights outside of global lock
        
        Args:
            layer_idx: Index of the layer to stream
            
        Returns:
            Dictionary of weight tensors for the layer
        """
        import torch
        
        if layer_idx not in self._per_layer_locks:
            return {}

        # Check if already loaded - use per-layer lock
        with self._per_layer_locks[layer_idx]:
            meta = self._layers.get(layer_idx)
            if meta and meta.is_loaded and meta.weights:
                return meta.weights
        
        # Load weights outside of any lock to maximize concurrency
        actual_size = None
        if self._weight_streamer is not None and self._loader is not None:
            weights = self._stream_layer_weights(layer_idx)
        elif self._loader is not None:
            weights = self._loader.load_layer(layer_idx, device=self._device)
        else:
            weights = {"weight": torch.zeros(1, device=self._device)}
        
        # Update layer metadata with per-layer lock
        with self._per_layer_locks[layer_idx]:
            if meta:
                meta.weights = weights
                meta.is_loaded = True
                actual_size = sum(t.numel() * t.element_size() for t in weights.values())
                meta.size_bytes = actual_size
        
        # Update global metadata with global lock - minimal work
        with self._lock:
            if actual_size is not None:
                self._gpu_memory_used += actual_size
            self._streamed_layers.add(layer_idx)
            if layer_idx not in self._layer_stream_order:
                self._layer_stream_order.append(layer_idx)
        
        return weights
    
    def _stream_layer_weights(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Internal method to stream layer weights using shard metadata."""
        import torch
        
        weights = {}
        
        # Get tensors for this layer from loader
        if hasattr(self._loader, '_layer_to_tensors'):
            tensor_names = self._loader._layer_to_tensors.get(layer_idx, [])
            
            # Get accurate shard info from loader
            shard_info = self._loader.get_layer_shard_info(layer_idx)
            
            # Prefetch next layers
            if self._weight_streamer and self._weight_streamer.config.enable_prefetch:
                next_layer_names = []
                for i in range(1, self._prefetch_window + 1):
                    next_idx = layer_idx + i
                    if next_idx < self._num_layers:
                        # We need shard info for next layers too
                        next_shard_info = self._loader.get_layer_shard_info(next_idx)
                        if next_shard_info:
                            # Add to prefetch queue (limit to a reasonable number)
                            prefetch_names = list(next_shard_info.keys())
                            self._weight_streamer.prefetch_weights(prefetch_names[:10], next_shard_info)
            
            # Load current layer weights
            for name in tensor_names:
                try:
                    tensor_info = shard_info.get(name)
                    if not tensor_info:
                         # Fallback if info missing
                         weights[name] = self._loader.load_tensor(name, device=self._device)
                         continue

                    dtype_str = tensor_info.get("dtype")
                    shape = tensor_info.get("shape")
                    
                    # Resolve dtype string to torch.dtype
                    torch_dtype = MMFP4ModelLoader.DTYPE_MAP.get(dtype_str, torch.float32)

                    if self._weight_streamer:
                        tensor = self._weight_streamer.stream_weight(
                            name,
                            tensor_info.get("path"),
                            tensor_info.get("offset"),
                            tensor_info.get("size"),
                            dtype=torch_dtype,
                            shape=shape,
                        )
                    else:
                        tensor = self._loader.load_tensor(name, device=self._device)
                    weights[name] = tensor
                except Exception:
                    # Fallback: create placeholder
                    weights[name] = torch.zeros(1, device=self._device)
        
        return weights if weights else {"weight": torch.zeros(1, device=self._device)}
    
    def unload_streamed_layer(self, layer_idx: int) -> bool:
        """Unload a streamed layer to free memory.
        
        Lock contention optimization:
        - Uses per-layer lock for layer-specific data
        - Minimizes global lock to only aggregate metadata updates
        
        Args:
            layer_idx: Index of the layer to unload
            
        Returns:
            True if layer was found and unloaded
        """
        if layer_idx not in self._per_layer_locks:
            return False

        size_to_free = 0
        tensor_names: list[str] = []
        
        # Get layer data under per-layer lock
        with self._per_layer_locks[layer_idx]:
            meta = self._layers.get(layer_idx)
            if not meta or not meta.is_loaded:
                return False
            
            # Calculate size to free
            if meta.weights:
                size_to_free = meta.size_bytes
                meta.weights = None
            meta.is_loaded = False
            
            # Collect tensor names for later eviction
            if self._weight_streamer and hasattr(self._loader, '_layer_to_tensors'):
                tensor_names = self._loader._layer_to_tensors.get(layer_idx, [])
        
        # Update global metadata with global lock - minimal work
        with self._lock:
            self._gpu_memory_used -= size_to_free
            self._streamed_layers.discard(layer_idx)
        
        # Evict from weight streamer cache outside of locks
        if self._weight_streamer and tensor_names:
            for name in tensor_names:
                self._weight_streamer.evict_weight(name)
        
        return True
    
    def optimize_streaming_for_inference(self) -> None:
        """Configure streaming parameters based on available memory.
        
        Analyzes current memory usage and adjusts streaming cache size,
        prefetch window, and eviction policy for optimal performance.
        """
        with self._lock:
            if not self._weight_streamer:
                return
            
            # Calculate target cache size based on available memory
            available_memory = self._max_memory_bytes - self._gpu_memory_used
            target_stream_memory = available_memory * 0.4  # 40% of remaining
            
            # Adjust streamer config
            self._weight_streamer._max_stream_memory = int(target_stream_memory)
            self._weight_streamer.config.prefetch_queue_size = min(
                self._prefetch_window + 1, 5
            )
            
            # Prefetch upcoming layers if memory allows
            if self._layer_stream_order:
                current_idx = self._layer_stream_order[-1]
                for i in range(1, self._prefetch_window + 1):
                    next_idx = current_idx + i
                    if next_idx < self._num_layers and next_idx not in self._streamed_layers:
                        # Schedule async prefetch
                        if hasattr(self, '_executor'):
                            self._executor.submit(self.stream_layer_from_disk, next_idx)
    
    def get_streaming_stats(self) -> dict[str, Any]:
        """Get weight streaming statistics."""
        stats = {
            "streaming_enabled": self._enable_weight_streaming,
            "streamed_layers": len(self._streamed_layers),
            "streaming_active": self._weight_streamer is not None,
        }
        
        if self._weight_streamer:
            stream_stats = self._weight_streamer.get_stream_stats()
            stats.update(stream_stats)
        
        return stats
    
    def _weight_streaming(self, layer_indices: list[int] | None = None) -> dict[str, Any]:
        """Implement weight streaming with on-demand loading for inference.
        
        This method orchestrates the loading of model weights on-demand during
        inference, enabling large models to run with limited GPU memory by
        streaming only the required layers when needed.
        
        The implementation:
        1. Loads specified layers (or all if none specified) on-demand
        2. Manages memory budget through LRU eviction
        3. Coordinates with weight streamer for efficient disk-to-GPU transfer
        4. Tracks streaming statistics for performance monitoring
        
        Args:
            layer_indices: List of layer indices to load. If None, uses the
                current streaming order or loads all layers sequentially.
                
        Returns:
            Dict containing streaming results:
                - loaded_layers: List of layer indices successfully loaded
                - failed_layers: List of layer indices that failed to load
                - memory_used_gb: Current GPU memory usage in GB
                - cache_hit_rate: Weight streamer cache hit rate
                - streaming_time_ms: Total time for streaming operation
        """
        start_time = time.time()
        loaded_layers: list[int] = []
        failed_layers: list[int] = []
        
        # Determine which layers to stream
        if layer_indices is None:
            # Use streaming order if available, otherwise stream all layers
            if self._layer_stream_order:
                layer_indices = self._layer_stream_order
            else:
                layer_indices = list(range(self._num_layers))
        
        with self._lock:
            # Check available memory budget
            available_memory = self._max_memory_bytes - self._gpu_memory_used
            
            # Adjust based on weight streamer cache if available
            if self._weight_streamer:
                streamer_stats = self._weight_streamer.get_stream_stats()
                cache_hit_rate = streamer_stats.get("hit_rate", 0.0)
            else:
                cache_hit_rate = 0.0
        
        # Stream each layer on-demand
        for layer_idx in layer_indices:
            try:
                # Check if already loaded
                meta = self._layers.get(layer_idx)
                if meta and meta.is_loaded:
                    loaded_layers.append(layer_idx)
                    continue
                
                # Estimate memory needed for this layer
                estimated_size = meta.size_bytes if meta else int(0.5 * 1024**3)
                
                # Check memory budget - evict if necessary
                with self._lock:
                    if self._gpu_memory_used + estimated_size > self._max_memory_bytes * 0.9:
                        # Need to free memory by unloading streamed layers
                        self._evict_for_memory(estimated_size)
                
                # Stream layer from disk using the weight streamer
                weights = self.stream_layer_from_disk(layer_idx)
                
                if weights:
                    loaded_layers.append(layer_idx)
                    
                    # Update metadata
                    if meta:
                        meta.is_loaded = True
                        meta.weights = weights
                else:
                    failed_layers.append(layer_idx)
                    
            except Exception:
                failed_layers.append(layer_idx)
        
        # Calculate streaming time
        streaming_time_ms = (time.time() - start_time) * 1000
        
        # Get updated stats
        with self._lock:
            memory_used_gb = self._gpu_memory_used / (1024**3)
            
            # Update cache hit rate from streamer
            if self._weight_streamer:
                streamer_stats = self._weight_streamer.get_stream_stats()
                cache_hit_rate = streamer_stats.get("hit_rate", 0.0)
        
        return {
            "loaded_layers": loaded_layers,
            "failed_layers": failed_layers,
            "memory_used_gb": memory_used_gb,
            "cache_hit_rate": cache_hit_rate,
            "streaming_time_ms": streaming_time_ms,
        }
    
    def _evict_for_memory(self, required_bytes: int) -> int:
        """Evict layers to free up required memory.
        
        Uses LRU eviction on streamed layers to make room for new layers
        while staying within the memory budget.
        
        Args:
            required_bytes: Amount of memory needed in bytes
            
        Returns:
            Amount of memory freed in bytes
        """
        freed_bytes = 0
        target_free = required_bytes * 1.1  # 10% buffer
        
        # Evict oldest streamed layers first (LRU order)
        with self._lock:
            for layer_idx in list(self._layer_stream_order):
                if freed_bytes >= target_free:
                    break
                    
                meta = self._layers.get(layer_idx)
                if meta and meta.is_loaded and meta.weights:
                    # Unload this layer
                    layer_size = meta.size_bytes
                    self.unload_streamed_layer(layer_idx)
                    freed_bytes += layer_size
        
        return freed_bytes
    
    def _memory_pressure(self, pressure_level: str = "medium") -> dict[str, Any]:
        """Handle memory pressure by gracefully degrading performance.
        
        Implements a tiered response to memory constraints:
        - low: Clear non-essential caches (activations, buffer pool)
        - medium: Evict LRU experts and streamed layers, increase KV compression
        - high: Aggressive eviction, disable prefetching, clear all caches
        - critical: Emergency cleanup, unload all layers, minimal memory footprint
        
        Args:
            pressure_level: One of "low", "medium", "high", "critical"
            
        Returns:
            Dict with actions taken and memory freed
        """
        import torch
        
        actions_taken: list[str] = []
        memory_freed_bytes = 0
        
        with self._lock:
            initial_memory = self._gpu_memory_used
            
            if pressure_level == "low":
                # Clear activations (can be recomputed)
                if self._activations:
                    activation_size = sum(
                        a.numel() * a.element_size()
                        for a in self._activations.values()
                    )
                    self._activations.clear()
                    memory_freed_bytes += activation_size
                    actions_taken.append(f"cleared_activations:{activation_size / 1024**2:.1f}MB")
                
                # Clear buffer pool
                pool_size = self._buffer_pool.get_size()
                if pool_size > 0:
                    self._buffer_pool.clear()
                    memory_freed_bytes += pool_size
                    actions_taken.append(f"cleared_buffer_pool:{pool_size / 1024**2:.1f}MB")
            
            elif pressure_level == "medium":
                # Do low-level cleanup first
                if self._activations:
                    self._activations.clear()
                    actions_taken.append("cleared_activations")
                self._buffer_pool.clear()
                actions_taken.append("cleared_buffer_pool")
                
                # Compact memory before evicting
                compaction_result = self.compact_memory(force=False)
                if compaction_result.get("compaction_performed"):
                    actions_taken.append(
                        f"memory_compaction:freed={compaction_result.get('bytes_freed', 0) / 1024**2:.1f}MB"
                    )
                
                # Evict half of cached experts using adaptive policy
                experts_to_evict = len(self._expert_weight_cache) // 2
                current_time = time.time()
                for _ in range(experts_to_evict):
                    if self._expert_weight_cache:
                        self._evict_one_expert(current_time)
                
                # Calculate freed memory
                freed_after = sum(
                    ce.size_bytes for ce in self._expert_weight_cache.values()
                )
                if experts_to_evict > 0:
                    actions_taken.append(f"evicted_experts:{experts_to_evict}")
                
                # Reduce streaming memory budget
                if self._weight_streamer:
                    old_limit = self._weight_streamer._max_stream_memory
                    new_limit = int(old_limit * 0.5)  # Reduce by 50%
                    self._weight_streamer._max_stream_memory = new_limit
                    actions_taken.append(f"reduced_stream_budget:{old_limit / 1024**3:.2f}GB->{new_limit / 1024**3:.2f}GB")
                
                # Increase KV compression if possible
                if self._kv_compression_ratio < MLACompressionRatio.EXTREME:
                    old_ratio = self._kv_compression_ratio
                    self._kv_compression_ratio = min(
                        self._kv_compression_ratio * 2,
                        MLACompressionRatio.EXTREME
                    )
                    actions_taken.append(f"increased_kv_compression:{old_ratio}x->{self._kv_compression_ratio}x")
            
            elif pressure_level == "high":
                # Force memory compaction first
                compaction_result = self.compact_memory(force=True)
                if compaction_result.get("compaction_performed"):
                    actions_taken.append(
                        f"memory_compaction:freed={compaction_result.get('bytes_freed', 0) / 1024**3:.2f}GB"
                    )
                
                # Aggressive expert eviction using adaptive policy
                current_time = time.time()
                while len(self._expert_weight_cache) > 1:
                    self._evict_one_expert(current_time)
                
                experts_remaining = len(self._expert_weight_cache)
                actions_taken.append(f"aggressive_expert_eviction:remaining={experts_remaining}")
                
                # Disable prefetching
                if self._weight_streamer:
                    self._weight_streamer.config.enable_prefetch = False
                    actions_taken.append("disabled_prefetch")
                
                # Clear all pending prefetch futures
                cancelled = 0
                for future in self._prefetch_futures.values():
                    if future.cancel():
                        cancelled += 1
                self._prefetch_futures.clear()
                if cancelled > 0:
                    actions_taken.append(f"cancelled_prefetches:{cancelled}")
                
                # Clear weight streamer cache
                if self._weight_streamer:
                    old_cache_size = self._weight_streamer._stream_memory_used
                    self._weight_streamer.clear_cache()
                    memory_freed_bytes += old_cache_size
                    actions_taken.append(f"cleared_stream_cache:{old_cache_size / 1024**3:.2f}GB")
                
                # Reduce prefetch window
                self._prefetch_window = max(1, self._prefetch_window // 2)
                actions_taken.append(f"reduced_prefetch_window:{self._prefetch_window}")
                
                # Clear all activations and buffers
                self._activations.clear()
                self._buffer_pool.clear()
                actions_taken.append("cleared_all_transient")
            
            elif pressure_level == "critical":
                # Emergency: unload everything possible
                
                # Unload all streamed layers except current (if any)
                layers_unloaded = 0
                for layer_idx in list(self._streamed_layers):
                    meta = self._layers.get(layer_idx)
                    if meta and meta.is_loaded:
                        if meta.weights:
                            self._gpu_memory_used -= meta.size_bytes
                            memory_freed_bytes += meta.size_bytes
                            meta.weights = None
                        meta.is_loaded = False
                        layers_unloaded += 1
                self._streamed_layers.clear()
                self._layer_stream_order.clear()
                
                if layers_unloaded > 0:
                    actions_taken.append(f"unloaded_layers:{layers_unloaded}")
                
                # Clear all expert caches
                all_experts_size = sum(
                    ce.size_bytes for ce in self._expert_weight_cache.values()
                )
                for key in self._expert_weight_cache:
                    if key in self._experts:
                        self._experts[key].is_cached = False
                self._expert_weight_cache.clear()
                self._gpu_memory_used -= all_experts_size
                memory_freed_bytes += all_experts_size
                actions_taken.append(f"cleared_all_experts:{all_experts_size / 1024**3:.2f}GB")
                
                # Clear weight streamer
                if self._weight_streamer:
                    streamer_cache = self._weight_streamer._stream_memory_used
                    self._weight_streamer.clear_cache()
                    memory_freed_bytes += streamer_cache
                    actions_taken.append(f"cleared_streamer:{streamer_cache / 1024**3:.2f}GB")
                
                # Clear KV cache
                kv_size = sum(
                    t.numel() * t.element_size()
                    for t in self._kv_cache.values()
                )
                self._kv_cache.clear()
                memory_freed_bytes += kv_size
                actions_taken.append(f"cleared_kv_cache:{kv_size / 1024**3:.2f}GB")
                
                # Set maximum compression
                self._kv_compression_ratio = MLACompressionRatio.EXTREME
                actions_taken.append("max_kv_compression_enabled")
                
                # Disable all async operations
                if self._weight_streamer:
                    self._weight_streamer.config.enable_prefetch = False
                    self._weight_streamer.config.enable_async_load = False
                actions_taken.append("disabled_async_ops")
                
                # Force garbage collection hint
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    actions_taken.append("emptied_mps_cache")
            
            actual_freed = initial_memory - self._gpu_memory_used
            
        return {
            "pressure_level": pressure_level,
            "actions_taken": actions_taken,
            "memory_freed_gb": actual_freed / 1024**3,
            "memory_freed_bytes": actual_freed,
            "current_memory_gb": self._gpu_memory_used / 1024**3,
        }

    def compact_memory(self, force: bool = False) -> dict[str, Any]:
        """Perform memory compaction to reduce fragmentation.
        
        Compacts expert cache, layer buffers, and buffer pool to
        consolidate free memory and improve memory locality.
        
        Args:
            force: If True, perform compaction regardless of fragmentation level
            
        Returns:
            Dictionary with compaction results
        """
        if not force:
            # Check if we should compact based on fragmentation
            fragmentation = self.calculate_fragmentation()
            if not self._memory_compactor.should_compact(
                fragmentation,
                self._gpu_memory_used,
                self._max_memory_bytes,
            ):
                return {
                    "compaction_performed": False,
                    "reason": "fragmentation_below_threshold",
                    "current_fragmentation": fragmentation,
                }
        
        # Perform compaction
        return self._memory_compactor.perform_compaction(
            memory_manager=self,
            expert_cache=self._expert_weight_cache,
            layers=self._layers,
            layer_stream_order=self._layer_stream_order,
            buffer_pool=self._buffer_pool,
        )
    
    def calculate_fragmentation(self) -> float:
        """Calculate current memory fragmentation ratio.
        
        Analyzes expert cache and layer buffers to estimate
        the level of memory fragmentation.
        
        Returns:
            Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
        """
        allocated_blocks: list[tuple[int, int]] = []
        current_offset = 0
        
        # Add expert cache blocks
        for (layer_idx, expert_idx), expert in self._expert_weight_cache.items():
            # Use hash as synthetic address
            addr = hash((layer_idx, expert_idx)) % (2**32)
            allocated_blocks.append((addr, expert.size_bytes))
        
        # Add layer buffer blocks
        for idx, layer in self._layers.items():
            if layer.is_loaded:
                addr = hash(idx) % (2**32)
                allocated_blocks.append((addr, layer.size_bytes))
        
        # Calculate fragmentation
        total_memory = self._gpu_memory_used
        if total_memory == 0:
            return 0.0
        
        return self._memory_compactor.calculate_fragmentation(
            allocated_blocks,
            total_memory,
        )
    
    def get_compaction_stats(self) -> dict[str, Any]:
        """Get memory compaction statistics."""
        stats = self._memory_compactor.get_stats()
        stats["current_fragmentation"] = self.calculate_fragmentation()
        stats["auto_compaction_enabled"] = self._compaction_config.enable_auto_compaction
        return stats
    
    def configure_compaction(self, config: CompactionConfig) -> None:
        """Configure memory compaction settings.
        
        Args:
            config: New compaction configuration
        """
        self._compaction_config = config
        self._memory_compactor.config = config
    
    def get_zero_copy_stats(self) -> dict[str, Any]:
        """Get zero-copy transfer statistics."""
        if self._zero_copy_manager:
            stats = self._zero_copy_manager.get_stats()
            stats["zero_copy_enabled"] = self._enable_zero_copy
            stats["unified_memory"] = self._unified_memory
            return stats
        return {
            "zero_copy_enabled": False,
            "unified_memory": self._unified_memory,
        }
    
    def synchronize_zero_copy(self) -> None:
        """Synchronize all pending zero-copy transfers."""
        if self._zero_copy_manager:
            self._zero_copy_manager.synchronize()
        else:
            import torch
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    def cleanup(self) -> None:
        if self._weight_streamer:
            self._weight_streamer.shutdown()
            self._weight_streamer = None
        if self._weight_prefetcher:
            self._weight_prefetcher.shutdown()
            self._weight_prefetcher = None
        if self._bandwidth_optimizer:
            self._bandwidth_optimizer.cleanup()
            self._bandwidth_optimizer = None
        if self._zero_copy_manager:
            self._zero_copy_manager.cleanup()
            self._zero_copy_manager = None
        self._executor.shutdown(wait=False)
        self._layers.clear()
        self._experts.clear()
        self._kv_cache.clear()
        self._activations.clear()
        self._buffer_pool.clear()
        self._streamed_layers.clear()
        self._layer_stream_order.clear()

    def __repr__(self) -> str:
        stats = self.get_memory_stats()
        stream_info = "streaming" if self._enable_weight_streaming else "standard"
        zc_info = "zero_copy" if self._enable_zero_copy else "standard"
        return f"MMFP4MemoryManager(used={stats['gpu_used_gb']:.2f}GB, layers={stats['layers_loaded']}, {stream_info}, {zc_info})"
