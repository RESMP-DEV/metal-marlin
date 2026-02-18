from __future__ import annotations

import json
import logging
import re
import struct
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

# Import buffer recycling for memory optimization
from metal_marlin.memory.buffer_recycler import BufferRecycler
from metal_marlin.memory.memory_pressure import (
    MemoryPressureConfig,
    get_global_memory_pressure_monitor,
)

logger = logging.getLogger(__name__)


@dataclass
class PrefetchConfig:
    """Configuration for weight prefetching.
    
    Attributes:
        enable_prefetch: Enable weight prefetching
        lookahead_count: Number of tensors to prefetch ahead
        max_prefetch_memory_mb: Maximum memory for prefetched weights
        prefetch_queue_size: Maximum number of concurrent prefetch operations
        adaptive_prefetch: Adapt prefetch count based on hit rate
        min_hit_rate_for_adaptation: Minimum hit rate before adjusting
    """
    enable_prefetch: bool = True
    lookahead_count: int = 5
    max_prefetch_memory_mb: float = 2048.0  # 2GB default
    prefetch_queue_size: int = 4
    adaptive_prefetch: bool = True
    min_hit_rate_for_adaptation: float = 0.5


@dataclass
class PrefetchStats:
    """Statistics for weight prefetching."""
    hits: int = 0
    misses: int = 0
    prefetches: int = 0
    evictions: int = 0
    total_prefetch_time_ms: float = 0.0
    last_access_time: float = field(default_factory=time.time)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "prefetches": self.prefetches,
            "evictions": self.evictions,
            "total_prefetch_time_ms": self.total_prefetch_time_ms,
        }


class WeightPrefetcher:
    """Weight prefetcher for memory-optimized model loading.
    
    Implements predictive prefetching of model weights:
    - Sequential prefetching: Prefetches next N tensors in sequence
    - Stride-based prefetching: Prefetches based on access patterns
    - LRU cache: Keeps recently used weights in memory
    - Async loading: Non-blocking prefetch operations
    
    This enables overlapping I/O with computation, reducing latency
    during inference by having weights ready before they're needed.
    """
    
    def __init__(
        self,
        loader: MMFP4ModelLoader,
        config: PrefetchConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize weight prefetcher.
        
        Args:
            loader: MMFP4ModelLoader instance for loading weights
            config: Prefetch configuration
            device: Target device for prefetched weights
        """
        self._loader = loader
        self.config = config or PrefetchConfig()
        self._device = device
        
        # Memory pressure monitoring
        self._pressure_monitor = get_global_memory_pressure_monitor()
        
        # Prefetch cache: tensor_name -> (tensor, access_time)
        self._cache: dict[str, tuple[torch.Tensor, float]] = {}
        self._cache_lock = threading.RLock()
        self._max_cache_bytes = int(self.config.max_prefetch_memory_mb * 1024 * 1024)
        self._current_cache_bytes = 0
        
        # Async prefetch executor
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.prefetch_queue_size,
            thread_name_prefix="weight_prefetch"
        )
        self._pending_prefetches: dict[str, Future[torch.Tensor | None]] = {}
        
        # Access pattern tracking for adaptive prefetching
        self._access_history: list[str] = []
        self._access_history_lock = threading.Lock()
        self._max_history = 20
        
        # Statistics
        self._stats = PrefetchStats()
        self._stats_lock = threading.Lock()
        
        # Sequence counter for LRU
        self._access_sequence = 0
    
    def get(self, tensor_name: str) -> torch.Tensor | None:
        """Get tensor from prefetch cache if available.
        
        Args:
            tensor_name: Name of the tensor to retrieve
            
        Returns:
            The tensor if in cache, None otherwise
        """
        with self._cache_lock:
            if tensor_name in self._cache:
                tensor, _ = self._cache[tensor_name]
                # Update access time
                self._access_sequence += 1
                self._cache[tensor_name] = (tensor, self._access_sequence)
                with self._stats_lock:
                    self._stats.hits += 1
                    self._stats.last_access_time = time.time()
                return tensor
        
        with self._stats_lock:
            self._stats.misses += 1
        return None
    
    def prefetch(self, tensor_names: list[str]) -> list[Future[torch.Tensor | None]]:
        """Queue tensors for prefetching.
        
        Args:
            tensor_names: List of tensor names to prefetch
            
        Returns:
            List of futures for the prefetch operations
        """
        if not self.config.enable_prefetch:
            return []
            
        # Check memory pressure
        is_warning, is_critical = self._pressure_monitor.check_pressure()
        if is_critical:
            # Drop prefetches and clear cache if critical
            self.clear_cache()
            return []
        
        futures: list[Future[torch.Tensor | None]] = []
        
        with self._cache_lock:
            for name in tensor_names:
                # Skip if already cached or being prefetched
                if name in self._cache or name in self._pending_prefetches:
                    continue
                
                # Submit prefetch task
                future = self._executor.submit(self._load_tensor, name)
                self._pending_prefetches[name] = future
                futures.append(future)
        
        return futures
    
    def prefetch_sequential(self, start_name: str, count: int | None = None) -> None:
        """Prefetch next N tensors in sequence.
        
        Args:
            start_name: Starting tensor name
            count: Number of tensors to prefetch (default: lookahead_count)
        """
        if not self.config.enable_prefetch:
            return
        
        count = count or self.config.lookahead_count
        
        # Get ordered list of tensors
        all_tensors = list(self._loader._tensor_metadata.keys())
        
        try:
            start_idx = all_tensors.index(start_name)
        except ValueError:
            return
        
        # Prefetch next N tensors
        end_idx = min(start_idx + count + 1, len(all_tensors))
        to_prefetch = all_tensors[start_idx + 1:end_idx]
        
        self.prefetch(to_prefetch)
    
    def prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch all tensors for a specific layer.
        
        Args:
            layer_idx: Layer index to prefetch
        """
        if not self.config.enable_prefetch:
            return
        
        tensor_names = self._loader._layer_to_tensors.get(layer_idx, [])
        if tensor_names:
            self.prefetch(list(tensor_names))
    
    def _load_tensor(self, tensor_name: str) -> torch.Tensor | None:
        """Load a single tensor (called by executor).
        
        Args:
            tensor_name: Name of tensor to load
            
        Returns:
            Loaded tensor or None if failed
        """
        # Abort if critical pressure
        _, is_critical = self._pressure_monitor.check_pressure()
        if is_critical:
            return None

        try:
            start_time = time.time()
            tensor = self._loader.load_tensor(tensor_name, device=self._device)
            
            # Add to cache if space available
            self._add_to_cache(tensor_name, tensor)
            
            with self._stats_lock:
                self._stats.prefetches += 1
                self._stats.total_prefetch_time_ms += (time.time() - start_time) * 1000
            
            return tensor
        except Exception as e:
            logger.warning(f"Failed to prefetch tensor {tensor_name}: {e}")
            return None
        finally:
            with self._cache_lock:
                self._pending_prefetches.pop(tensor_name, None)
    
    def _add_to_cache(self, name: str, tensor: torch.Tensor) -> None:
        """Add tensor to cache with LRU eviction.
        
        Args:
            name: Tensor name
            tensor: Tensor to cache
        """
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        with self._cache_lock:
            # Evict if necessary
            while (self._current_cache_bytes + tensor_bytes > self._max_cache_bytes and
                   self._cache):
                self._evict_lru()
            
            # Add to cache
            self._access_sequence += 1
            self._cache[name] = (tensor, self._access_sequence)
            self._current_cache_bytes += tensor_bytes
    
    def _evict_lru(self) -> None:
        """Evict least recently used tensor from cache."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_name = min(self._cache.items(), key=lambda x: x[1][1])[0]
        tensor, _ = self._cache[lru_name]
        
        # Remove from cache
        self._current_cache_bytes -= tensor.numel() * tensor.element_size()
        del self._cache[lru_name]
        
        with self._stats_lock:
            self._stats.evictions += 1
    
    def record_access(self, tensor_name: str) -> None:
        """Record tensor access for pattern detection.
        
        Args:
            tensor_name: Name of tensor that was accessed
        """
        with self._access_history_lock:
            self._access_history.append(tensor_name)
            if len(self._access_history) > self._max_history:
                self._access_history.pop(0)
        
        # Trigger sequential prefetch based on this access
        self.prefetch_sequential(tensor_name)
    
    def get_stats(self) -> dict[str, Any]:
        """Get prefetch statistics."""
        with self._stats_lock:
            stats = self._stats.to_dict()
        
        with self._cache_lock:
            stats["cache_size_bytes"] = self._current_cache_bytes
            stats["cache_max_bytes"] = self._max_cache_bytes
            stats["cached_tensors"] = len(self._cache)
            stats["pending_prefetches"] = len(self._pending_prefetches)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear prefetch cache."""
        with self._cache_lock:
            self._cache.clear()
            self._current_cache_bytes = 0
    
    def shutdown(self) -> None:
        """Shutdown prefetcher and release resources."""
        # Cancel pending prefetches
        with self._cache_lock:
            for future in self._pending_prefetches.values():
                future.cancel()
            self._pending_prefetches.clear()
        
        self._executor.shutdown(wait=True)
        self.clear_cache()


@dataclass
class TensorMetadata:
    """Metadata for a tensor in a safetensors file."""
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    size_bytes: int
    data_offsets: tuple[int, int]


class MMFP4ModelLoader:
    """Model loader with support for weight streaming from disk.
    
    Features:
        - Safetensors index parsing for multi-shard models
        - Layer-based weight loading
        - Quantized weight (FP4) retrieval
        - Memory-mapped streaming support
        - Tensor metadata extraction
        - Buffer recycling for reduced allocation overhead
    """
    
    # Module-level buffer recycler for memory optimization
    _buffer_recycler: BufferRecycler | None = None
    _recycler_lock = threading.Lock()
    
    DTYPE_MAP = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
        "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    DTYPE_SIZES = {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1,
        "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    }
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._tensor_metadata: dict[str, TensorMetadata] = {}
        self._shard_headers: dict[Path, dict] = {}
        self._parse_index()
        
    def _parse_index(self) -> None:
        """Parse model.safetensors.index.json for tensor->shard mapping."""
        index_path = self.model_path / "model.safetensors.index.json"
        
        if not index_path.exists():
            # Support single-file safetensors if index is missing
            single_file = self.model_path / "model.safetensors"
            if single_file.exists():
                self._tensor_to_shard = {}
                self._layer_to_tensors = defaultdict(list)
                self._shard_handles = {}
                self._register_shard(single_file)
                return
            
            raise FileNotFoundError(
                f"No model files found at {self.model_path}. "
                "Expected 'model.safetensors.index.json' or 'model.safetensors'."
            )
        
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        self._tensor_to_shard = {}
        self._layer_to_tensors = defaultdict(list)
        self._shard_handles = {}
        
        for tensor_name, shard_name in weight_map.items():
            shard_path = self.model_path / shard_name
            self._tensor_to_shard[tensor_name] = shard_path
            
            layer_idx = self._extract_layer_index(tensor_name)
            if layer_idx is not None:
                self._layer_to_tensors[layer_idx].append(tensor_name)
        
        # Parse shard headers for tensor metadata (needed for streaming)
        self._parse_shard_headers()
    
    def _register_shard(self, shard_path: Path) -> None:
        """Register all tensors in a single shard."""
        with safe_open(shard_path, framework="pt") as f:
            for name in f.keys():
                self._tensor_to_shard[name] = shard_path
                layer_idx = self._extract_layer_index(name)
                if layer_idx is not None:
                    self._layer_to_tensors[layer_idx].append(name)
    
    def _parse_shard_headers(self) -> None:
        """Parse safetensors headers to extract tensor metadata for streaming."""
        # Get unique shard paths
        shard_paths = set(self._tensor_to_shard.values())
        
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
                
            try:
                header, header_len = self._read_safetensors_header(shard_path)
                self._shard_headers[shard_path] = header
                
                # Calculate start of data section (8 bytes length + header JSON)
                data_start = 8 + header_len
                
                # Extract tensor metadata
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    
                    dtype = info.get("dtype", "F32")
                    shape = tuple(info.get("shape", []))
                    data_offsets = info.get("data_offsets", [0, 0])
                    # Offset is relative to data_start
                    offset = data_start + data_offsets[0]
                    size_bytes = data_offsets[1] - data_offsets[0]
                    
                    self._tensor_metadata[name] = TensorMetadata(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        offset=offset,
                        size_bytes=size_bytes,
                        data_offsets=(data_offsets[0], data_offsets[1]),
                    )
            except (OSError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse shard header for {shard_path}: {e}")
                continue
    
    def _read_safetensors_header(self, shard_path: Path) -> tuple[dict, int]:
        """Read and parse the safetensors file header.
        
        Returns:
            Tuple of (header_dict, header_length_bytes)
        """
        with open(shard_path, "rb") as f:
            # Read header length (first 8 bytes as little-endian unsigned long long)
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return {}, 0
            
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            
            # Read and parse header JSON
            header_bytes = f.read(header_len)
            return json.loads(header_bytes.decode("utf-8")), header_len
    
    def _extract_layer_index(self, name: str) -> int | None:
        """Extract layer index from tensor name using common patterns."""
        patterns = [
            r"layers\.(\d+)\.",
            r"h\.(\d+)\.",
            r"blocks\.(\d+)\.",
        ]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None
    
    def _get_shard_handle(self, shard_path: Path) -> Any:
        if shard_path not in self._shard_handles:
            self._shard_handles[shard_path] = safe_open(shard_path, framework="pt")
        return self._shard_handles[shard_path]
    
    def load_tensor(
        self,
        tensor_name: str,
        device: str = "cpu",
        zero_copy: bool = False,
    ) -> torch.Tensor:
        """Load a single tensor by name with optional device placement.
        
        Args:
            tensor_name: Full name of the tensor in the checkpoint
            device: Target device ("cpu", "mps", "cuda", etc.)
            zero_copy: Use zero-copy transfer for MPS Unified Memory
            
        Returns:
            The loaded tensor
        """
        shard_path = self._tensor_to_shard.get(tensor_name)
        if shard_path is None:
            raise KeyError(f"Tensor {tensor_name} not found in model checkpoint")
        
        handle = self._get_shard_handle(shard_path)
        tensor = handle.get_tensor(tensor_name)
        
        # Device placement with zero-copy optimization
        if device != "cpu":
            if device == "mps" and torch.backends.mps.is_available():
                if zero_copy:
                    # Zero-copy transfer for Unified Memory:
                    # 1. Pin memory for faster DMA transfer
                    # 2. Use non-blocking transfer for async copy
                    # On Apple Silicon, this enables zero-copy via unified memory architecture
                    if not tensor.is_pinned():
                        tensor = tensor.pin_memory()
                    tensor = tensor.to("mps", non_blocking=True)
                else:
                    tensor = tensor.to("mps")
            elif device.startswith("cuda") and torch.cuda.is_available():
                if zero_copy and hasattr(torch.cuda, 'HostRegister'):
                    # For CUDA with pinned memory support
                    if not tensor.is_pinned():
                        tensor = tensor.pin_memory()
                    tensor = tensor.to(device, non_blocking=True)
                else:
                    tensor = tensor.to(device)
        
        return tensor
    
    def load_tensor_streaming(
        self,
        tensor_name: str,
        device: str = "cpu",
        use_mmap: bool = True,
        buffer: bytearray | memoryview | None = None,
        zero_copy: bool = False,
        buffer_recycler: Any | None = None,
    ) -> torch.Tensor:
        """Load a tensor using memory-mapped I/O for efficient streaming.
        
        This method bypasses the safetensors handle and reads directly
        from the file using memory mapping, which is more efficient for
        large models when only specific tensors are needed.
        
        On Apple Silicon with unified memory, memory-mapped files can be
        accessed directly by the GPU without explicit CPU->GPU copies,
        enabling true zero-copy transfers.
        
        Args:
            tensor_name: Full name of the tensor
            device: Target device
            use_mmap: Whether to use memory mapping (faster for large files)
            buffer: Optional pre-allocated buffer to read into
            zero_copy: Use zero-copy transfer for unified memory systems
        buffer_recycler: Optional BufferRecycler for buffer reuse
            
        Returns:
            The loaded tensor
        """
        import mmap
        
        # Try to get recycled buffer if none provided
        if buffer is None and buffer_recycler is not None:
            metadata = self._tensor_metadata.get(tensor_name)
            if metadata is not None:
                recycled = buffer_recycler.get_buffer(metadata.size_bytes)
                if recycled is not None:
                    buffer = recycled
        
        metadata = self._tensor_metadata.get(tensor_name)
        shard_path = self._tensor_to_shard.get(tensor_name)
        
        if metadata is None or shard_path is None:
            # Fall back to regular loading
            return self.load_tensor(tensor_name, device, zero_copy=zero_copy)
        
        torch_dtype = self.DTYPE_MAP.get(metadata.dtype, torch.float32)
        tensor = None
        raw_bytes = None
        recycled_buffer = None
        
        # Determine if we should use buffer recycling
        use_recycled_buffer = buffer_recycler is not None or self._buffer_recycler is not None
        
        # Read tensor data
        # metadata.offset is now absolute (calculated in _parse_shard_headers)
        
        try:
            from contextlib import ExitStack
            with ExitStack() as stack:
                f = stack.enter_context(open(shard_path, "rb"))
                
                # Use mmap if requested
                reader = f
                if use_mmap:
                    import mmap
                    try:
                        mm = stack.enter_context(
                            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        )
                        reader = mm
                    except (ValueError, OSError) as e:
                        logger.warning(f"Failed to mmap {shard_path}, falling back to file IO: {e}")
                        # Fallback to file reader
                        reader = f

                if buffer is not None:
                    # Use provided buffer
                    target = buffer[:metadata.size_bytes]
                    reader.seek(metadata.offset)
                    reader.readinto(target)
                    raw_bytes = target
                elif use_recycled_buffer:
                    # Try to get a recycled buffer
                    recycler = buffer_recycler or self.get_buffer_recycler()
                    recycled_buffer = recycler.get_buffer(metadata.size_bytes)
                    if recycled_buffer is None:
                        recycled_buffer = bytearray(metadata.size_bytes)
                    
                    # Read directly into recycled buffer
                    target = recycled_buffer[:metadata.size_bytes]
                    reader.seek(metadata.offset)
                    reader.readinto(target)
                    raw_bytes = memoryview(recycled_buffer)
                else:
                    # Optimize: Zero-copy read directly into pre-allocated tensor
                    # Determine allocation shape/dtype
                    is_quantized = metadata.dtype in ("FP4", "NF4")
                    target_dtype = torch.uint8 if is_quantized else torch_dtype
                    
                    alloc_shape = metadata.shape if metadata.shape else (metadata.size_bytes,)
                    if is_quantized:
                        alloc_shape = (metadata.size_bytes,)
                    
                    # Use pinned memory if requested for zero-copy transfer
                    # Allocate pinned memory directly if possible
                    # On Apple Silicon, pinned CPU memory allows zero-copy access from GPU
                    pin_memory = zero_copy and (device != "cpu")
                    
                    try:
                        tensor = torch.empty(
                            alloc_shape, 
                            dtype=target_dtype, 
                            device="cpu",
                            pin_memory=pin_memory
                        )
                        
                        # Read directly into tensor memory via numpy view
                        if not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                        
                        # View as uint8 flat array for safe byte-level access
                        np_view = tensor.view(torch.uint8).flatten().numpy()
                        reader.seek(metadata.offset)
                        reader.readinto(np_view)
                        
                        # Reshape if needed
                        if metadata.shape and tensor.shape != metadata.shape:
                            tensor = tensor.view(target_dtype).reshape(metadata.shape)
                            
                    except (ValueError, OSError, RuntimeError) as e:
                        logger.debug(f"Direct read optimization failed for {tensor_name}: {e}")
                        # Fallback to standard read with optimization
                        if pin_memory:
                             # If we want pinned memory, try to read directly into it
                             # even in fallback path to avoid double allocation
                             try:
                                 # Re-attempt allocation if previous one failed/was skipped
                                 if 'tensor' not in locals() or tensor is None:
                                     tensor = torch.empty(
                                        alloc_shape, 
                                        dtype=target_dtype, 
                                        device="cpu",
                                        pin_memory=True
                                     )
                                 
                                 # Ensure contiguous and view as bytes
                                 if not tensor.is_contiguous():
                                     tensor = tensor.contiguous()
                                     
                                 np_view = tensor.view(torch.uint8).flatten().numpy()
                                 reader.seek(metadata.offset)
                                 reader.readinto(np_view)
                                 
                                 # Skip raw_bytes creation since we read into tensor
                                 raw_bytes = None
                             except Exception:
                                 # Ultimate fallback: read bytes
                                 reader.seek(metadata.offset)
                                 raw_bytes = reader.read(metadata.size_bytes)
                                 tensor = None
                        else:
                            reader.seek(metadata.offset)
                            raw_bytes = reader.read(metadata.size_bytes)

        except Exception as e:
             logger.error(f"Error loading {tensor_name}: {e}")
             raise
        
        # Return buffer to recycler after use
        if buffer is not None and buffer_recycler is not None:
            buffer_recycler.return_buffer(buffer)
        
        # Convert to tensor if not already created via zero-copy path
        if tensor is None:
            if metadata.dtype in ("FP4", "NF4"):
                tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
            else:
                tensor = torch.frombuffer(raw_bytes, dtype=torch_dtype)
            
            # Reshape if shape info available
            if metadata.shape:
                tensor = tensor.view(metadata.shape)
        
        # Move to device with zero-copy optimization
        if device != "cpu":
            if zero_copy and (device == "mps" or device.startswith("cuda")):
                # Zero-copy: use pinned memory + non-blocking transfer
                # If we pinned it already (in optimized path), this transfer is fast
                if not tensor.is_pinned():
                    tensor = tensor.pin_memory()
                tensor = tensor.to(device, non_blocking=True)
            else:
                tensor = tensor.to(device)
        
        # Return the recycled buffer to the pool after tensor creation
        # The tensor now owns its own copy of the data
        if recycled_buffer is not None and use_recycled_buffer:
            recycler = buffer_recycler or self.get_buffer_recycler()
            recycler.return_buffer(recycled_buffer)
        
        return tensor
    
    def get_tensor_metadata(self, tensor_name: str) -> TensorMetadata | None:
        """Get metadata for a tensor without loading it."""
        return self._tensor_metadata.get(tensor_name)
    
    def get_layer_shard_info(self, layer_idx: int) -> dict[str, Any]:
        """Get shard path and offset info for all tensors in a layer.
        
        Returns:
            Dict mapping tensor names to streaming info dicts with:
            - path: Path to shard file
            - offset: Byte offset within shard data section
            - size: Size in bytes
            - dtype: Tensor dtype string
            - shape: Tensor shape tuple
        """
        info = {}
        tensor_names = self._layer_to_tensors.get(layer_idx, [])
        
        for name in tensor_names:
            meta = self._tensor_metadata.get(name)
            shard_path = self._tensor_to_shard.get(name)
            
            if meta and shard_path:
                info[name] = {
                    "path": shard_path,
                    "offset": meta.offset,
                    "size": meta.size_bytes,
                    "dtype": meta.dtype,
                    "shape": meta.shape,
                }
        
        return info
    
    def estimate_layer_size(self, layer_idx: int) -> int:
        """Estimate the total size of a layer in bytes."""
        tensor_names = self._layer_to_tensors.get(layer_idx, [])
        total_size = 0
        
        for name in tensor_names:
            meta = self._tensor_metadata.get(name)
            if meta:
                total_size += meta.size_bytes
            else:
                # Estimate from dtype/size if metadata unavailable
                shard_path = self._tensor_to_shard.get(name)
                if shard_path and shard_path.exists():
                    # Rough estimate: assume average tensor is 1MB
                    total_size += 1024 * 1024
        
        return total_size
    
    def load_layer(self, layer_idx: int, device: str = "mps", zero_copy: bool = False) -> dict[str, torch.Tensor]:
        """Load all tensors for a single layer.
        
        Args:
            layer_idx: Index of the layer to load
            device: Target device ("cpu", "mps", "cuda")
            zero_copy: Use zero-copy transfer for unified memory systems
            
        Returns:
            Dictionary of tensor_name -> tensor
        """
        tensors = {}
        for name in self._layer_to_tensors.get(layer_idx, []):
            shard_path = self._tensor_to_shard[name]
            handle = self._get_shard_handle(shard_path)
            tensor = handle.get_tensor(name)
            
            if device != "cpu":
                if device == "mps" and torch.backends.mps.is_available():
                    if zero_copy:
                        # Zero-copy optimization for Unified Memory:
                        # 1. Pin memory to enable faster DMA transfer
                        # 2. Use non-blocking transfer for async copy
                        # On Apple Silicon, this leverages the unified memory architecture
                        # for near-zero-copy access between CPU and GPU
                        if not tensor.is_pinned():
                            tensor = tensor.pin_memory()
                        tensor = tensor.to("mps", non_blocking=True)
                    else:
                        tensor = tensor.to("mps")
                elif device.startswith("cuda") and torch.cuda.is_available():
                    if zero_copy:
                        # Zero-copy for CUDA with pinned memory
                        if not tensor.is_pinned():
                            tensor = tensor.pin_memory()
                        tensor = tensor.to(device, non_blocking=True)
                    else:
                        tensor = tensor.to(device)
                    
            tensors[name] = tensor
        return tensors
    
    def mmap_weights(
        self,
        device: str = "cpu",
        lazy: bool = True,
        cache_size_gb: float = 4.0,
        buffer_recycler: Any | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]] | dict[str, torch.Tensor]:
        """Load model weights using memory-mapped I/O for memory efficiency.

        This method enables loading models larger than available RAM by
        leveraging the OS virtual memory system. Only accessed weights
        are loaded into physical memory.

        Args:
            device: Target device for loaded tensors ("cpu", "mps", "cuda")
            lazy: If True, returns an iterator that loads on-demand.
                  If False, loads all weights and returns a dict.
            cache_size_gb: Size of in-memory cache for frequently accessed weights
            buffer_recycler: Optional BufferRecycler for buffer reuse

        Returns:
            If lazy=True: Iterator yielding (tensor_name, tensor) tuples
            If lazy=False: Dictionary mapping tensor_name -> tensor

        Example:
            # Lazy loading (memory-efficient)
            for name, tensor in loader.mmap_weights(device="mps"):
                print(f"Loaded {name}: {tensor.shape}")

            # Load all at once
            weights = loader.mmap_weights(device="mps", lazy=False)
        """
        from metal_marlin.memory.mmfp4_memory import MMAPWeightConfig, MMAPWeightManager

        config = MMAPWeightConfig(
            enable_mmap=True,
            cache_size_gb=cache_size_gb,
            enable_lazy_load=lazy,
            pin_memory=(device != "cpu"),
        )

        manager = MMAPWeightManager(self, config=config, device=device, buffer_recycler=buffer_recycler)

        if lazy:
            # Return iterator wrapper that manages manager lifecycle
            return self._mmap_weights_iterator(manager)
        else:
            # Load all weights into memory
            try:
                weights = {}
                for tensor_name in self._tensor_metadata:
                    try:
                        weights[tensor_name] = manager.load_weight(tensor_name)
                    except Exception:
                        continue
                return weights
            finally:
                manager.close()

    def _mmap_weights_iterator(
        self,
        manager: "MMAPWeightManager",
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterator for lazy loading weights via memory mapping.
        
        Args:
            manager: MMAPWeightManager instance to use for loading
            
        Yields:
            Tuples of (tensor_name, tensor)
        """
        try:
            for tensor_name in self._tensor_metadata:
                try:
                    tensor = manager.load_weight(tensor_name)
                    yield tensor_name, tensor
                except Exception:
                    # Skip corrupted/missing tensors
                    continue
        finally:
            # Clean up manager when iterator is exhausted or closed
            manager.close()

    def create_mmap_manager(
        self,
        device: str = "cpu",
        cache_size_gb: float = 4.0,
        buffer_recycler: Any | None = None,
    ) -> "MMAPWeightManager":
        """Create a reusable MMAP weight manager.

        This allows efficient repeated access to weights with caching.

        Args:
            device: Target device for loaded tensors
            cache_size_gb: Size of in-memory cache
            buffer_recycler: Optional BufferRecycler for buffer reuse

        Returns:
            MMAPWeightManager instance
        """
        from metal_marlin.memory.mmfp4_memory import MMAPWeightConfig, MMAPWeightManager

        config = MMAPWeightConfig(
            enable_mmap=True,
            cache_size_gb=cache_size_gb,
            pin_memory=(device != "cpu"),
        )
        return MMAPWeightManager(self, config=config, device=device, buffer_recycler=buffer_recycler)

    def get_quantized_weight(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (packed_weights, scales) tuple for a quantized weight."""
        # Look for packed weights (uint32 [K, N//8] where 8 FP4 nibbles packed per uint32)
        qweight_suffixes = [".qweight", ".weight", ".packed_weight"]
        scales_suffixes = [".scales", ".weight_scale", ".scales_fp4"]
        
        base_name = name
        for s in qweight_suffixes + scales_suffixes:
            if name.endswith(s):
                base_name = name[:-len(s)]
                break
        
        qweight_key = None
        for s in qweight_suffixes:
            cand = base_name + s
            if cand in self._tensor_to_shard:
                qweight_key = cand
                break
        
        scales_key = None
        for s in scales_suffixes:
            cand = base_name + s
            if cand in self._tensor_to_shard:
                scales_key = cand
                break
            
        if qweight_key is None or scales_key is None:
            raise KeyError(f"Could not find both qweight and scales for {name}")
            
        qweight = self._get_shard_handle(self._tensor_to_shard[qweight_key]).get_tensor(qweight_key)
        scales = self._get_shard_handle(self._tensor_to_shard[scales_key]).get_tensor(scales_key)
            
        return qweight, scales
    
    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Iterate over layers for streaming loading."""
        for layer_idx in sorted(self._layer_to_tensors.keys()):
            yield layer_idx, self.load_layer(layer_idx)
    
    def create_prefetcher(
        self,
        config: PrefetchConfig | None = None,
        device: str = "cpu",
    ) -> WeightPrefetcher:
        """Create a weight prefetcher for this loader.
        
        Args:
            config: Prefetch configuration
            device: Target device for prefetched weights
            
        Returns:
            WeightPrefetcher instance
        """
        return WeightPrefetcher(self, config=config, device=device)
    
    def load_with_prefetch(
        self,
        tensor_name: str,
        prefetcher: WeightPrefetcher | None = None,
        device: str = "cpu",
        zero_copy: bool = False,
    ) -> torch.Tensor:
        """Load a tensor with optional prefetch integration.
        
        First checks the prefetcher cache, then falls back to direct loading.
        Also records the access for pattern-based prefetching.
        
        Args:
            tensor_name: Name of the tensor to load
            prefetcher: Optional WeightPrefetcher to check first
            device: Target device
            zero_copy: Use zero-copy transfer for MPS
            
        Returns:
            The loaded tensor
        """
        # Try prefetcher cache first
        if prefetcher is not None:
            cached = prefetcher.get(tensor_name)
            if cached is not None:
                # Record access for pattern detection
                prefetcher.record_access(tensor_name)
                return cached
        
        # Fall back to direct loading
        tensor = self.load_tensor(tensor_name, device=device, zero_copy=zero_copy)
        
        # Record access if prefetcher available
        if prefetcher is not None:
            prefetcher.record_access(tensor_name)
        
        return tensor
    
    def load_layer_with_prefetch(
        self,
        layer_idx: int,
        prefetcher: WeightPrefetcher | None = None,
        device: str = "mps",
        zero_copy: bool = False,
        prefetch_next: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Load a layer with integrated prefetching.
        
        Loads all tensors for the specified layer, utilizing the prefetcher
        cache when available. Optionally prefetches the next layer.
        
        Args:
            layer_idx: Layer index to load
            prefetcher: Optional WeightPrefetcher for caching
            device: Target device
            zero_copy: Use zero-copy transfer
            prefetch_next: Whether to prefetch the next layer
            
        Returns:
            Dictionary of weight tensors for the layer
        """
        tensors = {}
        tensor_names = self._layer_to_tensors.get(layer_idx, [])
        
        for name in tensor_names:
            tensors[name] = self.load_with_prefetch(
                name, prefetcher=prefetcher, device=device, zero_copy=zero_copy
            )
        
        # Prefetch next layer
        if prefetch_next and prefetcher is not None:
            next_idx = layer_idx + 1
            if next_idx < max(self._layer_to_tensors.keys(), default=0) + 1:
                prefetcher.prefetch_layer(next_idx)
        
        return tensors
    
    def stream_weight(
        self,
        tensor_name: str,
        device: str = "cpu",
        use_mmap: bool = True,
        buffer: bytearray | memoryview | None = None,
        zero_copy: bool = False,
        buffer_recycler: Any | None = None,
    ) -> torch.Tensor:
        """Stream a single weight from disk (alias for load_tensor_streaming).
        
        Args:
            tensor_name: Full name of the tensor
            device: Target device
            use_mmap: Whether to use memory mapping
            buffer: Optional pre-allocated buffer
            zero_copy: Use zero-copy transfer
            buffer_recycler: Optional BufferRecycler
            
        Returns:
            The loaded tensor
        """
        return self.load_tensor_streaming(
            tensor_name,
            device=device,
            use_mmap=use_mmap,
            buffer=buffer,
            zero_copy=zero_copy,
            buffer_recycler=buffer_recycler,
        )
