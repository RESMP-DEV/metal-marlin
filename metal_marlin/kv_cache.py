"""Unified KV cache management for efficient autoregressive inference.

This module provides high-performance KV cache implementations for Metal Marlin,
supporting various model architectures and hardware backends.

Historical Context and Consolidation:
-------------------------------------
This file is the result of consolidating several specialized KV cache implementations
that previously existed in the codebase:
- `kv_cache_torch.py`: The original PyTorch-based implementation.
- `mla_kv_cache.py`: A cache specifically for Multi-head Latent Attention (MLA) models.
- `trellis/kv_cache.py`: A cache tailored for the Trellis architecture.

These have been unified into this single module to reduce code duplication and
streamline maintenance. Legacy aliases are provided at the end of the file for
backwards compatibility.

Cache Implementations:
----------------------
The module now contains the following primary classes:

1.  **KVCacheTorch (aliased as KVCache)**:
    - The standard, general-purpose KV cache.
    - Optimized for standard Multi-Head Attention (MHA) and Grouped-Query
      Attention (GQA) transformers.
    - Supports multiple quantization formats (FP8, FP4, INT8) and memory layouts
      to balance performance and memory usage.

2.  **MLAKVCache**:
    - A specialized cache for models using Multi-head Latent Attention (MLA).
    - It stores a compressed latent representation of the keys and values,
      significantly reducing the memory footprint for long-context models.
    - Optimized for Apple Silicon (Metal Performance Shaders).

3.  **TrellisKVCache**:
    - Trellis-focused behavior wrapper around `MLAKVCache`.
    - Preserves Trellis API semantics for global sequence tracking and memory accounting.

3.  **CompressedKVCache**:
    - An extension of `MLAKVCache` that adds features like sliding window attention
      and late quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._compat import require_torch, torch
from .dtypes import DTypeConfig, get_default_config

if TYPE_CHECKING:
    import torch as torch_typing

# Mapping from dtype config strings to PyTorch dtypes
if TYPE_CHECKING:
    _DTYPE_TO_TORCH_CACHE: dict[str, torch.dtype] = {}
else:
    _DTYPE_TO_TORCH_CACHE: dict[str, object] = {}

# Global KV cache pool to reuse buffers
# Key: (shape, dtype, device_str) -> List[torch.Tensor]
_kv_pool: dict[tuple[tuple[int, ...], torch.dtype, str], list[torch.Tensor]] = {}

# Pool metrics for monitoring
_pool_metrics: dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "returns": 0,
    "evictions": 0,
}


def _get_from_pool(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Get a tensor from the pool or allocate a new one.
    
    This is the core pooling function that enables buffer reuse:
    - On cache hit: Returns a zeroed tensor from the pool (fast path)
    - On cache miss: Allocates new tensor via torch.zeros() (slow path)
    
    The pool is keyed by (shape, dtype, device) for type-safe reuse.
    """
    device_str = str(device)
    key = (shape, dtype, device_str)
    if key in _kv_pool and _kv_pool[key]:
        tensor = _kv_pool[key].pop()
        tensor.zero_()
        _pool_metrics["hits"] += 1
        return tensor
    _pool_metrics["misses"] += 1
    return torch.zeros(shape, dtype=dtype, device=device)


def _return_to_pool(tensor: torch.Tensor | None) -> None:
    """Return a tensor to the pool for reuse.
    
    Returns the tensor to the global pool keyed by (shape, dtype, device).
    This enables subsequent _get_from_pool calls to reuse the buffer,
    avoiding expensive GPU memory allocations.
    """
    if tensor is None:
        return
    device_str = str(tensor.device)
    key = (tuple(tensor.shape), tensor.dtype, device_str)
    if key not in _kv_pool:
        _kv_pool[key] = []
    _kv_pool[key].append(tensor)
    _pool_metrics["returns"] += 1


def _return_list_to_pool(tensors: list[torch.Tensor] | None) -> None:
    """Return a list of tensors to the pool."""
    if tensors is None:
        return
    for tensor in tensors:
        _return_to_pool(tensor)


def clear_pool() -> None:
    """Clear all pooled buffers."""
    _pool_metrics["evictions"] += sum(len(v) for v in _kv_pool.values())
    _kv_pool.clear()


def get_pool_stats() -> dict[str, int]:
    """Get pool statistics.
    
    Returns:
        Dictionary with:
        - hits: Number of successful pool retrievals
        - misses: Number of allocations (pool miss)
        - returns: Number of tensors returned to pool
        - pooled_tensors: Current tensors in pool
        - hit_rate: Ratio of hits to total requests (0.0-1.0)
    """
    total_requests = _pool_metrics["hits"] + _pool_metrics["misses"]
    hit_rate = _pool_metrics["hits"] / total_requests if total_requests > 0 else 0.0
    pooled_tensors = sum(len(v) for v in _kv_pool.values())
    return {
        **_pool_metrics,
        "pooled_tensors": pooled_tensors,
        "hit_rate": hit_rate,
    }


def reset_pool_metrics() -> None:
    """Reset pool metrics counters."""
    _pool_metrics.update({"hits": 0, "misses": 0, "returns": 0, "evictions": 0})


def _kv_prealloc(
    num_layers: int,
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    cache_dtype: str = "bf16",
    device: str = "mps",
    layout: str = "BHSD",
) -> dict[str, list[torch.Tensor]]:
    """Pre-allocate KV cache buffers for maximum context length upfront.

    This function pre-allocates all KV cache buffers at their maximum size,
    eliminating incremental allocations during inference. This provides:
    - Zero-allocation inference path (all memory ready upfront)
    - Predictable memory usage (no mid-inference OOMs)
    - Better memory alignment (single large allocation vs fragments)
    - Improved GPU kernel performance (stable memory addresses)

    The pre-allocated buffers are returned as a dictionary that can be used
    directly by KVCacheTorch or stored in the pool for later use.

    Args:
        num_layers: Number of transformer layers
        batch_size: Batch size for generation
        max_seq_len: Maximum sequence length to allocate
        num_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per head
        cache_dtype: Storage dtype - "fp16", "bf16", "fp8", "fp4", "int8"
        device: Device to allocate on (default: "mps")
        layout: Memory layout - "BHSD" or "BSHD"

    Returns:
        Dictionary with keys 'k_cache', 'v_cache', 'k_scales', 'v_scales'
        containing lists of pre-allocated tensors for each layer.

    Example:
        >>> buffers = _kv_prealloc(
        ...     num_layers=32,
        ...     batch_size=1,
        ...     max_seq_len=8192,
        ...     num_kv_heads=8,
        ...     head_dim=128,
        ...     cache_dtype="fp8",
        ... )
        >>> # buffers['k_cache'] is a list of 32 pre-allocated tensors
    """
    require_torch("_kv_prealloc")

    # Determine storage dtype
    storage_dtype_override = None
    if cache_dtype == "fp8":
        quantize_mode = "fp8"
    elif cache_dtype in ("fp8_e5m2", "FP8-E5M2"):
        quantize_mode = "fp8_e5m2"
    elif cache_dtype == "fp4":
        quantize_mode = "fp4"
    elif cache_dtype == "int8":
        quantize_mode = "int8"
    elif cache_dtype in ("fp16", "bf16"):
        quantize_mode = "none"
        storage_dtype_override = cache_dtype
    else:
        quantize_mode = "none"

    # Get default dtype config for scales
    dtype_config = get_default_config()

    k_cache: list[torch.Tensor] = []
    v_cache: list[torch.Tensor] = []
    k_scales: list[torch.Tensor] | None = None
    v_scales: list[torch.Tensor] | None = None

    if quantize_mode == "fp4":
        if layout == "BSHD":
            raise ValueError("FP4 quantization only supports BHSD layout")
        packed_shape = (
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim // 8,
        )
        scale_dtype = _get_torch_dtype(dtype_config.scales)
        scale_shape = (batch_size, num_kv_heads, max_seq_len, 1)
        # Use pool for efficient buffer reuse
        k_cache = [
            _get_from_pool(packed_shape, dtype=torch.int32, device=device)
            for _ in range(num_layers)
        ]
        v_cache = [
            _get_from_pool(packed_shape, dtype=torch.int32, device=device)
            for _ in range(num_layers)
        ]
        k_scales = [
            _get_from_pool(scale_shape, dtype=scale_dtype, device=device)
            for _ in range(num_layers)
        ]
        v_scales = [
            _get_from_pool(scale_shape, dtype=scale_dtype, device=device)
            for _ in range(num_layers)
        ]
    elif quantize_mode in ("fp8", "fp8_e5m2"):
        if layout == "BSHD":
            raise ValueError("FP8 quantization only supports BHSD layout")
        fp8_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        scale_dtype = _get_torch_dtype(dtype_config.scales)
        scale_shape = (batch_size, num_kv_heads, max_seq_len, 1)
        # Use pool for efficient buffer reuse
        k_cache = [
            _get_from_pool(fp8_shape, dtype=torch.uint8, device=device)
            for _ in range(num_layers)
        ]
        v_cache = [
            _get_from_pool(fp8_shape, dtype=torch.uint8, device=device)
            for _ in range(num_layers)
        ]
        k_scales = [
            _get_from_pool(scale_shape, dtype=scale_dtype, device=device)
            for _ in range(num_layers)
        ]
        v_scales = [
            _get_from_pool(scale_shape, dtype=scale_dtype, device=device)
            for _ in range(num_layers)
        ]
    elif quantize_mode == "int8":
        if layout == "BSHD":
            raise ValueError("INT8 quantization only supports BHSD layout")
        int8_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        scale_dtype = _get_torch_dtype(dtype_config.scales)
        scale_shape = (batch_size, num_kv_heads, max_seq_len, 1)
        # Use pool for efficient buffer reuse
        k_cache = [
            _get_from_pool(int8_shape, dtype=torch.int8, device=device)
            for _ in range(num_layers)
        ]
        v_cache = [
            _get_from_pool(int8_shape, dtype=torch.int8, device=device)
            for _ in range(num_layers)
        ]
        k_scales = [
            _get_from_pool(scale_shape, dtype=scale_dtype, device=device)
            for _ in range(num_layers)
        ]
        v_scales = [
            _get_from_pool(scale_shape, dtype=scale_dtype, device=device)
            for _ in range(num_layers)
        ]
    else:
        # No quantization - use fp16/bf16
        storage_dtype_str = storage_dtype_override or dtype_config.kv_cache
        storage_dtype = _get_torch_dtype(storage_dtype_str)
        if layout == "BSHD":
            cache_shape = (
                batch_size,
                max_seq_len,
                num_kv_heads,
                head_dim,
            )
        else:  # BHSD (default)
            cache_shape = (
                batch_size,
                num_kv_heads,
                max_seq_len,
                head_dim,
            )
        # Use pool for efficient buffer reuse
        k_cache = [
            _get_from_pool(cache_shape, dtype=storage_dtype, device=device)
            for _ in range(num_layers)
        ]
        v_cache = [
            _get_from_pool(cache_shape, dtype=storage_dtype, device=device)
            for _ in range(num_layers)
        ]
        k_scales = None
        v_scales = None

    return {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scales": k_scales,
        "v_scales": v_scales,
    }


def require_mps(feature: str = "KV cache") -> None:
    """Raise if PyTorch MPS is not available."""
    require_torch(feature)
    # Check for MPS availability safely (torch is guaranteed not None by require_torch)
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():  # type: ignore
        raise RuntimeError(
            f"{feature} requires PyTorch MPS backend. "
            "Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to PyTorch dtype."""
    require_torch("dtype conversion")

    if not _DTYPE_TO_TORCH_CACHE:
        _DTYPE_TO_TORCH_CACHE.update(
            {
                "fp16": getattr(torch, "float16", torch.float32),
                "bf16": getattr(torch, "bfloat16", torch.float32),
                "fp32": getattr(torch, "float32", None),
                "fp8": getattr(torch, "float16", torch.float32),
                # Aliases
                "float16": getattr(torch, "float16", torch.float32),
                "bfloat16": getattr(torch, "bfloat16", torch.float32),
                "float32": getattr(torch, "float32", None),
            }
        )

    if dtype_str not in _DTYPE_TO_TORCH_CACHE:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    dtype = _DTYPE_TO_TORCH_CACHE[dtype_str]
    if dtype is None:
        raise RuntimeError(
            f"Torch dtype {dtype_str} not available in this environment")

    return dtype  # type: ignore[return-value]


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Resolve dtype string to torch dtype (helper for MLA)."""
    if isinstance(dtype, str):
        return _get_torch_dtype(dtype)
    return dtype


# Cache for shift tensors to avoid frequent CPU-GPU synchronization
_SHIFTS_CACHE: dict[str, torch.Tensor] = {}


def _get_shifts(device: torch.device | str) -> torch.Tensor:
    """Get cached shifts tensor for vectorized packing/unpacking."""
    device_str = str(device)
    if device_str not in _SHIFTS_CACHE:
        _SHIFTS_CACHE[device_str] = torch.tensor(
            [0, 4, 8, 12, 16, 20, 24, 28],
            dtype=torch.int32,
            device=device
        )
    return _SHIFTS_CACHE[device_str]


def _vectorized_unpack(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Vectorized unpack of FP4 uint32 tensor.
    
    Optimized implementation using pure bitwise operations - single uint32 op
    per nibble extraction vs 8 iterations. Unpacks all 8 nibbles in parallel
    using broadcasting and bitwise shifts.
    """
    # packed: [..., dim] (int32)
    # scale: [..., 1] (float16) or [..., dim] if per-channel
    
    # Shape handling
    orig_shape = packed.shape
    packed_dim = orig_shape[-1]
    
    # Reshape to [..., packed_dim, 1] for broadcasting against 8 shifts
    packed_expanded = packed.unsqueeze(-1)  # [..., dim, 1]
    
    # Create shift amounts: [0, 4, 8, 12, 16, 20, 24, 28]
    # Each uint32 contains 8 FP4 values (4 bits each)
    # Pack stores nibble 0 in least significant bits (shift 0), nibble 7 in most (shift 28)
    shifts = _get_shifts(packed.device)
    
    # Single vectorized operation: shift right by n*4 and mask with 0xF
    # This extracts all 8 nibbles in parallel
    # packed_expanded: [..., dim, 1], shifts: [8] -> broadcasts to [..., dim, 8]
    nibbles = (packed_expanded >> shifts) & 0xF  # [..., dim, 8]
    
    # Convert to float16 and dequantize: nibble value -> float
    # FP4 E2M1: 0-15 maps to -6.0 to +6.0 (bias 8, scale 0.5)
    unpacked = nibbles.to(torch.float16) - 8.0
    unpacked = unpacked * 0.5
    
    # Flatten last two dimensions: [..., dim, 8] -> [..., dim*8]
    output_shape = orig_shape[:-1] + (packed_dim * 8,)
    unpacked = unpacked.view(output_shape)
    
    return unpacked * scale.float()


def _vectorized_pack(quantized_reshaped: torch.Tensor) -> torch.Tensor:
    """Vectorized pack of 8 FP4 values (last dim) into uint32.
    
    Optimized implementation using summation reduction.
    Reduces from 7 sequential ops to single kernel launch.
    """
    # quantized_reshaped: [..., 8] (uint8) values in 0-15
    device = quantized_reshaped.device
    
    # Shift amounts: [0, 4, 8, 12, 16, 20, 24, 28] (LSB first)
    # First element (index 0) goes to bits 0-3, last element (index 7) to bits 28-31
    shifts = _get_shifts(device)
    
    # Convert to int32 and shift each nibble to its position
    shifted = quantized_reshaped.to(torch.int32) << shifts
    
    # Summation reduction (equivalent to OR for non-overlapping bits)
    packed = shifted.sum(dim=-1, dtype=torch.int32)
    
    return packed


# =============================================================================
# Standard KVCacheTorch Implementation
# =============================================================================


@dataclass
class CacheConfigTorch:
    """Configuration for PyTorch-based KV cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length to allocate.
        cache_dtype: Cache storage dtype.
            - "fp16": Half precision storage
            - "bf16": BFloat16 storage
            - "fp8": FP8 E4M3 quantized (2x memory savings)
            - "fp4": FP4 E2M1 quantized (4x memory savings)
            - "int8": INT8 symmetric quantization (2x memory savings)
        fp8_scale_method: Scaling method for FP8 quantization.
            - "tensor": Single scale per tensor (faster, lower memory)
            - "channel": Scale per head dimension (better accuracy for outliers)
        layout: Memory layout for cache storage.
            - "BHSD": [batch, num_heads, seq_len, head_dim] (default, standard)
            - "BSHD": [batch, seq_len, num_heads, head_dim] (faster decode writes)
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    cache_dtype: Literal["fp16", "bf16", "fp8", "fp8_e5m2", "FP8-E5M2", "fp4", "int8"] | None = None
    fp8_scale_method: Literal["tensor", "channel"] = "tensor"
    layout: Literal["BHSD", "BSHD"] = "BHSD"


class KVCacheTorch:
    """
    Key-Value cache for transformer inference using PyTorch MPS.

    Supports:
    - Standard MHA (num_heads == num_kv_heads)
    - Grouped Query Attention (num_kv_heads < num_heads)
    - Configurable precision via DTypeConfig:
        - BF16 (default): Better dynamic range
        - FP16: Maximum compatibility
        - FP8: 2x memory savings for long context
        - FP4: 4x memory savings for very long context
        - INT8: 2x memory savings with symmetric quantization

    The cache pre-allocates tensors for max_seq_len to avoid reallocation
    during generation. Only the slice up to current seq_len is used.
    """

    def __init__(
        self,
        config: CacheConfigTorch,
        batch_size: int = 1,
        dtype_config: DTypeConfig | None = None,
        device: str = "mps",
    ):
        require_mps("KV cache")

        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0
        self.cache_position = 0
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()
        self.device = device
        self._fp8_scale_method = config.fp8_scale_method
        self._update_id = 0
        self._last_eviction_update_id = -1
        self._layout = config.layout

        # Determine storage mode from cache_dtype
        self._storage_dtype_override = None
        
        # Pre-dequantization cache to avoid redundant dequantization on repeated access
        # Stores dequantized K/V tensors for each layer [layer_idx] -> (k_dequant, v_dequant)
        self._pre_dequant_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        
        if config.cache_dtype == "fp8":
            self._quantize_mode = "fp8"
        elif config.cache_dtype in ("fp8_e5m2", "FP8-E5M2"):
            self._quantize_mode = "fp8_e5m2"
        elif config.cache_dtype == "fp4":
            self._quantize_mode = "fp4"
        elif config.cache_dtype == "int8":
            self._quantize_mode = "int8"
        elif config.cache_dtype in ("fp16", "bf16"):
            self._quantize_mode = "none"
            self._storage_dtype_override = config.cache_dtype
        else:
            self._quantize_mode = "none"

        # Use _kv_prealloc to pre-allocate all buffers at max context upfront
        # This provides:
        # - Zero-allocation inference path (all memory ready upfront)
        # - Predictable memory usage (no mid-inference OOMs)
        # - Better memory alignment (single large allocation vs fragments)
        # - Improved GPU kernel performance (stable memory addresses)
        cache_dtype = config.cache_dtype or self.dtype_config.kv_cache
        buffers = _kv_prealloc(
            num_layers=config.num_layers,
            batch_size=batch_size,
            max_seq_len=config.max_seq_len,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            cache_dtype=cache_dtype,
            device=device,
            layout=self._layout,
        )
        self.k_cache = buffers["k_cache"]
        self.v_cache = buffers["v_cache"]
        self.k_scales = buffers["k_scales"]
        self.v_scales = buffers["v_scales"]

    def _pre_dequant_cache_clear(self, layer_idx: int | None = None) -> None:
        """Clear pre-dequantization cache for a specific layer or all layers.
        
        Must be called whenever the underlying quantized cache is modified
        to ensure consistency between cached dequantized values and source data.
        
        Args:
            layer_idx: Specific layer to clear, or None to clear all layers
        """
        if layer_idx is not None:
            self._pre_dequant_cache.pop(layer_idx, None)
        else:
            self._pre_dequant_cache.clear()

    def _pre_dequant_cache_get(
        self,
        layer_idx: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scales: torch.Tensor | None,
        v_scales: torch.Tensor | None,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get dequantized K/V from cache or return None if not cached/stale.
        
        The cache is valid when:
        1. We have a cached entry for this layer
        2. The cache was populated at the current sequence length
        
        Args:
            layer_idx: Layer index
            k_cache: Current K cache tensor (for staleness check)
            v_cache: Current V cache tensor (for staleness check)
            k_scales: K scale factors
            v_scales: V scale factors
            target_dtype: Target dtype for dequantized tensors
            
        Returns:
            Cached (k_dequant, v_dequant) tuple or None if cache miss
        """
        # Check if we have a valid cached entry for this layer at current seq_len
        if layer_idx not in self._pre_dequant_cache:
            return None
            
        # Validate the cached seq_len matches current state
        # This ensures cache consistency after updates
        # Handle both BHSD (seq at dim 2) and BSHD (seq at dim 1) layouts
        cached_k, cached_v = self._pre_dequant_cache[layer_idx]
        seq_dim = 2 if self._layout == "BHSD" else 1
        
        if cached_k.shape[seq_dim] != k_cache.shape[seq_dim]:
            self._pre_dequant_cache_clear(layer_idx)
            return None
            
        return cached_k, cached_v

    def _pre_dequant_cache_set(
        self,
        layer_idx: int,
        k_dequant: torch.Tensor,
        v_dequant: torch.Tensor,
    ) -> None:
        """Store dequantized K/V in cache.
        
        Args:
            layer_idx: Layer index
            k_dequant: Dequantized K tensor
            v_dequant: Dequantized V tensor
        """
        self._pre_dequant_cache[layer_idx] = (k_dequant, v_dequant)

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K, V and return full cached K, V."""
        # Clear pre-dequant cache for this layer since underlying data is changing
        self._pre_dequant_cache_clear(layer_idx)
        
        new_seq_len = k_new.shape[2]
        if new_seq_len > self.config.max_seq_len:
            k_new = k_new[:, :, -self.config.max_seq_len:, :]
            v_new = v_new[:, :, -self.config.max_seq_len:, :]
            new_seq_len = self.config.max_seq_len
            self._reset_positions()

        self._maybe_evict(new_seq_len)

        start_pos = self.cache_position
        end_pos = start_pos + new_seq_len
        act_dtype = _get_torch_dtype(self.dtype_config.activations)

        if self._quantize_mode == "fp4":
            k_packed, k_scale = self._quantize_fp4(k_new)
            v_packed, v_scale = self._quantize_fp4(v_new)
            self._kv_transfer(
                self.k_cache[layer_idx], k_packed, start_pos, new_seq_len)
            self._kv_transfer(
                self.v_cache[layer_idx], v_packed, start_pos, new_seq_len)
            self._kv_transfer(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._kv_transfer(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)
            
            # Try to get from pre-dequant cache first
            cached = self._pre_dequant_cache_get(
                layer_idx,
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
                act_dtype,
            )
            if cached is not None:
                k_full, v_full = cached
            else:
                k_full = self._dequant_fp4(
                    self.k_cache[layer_idx][:, :, :end_pos, :],
                    self.k_scales[layer_idx][:, :, :end_pos, :],
                ).to(act_dtype)
                v_full = self._dequant_fp4(
                    self.v_cache[layer_idx][:, :, :end_pos, :],
                    self.v_scales[layer_idx][:, :, :end_pos, :],
                ).to(act_dtype)
                # Cache the dequantized values
                self._pre_dequant_cache_set(layer_idx, k_full, v_full)
        elif self._quantize_mode in ("fp8", "fp8_e5m2"):
            # _quantize_fp8/_dequant_fp8 automatically handle E4M3 vs E5M2 based on _quantize_mode
            k_quant, k_scale = self._quantize_fp8(k_new)
            v_quant, v_scale = self._quantize_fp8(v_new)
            self._kv_transfer(
                self.k_cache[layer_idx], k_quant, start_pos, new_seq_len)
            self._kv_transfer(
                self.v_cache[layer_idx], v_quant, start_pos, new_seq_len)
            self._kv_transfer(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._kv_transfer(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)
            
            # Try to get from pre-dequant cache first
            cached = self._pre_dequant_cache_get(
                layer_idx,
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
                act_dtype,
            )
            if cached is not None:
                k_full, v_full = cached
            else:
                k_full = self._dequant_fp8(
                    self.k_cache[layer_idx][:, :, :end_pos, :],
                    self.k_scales[layer_idx][:, :, :end_pos, :],
                ).to(act_dtype)
                v_full = self._dequant_fp8(
                    self.v_cache[layer_idx][:, :, :end_pos, :],
                    self.v_scales[layer_idx][:, :, :end_pos, :],
                ).to(act_dtype)
                # Cache the dequantized values
                self._pre_dequant_cache_set(layer_idx, k_full, v_full)
        elif self._quantize_mode == "int8":
            k_quant, k_scale = self._quantize_int8(k_new)
            v_quant, v_scale = self._quantize_int8(v_new)
            self._kv_transfer(
                self.k_cache[layer_idx], k_quant, start_pos, new_seq_len)
            self._kv_transfer(
                self.v_cache[layer_idx], v_quant, start_pos, new_seq_len)
            self._kv_transfer(
                self.k_scales[layer_idx], k_scale, start_pos, new_seq_len)
            self._kv_transfer(
                self.v_scales[layer_idx], v_scale, start_pos, new_seq_len)
            
            # Try to get from pre-dequant cache first
            cached = self._pre_dequant_cache_get(
                layer_idx,
                self.k_cache[layer_idx][:, :, :end_pos, :],
                self.v_cache[layer_idx][:, :, :end_pos, :],
                self.k_scales[layer_idx][:, :, :end_pos, :],
                self.v_scales[layer_idx][:, :, :end_pos, :],
                act_dtype,
            )
            if cached is not None:
                k_full, v_full = cached
            else:
                k_full = self._dequant_int8(
                    self.k_cache[layer_idx][:, :, :end_pos, :],
                    self.k_scales[layer_idx][:, :, :end_pos, :],
                ).to(act_dtype)
                v_full = self._dequant_int8(
                    self.v_cache[layer_idx][:, :, :end_pos, :],
                    self.v_scales[layer_idx][:, :, :end_pos, :],
                ).to(act_dtype)
                # Cache the dequantized values
                self._pre_dequant_cache_set(layer_idx, k_full, v_full)
        else:
            storage_dtype_str = self._storage_dtype_override or self.dtype_config.kv_cache
            storage_dtype = _get_torch_dtype(storage_dtype_str)

            if self._layout == "BSHD":
                # BSHD layout: [batch, seq, heads, dim] - optimized for decode writes
                # Use _kv_contiguous_layout to fuse permute+contiguous for efficiency
                k_transposed = self._kv_contiguous_layout(
                    k_new.permute(0, 2, 1, 3), storage_dtype
                )
                v_transposed = self._kv_contiguous_layout(
                    v_new.permute(0, 2, 1, 3), storage_dtype
                )
                # Direct slice assignment (update is already contiguous and dtype-converted)
                self.k_cache[layer_idx].narrow(1, start_pos, new_seq_len).copy_(
                    k_transposed, non_blocking=True)
                self.v_cache[layer_idx].narrow(1, start_pos, new_seq_len).copy_(
                    v_transposed, non_blocking=True)
                # Retrieve with efficient layout conversion
                k_full = self._kv_contiguous_layout(
                    self.k_cache[layer_idx][:, :end_pos, :, :].permute(0, 2, 1, 3), act_dtype
                )
                v_full = self._kv_contiguous_layout(
                    self.v_cache[layer_idx][:, :end_pos, :, :].permute(0, 2, 1, 3), act_dtype
                )
            else:
                self._kv_transfer(
                    self.k_cache[layer_idx], k_new, start_pos, new_seq_len, storage_dtype
                )
                self._kv_transfer(
                    self.v_cache[layer_idx], v_new, start_pos, new_seq_len, storage_dtype
                )
                k_full = self.k_cache[layer_idx][:,
                                                 :, :end_pos, :].to(act_dtype)
                v_full = self.v_cache[layer_idx][:,
                                                 :, :end_pos, :].to(act_dtype)

        return k_full, v_full

    def advance(self, num_tokens: int = 1) -> None:
        """Advance sequence position after processing tokens."""
        self.seq_len += num_tokens
        self.cache_position = self.seq_len
        self._update_id += 1

    def reset(self) -> None:
        """Clear cache for new sequence and return tensors to pool for reuse.
        
        This method returns all cache tensors to the global pool and immediately
        re-allocates new tensors from the pool using _kv_prealloc. This enables
        efficient buffer reuse across sequence boundaries while keeping the cache
        usable with pre-allocated max context buffers.
        """
        # Clear pre-dequant cache before reset
        self._pre_dequant_cache_clear()
        
        # Return cache tensors to pool for reuse
        _return_list_to_pool(self.k_cache)
        _return_list_to_pool(self.v_cache)
        if self.k_scales is not None:
            _return_list_to_pool(self.k_scales)
        if self.v_scales is not None:
            _return_list_to_pool(self.v_scales)
        
        # Re-allocate using _kv_prealloc for consistent max context preallocation
        config = self.config
        cache_dtype = config.cache_dtype or self.dtype_config.kv_cache
        buffers = _kv_prealloc(
            num_layers=config.num_layers,
            batch_size=self.batch_size,
            max_seq_len=config.max_seq_len,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            cache_dtype=cache_dtype,
            device=self.device,
            layout=self._layout,
        )
        self.k_cache = buffers["k_cache"]
        self.v_cache = buffers["v_cache"]
        self.k_scales = buffers["k_scales"]
        self.v_scales = buffers["v_scales"]
        
        self._reset_positions()

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get current cached K, V for a layer (dequantized if needed).
        
        Uses pre-dequantization cache to avoid redundant dequantization
        operations when the same K/V values are accessed multiple times.
        """
        if self.seq_len == 0:
            return None, None

        act_dtype = _get_torch_dtype(self.dtype_config.activations)

        if self._quantize_mode == "fp4":
            # Try pre-dequant cache first
            cached = self._pre_dequant_cache_get(
                layer_idx,
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
                act_dtype,
            )
            if cached is not None:
                k, v = cached
            else:
                k = self._dequant_fp4(
                    self.k_cache[layer_idx][:, :, : self.seq_len, :],
                    self.k_scales[layer_idx][:, :, : self.seq_len, :],
                ).to(act_dtype)
                v = self._dequant_fp4(
                    self.v_cache[layer_idx][:, :, : self.seq_len, :],
                    self.v_scales[layer_idx][:, :, : self.seq_len, :],
                ).to(act_dtype)
                # Cache the dequantized values
                self._pre_dequant_cache_set(layer_idx, k, v)
        elif self._quantize_mode in ("fp8", "fp8_e5m2"):
            # Try pre-dequant cache first
            cached = self._pre_dequant_cache_get(
                layer_idx,
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
                act_dtype,
            )
            if cached is not None:
                k, v = cached
            else:
                # _dequant_fp8 automatically handles E4M3 vs E5M2 based on _quantize_mode
                k = self._dequant_fp8(
                    self.k_cache[layer_idx][:, :, : self.seq_len, :],
                    self.k_scales[layer_idx][:, :, : self.seq_len, :],
                ).to(act_dtype)
                v = self._dequant_fp8(
                    self.v_cache[layer_idx][:, :, : self.seq_len, :],
                    self.v_scales[layer_idx][:, :, : self.seq_len, :],
                ).to(act_dtype)
                # Cache the dequantized values
                self._pre_dequant_cache_set(layer_idx, k, v)
        elif self._quantize_mode == "int8":
            # Try pre-dequant cache first
            cached = self._pre_dequant_cache_get(
                layer_idx,
                self.k_cache[layer_idx][:, :, : self.seq_len, :],
                self.v_cache[layer_idx][:, :, : self.seq_len, :],
                self.k_scales[layer_idx][:, :, : self.seq_len, :],
                self.v_scales[layer_idx][:, :, : self.seq_len, :],
                act_dtype,
            )
            if cached is not None:
                k, v = cached
            else:
                k = self._dequant_int8(
                    self.k_cache[layer_idx][:, :, : self.seq_len, :],
                    self.k_scales[layer_idx][:, :, : self.seq_len, :],
                ).to(act_dtype)
                v = self._dequant_int8(
                    self.v_cache[layer_idx][:, :, : self.seq_len, :],
                    self.v_scales[layer_idx][:, :, : self.seq_len, :],
                ).to(act_dtype)
                # Cache the dequantized values
                self._pre_dequant_cache_set(layer_idx, k, v)
        else:
            if self._layout == "BSHD":
                k = self.k_cache[layer_idx][:, : self.seq_len,
                                            :, :].permute(0, 2, 1, 3).to(act_dtype)
                v = self.v_cache[layer_idx][:, : self.seq_len,
                                            :, :].permute(0, 2, 1, 3).to(act_dtype)
            else:
                k = self.k_cache[layer_idx][:, :,
                                            : self.seq_len, :].to(act_dtype)
                v = self.v_cache[layer_idx][:, :,
                                            : self.seq_len, :].to(act_dtype)

        return k, v

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        if self._quantize_mode == "fp4":
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 0.5 + scale_bytes / self.config.head_dim
        elif self._quantize_mode in ("fp8", "fp8_e5m2"):
            # Both E4M3 and E5M2 use 1 byte per element + scales
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 1.0 + scale_bytes / self.config.head_dim
        elif self._quantize_mode == "int8":
            scale_bytes = 4.0 if self.dtype_config.scales == "fp32" else 2.0
            bytes_per_element = 1.0 + scale_bytes / self.config.head_dim
        else:
            bytes_per_element = 2

        elements = (
            self.batch_size
            * self.config.num_kv_heads
            * self.seq_len
            * self.config.head_dim
            * 2
            * self.config.num_layers
        )
        return elements * bytes_per_element / 1024 / 1024

    def _reset_positions(self) -> None:
        self.seq_len = 0
        self.cache_position = 0
        self._update_id = 0
        self._last_eviction_update_id = -1

    def _maybe_evict(self, new_seq_len: int) -> None:
        """Evict old entries when cache would overflow."""
        end_pos = self.cache_position + new_seq_len
        if end_pos <= self.config.max_seq_len:
            return
        if self._last_eviction_update_id == self._update_id:
            return

        overflow = end_pos - self.config.max_seq_len
        self._shift_cache_left(overflow)
        self._last_eviction_update_id = self._update_id

    def _shift_cache_left(self, shift: int) -> None:
        if shift <= 0 or self.seq_len == 0:
            return
        
        # Clear pre-dequant cache when shifting (data is changing)
        self._pre_dequant_cache_clear()
        
        new_len = max(self.seq_len - shift, 0)

        if self._layout == "BSHD":
            def _shift_list_bshd(tensors: list[torch.Tensor]) -> None:
                for tensor in tensors:
                    if new_len == 0:
                        continue
                    tensor[:, :new_len, :, :] = tensor[:,
                                                       shift: shift + new_len, :, :]

            _shift_list_bshd(self.k_cache)
            _shift_list_bshd(self.v_cache)
        else:
            def _shift_list_bhsd(tensors: list[torch.Tensor]) -> None:
                for tensor in tensors:
                    if new_len == 0:
                        continue
                    tensor[:, :, :new_len, :] = tensor[:,
                                                       :, shift: shift + new_len, :]

            _shift_list_bhsd(self.k_cache)
            _shift_list_bhsd(self.v_cache)
            if self.k_scales is not None:
                _shift_list_bhsd(self.k_scales)
            if self.v_scales is not None:
                _shift_list_bhsd(self.v_scales)

        self.seq_len = new_len
        self.cache_position = new_len

    def _slice_update(
        self,
        target: torch.Tensor,
        update: torch.Tensor,
        start: int,
        length: int,
        dtype_override: torch.dtype | None = None,
    ) -> None:
        """In-place slice update without reallocating the cache tensor."""
        if dtype_override is not None and update.dtype != dtype_override:
            update = update.to(dtype_override)
        elif update.dtype != target.dtype:
            update = update.to(target.dtype)
        if not update.is_contiguous():
            update = update.contiguous()
        target.narrow(2, start, length).copy_(update)

    def _slice_update_bshd(
        self,
        target: torch.Tensor,
        update: torch.Tensor,
        start: int,
        length: int,
        dtype_override: torch.dtype | None = None,
    ) -> None:
        """In-place slice update for BSHD layout."""
        if dtype_override is not None and update.dtype != dtype_override:
            update = update.to(dtype_override)
        elif update.dtype != target.dtype:
            update = update.to(target.dtype)
        if not update.is_contiguous():
            update = update.contiguous()
        target.narrow(1, start, length).copy_(update)

    def _kv_contiguous_layout(
        self,
        tensor: torch.Tensor,
        target_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Ensure tensor has contiguous row-major layout for optimal memory operations.
        
        This method optimizes memory layout by:
        1. Ensuring contiguous row-major storage for coalesced memory access
        2. Performing dtype conversion AFTER ensuring contiguity (avoids temp tensors)
        3. Using non-blocking transfers when possible
        
        The key optimization is doing contiguity check BEFORE dtype conversion,
        which prevents creating a temporary non-contiguous tensor during conversion.
        
        Args:
            tensor: Input tensor that may be non-contiguous (e.g., from permute)
            target_dtype: Optional dtype to convert to after ensuring contiguity
            
        Returns:
            Tensor with contiguous row-major layout, optionally converted dtype
        """
        # Ensure contiguous layout FIRST, before any dtype conversion
        # This is the key optimization - avoids temp tensors during conversion
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Now perform dtype conversion on contiguous memory (more efficient)
        if target_dtype is not None and tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype, non_blocking=True)
        
        return tensor

    def _kv_transfer(
        self,
        target: torch.Tensor,
        update: torch.Tensor,
        start: int,
        length: int,
        dtype_override: torch.dtype | None = None,
    ) -> None:
        """Optimized KV cache transfer with efficient GPU memory operations.
        
        This method optimizes the transfer of key/value data to the cache by:
        1. Using non-blocking transfers when possible (MPS streams)
        2. Selecting the correct slice dimension based on layout (BHSD/BSHD)
        3. Minimizing dtype conversion overhead via _kv_contiguous_layout
        4. Ensuring contiguous row-major memory for optimal copy performance
        
        Args:
            target: The cache tensor to update (k_cache or v_cache)
            update: The new data to write (already quantized if needed)
            start: Starting position in the sequence dimension
            length: Number of tokens to write
            dtype_override: Optional dtype to convert update to before storing
        """
        # Determine target dtype for the cache
        target_dtype = dtype_override if dtype_override is not None else target.dtype
        
        # Use optimized layout conversion: ensure contiguous FIRST, then convert dtype
        # This avoids creating temporary non-contiguous tensors during conversion
        update = self._kv_contiguous_layout(update, target_dtype)
        
        # Select slice dimension based on layout
        # BHSD: [batch, heads, seq_len, dim] -> slice dim 2 (row-major: seq is minor)
        # BSHD: [batch, seq_len, heads, dim] -> slice dim 1 (row-major: heads is minor)
        if self._layout == "BSHD":
            # BSHD layout: slice on dimension 1 (seq_len)
            target.narrow(1, start, length).copy_(update, non_blocking=True)
        else:
            # BHSD layout (default): slice on dimension 2 (seq_len)
            # Row-major: contiguous reads along head_dim (last dim)
            target.narrow(2, start, length).copy_(update, non_blocking=True)

    def _quantize_fp4(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP4 packed format."""
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / 6.0
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -6.0, 6.0)
        quantized = torch.round(scaled * 2.0).to(torch.int8)
        quantized = torch.clamp(quantized + 8, 0, 15).to(torch.uint8)

        batch, heads, seq, dim = tensor.shape
        reshaped = quantized.view(batch, heads, seq, dim // 8, 8)

        packed = _vectorized_pack(reshaped)

        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return packed, scale.to(scale_dtype)

    def _dequant_fp4(self, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize FP4 packed tensor to float."""
        return _vectorized_unpack(packed, scale)

    def _quantize_fp8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP8 format (E4M3 or E5M2 based on mode)."""
        if self._quantize_mode == "fp8_e5m2":
            FP8_MAX = 57344.0  # E5M2 max finite value
        else:
            FP8_MAX = 448.0  # E4M3 max value

        # Compute per-channel scales (one per head_dim position)
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / FP8_MAX

        scaled = tensor / scale
        scaled = torch.clamp(scaled, -FP8_MAX, FP8_MAX)
        # Map to uint8 range [0, 255] with 128 as zero point
        quantized = torch.round(scaled / FP8_MAX * 127.0) + 128.0
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)

        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return quantized, scale.to(scale_dtype)

    def _dequant_fp8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 tensor to float (E4M3 or E5M2 based on mode)."""
        if self._quantize_mode == "fp8_e5m2":
            FP8_MAX = 57344.0  # E5M2 max finite value
        else:
            FP8_MAX = 448.0  # E4M3 max value
        signed = quantized.float() - 128.0
        return signed / 127.0 * FP8_MAX * scale.float()

    def _quantize_int8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8 format with symmetric quantization."""
        INT8_MAX = 127.0
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / INT8_MAX
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -INT8_MAX, INT8_MAX)
        quantized = torch.round(scaled).to(torch.int8)
        scale_dtype = _get_torch_dtype(self.dtype_config.scales)
        return quantized, scale.to(scale_dtype)

    def _dequant_int8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor to float."""
        return quantized.float() * scale.float()


# =============================================================================
# MLA KV Cache Implementation
# =============================================================================

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0  # E5M2 max finite value (training-compatible)


@dataclass
class MLAKVCache:
    """Unified KV cache for Multi-head Latent Attention (MLA).

    Caches the compressed KV representation (c_kv + k_pe) to reduce memory.
    Supports advanced features for long-context inference on Metal.

    Attributes:
        num_layers: Number of transformer layers.
        batch_size: Batch size for generation.
        max_seq_len: Initial maximum sequence length to allocate.
        kv_lora_rank: Compressed KV dimension (rank of latent space).
        qk_rope_head_dim: Dimension of rotary positional embedding.
        device: Device to store cache on (default: 'mps').
        dtype: Data type for cache tensors (default: float16).
        quantize_mode: Quantization mode - "none", "int8", or "fp8".
        fp8_scale_method: Scaling method for FP8/Int8 - "tensor" or "channel".
        auto_grow: Whether to automatically grow capacity when exceeded.
    """

    num_layers: int
    batch_size: int
    max_seq_len: int
    kv_lora_rank: int
    qk_rope_head_dim: int = 64
    device: str = "mps"
    dtype: torch.dtype | str = "float16"
    quantize_mode: Literal["none", "int8", "fp8", "fp8_e5m2", "fp4"] = "none"
    fp8_scale_method: Literal["tensor", "channel"] = "tensor"
    auto_grow: bool = True

    def __post_init__(self) -> None:
        require_mps("MLA KV cache")
        self.dtype = _resolve_dtype(self.dtype)
        self.cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._seq_lens: torch_typing.Tensor = torch.zeros(
            (self.num_layers, self.batch_size), dtype=torch.long, device=self.device
        )
        self._scale_dtype = torch.float16 if self.dtype == torch.float16 else torch.float32

        self.k_nope_cache: torch_typing.Tensor | None = None
        self.v_cache: torch_typing.Tensor | None = None
        self.decompressed_dims: tuple[int, int, int] | None = None

        self._allocate(self.max_seq_len)

    def __init__(
        self,
        num_layers: int,
        batch_size: int = 1,
        max_seq_len: int = 2048,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        device: str = "mps",
        dtype: torch.dtype | str = "float16",
        quantize_mode: Literal["none", "int8", "fp8", "fp4"] = "none",
        fp8_scale_method: Literal["tensor", "channel"] = "tensor",
        auto_grow: bool = True,
        max_batch_size: int | None = None,
    ) -> None:
        """Initialize MLA KV Cache with optional max_batch_size alias for compatibility."""
        # Handle max_batch_size alias for backward compatibility
        if max_batch_size is not None:
            batch_size = max_batch_size

        # Set dataclass fields manually
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.device = device
        self.dtype = dtype
        self.quantize_mode = quantize_mode
        self.fp8_scale_method = fp8_scale_method
        self.auto_grow = auto_grow

        # Call post-init for setup
        self.__post_init__()

    def init_decompressed_cache(
        self,
        num_kv_heads: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
    ) -> None:
        """Initialize caching for decompressed states to avoid re-projection."""
        self.decompressed_dims = (num_kv_heads, qk_nope_head_dim, v_head_dim)
        
        # BHSD layout: [layers, batch, heads, seq, dim]
        # We store layers in dim 0
        k_shape = (
            self.num_layers,
            self.batch_size,
            num_kv_heads,
            self.max_seq_len,
            qk_nope_head_dim,
        )
        v_shape = (
            self.num_layers,
            self.batch_size,
            num_kv_heads,
            self.max_seq_len,
            v_head_dim,
        )
        
        self.k_nope_cache = torch.zeros(
            k_shape, dtype=self.dtype, device=self.device
        )
        self.v_cache = torch.zeros(
            v_shape, dtype=self.dtype, device=self.device
        )

    def _allocate(self, max_seq_len: int) -> None:
        """Allocate cache tensors in BSHD layout using pool for buffer reuse.
        
        Uses the global KV cache pool to reuse previously allocated buffers,
        avoiding expensive GPU memory allocations when possible.
        """
        if self.quantize_mode == "none":
            self.kv_cache = _get_from_pool(
                (self.num_layers, self.batch_size, max_seq_len, self.cache_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self.kv_scales = None
            # Clear FP4-specific buffers
            self._k_packed = None
            self._v_packed = None
            self._k_scales = None
            self._v_scales = None
        elif self.quantize_mode == "int8":
            self.kv_cache = _get_from_pool(
                (self.num_layers, self.batch_size, max_seq_len, self.cache_dim),
                dtype=torch.int8,
                device=self.device,
            )
            scale_dim = self.cache_dim if self.fp8_scale_method == "channel" else 1
            self.kv_scales = _get_from_pool(
                (self.num_layers, self.batch_size, max_seq_len, scale_dim),
                dtype=self._scale_dtype,
                device=self.device,
            )
            # Clear FP4-specific buffers
            self._k_packed = None
            self._v_packed = None
            self._k_scales = None
            self._v_scales = None
        elif self.quantize_mode in ("fp8", "fp8_e5m2"):
            self.kv_cache = _get_from_pool(
                (self.num_layers, self.batch_size, max_seq_len, self.cache_dim),
                dtype=torch.uint8,
                device=self.device,
            )
            scale_dim = self.cache_dim if self.fp8_scale_method == "channel" else 1
            self.kv_scales = _get_from_pool(
                (self.num_layers, self.batch_size, max_seq_len, scale_dim),
                dtype=self._scale_dtype,
                device=self.device,
            )
            # Clear FP4-specific buffers
            self._k_packed = None
            self._v_packed = None
            self._k_scales = None
            self._v_scales = None
        elif self.quantize_mode == "fp4":
            if self.cache_dim % 8 != 0:
                raise ValueError(
                    f"FP4 quantization requires cache_dim ({self.cache_dim}) to be divisible by 8"
                )
            # For FP4 mode with unified compressed KV: packed_dim = cache_dim // 8
            packed_dim = self.cache_dim // 8
            # For separate K/V tensors: allocate dynamic buffers that grow as needed
            # These will be lazily allocated on first use with proper dimensions
            self._k_packed = None
            self._v_packed = None
            self._k_scales = None
            self._v_scales = None
            # Unified cache for compressed_kv input (backward compatible)
            self.kv_cache = _get_from_pool(
                (self.num_layers, self.batch_size, max_seq_len, packed_dim),
                dtype=torch.int32,
                device=self.device,
            )
            self.kv_scales = _get_from_pool(
                (self.num_layers, self.batch_size, max_seq_len, 1),
                dtype=self._scale_dtype,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unsupported quantize_mode: {self.quantize_mode}")
        
        # Allocate decompressed cache if enabled (also using pool)
        if self.decompressed_dims is not None:
            num_kv_heads, qk_nope_head_dim, v_head_dim = self.decompressed_dims
            k_shape = (
                self.num_layers,
                self.batch_size,
                num_kv_heads,
                max_seq_len,
                qk_nope_head_dim,
            )
            v_shape = (
                self.num_layers,
                self.batch_size,
                num_kv_heads,
                max_seq_len,
                v_head_dim,
            )
            # Use pool for decompressed cache buffers
            self.k_nope_cache = _get_from_pool(
                k_shape, dtype=self.dtype, device=self.device
            )
            self.v_cache = _get_from_pool(
                v_shape, dtype=self.dtype, device=self.device
            )

    def _ensure_capacity(self, required_len: int) -> None:
        """Grow cache if needed."""
        if required_len <= self.max_seq_len:
            return

        if not self.auto_grow:
            raise ValueError(
                f"Sequence length {required_len} exceeds max_seq_len {self.max_seq_len}")

        new_max = max(required_len, max(1, self.max_seq_len * 2))
        old_max = self.max_seq_len

        old_kv_cache = self.kv_cache
        old_kv_scales = self.kv_scales
        
        # Save old decompressed
        old_k_nope = self.k_nope_cache
        old_v = self.v_cache
        
        # Save old FP4 buffers and their dimensions
        old_k_packed = self._k_packed
        old_v_packed = self._v_packed
        old_k_scales = self._k_scales
        old_v_scales = self._v_scales
        old_k_packed_dim = getattr(self, '_k_packed_dim', None)
        old_v_packed_dim = getattr(self, '_v_packed_dim', None)

        self._allocate(new_max)

        self.kv_cache[:, :, :old_max, :] = old_kv_cache
        if self.kv_scales is not None and old_kv_scales is not None:
            self.kv_scales[:, :, :old_max, :] = old_kv_scales
            
        # Restore decompressed
        if self.k_nope_cache is not None and old_k_nope is not None:
            self.k_nope_cache[..., :old_max, :] = old_k_nope
        if self.v_cache is not None and old_v is not None:
            self.v_cache[..., :old_max, :] = old_v
        
        # Restore FP4 buffers with proper reallocation
        if old_k_packed is not None and old_k_packed_dim is not None:
            batch = old_k_packed.shape[1]
            self._k_packed = _get_from_pool(
                (self.num_layers, batch, new_max, old_k_packed_dim),
                dtype=torch.int32,
                device=self.device,
            )
            self._k_packed[:, :, :old_max, :] = old_k_packed
            self._k_packed_dim = old_k_packed_dim
            _return_to_pool(old_k_packed)
            
        if old_v_packed is not None and old_v_packed_dim is not None:
            batch = old_v_packed.shape[1]
            self._v_packed = _get_from_pool(
                (self.num_layers, batch, new_max, old_v_packed_dim),
                dtype=torch.int32,
                device=self.device,
            )
            self._v_packed[:, :, :old_max, :] = old_v_packed
            self._v_packed_dim = old_v_packed_dim
            _return_to_pool(old_v_packed)
            
        if old_k_scales is not None:
            batch = old_k_scales.shape[1]
            self._k_scales = _get_from_pool(
                (self.num_layers, batch, new_max, 1),
                dtype=self._scale_dtype,
                device=self.device,
            )
            self._k_scales[:, :, :old_max, :] = old_k_scales
            _return_to_pool(old_k_scales)
            
        if old_v_scales is not None:
            batch = old_v_scales.shape[1]
            self._v_scales = _get_from_pool(
                (self.num_layers, batch, new_max, 1),
                dtype=self._scale_dtype,
                device=self.device,
            )
            self._v_scales[:, :, :old_max, :] = old_v_scales
            _return_to_pool(old_v_scales)

        self.max_seq_len = new_max

    def _ensure_fp4_separate_buffers(
        self, batch: int, num_heads: int, k_head_dim: int, v_head_dim: int
    ) -> None:
        """Lazily allocate FP4 packed buffers for separate K/V storage.
        
        Args:
            batch: Batch size
            num_heads: Number of heads
            k_head_dim: Key head dimension (before packing)
            v_head_dim: Value head dimension (before packing)
        """
        k_packed_dim = k_head_dim // 8
        v_packed_dim = v_head_dim // 8
        
        # Allocate K packed buffer if needed
        # Shape: [layers, batch, max_seq, heads, packed_dim]
        if self._k_packed is None:
            self._k_packed = _get_from_pool(
                (self.num_layers, batch, self.max_seq_len, num_heads, k_packed_dim),
                dtype=torch.int32,
                device=self.device,
            )
            self._k_packed_dim = k_packed_dim
        elif getattr(self, '_k_packed_dim', None) != k_packed_dim:
            # Dimension mismatch - reallocate
            _return_to_pool(self._k_packed)
            self._k_packed = _get_from_pool(
                (self.num_layers, batch, self.max_seq_len, num_heads, k_packed_dim),
                dtype=torch.int32,
                device=self.device,
            )
            self._k_packed_dim = k_packed_dim
            
        # Allocate V packed buffer if needed
        if self._v_packed is None:
            self._v_packed = _get_from_pool(
                (self.num_layers, batch, self.max_seq_len, num_heads, v_packed_dim),
                dtype=torch.int32,
                device=self.device,
            )
            self._v_packed_dim = v_packed_dim
        elif getattr(self, '_v_packed_dim', None) != v_packed_dim:
            # Dimension mismatch - reallocate
            _return_to_pool(self._v_packed)
            self._v_packed = _get_from_pool(
                (self.num_layers, batch, self.max_seq_len, num_heads, v_packed_dim),
                dtype=torch.int32,
                device=self.device,
            )
            self._v_packed_dim = v_packed_dim
            
        # Allocate K scales buffer if needed
        # Shape: [layers, batch, max_seq, heads, 1]
        if self._k_scales is None:
            self._k_scales = _get_from_pool(
                (self.num_layers, batch, self.max_seq_len, num_heads, 1),
                dtype=self._scale_dtype,
                device=self.device,
            )
            
        # Allocate V scales buffer if needed
        if self._v_scales is None:
            self._v_scales = _get_from_pool(
                (self.num_layers, batch, self.max_seq_len, num_heads, 1),
                dtype=self._scale_dtype,
                device=self.device,
            )

    @property
    def seq_lens(self) -> list[int] | torch_typing.Tensor:
        """Sequence lengths per layer (first batch item for compatibility)."""
        return [int(v) for v in self._seq_lens[:, 0].tolist()]

    @property
    def seq_len(self) -> int:
        """Current sequence length across layers."""
        return int(self._seq_lens[:, 0].max().item())

    def get_seq_len(self) -> int:
        """Get current sequence length (for backward compatibility)."""
        return self.seq_len

    def update_decompressed(
        self,
        layer_idx: int,
        k_nope_new: torch_typing.Tensor,
        v_new: torch_typing.Tensor,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update decompressed cache with new tokens and return full sequence."""
        if self.k_nope_cache is None or self.v_cache is None:
            raise RuntimeError("Decompressed cache not initialized. Call init_decompressed_cache() first.")
            
        # k_nope_new: [batch, heads, seq, dim]
        # v_new: [batch, heads, seq, dim]
        
        batch, heads, seq_len, _ = k_nope_new.shape
        start = int(self._seq_lens[layer_idx, 0].item()) - seq_len # We assume seq_lens already updated by update_compressed
        
        # But wait, update_compressed updates seq_lens.
        # So if we call this AFTER update_compressed, start should be derived.
        # Let's trust seq_len passed in match.
        
        # We actually need to know where to write.
        # If this is called AFTER update_compressed, self._seq_lens is already advanced.
        # So start = current_len - seq_len
        
        current_len = int(self._seq_lens[layer_idx, 0].item())
        start = current_len - seq_len
        end = current_len
        
        if start < 0:
             # If seq_lens wasn't updated yet?
             # Let's assume this is called alongside update.
             # But if update_compressed was called, it advanced seq_lens.
             pass
             
        self.k_nope_cache[layer_idx, :batch, :, start:end, :] = k_nope_new
        self.v_cache[layer_idx, :batch, :, start:end, :] = v_new
        
        return (
            self.k_nope_cache[layer_idx, :batch, :, :end, :],
            self.v_cache[layer_idx, :batch, :, :end, :]
        )

    def get_decompressed(
        self,
        layer_idx: int
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor] | None:
        """Get full decompressed k_nope and v for a layer."""
        if self.k_nope_cache is None or self.v_cache is None:
            return None
            
        seq_len = int(self._seq_lens[layer_idx, 0].item())
        if seq_len == 0:
            return None
            
        return (
            self.k_nope_cache[layer_idx, :, :, :seq_len, :],
            self.v_cache[layer_idx, :, :, :seq_len, :]
        )

    def update(
        self,
        layer_idx: int,
        c_kv_new: torch_typing.Tensor | None = None,
        k_pe_new: torch_typing.Tensor | None = None,
        compressed_kv: torch_typing.Tensor | None = None,
        k_new: torch_typing.Tensor | None = None,
        v_new: torch_typing.Tensor | None = None,
    ) -> torch_typing.Tensor | tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update cache with new c_kv and k_pe and return full cached tensors.

        Supports both separate c_kv/k_pe inputs and unified compressed_kv input.
        If compressed_kv is provided, returns the full concatenated cache.
        Otherwise returns a tuple (c_kv_full, k_pe_full).
        
        For FP4 quantize_mode with k_new/v_new: packs K/V separately using _pack_fp4()
        and stores in _k_packed/_v_packed with separate scales.
        """
        if compressed_kv is not None:
            # Unified update (TrellisKVCache style)
            return self.update_compressed(layer_idx, compressed_kv)

        # Handle FP4 mode with separate K/V tensors
        if self.quantize_mode == "fp4" and k_new is not None and v_new is not None:
            return self._update_fp4_separate(layer_idx, k_new, v_new)

        if c_kv_new is None or k_pe_new is None:
            raise ValueError(
                "Must provide either (c_kv_new, k_pe_new) or compressed_kv, "
                "or (k_new, v_new) for FP4 mode")

        # Separate components update (MLAAttention style)
        # Concatenate for unified storage update
        combined_kv = torch.cat([c_kv_new, k_pe_new], dim=-1)
        full_kv = self.update_compressed(layer_idx, combined_kv)

        # Split back for return
        return full_kv[..., : self.kv_lora_rank], full_kv[..., self.kv_lora_rank:]

    def _update_fp4_separate(
        self,
        layer_idx: int,
        k_new: torch_typing.Tensor,
        v_new: torch_typing.Tensor,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update cache with separate K/V tensors using FP4 packing.
        
        Args:
            layer_idx: Layer index to update
            k_new: New key tensor [batch, heads, seq, head_dim] or [batch, seq, dim]
            v_new: New value tensor [batch, heads, seq, head_dim] or [batch, seq, dim]
            
        Returns:
            Tuple of (k_full, v_full) dequantized tensors with original shapes restored
        """
        # Handle different input shapes
        if k_new.dim() == 4:
            # [batch, heads, seq, head_dim]
            batch, heads, seq_len, head_dim = k_new.shape
            self._fp4_heads = heads
            self._fp4_head_dim = head_dim
        else:
            # [batch, seq, dim] -> treat as heads=1
            batch, seq_len, head_dim = k_new.shape
            heads = 1
            self._fp4_heads = None
            self._fp4_head_dim = None
        
        start = int(self._seq_lens[layer_idx, 0].item())
        end = start + seq_len
        self._ensure_capacity(end)
        
        # Lazily allocate FP4 buffers for separate K/V if needed
        self._ensure_fp4_separate_buffers(batch, heads, head_dim, head_dim)
        
        # Pack K and V separately using FP4
        # _pack_fp4 preserves leading dims: [batch, heads, seq, packed_dim] or [batch, seq, packed_dim]
        k_packed, k_scale = self._pack_fp4(k_new)
        v_packed, v_scale = self._pack_fp4(v_new)
        
        # Reshape/permute to [batch, seq, heads, dim] for storage
        if k_new.dim() == 4:
            k_packed = k_packed.permute(0, 2, 1, 3)
            v_packed = v_packed.permute(0, 2, 1, 3)
            k_scale = k_scale.permute(0, 2, 1, 3)
            v_scale = v_scale.permute(0, 2, 1, 3)
        else:
            # [batch, seq, dim] -> [batch, seq, 1, dim]
            k_packed = k_packed.unsqueeze(2)
            v_packed = v_packed.unsqueeze(2)
            k_scale = k_scale.unsqueeze(2)
            v_scale = v_scale.unsqueeze(2)
        
        # Store in separate packed buffers: [num_layers, batch, max_seq, heads, packed_dim]
        self._k_packed[layer_idx, :batch, start:end, :, :] = k_packed  # type: ignore[index]
        self._v_packed[layer_idx, :batch, start:end, :, :] = v_packed  # type: ignore[index]
        self._k_scales[layer_idx, :batch, start:end, :, :] = k_scale  # type: ignore[index]
        self._v_scales[layer_idx, :batch, start:end, :, :] = v_scale  # type: ignore[index]
        
        self._seq_lens[layer_idx, :batch] = end
        
        # Retrieve full cached K/V and restore original shapes
        return self._get_fp4_separate(layer_idx, end)

    def _get_fp4_separate(
        self, layer_idx: int, seq_len: int
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Retrieve and unpack separate K/V tensors for FP4 mode.
        
        Args:
            layer_idx: Layer index
            seq_len: Current sequence length
            
        Returns:
            Tuple of (k_full, v_full) dequantized tensors with original shapes restored
        """
        # Retrieve packed K/V: [batch, seq_len, heads, packed_dim]
        k_packed = self._k_packed[layer_idx, :, :seq_len, :, :]  # type: ignore[index]
        v_packed = self._v_packed[layer_idx, :, :seq_len, :, :]  # type: ignore[index]
        k_scales = self._k_scales[layer_idx, :, :seq_len, :, :]  # type: ignore[index]
        v_scales = self._v_scales[layer_idx, :, :seq_len, :, :]  # type: ignore[index]
        
        # Unpack and dequantize: returns [batch, seq_len, heads, cache_dim]
        k_full = self._unpack_fp4(k_packed, k_scales)
        v_full = self._unpack_fp4(v_packed, v_scales)
        
        # Restore original shape
        if hasattr(self, '_fp4_heads') and self._fp4_heads is not None:
            # Original: [batch, heads, seq, head_dim]
            # Current: [batch, seq, heads, head_dim]
            k_full = k_full.permute(0, 2, 1, 3)
            v_full = v_full.permute(0, 2, 1, 3)
        else:
            # Original: [batch, seq, dim]
            # Current: [batch, seq, 1, dim]
            k_full = k_full.squeeze(2)
            v_full = v_full.squeeze(2)
        
        return k_full, v_full

    def update_components(
        self,
        layer_idx: int,
        c_kv_new: torch_typing.Tensor,
        k_pe_new: torch_typing.Tensor,
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Update cache components and return full dequantized components."""
        res = self.update(layer_idx, c_kv_new=c_kv_new, k_pe_new=k_pe_new)
        assert isinstance(res, tuple)
        return res

    def update_compressed(
        self,
        layer_idx: int,
        compressed_kv: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Update cache with combined compressed KV [B, S, Rank+RoPE]."""
        batch, seq_len, input_dim = compressed_kv.shape

        if input_dim != self.cache_dim:
            raise ValueError(
                f"Input dimension {input_dim} != cache_dim {self.cache_dim}")

        start = int(self._seq_lens[layer_idx, 0].item())
        end = start + seq_len
        self._ensure_capacity(end)

        if self.quantize_mode == "int8":
            kv_q, kv_scales = self._quantize_int8(compressed_kv)
            self.kv_cache[layer_idx, :batch, start:end] = kv_q
            # type: ignore[index]
            self.kv_scales[layer_idx, :batch, start:end] = kv_scales
        elif self.quantize_mode == "fp8":
            kv_q, kv_scales = self._quantize_fp8(compressed_kv)
            self.kv_cache[layer_idx, :batch, start:end] = kv_q
            # type: ignore[index]
            self.kv_scales[layer_idx, :batch, start:end] = kv_scales
        elif self.quantize_mode == "fp8_e5m2":
            kv_q, kv_scales = self._quantize_fp8_e5m2(compressed_kv)
            self.kv_cache[layer_idx, :batch, start:end] = kv_q
            # type: ignore[index]
            self.kv_scales[layer_idx, :batch, start:end] = kv_scales
        elif self.quantize_mode == "fp4":
            kv_packed, kv_scale = self._pack_fp4(compressed_kv)
            self.kv_cache[layer_idx, :batch, start:end] = kv_packed
            # type: ignore[index]
            self.kv_scales[layer_idx, :batch, start:end] = kv_scale
        else:
            self.kv_cache[layer_idx, :batch,
                          start:end] = compressed_kv.to(self.dtype)

        self._seq_lens[layer_idx, :batch] = end

        # Retrieve full cache for this layer
        return self.get(layer_idx)  # type: ignore[return-value]

    def get(self, layer_idx: int) -> torch_typing.Tensor | None:
        """Retrieve full cached sequence for a layer."""
        seq_len = int(self._seq_lens[layer_idx, 0].item())
        if seq_len == 0:
            return None

        kv = self.kv_cache[layer_idx, :, :seq_len]

        if self.quantize_mode == "int8":
            kv = self._dequantize_int8(
                # type: ignore[index]
                kv, self.kv_scales[layer_idx, :, :seq_len])
        elif self.quantize_mode == "fp8":
            kv = self._dequantize_fp8(
                # type: ignore[index]
                kv, self.kv_scales[layer_idx, :, :seq_len])
        elif self.quantize_mode == "fp8_e5m2":
            kv = self._dequantize_fp8_e5m2(
                # type: ignore[index]
                kv, self.kv_scales[layer_idx, :, :seq_len])
        elif self.quantize_mode == "fp4":
            kv = self._unpack_fp4(
                # type: ignore[index]
                kv, self.kv_scales[layer_idx, :, :seq_len])

        return kv.contiguous()

    def _quantize_int8(self, tensor: torch_typing.Tensor) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Int8 per-token or per-channel symmetric quantization."""
        dim = -1 if self.fp8_scale_method == "channel" else (-1,)
        abs_max = tensor.abs().amax(dim=dim, keepdim=True).clamp(min=1e-5)
        scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale.to(self._scale_dtype)

    def _dequantize_int8(self, quantized: torch_typing.Tensor, scales: torch_typing.Tensor) -> torch_typing.Tensor:
        return (quantized.to(self.dtype) * scales.to(self.dtype))

    def _quantize_fp8(self, tensor: torch_typing.Tensor) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        dim = -1 if self.fp8_scale_method == "channel" else (-1,)
        abs_max = tensor.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
        scale = abs_max / _FP8_E4M3_MAX
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        quantized = torch.round(scaled / _FP8_E4M3_MAX * 127.0) + 128.0
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
        return quantized, scale.to(self._scale_dtype)

    def _dequantize_fp8(self, quantized: torch_typing.Tensor, scale: torch_typing.Tensor) -> torch_typing.Tensor:
        signed = quantized.float() - 128.0
        return (signed / 127.0 * _FP8_E4M3_MAX * scale.float()).to(self.dtype)

    def _quantize_fp8_e5m2(self, tensor: torch_typing.Tensor) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Quantize tensor to FP8 E5M2 format (training-compatible).
        
        FP8 E5M2 has wider dynamic range (max ~57344) compared to E4M3 (max 448),
        making it more suitable for gradients and training scenarios.
        """
        dim = -1 if self.fp8_scale_method == "channel" else (-1,)
        abs_max = tensor.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
        scale = abs_max / _FP8_E5M2_MAX
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -_FP8_E5M2_MAX, _FP8_E5M2_MAX)
        # Map to uint8 range [0, 255] with 128 as zero point
        quantized = torch.round(scaled / _FP8_E5M2_MAX * 127.0) + 128.0
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
        return quantized, scale.to(self._scale_dtype)

    def _dequantize_fp8_e5m2(self, quantized: torch_typing.Tensor, scale: torch_typing.Tensor) -> torch_typing.Tensor:
        """Dequantize FP8 E5M2 tensor to float."""
        signed = quantized.float() - 128.0
        return (signed / 127.0 * _FP8_E5M2_MAX * scale.float()).to(self.dtype)

    def _pack_fp4(
        self, tensor: torch_typing.Tensor
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Pack FP4 quantized tensor into uint32 format.

        FP4 E2M1 format: 2-bit exponent, 1-bit mantissa, 1-bit sign.
        Packs 8 FP4 values into 1 uint32 (4 bytes).

        Args:
            tensor: Input tensor to quantize [..., cache_dim]
            scales: Scale factors [..., 1]

        Returns:
            packed_uint32: Packed uint32 tensor [..., cache_dim // 8]
            scales_fp16: Scale factors in float16
        """
        require_torch("_pack_fp4")
        # Quantize: scale and convert to 4-bit indices (0-15)
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / 6.0
        scaled = tensor / scale
        scaled = torch.clamp(scaled, -6.0, 6.0)
        quantized = torch.round(scaled * 2.0).to(torch.int8)
        quantized = torch.clamp(quantized + 8, 0, 15).to(torch.uint8)

        # Reshape to pack 8 nibbles per uint32
        *dims, dim = tensor.shape
        reshaped = quantized.view(*dims, dim // 8, 8)

        packed = _vectorized_pack(reshaped)

        scales_fp16 = scale.to(torch.float16)
        return packed, scales_fp16

    def _unpack_fp4(
        self, packed: torch_typing.Tensor, scales: torch_typing.Tensor
    ) -> torch_typing.Tensor:
        """Unpack FP4 quantized tensor from uint32 format to float16.

        Args:
            packed: Packed uint32 tensor [batch, seq_len, cache_dim // 8]
            scales: Scale factors in float16 [batch, seq_len, 1] or [batch, seq_len, cache_dim]

        Returns:
            Dequantized float16 tensor [batch, seq_len, cache_dim]
        """
        require_torch("_unpack_fp4")
        return _vectorized_unpack(packed, scales).to(torch.float16)

    def reset(self) -> None:
        """Clear cache state and return tensors to pool for reuse.
        
        This method returns all cache tensors to the global pool and
        immediately re-allocates new tensors from the pool. This enables
        efficient buffer reuse while keeping the cache usable.
        """
        # Return cache tensors to pool
        _return_to_pool(self.kv_cache)
        _return_to_pool(self.kv_scales)
        _return_to_pool(self.k_nope_cache)
        _return_to_pool(self.v_cache)
        # Return FP4-specific buffers to pool
        _return_to_pool(self._k_packed)
        _return_to_pool(self._v_packed)
        _return_to_pool(self._k_scales)
        _return_to_pool(self._v_scales)
        
        # Re-allocate from pool (enables buffer reuse)
        self._allocate(self.max_seq_len)
        
        self._seq_lens.zero_()

    def advance(self, num_tokens: int = 1) -> None:
        """No-op: MLAKVCache auto-tracks position via update_compressed().

        This method exists for interface compatibility with KVCacheTorch
        which requires explicit position advancement. MLAKVCache tracks
        _seq_lens per-layer in update_compressed(), so advance() is not needed.
        """

    def memory_usage_mb(self) -> float:
        """Return memory usage in MB."""
        active_seq = int(self._seq_lens[:, 0].max().item())
        if active_seq == 0:
            return 0.0

        if self.quantize_mode == "none":
            bpe = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
            scale_elements = 0
        elif self.quantize_mode in ("int8", "fp8", "fp8_e5m2", "fp4"):
            # All quantized modes use 1 byte per element + scales
            bpe = 1
            scale_bpe = 2 if self._scale_dtype == torch.float16 else 4
            scale_dim = self.cache_dim if self.fp8_scale_method == "channel" else 1
            scale_elements = self.num_layers * self.batch_size * active_seq * scale_dim
        else:
            raise ValueError(f"Unknown quantize_mode: {self.quantize_mode}")

        kv_elements = self.num_layers * self.batch_size * active_seq * self.cache_dim
        if self.quantize_mode == "none":
            return kv_elements * bpe / 1024 / 1024
        return (kv_elements * bpe + scale_elements * scale_bpe) / 1024 / 1024

    def prefetch_layer(self, layer_idx: int) -> torch_typing.Tensor | None:
        """Warm GPU caches for a layer."""
        kv = self.get(layer_idx)
        if kv is not None:
            return kv[0, 0, 0]
        return None

    def get_layer_slices(self, layer_idx: int) -> tuple[torch_typing.Tensor, torch_typing.Tensor] | None:
        """Get raw slice views split into c_kv and k_pe components."""
        kv = self.get(layer_idx)
        if kv is None:
            return None
        return kv[..., : self.kv_lora_rank], kv[..., self.kv_lora_rank:]

    def get_layer_for_attention(self, layer_idx: int) -> torch_typing.Tensor | None:
        """Get cached KV in BHSD format for attention kernels (zero-copy transpose)."""
        kv = self.get(layer_idx)
        if kv is None:
            return None
        # Transpose from BSHD [batch, seq_len, cache_dim] to BHSD [batch, cache_dim, seq_len]
        return kv.permute(0, 2, 1)

    def get_snapshot(self) -> dict[str, torch_typing.Tensor]:
        """Get a snapshot of the KV cache state for prompt caching."""
        snapshot = {
            "kv_cache": self.kv_cache.clone(),
            "seq_lens": self._seq_lens.clone(),
        }
        if self.kv_scales is not None:
            snapshot["kv_scales"] = self.kv_scales.clone()
        return snapshot

    def restore_snapshot(self, snapshot: dict[str, torch_typing.Tensor]) -> None:
        """Restore KV cache state from a snapshot."""
        self.kv_cache.copy_(snapshot["kv_cache"])
        self._seq_lens.copy_(snapshot["seq_lens"])
        if self.kv_scales is not None and "kv_scales" in snapshot:
            self.kv_scales.copy_(snapshot["kv_scales"])

    # Backwards compatibility properties
    @property
    def c_kv(self) -> torch_typing.Tensor:
        return self.kv_cache[..., : self.kv_lora_rank].contiguous()

    @property
    def k_pe(self) -> torch_typing.Tensor:
        return self.kv_cache[..., self.kv_lora_rank:].contiguous()

    @property
    def c_kv_scales(self) -> torch_typing.Tensor | None:
        return self.kv_scales

    @property
    def k_pe_scales(self) -> torch_typing.Tensor | None:
        return self.kv_scales


@dataclass
class CompressedKVCache(MLAKVCache):
    """Extended MLA cache with sliding window and late quantization."""

    compression: str = "int8"
    quantization_start_seq_len: int = 128
    sliding_window_layer_threshold: int = 32
    sliding_window_size: int = 4096

    def __post_init__(self) -> None:
        super().__post_init__()
        self._compression_enabled: list[bool] = [False] * self.num_layers
        self._window_start = torch.zeros(
            self.num_layers, dtype=torch.long, device=self.device)

    def _should_use_sliding_window(self, layer_idx: int) -> bool:
        return layer_idx >= self.sliding_window_layer_threshold

    def update_compressed(self, layer_idx: int, compressed_kv: torch_typing.Tensor) -> torch_typing.Tensor:
        # Simple override for now, can be extended with full logic from Trellis version
        return super().update_compressed(layer_idx, compressed_kv)


# Trellis-specific wrapper preserving legacy Trellis semantics.
class TrellisKVCache(MLAKVCache):
    """Trellis-oriented MLA cache behavior wrapper."""

    @property
    def seq_lens(self) -> torch_typing.Tensor:
        """Per-layer, per-batch sequence lengths tensor."""
        return self._seq_lens

    @property
    def seq_len(self) -> int:
        """Committed sequence length of the last layer."""
        return int(self._seq_lens[-1, 0].item())

    def memory_usage_mb(self) -> float:
        """Return allocated memory usage in MB (Trellis compatibility)."""
        if self.quantize_mode == "none":
            bpe = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
            scale_bpe = 0
        else:
            bpe = 1
            scale_bpe = 2 if self._scale_dtype == torch.float16 else 4

        kv_elements = self.num_layers * self.batch_size * \
            self.max_seq_len * self.cache_dim
        scale_elements = 0
        if self.kv_scales is not None:
            scale_elements = self.kv_scales.numel()

        return (kv_elements * bpe + scale_elements * scale_bpe) / 1024 / 1024


# Legacy aliases
KVCache = KVCacheTorch
CacheConfig = CacheConfigTorch

__all__ = [
    "KVCacheTorch",
    "CacheConfigTorch",

    "KVCache",
    "CacheConfig",

    "MLAKVCache",
    "CompressedKVCache",
    "TrellisKVCache",

    "require_mps",
    "clear_pool",
    "get_pool_stats",
    "reset_pool_metrics",
    "_kv_prealloc",
]
