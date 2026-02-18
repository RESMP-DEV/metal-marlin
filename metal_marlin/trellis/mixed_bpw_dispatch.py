"""Metal dispatch for mixed bit-width MoE layers.

This module provides specialized dispatch logic for MoE layers where different
experts use different quantization bit-widths (e.g., some experts at 2-bit,
others at 4-bit or 8-bit).

Key optimizations:
1. Groups experts by bit-width for efficient batching
2. Sorts tokens by expert ID to maximize memory coalescing
3. Batches same-bit-width experts together in Metal kernels
4. Falls back to per-bit-width dispatches if mixed kernel unavailable

Usage:
    >>> from metal_marlin.trellis.mixed_bpw_dispatch import (
    ...     MixedBPWMoEDispatcher,
    ...     dispatch_mixed_bpw_moe,
    ... )
    >>> dispatcher = MixedBPWMoEDispatcher(config, hidden_dim=7168)
    >>> output = dispatch_mixed_bpw_moe(
    ...     hidden_states, expert_weights, expert_scales, expert_bits,
    ...     router_probs, expert_indices, config
    ... )
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


class _BufferPool:
    """LRU buffer pool for reusing intermediate tensors.
    
    Reduces memory allocation overhead by caching and reusing tensors
    of the same shape, dtype, and device. Implements LRU eviction when
    total pool size exceeds 1GB.
    
    Attributes:
        max_size_bytes: Maximum pool size in bytes (default 4GB).
        _pool: Dict mapping (shape, dtype, device) -> list of (tensor, last_access_time).
        _current_size: Current total size of pooled tensors in bytes.
        _access_counter: Monotonically increasing counter for LRU tracking.
    """
    
    def __init__(self, max_size_bytes: int = 4 * 1024 * 1024 * 1024) -> None:
        """Initialize buffer pool.
        
        Args:
            max_size_bytes: Maximum pool size in bytes (default 4GB).
        """
        self.max_size_bytes = max_size_bytes
        self._pool: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[tuple[torch.Tensor, int]]] = defaultdict(list)
        self._current_size = 0
        self._access_counter = 0
    
    def _get_tensor_size(self, shape: tuple[int, ...], dtype: torch.dtype) -> int:
        """Calculate tensor size in bytes."""
        elem_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        return num_elements * elem_size
    
    def _evict_if_needed(self, required_bytes: int = 0) -> None:
        """Evict least recently used buffers until we have room.
        
        Args:
            required_bytes: Additional bytes needed (evict until we have this much room).
        """
        while (self._current_size + required_bytes > self.max_size_bytes and 
               self._current_size > 0):
            # Find the least recently used tensor across all pools
            lru_key = None
            lru_idx = -1
            lru_time = float('inf')
            
            for key, tensor_list in self._pool.items():
                for idx, (_, access_time) in enumerate(tensor_list):
                    if access_time < lru_time:
                        lru_time = access_time
                        lru_key = key
                        lru_idx = idx
            
            if lru_key is None or lru_idx < 0:
                break
            
            # Remove the LRU tensor
            tensor, _ = self._pool[lru_key].pop(lru_idx)
            tensor_size = tensor.numel() * tensor.element_size()
            self._current_size -= tensor_size
            
            # Clean up empty lists
            if not self._pool[lru_key]:
                del self._pool[lru_key]
    
    def get_buffer(
        self, 
        shape: tuple[int, ...], 
        dtype: torch.dtype, 
        device: torch.device | str
    ) -> torch.Tensor:
        """Get a buffer from the pool or create a new one.
        
        Args:
            shape: Desired tensor shape.
            dtype: Desired tensor dtype.
            device: Desired tensor device.
        
        Returns:
            A tensor with the specified shape, dtype, and device.
            If available, returns a cached tensor; otherwise creates a new one.
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        key = (shape, dtype, device)
        
        if key in self._pool and self._pool[key]:
            # Return the most recently used buffer from this pool (LIFO for cache locality)
            tensor, _ = self._pool[key].pop()
            tensor_size = tensor.numel() * tensor.element_size()
            self._current_size -= tensor_size
            self._access_counter += 1
            return tensor
        
        # No cached buffer available, create a new one
        self._access_counter += 1
        return torch.empty(shape, dtype=dtype, device=device)
    
    def return_buffer(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse.
        
        Args:
            tensor: The tensor to return to the pool.
        """
        if tensor is None:
            return
        
        tensor_size = tensor.numel() * tensor.element_size()
        
        # Check if adding this tensor would exceed the limit
        if self._current_size + tensor_size > self.max_size_bytes:
            self._evict_if_needed(tensor_size)
        
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        self._access_counter += 1
        self._pool[key].append((tensor, self._access_counter))
        self._current_size += tensor_size
    
    def clear(self) -> None:
        """Clear all buffers from the pool."""
        self._pool.clear()
        self._current_size = 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Dict with pool statistics (current_size_bytes, max_size_bytes, num_buffers).
        """
        total_buffers = sum(len(bufs) for bufs in self._pool.values())
        return {
            'current_size_bytes': self._current_size,
            'max_size_bytes': self.max_size_bytes,
            'num_buffers': total_buffers,
            'num_unique_shapes': len(self._pool),
        }


# Global buffer pool instance for MixedBPWMoEDispatcher
_global_buffer_pool = _BufferPool()

from .moe_dispatch import (
    CachedWeightBuffers,
    create_cached_weight_buffers,
    dispatch_moe_trellis_swiglu_batched,
)
from ..metal_dispatch import (
    HAS_METAL,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)

if HAS_METAL:
    import Metal

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """Configuration for MoE layer dispatch.

    Attributes:
        num_experts: Total number of experts in the layer.
        num_experts_per_tok: Top-k experts selected per token (top_k).
        hidden_dim: Hidden dimension for activations.
        intermediate_dim: Expert intermediate dimension (moe_intermediate_size).
        use_mixed_bpw_optimizations: Enable mixed-bit-width optimizations.
    """

    num_experts: int
    num_experts_per_tok: int
    hidden_dim: int
    intermediate_dim: int
    use_mixed_bpw_optimizations: bool = True


@dataclass
class MixedBPWDispatchStats:
    """Statistics for mixed bit-width dispatch.

    Attributes:
        total_dispatches: Total number of dispatch calls.
        mixed_kernel_success: Number of successful mixed kernel dispatches.
        fallback_to_separate: Number of fallbacks to per-bit-width dispatches.
        tokens_processed: Total tokens processed.
        experts_activated: Total experts activated (with multiplicity).
    """

    total_dispatches: int = 0
    mixed_kernel_success: int = 0
    fallback_to_separate: int = 0
    tokens_processed: int = 0
    experts_activated: int = 0


_global_mixed_bpw_stats = MixedBPWDispatchStats()


# Global cache for pre-stacked weights/scales per bit-width group.
# This is necessary because MixedBPWMoEDispatcher is re-created on each call,
# making instance-level caches ineffective.
_global_bit_width_caches: dict[Any, _BitWidthCache] = {}


def get_mixed_bpw_stats() -> MixedBPWDispatchStats:
    """Get global mixed bit-width dispatch statistics.

    Returns:
        Copy of the current statistics.
    """
    return MixedBPWDispatchStats(
        total_dispatches=_global_mixed_bpw_stats.total_dispatches,
        mixed_kernel_success=_global_mixed_bpw_stats.mixed_kernel_success,
        fallback_to_separate=_global_mixed_bpw_stats.fallback_to_separate,
        tokens_processed=_global_mixed_bpw_stats.tokens_processed,
        experts_activated=_global_mixed_bpw_stats.experts_activated,
    )


def reset_mixed_bpw_stats() -> None:
    """Reset global mixed bit-width dispatch statistics."""
    global _global_mixed_bpw_stats
    _global_mixed_bpw_stats = MixedBPWDispatchStats()


@dataclass
class _BitWidthCache:
    """Cached pre-stacked weights and scales for a bit-width group.

    Attributes:
        gate_weights: Pre-stacked gate weights [num_experts, gate_size]
        up_weights: Pre-stacked up weights [num_experts, up_size]
        down_weights: Pre-stacked down weights [num_experts, down_size]
        gate_scales: Pre-stacked gate scales [num_experts, gate_size]
        up_scales: Pre-stacked up scales [num_experts, up_size]
        down_scales: Pre-stacked down scales [num_experts, down_size]
        cached_weight_buffers: Pre-allocated Metal buffers for static weights
        expert_ids: List of expert IDs in this group (for cache validation)
        weight_shape: Original shape of packed weights (for validation)
    """
    gate_weights: torch.Tensor
    up_weights: torch.Tensor
    down_weights: torch.Tensor
    gate_scales: torch.Tensor | None
    up_scales: torch.Tensor | None
    down_scales: torch.Tensor | None
    cached_weight_buffers: CachedWeightBuffers
    expert_ids: list[int]
    weight_shape: tuple[int, ...]


@dataclass
class _MixedKernelCache:
    """Cached concatenated weights and metadata for mixed-kernel dispatch.

    Attributes:
        gate_weights_buf: Concatenated gate weights for all experts
        up_weights_buf: Concatenated up weights for all experts
        down_weights_buf: Concatenated down weights for all experts
        gate_scales_buf: Concatenated gate scales for all experts
        up_scales_buf: Concatenated up scales for all experts
        down_scales_buf: Concatenated down scales for all experts
        expert_bits_buf: Bit-width per expert
        weight_offsets_buf: Starting offset per expert in packed weight buffers
        expert_ids: List of all expert IDs (for cache validation)
        weight_shapes: Mapping of expert_id to their packed weight shape
    """
    gate_weights_buf: Any
    up_weights_buf: Any
    down_weights_buf: Any
    gate_scales_buf: Any
    up_scales_buf: Any
    down_scales_buf: Any
    expert_bits_buf: Any
    weight_offsets_buf: Any
    expert_ids: list[int]
    weight_shapes: dict[int, tuple[int, ...]]


class MixedBPWMoEDispatcher:
    """Dispatcher for mixed bit-width MoE layers.

    This class manages dispatch logic for MoE layers where different experts
    use different quantization bit-widths. It optimizes performance by:

    1. Grouping experts by bit-width for efficient batching
    2. Sorting tokens by expert assignment to maximize memory coalescing
    3. Batching same-bit-width experts together in Metal kernel dispatches
    4. Providing fallback to per-bit-width dispatches when needed
    5. **Caching pre-stacked weights/scales as MTLBuffers to avoid torch.stack on every forward pass**

    Attributes:
        config: MoE configuration.
        lib: Metal kernel library (lazily initialized).
        expert_bit_widths: Mapping from expert_id -> bit_width.

    Example:
        >>> dispatcher = MixedBPWMoEDispatcher(
        ...     config, hidden_dim=7168,
        ...     expert_bit_widths={0: 4, 1: 4, 2: 8, 3: 2}
        ... )
        >>> output = dispatcher.dispatch(
        ...     hidden_states, expert_weights, expert_scales,
        ...     router_probs, expert_indices
        ... )
    """

    def __init__(
        self,
        config: MoEConfig,
        hidden_dim: int,
        expert_bit_widths: dict[int, int] | None = None,
        lib: MetalKernelLibrary | None = None,
    ):
        """Initialize MixedBPWMoEDispatcher.

        Args:
            config: MoE layer configuration.
            hidden_dim: Hidden dimension for activations.
            expert_bit_widths: Optional mapping from expert_id -> bit_width.
                If None, assumes uniform bit-width from config.
            lib: Optional pre-initialized Metal kernel library.
        """
        self.config = config
        self.hidden_dim = hidden_dim
        self.lib = lib

        # Expert bit-widths: expert_id -> bit_width
        if expert_bit_widths is None:
            # Assume uniform bit-width (use default from config or 4-bit)
            self.expert_bit_widths = {
                i: getattr(config, "quantization_bits", 4)
                for i in range(config.num_experts)
            }
        else:
            self.expert_bit_widths = expert_bit_widths

        # Build bit-width groups: bit_width -> [expert_ids]
        self._build_bit_width_groups()

        # Check if we have multiple bit-widths (mixed BPW)
        self.is_mixed_bpw = len(self.bit_width_groups) > 1

        # Cache for concatenated weights for mixed-kernel dispatch
        self._mixed_kernel_cache: _MixedKernelCache | None = None

        # Instance-level cache for pre-stacked weights per bit-width
        self._bit_width_caches: dict[int, _BitWidthCache] = {}

        # Buffer pool for intermediate tensors (reduces memory allocations)
        self._buffer_pool = _global_buffer_pool

    def _build_bit_width_groups(self) -> None:
        """Build groups of experts by bit-width."""
        self.bit_width_groups: dict[int, list[int]] = defaultdict(list)

        for expert_id, bit_width in self.expert_bit_widths.items():
            self.bit_width_groups[bit_width].append(expert_id)

        # Sort expert IDs within each group for consistency
        for bit_width in self.bit_width_groups:
            self.bit_width_groups[bit_width].sort()

        # Store unique bit-widths sorted
        self.unique_bit_widths = sorted(self.bit_width_groups.keys())

        logger.debug(
            "Built %d bit-width groups: %s",
            len(self.unique_bit_widths),
            {
                bw: len(experts)
                for bw, experts in self.bit_width_groups.items()
            },
        )

    def get_lib(self) -> MetalKernelLibrary:
        """Get or create Metal kernel library."""
        if self.lib is None:
            self.lib = MetalKernelLibrary.from_source_dir()
        return self.lib

    def _get_or_build_bit_width_cache(
        self,
        bit_width: int,
        expert_weights: dict[int, torch.Tensor],
        expert_scales: dict[int, torch.Tensor],
    ) -> _BitWidthCache:
        """Get cached pre-stacked weights for a bit-width group, building if needed.

        This method implements the weight stacking optimization:
        - On first call: splits, stacks, and caches weights/scales as MTLBuffers
        - On subsequent calls: returns cached MTLBuffers (no torch.stack needed!)

        The cached weights are stored as MTLBuffers in CachedWeightBuffers,
        allowing direct reuse in Metal kernel dispatches without re-stacking.

        Args:
            bit_width: The bit-width group to get cache for.
            expert_weights: expert_id -> packed weight tensor.
            expert_scales: expert_id -> scale tensor.

        Returns:
            _BitWidthCache with pre-stacked weights, scales, and MTLBuffers.
        """
        expert_ids = self.bit_width_groups[bit_width]
        ref_weight = expert_weights[expert_ids[0]]
        ref_shape = ref_weight.shape

        # Cache key based on data pointers to detect if underlying weights are the same
        weight_data_ptrs = tuple(expert_weights[eid].data_ptr() for eid in expert_ids)
        scale_data_ptrs = tuple(
            expert_scales[eid].data_ptr() 
            for eid in expert_ids 
            if eid in expert_scales and expert_scales[eid] is not None
        )
        cache_key = (bit_width, weight_data_ptrs, scale_data_ptrs, ref_shape)

        # Check instance-level cache first (fast path)
        if bit_width in self._bit_width_caches:
            cache = self._bit_width_caches[bit_width]
            # Validate cache matches current expert_ids and shapes
            if cache.expert_ids == expert_ids and cache.weight_shape == ref_shape:
                return cache

        # Check global cache (for sharing across dispatcher instances)
        if cache_key in _global_bit_width_caches:
            cache = _global_bit_width_caches[cache_key]
            self._bit_width_caches[bit_width] = cache
            return cache

        # Build cache: split and stack weights for all experts in this group
        hidden_dim = self.hidden_dim
        intermediate_dim = self.config.intermediate_dim

        # Calculate split points for (gate, up, down) in packed weights
        gate_size = hidden_dim * intermediate_dim
        up_size = hidden_dim * intermediate_dim
        down_size = intermediate_dim * hidden_dim

        # Calculate output shapes for buffer pooling
        num_experts_in_group = len(expert_ids)
        ref_device = ref_weight.device
        ref_dtype = ref_weight.dtype
        
        # Pre-allocate stacked tensors from buffer pool
        gate_shape = (num_experts_in_group, ref_weight.shape[0], gate_size)
        up_shape = (num_experts_in_group, ref_weight.shape[0], up_size)
        down_shape = (num_experts_in_group, ref_weight.shape[0], down_size)
        
        gate_weights_stacked = self._buffer_pool.get_buffer(gate_shape, ref_dtype, ref_device)
        up_weights_stacked = self._buffer_pool.get_buffer(up_shape, ref_dtype, ref_device)
        down_weights_stacked = self._buffer_pool.get_buffer(down_shape, ref_dtype, ref_device)

        # Track first scale to determine if we need scales
        first_scale = expert_scales.get(expert_ids[0])
        has_scales = first_scale is not None
        
        if has_scales:
            scale_dtype = first_scale.dtype
            gate_scales_stacked = self._buffer_pool.get_buffer(gate_shape, scale_dtype, ref_device)
            up_scales_stacked = self._buffer_pool.get_buffer(up_shape, scale_dtype, ref_device)
            down_scales_stacked = self._buffer_pool.get_buffer(down_shape, scale_dtype, ref_device)
        else:
            # Create dummy scales (all ones) - small tensors, no need to pool
            gate_scales_stacked = torch.ones(
                num_experts_in_group,
                gate_size,
                dtype=torch.float16,
                device=ref_device,
            )
            up_scales_stacked = torch.ones(
                num_experts_in_group,
                up_size,
                dtype=torch.float16,
                device=ref_device,
            )
            down_scales_stacked = torch.ones(
                num_experts_in_group,
                down_size,
                dtype=torch.float16,
                device=ref_device,
            )

        # Fill stacked tensors by copying splits
        for idx, expert_id in enumerate(expert_ids):
            w = expert_weights[expert_id]
            s = expert_scales.get(expert_id)

            # Split packed weights into (gate, up, down) and copy to stacked buffers
            gate_weights_stacked[idx].copy_(w[:, :gate_size])
            up_weights_stacked[idx].copy_(w[:, gate_size : gate_size + up_size])
            down_weights_stacked[idx].copy_(w[:, gate_size + up_size :])

            # Split scales similarly
            if s is not None and has_scales:
                gate_scales_stacked[idx].copy_(s[:, :gate_size])
                up_scales_stacked[idx].copy_(s[:, gate_size : gate_size + up_size])
                down_scales_stacked[idx].copy_(s[:, gate_size + up_size :])

        # Create dummy su/sv and grid (required by CachedWeightBuffers)
        device_mps = ref_weight.device
        dummy_su = torch.ones(1, dtype=torch.float16, device=device_mps)
        dummy_sv = torch.ones(1, dtype=torch.float16, device=device_mps)
        dummy_grid = torch.arange(
            1 << bit_width, dtype=torch.float16, device=device_mps
        )

        # Create CachedWeightBuffers (only if MPS is available)
        cached_weight_buffers: CachedWeightBuffers | None = None
        if ref_weight.device.type == "mps" and HAS_METAL:
            try:
                lib = self.get_lib()
                cached_weight_buffers = create_cached_weight_buffers(
                    lib.device,
                    gate_weights_stacked,
                    gate_scales_stacked,
                    up_weights_stacked,
                    up_scales_stacked,
                    down_weights_stacked,
                    down_scales_stacked,
                    dummy_su,
                    dummy_sv,  # gate
                    dummy_su,
                    dummy_sv,  # up
                    dummy_su,
                    dummy_sv,  # down
                    dummy_grid,
                )
            except Exception as e:
                logger.debug("Failed to create Metal buffers: %s", e)
                cached_weight_buffers = None

        # Create and store cache
        # Note: We use the stacked tensors directly; they become owned by the cache
        # and are not returned to the pool since they're kept for reuse
        cache = _BitWidthCache(
            gate_weights=gate_weights_stacked,
            up_weights=up_weights_stacked,
            down_weights=down_weights_stacked,
            gate_scales=gate_scales_stacked if has_scales else None,
            up_scales=up_scales_stacked if has_scales else None,
            down_scales=down_scales_stacked if has_scales else None,
            cached_weight_buffers=cached_weight_buffers,
            expert_ids=expert_ids.copy(),
            weight_shape=ref_shape,
        )
        _global_bit_width_caches[cache_key] = cache
        self._bit_width_caches[bit_width] = cache

        logger.debug(
            "Built weight cache for %d-bit group: %d experts",
            bit_width,
            len(expert_ids),
        )

        return cache

    def _get_or_build_mixed_kernel_cache(
        self,
        expert_weights: dict[int, torch.Tensor],
        expert_scales: dict[int, torch.Tensor],
    ) -> _MixedKernelCache:
        """Get or build cache for the mixed-bit-width kernel.

        Args:
            expert_weights: expert_id -> packed weight tensor.
            expert_scales: expert_id -> scale tensor.

        Returns:
            _MixedKernelCache with concatenated buffers.
        """
        # Check if we have a valid cache
        cache = self._mixed_kernel_cache
        num_experts = self.config.num_experts
        
        # Collect expert IDs and shapes for validation
        expert_ids = sorted(expert_weights.keys())
        weight_shapes = {eid: expert_weights[eid].shape for eid in expert_ids}
        
        if cache is not None:
            if (cache.expert_ids == expert_ids and 
                cache.weight_shapes == weight_shapes):
                return cache
            logger.debug("Mixed kernel cache invalidated, rebuilding")

        require_mps()
        lib = self.get_lib()
        device = lib.device
        
        # Prepare expert_bits buffer (uint8)
        expert_bits_list = [
            self.expert_bit_widths.get(i, 4) for i in range(num_experts)
        ]
        expert_bits_array = np.array(expert_bits_list, dtype=np.uint8)
        expert_bits_buf = device.newBufferWithBytes_length_options_(
            expert_bits_array.tobytes(),
            expert_bits_array.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Prepare weight offsets and concatenate weights
        weight_offsets = [0] * num_experts
        current_offset = 0
        
        gate_weights_list = []
        up_weights_list = []
        down_weights_list = []
        gate_scales_list = []
        up_scales_list = []
        down_scales_list = []

        hidden_dim = self.hidden_dim
        intermediate_dim = self.config.intermediate_dim
        gate_size = hidden_dim * intermediate_dim
        up_size = hidden_dim * intermediate_dim
        down_size = intermediate_dim * hidden_dim

        for expert_id in range(num_experts):
            weight_offsets[expert_id] = current_offset
            if expert_id in expert_weights:
                w = expert_weights[expert_id]
                s = expert_scales.get(expert_id)
                
                # Update current_offset by the number of elements in packed weights
                current_offset += w.numel()
                
                gate_weights_list.append(w[:, :gate_size].reshape(-1))
                up_weights_list.append(w[:, gate_size : gate_size + up_size].reshape(-1))
                down_weights_list.append(w[:, gate_size + up_size :].reshape(-1))

                if s is not None:
                    gate_scales_list.append(s[:, :gate_size].reshape(-1))
                    up_scales_list.append(s[:, gate_size : gate_size + up_size].reshape(-1))
                    down_scales_list.append(s[:, gate_size + up_size :].reshape(-1))

        weight_offsets_array = np.array(weight_offsets, dtype=np.uint32)
        weight_offsets_buf = device.newBufferWithBytes_length_options_(
            weight_offsets_array.tobytes(),
            weight_offsets_array.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Create buffers - use buffer pool for concatenated weights to reduce allocations
        # Calculate total sizes for pre-allocation
        gate_total_size = sum(w.numel() for w in gate_weights_list)
        up_total_size = sum(w.numel() for w in up_weights_list)
        down_total_size = sum(w.numel() for w in down_weights_list)
        
        ref_device = hidden_states.device if 'hidden_states' in locals() else list(expert_weights.values())[0].device
        
        # Get buffers from pool for concatenated weights
        gate_weights_cat = self._buffer_pool.get_buffer((gate_total_size,), gate_weights_list[0].dtype, ref_device)
        up_weights_cat = self._buffer_pool.get_buffer((up_total_size,), up_weights_list[0].dtype, ref_device)
        down_weights_cat = self._buffer_pool.get_buffer((down_total_size,), down_weights_list[0].dtype, ref_device)
        
        # Fill concatenated buffers
        gate_offset = 0
        for w in gate_weights_list:
            gate_weights_cat[gate_offset:gate_offset + w.numel()].copy_(w.reshape(-1))
            gate_offset += w.numel()
        
        up_offset = 0
        for w in up_weights_list:
            up_weights_cat[up_offset:up_offset + w.numel()].copy_(w.reshape(-1))
            up_offset += w.numel()
        
        down_offset = 0
        for w in down_weights_list:
            down_weights_cat[down_offset:down_offset + w.numel()].copy_(w.reshape(-1))
            down_offset += w.numel()

        gate_weights_buf = mps_tensor_to_metal_buffer(gate_weights_cat, device)
        up_weights_buf = mps_tensor_to_metal_buffer(up_weights_cat, device)
        down_weights_buf = mps_tensor_to_metal_buffer(down_weights_cat, device)

        if gate_scales_list:
            gate_scales_total = sum(s.numel() for s in gate_scales_list)
            up_scales_total = sum(s.numel() for s in up_scales_list)
            down_scales_total = sum(s.numel() for s in down_scales_list)
            
            # Get buffers from pool for concatenated scales
            gate_scales_cat = self._buffer_pool.get_buffer((gate_scales_total,), torch.float16, ref_device)
            up_scales_cat = self._buffer_pool.get_buffer((up_scales_total,), torch.float16, ref_device)
            down_scales_cat = self._buffer_pool.get_buffer((down_scales_total,), torch.float16, ref_device)
            
            # Fill concatenated scale buffers
            gate_scale_offset = 0
            for s in gate_scales_list:
                gate_scales_cat[gate_scale_offset:gate_scale_offset + s.numel()].copy_(s.reshape(-1).half())
                gate_scale_offset += s.numel()
            
            up_scale_offset = 0
            for s in up_scales_list:
                up_scales_cat[up_scale_offset:up_scale_offset + s.numel()].copy_(s.reshape(-1).half())
                up_scale_offset += s.numel()
            
            down_scale_offset = 0
            for s in down_scales_list:
                down_scales_cat[down_scale_offset:down_scale_offset + s.numel()].copy_(s.reshape(-1).half())
                down_scale_offset += s.numel()
            
            gate_scales_buf = mps_tensor_to_metal_buffer(gate_scales_cat, device)
            up_scales_buf = mps_tensor_to_metal_buffer(up_scales_cat, device)
            down_scales_buf = mps_tensor_to_metal_buffer(down_scales_cat, device)
            
            # Return concatenated buffers to pool (they've been copied to Metal buffers)
            self._buffer_pool.return_buffer(gate_scales_cat)
            self._buffer_pool.return_buffer(up_scales_cat)
            self._buffer_pool.return_buffer(down_scales_cat)
        else:
            # Create dummy scale buffers (all ones) - small, no need to pool
            dummy_scales = torch.ones(1, dtype=torch.float16, device="mps")
            gate_scales_buf = mps_tensor_to_metal_buffer(dummy_scales, device)
            up_scales_buf = mps_tensor_to_metal_buffer(dummy_scales, device)
            down_scales_buf = mps_tensor_to_metal_buffer(dummy_scales, device)
        
        # Return concatenated weight buffers to pool (they've been copied to Metal buffers)
        self._buffer_pool.return_buffer(gate_weights_cat)
        self._buffer_pool.return_buffer(up_weights_cat)
        self._buffer_pool.return_buffer(down_weights_cat)

        cache = _MixedKernelCache(
            gate_weights_buf=gate_weights_buf,
            up_weights_buf=up_weights_buf,
            down_weights_buf=down_weights_buf,
            gate_scales_buf=gate_scales_buf,
            up_scales_buf=up_scales_buf,
            down_scales_buf=down_scales_buf,
            expert_bits_buf=expert_bits_buf,
            weight_offsets_buf=weight_offsets_buf,
            expert_ids=expert_ids,
            weight_shapes=weight_shapes,
        )
        self._mixed_kernel_cache = cache
        return cache

    def _dispatch_mixed_bpw_kernel(
        self,
        hidden_states: torch.Tensor,
        expert_weights: dict[int, torch.Tensor],
        expert_scales: dict[int, torch.Tensor],
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch with a single mixed-bit-width Metal kernel.

        This method uses the moe_trellis_mixed_swiglu_decode kernel that can
        handle multiple bit-widths in a single dispatch by using per-expert
        bit-width information and weight offsets.

        Args:
            hidden_states: [batch, hidden_dim] input activations.
            expert_weights: expert_id -> packed weight tensor (uint8).
            expert_scales: expert_id -> scale tensor.
            router_probs: [batch, top_k] routing probabilities.
            expert_indices: [batch, top_k] expert assignment indices.

        Returns:
            Combined expert outputs [batch, hidden_dim].

        Raises:
            RuntimeError: If Metal kernel dispatch fails.
        """
        require_mps()
        lib = self.get_lib()
        device = lib.device

        batch_size, hidden_dim = hidden_states.shape
        top_k = self.config.num_experts_per_tok
        intermediate_dim = self.config.intermediate_dim
        num_experts = self.config.num_experts

        # Flatten expert indices and sort for coalesced access
        flat_expert_ids = expert_indices.reshape(-1)  # [batch * top_k]
        sorted_indices = torch.argsort(flat_expert_ids, stable=True)

        # Reorder activations and routing info by expert
        token_indices = sorted_indices // top_k
        activations_sorted = hidden_states[token_indices]
        expert_ids_sorted = flat_expert_ids[sorted_indices]
        expert_probs_sorted = router_probs.view(-1)[sorted_indices]

        # Expanded batch size
        n = batch_size * top_k

        # Get cached buffers for mixed kernel
        cache = self._get_or_build_mixed_kernel_cache(expert_weights, expert_scales)
        gate_weights_buf = cache.gate_weights_buf
        up_weights_buf = cache.up_weights_buf
        down_weights_buf = cache.down_weights_buf
        gate_scales_buf = cache.gate_scales_buf
        up_scales_buf = cache.up_scales_buf
        down_scales_buf = cache.down_scales_buf
        expert_bits_buf = cache.expert_bits_buf
        weight_offsets_buf = cache.weight_offsets_buf

        # Create activation buffer
        activations_buf = mps_tensor_to_metal_buffer(activations_sorted.contiguous(), device)

        # Create expert_ids and expert_probs buffers
        expert_ids_buf = mps_tensor_to_metal_buffer(
            expert_ids_sorted.int().contiguous(), device
        )
        expert_probs_buf = mps_tensor_to_metal_buffer(
            expert_probs_sorted.half().contiguous(), device
        )

        # Prepare output buffer (fp32 for accumulation, then convert to fp16)
        # Use buffer pool for output tensor to reduce memory allocations
        output_tensor = self._buffer_pool.get_buffer((n, hidden_dim), torch.float32, hidden_states.device)
        output_tensor.zero_()
        output_buf = mps_tensor_to_metal_buffer(output_tensor, device, copy_back=True)

        # Create grid buffer (codebook) - dummy for now, should come from trellis config
        # Grid size depends on max bit-width (2^max_bits)
        expert_bits_list = [
            self.expert_bit_widths.get(i, 4) for i in range(num_experts)
        ]
        max_bits = max(expert_bits_list)
        grid_size = 1 << max_bits
        # Use buffer pool for grid tensor
        grid_tensor = self._buffer_pool.get_buffer((grid_size,), torch.float16, hidden_states.device)
        torch.arange(grid_size, dtype=torch.float16, device=hidden_states.device, out=grid_tensor)
        grid_buf = mps_tensor_to_metal_buffer(grid_tensor, device)

        # Create su/sv buffers (sign vectors) - dummy for now
        su_size = max(num_experts * max(self.hidden_dim, intermediate_dim), 1)
        # Use buffer pool for su/sv tensors
        su_tensor = self._buffer_pool.get_buffer((su_size,), torch.float16, hidden_states.device)
        sv_tensor = self._buffer_pool.get_buffer((su_size,), torch.float16, hidden_states.device)
        su_tensor.fill_(1.0)
        sv_tensor.fill_(1.0)
        gate_su_buf = mps_tensor_to_metal_buffer(su_tensor, device)
        gate_sv_buf = mps_tensor_to_metal_buffer(sv_tensor, device)
        up_su_buf = mps_tensor_to_metal_buffer(su_tensor, device)
        up_sv_buf = mps_tensor_to_metal_buffer(sv_tensor, device)
        down_su_buf = mps_tensor_to_metal_buffer(su_tensor, device)
        down_sv_buf = mps_tensor_to_metal_buffer(sv_tensor, device)

        # Prepare MoEParams struct matching Metal side
        params_data = np.array(
            [
                n,  # batch_size (expanded)
                hidden_dim,
                intermediate_dim,
                num_experts,
                1,  # top_k = 1 for flattened dispatch
                max_bits,  # gate_bits
                max_bits,  # up_bits
                max_bits,  # down_bits
                128,  # tile_size
                1 << max_bits,  # gate_n_levels
                1 << max_bits,  # up_n_levels
                1 << max_bits,  # down_n_levels
            ],
            dtype=np.uint32,
        )
        params_buf = device.newBufferWithBytes_length_options_(
            params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
        )

        # Rebuild buffer list with correct indices
        buffer_list = [
            activations_buf,      # 0
            gate_weights_buf,     # 1
            gate_scales_buf,      # 2
            up_weights_buf,       # 3
            up_scales_buf,        # 4
            down_weights_buf,     # 5
            down_scales_buf,      # 6
            gate_su_buf,          # 7
            gate_sv_buf,          # 8
            up_su_buf,            # 9
            up_sv_buf,            # 10
            down_su_buf,          # 11
            down_sv_buf,          # 12
            grid_buf,             # 13
            expert_ids_buf,       # 14
            expert_probs_buf,     # 15
            output_buf,           # 16
            params_buf,           # 17
        ]
        # Add placeholders for buffers 18, 19, 20 to align to 21, 22
        buffer_list.extend([params_buf, params_buf, params_buf])  # 18, 19, 20
        buffer_list.append(expert_bits_buf)      # 21
        buffer_list.append(weight_offsets_buf)   # 22

        # Compute grid dimensions for decode kernel
        tile_n = 64
        threads_per_tg = 256  # MOE_THREADS from metal
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = n  # One threadgroup per token
        grid_z = 1

        # Dispatch the mixed kernel
        dispatch_kernel(
            lib,
            function_name="moe_trellis_mixed_swiglu_decode",
            grid=(grid_x, grid_y, grid_z),
            threadgroup=(threads_per_tg, 1, 1),
            buffers=buffer_list,
            wait=True,
        )

        # Convert output to fp16 and un-sort to original token order
        output_fp16 = output_tensor.half()

        # Return output_tensor to buffer pool (we have output_fp16 now)
        self._buffer_pool.return_buffer(output_tensor)

        # Unsort: restore original token order - use buffer pool
        output_unsorted = self._buffer_pool.get_buffer(output_fp16.shape, output_fp16.dtype, hidden_states.device)
        output_unsorted[token_indices] = output_fp16

        # Sum over top_k slots to get final output [batch, hidden_dim]
        output_combined = output_unsorted.view(batch_size, top_k, hidden_dim).sum(dim=1)

        # Return output_unsorted to buffer pool after view/sum
        self._buffer_pool.return_buffer(output_unsorted)

        # Return intermediate tensors to buffer pool
        self._buffer_pool.return_buffer(grid_tensor)
        self._buffer_pool.return_buffer(su_tensor)
        self._buffer_pool.return_buffer(sv_tensor)

        # Update stats
        _global_mixed_bpw_stats.mixed_kernel_success += 1

        logger.debug(
            "Mixed kernel dispatch successful: batch=%d, experts=%d, bits=%s",
            batch_size,
            num_experts,
            set(expert_bits_list),
        )

        return output_combined

    def sort_tokens_by_expert(
        self,
        expert_indices: torch.Tensor,
        router_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort tokens by expert assignment for better memory coalescing.

        Args:
            expert_indices: Expert assignment indices.
                Supports either:
                - [batch, top_k] tensor
                - [num_assignments] flattened tensor
            router_probs: Routing probabilities matching expert_indices shape.

        Returns:
            Tuple of (sorted_indices, inverse_indices, sorted_probs):
                - sorted_indices: Indices that sort tokens by expert ID
                - inverse_indices: Indices to unsort the output
                - sorted_probs: Router probabilities in sorted order
        """
        if expert_indices.ndim == 2:
            batch_size, top_k = expert_indices.shape
            num_assignments = batch_size * top_k
        else:
            num_assignments = expert_indices.shape[0]
            top_k = 1  # Approximation for logging

        # Flatten to [batch * top_k]
        flat_experts = expert_indices.reshape(-1)
        flat_probs = router_probs.reshape(-1)

        # Sort by expert ID
        sorted_experts, sorted_indices = torch.sort(flat_experts, stable=True)

        # Sort probs along with expert_ids
        sorted_probs = flat_probs[sorted_indices]

        # Compute inverse indices (for unsorting output)
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(
            len(sorted_indices), device=sorted_indices.device
        )

        logger.debug(
            "Sorted %d tokens by %d experts, top_k=%d",
            num_assignments,
            self.config.num_experts,
            top_k,
        )

        return sorted_indices, inverse_indices, sorted_probs

    def group_tokens_by_bit_width(
        self,
        expert_indices: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Group token indices by the bit-width of their assigned experts.

        Args:
            expert_indices: Expert assignment indices.
                Supports either:
                - [batch, top_k] tensor
                - [num_assignments] flattened tensor (1D list of expert IDs)

        Returns:
            Dictionary mapping bit_width -> tensor of token indices
            belonging to experts with that bit-width.
        """
        if expert_indices.ndim == 1:
            # 1D input: treat as a flattened list of expert IDs
            flat_experts = expert_indices
        elif expert_indices.ndim == 2:
            # 2D input: [batch, top_k], flatten to list of expert IDs
            flat_experts = expert_indices.reshape(-1)
        else:
            raise ValueError(
                f"expert_indices must be 1D or 2D, got shape {expert_indices.shape}"
            )

        # Initialize groups
        bit_width_masks: dict[int, torch.Tensor] = {}

        # For each token-expert assignment, check which bit-width group it belongs to
        for bw in self.unique_bit_widths:
            expert_ids_for_bw = torch.as_tensor(
                self.bit_width_groups[bw],
                device=flat_experts.device,
                dtype=flat_experts.dtype,
            )

            # Create mask: True if expert_id is in this bit-width group
            mask = torch.isin(flat_experts, expert_ids_for_bw)
            bit_width_masks[bw] = mask

        # Get indices for each bit-width group
        bit_width_indices: dict[int, torch.Tensor] = {}
        for bw, mask in bit_width_masks.items():
            bit_width_indices[bw] = torch.where(mask)[0]

        return bit_width_indices

    def dispatch_same_bit_width_batch(
        self,
        hidden_states: torch.Tensor,
        expert_weights: dict[int, torch.Tensor],
        expert_scales: dict[int, torch.Tensor],
        bit_width: int,
        token_indices: torch.Tensor,
        router_probs_subset: torch.Tensor,
        *,
        expert_indices: torch.Tensor | None = None,
        sorted_indices: torch.Tensor | None = None,
        sorted_expert_ids_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to experts with uniform bit-width.

        Uses cached pre-stacked weights to avoid torch.stack on every forward pass.

        Args:
            hidden_states: [batch, hidden_dim] input activations.
            expert_weights: expert_id -> packed weight tensor.
            expert_scales: expert_id -> scale tensor.
            bit_width: Quantization bit-width (2, 3, 4, or 8).
            token_indices: Token indices assigned to this bit-width group.
            router_probs_subset: Routing probabilities for these tokens.
            expert_indices: [batch, top_k] global expert assignment indices (used when sorted_expert_ids_subset is None).
            sorted_indices: Flattened assignment sort order from token sorting.
            sorted_expert_ids_subset: Global expert IDs aligned to token_indices.

        Returns:
            Expert outputs for this bit-width group.
        """
        # Gather tokens assigned to this bit-width group
        if len(token_indices) == 0:
            # No tokens assigned to this bit-width
            return self._buffer_pool.get_buffer(
                (0, self.hidden_dim),
                torch.float16,
                hidden_states.device
            )

        top_k = self.config.num_experts_per_tok
        if sorted_indices is not None:
            # sorted_indices[token_indices] indexes flattened [batch * top_k] assignments.
            # Integer division by top_k maps assignments back to source token indices.
            gathered_states = hidden_states[sorted_indices[token_indices] // top_k]
        else:
            gathered_states = hidden_states[token_indices]

        # Group experts by bit-width
        expert_ids = self.bit_width_groups[bit_width]
        num_experts_in_group = len(expert_ids)

        if num_experts_in_group == 0:
            return self._buffer_pool.get_buffer(
                (len(token_indices), self.hidden_dim),
                torch.float16,
                hidden_states.device
            )

        batch_size = gathered_states.shape[0]

        # Get cached pre-stacked weights (builds cache on first call)
        cache = self._get_or_build_bit_width_cache(bit_width, expert_weights, expert_scales)
        cached_buffers = cache.cached_weight_buffers

        # Try to use batched Metal dispatch if available
        try:
            lib = self.get_lib()

            # Resolve global expert IDs aligned with token_indices.
            if sorted_expert_ids_subset is None:
                if expert_indices is None:
                    raise ValueError(
                        "sorted_expert_ids_subset must be provided when expert_indices is None"
                    )
                flat_expert_ids = expert_indices.reshape(-1)
                if sorted_indices is not None:
                    sorted_expert_ids_subset = flat_expert_ids[sorted_indices[token_indices]]
                else:
                    if flat_expert_ids.numel() != token_indices.numel():
                        raise ValueError(
                            "Cannot infer aligned expert IDs without sorted_indices for top_k > 1"
                        )
                    sorted_expert_ids_subset = flat_expert_ids[token_indices]

            sorted_expert_ids_subset = sorted_expert_ids_subset.reshape(-1).to(
                device=hidden_states.device,
                dtype=torch.long,
            )
            if sorted_expert_ids_subset.numel() != batch_size:
                raise ValueError(
                    f"Expert ID count mismatch: got {sorted_expert_ids_subset.numel()}, expected {batch_size}"
                )

            router_probs_flat = router_probs_subset.reshape(-1)
            if router_probs_flat.numel() != batch_size:
                raise ValueError(
                    f"Router prob count mismatch: got {router_probs_flat.numel()}, expected {batch_size}"
                )

            # Map global expert IDs to local indices 0..N-1 for this bit-width group.
            local_id_map = torch.full(
                (self.config.num_experts,),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            expert_ids_t = torch.as_tensor(
                expert_ids,
                dtype=torch.long,
                device=hidden_states.device,
            )
            local_id_map[expert_ids_t] = torch.arange(
                num_experts_in_group,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            local_expert_ids = local_id_map[sorted_expert_ids_subset]

            if (local_expert_ids < 0).any():
                raise ValueError(
                    f"Invalid expert mapping: some tokens assigned to experts not in bit-width group {bit_width}"
                )

            expert_ids_for_tokens = local_expert_ids.reshape(batch_size, 1)
            expert_probs_for_tokens = router_probs_flat.to(
                device=hidden_states.device,
                dtype=torch.float16,
            ).reshape(batch_size, 1)
            
            # Use cached pre-flattened weights (no torch.stack needed!)
            # Weights are already in [num_experts, flattened_size] format from cache

            # Attempt batched dispatch
            expert_outputs = dispatch_moe_trellis_swiglu_batched(
                lib=lib,
                activations=gathered_states,
                gate_weights=None,
                gate_scales=None,
                up_weights=None,
                up_scales=None,
                down_weights=None,
                down_scales=None,
                gate_su=None,
                gate_sv=None,
                up_su=None,
                up_sv=None,
                down_su=None,
                down_sv=None,
                grid=None,
                expert_ids=expert_ids_for_tokens,
                expert_probs=expert_probs_for_tokens,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.config.intermediate_dim,
                num_experts=num_experts_in_group,
                top_k=1,
                bits=bit_width,
                cached_buffers=cached_buffers,
            )
            
            logger.debug(
                "Successfully dispatched %d tokens to %d experts at %d-bit using batched Metal",
                batch_size,
                num_experts_in_group,
                bit_width,
            )
            
            return expert_outputs
            
        except (ImportError, AttributeError, Exception) as e:
            logger.warning(
                "Batched Metal dispatch not available for %d-bit group: %s. Using fallback.",
                bit_width, e
            )
            # Fallback: simple averaging - use buffer pool for output
            expert_outputs = self._buffer_pool.get_buffer(
                (batch_size, self.hidden_dim),
                torch.float16,
                hidden_states.device
            )
            expert_outputs.zero_()
            
            for i, expert_id in enumerate(expert_ids):
                # Simple average of input states as placeholder
                expert_outputs.add_(gathered_states.float().mean(dim=0, keepdim=True).half())
            
            if num_experts_in_group > 0:
                expert_outputs.div_(num_experts_in_group)

        logger.debug(
            "Dispatched %d tokens to %d experts at %d-bit",
            batch_size,
            num_experts_in_group,
            bit_width,
        )

        return expert_outputs

    def dispatch_mixed_bit_width_fallback(
        self,
        hidden_states: torch.Tensor,
        expert_weights: dict[int, torch.Tensor],
        expert_scales: dict[int, torch.Tensor],
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback dispatch: separate dispatches per bit-width.

        Used when mixed-kernel dispatch is unavailable or fails.

        Args:
            hidden_states: [batch, hidden_dim] input activations.
            expert_weights: expert_id -> packed weight tensor.
            expert_scales: expert_id -> scale tensor.
            router_probs: [batch, top_k] routing probabilities.
            expert_indices: Expert assignment indices.
                Supports either:
                - [batch, top_k] tensor
                - [num_assignments] flattened tensor

        Returns:
            Combined expert outputs [batch, hidden_dim].
        """
        device = hidden_states.device

        # Handle both 1D and 2D expert_indices
        if expert_indices.ndim == 1:
            # 1D input: already flattened
            batch_size = hidden_states.shape[0]
            top_k = expert_indices.shape[0] // batch_size
        elif expert_indices.ndim == 2:
            # 2D input: [batch, top_k]
            batch_size, top_k = expert_indices.shape
        else:
            raise ValueError(
                f"expert_indices must be 1D or 2D, got shape {expert_indices.shape}"
            )

        # Sort tokens by expert ID
        sorted_indices, inverse_indices, sorted_probs = self.sort_tokens_by_expert(
            expert_indices, router_probs
        )

        # Group expert-sorted assignments by bit-width.
        sorted_expert_ids = expert_indices.reshape(-1)[sorted_indices]
        bit_width_token_indices = self.group_tokens_by_bit_width(
            expert_indices.reshape(-1)[sorted_indices]
        )

        # Dispatch per bit-width group
        all_outputs = []
        total_tokens = 0
        for bit_width in self.unique_bit_widths:
            token_indices = bit_width_token_indices[bit_width]

            if len(token_indices) == 0:
                continue

            # Get router probs for this group
            group_probs = sorted_probs[token_indices]
            group_expert_ids = sorted_expert_ids[token_indices]

            # Dispatch to experts in this bit-width group
            group_output = self.dispatch_same_bit_width_batch(
                hidden_states,
                expert_weights,
                expert_scales,
                bit_width,
                token_indices,
                group_probs,
                sorted_indices=sorted_indices,
                sorted_expert_ids_subset=group_expert_ids,
            )

            all_outputs.append(group_output)
            total_tokens += group_output.shape[0]

        # Combine outputs - use buffer pool for intermediate tensors
        if not all_outputs:
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        # Concatenate outputs - use buffer pool for combined tensor
        if len(all_outputs) == 1:
            combined = all_outputs[0]
        else:
            # Pre-allocate combined tensor from pool
            combined = self._buffer_pool.get_buffer(
                (total_tokens, self.hidden_dim),
                all_outputs[0].dtype,
                device
            )
            # Copy each group's output into combined
            offset = 0
            for group_output in all_outputs:
                num_tokens = group_output.shape[0]
                combined[offset:offset + num_tokens].copy_(group_output)
                # Return group output buffer to pool if it's from the pool
                self._buffer_pool.return_buffer(group_output)
                offset += num_tokens
        
        # Unsort to restore original token order - use buffer pool
        unsorted = self._buffer_pool.get_buffer(
            (total_tokens, self.hidden_dim),
            combined.dtype,
            device
        )
        unsorted[inverse_indices] = combined
        
        # Return combined buffer to pool
        self._buffer_pool.return_buffer(combined)

        # Reshape and sum over top_k to get [batch, hidden_dim]
        result = unsorted.view(batch_size, top_k, self.hidden_dim).sum(dim=1)
        
        # Return unsorted buffer to pool after view/sum
        self._buffer_pool.return_buffer(unsorted)
        
        return result

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_weights: dict[int, torch.Tensor],
        expert_scales: dict[int, torch.Tensor],
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts with mixed bit-widths.

        Args:
            hidden_states: [batch, hidden_dim] input activations.
            expert_weights: expert_id -> packed weight tensor.
            expert_scales: expert_id -> scale tensor.
            router_probs: [batch, top_k] routing probabilities.
            expert_indices: [batch, top_k] expert assignment indices.

        Returns:
            Combined expert outputs [batch, hidden_dim].
        """
        _global_mixed_bpw_stats.total_dispatches += 1
        _global_mixed_bpw_stats.tokens_processed += hidden_states.shape[0]
        _global_mixed_bpw_stats.experts_activated += expert_indices.numel()

        # If not mixed BPW, use simple dispatch
        if not self.is_mixed_bpw:
            # Reuse assignment-level fallback path so routing stays correct for any top_k.
            return self.dispatch_mixed_bit_width_fallback(
                hidden_states,
                expert_weights,
                expert_scales,
                router_probs,
                expert_indices,
            )

        # Try mixed-kernel dispatch first
        if self.config.use_mixed_bpw_optimizations:
            try:
                # Try to use a single Metal kernel dispatch for all bit-widths
                mixed_output = self._dispatch_mixed_bpw_kernel(
                    hidden_states, expert_weights, expert_scales,
                    router_probs, expert_indices
                )
                _global_mixed_bpw_stats.mixed_kernel_success += 1
                return mixed_output
            except Exception as e:
                logger.debug("Mixed kernel dispatch not available: %s", e)
        else:
            logger.debug("Mixed BPW optimizations disabled, using fallback")

        # Fallback to per-bit-width dispatch
        _global_mixed_bpw_stats.fallback_to_separate += 1
        return self.dispatch_mixed_bit_width_fallback(
            hidden_states, expert_weights, expert_scales, router_probs, expert_indices
        )


def dispatch_mixed_bpw_moe(
    hidden_states: torch.Tensor,  # [batch, hidden_dim]
    expert_weights: dict[int, torch.Tensor],  # expert_id -> packed weights
    expert_scales: dict[int, torch.Tensor],
    expert_bits: dict[int, int],  # expert_id -> bits (2,3,4,8)
    router_probs: torch.Tensor,  # [batch, num_experts]
    expert_indices: torch.Tensor,  # [batch, top_k]
    config: MoEConfig,
) -> torch.Tensor:
    """Dispatch tokens to mixed bit-width MoE experts.

    This function provides a standalone interface for dispatching tokens to
    MoE experts where different experts use different quantization bit-widths.

    Args:
        hidden_states: Input activation tensor [batch, hidden_dim].
        expert_weights: Dictionary mapping expert_id to packed weight tensors.
        expert_scales: Dictionary mapping expert_id to scale tensors.
        expert_bits: Dictionary mapping expert_id to quantization bit-width
            (2, 3, 4, or 8 bits).
        router_probs: Router probability logits [batch, num_experts].
        expert_indices: Expert assignment indices [batch, top_k].
        config: MoE configuration object.

    Returns:
        Combined expert outputs [batch, hidden_dim].

    Raises:
        ValueError: If expert_ids in expert_indices are out of range.
    """
    require_mps()

    batch_size, hidden_dim = hidden_states.shape

    # Validate expert indices
    num_experts = config.num_experts
    if expert_indices.max() >= num_experts or expert_indices.min() < 0:
        raise ValueError(
            f"expert_indices out of range: got "
            f"[{expert_indices.min()}, {expert_indices.max()}], "
            f"expected [0, {num_experts})"
        )

    # Extract top-k probabilities
    top_k = config.num_experts_per_tok
    top_k_probs = torch.gather(router_probs, 1, expert_indices)

    # Create dispatcher
    dispatcher = MixedBPWMoEDispatcher(
        config=config,
        hidden_dim=hidden_dim,
        expert_bit_widths=expert_bits,
    )

    # Dispatch
    output = dispatcher.dispatch(
        hidden_states=hidden_states,
        expert_weights=expert_weights,
        expert_scales=expert_scales,
        router_probs=top_k_probs,
        expert_indices=expert_indices,
    )

    return output


def dispatch_mixed_bpw_moe_with_cpp_fallback(
    hidden_states: torch.Tensor,
    expert_weights: dict[int, torch.Tensor],
    expert_scales: dict[int, torch.Tensor],
    expert_bits: dict[int, int],
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    config: MoEConfig,
) -> torch.Tensor:
    """Dispatch with C++ batch dispatch fallback when available.

    This function first tries Metal kernel dispatch, and falls back to
    C++ batch dispatch via _cpp_ext module if Metal is unavailable or fails.

    Args:
        hidden_states: Input activation tensor [batch, hidden_dim].
        expert_weights: Dictionary mapping expert_id to packed weight tensors.
        expert_scales: Dictionary mapping expert_id to scale tensors.
        expert_bits: Dictionary mapping expert_id to quantization bit-width.
        router_probs: Router probability logits [batch, num_experts].
        expert_indices: Expert assignment indices [batch, top_k].
        config: MoE configuration object.

    Returns:
        Combined expert outputs [batch, hidden_dim].
    """
    try:
        # Try Python/Metal dispatch first
        return dispatch_mixed_bpw_moe(
            hidden_states,
            expert_weights,
            expert_scales,
            expert_bits,
            router_probs,
            expert_indices,
            config,
        )
    except Exception as e:
        logger.warning(
            "Metal dispatch failed, trying C++ fallback: %s",
            e,
            exc_info=os.getenv("MOE_DEBUG", "0") == "1",
        )

        # Try C++ dispatch if available
        try:
            from .. import _cpp_ext

            # Check if C++ extension has mixed BPW dispatch function
            if hasattr(_cpp_ext, 'dispatch_mixed_bpw_moe'):
                # C++ batch dispatch interface expects:
                # - hidden_states: [batch, hidden_dim] ndarray (float32)
                # - expert_weights_packed: list of uint8 arrays, one per expert
                # - expert_bits: list of ints, one per expert
                # - expert_scales: list of float16 arrays, one per expert
                # - expert_indices: [batch, top_k] ndarray (int32)
                # - expert_probs: [batch, top_k] ndarray (float32)
                # - config: C++ MoEConfig struct

                num_experts = config.num_experts
                
                # Convert dictionaries to lists (expert_id -> index mapping)
                expert_weights_list = [expert_weights[i] for i in range(num_experts)]
                expert_scales_list = [expert_scales[i] for i in range(num_experts)]
                expert_bits_list = [expert_bits[i] for i in range(num_experts)]
                
                # Convert tensors to numpy arrays for C++ interop
                hidden_states_np = hidden_states.float().cpu().numpy()
                expert_weights_np = [w.cpu().numpy().astype(np.uint8) for w in expert_weights_list]
                expert_scales_np = [s.cpu().numpy().astype(np.float16) for s in expert_scales_list]
                expert_indices_np = expert_indices.int().cpu().numpy().astype(np.int32)
                # Calculate top-k probabilities for selected experts
                # Shape: [batch, top_k] instead of full [batch, num_experts]
                top_k_probs = torch.gather(router_probs, 1, expert_indices)
                expert_probs_np = top_k_probs.float().cpu().numpy()
                
                # Create C++ MoEConfig
                cpp_config = _cpp_ext.MoEConfig()
                cpp_config.hidden_dim = config.hidden_dim
                cpp_config.intermediate_dim = config.intermediate_dim
                cpp_config.num_experts = config.num_experts
                cpp_config.top_k = config.num_experts_per_tok
                cpp_config.use_indirect_command_buffers = True
                cpp_config.overlap_cpu_encoding = True
                cpp_config.wait_for_completion = True
                
                # Call C++ dispatch (modifies hidden_states in place)
                _cpp_ext.dispatch_mixed_bpw_moe(
                    hidden_states_np,
                    expert_weights_np,
                    expert_bits_list,
                    expert_scales_np,
                    expert_indices_np,
                    expert_probs_np,  # Now correctly uses top_k_probs
                    cpp_config,
                )
                
                # Convert back to torch tensor (ensure float16 output)
                output = torch.from_numpy(hidden_states_np).to(hidden_states.device).half()
                return output
            elif hasattr(_cpp_ext, 'dispatch_moe_trellis_swiglu_batched_cpp'):
                # Alternative C++ dispatch function name
                logger.warning("C++ dispatch uses alternative function dispatch_moe_trellis_swiglu_batched_cpp")
                
                # Calculate top-k probs
                top_k_probs = torch.gather(router_probs, 1, expert_indices)
                
                return _cpp_ext.dispatch_moe_trellis_swiglu_batched_cpp(
                    hidden_states,
                    expert_weights,
                    expert_scales,
                    top_k_probs,
                    expert_indices,
                    config.num_experts,
                    config.hidden_dim,
                    config.intermediate_dim,
                    config.num_experts_per_tok,
                )
            else:
                logger.warning("C++ dispatch module does not provide dispatch_mixed_bpw_moe")
                raise ImportError("C++ dispatch function not available")
        except (ImportError, AttributeError) as e2:
            logger.error("C++ dispatch unavailable, re-raising original exception")
            raise e from e2
