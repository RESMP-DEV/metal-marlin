"""Metal dispatch for Trellis MoE kernels.

Provides Python wrappers for gemm_trellis_moe.metal kernels:
- dispatch_moe_trellis_swiglu: Fused MoE GEMM with SwiGLU activation
- dispatch_moe_trellis_swiglu_grouped_fairway: Grouped fairway dispatch path

CRITICAL: The kernel uses per-token expert routing. Each token uses its OWN
assigned expert from expert_ids[token * top_k + slot], NOT a shared expert
for all tokens in a tile. The grid is 3D: (n_blocks, tokens, slots).

This replaces the slow sequential expert iteration in TrellisMoEMLP with a
single batched kernel that processes all experts in parallel.
"""

from __future__ import annotations

import logging
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from ..metal_dispatch import (
    HAS_METAL,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)
from .kernel_selection import get_kernel_for_batch_size

if "metal_marlin.moe_dispatch" not in sys.modules:
    from metal_marlin import moe_dispatch as _parent_moe_dispatch
else:
    _parent_moe_dispatch = sys.modules["metal_marlin.moe_dispatch"]

gather_for_experts = _parent_moe_dispatch.gather_for_experts
group_tokens_by_expert_full = _parent_moe_dispatch.group_tokens_by_expert_full
group_tokens_by_expert_full_gpu = getattr(
    _parent_moe_dispatch, "group_tokens_by_expert_full_gpu", None
)

if HAS_METAL:
    import Metal

if TYPE_CHECKING:
    from Metal import MTLCommandBuffer
else:
    MTLCommandBuffer = Any


logger = logging.getLogger(__name__)

_MOE_PARAMS_CACHE_MAX_ENTRIES = 512
_moe_params_buffer_cache: dict[
    tuple[int, tuple[int, int, int], tuple[int, int, int, int, int]],
    Any,
] = {}

# Lightweight diagnostics/counters for mixed-BPW token grouping.
_MIXED_BPW_GROUPING_BASE_COUNTERS: dict[str, int] = {
    "grouping_calls_total": 0,
    "grouping_gpu_primary_success_total": 0,
    "grouping_cpu_fallback_total": 0,
}
_mixed_bpw_grouping_counters: dict[str, int] = _MIXED_BPW_GROUPING_BASE_COUNTERS.copy()
_mixed_bpw_last_fallback: dict[str, Any] | None = None


def get_mixed_bpw_grouping_fallback_counters() -> dict[str, int]:
    """Return mixed-BPW token-grouping fallback counters."""
    return _mixed_bpw_grouping_counters.copy()


def get_mixed_bpw_grouping_fallback_diagnostics() -> dict[str, Any]:
    """Return counters plus the last reason-coded CPU fallback event."""
    diagnostics: dict[str, Any] = {"counters": get_mixed_bpw_grouping_fallback_counters()}
    if _mixed_bpw_last_fallback is not None:
        diagnostics["last_fallback"] = _mixed_bpw_last_fallback.copy()
    return diagnostics


def reset_mixed_bpw_grouping_fallback_counters() -> None:
    """Reset mixed-BPW token-grouping fallback counters and last event."""
    global _mixed_bpw_grouping_counters
    global _mixed_bpw_last_fallback
    _mixed_bpw_grouping_counters = _MIXED_BPW_GROUPING_BASE_COUNTERS.copy()
    _mixed_bpw_last_fallback = None


def _bump_mixed_bpw_grouping_counter(key: str, amount: int = 1) -> None:
    _mixed_bpw_grouping_counters[key] = _mixed_bpw_grouping_counters.get(key, 0) + amount


def _record_mixed_bpw_grouping_fallback(
    *,
    reason_code: str,
    detail: str,
    batch_size: int,
    top_k: int,
    num_experts: int,
) -> None:
    """Record reason-coded diagnostics when grouping falls back from GPU to CPU."""
    global _mixed_bpw_last_fallback
    _bump_mixed_bpw_grouping_counter("grouping_cpu_fallback_total")
    _bump_mixed_bpw_grouping_counter(f"grouping_cpu_fallback_reason_{reason_code}")
    _mixed_bpw_last_fallback = {
        "reason_code": reason_code,
        "detail": detail,
        "batch_size": batch_size,
        "top_k": top_k,
        "num_experts": num_experts,
    }
    logger.debug(
        "Mixed-BPW grouping CPU fallback reason=%s detail=%s batch=%d top_k=%d num_experts=%d",
        reason_code,
        detail,
        batch_size,
        top_k,
        num_experts,
    )


def _group_tokens_mixed_bpw_primary_gpu(
    *,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    num_experts: int,
):
    """Use GPU token grouping when available, otherwise CPU fallback with diagnostics."""
    batch_size, top_k = expert_ids.shape
    _bump_mixed_bpw_grouping_counter("grouping_calls_total")

    if group_tokens_by_expert_full_gpu is None:
        _record_mixed_bpw_grouping_fallback(
            reason_code="gpu_grouping_unavailable",
            detail="group_tokens_by_expert_full_gpu missing",
            batch_size=batch_size,
            top_k=top_k,
            num_experts=num_experts,
        )
        return group_tokens_by_expert_full(expert_ids, num_experts)

    if expert_ids.device.type != "mps" or expert_probs.device.type != "mps":
        _record_mixed_bpw_grouping_fallback(
            reason_code="non_mps_inputs",
            detail=f"expert_ids={expert_ids.device.type} expert_probs={expert_probs.device.type}",
            batch_size=batch_size,
            top_k=top_k,
            num_experts=num_experts,
        )
        return group_tokens_by_expert_full(expert_ids, num_experts)

    try:
        dispatch_info = group_tokens_by_expert_full_gpu(
            expert_ids, expert_probs, num_experts
        )

        # Validate shape invariants to guard against malformed kernel output.
        total_assignments = expert_ids.numel()
        if dispatch_info.expert_offsets.numel() != (num_experts + 1):
            raise RuntimeError(
                f"invalid expert_offsets size={dispatch_info.expert_offsets.numel()} "
                f"expected={num_experts + 1}"
            )
        if dispatch_info.sorted_token_indices.numel() != total_assignments:
            raise RuntimeError(
                f"invalid sorted_token_indices size={dispatch_info.sorted_token_indices.numel()} "
                f"expected={total_assignments}"
            )
        if dispatch_info.sorted_expert_indices.numel() != total_assignments:
            raise RuntimeError(
                f"invalid sorted_expert_indices size={dispatch_info.sorted_expert_indices.numel()} "
                f"expected={total_assignments}"
            )

        _bump_mixed_bpw_grouping_counter("grouping_gpu_primary_success_total")
        return dispatch_info
    except Exception as exc:
        _record_mixed_bpw_grouping_fallback(
            reason_code="gpu_grouping_exception",
            detail=f"{type(exc).__name__}: {exc}",
            batch_size=batch_size,
            top_k=top_k,
            num_experts=num_experts,
        )
        return group_tokens_by_expert_full(expert_ids, num_experts)


def prepare_fairway_grouped_inputs(
    *,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare grouped tensors for mixed-bit fairway grouped-kernel dispatch.

    Uses GPU token grouping as the primary path and falls back to CPU grouping
    via the existing mixed-BPW fallback diagnostics/counters when needed.

    Args:
        expert_ids: [batch, top_k] selected expert IDs.
        expert_probs: [batch, top_k] routing probabilities.
        num_experts: Total number of experts.

    Returns:
        Tuple of:
        - sorted_token_ids: [batch * top_k] int32 token indices grouped by expert.
        - expert_offsets: [num_experts + 1] int32 cumulative expert offsets.
        - sorted_probs: [batch * top_k] float16 probabilities aligned to sorted_token_ids.
    """
    if expert_ids.dim() != 2 or expert_probs.dim() != 2:
        raise ValueError(
            "expert_ids and expert_probs must be rank-2 tensors with shape [batch, top_k]"
        )
    if expert_ids.shape != expert_probs.shape:
        raise ValueError(
            f"expert_ids shape {tuple(expert_ids.shape)} must match "
            f"expert_probs shape {tuple(expert_probs.shape)}"
        )

    dispatch_info = _group_tokens_mixed_bpw_primary_gpu(
        expert_ids=expert_ids,
        expert_probs=expert_probs,
        num_experts=num_experts,
    )

    sorted_token_ids = dispatch_info.sorted_token_indices.to(torch.int32).contiguous()
    expert_offsets = dispatch_info.expert_offsets.to(torch.int32).contiguous()
    sorted_probs = expert_probs[
        dispatch_info.sorted_token_indices, dispatch_info.sorted_expert_indices
    ].to(torch.float16).contiguous()

    total_assignments = expert_ids.numel()
    if sorted_token_ids.numel() != total_assignments:
        raise RuntimeError(
            f"invalid sorted_token_ids size={sorted_token_ids.numel()} "
            f"expected={total_assignments}"
        )
    if expert_offsets.numel() != (num_experts + 1):
        raise RuntimeError(
            f"invalid expert_offsets size={expert_offsets.numel()} "
            f"expected={num_experts + 1}"
        )
    if sorted_probs.numel() != total_assignments:
        raise RuntimeError(
            f"invalid sorted_probs size={sorted_probs.numel()} "
            f"expected={total_assignments}"
        )

    return sorted_token_ids, expert_offsets, sorted_probs


def _normalize_bit_group_entry(
    *,
    bit_tuple: tuple[int, int, int],
    entry: tuple[Any, Any],
) -> tuple[list[int], Any]:
    """Normalize both supported tuple layouts to (expert_list, cached_buffers)."""
    first, second = entry
    expert_list: list[int]
    cached_buffers: Any

    if isinstance(first, list):
        expert_list = first
        cached_buffers = second
    elif isinstance(second, list):
        expert_list = second
        cached_buffers = first
    else:
        raise RuntimeError(
            f"bit_group_buffers[{bit_tuple}] must contain an expert list and cached buffers"
        )

    return expert_list, cached_buffers


def _normalize_moe_bits(bits: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Normalize uniform/per-projection bits to (gate_bits, up_bits, down_bits)."""
    if isinstance(bits, tuple):
        return bits
    return bits, bits, bits


def _get_or_create_moe_params_buffer(
    *,
    device: Any,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    bits: int | tuple[int, int, int],
) -> Any:
    """Get/create cached params buffer keyed by device, bits, and dispatch shape."""
    gate_bits, up_bits, down_bits = _normalize_moe_bits(bits)
    shape_sig = (batch_size, hidden_dim, intermediate_dim, num_experts, top_k)
    cache_key = (id(device), (gate_bits, up_bits, down_bits), shape_sig)

    params_buf = _moe_params_buffer_cache.get(cache_key)
    if params_buf is not None:
        return params_buf

    # Metal struct: batch_size, hidden_dim, intermediate_dim, num_experts, top_k,
    #               gate_bits, up_bits, down_bits, tile_size,
    #               gate_n_levels, up_n_levels, down_n_levels
    params_data = np.array(
        [
            batch_size,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            gate_bits,
            up_bits,
            down_bits,
            128,
            1 << gate_bits,
            1 << up_bits,
            1 << down_bits,
        ],
        dtype=np.uint32,
    )
    params_buf = device.newBufferWithBytes_length_options_(
        params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
    )

    if len(_moe_params_buffer_cache) >= _MOE_PARAMS_CACHE_MAX_ENTRIES:
        _moe_params_buffer_cache.clear()
    _moe_params_buffer_cache[cache_key] = params_buf

    return params_buf


class MoEDispatchValidationError(ValueError):
    """Raised when MoE dispatch input validation fails."""

    pass


@dataclass
class QueuedDispatch:
    """A queued kernel dispatch waiting to be executed.

    Stores all information needed to encode a dispatch into a command buffer.
    """

    pipeline: Any  # MTLComputePipelineState
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    buffers: list[Any]  # List of MTLBuffer
    output_tensor: torch.Tensor  # fp32 output tensor for copy-back
    output_buffer: Any  # MTLBuffer for output
    output_fp16: torch.Tensor  # Pre-allocated fp16 buffer for result


class BatchedDispatcher:
    """Batched Metal kernel dispatcher for MoE layers.

    Accumulates kernel dispatches from multiple MoE layers into a single
    command buffer, reducing Metal API overhead from 45 command buffer
    creations + commits to just 1.

    Usage:
        dispatcher = BatchedDispatcher(lib)

        # Queue dispatches from each MoE layer
        for layer in moe_layers:
            dispatcher.queue_moe_dispatch(...)

        # Execute all queued dispatches in one commit
        dispatcher.commit_and_wait()

    Thread safety: NOT thread-safe. Use one dispatcher per inference thread.
    """

    def __init__(self, lib: MetalKernelLibrary):
        """Initialize BatchedDispatcher.

        Args:
            lib: MetalKernelLibrary with compiled MoE kernels.
        """
        self._lib = lib
        self._queue: list[QueuedDispatch] = []
        self._committed = False

    @property
    def pending_count(self) -> int:
        """Number of dispatches queued but not yet committed."""
        return len(self._queue)

    def queue_moe_dispatch(
        self,
        *,
        activations: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        bits: int,
        cached_buffers: CachedWeightBuffers,
        buffer_pool: MoEBufferPool,
        use_fp32_acc: bool = False,
    ) -> torch.Tensor:
        """Queue an MoE dispatch for later execution.

        Returns a tensor that will contain the result after commit_and_wait().
        The returned tensor is INVALID until commit_and_wait() is called.

        Args:
            activations: Input activations [batch, hidden_dim]
            expert_ids: Selected expert IDs [batch, top_k]
            expert_probs: Expert routing weights [batch, top_k]
            hidden_dim: Hidden dimension
            intermediate_dim: Intermediate (FFN) dimension
            num_experts: Total number of experts
            top_k: Number of experts per token
            bits: Quantization bits
            cached_buffers: Pre-cached weight buffers
            buffer_pool: Buffer pool for dynamic allocations
            use_fp32_acc: Use FP32 accumulation

        Returns:
            Output tensor (fp16) - valid only after commit_and_wait()
        """
        device = self._lib.device
        batch_size = activations.shape[0]

        # Get buffers from pool
        activations_buf = buffer_pool.get_activation_buffer(
            batch_size, activations)
        expert_ids_buf = buffer_pool.get_expert_ids_buffer(
            batch_size, top_k, expert_ids)
        expert_probs_buf = buffer_pool.get_expert_probs_buffer(
            batch_size, top_k, expert_probs)

        # Get output buffers
        output_fp32, output_buf = buffer_pool.get_output_buffer(batch_size)
        output_fp16 = buffer_pool.get_output_fp16(batch_size)

        # Get params buffer
        params_buf = buffer_pool.get_params_buffer(
            batch_size, hidden_dim, intermediate_dim, num_experts, top_k, bits
        )

        # Select kernel
        kernel_name, tile_n = select_moe_kernel(batch_size, use_fp32_acc)
        pipeline = self._lib.get_pipeline(kernel_name)

        # Compute grid
        is_decode_kernel = kernel_name == "moe_trellis_swiglu_decode"
        is_prefill4_kernel = "prefill4" in kernel_name

        if is_decode_kernel:
            threads_per_tg = 128  # Matches DECODE_THREADS in gemm_trellis_moe.metal
            grid_x = (hidden_dim + tile_n - 1) // tile_n
            grid_y = top_k
            grid_z = 1
        elif is_prefill4_kernel:
            threads_per_tg = 128
            grid_x = (hidden_dim + tile_n - 1) // tile_n
            grid_y = (batch_size + 3) // 4
            grid_z = top_k
        else:
            threads_per_tg = 128
            grid_x = (hidden_dim + tile_n - 1) // tile_n
            grid_y = batch_size
            grid_z = top_k

        # Build buffer list
        buffer_list = [
            activations_buf,
            cached_buffers.gate_weights,
            cached_buffers.gate_scales,
            cached_buffers.up_weights,
            cached_buffers.up_scales,
            cached_buffers.down_weights,
            cached_buffers.down_scales,
            cached_buffers.gate_su,
            cached_buffers.gate_sv,
            cached_buffers.up_su,
            cached_buffers.up_sv,
            cached_buffers.down_su,
            cached_buffers.down_sv,
            cached_buffers.grid,
            expert_ids_buf,
            expert_probs_buf,
            output_buf,
            params_buf,
        ]

        # Queue the dispatch
        self._queue.append(
            QueuedDispatch(
                pipeline=pipeline,
                grid=(grid_x, grid_y, grid_z),
                threadgroup=(threads_per_tg, 1, 1),
                buffers=buffer_list,
                output_tensor=output_fp32,
                output_buffer=output_buf,
                output_fp16=output_fp16,
            )
        )

        # Return the fp16 output tensor (will be filled after commit)
        return output_fp16

    def commit_and_wait(self) -> None:
        """Execute all queued dispatches with parallel expert GEMM execution.

        Each expert dispatch uses a separate command encoder, allowing the Metal
        scheduler to execute multiple expert GEMMs concurrently when GPU has capacity.

        After this call completes, all output tensors returned by
        queue_moe_dispatch() contain valid results.
        """
        if not self._queue:
            return

        command_buffer = self._lib.command_queue.commandBuffer()

        # Create separate encoders for each dispatch to enable parallel execution
        for dispatch in self._queue:
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(dispatch.pipeline)

            # Bind buffers
            for i, buf in enumerate(dispatch.buffers):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)

            # Dispatch this expert's GEMM
            grid_size = Metal.MTLSizeMake(*dispatch.grid)
            tg_size = Metal.MTLSizeMake(*dispatch.threadgroup)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                grid_size, tg_size)

            encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Copy results to fp16 output tensors
        for dispatch in self._queue:
            dispatch.output_fp16.copy_(dispatch.output_tensor)

        # Clear queue
        self._queue.clear()
        self._committed = True

    def clear(self) -> None:
        """Discard all queued dispatches without executing."""
        self._queue.clear()
        self._committed = False


@dataclass
class CachedWeightBuffers:
    """Pre-allocated Metal buffers for static MoE weights."""

    gate_weights: Any
    gate_scales: Any
    up_weights: Any
    up_scales: Any
    down_weights: Any
    down_scales: Any
    gate_su: Any
    gate_sv: Any
    up_su: Any
    up_sv: Any
    down_su: Any
    down_sv: Any
    grid: Any


def create_cached_weight_buffers(
    device: Any,
    gate_weights: torch.Tensor,
    gate_scales: torch.Tensor,
    up_weights: torch.Tensor,
    up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
    gate_su: torch.Tensor,
    gate_sv: torch.Tensor,
    up_su: torch.Tensor,
    up_sv: torch.Tensor,
    down_su: torch.Tensor,
    down_sv: torch.Tensor,
    grid: torch.Tensor,
) -> CachedWeightBuffers:
    """Create cached Metal buffers for static MoE weights.

    Call this once during model initialization, then pass the returned
    CachedWeightBuffers to dispatch_moe_trellis_swiglu for each forward pass.

    Uses batched buffer creation to reduce Metal API overhead.
    """
    require_mps()
    from ..metal_dispatch import mps_tensors_to_metal_buffers

    def ensure_half(t: torch.Tensor) -> torch.Tensor:
        if t.dtype == torch.float16:
            return t.contiguous()
        return t.half().contiguous()

    # Prepare all tensors
    tensors = [
        gate_weights.contiguous(),
        ensure_half(gate_scales),
        up_weights.contiguous(),
        ensure_half(up_scales),
        down_weights.contiguous(),
        ensure_half(down_scales),
        ensure_half(gate_su),
        ensure_half(gate_sv),
        ensure_half(up_su),
        ensure_half(up_sv),
        ensure_half(down_su),
        ensure_half(down_sv),
        ensure_half(grid),
    ]

    # Batch create all buffers in a single call
    buffers = mps_tensors_to_metal_buffers(tensors, device)

    return CachedWeightBuffers(
        gate_weights=buffers[0],
        gate_scales=buffers[1],
        up_weights=buffers[2],
        up_scales=buffers[3],
        down_weights=buffers[4],
        down_scales=buffers[5],
        gate_su=buffers[6],
        gate_sv=buffers[7],
        up_su=buffers[8],
        up_sv=buffers[9],
        down_su=buffers[10],
        down_sv=buffers[11],
        grid=buffers[12],
    )


def create_cached_weight_buffers_from_cpu(
    device: Any,
    gate_weights: torch.Tensor,
    gate_scales: torch.Tensor,
    up_weights: torch.Tensor,
    up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
    gate_su: torch.Tensor,
    gate_sv: torch.Tensor,
    up_su: torch.Tensor,
    up_sv: torch.Tensor,
    down_su: torch.Tensor,
    down_sv: torch.Tensor,
    grid: torch.Tensor,
) -> CachedWeightBuffers:
    """Create cached Metal buffers from CPU tensors.

    Uses batched buffer creation to reduce Metal API overhead.
    """
    from ..metal_dispatch import cpu_tensors_to_metal_buffers

    def ensure_half_cpu(t: torch.Tensor) -> torch.Tensor:
        if t.is_mps or t.is_cuda:
            raise ValueError(f"Tensor must be on CPU, got device={t.device}")
        if t.dtype == torch.float16:
            return t.contiguous()
        return t.half().contiguous()

    # Prepare all tensors
    tensors = [
        gate_weights.contiguous(),
        ensure_half_cpu(gate_scales),
        up_weights.contiguous(),
        ensure_half_cpu(up_scales),
        down_weights.contiguous(),
        ensure_half_cpu(down_scales),
        ensure_half_cpu(gate_su),
        ensure_half_cpu(gate_sv),
        ensure_half_cpu(up_su),
        ensure_half_cpu(up_sv),
        ensure_half_cpu(down_su),
        ensure_half_cpu(down_sv),
        ensure_half_cpu(grid),
    ]

    # Batch create all buffers in a single call
    buffers = cpu_tensors_to_metal_buffers(tensors, device)

    return CachedWeightBuffers(
        gate_weights=buffers[0],
        gate_scales=buffers[1],
        up_weights=buffers[2],
        up_scales=buffers[3],
        down_weights=buffers[4],
        down_scales=buffers[5],
        gate_su=buffers[6],
        gate_sv=buffers[7],
        up_su=buffers[8],
        up_sv=buffers[9],
        down_su=buffers[10],
        down_sv=buffers[11],
        grid=buffers[12],
    )


class MoEBufferPool:
    """Reusable buffer pool for MoE kernel dispatch with memory optimization.

    Preallocates buffers for common batch sizes (1, 2, 4, 8, 16, 32) and top_k
    values to eliminate allocation during forward pass. Separate pools are
    maintained for:
    - Activations: [batch_size, hidden_dim] fp16 input buffers
    - Expert IDs: [batch_size, top_k] int32 routing buffers
    - Expert probs: [batch_size, top_k] fp16 weight buffers
    - Outputs: [batch_size, hidden_dim] fp32 and fp16 output buffers

    Memory Optimization Features:
    1. Power-of-2 bucket sizes for buffer allocation to reduce fragmentation
    2. Buffer coalescing for small allocations to improve cache locality
    3. Memory pressure callback to release unused buffers
    """

    # Standard batch sizes to preallocate
    STANDARD_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    # Default top_k value (Qwen3-235B uses top_k=8)
    DEFAULT_TOP_K: int = 8

    # Power-of-2 bucket sizes for memory optimization (reduces fragmentation)
    BUCKET_SIZES: list[int] = [64, 128, 256, 512, 1024, 2048, 4096]

    # Threshold for coalescing small allocations (bytes)
    COALESCE_THRESHOLD: int = 256

    def __init__(
        self,
        device: Any,
        hidden_dim: int,
        max_batch: int = 32,
        top_k_values: tuple[int, ...] | None = None,
        enable_metrics: bool = False,
        memory_pressure_threshold: float = 5.0,
        enable_coalescing: bool = True,
    ):
        """Initialize MoEBufferPool with preallocated buffers.

        Args:
            device: Metal device for buffer allocation.
            hidden_dim: Hidden dimension for activation/output tensors.
            max_batch: Maximum batch size to support.
            top_k_values: Tuple of top_k values to preallocate for.
                          Defaults to (8,) which covers Qwen3-235B.
            enable_metrics: Whether to track buffer pool efficiency metrics.
            memory_pressure_threshold: Seconds of inactivity before releasing buffers.
            enable_coalescing: Whether to enable buffer coalescing for small allocations.
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_batch = max_batch
        self._top_k_values = top_k_values if top_k_values is not None else (
            self.DEFAULT_TOP_K,)
        self.enable_metrics = enable_metrics
        self.memory_pressure_threshold = memory_pressure_threshold
        self.enable_coalescing = enable_coalescing

        # Separate pools for each buffer type
        self._activation_buffers: dict[int, tuple[torch.Tensor, Any]] = {}
        self._expert_ids_buffers: dict[tuple[int,
                                             int], tuple[torch.Tensor, Any]] = {}
        self._expert_probs_buffers: dict[tuple[int,
                                               int], tuple[torch.Tensor, Any]] = {}
        self._output_buffers: dict[int, tuple[torch.Tensor, Any]] = {}
        self._output_fp16_buffers: dict[int, torch.Tensor] = {}
        self._params_buffers: dict[tuple[int,
                                         int, int, int, int, int], Any] = {}

        # Coalesced buffer pools for small allocations
        # Maps bucket_size -> (tensor, metal_buffer, used_bytes)
        self._coalesced_buffers: dict[int, tuple[torch.Tensor, Any, int]] = {}
        self._coalesced_allocations: dict[
            int, list[tuple[int, int]]
        ] = {}  # bucket_size -> [(offset, size), ...]

        # Last used timestamps for memory pressure management
        self._last_used: dict[str, float] = {}
        self._access_count: dict[str, int] = {}
        self._total_accesses: int = 0

        # Memory pressure callback
        self._memory_pressure_callback: Callable[[], None] | None = None

        # Metrics tracking (only when enable_metrics=True)
        self._hits = 0
        self._misses = 0
        self._peak_buffers = 0
        self._buffer_lifetimes: list[int] = []
        self._forward_calls = 0
        self._coalesced_hits = 0
        self._released_buffers = 0

        # Preallocate all buffers for common sizes
        self._preallocate_all()

    def _get_bucket_size(self, size: int) -> int:
        """Get the power-of-2 bucket size for a given allocation size.

        Args:
            size: Required buffer size in bytes.

        Returns:
            Power-of-2 bucket size that can accommodate the allocation.
            Returns the exact size if larger than max bucket.
        """
        for bucket in self.BUCKET_SIZES:
            if size <= bucket:
                return bucket
        # For large sizes, round up to next power of 2
        return 1 << (size - 1).bit_length()

    def _should_coalesce(self, size: int) -> bool:
        """Check if a buffer size should use coalesced allocation.

        Args:
            size: Buffer size in bytes.

        Returns:
            True if coalescing should be used for this size.
        """
        return self.enable_coalescing and size <= self.COALESCE_THRESHOLD

    def _allocate_coalesced(
        self, size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, Any, int] | None:
        """Allocate space from a coalesced buffer.

        Args:
            size: Required size in bytes.
            dtype: Data type for the tensor view.

        Returns:
            Tuple of (tensor_view, metal_buffer, offset) or None if allocation fails.
        """
        bucket_size = self._get_bucket_size(size)

        if bucket_size not in self._coalesced_buffers:
            # Create new coalesced buffer
            num_bytes = bucket_size * 16  # 16 allocations per bucket
            tensor = torch.zeros(num_bytes // 2, dtype=torch.float16, device="mps")
            buf = mps_tensor_to_metal_buffer(tensor, self.device)
            self._coalesced_buffers[bucket_size] = (tensor, buf, 0)
            self._coalesced_allocations[bucket_size] = []

        tensor, buf, used_bytes = self._coalesced_buffers[bucket_size]
        allocations = self._coalesced_allocations[bucket_size]

        # Simple first-fit allocation
        offset = 0
        for alloc_offset, alloc_size in sorted(allocations):
            if offset + size <= alloc_offset:
                # Found gap
                break
            offset = alloc_offset + alloc_size

        if offset + size > tensor.numel() * 2:  # 2 bytes per float16
            return None  # No space in coalesced buffer

        allocations.append((offset, size))
        self._coalesced_buffers[bucket_size] = (tensor, buf, used_bytes + size)

        # Create a view into the coalesced buffer
        num_elements = size // 2  # float16 = 2 bytes
        tensor_view = tensor[offset // 2:offset // 2 + num_elements]

        if self.enable_metrics:
            self._coalesced_hits += 1

        return tensor_view, buf, offset

    def _free_coalesced(self, offset: int, size: int) -> None:
        """Free space from a coalesced buffer.

        Args:
            offset: Offset in the coalesced buffer.
            size: Size of the allocation.
        """
        for bucket_size, allocations in self._coalesced_allocations.items():
            for i, (alloc_offset, alloc_size) in enumerate(allocations):
                if alloc_offset == offset and alloc_size == size:
                    allocations.pop(i)
                    _, buf, used_bytes = self._coalesced_buffers[bucket_size]
                    self._coalesced_buffers[bucket_size] = (
                        self._coalesced_buffers[bucket_size][0],
                        buf,
                        used_bytes - size,
                    )
                    return

    def on_memory_pressure(self) -> int:
        """Release buffers not used in the last N seconds.

        This method should be called when system memory pressure is detected
        or periodically to prevent unbounded memory growth.

        Returns:
            Number of buffers released.
        """
        current_time = time.monotonic()
        released = 0

        # Release old activation buffers
        for key in list(self._activation_buffers.keys()):
            key_str = f"act_{key}"
            last_used = self._last_used.get(key_str, current_time)
            if current_time - last_used > self.memory_pressure_threshold:
                del self._activation_buffers[key]
                self._last_used.pop(key_str, None)
                self._access_count.pop(key_str, None)
                released += 1

        # Release old expert_ids buffers
        for key in list(self._expert_ids_buffers.keys()):
            key_str = f"ids_{key}"
            last_used = self._last_used.get(key_str, current_time)
            if current_time - last_used > self.memory_pressure_threshold:
                del self._expert_ids_buffers[key]
                self._last_used.pop(key_str, None)
                self._access_count.pop(key_str, None)
                released += 1

        # Release old expert_probs buffers
        for key in list(self._expert_probs_buffers.keys()):
            key_str = f"probs_{key}"
            last_used = self._last_used.get(key_str, current_time)
            if current_time - last_used > self.memory_pressure_threshold:
                del self._expert_probs_buffers[key]
                self._last_used.pop(key_str, None)
                self._access_count.pop(key_str, None)
                released += 1

        # Release old output buffers
        for key in list(self._output_buffers.keys()):
            key_str = f"out_{key}"
            last_used = self._last_used.get(key_str, current_time)
            if current_time - last_used > self.memory_pressure_threshold:
                del self._output_buffers[key]
                del self._output_fp16_buffers[key]
                self._last_used.pop(key_str, None)
                self._access_count.pop(key_str, None)
                released += 1

        # Release old params buffers
        for key in list(self._params_buffers.keys()):
            key_str = f"params_{key}"
            last_used = self._last_used.get(key_str, current_time)
            if current_time - last_used > self.memory_pressure_threshold:
                del self._params_buffers[key]
                self._last_used.pop(key_str, None)
                self._access_count.pop(key_str, None)
                released += 1

        # Clean up empty coalesced buffers
        for bucket_size in list(self._coalesced_buffers.keys()):
            allocations = self._coalesced_allocations.get(bucket_size, [])
            if not allocations:
                del self._coalesced_buffers[bucket_size]
                self._coalesced_allocations.pop(bucket_size, None)
                released += 1

        if self.enable_metrics:
            self._released_buffers += released

        return released

    def set_memory_pressure_callback(self, callback: Callable[[], None]) -> None:
        """Set a callback to be invoked when memory pressure is detected.

        Args:
            callback: Function to call when memory pressure is detected.
        """
        self._memory_pressure_callback = callback

    def _update_access_time(self, key: str) -> None:
        """Update the last access time and count for a buffer.

        Args:
            key: Buffer identifier string.
        """
        current_time = time.monotonic()
        self._last_used[key] = current_time
        self._access_count[key] = self._access_count.get(key, 0) + 1
        self._total_accesses += 1

    def _preallocate_all(self) -> None:
        """Preallocate buffers for all standard batch sizes and top_k values."""
        batch_sizes = [
            b for b in self.STANDARD_BATCH_SIZES if b <= self.max_batch]

        for batch_size in batch_sizes:
            # Preallocate activation buffers
            self._preallocate_activation(batch_size)

            # Preallocate output buffers (fp32 and fp16)
            self._preallocate_output(batch_size)

            # Preallocate expert_ids and expert_probs for each top_k
            for top_k in self._top_k_values:
                self._preallocate_expert_ids(batch_size, top_k)
                self._preallocate_expert_probs(batch_size, top_k)

    def _preallocate_activation(self, batch_size: int) -> None:
        """Preallocate activation buffer for given batch size."""
        if batch_size in self._activation_buffers:
            return
        act_tensor = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.float16, device="mps")
        act_buf = mps_tensor_to_metal_buffer(act_tensor, self.device)
        self._activation_buffers[batch_size] = (act_tensor, act_buf)

    def _preallocate_output(self, batch_size: int) -> None:
        """Preallocate fp32 and fp16 output buffers for given batch size."""
        if batch_size in self._output_buffers:
            return
        out_tensor = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.float32, device="mps")
        out_buf = mps_tensor_to_metal_buffer(
            out_tensor, self.device, copy_back=True)
        self._output_buffers[batch_size] = (out_tensor, out_buf)

        # Pre-allocate fp16 output buffer for fast conversion
        self._output_fp16_buffers[batch_size] = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.float16, device="mps"
        )

    def _preallocate_expert_ids(self, batch_size: int, top_k: int) -> None:
        """Preallocate expert_ids buffer for given batch size and top_k."""
        key = (batch_size, top_k)
        if key in self._expert_ids_buffers:
            return
        tensor = torch.zeros(
            batch_size, top_k, dtype=torch.int32, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device)
        self._expert_ids_buffers[key] = (tensor, buf)

    def _preallocate_expert_probs(self, batch_size: int, top_k: int) -> None:
        """Preallocate expert_probs buffer for given batch size and top_k."""
        key = (batch_size, top_k)
        if key in self._expert_probs_buffers:
            return
        tensor = torch.zeros(
            batch_size, top_k, dtype=torch.float16, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device)
        self._expert_probs_buffers[key] = (tensor, buf)

    def get_activation_buffer(self, batch_size: int, activations: torch.Tensor) -> Any:
        key_str = f"act_{batch_size}"
        self._update_access_time(key_str)

        if batch_size in self._activation_buffers:
            if self.enable_metrics:
                self._hits += 1
            tensor, buf = self._activation_buffers[batch_size]
            tensor.copy_(activations)
            return buf
        if self.enable_metrics:
            self._misses += 1
        return mps_tensor_to_metal_buffer(activations.contiguous(), self.device)

    def get_expert_ids_buffer(self, batch_size: int, top_k: int, expert_ids: torch.Tensor) -> Any:
        key = (batch_size, top_k)
        key_str = f"ids_{key}"
        self._update_access_time(key_str)

        if key not in self._expert_ids_buffers:
            tensor = torch.zeros(
                batch_size, top_k, dtype=torch.int32, device="mps")
            buf = mps_tensor_to_metal_buffer(tensor, self.device)
            self._expert_ids_buffers[key] = (tensor, buf)
        tensor, buf = self._expert_ids_buffers[key]
        tensor.copy_(expert_ids.int())
        return buf

    def get_expert_probs_buffer(
        self, batch_size: int, top_k: int, expert_probs: torch.Tensor
    ) -> Any:
        key = (batch_size, top_k)
        key_str = f"probs_{key}"
        self._update_access_time(key_str)

        if key not in self._expert_probs_buffers:
            tensor = torch.zeros(
                batch_size, top_k, dtype=torch.float16, device="mps")
            buf = mps_tensor_to_metal_buffer(tensor, self.device)
            self._expert_probs_buffers[key] = (tensor, buf)
        tensor, buf = self._expert_probs_buffers[key]
        # Avoid dtype conversion if already fp16 (common case after softmax optimization)
        if expert_probs.dtype == torch.float16:
            tensor.copy_(expert_probs)
        else:
            tensor.copy_(expert_probs.half())
        return buf

    def get_output_buffer(self, batch_size: int) -> tuple[torch.Tensor, Any]:
        key_str = f"out_{batch_size}"
        self._update_access_time(key_str)

        if batch_size in self._output_buffers:
            tensor, buf = self._output_buffers[batch_size]
            tensor.zero_()
            return tensor, buf
        tensor = torch.zeros(batch_size, self.hidden_dim,
                             dtype=torch.float32, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device, copy_back=True)
        return tensor, buf

    def get_output_fp16(self, batch_size: int) -> torch.Tensor:
        """Get pre-allocated fp16 output buffer for fast dtype conversion.

        For batch=1 decode, this avoids allocating a new tensor on every call.
        The caller should copy fp32 output into this buffer via .copy_().
        """
        if batch_size in self._output_fp16_buffers:
            return self._output_fp16_buffers[batch_size]
        # Fallback: allocate (should not happen for common batch sizes)
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float16, device="mps")

    def preallocate_top_k(self, top_k: int, batch_sizes: list[int] | None = None) -> None:
        """Preallocate expert_ids and expert_probs buffers for additional top_k value.

        Use this to add support for top_k values not included during __init__.

        Args:
            top_k: The top_k value to preallocate buffers for.
            batch_sizes: Optional list of batch sizes. Defaults to standard sizes.
        """
        if batch_sizes is None:
            batch_sizes = [
                b for b in self.STANDARD_BATCH_SIZES if b <= self.max_batch]
        for bs in batch_sizes:
            self._preallocate_expert_ids(bs, top_k)
            self._preallocate_expert_probs(bs, top_k)

    def get_params_buffer(
        self,
        batch_size: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        bits: int | tuple[int, int, int],
    ) -> Any:
        """Get or create a cached params buffer for the given parameters.

        Uses power-of-2 bucket sizing to reduce memory fragmentation.

        Args:
            bits: Either uniform bits (int) or per-projection (gate, up, down) tuple.
        """
        # Handle per-projection bits
        if isinstance(bits, tuple):
            gate_bits, up_bits, down_bits = bits
        else:
            gate_bits = up_bits = down_bits = bits

        # Use power-of-2 bucketing for params to reduce fragmentation
        # Bucket the dimensions to reduce unique buffer sizes
        bucketed_batch = self._get_bucket_size(batch_size)
        bucketed_hidden = self._get_bucket_size(hidden_dim)
        bucketed_intermediate = self._get_bucket_size(intermediate_dim)

        key = (bucketed_batch, bucketed_hidden, bucketed_intermediate,
               num_experts, top_k, gate_bits, up_bits, down_bits)
        key_str = f"params_{key}"
        self._update_access_time(key_str)

        if key not in self._params_buffers:
            # Metal struct: batch_size, hidden_dim, intermediate_dim, num_experts, top_k,
            #               gate_bits, up_bits, down_bits, tile_size,
            #               gate_n_levels, up_n_levels, down_n_levels
            params_data = np.array(
                [batch_size, hidden_dim, intermediate_dim, num_experts, top_k,
                 gate_bits, up_bits, down_bits, 128,
                 1 << gate_bits, 1 << up_bits, 1 << down_bits],
                dtype=np.uint32,
            )
            self._params_buffers[key] = self.device.newBufferWithBytes_length_options_(
                params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
            )
        return self._params_buffers[key]

    def clear(self) -> None:
        """Clear all buffers and reset the pool state."""
        self._activation_buffers.clear()
        self._expert_ids_buffers.clear()
        self._expert_probs_buffers.clear()
        self._output_buffers.clear()
        self._output_fp16_buffers.clear()
        self._params_buffers.clear()
        self._coalesced_buffers.clear()
        self._coalesced_allocations.clear()
        self._last_used.clear()
        self._access_count.clear()
        self._total_accesses = 0


def select_moe_kernel(
    batch_size: int,
    use_fp32_acc: bool,
    gate_bits: int | None = None,
    up_bits: int | None = None,
    down_bits: int | None = None,
) -> tuple[str, int]:
    """Select optimal MoE kernel and tile size for given batch size.

    Args:
        batch_size: Number of tokens in the batch
        use_fp32_acc: Whether to use FP32 accumulation
        gate_bits: Bit width for gate projection weights (optional)
        up_bits: Bit width for up projection weights (optional)
        down_bits: Bit width for down projection weights (optional)

    Returns:
        Tuple of (kernel_name, tile_n)
    """
    return get_kernel_for_batch_size(
        batch_size=batch_size,
        use_fp32_acc=use_fp32_acc,
        gate_bits=gate_bits,
        up_bits=up_bits,
        down_bits=down_bits,
    )


def _get_moe_tile_n_for_kernel(kernel_name: str) -> int:
    """Infer tile_n from kernel name for benchmark-only kernel overrides."""
    if "large_batch" in kernel_name:
        return 128
    return 64


def get_moe_kernel(
    batch_size: int,
    use_fp32_acc: bool = False,
    gate_bits: int | None = None,
    up_bits: int | None = None,
    down_bits: int | None = None,
) -> tuple[str, int]:
    """Get MoE kernel name and tile size for given parameters.

    This is a convenience wrapper around select_moe_kernel that provides
    a clear API for kernel dispatch selection.

    Args:
        batch_size: Number of tokens in the batch
        use_fp32_acc: Whether to use FP32 accumulation
        gate_bits: Bit width for gate projection weights (optional)
        up_bits: Bit width for up projection weights (optional)
        down_bits: Bit width for down projection weights (optional)

    Returns:
        Tuple of (kernel_name, tile_n)

    Specialized kernels:
        - moe_trellis_swiglu_decode_6_2_3: gate=6-bit, up=2-bit, down=3-bit (dominant)
        - moe_trellis_swiglu_decode_6_3_4: gate=6-bit, up=3-bit, down=4-bit
        - moe_trellis_swiglu_decode_6_2_4: gate=6-bit, up=2-bit, down=4-bit

    Specialization benefits:
        - Compile-time known dequant parameters (bit shifts/masks)
        - Better instruction scheduling
        - Reduced register pressure
    """
    return select_moe_kernel(batch_size, use_fp32_acc, gate_bits, up_bits, down_bits)


def dispatch_moe_trellis_swiglu_batched(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    gate_weights: torch.Tensor | None,
    gate_scales: torch.Tensor | None,
    up_weights: torch.Tensor | None,
    up_scales: torch.Tensor | None,
    down_weights: torch.Tensor | None,
    down_scales: torch.Tensor | None,
    gate_su: torch.Tensor | None,
    gate_sv: torch.Tensor | None,
    up_su: torch.Tensor | None,
    up_sv: torch.Tensor | None,
    down_su: torch.Tensor | None,
    down_sv: torch.Tensor | None,
    grid: torch.Tensor | None,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    bits: int,
    *,
    cached_buffers: CachedWeightBuffers | None = None,
    buffer_pool: MoEBufferPool | None = None,
    use_fp32_acc: bool = False,
) -> torch.Tensor:
    """Fused MoE GEMM with expert batching for improved efficiency.

    Groups tokens by selected expert, processes each expert with batched tokens,
    and scatters results back to original positions.

    This provides significant bandwidth reduction by loading each expert's weights
    once instead of per-token, especially beneficial for batch > 1.

    Args:
        Same as dispatch_moe_trellis_swiglu

    Returns:
        Output tensor [batch, hidden_dim] fp16
    """
    device = lib.device
    batch_size = activations.shape[0]

    # Group tokens by expert
    dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)

    # Gather activations in expert-sorted order
    activations_sorted = gather_for_experts(activations, dispatch_info)

    # Prepare cached buffers
    if cached_buffers is None:
        gate_weights = gate_weights.contiguous()
        gate_scales = gate_scales.contiguous()
        up_weights = up_weights.contiguous()
        up_scales = up_scales.contiguous()
        down_weights = down_weights.contiguous()
        down_scales = down_scales.contiguous()
        gate_su = gate_su.contiguous()
        gate_sv = gate_sv.contiguous()
        up_su = up_su.contiguous()
        up_sv = up_sv.contiguous()
        down_su = down_su.contiguous()
        down_sv = down_sv.contiguous()
        grid = grid.contiguous()

        cached_temp = create_cached_weight_buffers(
            device,
            gate_weights,
            gate_scales,
            up_weights,
            up_scales,
            down_weights,
            down_scales,
            gate_su,
            gate_sv,
            up_su,
            up_sv,
            down_su,
            down_sv,
            grid,
        )
    else:
        cached_temp = cached_buffers

    # Output accumulator for scattering
    output_accum = torch.zeros(
        batch_size, hidden_dim, dtype=torch.float16, device="mps")

    # OPTIMIZATION: Use single encoder for all expert dispatches
    # Previous: 1 encoder per expert = num_experts encoders (30+ for large models)
    # Current: 1 shared encoder for all experts = 1 encoder total
    # Result: <10 dispatches per layer (meets target: <10 dispatches)
    command_buffer = lib.command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    scatter_tasks = []
    keep_alive = []

    # Process each expert with its batched tokens
    for expert_id in range(num_experts):
        start_idx = int(dispatch_info.expert_offsets[expert_id])
        end_idx = int(dispatch_info.expert_offsets[expert_id + 1])

        if end_idx == start_idx:
            continue

        expert_batch_size = end_idx - start_idx

        # Get activations for this expert
        expert_acts = activations_sorted[start_idx:end_idx]

        # Get Metal buffer for expert activations
        expert_acts_buf = mps_tensor_to_metal_buffer(expert_acts, device)

        # Create expert-specific output buffer
        expert_output_fp32 = torch.zeros(
            expert_batch_size, hidden_dim, dtype=torch.float32, device="mps"
        )
        expert_output_buf = mps_tensor_to_metal_buffer(
            expert_output_fp32, device, copy_back=True)

        # Create temp tensors for unused inputs (ids/probs are not used in kernel for batched mode)
        # We set top_k=1 for the kernel dispatch, so we need (N, 1) tensors
        temp_ids = torch.full(
            (expert_batch_size, 1), expert_id, dtype=torch.int32, device="mps"
        )
        temp_probs = torch.full(
            (expert_batch_size, 1), 1.0, dtype=torch.float16, device="mps"
        )

        # Create params buffer for this expert batch
        # NOTE: top_k=1 because we process one expert at a time
        params_data = np.array(
            [
                expert_batch_size,
                hidden_dim,
                intermediate_dim,
                num_experts,
                1,  # top_k
                bits,
                128,
                1 << bits,
            ],
            dtype=np.uint32,
        )
        params_buf = device.newBufferWithBytes_length_options_(
            params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
        )

        # Select kernel for this batch size
        kernel_name, tile_n = select_moe_kernel(
            expert_batch_size, use_fp32_acc)
        pipeline = lib.get_pipeline(kernel_name)

        # Compute grid
        is_decode_kernel = kernel_name == "moe_trellis_swiglu_decode"
        is_prefill4_kernel = "prefill4" in kernel_name

        if is_decode_kernel:
            threads_per_tg = 128
            grid_x = (hidden_dim + tile_n - 1) // tile_n
            grid_y = 1
            grid_z = 1
        elif is_prefill4_kernel:
            threads_per_tg = 128
            grid_x = (hidden_dim + tile_n - 1) // tile_n
            grid_y = (expert_batch_size + 3) // 4
            grid_z = 1
        else:
            threads_per_tg = 128
            grid_x = (hidden_dim + tile_n - 1) // tile_n
            grid_y = expert_batch_size
            grid_z = 1

        # Build buffer list for this expert
        raw_buffer_list = [
            expert_acts_buf,
            cached_temp.gate_weights,
            cached_temp.gate_scales,
            cached_temp.up_weights,
            cached_temp.up_scales,
            cached_temp.down_weights,
            cached_temp.down_scales,
            cached_temp.gate_su,
            cached_temp.gate_sv,
            cached_temp.up_su,
            cached_temp.up_sv,
            cached_temp.down_su,
            cached_temp.down_sv,
            cached_temp.grid,
            temp_ids,  # Pass tensor, convert later
            temp_probs,  # Pass tensor, convert later
            expert_output_buf,
            params_buf,
        ]

        # Process buffers: convert tensors and unwrap CopyBackBuffers
        buffer_list = []
        for b in raw_buffer_list:
            if isinstance(b, torch.Tensor):
                b = mps_tensor_to_metal_buffer(b, device)

            # Unwrap _CopyBackBuffer if needed
            if hasattr(b, "buffer") and hasattr(b, "tensor"):
                b = b.buffer

            buffer_list.append(b)

        # Keep everything alive
        keep_alive.append((expert_acts, expert_output_fp32,
                          temp_ids, temp_probs, buffer_list, pipeline))

        # Reuse single encoder - just update pipeline state and buffers for this expert
        encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(buffer_list):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        grid_size = Metal.MTLSizeMake(grid_x, grid_y, grid_z)
        tg_size = Metal.MTLSizeMake(threads_per_tg, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

        # Store task info for post-dispatch scattering
        scatter_tasks.append((start_idx, end_idx, expert_output_fp32))

    # End encoding once after all expert dispatches
    encoder.endEncoding()

    # Commit and wait for all experts to complete
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    # Scatter results to accumulator
    for start_idx, end_idx, expert_output_fp32 in scatter_tasks:
        # Get token indices for this expert's assignments
        expert_token_indices = dispatch_info.sorted_token_indices[start_idx:end_idx]
        expert_slot_indices = dispatch_info.sorted_expert_indices[start_idx:end_idx]

        # Weight outputs by expert_probs
        probs_for_expert = expert_probs[expert_token_indices,
                                        expert_slot_indices]
        expert_output_fp16 = expert_output_fp32.half()
        weighted_outputs = expert_output_fp16 * probs_for_expert.unsqueeze(1)

        # Scatter to accumulator
        output_accum.index_add_(
            0, expert_token_indices.long(), weighted_outputs)

    return output_accum


def dispatch_moe_trellis_swiglu(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    gate_weights: torch.Tensor | None,
    gate_scales: torch.Tensor | None,
    up_weights: torch.Tensor | None,
    up_scales: torch.Tensor | None,
    down_weights: torch.Tensor | None,
    down_scales: torch.Tensor | None,
    gate_su: torch.Tensor | None,
    gate_sv: torch.Tensor | None,
    up_su: torch.Tensor | None,
    up_sv: torch.Tensor | None,
    down_su: torch.Tensor | None,
    down_sv: torch.Tensor | None,
    grid: torch.Tensor | None,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    bits: int | tuple[int, int, int],
    *,
    cached_buffers: CachedWeightBuffers | None = None,
    buffer_pool: MoEBufferPool | None = None,
    use_fp32_acc: bool = False,
    kernel_name_override: str | None = None,
    cmd_manager: Any | None = None,
) -> Optional[MTLCommandBuffer]:
    """Fused MoE GEMM with Trellis quantization and SwiGLU activation.

    HOT PATH: This function is called for every forward pass. Keep it minimal:
    1. Get cached buffers (dict lookup)
    2. Create activation buffer (minimal)
    3. Dispatch kernel
    4. Return result

    Args:
        bits: Either uniform bit width (int) or per-projection (gate, up, down) tuple.
        kernel_name_override: Optional benchmark-only kernel override. When set,
            dispatch uses this kernel directly instead of select_moe_kernel(...).
        cmd_manager: Optional AsyncCommandBufferManager for batched dispatch.

    Note: MPS availability is checked at module import time (HAS_MPS) and by
    callers (x.is_mps check in forward()). No per-call validation needed here.

    Synchronization: This function does not wait for completion. Callers must
    call waitUntilCompleted() at a layer boundary after batching dispatches.
    """
    # Validation (required for correctness tests)
    if activations.dim() != 2:
        raise MoEDispatchValidationError(
            f"activations must be rank-2 [batch, hidden_dim], got shape={tuple(activations.shape)}"
        )
    if activations.shape[1] != hidden_dim:
        raise MoEDispatchValidationError(
            f"activations hidden dim mismatch: expected {hidden_dim}, got {activations.shape[1]}"
        )
    if activations.dtype != torch.float16:
        raise MoEDispatchValidationError(
            f"activations must be float16 for optimized MoE kernels, got {activations.dtype}"
        )
    if expert_ids.shape != expert_probs.shape:
        raise MoEDispatchValidationError(
            f"expert_ids shape {tuple(expert_ids.shape)} must match expert_probs shape {tuple(expert_probs.shape)}"
        )

    device = lib.device
    batch_size = activations.shape[0]

    # Sort tokens by expert for coalesced memory access
    # Flatten expert_ids from [batch, top_k] to [batch * top_k] and sort by expert_id
    expert_ids_flat = expert_ids.view(-1)
    sort_order = torch.argsort(expert_ids_flat, stable=True)

    # Reorder tensors by expert grouping
    activations_sorted = activations[sort_order // top_k]
    expert_ids_sorted = expert_ids_flat[sort_order]
    expert_probs_sorted = expert_probs.view(-1)[sort_order]

    # Expanded batch size for flattened expert dispatch
    n = batch_size * top_k

    # Get buffers (fast path with pool avoids contiguous/dtype copies)
    if buffer_pool is not None:
        # Buffer pool handles the copy internally, avoiding intermediate allocations
        # NOTE: Use n (expanded batch) for activation buffer size
        activations_buf = buffer_pool.get_activation_buffer(
            n, activations_sorted)

        # Get expert buffers from pool
        # Reshape to (batch_size, top_k) to match pool's preallocated tensor shape
        expert_ids_buf = buffer_pool.get_expert_ids_buffer(
            batch_size, top_k, expert_ids_sorted.view(batch_size, top_k)
        )
        expert_probs_buf = buffer_pool.get_expert_probs_buffer(
            batch_size, top_k, expert_probs_sorted.view(batch_size, top_k)
        )
    else:
        # Slow path - need contiguous tensors for buffer creation
        activations_sorted = activations_sorted.contiguous()
        expert_ids_sorted = expert_ids_sorted.int().contiguous()
        # Avoid dtype conversion if already fp16 (common case)
        if expert_probs_sorted.dtype != torch.float16:
            expert_probs_sorted = expert_probs_sorted.half()
        expert_probs_sorted = expert_probs_sorted.contiguous()
        activations_buf = mps_tensor_to_metal_buffer(
            activations_sorted, device)
        expert_ids_buf = mps_tensor_to_metal_buffer(expert_ids_sorted, device)
        expert_probs_buf = mps_tensor_to_metal_buffer(
            expert_probs_sorted, device)

    # Get cached weight buffers or create new ones
    if cached_buffers is not None:
        gate_weights_buf = cached_buffers.gate_weights
        gate_scales_buf = cached_buffers.gate_scales
        up_weights_buf = cached_buffers.up_weights
        up_scales_buf = cached_buffers.up_scales
        down_weights_buf = cached_buffers.down_weights
        down_scales_buf = cached_buffers.down_scales
        gate_su_buf = cached_buffers.gate_su
        gate_sv_buf = cached_buffers.gate_sv
        up_su_buf = cached_buffers.up_su
        up_sv_buf = cached_buffers.up_sv
        down_su_buf = cached_buffers.down_su
        down_sv_buf = cached_buffers.down_sv
        grid_buf = cached_buffers.grid
    else:
        # Slow path - should rarely happen in production
        gate_weights = gate_weights.contiguous()  # type: ignore[union-attr]
        gate_scales = gate_scales.contiguous()  # type: ignore[union-attr]
        up_weights = up_weights.contiguous()  # type: ignore[union-attr]
        up_scales = up_scales.contiguous()  # type: ignore[union-attr]
        down_weights = down_weights.contiguous()  # type: ignore[union-attr]
        down_scales = down_scales.contiguous()  # type: ignore[union-attr]
        gate_su = gate_su.contiguous()  # type: ignore[union-attr]
        gate_sv = gate_sv.contiguous()  # type: ignore[union-attr]
        up_su = up_su.contiguous()  # type: ignore[union-attr]
        up_sv = up_sv.contiguous()  # type: ignore[union-attr]
        down_su = down_su.contiguous()  # type: ignore[union-attr]
        down_sv = down_sv.contiguous()  # type: ignore[union-attr]
        grid = grid.contiguous()  # type: ignore[union-attr]

        gate_weights_buf = mps_tensor_to_metal_buffer(gate_weights, device)
        gate_scales_buf = mps_tensor_to_metal_buffer(gate_scales, device)
        up_weights_buf = mps_tensor_to_metal_buffer(up_weights, device)
        up_scales_buf = mps_tensor_to_metal_buffer(up_scales, device)
        down_weights_buf = mps_tensor_to_metal_buffer(down_weights, device)
        down_scales_buf = mps_tensor_to_metal_buffer(down_scales, device)
        gate_su_buf = mps_tensor_to_metal_buffer(gate_su, device)
        gate_sv_buf = mps_tensor_to_metal_buffer(gate_sv, device)
        up_su_buf = mps_tensor_to_metal_buffer(up_su, device)
        up_sv_buf = mps_tensor_to_metal_buffer(up_sv, device)
        down_su_buf = mps_tensor_to_metal_buffer(down_su, device)
        down_sv_buf = mps_tensor_to_metal_buffer(down_sv, device)
        grid_buf = mps_tensor_to_metal_buffer(grid, device)

    # Allocate output buffer
    dispatch_top_k = 1  # flattened expert dispatch
    if buffer_pool is not None:
        output_tensor, output_buf = buffer_pool.get_output_buffer(n)
        output_fp16 = buffer_pool.get_output_fp16(n)
    else:
        output_tensor = torch.zeros(n, hidden_dim, dtype=torch.float32, device="mps")
        output_buf = mps_tensor_to_metal_buffer(
            output_tensor,
            device,
            copy_back=True,
        )
        output_fp16 = torch.zeros(n, hidden_dim, dtype=torch.float16, device="mps")

    # Mixed-BPW and no-pool paths use dedicated cache keyed by device + bits + shape.
    if isinstance(bits, tuple) or buffer_pool is None:
        params_buf = _get_or_create_moe_params_buffer(
            device=device,
            batch_size=n,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=dispatch_top_k,
            bits=bits,
        )
    else:
        params_buf = buffer_pool.get_params_buffer(
            n,
            hidden_dim,
            intermediate_dim,
            num_experts,
            dispatch_top_k,
            bits,
        )

    # Select kernel and compute grid
    # NOTE: Use n (expanded batch) for kernel selection
    if kernel_name_override is None:
        gate_bits, up_bits, down_bits = (
            bits if isinstance(bits, tuple) else (None, None, None)
        )
        kernel_name, tile_n = select_moe_kernel(
            n, use_fp32_acc, gate_bits=gate_bits, up_bits=up_bits, down_bits=down_bits
        )
    else:
        # Handle short names from benchmark scripts
        if kernel_name_override == "decode":
            kernel_name = "moe_trellis_swiglu_decode"
        elif kernel_name_override == "prefill4":
            kernel_name = "moe_trellis_swiglu_prefill4" + ("_fp32acc" if use_fp32_acc else "")
        elif kernel_name_override == "base":
            kernel_name = "moe_trellis_swiglu" + ("_fp32acc" if use_fp32_acc else "")
        elif kernel_name_override == "large_batch":
            kernel_name = "moe_trellis_swiglu_large_batch"
        else:
            kernel_name = kernel_name_override
        tile_n = _get_moe_tile_n_for_kernel(kernel_name)

    is_decode_kernel = kernel_name.startswith("moe_trellis_swiglu_decode")
    is_prefill4_kernel = "prefill4" in kernel_name

    if is_decode_kernel:
        threads_per_tg = 128  # Matches DECODE_THREADS in gemm_trellis_moe.metal
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = 1  # top_k is 1
        grid_z = 1
    elif is_prefill4_kernel:
        threads_per_tg = 128
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = (n + 3) // 4  # Use n
        grid_z = 1  # top_k is 1
    else:
        threads_per_tg = 128
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = n  # Use n
        grid_z = 1  # top_k is 1

    # Dispatch kernel
    buffer_list = [
        activations_buf,
        gate_weights_buf,
        gate_scales_buf,
        up_weights_buf,
        up_scales_buf,
        down_weights_buf,
        down_scales_buf,
        gate_su_buf,
        gate_sv_buf,
        up_su_buf,
        up_sv_buf,
        down_su_buf,
        down_sv_buf,
        grid_buf,
        expert_ids_buf,
        expert_probs_buf,
        output_buf,
        params_buf,
    ]

    if cmd_manager is not None:
        cmd_manager.dispatch_kernel(
            function_name=kernel_name,
            grid=(grid_x, grid_y, grid_z),
            threadgroup=(threads_per_tg, 1, 1),
            buffers=buffer_list,
        )
        # Schedule copy from fp32 output to fp16
        output_fp16.copy_(output_tensor)
        return output_fp16

    cmd_buf = dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, grid_z),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=buffer_list,
        wait=False,  # Don't block - allow batching
    )
    
    # Schedule copy from fp32 output to fp16
    output_fp16.copy_(output_tensor)
    return output_fp16


# ---------------------------------------------------------------------------
# Fused Router Dispatch: matmul + softmax + top-k in single kernel
# ---------------------------------------------------------------------------


@dataclass
class CachedRouterBuffers:
    """Pre-allocated Metal buffers for router weights."""

    router_weights: Any  # [hidden_dim, num_experts] half, column-major


class RouterBufferPool:
    """Reusable buffer pool for fused router kernel dispatch.

    Preallocates buffers for common batch sizes to eliminate allocation
    during forward pass.
    """

    STANDARD_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    def __init__(
        self,
        device: Any,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 8,
        max_batch: int = 32,
    ):
        """Initialize RouterBufferPool.

        Args:
            device: Metal device for buffer allocation.
            hidden_dim: Hidden dimension (router input).
            num_experts: Number of experts (router output).
            top_k: Number of experts selected per token.
            max_batch: Maximum batch size to support.
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_batch = max_batch

        # Buffer pools
        self._hidden_buffers: dict[int, tuple[torch.Tensor, Any]] = {}
        self._expert_ids_buffers: dict[int, tuple[torch.Tensor, Any]] = {}
        self._expert_probs_buffers: dict[int, tuple[torch.Tensor, Any]] = {}
        self._params_buffers: dict[tuple[int, int, int, int], Any] = {}

        self._preallocate_all()

    def _preallocate_all(self) -> None:
        batch_sizes = [
            b for b in self.STANDARD_BATCH_SIZES if b <= self.max_batch]
        for batch_size in batch_sizes:
            self._preallocate_hidden(batch_size)
            self._preallocate_expert_ids(batch_size)
            self._preallocate_expert_probs(batch_size)

    def _preallocate_hidden(self, batch_size: int) -> None:
        if batch_size in self._hidden_buffers:
            return
        tensor = torch.zeros(batch_size, self.hidden_dim,
                             dtype=torch.float16, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device)
        self._hidden_buffers[batch_size] = (tensor, buf)

    def _preallocate_expert_ids(self, batch_size: int) -> None:
        if batch_size in self._expert_ids_buffers:
            return
        tensor = torch.zeros(batch_size, self.top_k,
                             dtype=torch.uint32, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device, copy_back=True)
        self._expert_ids_buffers[batch_size] = (tensor, buf)

    def _preallocate_expert_probs(self, batch_size: int) -> None:
        if batch_size in self._expert_probs_buffers:
            return
        tensor = torch.zeros(batch_size, self.top_k,
                             dtype=torch.float16, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device, copy_back=True)
        self._expert_probs_buffers[batch_size] = (tensor, buf)

    def get_hidden_buffer(self, batch_size: int, hidden: torch.Tensor) -> Any:
        """Get or create hidden state buffer, copying data in."""
        if batch_size in self._hidden_buffers:
            tensor, buf = self._hidden_buffers[batch_size]
            tensor.copy_(hidden)
            return buf
        return mps_tensor_to_metal_buffer(hidden.contiguous(), self.device)

    def get_expert_ids_output(self, batch_size: int) -> tuple[torch.Tensor, Any]:
        """Get output buffer for expert IDs."""
        if batch_size in self._expert_ids_buffers:
            tensor, buf = self._expert_ids_buffers[batch_size]
            return tensor, buf
        tensor = torch.zeros(batch_size, self.top_k,
                             dtype=torch.uint32, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device, copy_back=True)
        return tensor, buf

    def get_expert_probs_output(self, batch_size: int) -> tuple[torch.Tensor, Any]:
        """Get output buffer for expert probabilities."""
        if batch_size in self._expert_probs_buffers:
            tensor, buf = self._expert_probs_buffers[batch_size]
            return tensor, buf
        tensor = torch.zeros(batch_size, self.top_k,
                             dtype=torch.float16, device="mps")
        buf = mps_tensor_to_metal_buffer(tensor, self.device, copy_back=True)
        return tensor, buf

    def get_params_buffer(
        self, batch_size: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> Any:
        """Get or create params buffer."""
        key = (batch_size, hidden_dim, num_experts, top_k)
        if key not in self._params_buffers:
            params_data = np.array(
                [batch_size, hidden_dim, num_experts, top_k], dtype=np.uint32)
            self._params_buffers[key] = self.device.newBufferWithBytes_length_options_(
                params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
            )
        return self._params_buffers[key]

    def clear(self) -> None:
        self._hidden_buffers.clear()
        self._expert_ids_buffers.clear()
        self._expert_probs_buffers.clear()
        self._params_buffers.clear()


def create_cached_router_buffers(device: Any, router_weights: torch.Tensor) -> CachedRouterBuffers:
    """Create cached Metal buffer for router weights.

    Router weights must be in column-major layout [hidden_dim, num_experts]
    where each column is one expert's weight vector.

    Args:
        device: Metal device.
        router_weights: Router weight tensor [num_experts, hidden_dim] or
            [hidden_dim, num_experts]. Will be transposed to column-major if needed.

    Returns:
        CachedRouterBuffers with Metal buffer.
    """
    require_mps()

    # Router weights from nn.Linear are [out_features, in_features] = [num_experts, hidden_dim]
    # The Metal kernel expects [hidden_dim, num_experts] column-major
    # So we need to transpose: [num_experts, hidden_dim] -> [hidden_dim, num_experts]
    if router_weights.shape[0] < router_weights.shape[1]:
        # Already [hidden_dim, num_experts] - likely
        weights_col_major = router_weights.half().contiguous()
    else:
        # [num_experts, hidden_dim] -> transpose to [hidden_dim, num_experts]
        weights_col_major = router_weights.t().half().contiguous()

    return CachedRouterBuffers(router_weights=mps_tensor_to_metal_buffer(weights_col_major, device))


def dispatch_moe_router_fused(
    lib: MetalKernelLibrary,
    hidden: torch.Tensor,
    router_weights: torch.Tensor | None,
    num_experts: int,
    top_k: int,
    *,
    cached_router: CachedRouterBuffers | None = None,
    router_pool: RouterBufferPool | None = None,
    use_coalesced: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused router: matmul + softmax + top-k in single kernel launch.

    Replaces 3 PyTorch operations:
        logits = hidden @ router_weights.T
        probs = softmax(logits, dim=-1)
        topk_probs, topk_ids = topk(probs, k=top_k)

    With a single Metal kernel that avoids intermediate tensor allocations
    and 2 kernel launch boundaries.

    Args:
        lib: MetalKernelLibrary with moe_router.metal compiled.
        hidden: Input hidden states [batch, hidden_dim].
        router_weights: Router weight tensor [num_experts, hidden_dim].
            Ignored if cached_router is provided.
        num_experts: Number of experts.
        top_k: Number of experts to select per token.
        cached_router: Pre-cached router weight buffer.
        router_pool: Buffer pool for input/output buffers.
        use_coalesced: If True (default), use the coalesced kernel variant with
            row-major weights [num_experts, hidden_dim]. This provides ~10x better
            memory bandwidth than the strided column-major variant.

    Returns:
        Tuple of (expert_ids [batch, top_k] uint32, expert_probs [batch, top_k] fp16).
        Probabilities are renormalized to sum to 1.
    """
    device = lib.device
    batch_size = hidden.shape[0]
    hidden_dim = hidden.shape[1]

    # Get router weights buffer
    if cached_router is not None:
        router_buf = cached_router.router_weights
        # cached_router uses the layout that was cached; assume coalesced if use_coalesced
    elif router_weights is not None:
        # Determine kernel layout requirements
        # moe_router_fused_small: expects [hidden_dim, num_experts] column-major
        # moe_router_fused: expects [hidden_dim, num_experts] column-major
        # moe_router_fused_coalesced: expects [num_experts, hidden_dim] row-major
        if num_experts <= 32 and hidden_dim <= 256:
            # Small kernel uses column-major layout regardless of use_coalesced
            weights_col_major = router_weights.t().half().contiguous()
            router_buf = mps_tensor_to_metal_buffer(weights_col_major, device)
        elif use_coalesced:
            # Keep as row-major [num_experts, hidden_dim] for coalesced access
            # The coalesced kernel accesses weights as w_row[d] sequentially
            weights_row_major = router_weights.half().contiguous()
            router_buf = mps_tensor_to_metal_buffer(weights_row_major, device)
        else:
            # Legacy: transpose to column-major [hidden_dim, num_experts]
            # The original kernel accesses w_col[d * num_experts] (strided)
            weights_col_major = router_weights.t().half().contiguous()
            router_buf = mps_tensor_to_metal_buffer(weights_col_major, device)
    else:
        raise ValueError(
            "Either router_weights or cached_router must be provided")

    # Get input/output buffers
    if router_pool is not None:
        hidden_buf = router_pool.get_hidden_buffer(batch_size, hidden)
        expert_ids_tensor, expert_ids_buf = router_pool.get_expert_ids_output(
            batch_size)
        expert_probs_tensor, expert_probs_buf = router_pool.get_expert_probs_output(
            batch_size)
        params_buf = router_pool.get_params_buffer(
            batch_size, hidden_dim, num_experts, top_k)
    else:
        hidden_buf = mps_tensor_to_metal_buffer(
            hidden.half().contiguous(), device)
        expert_ids_tensor = torch.zeros(
            batch_size, top_k, dtype=torch.uint32, device="mps")
        expert_ids_buf = mps_tensor_to_metal_buffer(
            expert_ids_tensor, device, copy_back=True)
        expert_probs_tensor = torch.zeros(
            batch_size, top_k, dtype=torch.float16, device="mps")
        expert_probs_buf = mps_tensor_to_metal_buffer(
            expert_probs_tensor, device, copy_back=True)
        params_data = np.array(
            [batch_size, hidden_dim, num_experts, top_k], dtype=np.uint32)
        params_buf = device.newBufferWithBytes_length_options_(
            params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
        )

    # Select kernel variant
    # Use moe_router_fused_small for small dimensions (more efficient register usage)
    # Use moe_router_fused_coalesced for larger dimensions with row-major weights
    if num_experts <= 32 and hidden_dim <= 256:
        kernel_name = "moe_router_fused_small"
    elif use_coalesced:
        # Coalesced variant: weights in [num_experts, hidden_dim] row-major
        # Eliminates strided memory access (stride=num_experts per element)
        kernel_name = "moe_router_fused_coalesced"
    else:
        # Legacy strided variant: weights in [hidden_dim, num_experts]
        kernel_name = "moe_router_fused"

    # Dispatch: 1 threadgroup per token, ROUTER_THREADS=128 threads per group
    buffer_list = [
        hidden_buf,  # buffer(0): hidden [batch, hidden_dim]
        router_buf,  # buffer(1): router_weights [hidden_dim, num_experts]
        expert_ids_buf,  # buffer(2): expert_ids [batch, top_k] output
        expert_probs_buf,  # buffer(3): expert_probs [batch, top_k] output
        params_buf,  # buffer(4-7): batch_size, hidden_dim, num_experts, top_k
    ]

    # The kernel expects individual constant parameters, but we packed them
    # Actually looking at the kernel signature:
    #   constant uint& batch_size [[buffer(4)]]
    #   constant uint& hidden_dim [[buffer(5)]]
    #   constant uint& num_experts [[buffer(6)]]
    #   constant uint& top_k [[buffer(7)]]
    # So we need separate buffers for each parameter

    # Create individual parameter buffers
    batch_size_data = np.array([batch_size], dtype=np.uint32)
    hidden_dim_data = np.array([hidden_dim], dtype=np.uint32)
    num_experts_data = np.array([num_experts], dtype=np.uint32)
    top_k_data = np.array([top_k], dtype=np.uint32)

    batch_size_buf = device.newBufferWithBytes_length_options_(
        batch_size_data.tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    hidden_dim_buf = device.newBufferWithBytes_length_options_(
        hidden_dim_data.tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    num_experts_buf = device.newBufferWithBytes_length_options_(
        num_experts_data.tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    top_k_buf = device.newBufferWithBytes_length_options_(
        top_k_data.tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    buffer_list = [
        hidden_buf,  # buffer(0)
        router_buf,  # buffer(1)
        expert_ids_buf,  # buffer(2)
        expert_probs_buf,  # buffer(3)
        batch_size_buf,  # buffer(4)
        hidden_dim_buf,  # buffer(5)
        num_experts_buf,  # buffer(6)
        top_k_buf,  # buffer(7)
    ]

    cmd_buf = dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(batch_size, 1, 1),  # 1 threadgroup per batch element
        threadgroup=(128, 1, 1),  # ROUTER_THREADS = 128
        buffers=buffer_list,
        wait=False,
    )
    if cmd_buf is not None:
        cmd_buf.waitUntilCompleted()

    return expert_ids_tensor, expert_probs_tensor


def dispatch_moe_fused_router_sorted(
    lib: MetalKernelLibrary,
    hidden: torch.Tensor,
    router_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused router with sorted expert indices in a single kernel.

    Performs in one GPU pass:
        1. Router GEMV: hidden @ router_weights -> logits
        2. Softmax normalization
        3. Top-k expert selection per token
        4. Grouping: sort token-expert pairs by expert ID

    This eliminates CPU-side sorting and grouping, providing sorted indices
    directly for efficient batched expert GEMM execution.

    Output format:
        - sorted_indices: [batch * top_k] - indices into expert_ids, grouped by expert
        - expert_offsets: [num_experts + 1] - start/end indices for each expert
        - topk_expert_ids: [batch, top_k] - top-k expert selections (for reference)
        - topk_probs: [batch, top_k] - normalized probabilities

    Memory layout for sorted output:
        expert 0: sorted_indices[offsets[0]:offsets[1]]
        expert 1: sorted_indices[offsets[1]:offsets[2]]
        ...

    Args:
        lib: MetalKernelLibrary with moe_fused_router.metal compiled.
        hidden: Input hidden states [batch, hidden_dim] (fp16).
        router_weights: Router weight tensor [hidden_dim, num_experts] or
            [num_experts, hidden_dim] (fp16). The coalesced variant
            expects row-major [num_experts, hidden_dim] for better memory access.
        num_experts: Number of experts.
        top_k: Number of experts to select per token.

    Returns:
        Tuple of (expert_ids, expert_probs, sorted_indices, expert_offsets).
        expert_ids: [batch, top_k] uint32 tensor.
        expert_probs: [batch, top_k] fp16 tensor (renormalized).
        sorted_indices: [batch * top_k] uint32 tensor grouped by expert.
        expert_offsets: [num_experts + 1] uint32 tensor of cumulative counts.
    """
    require_mps()
    device = lib.device
    batch_size = hidden.shape[0]
    hidden_dim = hidden.shape[1]
    total_assignments = batch_size * top_k

    # Prepare input buffers
    hidden_buf = mps_tensor_to_metal_buffer(hidden.half().contiguous(), device)

    # Use coalesced variant with transposed weights for better memory access
    # The kernel expects [num_experts, hidden_dim] layout
    weights_coalesced = router_weights.t().half().contiguous()
    router_buf = mps_tensor_to_metal_buffer(weights_coalesced, device)

    # Prepare output buffers
    expert_offsets_tensor = torch.zeros(
        num_experts + 1, dtype=torch.uint32, device="mps")
    expert_offsets_buf = mps_tensor_to_metal_buffer(
        expert_offsets_tensor, device, copy_back=True)

    sorted_indices_tensor = torch.zeros(
        total_assignments, dtype=torch.uint32, device="mps")
    sorted_indices_buf = mps_tensor_to_metal_buffer(
        sorted_indices_tensor, device, copy_back=True)

    topk_expert_ids_tensor = torch.zeros(
        batch_size, top_k, dtype=torch.uint32, device="mps")
    topk_expert_ids_buf = mps_tensor_to_metal_buffer(
        topk_expert_ids_tensor, device, copy_back=True)

    topk_probs_tensor = torch.zeros(
        batch_size, top_k, dtype=torch.float16, device="mps")
    topk_probs_buf = mps_tensor_to_metal_buffer(
        topk_probs_tensor, device, copy_back=True)

    # Create RouterParams struct
    # batch_size, hidden_dim, num_experts, top_k (all uint32)
    RouterParams = struct.Struct('IIII')
    params_data = RouterParams.pack(batch_size, hidden_dim, num_experts, top_k)
    params_buf = device.newBufferWithBytes_length_options_(
        params_data, params_data[1], Metal.MTLResourceStorageModeShared
    )

    # Dispatch the coalesced variant for best memory access pattern
    buffer_list = [
        hidden_buf,  # buffer(0): hidden [batch, hidden_dim]
        # buffer(1): router_weights [num_experts, hidden_dim] TRANSPOSED
        router_buf,
        # buffer(2): expert_offsets [num_experts + 1] output (atomic counters)
        expert_offsets_buf,
        sorted_indices_buf,  # buffer(3): sorted_indices [batch * top_k] output
        # buffer(4): topk_expert_ids [batch, top_k] output
        topk_expert_ids_buf,
        topk_probs_buf,  # buffer(5): topk_probs [batch, top_k] output
        params_buf,  # buffer(6): RouterParams struct
    ]

    # Grid: 1 threadgroup per batch element
    # Threadgroup: 256 threads per group (ROUTER_TG_SIZE)
    cmd_buf = dispatch_kernel(
        lib,
        function_name="moe_fused_router_sorted_coalesced",
        grid=(batch_size, 1, 1),
        threadgroup=(256, 1, 1),
        buffers=buffer_list,
        wait=True,
    )

    # Convert to int64 for consistency with moe_dispatch.py
    expert_ids = topk_expert_ids_tensor.to(torch.int64)
    expert_probs = topk_probs_tensor.to(torch.float32)
    sorted_indices = sorted_indices_tensor.to(torch.int64)
    expert_offsets = expert_offsets_tensor.to(torch.int64)

    return expert_ids, expert_probs, sorted_indices, expert_offsets


def dispatch_moe_trellis_swiglu_grouped_fairway(
    *,
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_probs: torch.Tensor,
    cached_buffers: CachedWeightBuffers,
    bit_tuple: tuple[int, int, int],
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    buffer_pool: MoEBufferPool | None = None,
    use_fp32_acc: bool = False,
) -> torch.Tensor:
    """Primary fairway dispatch path using grouped expert inputs.

    Dispatches `moe_trellis_swiglu_grouped` with pre-grouped routing tensors:
    - `sorted_token_ids` grouped by expert
    - `expert_offsets` cumulative expert ranges
    - `sorted_probs` aligned probabilities

    Args:
        activations: Grouped activations [batch, hidden_dim].
        sorted_token_ids: [total_assignments] token IDs grouped by expert.
        expert_offsets: [num_experts + 1] cumulative offsets.
        sorted_probs: [total_assignments] probabilities aligned to sorted tokens.
        cached_buffers: Metal buffers for this bit tuple's expert weights.
        bit_tuple: (gate_bits, up_bits, down_bits) for this group.
        use_fp32_acc: Ignored for grouped kernel (kept for API compatibility).

    Returns:
        Output [batch, hidden_dim] in fp16 with weighted accumulation.
    """
    del use_fp32_acc  # grouped kernel has no fp32acc variant

    batch_size = activations.shape[0]
    if activations.dim() != 2:
        raise MoEDispatchValidationError(
            f"activations must be rank-2 [batch, hidden_dim], got shape={tuple(activations.shape)}"
        )
    if activations.shape[1] != hidden_dim:
        raise MoEDispatchValidationError(
            f"activations hidden dim mismatch: expected {hidden_dim}, got {activations.shape[1]}"
        )
    if sorted_token_ids.dim() != 1 or expert_offsets.dim() != 1 or sorted_probs.dim() != 1:
        raise MoEDispatchValidationError(
            "sorted_token_ids, expert_offsets, and sorted_probs must all be rank-1 tensors"
        )
    if sorted_token_ids.numel() != sorted_probs.numel():
        raise MoEDispatchValidationError(
            f"sorted_token_ids size {sorted_token_ids.numel()} must match "
            f"sorted_probs size {sorted_probs.numel()}"
        )
    if expert_offsets.numel() != (num_experts + 1):
        raise MoEDispatchValidationError(
            f"expert_offsets size {expert_offsets.numel()} must be num_experts+1 ({num_experts + 1})"
        )

    if batch_size == 0:
        return torch.zeros(0, hidden_dim, dtype=torch.float16, device=activations.device)

    device = lib.device

    # Reuse pooled activation/output/params buffers when available.
    if buffer_pool is not None:
        activations_buf = buffer_pool.get_activation_buffer(batch_size, activations)
        output_fp32, output_buf = buffer_pool.get_output_buffer(batch_size)
    else:
        activations_fp16 = (
            activations.contiguous()
            if activations.dtype == torch.float16
            else activations.half().contiguous()
        )
        activations_buf = mps_tensor_to_metal_buffer(activations_fp16, device)
        output_fp32 = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device="mps")
        output_buf = mps_tensor_to_metal_buffer(output_fp32, device, copy_back=True)

    sorted_token_ids_i32 = (
        sorted_token_ids
        if sorted_token_ids.dtype == torch.int32 and sorted_token_ids.is_contiguous()
        else sorted_token_ids.to(torch.int32).contiguous()
    )
    expert_offsets_i32 = (
        expert_offsets
        if expert_offsets.dtype == torch.int32 and expert_offsets.is_contiguous()
        else expert_offsets.to(torch.int32).contiguous()
    )
    sorted_probs_f16 = (
        sorted_probs
        if sorted_probs.dtype == torch.float16 and sorted_probs.is_contiguous()
        else sorted_probs.to(torch.float16).contiguous()
    )

    sorted_token_ids_buf = mps_tensor_to_metal_buffer(sorted_token_ids_i32, device)
    expert_offsets_buf = mps_tensor_to_metal_buffer(expert_offsets_i32, device)
    sorted_probs_buf = mps_tensor_to_metal_buffer(sorted_probs_f16, device)

    # Grouped kernel does not consume top_k directly; keep params stable for caching.
    params_top_k = 1
    if buffer_pool is None:
        params_buf = _get_or_create_moe_params_buffer(
            device=device,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=params_top_k,
            bits=bit_tuple,
        )
    else:
        params_buf = buffer_pool.get_params_buffer(
            batch_size,
            hidden_dim,
            intermediate_dim,
            num_experts,
            params_top_k,
            bit_tuple,
        )

    tile_n = 64
    grid_x = (hidden_dim + tile_n - 1) // tile_n

    buffer_list = [
        activations_buf,
        cached_buffers.gate_weights,
        cached_buffers.gate_scales,
        cached_buffers.up_weights,
        cached_buffers.up_scales,
        cached_buffers.down_weights,
        cached_buffers.down_scales,
        cached_buffers.gate_su,
        cached_buffers.gate_sv,
        cached_buffers.up_su,
        cached_buffers.up_sv,
        cached_buffers.down_su,
        cached_buffers.down_sv,
        cached_buffers.grid,
        sorted_token_ids_buf,
        expert_offsets_buf,
        sorted_probs_buf,
        output_buf,
        params_buf,
    ]

    dispatch_kernel(
        lib,
        function_name="moe_trellis_swiglu_grouped",
        grid=(grid_x, num_experts, 1),
        threadgroup=(128, 1, 1),  # GROUPED_THREADS in gemm_trellis_moe.metal
        buffers=buffer_list,
        wait=True,
    )

    if buffer_pool is not None:
        output_fp16 = buffer_pool.get_output_fp16(batch_size)
        output_fp16.copy_(output_fp32)
        return output_fp16

    return output_fp32.half()


def dispatch_moe_per_bit_tuple(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,        # [batch, hidden]
    expert_ids: torch.Tensor,         # [batch, top_k]
    expert_probs: torch.Tensor,       # [batch, top_k]
    bit_group_buffers: dict[tuple[int, int, int], tuple[Any, Any]],
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    buffer_pool: MoEBufferPool,
    use_fp32_acc: bool = True,
    output_accum: torch.Tensor | None = None,
    output_fp16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch MoE by grouping experts with same bit-width tuple.

    Reduces dispatch calls from O(selected_experts) to O(unique_bit_tuples) ~6.
    Each unique (gate_bits, up_bits, down_bits) tuple is processed once with
    all tokens that selected experts using that bit configuration.

    Args:
        lib: MetalKernelLibrary with compiled MoE kernels.
        activations: Input activations [batch, hidden_dim].
        expert_ids: Selected expert IDs [batch, top_k].
        expert_probs: Expert routing weights [batch, top_k].
        bit_group_buffers: Dict mapping bit tuple to one of:
            - (expert_list, cached_buffers)
            - (cached_buffers, expert_list)
            Each key is (gate_bits, up_bits, down_bits).
        hidden_dim: Hidden dimension.
        intermediate_dim: Intermediate (FFN) dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        buffer_pool: Buffer pool for dynamic allocations.
        use_fp32_acc: Use FP32 accumulation (default True).

    Returns:
        Output tensor [batch, hidden_dim] in fp16.
    """
    batch_size = activations.shape[0]

    # Create or reuse output accumulator in fp32
    if output_accum is None:
        output_accum = torch.zeros(
            batch_size, hidden_dim, dtype=torch.float32, device="mps"
        )
    else:
        output_accum.zero_()

    def _dispatch_legacy_per_tuple(
        *,
        group_activations: torch.Tensor,
        group_expert_ids: torch.Tensor,
        group_expert_probs: torch.Tensor,
        bit_tuple: tuple[int, int, int],
        cached_buffers: CachedWeightBuffers,
    ) -> torch.Tensor:
        """Legacy per-tuple dispatch used as compatibility fallback."""
        return dispatch_moe_trellis_swiglu(
            lib,
            activations=group_activations,
            gate_weights=cached_buffers.gate_weights,
            gate_scales=cached_buffers.gate_scales,
            up_weights=cached_buffers.up_weights,
            up_scales=cached_buffers.up_scales,
            down_weights=cached_buffers.down_weights,
            down_scales=cached_buffers.down_scales,
            gate_su=cached_buffers.gate_su,
            gate_sv=cached_buffers.gate_sv,
            up_su=cached_buffers.up_su,
            up_sv=cached_buffers.up_sv,
            down_su=cached_buffers.down_su,
            down_sv=cached_buffers.down_sv,
            grid=cached_buffers.grid,
            expert_ids=group_expert_ids,
            expert_probs=group_expert_probs,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=1,  # Each token has 1 expert in this group
            bits=bit_tuple,
            cached_buffers=cached_buffers,
            buffer_pool=buffer_pool,
            use_fp32_acc=use_fp32_acc,
        )

    # Process each unique bit tuple group
    for bit_tuple, entry in bit_group_buffers.items():
        expert_list, cached_buffers = _normalize_bit_group_entry(
            bit_tuple=bit_tuple,
            entry=entry,
        )

        # Skip if cached_buffers is a dict (CPU tensors not yet converted)
        if isinstance(cached_buffers, dict):
            continue

        # Build mask: True where expert_ids routes into this bit tuple.
        mask = torch.zeros_like(expert_ids, dtype=torch.bool)
        for expert_id in expert_list:
            mask |= (expert_ids == expert_id)

        # If no hits, skip this group
        if not mask.any():
            continue

        # Get indices of matching (batch, slot) pairs
        batch_indices, slot_indices = mask.nonzero(as_tuple=True)

        if len(batch_indices) == 0:
            continue

        # Gather inputs for this group
        group_activations = activations[batch_indices]
        group_expert_ids = expert_ids[batch_indices, slot_indices].unsqueeze(1)
        group_expert_probs = expert_probs[batch_indices, slot_indices].unsqueeze(1)

        # Primary path: grouped fairway dispatch for this active tuple.
        # Fallback path: preserve previous dispatch_moe_trellis_swiglu behavior.
        try:
            sorted_token_ids, expert_offsets, sorted_probs = prepare_fairway_grouped_inputs(
                expert_ids=group_expert_ids,
                expert_probs=group_expert_probs,
                num_experts=num_experts,
            )
            group_output = dispatch_moe_trellis_swiglu_grouped_fairway(
                lib=lib,
                activations=group_activations,
                sorted_token_ids=sorted_token_ids,
                expert_offsets=expert_offsets,
                sorted_probs=sorted_probs,
                cached_buffers=cached_buffers,
                bit_tuple=bit_tuple,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=num_experts,
                buffer_pool=buffer_pool,
            )
        except Exception as exc:
            logger.debug(
                "grouped fairway dispatch failed for bits=%s, falling back to legacy path: %s",
                bit_tuple,
                exc,
            )
            group_output = _dispatch_legacy_per_tuple(
                group_activations=group_activations,
                group_expert_ids=group_expert_ids,
                group_expert_probs=group_expert_probs,
                bit_tuple=bit_tuple,
                cached_buffers=cached_buffers,
            )

        # Accumulate weighted outputs in fp32
        # group_output is [num_matches, hidden_dim] in fp16
        # We need to accumulate back to the original batch positions
        for i, batch_idx in enumerate(batch_indices):
            # Weight is already applied in the kernel via expert_probs
            output_accum[batch_idx] += group_output[i].float()

    # Return final output in fp16
    if output_fp16 is None:
        return output_accum.half()
    output_fp16.copy_(output_accum)
    return output_fp16
