"""Metal dispatch for Trellis MoE kernels.

Provides Python wrappers for gemm_trellis_moe.metal kernels:
- dispatch_moe_trellis_swiglu: Fused MoE GEMM with SwiGLU activation

CRITICAL: The kernel uses per-token expert routing. Each token uses its OWN
assigned expert from expert_ids[token * top_k + slot], NOT a shared expert
for all tokens in a tile. The grid is 3D: (n_blocks, tokens, slots).

This replaces the slow sequential expert iteration in TrellisMoEMLP with a
single batched kernel that processes all experts in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..metal_dispatch import (
    HAS_METAL,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)

if HAS_METAL:
    import Metal


class MoEDispatchValidationError(ValueError):
    """Raised when MoE dispatch input validation fails."""

    pass


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
    """
    require_mps()

    def ensure_half(t: torch.Tensor) -> torch.Tensor:
        if t.dtype == torch.float16:
            return t.contiguous()
        return t.half().contiguous()

    return CachedWeightBuffers(
        gate_weights=mps_tensor_to_metal_buffer(gate_weights.contiguous(), device),
        gate_scales=mps_tensor_to_metal_buffer(ensure_half(gate_scales), device),
        up_weights=mps_tensor_to_metal_buffer(up_weights.contiguous(), device),
        up_scales=mps_tensor_to_metal_buffer(ensure_half(up_scales), device),
        down_weights=mps_tensor_to_metal_buffer(down_weights.contiguous(), device),
        down_scales=mps_tensor_to_metal_buffer(ensure_half(down_scales), device),
        gate_su=mps_tensor_to_metal_buffer(ensure_half(gate_su), device),
        gate_sv=mps_tensor_to_metal_buffer(ensure_half(gate_sv), device),
        up_su=mps_tensor_to_metal_buffer(ensure_half(up_su), device),
        up_sv=mps_tensor_to_metal_buffer(ensure_half(up_sv), device),
        down_su=mps_tensor_to_metal_buffer(ensure_half(down_su), device),
        down_sv=mps_tensor_to_metal_buffer(ensure_half(down_sv), device),
        grid=mps_tensor_to_metal_buffer(ensure_half(grid), device),
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
    """Create cached Metal buffers from CPU tensors."""
    from ..metal_dispatch import cpu_tensor_to_metal_buffer

    def ensure_half_cpu(t: torch.Tensor) -> torch.Tensor:
        if t.is_mps or t.is_cuda:
            raise ValueError(f"Tensor must be on CPU, got device={t.device}")
        if t.dtype == torch.float16:
            return t.contiguous()
        return t.half().contiguous()

    return CachedWeightBuffers(
        gate_weights=cpu_tensor_to_metal_buffer(gate_weights.contiguous(), device),
        gate_scales=cpu_tensor_to_metal_buffer(ensure_half_cpu(gate_scales), device),
        up_weights=cpu_tensor_to_metal_buffer(up_weights.contiguous(), device),
        up_scales=cpu_tensor_to_metal_buffer(ensure_half_cpu(up_scales), device),
        down_weights=cpu_tensor_to_metal_buffer(down_weights.contiguous(), device),
        down_scales=cpu_tensor_to_metal_buffer(ensure_half_cpu(down_scales), device),
        gate_su=cpu_tensor_to_metal_buffer(ensure_half_cpu(gate_su), device),
        gate_sv=cpu_tensor_to_metal_buffer(ensure_half_cpu(gate_sv), device),
        up_su=cpu_tensor_to_metal_buffer(ensure_half_cpu(up_su), device),
        up_sv=cpu_tensor_to_metal_buffer(ensure_half_cpu(up_sv), device),
        down_su=cpu_tensor_to_metal_buffer(ensure_half_cpu(down_su), device),
        down_sv=cpu_tensor_to_metal_buffer(ensure_half_cpu(down_sv), device),
        grid=cpu_tensor_to_metal_buffer(ensure_half_cpu(grid), device),
    )


class MoEBufferPool:
    """Reusable buffer pool for MoE kernel dispatch."""

    def __init__(self, device: Any, hidden_dim: int, max_batch: int = 32):
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_batch = max_batch

        self._activation_buffers: dict[int, tuple[torch.Tensor, Any]] = {}
        self._expert_ids_buffers: dict[tuple[int, int], tuple[torch.Tensor, Any]] = {}
        self._expert_probs_buffers: dict[tuple[int, int], tuple[torch.Tensor, Any]] = {}
        self._output_buffers: dict[int, tuple[torch.Tensor, Any]] = {}
        self._params_buffers: dict[tuple[int, int, int, int, int, int], Any] = {}  # keyed by params tuple

        for batch in [1, 2, 4, 8, 16, 32]:
            if batch <= max_batch:
                self._preallocate(batch)

    def _preallocate(self, batch_size: int) -> None:
        act_tensor = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.float16, device="mps"
        )
        act_buf = mps_tensor_to_metal_buffer(act_tensor, self.device)
        self._activation_buffers[batch_size] = (act_tensor, act_buf)

        out_tensor = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.float32, device="mps"
        )
        out_buf = mps_tensor_to_metal_buffer(out_tensor, self.device, copy_back=True)
        self._output_buffers[batch_size] = (out_tensor, out_buf)

    def get_activation_buffer(
        self, batch_size: int, activations: torch.Tensor
    ) -> Any:
        if batch_size in self._activation_buffers:
            tensor, buf = self._activation_buffers[batch_size]
            tensor.copy_(activations)
            return buf
        return mps_tensor_to_metal_buffer(activations.contiguous(), self.device)

    def get_expert_ids_buffer(
        self, batch_size: int, top_k: int, expert_ids: torch.Tensor
    ) -> Any:
        key = (batch_size, top_k)
        if key not in self._expert_ids_buffers:
            tensor = torch.zeros(batch_size, top_k, dtype=torch.int32, device="mps")
            buf = mps_tensor_to_metal_buffer(tensor, self.device)
            self._expert_ids_buffers[key] = (tensor, buf)
        tensor, buf = self._expert_ids_buffers[key]
        tensor.copy_(expert_ids.int())
        return buf

    def get_expert_probs_buffer(
        self, batch_size: int, top_k: int, expert_probs: torch.Tensor
    ) -> Any:
        key = (batch_size, top_k)
        if key not in self._expert_probs_buffers:
            tensor = torch.zeros(batch_size, top_k, dtype=torch.float16, device="mps")
            buf = mps_tensor_to_metal_buffer(tensor, self.device)
            self._expert_probs_buffers[key] = (tensor, buf)
        tensor, buf = self._expert_probs_buffers[key]
        tensor.copy_(expert_probs.to(torch.float16))
        return buf

    def get_output_buffer(self, batch_size: int) -> tuple[torch.Tensor, Any]:
        if batch_size in self._output_buffers:
            tensor, buf = self._output_buffers[batch_size]
            tensor.zero_()
            return tensor, buf
        tensor = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.float32, device="mps"
        )
        buf = mps_tensor_to_metal_buffer(tensor, self.device, copy_back=True)
        return tensor, buf

    def preallocate_top_k(self, top_k: int, batch_sizes: list[int] | None = None) -> None:
        if batch_sizes is None:
            batch_sizes = [b for b in [1, 2, 4, 8, 16, 32] if b <= self.max_batch]
        for bs in batch_sizes:
            key = (bs, top_k)
            if key not in self._expert_ids_buffers:
                tensor = torch.zeros(bs, top_k, dtype=torch.int32, device="mps")
                buf = mps_tensor_to_metal_buffer(tensor, self.device)
                self._expert_ids_buffers[key] = (tensor, buf)
            if key not in self._expert_probs_buffers:
                tensor = torch.zeros(bs, top_k, dtype=torch.float16, device="mps")
                buf = mps_tensor_to_metal_buffer(tensor, self.device)
                self._expert_probs_buffers[key] = (tensor, buf)

    def get_params_buffer(
        self,
        batch_size: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        bits: int,
    ) -> Any:
        """Get or create a cached params buffer for the given parameters."""
        key = (batch_size, hidden_dim, intermediate_dim, num_experts, top_k, bits)
        if key not in self._params_buffers:
            n_levels = 1 << bits
            params_data = np.array(
                [batch_size, hidden_dim, intermediate_dim, num_experts, top_k, bits, 128, n_levels],
                dtype=np.uint32,
            )
            self._params_buffers[key] = self.device.newBufferWithBytes_length_options_(
                params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
            )
        return self._params_buffers[key]

    def clear(self) -> None:
        self._activation_buffers.clear()
        self._expert_ids_buffers.clear()
        self._expert_probs_buffers.clear()
        self._output_buffers.clear()
        self._params_buffers.clear()


def get_buffer_stats() -> dict[str, int]:
    """Get buffer caching statistics for debugging.

    DEPRECATED: Stats tracking removed from hot path for performance.
    Returns zeros for backward compatibility.
    """
    return {
        "dispatch_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }


def reset_buffer_stats() -> None:
    """Reset buffer caching statistics.

    DEPRECATED: Stats tracking removed from hot path for performance.
    No-op for backward compatibility.
    """
    pass


def select_moe_kernel(batch_size: int, use_fp32_acc: bool) -> tuple[str, int]:
    """Select optimal MoE kernel and tile size for given batch size."""
    if batch_size == 1:
        base = "moe_trellis_swiglu_decode"
        tile_n = 32
    elif batch_size > 8:
        base = "moe_trellis_swiglu_large_batch"
        tile_n = 128
    elif batch_size >= 2:
        base = "moe_trellis_swiglu_prefill4"
        tile_n = 64
    else:
        base = "moe_trellis_swiglu"
        tile_n = 64

    if use_fp32_acc:
        if base in ("moe_trellis_swiglu_decode", "moe_trellis_swiglu_large_batch"):
            return base, tile_n
        return base + "_fp32acc", tile_n
    return base, tile_n


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
    bits: int,
    *,
    cached_buffers: CachedWeightBuffers | None = None,
    buffer_pool: MoEBufferPool | None = None,
    use_fp32_acc: bool = False,
) -> torch.Tensor:
    """Fused MoE GEMM with Trellis quantization and SwiGLU activation.

    HOT PATH: This function is called for every forward pass. Keep it minimal:
    1. Get cached buffers (dict lookup)
    2. Create activation buffer (minimal)
    3. Dispatch kernel
    4. Return result

    No validation between steps for maximum performance.
    """
    require_mps()

    device = lib.device
    batch_size = activations.shape[0]

    # Get buffers (fast path with pool avoids contiguous/dtype copies)
    if buffer_pool is not None:
        # Buffer pool handles the copy internally, avoiding intermediate allocations
        activations_buf = buffer_pool.get_activation_buffer(batch_size, activations)
        expert_ids_buf = buffer_pool.get_expert_ids_buffer(batch_size, top_k, expert_ids)
        expert_probs_buf = buffer_pool.get_expert_probs_buffer(batch_size, top_k, expert_probs)
    else:
        # Slow path - need contiguous tensors for buffer creation
        activations = activations.contiguous()
        expert_ids = expert_ids.int().contiguous()
        expert_probs = expert_probs.to(torch.float16).contiguous()
        activations_buf = mps_tensor_to_metal_buffer(activations, device)
        expert_ids_buf = mps_tensor_to_metal_buffer(expert_ids, device)
        expert_probs_buf = mps_tensor_to_metal_buffer(expert_probs, device)

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
    if buffer_pool is not None:
        output_fp32, output_buf = buffer_pool.get_output_buffer(batch_size)
        # Get cached params buffer from pool
        params_buf = buffer_pool.get_params_buffer(
            batch_size, hidden_dim, intermediate_dim, num_experts, top_k, bits
        )
    else:
        output_fp32 = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device="mps")
        output_buf = mps_tensor_to_metal_buffer(output_fp32, device, copy_back=True)
        # Create params buffer (slow path)
        n_levels = 1 << bits
        params_data = np.array(
            [batch_size, hidden_dim, intermediate_dim, num_experts, top_k, bits, 128, n_levels],
            dtype=np.uint32,
        )
        params_buf = device.newBufferWithBytes_length_options_(
            params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
        )

    # Select kernel and compute grid
    kernel_name, tile_n = select_moe_kernel(batch_size, use_fp32_acc)
    is_decode_kernel = kernel_name == "moe_trellis_swiglu_decode"
    is_prefill4_kernel = "prefill4" in kernel_name

    if is_decode_kernel:
        threads_per_tg = 64
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

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, grid_z),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=buffer_list,
        wait=True,
    )

    return output_fp32.half()
