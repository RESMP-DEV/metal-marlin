"""Complete trellis-quantized model for inference.

Provides a high-level nn.Module interface for loading and running
trellis-quantized models with support for dense and MoE layers.

Buffer Caching Strategy:
-----------------------
Weight buffers are created lazily on first forward pass and cached for reuse.
This achieves <1ms overhead for repeated inference by avoiding Metal buffer
creation during the decode phase.

- MoE layers: CachedWeightBuffers holds 13 Metal buffers for stacked expert weights
- Output buffers: OutputBufferPool pre-allocates common batch sizes (1, 2, 4, 8, 16)

Cache invalidation: Buffers are invalidated when model weights change (e.g., after
quantization or fine-tuning). Call invalidate_buffer_cache() to clear all cached
buffers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metal_dispatch import HAS_METAL, MetalKernelLibrary, mps_tensor_to_metal_buffer
from ..transformer import RMSNorm
from .attention import TrellisMLAConfig, TrellisMLAttention
from .config import TrellisModelConfig
from .kv_cache import TrellisKVCache
from .layer import TrellisDenseMLP
from .linear import TrellisLinear
from .moe_dispatch import (
    CachedWeightBuffers,
    MoEBufferPool,
    create_cached_weight_buffers,
    dispatch_moe_trellis_swiglu,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .loader import TrellisModelLoader

if HAS_METAL:
    pass


# Module-level timing stats for buffer caching performance measurement
_buffer_timing_stats: dict[str, list[float]] = {
    "first_call_ms": [],
    "cached_call_ms": [],
    "buffer_creation_ms": [],
}


def get_buffer_timing_stats() -> dict[str, Any]:
    """Get timing statistics for buffer caching performance.

    Returns:
        Dictionary with timing stats:
        - first_call_ms: List of first-call times (buffer creation)
        - cached_call_ms: List of subsequent call times (cached path)
        - first_call_avg_ms: Average first-call time
        - cached_call_avg_ms: Average cached-call time
        - speedup: Ratio of first_call_avg to cached_call_avg
    """
    stats = dict(_buffer_timing_stats)
    if stats["first_call_ms"]:
        stats["first_call_avg_ms"] = sum(stats["first_call_ms"]) / len(stats["first_call_ms"])
    else:
        stats["first_call_avg_ms"] = 0.0
    if stats["cached_call_ms"]:
        stats["cached_call_avg_ms"] = sum(stats["cached_call_ms"]) / len(stats["cached_call_ms"])
    else:
        stats["cached_call_avg_ms"] = 0.0
    if stats["buffer_creation_ms"]:
        stats["buffer_creation_avg_ms"] = sum(stats["buffer_creation_ms"]) / len(
            stats["buffer_creation_ms"]
        )
    else:
        stats["buffer_creation_avg_ms"] = 0.0
    if stats["cached_call_avg_ms"] > 0:
        stats["speedup"] = stats["first_call_avg_ms"] / stats["cached_call_avg_ms"]
    else:
        stats["speedup"] = float("inf")
    return stats


def reset_buffer_timing_stats() -> None:
    """Reset buffer timing statistics."""
    global _buffer_timing_stats
    _buffer_timing_stats = {
        "first_call_ms": [],
        "cached_call_ms": [],
        "buffer_creation_ms": [],
    }


@dataclass
class OutputBufferPool:
    """Pre-allocated output buffer pool for common batch sizes.

    Avoids repeated allocation during autoregressive decode by reusing
    output tensors of the same shape.
    """

    hidden_dim: int
    device: Any  # MTLDevice
    _buffers: dict[int, tuple[torch.Tensor, Any]] = field(default_factory=dict)

    def get_output_buffer(
        self, batch_size: int, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, Any]:
        """Get or create an output buffer for the given batch size.

        Args:
            batch_size: Number of tokens in the batch.
            dtype: Data type for the buffer.

        Returns:
            Tuple of (PyTorch tensor, Metal buffer) for output.
        """
        key = batch_size
        if key not in self._buffers:
            tensor = torch.zeros(batch_size, self.hidden_dim, dtype=dtype, device="mps")
            metal_buf = mps_tensor_to_metal_buffer(tensor, self.device, copy_back=True)
            self._buffers[key] = (tensor, metal_buf)
        return self._buffers[key]

    def preallocate(self, batch_sizes: list[int], dtype: torch.dtype = torch.float32) -> None:
        """Pre-allocate buffers for common batch sizes.

        Args:
            batch_sizes: List of batch sizes to pre-allocate.
            dtype: Data type for the buffers.
        """
        for bs in batch_sizes:
            self.get_output_buffer(bs, dtype)

    def clear(self) -> None:
        """Clear all cached buffers."""
        self._buffers.clear()

    def memory_usage_bytes(self) -> int:
        """Get total memory usage of cached buffers in bytes."""
        total = 0
        for tensor, _ in self._buffers.values():
            total += tensor.numel() * tensor.element_size()
        return total


class TrellisMoEMLP(nn.Module):
    """MoE MLP with trellis-quantized weights for MoE layers.

    Implements a mixture of experts with:
    - Router: selects top-k experts per token
    - Multiple experts: each is a dense MLP with SwiGLU
    - Shared expert: always applied (for model stability)

    This is used for layers >= first_moe_layer in GLM-4.7-Flash.

    Attributes:
        router: Linear layer for expert selection.
        experts: List of expert MLPs (TrellisDenseMLP).
        shared_expert: Always-active expert for stability.
        num_experts_per_tok: Number of experts to activate per token.
    """

    def __init__(
        self,
        router: nn.Linear,
        experts: list[TrellisDenseMLP],
        shared_expert: TrellisDenseMLP,
        num_experts_per_tok: int = 8,
        eager_buffers: bool = True,
    ):
        """Initialize TrellisMoEMLP.

        Args:
            router: Linear layer for expert selection scores.
            experts: List of expert MLPs (each a TrellisDenseMLP).
            shared_expert: Always-active expert MLP.
            num_experts_per_tok: Number of experts to activate per token.
            eager_buffers: If True (default), skip pre-stacking expert weights
                and create Metal buffers on-demand for better memory efficiency.
                If False, pre-stack weights into contiguous tensors (deprecated).
        """
        super().__init__()
        self._eager_buffers = eager_buffers
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.shared_expert = shared_expert
        self.num_experts_per_tok = num_experts_per_tok

        # Prepare contiguous expert weights for fast dispatch
        self._prepare_expert_weights()

        # Lazy Metal library and cached buffers
        self._lib: MetalKernelLibrary | None = None
        self._cached_weight_buffers: CachedWeightBuffers | None = None
        self._output_buffer_pool: OutputBufferPool | None = None
        self._buffer_pool: MoEBufferPool | None = None

        # Buffer validity tracking for cache invalidation
        self._buffer_version: int = 0
        self._weights_hash: int | None = None

        # Fast MoE kernel state management
        self._use_fast_moe = True
        self._fast_moe_failure_count = 0
        self._fast_moe_max_retries = 3  # Retries before permanent fallback
        self._fast_moe_backoff_until: float = 0.0  # Timestamp for exponential backoff
        self._fast_moe_backoff_multiplier = 1.0  # Grows with each failure
        self._fast_moe_permanently_disabled = False

        # Timing instrumentation (disabled by default for performance)
        # Set _track_timing = True to enable timing stats collection
        self._first_forward_done = False
        self._track_timing = False

        # Optional eager buffer creation for memory efficiency
        if eager_buffers:
            self._create_buffers_eagerly()

    def _prepare_expert_weights(self) -> None:
        """Prepare expert weights in contiguous format for fast MoE dispatch.

        When self._eager_buffers is True, this method does minimal work
        since _create_buffers_eagerly() will handle buffer creation.
        """
        # If using eager buffers, skip creating stacked MPS tensors
        # _create_buffers_eagerly() will create buffers directly from CPU
        if self._eager_buffers:
            # Just store dimensions for later use
            first_expert = self.experts[0]
            self.hidden_dim = first_expert.gate_proj.in_features
            self.intermediate_dim = first_expert.gate_proj.out_features
            self.bits = first_expert.gate_proj.bits
            return

        # Deprecated: using non-optimized memory path
        import warnings

        warnings.warn(
            "Using non-optimized memory path. Set eager_buffers=True "
            "or use TrellisForCausalLM.from_pretrained() for better memory efficiency.",
            DeprecationWarning,
            stacklevel=2,
        )

        num_experts = len(self.experts)

        # Get dimensions from first expert
        first_expert = self.experts[0]
        hidden_dim = first_expert.gate_proj.in_features
        intermediate_dim = first_expert.gate_proj.out_features
        bits = first_expert.gate_proj.bits

        # Stack expert weights
        gate_weights_list = []
        gate_scales_list = []
        up_weights_list = []
        up_scales_list = []
        down_weights_list = []
        down_scales_list = []

        gate_su_list = []
        gate_sv_list = []
        up_su_list = []
        up_sv_list = []
        down_su_list = []
        down_sv_list = []

        for expert in self.experts:
            # Transpose packed weights from TrellisWeight [tiles_out, tiles_in, packed]
            # to GEMM convention [tiles_in, tiles_out, packed] for MoE kernel
            gate_weights_list.append(expert.gate_proj.packed_indices.permute(1, 0, 2).contiguous())
            gate_scales_list.append(expert.gate_proj.scales)
            up_weights_list.append(expert.up_proj.packed_indices.permute(1, 0, 2).contiguous())
            up_scales_list.append(expert.up_proj.scales)
            down_weights_list.append(expert.down_proj.packed_indices.permute(1, 0, 2).contiguous())
            down_scales_list.append(expert.down_proj.scales)

            gate_su_list.append(expert.gate_proj.su)
            gate_sv_list.append(expert.gate_proj.sv)
            up_su_list.append(expert.up_proj.su)
            up_sv_list.append(expert.up_proj.sv)
            down_su_list.append(expert.down_proj.su)
            down_sv_list.append(expert.down_proj.sv)

        # Stack along new dimension 0 (num_experts)
        # Convert scales/su/sv to float16 NOW to avoid dtype conversion copies
        # during Metal buffer creation later
        self.register_buffer("gate_weights_stacked", torch.stack(gate_weights_list, dim=0))
        self.register_buffer("gate_scales_stacked", torch.stack(gate_scales_list, dim=0).half())
        self.register_buffer("up_weights_stacked", torch.stack(up_weights_list, dim=0))
        self.register_buffer("up_scales_stacked", torch.stack(up_scales_list, dim=0).half())
        self.register_buffer("down_weights_stacked", torch.stack(down_weights_list, dim=0))
        self.register_buffer("down_scales_stacked", torch.stack(down_scales_list, dim=0).half())

        self.register_buffer("gate_su_stacked", torch.stack(gate_su_list, dim=0).half())
        self.register_buffer("gate_sv_stacked", torch.stack(gate_sv_list, dim=0).half())
        self.register_buffer("up_su_stacked", torch.stack(up_su_list, dim=0).half())
        self.register_buffer("up_sv_stacked", torch.stack(up_sv_list, dim=0).half())
        self.register_buffer("down_su_stacked", torch.stack(down_su_list, dim=0).half())
        self.register_buffer("down_sv_stacked", torch.stack(down_sv_list, dim=0).half())

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.bits = bits

    def _prepare_expert_weights_cpu(self) -> dict[str, torch.Tensor]:
        """Prepare expert weights on CPU for direct Metal buffer creation.

        Returns a dict of stacked CPU tensors that can be passed directly
        to create_cached_weight_buffers_from_cpu().

        Returns:
            Dict with keys: gate_weights, gate_scales, up_weights, up_scales,
            down_weights, down_scales, gate_su, gate_sv, up_su, up_sv,
            down_su, down_sv, grid. All tensors on CPU.
        """
        # Collect weights from experts
        gate_weights_list = []
        gate_scales_list = []
        up_weights_list = []
        up_scales_list = []
        down_weights_list = []
        down_scales_list = []
        gate_su_list = []
        gate_sv_list = []
        up_su_list = []
        up_sv_list = []
        down_su_list = []
        down_sv_list = []

        for expert in self.experts:
            # Transpose packed weights from TrellisWeight [tiles_out, tiles_in, packed]
            # to GEMM convention [tiles_in, tiles_out, packed] for MoE kernel
            # Keep on CPU by calling .cpu() explicitly
            gate_weights_list.append(
                expert.gate_proj.packed_indices.cpu().permute(1, 0, 2).contiguous()
            )
            gate_scales_list.append(expert.gate_proj.scales.cpu().half())
            up_weights_list.append(
                expert.up_proj.packed_indices.cpu().permute(1, 0, 2).contiguous()
            )
            up_scales_list.append(expert.up_proj.scales.cpu().half())
            down_weights_list.append(
                expert.down_proj.packed_indices.cpu().permute(1, 0, 2).contiguous()
            )
            down_scales_list.append(expert.down_proj.scales.cpu().half())

            gate_su_list.append(expert.gate_proj.su.cpu().half())
            gate_sv_list.append(expert.gate_proj.sv.cpu().half())
            up_su_list.append(expert.up_proj.su.cpu().half())
            up_sv_list.append(expert.up_proj.sv.cpu().half())
            down_su_list.append(expert.down_proj.su.cpu().half())
            down_sv_list.append(expert.down_proj.sv.cpu().half())

        # Stack on CPU
        return {
            "gate_weights": torch.stack(gate_weights_list, dim=0),
            "gate_scales": torch.stack(gate_scales_list, dim=0),
            "up_weights": torch.stack(up_weights_list, dim=0),
            "up_scales": torch.stack(up_scales_list, dim=0),
            "down_weights": torch.stack(down_weights_list, dim=0),
            "down_scales": torch.stack(down_scales_list, dim=0),
            "gate_su": torch.stack(gate_su_list, dim=0),
            "gate_sv": torch.stack(gate_sv_list, dim=0),
            "up_su": torch.stack(up_su_list, dim=0),
            "up_sv": torch.stack(up_sv_list, dim=0),
            "down_su": torch.stack(down_su_list, dim=0),
            "down_sv": torch.stack(down_sv_list, dim=0),
            "grid": self.experts[0].gate_proj.grid.cpu().half(),
        }

    def _get_lib(self) -> MetalKernelLibrary:
        """Get or create Metal kernel library."""
        if self._lib is None:
            self._lib = MetalKernelLibrary.from_source_dir()
        return self._lib

    def _check_fast_moe_available(self) -> bool:
        """Check if fast MoE kernel is available."""
        try:
            lib = self._get_lib()
            # Try to get pipeline - will raise if not available
            lib.get_pipeline("moe_trellis_swiglu")
            return True
        except Exception as e:
            import warnings

            warnings.warn(
                f"Fast MoE kernel unavailable, using slow path: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _get_cached_buffers(self) -> CachedWeightBuffers:
        """Get or create cached Metal buffers for static weights.

        Caches weight buffers on first call to avoid creating 13 Metal buffers
        per dispatch during decode. Only dynamic inputs (activations, expert_ids,
        expert_probs) need new buffers per call.
        """
        if self._cached_weight_buffers is not None:
            return self._cached_weight_buffers

        # If eager_buffers mode but buffers weren't created, create them now
        if getattr(self, "_eager_buffers", False):
            self._create_buffers_eagerly()
            return self._cached_weight_buffers

        # Original lazy creation path (from MPS tensors)
        start_time = time.perf_counter()
        lib = self._get_lib()
        self._cached_weight_buffers = create_cached_weight_buffers(
            device=lib.device,
            gate_weights=self.gate_weights_stacked,
            gate_scales=self.gate_scales_stacked,
            up_weights=self.up_weights_stacked,
            up_scales=self.up_scales_stacked,
            down_weights=self.down_weights_stacked,
            down_scales=self.down_scales_stacked,
            gate_su=self.gate_su_stacked,
            gate_sv=self.gate_sv_stacked,
            up_su=self.up_su_stacked,
            up_sv=self.up_sv_stacked,
            down_su=self.down_su_stacked,
            down_sv=self.down_sv_stacked,
            grid=self.experts[0].gate_proj.grid,
        )
        # Record buffer creation time
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _buffer_timing_stats["buffer_creation_ms"].append(elapsed_ms)

        # Compute weights hash for cache invalidation
        self._weights_hash = hash(
            (
                self.gate_weights_stacked.data_ptr(),
                self.up_weights_stacked.data_ptr(),
                self.down_weights_stacked.data_ptr(),
            )
        )
        self._buffer_version += 1

        return self._cached_weight_buffers

    def _get_output_buffer_pool(self) -> OutputBufferPool:
        """Get or create output buffer pool for common batch sizes.

        Pre-allocates output tensors for batch sizes 1, 2, 4, 8, 16 to avoid
        repeated allocation during autoregressive decode.
        """
        if self._output_buffer_pool is None:
            lib = self._get_lib()
            self._output_buffer_pool = OutputBufferPool(
                hidden_dim=self.hidden_dim,
                device=lib.device,
            )
            # Pre-allocate common decode batch sizes
            self._output_buffer_pool.preallocate([1, 2, 4, 8, 16], dtype=torch.float32)
        return self._output_buffer_pool

    def _get_buffer_pool(self) -> MoEBufferPool:
        """Get or create MoE buffer pool for dynamic input/output buffers.

        Pre-allocates activation, expert_ids, expert_probs, and output buffers
        for common batch sizes to avoid repeated buffer creation during decode.
        """
        if self._buffer_pool is None:
            lib = self._get_lib()
            self._buffer_pool = MoEBufferPool(
                device=lib.device,
                hidden_dim=self.hidden_dim,
                max_batch=32,
            )
            # Pre-allocate buffers for top_k
            self._buffer_pool.preallocate_top_k(self.num_experts_per_tok)
        return self._buffer_pool

    def invalidate_buffer_cache(self) -> None:
        """Invalidate all cached buffers.

        Call this after modifying model weights (e.g., after quantization
        or fine-tuning) to ensure buffers are recreated on next forward pass.
        """
        self._cached_weight_buffers = None
        if self._output_buffer_pool is not None:
            self._output_buffer_pool.clear()
        self._output_buffer_pool = None
        if self._buffer_pool is not None:
            self._buffer_pool.clear()
        self._buffer_pool = None
        self._buffer_version += 1
        self._weights_hash = None
        logger.debug("Buffer cache invalidated, version=%d", self._buffer_version)

    def _create_buffers_eagerly(self) -> None:
        """Create Metal buffers eagerly from CPU tensors.

        This method:
        1. Prepares expert weights on CPU (no MPS copy)
        2. Creates Metal buffers directly from CPU
        3. Deletes the stacked PyTorch tensors to free memory
        4. Optionally deletes individual expert weight tensors
        5. Pre-allocates buffer pool for decode (batch=1) fast path

        Called during __init__ when eager_buffers=True.
        """
        import gc

        from .moe_dispatch import create_cached_weight_buffers_from_cpu

        # Get Metal device - cache lib for decode fast path
        lib = self._get_lib()

        # Prepare weights on CPU
        cpu_weights = self._prepare_expert_weights_cpu()

        # Create Metal buffers directly from CPU
        self._cached_weight_buffers = create_cached_weight_buffers_from_cpu(
            device=lib.device, **cpu_weights
        )

        # Delete CPU weight copies
        del cpu_weights

        # Delete the stacked MPS tensors if they exist
        # (from _prepare_expert_weights which runs before this)
        for name in [
            "gate_weights_stacked",
            "gate_scales_stacked",
            "up_weights_stacked",
            "up_scales_stacked",
            "down_weights_stacked",
            "down_scales_stacked",
            "gate_su_stacked",
            "gate_sv_stacked",
            "up_su_stacked",
            "up_sv_stacked",
            "down_su_stacked",
            "down_sv_stacked",
        ]:
            if hasattr(self, name):
                delattr(self, name)

        # CRITICAL: Clear the original expert weights (the MPS tensors)
        # These are the source data that was copied to Metal buffers - now redundant
        # This is what actually frees the ~6GB of MPS memory
        for expert in self.experts:
            for proj in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                # Replace with empty tensors to free memory but keep module structure
                proj.register_buffer("packed_indices", torch.empty(0, dtype=torch.uint8))
                proj.register_buffer("scales", torch.empty(0, dtype=torch.float16))
                proj.register_buffer("su", torch.empty(0, dtype=torch.float16))
                proj.register_buffer("sv", torch.empty(0, dtype=torch.float16))

        # Pre-allocate buffer pool for decode fast path (batch=1 is most common)
        # This ensures _buffer_pool is ready on first forward call
        self._get_buffer_pool()

        # Force garbage collection
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.debug("Created Metal buffers eagerly, freed PyTorch tensors")

    def forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Fast forward pass using fused MoE dispatch.

        Replaces the slow sequential expert iteration (512 calls) with a
        single batched Metal kernel that processes all experts in parallel.

        For batch=1 (decode), uses an optimized path that:
        - Avoids redundant dtype conversions
        - Minimizes intermediate tensor allocations
        - Uses pre-cached buffers directly
        """
        batch_size = x.shape[0] if x.dim() == 2 else x.numel() // self.hidden_dim

        # Fast path for decode (batch=1) - minimize overhead
        if batch_size == 1:
            # All resources must be pre-initialized - fallback immediately if not
            cached = self._cached_weight_buffers
            lib = self._lib
            buffer_pool = self._buffer_pool
            if cached is None or lib is None or buffer_pool is None:
                return self._forward_slow(x)

            orig_dtype = x.dtype

            # For batch=1, x is [1, hidden_dim] - use view only if needed
            if x.dim() != 2:
                x = x.view(1, self.hidden_dim)

            # Convert to fp16 if needed (most inputs already fp16)
            if x.dtype != torch.float16:
                x = x.half()

            # Route - compute logits in router's dtype
            # Most models use fp16 router, so this is typically a no-op
            router_logits = self.router(x.to(self.router.weight.dtype))

            # Top-k selection: softmax -> topk -> normalize (in-place where possible)
            # For batch=1, this is [1, num_experts] -> [1, top_k]
            routing_weights, selected_experts = torch.topk(
                F.softmax(router_logits, dim=-1, dtype=torch.float),
                k=self.num_experts_per_tok,
                dim=-1,
            )
            # In-place normalize to avoid allocation
            routing_weights.div_(routing_weights.sum(dim=-1, keepdim=True))

            # Direct dispatch with all cached resources
            output = dispatch_moe_trellis_swiglu(
                lib=lib,
                activations=x,
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
                expert_ids=selected_experts,
                expert_probs=routing_weights,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_experts=len(self.experts),
                top_k=self.num_experts_per_tok,
                bits=self.bits,
                cached_buffers=cached,
                buffer_pool=buffer_pool,
                use_fp32_acc=self.hidden_dim >= 1024,
            )

            # Add shared expert - output is already fp16 from dispatch
            output = output + self.shared_expert(x)

            # Restore original dtype if needed
            if orig_dtype != torch.float16:
                output = output.to(orig_dtype)
            return output

        # Standard path for batch > 1 (prefill)
        orig_dtype = x.dtype

        # Flatten for processing - Metal kernels expect float16
        batch_shape = x.shape[:-1]
        x_fp16 = x.to(torch.float16) if x.dtype != torch.float16 else x
        x_flat = x_fp16.view(-1, self.hidden_dim)

        # Convert input to router's dtype for routing computation
        x_router = x_flat.to(self.router.weight.dtype)

        # Get router scores
        router_logits = self.router(x_router)

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float),
            k=self.num_experts_per_tok,
            dim=-1,
        )

        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Get cached buffers (required for fast path)
        cached_buffers = self._get_cached_buffers()
        buffer_pool = self._get_buffer_pool()

        # When using eager buffers, weight tensors have been cleared - pass None
        # The dispatch function will use cached_buffers instead
        if self._eager_buffers:
            # Pass None for weight tensors since cached_buffers will be used
            output = dispatch_moe_trellis_swiglu(
                lib=self._get_lib(),
                activations=x_flat,
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
                expert_ids=selected_experts,
                expert_probs=routing_weights,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_experts=len(self.experts),
                top_k=self.num_experts_per_tok,
                bits=self.bits,
                cached_buffers=cached_buffers,
                buffer_pool=buffer_pool,
                use_fp32_acc=self.hidden_dim >= 1024,
            )
        else:
            # Fast fused dispatch with weight tensors (for lazy buffer creation)
            output = dispatch_moe_trellis_swiglu(
                lib=self._get_lib(),
                activations=x_flat,
                gate_weights=self.gate_weights_stacked,
                gate_scales=self.gate_scales_stacked,
                up_weights=self.up_weights_stacked,
                up_scales=self.up_scales_stacked,
                down_weights=self.down_weights_stacked,
                down_scales=self.down_scales_stacked,
                gate_su=self.gate_su_stacked,
                gate_sv=self.gate_sv_stacked,
                up_su=self.up_su_stacked,
                up_sv=self.up_sv_stacked,
                down_su=self.down_su_stacked,
                down_sv=self.down_sv_stacked,
                grid=self.experts[0].gate_proj.grid,
                expert_ids=selected_experts,
                expert_probs=routing_weights,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_experts=len(self.experts),
                top_k=self.num_experts_per_tok,
                bits=self.bits,
                cached_buffers=cached_buffers,
                buffer_pool=buffer_pool,
                use_fp32_acc=self.hidden_dim >= 1024,
            )

        # Add shared expert (always applied)
        shared_output = self.shared_expert(x)
        output = output + shared_output

        # Restore shape
        output = output.view(*batch_shape, self.hidden_dim)

        return output.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MoE routing.

        Uses fast fused Metal kernel when available, with graceful fallback
        to sequential processing on failure. Implements retry logic with
        exponential backoff for transient errors.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        # Hot path - minimal overhead, no per-call validation
        if self._use_fast_moe and not self._fast_moe_permanently_disabled and x.is_mps:
            # Fast bailout for backoff (rare after failures)
            if self._fast_moe_backoff_until == 0.0 or time.monotonic() >= self._fast_moe_backoff_until:
                try:
                    return self.forward_fast(x)
                except (RuntimeError, MemoryError) as e:
                    self._on_fast_moe_failure(e)
                except Exception as e:
                    self._on_fast_moe_failure(e, unexpected=True)

        return self._forward_slow(x)

    def _on_fast_moe_failure(self, error: Exception, unexpected: bool = False) -> None:
        """Handle fast MoE execution failure.

        Implements retry counting with exponential backoff. After max_retries
        failures, permanently disables fast path.

        Args:
            error: The exception that caused the failure.
            unexpected: If True, this was an unexpected error type.
        """
        self._fast_moe_failure_count += 1
        error_type = type(error).__name__

        # Log the failure with appropriate level
        if unexpected:
            logger.warning(
                "Unexpected fast MoE error (attempt %d/%d): %s: %s",
                self._fast_moe_failure_count,
                self._fast_moe_max_retries,
                error_type,
                error,
            )
        else:
            logger.debug(
                "Fast MoE failed (attempt %d/%d): %s: %s",
                self._fast_moe_failure_count,
                self._fast_moe_max_retries,
                error_type,
                error,
            )

        # Check if we should permanently disable
        if self._fast_moe_failure_count >= self._fast_moe_max_retries:
            self._fast_moe_permanently_disabled = True
            logger.warning(
                "Fast MoE permanently disabled after %d failures. "
                "Last error: %s: %s. Use _check_fast_path_health() to re-enable.",
                self._fast_moe_failure_count,
                error_type,
                error,
            )
        else:
            # Apply exponential backoff: 0.1s, 0.2s, 0.4s, ...
            backoff_seconds = 0.1 * self._fast_moe_backoff_multiplier
            self._fast_moe_backoff_until = time.monotonic() + backoff_seconds
            self._fast_moe_backoff_multiplier *= 2.0
            logger.debug(
                "Fast MoE backoff for %.2fs before next retry",
                backoff_seconds,
            )

    def _check_fast_path_health(self) -> bool:
        """Run a small test to verify fast MoE path is functional.

        Creates a small test input and runs both fast and slow paths,
        comparing outputs. If the fast path works and produces correct
        results, re-enables it.

        Returns:
            True if fast path is healthy and re-enabled, False otherwise.
        """
        if not torch.backends.mps.is_available():
            logger.info("MPS not available, fast path cannot be enabled")
            return False

        try:
            # Create small test input (2 tokens, hidden_dim)
            test_input = torch.randn(2, self.hidden_dim, dtype=torch.float16, device="mps")

            # Temporarily force slow path for reference
            old_use_fast = self._use_fast_moe
            old_permanently_disabled = self._fast_moe_permanently_disabled
            self._use_fast_moe = False
            self._fast_moe_permanently_disabled = False

            with torch.no_grad():
                slow_output = self._forward_slow(test_input.clone())

            # Now try fast path
            self._use_fast_moe = True
            self._fast_moe_permanently_disabled = False

            with torch.no_grad():
                fast_output = self.forward_fast(test_input.clone())

            # Restore original state temporarily for comparison
            self._use_fast_moe = old_use_fast
            self._fast_moe_permanently_disabled = old_permanently_disabled

            # Check outputs match (with tolerance for FP16)
            if not torch.allclose(fast_output, slow_output, rtol=0.1, atol=0.1):
                max_diff = (fast_output - slow_output).abs().max().item()
                logger.warning(
                    "Fast path produces different results (max diff: %.4f), keeping disabled",
                    max_diff,
                )
                return False

            # Fast path is healthy - re-enable
            self._use_fast_moe = True
            self._fast_moe_permanently_disabled = False
            self._fast_moe_failure_count = 0
            self._fast_moe_backoff_multiplier = 1.0
            self._fast_moe_backoff_until = 0.0

            logger.info("Fast MoE health check passed, re-enabled")
            return True

        except Exception as e:
            logger.warning("Fast path health check failed: %s: %s", type(e).__name__, e)
            return False

    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-optimized sequential forward pass.

        Batches by unique experts to reduce tensor allocations from O(slots × experts)
        to O(unique_experts). For top-8 routing with 64 experts, typically only ~8-16
        unique experts are active, reducing iterations from ~512 to ~12.

        Also lazily initializes fast path resources if not yet ready.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        # Lazy init: ensure fast path resources are ready for next call
        if self._cached_weight_buffers is None and self._eager_buffers:
            self._create_buffers_eagerly()
        if self._lib is None:
            self._get_lib()
        if self._buffer_pool is None:
            self._get_buffer_pool()

        # Convert input to router's dtype for routing computation
        x_router = x.to(self.router.weight.dtype)

        # Get router scores
        router_logits = self.router(x_router)  # [..., num_experts]

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float),
            k=self.num_experts_per_tok,
            dim=-1,
        )

        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Find unique experts across ALL slots (typically ~8-16 out of 64)
        unique_experts = selected_experts.unique().tolist()

        # Initialize output
        final_hidden_states = torch.zeros_like(x)

        # Process only active experts - each expert called exactly once
        for expert_id in unique_experts:
            # Find all (slot, position) pairs using this expert across all k slots
            expert_mask = selected_experts == expert_id  # [..., k]

            # Sum weights across all slots for this expert
            # torch.where keeps routing_weights where mask is True, else 0
            weights_for_expert = torch.where(
                expert_mask,
                routing_weights,
                torch.zeros_like(routing_weights),
            ).sum(dim=-1)  # [...] - summed weight per position

            # Apply expert once to all tokens, then weight the output
            expert_output = self.experts[expert_id](x)
            final_hidden_states += expert_output * weights_for_expert.unsqueeze(-1)

            # Explicit cleanup for MPS memory pressure
            del expert_output, weights_for_expert, expert_mask

        # Add shared expert (always applied)
        shared_output = self.shared_expert(x)
        final_hidden_states = final_hidden_states + shared_output

        # Sync MPS to release memory before returning
        if x.is_mps:
            torch.mps.synchronize()

        return final_hidden_states

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        config: TrellisModelConfig,
        layer_idx: int,
        router_weights: dict[str, torch.Tensor],
        device: str = "mps",
    ) -> TrellisMoEMLP:
        """Create TrellisMoEMLP from a TrellisModelLoader.

        Args:
            loader: TrellisModelLoader instance for the model.
            config: Model configuration.
            layer_idx: Layer index to load.
            router_weights: Router weights dictionary.
            device: Device to place modules on.

        Returns:
            TrellisMoEMLP module initialized with layer weights.
        """
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.mlp"

        # Get router weight to determine dtype
        router_weight = router_weights[f"{prefix}.gate.weight"]

        # Create router with same dtype as weights
        router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            device=device,
            dtype=router_weight.dtype,
        )
        router.weight.data = router_weight.to(device)

        # Create experts
        experts = []
        for expert_idx in range(config.num_experts):
            expert_prefix = f"{prefix}.experts.{expert_idx}"
            expert = TrellisDenseMLP(
                gate_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.gate_proj.weight"],
                    device=device,
                ),
                up_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.up_proj.weight"],
                    device=device,
                ),
                down_proj=TrellisLinear.from_trellis_weight(
                    layer_weights[f"{expert_prefix}.down_proj.weight"],
                    device=device,
                ),
            )
            experts.append(expert)

        # Create shared expert (GLM uses 'shared_experts' plural in weights)
        shared_expert = TrellisDenseMLP(
            gate_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.gate_proj.weight"],
                device=device,
            ),
            up_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.up_proj.weight"],
                device=device,
            ),
            down_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.shared_experts.down_proj.weight"],
                device=device,
            ),
        )

        return cls(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            num_experts_per_tok=config.num_experts_per_tok,
            eager_buffers=True,  # Memory-optimized: create Metal buffers from CPU
        )


class TrellisDecoderLayer(nn.Module):
    """Complete transformer decoder layer with trellis-quantized weights.

    Implements a GLM-style decoder layer with:
    - MLA attention (Multi-head Latent Attention)
    - RMSNorm pre-normalization
    - Dense or MoE MLP (depending on layer index)
    - Residual connections

    Attributes:
        self_attn: Attention module (to be implemented).
        mlp: Dense or MoE MLP module.
        input_layernorm: Pre-attention normalization.
        post_attention_layernorm: Post-attention normalization.
        config: Layer configuration.
    """

    def __init__(
        self,
        config: TrellisModelConfig,
        layer_idx: int,
        device: str = "mps",
    ):
        """Initialize TrellisDecoderLayer.

        Args:
            config: Model configuration.
            layer_idx: Layer index (0-indexed).
            device: Device to place modules on.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # MLA attention will be created in from_loader
        self.self_attn = None

        # MLP (dense or MoE)
        self.mlp = None  # Will be set in from_loader

        # Normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor [..., seq_len, hidden_size].
            attention_mask: Causal attention mask.
            position_ids: Position IDs for RoPE.
            kv_cache: KV cache for generation.

        Returns:
            Output tensor [..., seq_len, hidden_size].
        """
        # Pre-attention normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output = self.self_attn(
            hidden_states,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # Post-attention normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        mlp_output = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + mlp_output

        return hidden_states

    @classmethod
    def from_loader(
        cls,
        loader: TrellisModelLoader,
        config: TrellisModelConfig,
        layer_idx: int,
        router_weights: dict[str, torch.Tensor],
        base_weights: dict[str, torch.Tensor],
        device: str = "mps",
    ) -> TrellisDecoderLayer:
        """Create TrellisDecoderLayer from a TrellisModelLoader.

        Args:
            loader: TrellisModelLoader instance.
            config: Model configuration.
            layer_idx: Layer index.
            router_weights: Router weights for MoE layers.
            base_weights: Base model weights.
            device: Device to place modules on.

        Returns:
            TrellisDecoderLayer module initialized with layer weights.
        """
        layer = cls(config, layer_idx, device)

        # Load layer weights
        layer_weights = loader.load_layer(layer_idx)
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Load layernorm weights from base_weights.safetensors
        layernorm_weights = loader.load_layernorm_weights(layer_idx)

        # Create MLA attention config with GLM-4 dimensions
        mla_config = TrellisMLAConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 192),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(config, "v_head_dim", 256),
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

        # Get attention projections
        # GLM uses low-rank Q: q_a_proj + q_b_proj
        q_a_proj = None
        q_b_proj = None
        if mla_config.q_lora_rank:
            q_a_key = f"{prefix}.q_a_proj.weight"
            q_b_key = f"{prefix}.q_b_proj.weight"
            if q_a_key in layer_weights:
                q_a_proj = TrellisLinear.from_trellis_weight(
                    layer_weights[q_a_key],
                    device=device,
                )
            if q_b_key in layer_weights:
                q_b_proj = TrellisLinear.from_trellis_weight(
                    layer_weights[q_b_key],
                    device=device,
                )

        # GLM uses kv_a_proj_with_mqa (includes MQA heads)
        kv_a_key = f"{prefix}.kv_a_proj_with_mqa.weight"
        if kv_a_key not in layer_weights:
            kv_a_key = f"{prefix}.kv_a_proj.weight"  # Fallback
        kv_b_key = f"{prefix}.kv_b_proj.weight"
        o_key = f"{prefix}.o_proj.weight"

        kv_a_proj = TrellisLinear.from_trellis_weight(
            layer_weights[kv_a_key],
            device=device,
        )
        kv_b_proj = TrellisLinear.from_trellis_weight(
            layer_weights[kv_b_key],
            device=device,
        )
        o_proj = TrellisLinear.from_trellis_weight(
            layer_weights[o_key],
            device=device,
        )

        # Load MLA layernorms (q_a_layernorm and kv_a_layernorm)
        q_a_layernorm = None
        kv_a_layernorm = None
        if "self_attn.q_a_layernorm.weight" in layernorm_weights:
            q_a_ln_weight = layernorm_weights["self_attn.q_a_layernorm.weight"]
            q_a_layernorm = RMSNorm(q_a_ln_weight.shape[0], eps=config.rms_norm_eps)
            q_a_layernorm.weight.data = q_a_ln_weight.to(device)
        if "self_attn.kv_a_layernorm.weight" in layernorm_weights:
            kv_a_ln_weight = layernorm_weights["self_attn.kv_a_layernorm.weight"]
            kv_a_layernorm = RMSNorm(kv_a_ln_weight.shape[0], eps=config.rms_norm_eps)
            kv_a_layernorm.weight.data = kv_a_ln_weight.to(device)

        layer.self_attn = TrellisMLAttention(
            config=mla_config,
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            kv_b_proj=kv_b_proj,
            o_proj=o_proj,
            q_a_layernorm=q_a_layernorm,
            kv_a_layernorm=kv_a_layernorm,
        )

        # Load input/post-attention layernorms
        if "input_layernorm.weight" in layernorm_weights:
            layer.input_layernorm.weight.data = layernorm_weights["input_layernorm.weight"].to(
                device
            )
        if "post_attention_layernorm.weight" in layernorm_weights:
            layer.post_attention_layernorm.weight.data = layernorm_weights[
                "post_attention_layernorm.weight"
            ].to(device)

        # Create MLP (dense or MoE)
        if config.is_moe_layer(layer_idx):
            layer.mlp = TrellisMoEMLP.from_loader(loader, config, layer_idx, router_weights, device)
        else:
            layer.mlp = TrellisDenseMLP.from_loader(loader, layer_idx, device)

        return layer


class TrellisModel(nn.Module):
    """Complete trellis-quantized model for inference."""

    def __init__(self, config: TrellisModelConfig):
        super().__init__()
        self.config = config

        # Embedding (not quantized, from base model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Layers
        self.layers = nn.ModuleList()

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask
        if attention_mask is None:
            seq_len = input_ids.shape[1]
            attention_mask = self._make_causal_mask(seq_len, hidden_states.device)

        is_mps = hidden_states.is_mps
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
            # Sync MPS every 8 layers to bound peak memory
            if is_mps and (i + 1) % 8 == 0:
                torch.mps.synchronize()

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Clear transient allocations
        if is_mps:
            torch.mps.synchronize()

        return hidden_states

    def _make_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "mps",
        load_in_layers: bool = True,
    ) -> TrellisModel:
        """Load model from trellis-quantized checkpoint.

        Args:
            model_path: Path to quantized model directory
            device: Device to load model on
            load_in_layers: If True, load one layer at a time (memory efficient)
        """

        config = TrellisModelConfig.from_pretrained(model_path)
        model = cls(config)

        from .loader import TrellisModelLoader

        loader = TrellisModelLoader(model_path)

        # Load non-quantized weights (embedding, norms, lm_head)
        base_weights = cls._load_base_weights(model_path)
        model.embed_tokens.weight.data = base_weights["model.embed_tokens.weight"].to(device)
        model.norm.weight.data = base_weights["model.norm.weight"].to(device)

        # Load router weights if MoE
        router_weights = {}
        if config.num_experts > 1:
            router_weights = loader.load_router_weights()

        # Load layers
        for layer_idx in range(config.num_hidden_layers):
            layer = TrellisDecoderLayer.from_loader(
                loader, config, layer_idx, router_weights, base_weights, device
            )
            model.layers.append(layer)

            if load_in_layers:
                # Clear loader cache to save memory
                loader.clear_layer_cache(layer_idx)

        return model.to(device)

    @staticmethod
    def _load_base_weights(model_path: str) -> dict[str, torch.Tensor]:
        """Load non-quantized weights (embedding, norms, lm_head)."""
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(model_path)

        # Try loading from quantized model directory
        base_weights_path = path / "base_weights.safetensors"
        if base_weights_path.exists():
            return load_file(base_weights_path)

        # Fall back to HuggingFace
        raise FileNotFoundError(
            f"base_weights.safetensors not found in {model_path}. "
            "Run extract_base_weights.py first."
        )


class TrellisForCausalLM(nn.Module):
    """Trellis model with language modeling head for text generation.

    Wraps TrellisModel with an LM head projection for generating logits.
    Supports autoregressive generation with temperature, top-k, and top-p sampling.

    Attributes:
        model: The underlying TrellisModel.
        config: Model configuration.
        lm_head: Linear projection from hidden_size to vocab_size.
    """

    def __init__(self, config: TrellisModelConfig):
        """Initialize TrellisForCausalLM.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model = TrellisModel(config)

        # LM head (not quantized, tied to embedding or separate)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask [batch, seq_len].
            position_ids: Optional position IDs [batch, seq_len].
            kv_cache: Optional KV cache for generation.

        Returns:
            Logits tensor [batch, seq_len, vocab_size].
        """
        hidden_states = self.model(input_ids, attention_mask, position_ids, kv_cache)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache.

        Generates tokens autoregressively using the model with efficient
        KV caching for improved performance on long sequences.

        Args:
            input_ids: Initial token IDs [batch, seq_len].
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (1.0 = greedy, <1.0 = focused, >1.0 = random).
            top_k: Number of highest probability tokens to keep for top-k sampling.
            top_p: Cumulative probability threshold for nucleus (top-p) sampling.

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize MLA KV cache for efficient generation
        # MLA caches compressed representation (kv_lora_rank + qk_rope_head_dim)
        # instead of full K,V tensors, reducing cache size by ~8x
        kv_cache = TrellisKVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=seq_len + max_new_tokens,
            kv_lora_rank=self.config.kv_lora_rank,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            device=str(device),
        )

        # Track which sequences are finished (for batched generation)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initial forward pass to fill KV cache with prompt
        _ = self.forward(input_ids, kv_cache=kv_cache)

        # Get current sequence length from cache
        current_len = kv_cache.seq_len

        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Get logits for the last position only
            logits = self.forward(
                input_ids[:, -1:],
                kv_cache=kv_cache,
            )  # [batch, 1, vocab_size]
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Mark sequences as finished if EOS token is generated
            if hasattr(self.config, "eos_token_id") and self.config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == self.config.eos_token_id)
            else:
                # Default EOS token ID (commonly 2 for many models)
                finished = finished | (next_token.squeeze(-1) == 2)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if all sequences are finished
            if finished.all():
                break

        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str = "mps",
        optimize_memory: bool = True,
    ) -> TrellisForCausalLM:
        """Load a TrellisForCausalLM model from path.

        Loads the model configuration, base model weights, and LM head weights
        from the specified path. Supports both tied and separate LM heads.

        Args:
            model_path: Path to the model directory containing config.json
                and base_weights.safetensors.
            device: Device to load the model on (default: "mps").

        Returns:
            Loaded TrellisForCausalLM instance.
        """
        config = TrellisModelConfig.from_pretrained(model_path)
        model = cls(config)

        # Load base model
        model.model = TrellisModel.from_pretrained(model_path, device)

        # Load lm_head weight from base_weights.safetensors (may be tied to embed_tokens)
        base_weights = TrellisModel._load_base_weights(model_path)
        if "lm_head.weight" in base_weights:
            model.lm_head.weight.data = base_weights["lm_head.weight"].to(device)
        else:
            # Tied embeddings - share weight with embed_tokens
            model.lm_head.weight = model.model.embed_tokens.weight

        # Optimize memory if requested
        if optimize_memory:
            model.optimize_memory(verbose=False)

        return model.to(device)

    def optimize_memory(self, verbose: bool = False) -> dict:
        """Optimize memory by creating Metal buffers and freeing tensors.

        Call after loading the model to minimize memory footprint.

        Args:
            verbose: If True, print memory stats during optimization.

        Returns:
            Dict with memory stats before/after optimization.
        """
        import gc

        stats = {"layers_optimized": 0}

        if verbose:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            stats["before_rss_gb"] = process.memory_info().rss / 1e9

        # Optimize each MoE layer
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer.mlp, "_cached_weight_buffers"):
                # Force eager buffer creation if not already done
                if layer.mlp._cached_weight_buffers is None:
                    layer.mlp._get_cached_buffers()
                    stats["layers_optimized"] += 1

                    if verbose:
                        print(f"  Layer {i}: created Metal buffers")

        # Force garbage collection
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        gc.collect()  # Second pass to catch released MPS memory

        if verbose:
            stats["after_rss_gb"] = process.memory_info().rss / 1e9
            stats["freed_gb"] = stats["before_rss_gb"] - stats["after_rss_gb"]
            print(f"  Memory freed: {stats['freed_gb']:.2f} GB")

        return stats
