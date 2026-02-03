"""Trellis MoE: Mixture-of-Experts with trellis-quantized weights.

This module provides expert implementations using trellis-quantized weights
for efficient inference on Apple Silicon via Metal Performance Shaders.
"""

from __future__ import annotations

import logging
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._compat import HAS_MPS
from ..metal_dispatch import HAS_METAL, MetalKernelLibrary, get_default_library
from .config import TrellisModelConfig
from .layer import TrellisDenseMLP
from .linear import TrellisLinear
from .loader import TrellisWeight
from .moe_dispatch import (
    CachedWeightBuffers,
    MoEBufferPool,
    create_cached_weight_buffers,
    dispatch_moe_router_fused,
    dispatch_moe_trellis_swiglu,
)

__all__ = [
    "TrellisMoELayer",
    "TrellisExpert",
    "ExpertCache",
    "get_moe_dispatch_stats",
    "reset_moe_dispatch_stats",
]

# Debug mode: use only expert 0 for all tokens to isolate routing vs execution issues
DEBUG_MOE_SIMPLE = os.getenv("DEBUG_MOE_SIMPLE", "0") == "1"

# MoE dispatch logging - enable with MOE_DEBUG=1 environment variable
logger = logging.getLogger(__name__)
_MOE_DEBUG = os.getenv("MOE_DEBUG", "0") == "1"

# Statistics tracking for MoE dispatch
_moe_stats = {
    "metal_router_calls": 0,
    "metal_router_success": 0,
    "cpu_router_fallback": 0,
    "total_tokens_processed": 0,
    "total_experts_activated": 0,
}


def get_moe_dispatch_stats() -> dict[str, int]:
    """Get MoE dispatch statistics.

    Returns:
        Dictionary with dispatch statistics:
        - metal_router_calls: Total router dispatch attempts
        - metal_router_success: Successful Metal kernel dispatches
        - cpu_router_fallback: CPU fallback count
        - total_tokens_processed: Cumulative tokens processed
        - total_experts_activated: Cumulative experts activated
    """
    return _moe_stats.copy()


def reset_moe_dispatch_stats() -> None:
    """Reset MoE dispatch statistics."""
    global _moe_stats
    _moe_stats = {
        "metal_router_calls": 0,
        "metal_router_success": 0,
        "cpu_router_fallback": 0,
        "total_tokens_processed": 0,
        "total_experts_activated": 0,
    }


def _log_moe_dispatch(
    stage: str,
    *,
    num_experts: int | None = None,
    num_tokens: int | None = None,
    topk: int | None = None,
    kernel: str | None = None,
    success: bool = True,
    error: Exception | None = None,
) -> None:
    """Log MoE dispatch events for debugging.

    Enable with MOE_DEBUG=1 environment variable or DEBUG log level.

    Args:
        stage: Dispatch stage (e.g., "router", "expert_gemm", "fallback")
        num_experts: Number of experts in the layer
        num_tokens: Number of tokens being processed
        topk: Top-k experts selected per token
        kernel: Name of the kernel being used
        success: Whether the operation succeeded
        error: Exception if operation failed
    """
    if not (_MOE_DEBUG or logger.isEnabledFor(logging.DEBUG)):
        return

    parts = [f"MoE {stage}:"]
    if num_experts is not None:
        parts.append(f"experts={num_experts}")
    if num_tokens is not None:
        parts.append(f"tokens={num_tokens}")
    if topk is not None:
        parts.append(f"topk={topk}")
    if kernel is not None:
        parts.append(f"kernel={kernel}")

    msg = " ".join(parts)

    if not success and error is not None:
        logger.warning(f"{msg} FAILED: {error}")
    elif not success:
        logger.warning(f"{msg} - falling back to CPU")
    else:
        logger.info(msg)


class ExpertCache:
    """Cache for frequently-used expert weights.

    Tracks expert selection frequency over N samples and maintains a fast
    memory cache of the top-K most used experts. Prefetching is triggered
    based on router predictions.

    Rarely-used experts (selected <1% of the time) are kept on CPU and streamed
    to GPU asynchronously when needed.

    Speculative loading: After routing, predict next layer's experts and prefetch.
    If prediction correct, no wait; if wrong, fallback to normal loading.
    """

    def __init__(
        self,
        num_experts: int,
        cache_size: int = 8,
        window_size: int = 128,
        device: str = "mps",
        stream_threshold: float = 0.01,
        enable_speculation: bool = True,
        speculation_depth: int = 1,
    ) -> None:
        """Initialize expert cache.

        Args:
            num_experts: Total number of experts in the layer.
            cache_size: Number of experts to keep in fast memory cache.
            window_size: Number of samples to track for frequency statistics.
            device: Device to place cached weights on.
            stream_threshold: Frequency threshold below which experts stream from CPU.
            enable_speculation: Enable speculative prefetching of next layer's experts.
            speculation_depth: Number of future layers to predict (default: 1).
        """
        self.num_experts = num_experts
        self.cache_size = cache_size
        self.window_size = window_size
        self.device = device
        self.stream_threshold = stream_threshold
        self.enable_speculation = enable_speculation
        self.speculation_depth = speculation_depth

        self.expert_frequency: torch.Tensor = torch.zeros(
            num_experts, device=device)
        self.selection_history: deque[torch.Tensor] = deque(maxlen=window_size)
        self.cached_experts: set[int] = set()
        self.expert_weights: dict[int, nn.Module] = {}
        self.cpu_experts: dict[int, nn.Module] = {}
        self.streaming_experts: dict[int, torch.cuda.Stream | None] = {}

        # Speculative prefetching state
        # Experts currently prefetching
        self.speculative_experts: set[int] = set()
        self.speculation_hits: int = 0  # Correct predictions
        self.speculation_misses: int = 0  # Wrong predictions
        self.last_expert_ids: torch.Tensor | None = None  # Last selected experts

    def record_selection(self, expert_ids: torch.Tensor) -> None:
        """Record expert selections for this batch.

        Args:
            expert_ids: Selected expert indices [num_assignments].
        """
        flat_expert_ids = expert_ids.reshape(-1)

        # Add new selections
        for expert_id in flat_expert_ids.cpu().numpy().tolist():
            self.expert_frequency[expert_id] += 1

        # Remove old selections if history is full
        if len(self.selection_history) == self.selection_history.maxlen:
            oldest_selections = self.selection_history[0]
            for expert_id in oldest_selections.cpu().numpy().tolist():
                self.expert_frequency[expert_id] -= 1

        self.selection_history.append(flat_expert_ids.clone())

    def get_top_experts(self) -> list[int]:
        """Get top-K most frequently used experts.

        Returns:
            List of expert indices sorted by frequency (descending).
        """
        _, top_indices = torch.topk(self.expert_frequency, self.cache_size)
        return top_indices.cpu().numpy().tolist()

    def should_prefetch(self, router_logits: torch.Tensor, threshold: float = 0.5) -> list[int]:
        """Predict which experts will be selected and prefetch them.

        Args:
            router_logits: Router output [tokens, num_experts].
            threshold: Probability threshold for prediction.

        Returns:
            List of expert indices to prefetch.
        """
        routing_probs = F.softmax(router_logits, dim=-1)

        top_predictions = []

        for token_idx in range(routing_probs.shape[0]):
            token_probs = routing_probs[token_idx]
            experts_above_threshold = torch.where(token_probs > threshold)[0]
            top_predictions.extend(experts_above_threshold.cpu().numpy().tolist())

        unique_predictions = torch.unique(
            torch.tensor(top_predictions)).cpu().numpy().tolist()

        return [e for e in unique_predictions if e not in self.cached_experts]

    def update_cache(
        self, experts: nn.ModuleList, prefetch_indices: list[int] | None = None
    ) -> list[int]:
        """Update cache with top-K most used experts and prefetched experts.

        Moves rarely-used experts to CPU for streaming on-demand.

        Args:
            experts: List of all expert modules.
            prefetch_indices: List of expert indices to prefetch.

        Returns:
            List of expert indices newly added to cache.
        """
        top_experts = self.get_top_experts()
        target_experts = set(top_experts)
        if prefetch_indices:
            target_experts.update(prefetch_indices)

        # Identify rare experts to move to CPU
        rare_experts = set(self.get_rare_experts())

        # Don't move frequently-accessed experts to CPU
        rare_experts = rare_experts - target_experts

        # Evict experts not in target set
        current_cached = set(self.cached_experts)
        to_evict = current_cached - target_experts

        for expert_idx in to_evict:
            # Move to CPU if rarely used
            if expert_idx in rare_experts:
                self.move_to_cpu(expert_idx, experts[expert_idx])
            else:
                # Just evict from cache
                self.cached_experts.remove(expert_idx)
                if expert_idx in self.expert_weights:
                    del self.expert_weights[expert_idx]

        newly_cached = []

        for expert_idx in target_experts:
            if expert_idx not in self.cached_experts:
                # Stream from CPU if available
                if expert_idx in self.cpu_experts:
                    self.stream_to_gpu(expert_idx)
                else:
                    self.cached_experts.add(expert_idx)
                    self.expert_weights[expert_idx] = experts[expert_idx]
                newly_cached.append(expert_idx)

        return newly_cached

    def is_cached(self, expert_idx: int) -> bool:
        """Check if expert is in cache."""
        return expert_idx in self.cached_experts

    def get_cached_experts(self) -> list[int]:
        """Get all cached expert indices."""
        return sorted(self.cached_experts)

    def get_frequency(self) -> torch.Tensor:
        """Get current expert selection frequencies."""
        return self.expert_frequency.clone()

    def get_frequency_ratio(self) -> torch.Tensor:
        """Get expert selection frequency as ratio of total selections."""
        total_selections = self.expert_frequency.sum()
        if total_selections == 0:
            return torch.zeros_like(self.expert_frequency)
        return self.expert_frequency / total_selections

    def get_rare_experts(self) -> list[int]:
        """Get experts with selection frequency below stream threshold.

        Returns:
            List of expert indices with frequency < stream_threshold.
        """
        freq_ratio = self.get_frequency_ratio()
        rare_mask = freq_ratio < self.stream_threshold
        return torch.where(rare_mask)[0].cpu().numpy().tolist()

    def move_to_cpu(self, expert_idx: int, expert_module: nn.Module) -> None:
        """Move expert weights to CPU for streaming.

        Args:
            expert_idx: Expert index to move to CPU.
            expert_module: Expert module to move.
        """
        if expert_idx not in self.cpu_experts:
            # Move expert to CPU
            cpu_expert = expert_module.cpu()

            # Pin memory for faster transfer
            # Note: On MPS/Apple Silicon, this might fail or be unnecessary due to Unified Memory
            try:
                for param in cpu_expert.parameters():
                    param.pin_memory()
                for buf in cpu_expert.buffers():
                    buf.pin_memory()
            except RuntimeError:
                # Ignore pin_memory failures on MPS
                pass

            self.cpu_experts[expert_idx] = cpu_expert

            # Remove from GPU cache
            if expert_idx in self.cached_experts:
                self.cached_experts.remove(expert_idx)
            if expert_idx in self.expert_weights:
                del self.expert_weights[expert_idx]

    def stream_to_gpu(self, expert_idx: int, non_blocking: bool = False) -> nn.Module:
        """Stream expert from CPU to GPU asynchronously.

        Args:
            expert_idx: Expert index to stream.
            non_blocking: Whether to perform async transfer.

        Returns:
            Expert module on GPU (may be still transferring).
        """
        if expert_idx in self.expert_weights:
            return self.expert_weights[expert_idx]

        if expert_idx not in self.cpu_experts:
            raise ValueError(f"Expert {expert_idx} not in CPU cache")

        # Create stream for async transfer (MPS doesn't support streams, use sync)
        cpu_expert = self.cpu_experts[expert_idx]

        # For MPS, just do synchronous transfer
        # For CUDA, this would be: with torch.cuda.stream(stream)
        gpu_expert = cpu_expert.to(self.device, non_blocking=non_blocking)

        # Cache on GPU temporarily
        self.expert_weights[expert_idx] = gpu_expert
        self.cached_experts.add(expert_idx)

        return gpu_expert

    def predict_next_experts(self, current_expert_ids: torch.Tensor, top_k: int = 4) -> list[int]:
        """Predict which experts will be selected in the next layer.

        Uses simple heuristic: experts selected in current layer are likely
        to be selected in next layer (spatial/temporal locality).

        Args:
            current_expert_ids: Currently selected expert IDs [tokens, k].
            top_k: Number of experts to predict (default: 4).

        Returns:
            List of predicted expert indices.
        """
        if not self.enable_speculation:
            return []

        # Count frequency of experts in current selection
        flat_ids = current_expert_ids.reshape(-1)
        unique_ids, counts = torch.unique(flat_ids, return_counts=True)

        # Predict top-k most frequent experts from current selection
        if len(unique_ids) <= top_k:
            predictions = unique_ids.cpu().numpy().tolist()
        else:
            _, top_indices = torch.topk(counts, min(top_k, len(counts)))
            predictions = unique_ids[top_indices].cpu().numpy().tolist()

        return predictions

    def start_speculative_prefetch(
        self,
        predicted_experts: list[int],
        next_layer_experts: nn.ModuleList,
        next_layer_cache: ExpertCache | None = None,
    ) -> None:
        """Start prefetching predicted experts for next layer.

        Args:
            predicted_experts: List of expert indices to prefetch.
            next_layer_experts: Expert modules from next layer.
            next_layer_cache: ExpertCache instance for next layer.
        """
        if not self.enable_speculation or next_layer_cache is None:
            return

        # Filter to experts not already cached
        to_prefetch = [
            e
            for e in predicted_experts
            if e not in next_layer_cache.cached_experts
            and e not in next_layer_cache.speculative_experts
        ]

        if not to_prefetch:
            return

        # Mark as speculative
        next_layer_cache.speculative_experts.update(to_prefetch)

        # Trigger async prefetch
        for expert_idx in to_prefetch:
            if expert_idx in next_layer_cache.cpu_experts:
                # Stream from CPU asynchronously
                next_layer_cache.stream_to_gpu(expert_idx, non_blocking=True)
            else:
                # Already resident, just add to cache
                next_layer_cache.cached_experts.add(expert_idx)
                next_layer_cache.expert_weights[expert_idx] = next_layer_experts[expert_idx]

    def validate_speculation(self, actual_expert_ids: torch.Tensor) -> None:
        """Check if speculative prefetch was correct.

        Args:
            actual_expert_ids: Actually selected expert IDs [tokens, k].
        """
        if not self.speculative_experts:
            return

        actual_set = set(actual_expert_ids.reshape(-1).cpu().numpy().tolist())
        speculative_set = self.speculative_experts

        # Count hits (correctly predicted) and misses (not used)
        hits = speculative_set & actual_set
        misses = speculative_set - actual_set

        self.speculation_hits += len(hits)
        self.speculation_misses += len(misses)

        # Clear speculative state
        self.speculative_experts.clear()

    def reset(self) -> None:
        """Reset cache statistics."""
        self.expert_frequency.zero_()
        self.selection_history.clear()
        self.cached_experts.clear()
        self.expert_weights.clear()
        self.cpu_experts.clear()
        self.streaming_experts.clear()
        self.speculative_experts.clear()
        self.speculation_hits = 0
        self.speculation_misses = 0
        self.last_expert_ids = None


def _fused_router_topk(
    x: torch.Tensor,
    W_router: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused router computation + top-k selection.

    Uses partial top-k (O(N log k)) instead of full sort (O(N log N)).
    For 64 experts with top-8 selection: ~2x faster.

    Args:
        x: Input tensor [tokens, hidden].
        W_router: Router weights [num_experts, hidden].
        k: Number of top experts to select.

    Returns:
        expert_ids: Top-k expert indices [tokens, k].
        expert_weights: Normalized weights for top-k experts [tokens, k].
    """
    num_tokens = x.shape[0]
    num_experts = W_router.shape[0]

    # Update statistics
    _moe_stats["metal_router_calls"] += 1
    _moe_stats["total_tokens_processed"] += num_tokens

    # Try using fused Metal kernel if available
    if HAS_METAL and x.is_mps:
        _log_moe_dispatch(
            "router_attempt",
            num_experts=num_experts,
            num_tokens=num_tokens,
            topk=k,
            kernel="moe_router_fused",
        )
        try:
            lib = get_default_library()
            # Dispatch fused router
            expert_ids, expert_weights = dispatch_moe_router_fused(
                lib, x, W_router, num_experts=num_experts, top_k=k
            )

            # Sanity check: if kernel returns all zeros, it failed silently
            # (expert 0 with weight 0 is theoretically possible but unlikely for all tokens)
            if expert_weights.abs().sum() == 0:
                raise RuntimeError("Fused router kernel returned all zeros")

            _moe_stats["metal_router_success"] += 1
            _log_moe_dispatch(
                "router_success",
                num_experts=num_experts,
                num_tokens=num_tokens,
                topk=k,
                kernel="moe_router_fused (Metal)",
                success=True,
            )
            return expert_ids.long(), expert_weights.float()
        except Exception as e:
            # Fallback to PyTorch implementation
            _moe_stats["cpu_router_fallback"] += 1
            _log_moe_dispatch(
                "router_fallback",
                num_experts=num_experts,
                num_tokens=num_tokens,
                topk=k,
                kernel="PyTorch CPU",
                success=False,
                error=e,
            )
            import warnings

            warnings.warn(
                f"Metal MoE router dispatch failed, falling back to CPU: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            pass
    else:
        # Metal not available or tensor not on MPS
        _moe_stats["cpu_router_fallback"] += 1
        reason = "HAS_METAL=False" if not HAS_METAL else "tensor not on MPS"
        _log_moe_dispatch(
            "router_cpu",
            num_experts=num_experts,
            num_tokens=num_tokens,
            topk=k,
            kernel=f"PyTorch ({reason})",
        )

    # 1. Compute logits
    router_logits = x @ W_router.T

    # 2. Top-k selection using partial sort for efficiency
    # For typical MoE configs (64 experts, top-8), use heap-based selection

    if num_experts >= 64 and k <= num_experts // 8:
        # Use heap-based partial selection (O(N log k)) for large expert counts
        # torch.topk with sorted=False uses a binary heap internally
        # This is ~2x faster than full sort for 64 experts with top-8
        top_logits, expert_ids = torch.topk(
            router_logits, k, dim=-1, largest=True, sorted=False)
        # Note: No need to sort - softmax only cares about relative magnitudes
        # and the weighted sum is order-independent
    else:
        # Standard topk for smaller configurations (sorted=True is fine here)
        top_logits, expert_ids = torch.topk(router_logits, k, dim=-1)

    # 3. Compute weights for top-k experts
    expert_weights = F.softmax(top_logits, dim=-1)

    return expert_ids, expert_weights


class TrellisMoELayer(nn.Module):
    """Mixture-of-Experts layer with trellis-quantized expert weights.

    Implements sparse MoE where each token is routed to top-k experts.
    Expert weights are stored in trellis-quantized format for memory efficiency.

    Forward pass:
        1. Router computes expert logits for each token
        2. Top-k experts are selected per token
        3. Selected experts are executed
        4. Outputs are combined weighted by routing probabilities
    """

    def __init__(
        self,
        config: TrellisModelConfig,
        layer_weights: dict[str, dict],
        router_weight: torch.Tensor,
        layer_idx: int,
        device: str = "mps",
        enable_cache: bool = True,
        cache_size: int = 8,
        cache_window_size: int = 128,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.0,
    ) -> None:
        """Initialize TrellisMoELayer.

        Args:
            config: MoE configuration.
            layer_weights: Dictionary of expert weights. Keys should be like
                "experts.{i}.gate_proj", "experts.{i}.up_proj", "experts.{i}.down_proj"
                or "experts.{i}.gate_up_proj" (fused).
            router_weight: Router weight tensor [num_experts, hidden_size].
            layer_idx: Layer index for weight key prefixes.
            device: Device to place weights on.
            enable_cache: Enable expert caching for fast memory.
            cache_size: Number of experts to cache (default: 8).
            cache_window_size: Number of samples to track for frequency (default: 128).
            capacity_factor: Max tokens per expert = capacity_factor * (batch_size / num_experts).
                Default 1.25 allows 25% imbalance. Set to float('inf') to disable.
            aux_loss_weight: Weight for auxiliary balance loss (default: 0.0 = disabled).
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.enable_cache = enable_cache
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

        # Router: linear projection to expert logits
        self.router = nn.Linear(
            config.hidden_size, config.num_experts, bias=False, device=device)
        self.router.weight.data = router_weight.to(
            device=device, dtype=torch.float32)

        # Build experts from weights
        self.experts = nn.ModuleList()
        for i in range(config.num_experts):
            expert = self._build_expert(layer_weights, i, device)
            self.experts.append(expert)

        # Packed expert weights for batched dispatch
        self.expert_weights_packed: torch.Tensor | None = None
        self.expert_qweights_packed: torch.Tensor | None = None

        # Pack expert weights after loading
        self._pack_expert_weights()

        # Initialize expert cache
        self.expert_cache: ExpertCache | None = None
        if self.enable_cache and config.num_experts > 1:
            self.expert_cache = ExpertCache(
                num_experts=config.num_experts,
                cache_size=min(cache_size, config.num_experts),
                window_size=cache_window_size,
                device=device,
                enable_speculation=True,
            )

        # Buffer for balance loss
        self.register_buffer("aux_loss", torch.tensor(0.0, device=device))

        # Reference to next layer for speculation (set externally)
        self.next_layer: TrellisMoELayer | None = None

        # Metal resources for batched dispatch (lazily initialized)
        self._lib: MetalKernelLibrary | None = None
        self._cached_weight_buffers: CachedWeightBuffers | None = None
        self._buffer_pool: MoEBufferPool | None = None
        self._use_batched_dispatch: bool = False  # Set to True after packing

        # Pack expert weights for efficient batched dispatch
        self._pack_expert_weights_for_batched_dispatch()

    def _get_lib(self) -> MetalKernelLibrary:
        """Get or create Metal kernel library."""
        if self._lib is None:
            self._lib = MetalKernelLibrary.from_source_dir()
        return self._lib

    def _get_buffer_pool(self) -> MoEBufferPool:
        """Get or create MoE buffer pool for dynamic input/output buffers."""
        if self._buffer_pool is None:
            lib = self._get_lib()
            self._buffer_pool = MoEBufferPool(
                device=lib.device,
                hidden_dim=self.config.hidden_size,
                max_batch=128,  # Support up to 128 tokens
                top_k_values=(self.config.num_experts_per_tok,),
            )
        return self._buffer_pool

    def _get_cached_buffers(self) -> CachedWeightBuffers | None:
        """Get or create cached Metal buffers for stacked expert weights.

        Returns None if batched dispatch is not available (e.g., experts not
        using TrellisLinear, or Metal not available).
        """
        if not self._use_batched_dispatch:
            return None

        if self._cached_weight_buffers is not None:
            return self._cached_weight_buffers

        if not HAS_METAL or not HAS_MPS:
            return None

        lib = self._get_lib()

        # Create cached buffers from stacked tensors
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
            grid=self.grid,
        )

        return self._cached_weight_buffers

    def _pack_expert_weights_for_batched_dispatch(self) -> None:
        """Pack all expert weights into stacked tensors for batched Metal dispatch.

        This stacks gate, up, and down projection weights from all experts into
        contiguous tensors with shape [num_experts, ...]. The packed format
        matches what dispatch_moe_trellis_swiglu expects.

        Sets self._use_batched_dispatch = True if packing succeeds.
        """
        # Check if experts use TrellisLinear (required for batched dispatch)
        if not self.experts:
            return

        first_expert = self.experts[0]

        # Determine expert type and check for TrellisLinear
        if isinstance(first_expert, (TrellisExpert, TrellisDenseMLP)):
            # Separate gate/up/down projections
            if not isinstance(first_expert.gate_proj, TrellisLinear):
                return
            self._pack_trellis_expert_weights()
        elif isinstance(first_expert, _FusedTrellisExpert):
            # Fused gate_up + down - not yet supported for batched dispatch
            # Would need a different kernel that handles fused projections
            return
        else:
            # Unknown expert type
            return

    def _pack_trellis_expert_weights(self) -> None:
        """Pack TrellisExpert weights into stacked buffers."""
        gate_indices_list = []
        gate_scales_list = []
        gate_su_list = []
        gate_sv_list = []
        up_indices_list = []
        up_scales_list = []
        up_su_list = []
        up_sv_list = []
        down_indices_list = []
        down_scales_list = []
        down_su_list = []
        down_sv_list = []

        for expert in self.experts:
            if not isinstance(expert, (TrellisExpert, TrellisDenseMLP)):
                # Inconsistent expert types - bail out
                return

            # Gate projection
            gate_indices_list.append(expert.gate_proj.packed_indices)
            gate_scales_list.append(expert.gate_proj.scales)
            gate_su_list.append(expert.gate_proj.su)
            gate_sv_list.append(expert.gate_proj.sv)

            # Up projection
            up_indices_list.append(expert.up_proj.packed_indices)
            up_scales_list.append(expert.up_proj.scales)
            up_su_list.append(expert.up_proj.su)
            up_sv_list.append(expert.up_proj.sv)

            # Down projection
            down_indices_list.append(expert.down_proj.packed_indices)
            down_scales_list.append(expert.down_proj.scales)
            down_su_list.append(expert.down_proj.su)
            down_sv_list.append(expert.down_proj.sv)

        # Stack all tensors along expert dimension
        # Transpose packed indices from [tiles_out, tiles_in, packed] to
        # [tiles_in, tiles_out, packed] for the Metal kernel
        self.register_buffer(
            "gate_weights_stacked",
            torch.stack(gate_indices_list, dim=0).permute(
                0, 2, 1, 3).contiguous(),
        )
        self.register_buffer(
            "gate_scales_stacked",
            torch.stack(gate_scales_list, dim=0).half().contiguous(),
        )
        self.register_buffer(
            "gate_su_stacked",
            torch.stack(gate_su_list, dim=0).half().contiguous(),
        )
        self.register_buffer(
            "gate_sv_stacked",
            torch.stack(gate_sv_list, dim=0).half().contiguous(),
        )

        self.register_buffer(
            "up_weights_stacked",
            torch.stack(up_indices_list, dim=0).permute(
                0, 2, 1, 3).contiguous(),
        )
        self.register_buffer(
            "up_scales_stacked",
            torch.stack(up_scales_list, dim=0).half().contiguous(),
        )
        self.register_buffer(
            "up_su_stacked",
            torch.stack(up_su_list, dim=0).half().contiguous(),
        )
        self.register_buffer(
            "up_sv_stacked",
            torch.stack(up_sv_list, dim=0).half().contiguous(),
        )

        self.register_buffer(
            "down_weights_stacked",
            torch.stack(down_indices_list, dim=0).permute(
                0, 2, 1, 3).contiguous(),
        )
        self.register_buffer(
            "down_scales_stacked",
            torch.stack(down_scales_list, dim=0).half().contiguous(),
        )
        self.register_buffer(
            "down_su_stacked",
            torch.stack(down_su_list, dim=0).half().contiguous(),
        )
        self.register_buffer(
            "down_sv_stacked",
            torch.stack(down_sv_list, dim=0).half().contiguous(),
        )

        # Get codebook grid from first expert (shared across all)
        first_expert = self.experts[0]
        self.register_buffer(
            "grid", first_expert.gate_proj.grid.half().contiguous())
        self.bits = first_expert.gate_proj.bits

        # Mark batched dispatch as available
        self._use_batched_dispatch = True

    def _build_expert(
        self, layer_weights: dict[str, dict], expert_idx: int, device: str
    ) -> nn.Module:
        """Build a single expert from weight dict.

        Handles both fused (gate_up_proj) and separate (gate_proj, up_proj) formats.
        """
        # Check for fused vs separate weight format
        fused_key = f"experts.{expert_idx}.gate_up_proj"
        gate_key = f"experts.{expert_idx}.gate_proj"
        up_key = f"experts.{expert_idx}.up_proj"
        down_key = f"experts.{expert_idx}.down_proj"

        if fused_key in layer_weights:
            # Fused gate_up format - use FusedExpert
            return _FusedTrellisExpert(
                gate_up_weight=self._dict_to_linear(
                    layer_weights[fused_key], device),
                down_weight=self._dict_to_linear(
                    layer_weights[down_key], device),
                config=self.config,
            )
        elif gate_key in layer_weights:
            # Separate gate/up format
            return TrellisExpert(
                gate_proj=self._dict_to_linear(
                    layer_weights[gate_key], device),
                up_proj=self._dict_to_linear(layer_weights[up_key], device),
                down_proj=self._dict_to_linear(
                    layer_weights[down_key], device),
            )
        else:
            # Fallback: create dummy expert for testing
            return _DummyExpert(self.config.hidden_size, self.config.intermediate_size, device)

    def _dict_to_linear(self, weight_dict: dict, device: str) -> TrellisLinear:
        """Convert weight dict to TrellisLinear module."""
        # Create TrellisWeight-like object from dict
        weight = _DictWeight(weight_dict)
        return TrellisLinear.from_trellis_weight(weight, device=device)

    def _apply_capacity_limit(
        self,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply expert capacity limits to prevent bottlenecks.

        Tokens exceeding an expert's capacity are dropped (weights set to 0).
        Capacity = capacity_factor * (num_tokens / num_experts).

        Args:
            expert_ids: Expert assignments [num_tokens, num_experts_per_tok].
            expert_weights: Routing weights [num_tokens, num_experts_per_tok].
            num_tokens: Total number of tokens in batch.

        Returns:
            (expert_ids, expert_weights): Capacity-limited assignments and weights.
        """
        # Compute capacity per expert
        capacity = int(self.capacity_factor *
                       num_tokens / self.config.num_experts)

        if capacity <= 0:
            capacity = 1

        # Track token counts per expert
        expert_token_counts = torch.zeros(
            self.config.num_experts, dtype=torch.long, device=expert_ids.device
        )

        # Create mask for tokens exceeding capacity
        keep_mask = torch.ones_like(expert_ids, dtype=torch.bool)

        # Process each slot (expert choice) for all tokens
        for slot_idx in range(self.config.num_experts_per_tok):
            slot_expert_ids = expert_ids[:, slot_idx]

            # For each token, check if its chosen expert is at capacity
            for token_idx in range(num_tokens):
                expert_id = int(slot_expert_ids[token_idx].cpu().numpy())

                if expert_token_counts[expert_id] >= capacity:
                    # Drop this assignment - expert at capacity
                    keep_mask[token_idx, slot_idx] = False
                else:
                    # Accept this assignment
                    expert_token_counts[expert_id] += 1

        # Zero out weights for dropped assignments
        expert_weights = expert_weights * keep_mask.float()

        # Renormalize weights for each token (to maintain output scale)
        weight_sum = expert_weights.sum(dim=-1, keepdim=True)
        # Avoid division by zero - if all experts dropped, leave weights as 0
        expert_weights = torch.where(
            weight_sum > 0,
            expert_weights / weight_sum,
            expert_weights,
        )

        return expert_ids, expert_weights

    def _process_expert_tokens(
        self,
        expert_idx: int,
        start: int,
        end: int,
        sorted_token_indices: torch.Tensor,
        sorted_slot_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        x: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Process tokens assigned to a specific expert."""
        # Get all tokens assigned to this expert
        token_indices = sorted_token_indices[start:end]
        slot_indices = sorted_slot_indices[start:end]
        expert_tokens = x[token_indices]

        # Get routing weights for these assignments
        expert_assign_weights = expert_weights[token_indices, slot_indices]

        # Load expert weights once, process all tokens
        # Note: Transfer should have been triggered already if needed
        if self.expert_cache is not None and expert_idx in self.expert_cache.expert_weights:
            expert_module = self.expert_cache.expert_weights[expert_idx]
        elif self.expert_cache is not None and expert_idx in self.expert_cache.cpu_experts:
            # Fallback if not triggered (should not happen with new forward logic)
            expert_module = self.expert_cache.stream_to_gpu(expert_idx)
        else:
            expert_module = self.experts[expert_idx]

        expert_out = expert_module(expert_tokens)

        # Accumulate weighted output
        # Ensure weights match output dtype (e.g. fp16)
        weighted_out = expert_out * \
            expert_assign_weights.to(dtype=output.dtype).unsqueeze(-1)
        output.index_add_(0, token_indices, weighted_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer with batched Metal dispatch.

        Uses efficient batched Metal kernel when available (all experts processed
        in a single kernel launch), falling back to Python-level iteration for
        unsupported expert types.

        Args:
            x: Input tensor of shape [batch, seq, hidden] or [tokens, hidden].

        Returns:
            Output tensor with same shape as input.
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, hidden = x.shape
            x = x.view(-1, hidden)  # [batch*seq, hidden]
        else:
            batch, seq_len = None, None

        num_tokens, hidden = x.shape

        # DEBUG MODE: Use only expert 0 for all tokens
        # This isolates whether slowness is from routing/dispatch vs expert execution
        if DEBUG_MOE_SIMPLE:
            print(
                f"DEBUG: Using simplified MoE (expert 0 only) for layer {self.layer_idx}")
            expert_out = self.experts[0](x)
            if batch is not None:
                expert_out = expert_out.view(batch, seq_len, hidden)
            return expert_out.to(x.dtype)

        # Fused router: computes logits + topk in one kernel
        expert_ids, expert_weights = _fused_router_topk(
            x, self.router.weight, self.config.num_experts_per_tok
        )

        # Compute auxiliary balance loss if enabled
        # Note: This requires full logits which the fused kernel doesn't return
        # So we only compute them if absolutely necessary
        if self.aux_loss_weight > 0.0 and self.training:
            # Recompute logits for loss (slow path, but training only)
            router_logits = self.router(x)
            self._compute_balance_loss(router_logits)

        # Apply expert capacity limiting if configured
        if self.capacity_factor < float("inf"):
            expert_ids, expert_weights = self._apply_capacity_limit(
                expert_ids, expert_weights, num_tokens
            )

        # Try batched Metal dispatch (FAST path)
        if self._use_batched_dispatch and x.is_mps:
            output = self._forward_batched_metal(x, expert_ids, expert_weights)
            if output is not None:
                # Restore original shape
                if batch is not None:
                    output = output.view(batch, seq_len, hidden)
                return output.to(x.dtype)

        # Fallback to Python-level iteration (SLOW path)
        output = self._forward_python_loop(
            x, expert_ids, expert_weights, num_tokens)

        # Restore original shape
        if batch is not None:
            output = output.view(batch, seq_len, hidden)

        return output.to(x.dtype)

    def _forward_batched_metal(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """Execute forward pass using batched Metal kernel.

        All expert computations happen in a single Metal kernel launch,
        avoiding Python-level loops and per-expert dispatch overhead.

        Args:
            x: Input activations [num_tokens, hidden_dim].
            expert_ids: Selected expert indices [num_tokens, top_k].
            expert_weights: Expert routing weights [num_tokens, top_k].

        Returns:
            Output tensor [num_tokens, hidden_dim], or None if dispatch fails.
        """
        try:
            lib = self._get_lib()
            cached_buffers = self._get_cached_buffers()
            buffer_pool = self._get_buffer_pool()

            if cached_buffers is None:
                return None

            # Update statistics
            num_tokens = x.shape[0]
            _moe_stats["total_experts_activated"] += self.config.num_experts_per_tok
            _log_moe_dispatch(
                "expert_execution",
                num_experts=self.config.num_experts,
                num_tokens=num_tokens,
                topk=self.config.num_experts_per_tok,
                kernel="dispatch_moe_trellis_swiglu (batched Metal)",
            )

            # Dispatch batched kernel - processes all experts in single launch
            output = dispatch_moe_trellis_swiglu(
                lib=lib,
                activations=x,
                gate_weights=None,  # Using cached buffers
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
                expert_ids=expert_ids,
                expert_probs=expert_weights,
                hidden_dim=self.config.hidden_size,
                intermediate_dim=self.config.intermediate_size,
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_tok,
                bits=self.bits,
                cached_buffers=cached_buffers,
                buffer_pool=buffer_pool,
                use_fp32_acc=False,
            )

            return output

        except Exception as e:
            # Log warning and fall back to Python path
            import warnings
            warnings.warn(
                f"Batched Metal MoE dispatch failed, falling back to Python: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _forward_python_loop(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Execute forward pass using Python-level expert iteration.

        This is the fallback path when batched Metal dispatch is not available.
        Includes expert caching and speculative prefetching optimizations.

        Args:
            x: Input activations [num_tokens, hidden_dim].
            expert_ids: Selected expert indices [num_tokens, top_k].
            expert_weights: Expert routing weights [num_tokens, top_k].
            num_tokens: Number of input tokens.

        Returns:
            Output tensor [num_tokens, hidden_dim].
        """
        # Update expert cache: record selections and prefetch
        if self.expert_cache is not None:
            # Validate speculation from previous layer (if any)
            self.expert_cache.validate_speculation(expert_ids)

            self.expert_cache.record_selection(expert_ids)

            # Prefetch based on selected experts (definitive)
            prefetch_experts = expert_ids.unique().cpu().numpy().tolist()

            # Update cache if we have prefetch or periodically
            should_update = (
                bool(prefetch_experts)
                or len(self.expert_cache.selection_history) >= self.expert_cache.window_size
            )

            if should_update:
                self.expert_cache.update_cache(
                    self.experts, prefetch_indices=prefetch_experts)

            # Speculative prefetch for next layer
            if self.next_layer is not None and self.next_layer.expert_cache is not None:
                predicted_experts = self.expert_cache.predict_next_experts(
                    expert_ids, top_k=self.config.num_experts_per_tok
                )
                if predicted_experts:
                    self.expert_cache.start_speculative_prefetch(
                        predicted_experts,
                        self.next_layer.experts,
                        self.next_layer.expert_cache,
                    )

        # Group tokens by expert to reduce redundant weight loads
        total_assignments = num_tokens * self.config.num_experts_per_tok
        flat_expert_ids = expert_ids.reshape(-1)
        positions = torch.arange(total_assignments, device=x.device)
        sort_keys = flat_expert_ids * total_assignments + positions

        # Sort by expert ID to group tokens
        sorted_order = torch.argsort(sort_keys, stable=True)
        sorted_expert_ids = flat_expert_ids[sorted_order]
        sorted_token_indices = sorted_order // self.config.num_experts_per_tok
        sorted_slot_indices = sorted_order % self.config.num_experts_per_tok

        # Count tokens per expert
        expert_counts = torch.bincount(
            sorted_expert_ids, minlength=self.config.num_experts)
        expert_offsets = torch.zeros(
            self.config.num_experts + 1, dtype=torch.long, device=x.device)
        expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

        # Compute expert outputs with weight reuse
        output = torch.zeros_like(x)

        # Identify all active experts
        active_experts = []
        for i in range(self.config.num_experts):
            if expert_offsets[i] != expert_offsets[i + 1]:
                active_experts.append(i)

        # Update statistics and log
        _moe_stats["total_experts_activated"] += len(active_experts)
        _log_moe_dispatch(
            "expert_execution",
            num_experts=len(active_experts),
            num_tokens=num_tokens,
            topk=self.config.num_experts_per_tok,
            kernel="TrellisLinear (Python loop)" if x.is_mps else "CPU",
        )

        # Separate into resident (already on GPU) and streamed (on CPU)
        resident_experts = []
        streamed_experts = []

        for idx in active_experts:
            if self.expert_cache is not None and idx in self.expert_cache.cpu_experts:
                streamed_experts.append(idx)
            else:
                resident_experts.append(idx)

        if streamed_experts and (_MOE_DEBUG or logger.isEnabledFor(logging.DEBUG)):
            logger.info(
                f"MoE layer {self.layer_idx}: streaming {len(streamed_experts)} experts from CPU, "
                f"{len(resident_experts)} resident on GPU"
            )

        # 1. Trigger async transfers for streamed experts FIRST
        if self.expert_cache is not None:
            for expert_idx in streamed_experts:
                self.expert_cache.stream_to_gpu(expert_idx, non_blocking=True)

        # 2. Process resident experts (while transfers happen in background)
        for expert_idx in resident_experts:
            self._process_expert_tokens(
                expert_idx,
                expert_offsets[expert_idx],
                expert_offsets[expert_idx + 1],
                sorted_token_indices,
                sorted_slot_indices,
                expert_weights,
                x,
                output,
            )

        # 3. Process streamed experts (will wait for transfer if not finished)
        for expert_idx in streamed_experts:
            self._process_expert_tokens(
                expert_idx,
                expert_offsets[expert_idx],
                expert_offsets[expert_idx + 1],
                sorted_token_indices,
                sorted_slot_indices,
                expert_weights,
                x,
                output,
            )

        return output

    def _compute_balance_loss(self, router_logits: torch.Tensor) -> None:
        """Compute auxiliary load balance loss.

        Balance loss penalizes uneven expert utilization by computing
        the variance of expert loads across the batch. Uses soft probabilities
        to ensure differentiability.

        Args:
            router_logits: Router output logits [tokens, num_experts].
        """
        # Compute soft routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Compute average load per expert (differentiable)
        expert_loads = routing_probs.mean(dim=0)  # [num_experts]

        # Compute coefficient of variation (std/mean) as balance loss
        # Lower CV = more balanced expert usage
        mean_load = expert_loads.mean()
        std_load = expert_loads.std()

        # CV loss: higher std relative to mean = worse balance
        # Add epsilon to mean to avoid division by zero
        cv_loss = std_load / (mean_load + 1e-6)

        # Store weighted loss
        self.aux_loss = self.aux_loss_weight * cv_loss

    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary balance loss for backprop.

        Returns:
            Auxiliary loss tensor (scalar).
        """
        return self.aux_loss


class _DictWeight:
    """Adapter to make a dict look like TrellisWeight for from_trellis_weight."""

    def __init__(self, d: dict) -> None:
        self.packed_indices = d["indices"]
        self.scales = d["scales"]
        self.su = d["su"]
        self.sv = d["sv"]
        self.bits = d["bits"]
        self.original_shape = d["original_shape"]


class _FusedTrellisExpert(nn.Module):
    """Expert with fused gate_up projection."""

    def __init__(
        self,
        gate_up_weight: TrellisLinear,
        down_weight: TrellisLinear,
        config: TrellisModelConfig,
    ) -> None:
        super().__init__()
        self.gate_up = gate_up_weight
        self.down = down_weight
        self.intermediate_size = config.intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate_up: output is [gate, up] concatenated
        gate_up_out = self.gate_up(x)

        # Check if truly fused (2x intermediate) or just named oddly
        if gate_up_out.shape[-1] == 2 * self.intermediate_size:
            gate, up = gate_up_out.chunk(2, dim=-1)
        else:
            # Not truly fused - use same output for both (fallback)
            gate = gate_up_out
            up = gate_up_out

        return self.down(F.silu(gate) * up)


class _DummyExpert(nn.Module):
    """Dummy expert for testing when weights aren't available."""

    def __init__(self, hidden_size: int, intermediate_size: int, device: str) -> None:
        super().__init__()
        # Use regular linear for testing
        self.gate = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device)
        self.up = nn.Linear(hidden_size, intermediate_size,
                            bias=False, device=device)
        self.down = nn.Linear(
            intermediate_size, hidden_size, bias=False, device=device)

        # Initialize with small values
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TrellisExpert(nn.Module):
    """Single expert with trellis-quantized weights.

    Implements the SwiGLU structure:
        out = down_proj(silu(gate_proj(x)) * up_proj(x))

    Each projection uses trellis-quantized weights for memory efficiency.
    """

    def __init__(
        self,
        gate_proj: TrellisLinear,
        up_proj: TrellisLinear,
        down_proj: TrellisLinear,
    ):
        """Initialize TrellisExpert.

        Args:
            gate_proj: TrellisLinear for gate projection.
            up_proj: TrellisLinear for up projection.
            down_proj: TrellisLinear for down projection.
        """
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    @classmethod
    def from_trellis_weights(
        cls,
        layer_weights: dict[str, TrellisWeight],
        expert_idx: int,
        layer_idx: int,
        device: str = "mps",
    ) -> TrellisExpert:
        """Create TrellisExpert from layer weights dictionary.

        Args:
            layer_weights: Dictionary mapping weight names to TrellisWeight objects.
            expert_idx: Index of the expert to load.
            layer_idx: Index of the transformer layer.
            device: Device to place weights on (default: "mps").

        Returns:
            TrellisExpert initialized with the specified expert's weights.

        Raises:
            KeyError: If required weights are not found in layer_weights.
        """
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        return cls(
            gate_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.gate_proj.weight"], device=device
            ),
            up_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.up_proj.weight"], device=device
            ),
            down_proj=TrellisLinear.from_trellis_weight(
                layer_weights[f"{prefix}.down_proj.weight"], device=device
            ),
        )

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"gate_proj={self.gate_proj.in_features}x{self.gate_proj.out_features}, "
            f"up_proj={self.up_proj.in_features}x{self.up_proj.out_features}, "
            f"down_proj={self.down_proj.in_features}x{self.down_proj.out_features}"
        )
