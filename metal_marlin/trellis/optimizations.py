"""Optimization components for Trellis MoE models.

This module provides specialized components for optimizing MoE inference:
- MixedBPWMoEDispatcher: Handles dispatch for mixed bit-width experts
- ExpertSelectionCache: Caches routing decisions for decode
- ExpertMemoryPool: Manages hot/cold expert loading for large models
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..metal_dispatch import MetalKernelLibrary
from .moe_dispatch import (
    CachedWeightBuffers,
    MoEBufferPool,
    create_cached_weight_buffers_from_cpu,
    dispatch_moe_trellis_swiglu,
)

if TYPE_CHECKING:
    from .model import TrellisMoEMLP

logger = logging.getLogger(__name__)


class MixedBPWMoEDispatcher:
    """Dispatcher for mixed-precision MoE models (sensitivity-aware quantization).

    Handles grouping experts by bit width and dispatching each group efficiently
    using Metal kernels.
    """

    def __init__(self, layer: "TrellisMoEMLP"):
        self.layer = layer
        self._bit_group_buffers: dict[
            tuple[int, int, int], tuple[CachedWeightBuffers, list[int]]
        ] | None = None
        self._bit_group_lookup_cache: dict[
            str, dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]
        ] = {}

    def ensure_buffers(self) -> None:
        """Ensure per-bit-group buffers are created."""
        if self._bit_group_buffers is not None:
            return

        # If layer has global buffers, try to create zero-copy views
        # This is optimization logic from _ensure_bit_group_buffers
        pass  # Logic handled by the layer or we move it here? 
        # For now, we'll assume the layer manages creation via _create_bit_group_buffers
        # or we rely on the layer's state.
        # Actually, let's just use what the layer has.
        self._bit_group_buffers = getattr(self.layer, "_bit_group_buffers", None)

    def dispatch(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        lib: MetalKernelLibrary,
        buffer_pool: MoEBufferPool,
    ) -> torch.Tensor:
        """Grouped dispatch for mixed-precision experts."""
        batch_size = x.shape[0]
        device = x.device

        # Initialize output accumulator using workspace buffer pool
        workspace_pool = self.layer._get_workspace_buffer_pool()
        output = workspace_pool.get_output_buffer(batch_size).zero_()

        self.ensure_buffers()
        if not self._bit_group_buffers:
            # Fallback if buffers not available (should be handled by layer fallback)
            return output

        device_key = str(device)
        device_lookup_cache = self._bit_group_lookup_cache.setdefault(device_key, {})
        num_experts_total = len(self.layer.experts)

        # Group selected expert slots by bit width and dispatch once per active group.
        for bits, (cached_buffers, group_expert_indices) in self._bit_group_buffers.items():
            if not group_expert_indices:
                continue

            lookup_entry = device_lookup_cache.get(bits)
            if lookup_entry is None:
                group_ids_t = torch.tensor(
                    group_expert_indices,
                    dtype=torch.long,
                    device=device,
                )
                local_id_map = torch.full(
                    (num_experts_total,),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                local_id_map[group_ids_t] = torch.arange(
                    group_ids_t.numel(),
                    dtype=torch.long,
                    device=device,
                )
                lookup_entry = (group_ids_t, local_id_map)
                device_lookup_cache[bits] = lookup_entry

            _group_ids_t, local_id_map = lookup_entry

            # selected_local_ids: [batch, top_k], -1 where not in this bit group.
            selected_local_ids = local_id_map[selected_experts]
            batch_indices, slot_indices = torch.nonzero(
                selected_local_ids >= 0, as_tuple=True
            )
            if batch_indices.numel() == 0:
                continue

            group_activations = x[batch_indices]
            local_expert_ids = selected_local_ids[batch_indices, slot_indices].unsqueeze(1)
            group_probs = routing_weights[batch_indices, slot_indices].unsqueeze(1)

            group_output = dispatch_moe_trellis_swiglu(
                lib=lib,
                activations=group_activations,
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
                expert_ids=local_expert_ids,
                expert_probs=group_probs,
                hidden_dim=self.layer.hidden_dim,
                intermediate_dim=self.layer.intermediate_dim,
                num_experts=len(group_expert_indices),
                top_k=1,
                bits=bits,
                cached_buffers=cached_buffers,
                buffer_pool=buffer_pool,
                use_fp32_acc=self.layer.hidden_dim >= 1024,
            )

            output.index_add_(0, batch_indices, group_output)

        return output


class ExpertSelectionCache:
    """Cache for MoE routing decisions during decode.

    Consecutive tokens often select the same experts. This cache allows reusing
    the routing decision (selected experts and weights) if the input embedding
    is similar enough to the previous one (cosine similarity > threshold).
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self._cached_routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.hits = 0
        self.misses = 0

    def check(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Check cache for reusable routing.

        Args:
            x: Input tensor [1, hidden_dim].

        Returns:
            Tuple (selected_experts, routing_weights) or None.
        """
        if self._cached_routing is None:
            return None

        cached_norm, cached_experts, cached_weights = self._cached_routing

        # Fast cosine similarity
        x_flat = x.view(-1)
        x_norm = x_flat / (x_flat.norm() + 1e-8)
        similarity = float(torch.dot(x_norm, cached_norm).cpu().numpy())

        if similarity >= self.threshold:
            self.hits += 1
            return cached_experts, cached_weights

        self.misses += 1
        return None

    def update(
        self, x: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor
    ) -> None:
        """Update cache with new routing decision."""
        x_flat = x.view(-1)
        x_norm = x_flat / (x_flat.norm() + 1e-8)
        self._cached_routing = (
            x_norm.clone(),
            selected_experts.clone(),
            routing_weights.clone(),
        )

    def clear(self) -> None:
        self._cached_routing = None

    def get_stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


class ExpertMemoryPool:
    """Manages memory for hot/cold experts in large models.

    Keeps frequently used (hot) experts in dedicated buffers and loads
    infrequently used (cold) experts on demand into a shared pool.
    """

    def __init__(self, experts: Any, hot_threshold: float = 0.5):
        self.experts = experts  # nn.ModuleList
        self.hot_threshold = hot_threshold
        self.hot_experts: set[int] = set(range(len(experts)))
        self.cold_buffer_pool: dict[int, CachedWeightBuffers] = {}
        self.selection_counts: list[int] = [0] * len(experts)
        self.call_count = 0
        self.update_interval = 100

    def record_selection(self, selected_experts: torch.Tensor) -> None:
        """Record expert usage."""
        flat = selected_experts.flatten().tolist()
        for eid in flat:
            self.selection_counts[eid] += 1
        
        self.call_count += 1
        if self.call_count % self.update_interval == 0:
            self._recompute_hot_experts()

    def _recompute_hot_experts(self) -> None:
        num_experts = len(self.experts)
        if num_experts == 0:
            return

        counts = [(i, c) for i, c in enumerate(self.selection_counts)]
        counts.sort(key=lambda x: x[1], reverse=True)

        num_hot = max(1, int(num_experts * self.hot_threshold))
        self.hot_experts = set(eid for eid, _ in counts[:num_hot])
        
        # Clear cold buffers to free memory
        self.cold_buffer_pool.clear()
        
        logger.debug(
            f"Recomputed hot experts: {len(self.hot_experts)} hot, {num_experts - len(self.hot_experts)} cold"
        )

    def get_cold_expert_buffers(self, expert_id: int, lib: MetalKernelLibrary) -> CachedWeightBuffers | None:
        """Load cold expert buffers on demand."""
        if expert_id in self.cold_buffer_pool:
            return self.cold_buffer_pool[expert_id]
        
        try:
            expert = self.experts[expert_id]
            
            # Create buffers from CPU (assuming weights available on CPU)
            # This logic assumes expert weights are on CPU or can be moved
            # The original logic in model.py accessed .packed_indices.cpu()
            
            cpu_weights = {
                "gate_weights": expert.gate_proj.packed_indices.cpu().permute(1, 0, 2).contiguous(),
                "gate_scales": expert.gate_proj.scales.cpu().half(),
                "up_weights": expert.up_proj.packed_indices.cpu().permute(1, 0, 2).contiguous(),
                "up_scales": expert.up_proj.scales.cpu().half(),
                "down_weights": expert.down_proj.packed_indices.cpu().permute(1, 0, 2).contiguous(),
                "down_scales": expert.down_proj.scales.cpu().half(),
                "gate_su": expert.gate_proj.su.cpu().half(),
                "gate_sv": expert.gate_proj.sv.cpu().half(),
                "up_su": expert.up_proj.su.cpu().half(),
                "up_sv": expert.up_proj.sv.cpu().half(),
                "down_su": expert.down_proj.su.cpu().half(),
                "down_sv": expert.down_proj.sv.cpu().half(),
                "grid": expert.gate_proj.grid.cpu().half(),
            }
            
            buffers = create_cached_weight_buffers_from_cpu(device=lib.device, **cpu_weights)
            self.cold_buffer_pool[expert_id] = buffers
            return buffers
        except Exception as e:
            logger.warning(f"Failed to load cold expert {expert_id}: {e}")
            return None

    def reset_stats(self) -> None:
        self.selection_counts = [0] * len(self.experts)
        self.call_count = 0
        self.hot_experts = set(range(len(self.experts)))
        self.cold_buffer_pool.clear()
