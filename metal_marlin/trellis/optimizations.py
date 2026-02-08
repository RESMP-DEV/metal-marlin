"""Optimization components for Trellis MoE models.

This module provides specialized components for optimizing MoE inference:
- MixedBPWMoEDispatcher: Handles dispatch for mixed bit-width experts
- ExpertSelectionCache: Caches routing decisions for decode
- ExpertMemoryPool: Manages hot/cold expert loading for large models
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..metal_dispatch import (MetalKernelLibrary, _copy_buffer_to_tensor,
                              _CopyBackBuffer, mps_tensor_to_metal_buffer)
from .moe_dispatch import (CachedWeightBuffers, MoEBufferPool,
                           create_cached_weight_buffers_from_cpu,
                           select_moe_kernel)

if TYPE_CHECKING:
    from .async_dispatch import AsyncCommandBufferManager
    from .model import TrellisMoEMLP

logger = logging.getLogger(__name__)


class MixedBPWMoEDispatcher:
    """Dispatcher for mixed-precision MoE models (sensitivity-aware quantization).

    Handles grouping experts by bit width and dispatching each group efficiently
    using Metal kernels.
    """

    _warned_no_batch = False

    def __init__(self, layer: TrellisMoEMLP):
        self.layer = layer
        self._bit_group_buffers: dict[
            tuple[int, int, int], tuple[CachedWeightBuffers, list[int]]
        ] | None = None
        self._bit_group_lookup_cache: dict[
            str, dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        # Share routing cache with layer when available.
        self._expert_selection_cache = getattr(
            layer, "expert_selection_cache", ExpertSelectionCache()
        )

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
        self._bit_group_buffers = getattr(
            self.layer, "_bit_group_buffers", None)

    def dispatch(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
        lib: MetalKernelLibrary | None = None,
        buffer_pool: MoEBufferPool | None = None,
        cmd_manager: AsyncCommandBufferManager | None = None,
    ) -> torch.Tensor:
        """Grouped dispatch for mixed-precision experts.

        When ``cmd_manager`` is provided, kernels are queued and the caller must
        invoke ``cmd_manager.commit_and_wait()`` once after all dispatches.
        When omitted, this method creates a local batch and commits before
        returning for backward compatibility.
        """
        batch_size = x.shape[0]
        device = x.device
        if lib is None:
            raise ValueError("lib must be provided")
        if buffer_pool is None:
            raise ValueError("buffer_pool must be provided")

        if (selected_experts is None) != (routing_weights is None):
            raise ValueError(
                "selected_experts and routing_weights must both be provided or both omitted"
            )

        # Decode path can skip router when hidden state matches a recent token.
        if selected_experts is None or routing_weights is None:
            if batch_size == 1:
                cached_routing = self._expert_selection_cache.lookup(x)
                if cached_routing is not None:
                    selected_experts, routing_weights = cached_routing
                else:
                    router_logits = self.layer.router(x)
                    routing_weights, selected_experts = torch.topk(
                        F.softmax(router_logits, dim=-1, dtype=torch.float16),
                        k=self.layer.num_experts_per_tok,
                        dim=-1,
                    )
                    routing_weights = routing_weights / routing_weights.sum(
                        dim=-1, keepdim=True
                    )
                    self._expert_selection_cache.store(
                        x, selected_experts, routing_weights
                    )
            else:
                router_logits = self.layer.router(x)
                routing_weights, selected_experts = torch.topk(
                    F.softmax(router_logits, dim=-1, dtype=torch.float16),
                    k=self.layer.num_experts_per_tok,
                    dim=-1,
                )
                routing_weights = routing_weights / routing_weights.sum(
                    dim=-1, keepdim=True
                )
        elif batch_size == 1:
            # Keep decode cache warm when caller computes routing externally.
            self._expert_selection_cache.store(
                x, selected_experts, routing_weights)

        # Initialize output accumulator using workspace buffer pool
        workspace_pool = self.layer._get_workspace_buffer_pool()
        output = workspace_pool.get_output_buffer(batch_size).zero_()
        owns_cmd_batch = False
        dispatch_unbatched = False
        if cmd_manager is None:
            cmd_manager = self.layer._get_async_cmd_manager()
            cmd_manager.begin_batch()
            owns_cmd_batch = True
        elif not cmd_manager.has_active_batch():
            dispatch_unbatched = True
            if not MixedBPWMoEDispatcher._warned_no_batch:
                logger.warning(
                    "MixedBPWMoEDispatcher: No active batch, using unbatched dispatch"
                )
                MixedBPWMoEDispatcher._warned_no_batch = True

        self.ensure_buffers()
        if not self._bit_group_buffers:
            # Fallback if buffers not available (should be handled by layer fallback)
            if owns_cmd_batch:
                cmd_manager.commit_and_wait()
            return output

        device_key = str(device)
        device_lookup_cache = self._bit_group_lookup_cache.setdefault(
            device_key, {})
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
            local_expert_ids = selected_local_ids[batch_indices, slot_indices].unsqueeze(
                1)
            group_probs = routing_weights[batch_indices,
                                          slot_indices].unsqueeze(1)

            # Group by local expert for better memory locality in the kernel.
            sort_order = torch.argsort(local_expert_ids.view(-1), stable=True)
            group_activations = group_activations[sort_order].contiguous()
            local_expert_ids = local_expert_ids[sort_order].int().contiguous()
            group_probs = group_probs[sort_order]
            if group_probs.dtype != torch.float16:
                group_probs = group_probs.half()
            group_probs = group_probs.contiguous()
            scatter_indices = batch_indices[sort_order]

            group_batch = group_activations.shape[0]
            use_fp32_acc = self.layer.hidden_dim >= 1024

            if isinstance(bits, tuple):
                gate_bits, up_bits, down_bits = bits
            else:
                gate_bits = up_bits = down_bits = bits
            kernel_name, tile_n = select_moe_kernel(
                group_batch,
                use_fp32_acc,
                gate_bits=gate_bits,
                up_bits=up_bits,
                down_bits=down_bits,
            )

            grid_x = (self.layer.hidden_dim + tile_n - 1) // tile_n
            if kernel_name.startswith("moe_trellis_swiglu_decode"):
                grid_y = 1
            elif "prefill4" in kernel_name:
                grid_y = (group_batch + 3) // 4
            else:
                grid_y = group_batch

            # Do not use pooled dynamic buffers here: queued dispatches in one batch
            # must not alias mutable input/output buffers before commit.
            activations_buf = mps_tensor_to_metal_buffer(
                group_activations, lib.device)
            expert_ids_buf = mps_tensor_to_metal_buffer(
                local_expert_ids, lib.device)
            expert_probs_buf = mps_tensor_to_metal_buffer(
                group_probs, lib.device)

            group_output_fp32 = torch.zeros(
                group_batch, self.layer.hidden_dim, dtype=torch.float32, device=device
            )
            group_output_buf = mps_tensor_to_metal_buffer(
                group_output_fp32, lib.device, copy_back=True
            )
            params_buf = buffer_pool.get_params_buffer(
                group_batch,
                self.layer.hidden_dim,
                self.layer.intermediate_dim,
                len(group_expert_indices),
                1,
                bits,
            )

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
                group_output_buf,
                params_buf,
            ]

            keep_alive = (
                group_activations,
                local_expert_ids,
                group_probs,
                group_output_fp32,
                buffer_list,
            )

            def _accumulate_group(
                *,
                _scatter_indices: torch.Tensor = scatter_indices,
                _group_output_fp32: torch.Tensor = group_output_fp32,
                _keep_alive: tuple[Any, ...] = keep_alive,
            ) -> None:
                output.index_add_(0, _scatter_indices,
                                  _group_output_fp32.half())

            if dispatch_unbatched:
                immediate_buffers: list[Any] = []
                copy_back_buffers: list[_CopyBackBuffer] = []
                for item in buffer_list:
                    if isinstance(item, _CopyBackBuffer):
                        immediate_buffers.append(item.buffer)
                        copy_back_buffers.append(item)
                    else:
                        immediate_buffers.append(item)

                cmd_manager.dispatch_immediate(
                    pipeline=lib.get_pipeline(kernel_name),
                    grid=(grid_x, grid_y, 1),
                    threadgroup=(128, 1, 1),
                    buffers=immediate_buffers,
                )
                for copy_back in copy_back_buffers:
                    _copy_buffer_to_tensor(copy_back.buffer, copy_back.tensor)
                _accumulate_group()
            else:
                cmd_manager.dispatch_kernel(
                    function_name=kernel_name,
                    grid=(grid_x, grid_y, 1),
                    threadgroup=(128, 1, 1),
                    buffers=buffer_list,
                )
                cmd_manager.register_post_commit(_accumulate_group)

        if owns_cmd_batch:
            cmd_manager.commit_and_wait()
        return output


class ExpertSelectionCache:
    """Cache expert selections for similar hidden states."""

    def __init__(self, max_entries: int = 16, similarity_threshold: float = 0.95):
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hidden_hashes: dict[int, torch.Tensor] = {}
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @hits.setter
    def hits(self, value: int) -> None:
        self._hits = value

    @property
    def misses(self) -> int:
        return self._misses

    @misses.setter
    def misses(self, value: int) -> None:
        self._misses = value

    def _compute_hash(self, hidden: torch.Tensor) -> int:
        # Use downsampled hidden state as key.
        downsampled = hidden.detach()[..., ::64].float().cpu().numpy()
        return hash(downsampled.tobytes())

    def lookup(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        h = self._compute_hash(hidden)
        if h in self._cache:
            cached_hidden = self._hidden_hashes[h]
            similarity = torch.cosine_similarity(
                hidden.detach().flatten().float(),
                cached_hidden.flatten().float(),
                dim=0,
            )
            if float(similarity.item()) > self.similarity_threshold:
                self._hits += 1
                return self._cache[h]
        self._misses += 1
        return None

    def store(
        self, hidden: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor
    ) -> None:
        if len(self._cache) >= self.max_entries:
            # Evict oldest entry.
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            del self._hidden_hashes[oldest]
        h = self._compute_hash(hidden)
        self._cache[h] = (experts.clone(), weights.clone())
        self._hidden_hashes[h] = hidden.detach().clone()

    # Backward-compatible aliases for existing call sites.
    def check(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.lookup(x)

    def update(
        self, x: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor
    ) -> None:
        self.store(x, selected_experts, routing_weights)

    def clear(self) -> None:
        self._cache.clear()
        self._hidden_hashes.clear()

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
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

            buffers = create_cached_weight_buffers_from_cpu(
                device=lib.device, **cpu_weights)
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
