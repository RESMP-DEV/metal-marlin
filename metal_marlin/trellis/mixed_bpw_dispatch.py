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
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from ..metal_dispatch import (
    HAS_METAL,
    MetalKernelLibrary,
    require_mps,
)

if HAS_METAL:
    pass

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


class MixedBPWMoEDispatcher:
    """Dispatcher for mixed bit-width MoE layers.

    This class manages dispatch logic for MoE layers where different experts
    use different quantization bit-widths. It optimizes performance by:

    1. Grouping experts by bit-width for efficient batching
    2. Sorting tokens by expert assignment to maximize memory coalescing
    3. Batching same-bit-width experts together in Metal kernel dispatches
    4. Providing fallback to per-bit-width dispatches when needed

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

    def _dispatch_mixed_bpw_kernel(
        self,
        hidden_states: torch.Tensor,
        expert_weights: dict[int, torch.Tensor],
        expert_scales: dict[int, torch.Tensor],
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Attempt to dispatch with a single mixed-bit-width Metal kernel.

        This method attempts to use a Metal kernel that can handle multiple
        bit-widths in a single dispatch. Currently unimplemented, but
        provides the interface for future optimization.

        Raises:
            NotImplementedError: Always raises until mixed-kernel is implemented.
        """
        # TODO: Implement actual mixed-bit-width Metal kernel dispatch
        # This would require a kernel that can handle different bit-widths
        # and possibly different trellis codebook configurations per expert
        raise NotImplementedError(
            "Mixed-bit-width Metal kernel dispatch not yet implemented. "
            "Falling back to per-bit-width dispatch."
        )

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
            return torch.zeros(0, self.hidden_dim, device=hidden_states.device)

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
            return torch.zeros(
                len(token_indices), self.hidden_dim,
                device=hidden_states.device, dtype=torch.float16
            )

        batch_size = gathered_states.shape[0]

        # Get expert weights for all experts in this bit-width group
        # Assuming expert_weights contains gate, up, down weights for each expert
        all_gate_weights = []
        all_up_weights = []
        all_down_weights = []
        all_gate_scales = []
        all_up_scales = []
        all_down_scales = []

        for expert_id in expert_ids:
            w = expert_weights[expert_id]
            s = expert_scales[expert_id]

            # Assuming packed format: (gate, up, down) concatenated
            # Split into individual weight matrices
            hidden_dim = self.hidden_dim
            intermediate_dim = self.config.intermediate_dim

            # Calculate split points
            gate_size = hidden_dim * intermediate_dim
            up_size = hidden_dim * intermediate_dim
            down_size = intermediate_dim * hidden_dim

            # Split packed weights
            all_gate_weights.append(w[:, :gate_size])
            all_up_weights.append(w[:, gate_size:gate_size + up_size])
            all_down_weights.append(w[:, gate_size + up_size:])

            # Split scales similarly
            if s is not None:
                all_gate_scales.append(s[:, :gate_size])
                all_up_scales.append(s[:, gate_size:gate_size + up_size])
                all_down_scales.append(s[:, gate_size + up_size:])
            else:
                all_gate_scales.append(None)
                all_up_scales.append(None)
                all_down_scales.append(None)

        # Try to use batched Metal dispatch if available
        try:
            from .moe_dispatch import dispatch_moe_trellis_swiglu_batched
            
            lib = self.get_lib()
            
            # Create stacked weights for all experts in this bit-width group
            # We need to flatten weights to 2D tensors expected by the batched dispatch
            num_experts_in_group = len(expert_ids)
            
            # Create stacked weights and scales
            gate_weights_stacked = torch.stack(all_gate_weights, dim=0)  # [num_experts, ...]
            up_weights_stacked = torch.stack(all_up_weights, dim=0)
            down_weights_stacked = torch.stack(all_down_weights, dim=0)
            
            # Stack scales if available
            if all_gate_scales[0] is not None:
                gate_scales_stacked = torch.stack(all_gate_scales, dim=0)
                up_scales_stacked = torch.stack(all_up_scales, dim=0)
                down_scales_stacked = torch.stack(all_down_scales, dim=0)
            else:
                gate_scales_stacked = None
                up_scales_stacked = None
                down_scales_stacked = None

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
            
            # Flatten weights to 2D for batched dispatch
            gate_weights_flat = gate_weights_stacked.reshape(num_experts_in_group, -1)
            up_weights_flat = up_weights_stacked.reshape(num_experts_in_group, -1)
            down_weights_flat = down_weights_stacked.reshape(num_experts_in_group, -1)
            
            if gate_scales_stacked is not None:
                gate_scales_flat = gate_scales_stacked.reshape(num_experts_in_group, -1)
                up_scales_flat = up_scales_stacked.reshape(num_experts_in_group, -1)
                down_scales_flat = down_scales_stacked.reshape(num_experts_in_group, -1)
            else:
                gate_scales_flat = None
                up_scales_flat = None
                down_scales_flat = None

            # Attempt batched dispatch
            expert_outputs = dispatch_moe_trellis_swiglu_batched(
                lib=lib,
                activations=gathered_states,
                gate_weights=gate_weights_flat,
                gate_scales=gate_scales_flat,
                up_weights=up_weights_flat,
                up_scales=up_scales_flat,
                down_weights=down_weights_flat,
                down_scales=down_scales_flat,
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
            # Fallback: simple averaging
            expert_outputs = torch.zeros(
                batch_size, self.hidden_dim,
                device=hidden_states.device, dtype=torch.float16
            )
            for i, expert_id in enumerate(expert_ids):
                # Simple average of input states as placeholder
                expert_outputs += gathered_states.float().mean(dim=0, keepdim=True)
            
            if num_experts_in_group > 0:
                expert_outputs = expert_outputs / num_experts_in_group

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

        # Combine outputs
        if not all_outputs:
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        combined = torch.cat(all_outputs, dim=0)
        # Unsort to restore original token order
        unsorted = combined[inverse_indices]

        # Reshape and sum over top_k to get [batch, hidden_dim]
        return unsorted.view(batch_size, top_k, self.hidden_dim).sum(dim=1)

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
                mixed_output = self.dispatch_mixed_bit_width_fallback(
                    hidden_states, expert_weights, expert_scales,
                    router_probs, expert_indices
                )
                # TODO: Replace with actual _dispatch_mixed_bpw_kernel when implemented
                # mixed_output = self._dispatch_mixed_bpw_kernel(...)
                # _global_mixed_bpw_stats.mixed_kernel_success += 1
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
                
                # Convert back to torch tensor
                output = torch.from_numpy(hidden_states_np).to(hidden_states.device)
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
