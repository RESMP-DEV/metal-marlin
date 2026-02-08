"""Example integration of mixed kernel selection with MoE dispatch.

This module demonstrates how to use the intelligent mixed-precision kernel
selector in conjunction with the MoE dispatch system.

Usage Example:
    ```python
    from metal_marlin.trellis.kernel_selection_mixed import (
        get_mixed_kernel,
        record_kernel_latency,
    )
    from metal_marlin.trellis.moe_dispatch import dispatch_moe_trellis_swiglu

    # Get active expert bit-widths (from expert_ids and metadata)
    active_bits = get_expert_bits(expert_ids, expert_bit_metadata)

    # Select optimal kernel
    kernel_name, metadata = get_mixed_kernel(
        batch_size=1,
        active_expert_bits=active_bits,
        gpu_memory_pressure=0.3,
        gate_bits=6,
        up_bits=2,
        down_bits=3,
        use_fp32_acc=False,
        available_kernels=get_available_kernels(lib),
    )

    # Dispatch with selected kernel
    output = dispatch_moe_trellis_swiglu(
        lib=lib,
        activations=activations,
        gate_weights=gate_weights,
        gate_scales=gate_scales,
        up_weights=up_weights,
        up_scales=up_scales,
        down_weights=down_weights,
        down_scales=down_scales,
        gate_su=gate_su,
        gate_sv=gate_sv,
        up_su=up_su,
        up_sv=up_sv,
        down_su=down_su,
        down_sv=down_sv,
        grid=grid,
        expert_ids=expert_ids,
        expert_probs=expert_probs,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
        bits=(gate_bits, up_bits, down_bits),
        cached_buffers=cached_buffers,
        buffer_pool=buffer_pool,
        use_fp32_acc=use_fp32_acc,
        kernel_name_override=kernel_name,  # Use selected kernel
    )

    # Record timing for feedback loop
    record_kernel_latency(kernel_name, batch_size, latency_ms)
    ```
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch


def get_expert_bits(
    expert_ids: torch.Tensor,
    expert_bit_metadata: Dict[int, Tuple[int, int, int]],
) -> List[int]:
    """Extract bit-width information for activated experts.

    Args:
        expert_ids: Tensor of expert IDs [batch, top_k]
        expert_bit_metadata: Mapping from expert_id to (gate_bits, up_bits, down_bits)

    Returns:
        List of bit-widths for all unique activated experts
    """
    unique_experts = torch.unique(expert_ids).tolist()
    active_bits = []

    for expert_id in unique_experts:
        if expert_id in expert_bit_metadata:
            gate_bits, up_bits, down_bits = expert_bit_metadata[expert_id]
            # Use up_bits as representative (or could use max/min)
            active_bits.append(up_bits)

    return active_bits


def estimate_gpu_memory_pressure(device: torch.device) -> float:
    """Estimate current GPU memory pressure (0.0 to 1.0).

    Args:
        device: PyTorch device

    Returns:
        Memory pressure as a float between 0.0 and 1.0
    """
    if device.type != "mps":
        return 0.0

    # For MPS, we don't have direct memory info
    # Return 0.0 for now (could be enhanced with MPS API)
    return 0.0


def get_available_kernels(lib: object) -> Set[str]:
    """Get the set of available kernel names from the Metal library.

    Args:
        lib: MetalKernelLibrary instance

    Returns:
        Set of available kernel names
    """
    # This is a placeholder - actual implementation would query the Metal library
    # For now, return common kernels
    return {
        "moe_trellis_swiglu_decode",
        "moe_trellis_swiglu_prefill4",
        "moe_trellis_swiglu",
        "moe_trellis_swiglu_large_batch",
        "moe_trellis_swiglu_prefill4_fp32acc",
        "moe_trellis_swiglu_fp32acc",
    }


def analyze_expert_activation_pattern(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> Dict[str, float]:
    """Analyze the expert activation pattern.

    Args:
        expert_ids: Tensor of expert IDs [batch, top_k]
        num_experts: Total number of experts in the model

    Returns:
        Dictionary with pattern metrics
    """
    unique_experts = torch.unique(expert_ids).tolist()
    num_activated = len(unique_experts)
    total_tokens = expert_ids.numel()

    return {
        "activation_rate": num_activated / num_experts,
        "tokens_per_expert_avg": total_tokens / num_activated if num_activated > 0 else 0.0,
        "num_activated_experts": num_activated,
        "expert_utilization": num_activated / num_experts,
    }


# Example integration class
class MixedBpwMoEDispatcher:
    """Intelligent dispatcher for mixed-precision MoE layers."""

    def __init__(
        self,
        lib,
        expert_bit_metadata: Dict[int, Tuple[int, int, int]],
        use_fp32_acc: bool = False,
        enable_ab_testing: bool = True,
        exploration_rate: float = 0.05,
    ):
        """Initialize the dispatcher.

        Args:
            lib: MetalKernelLibrary instance
            expert_bit_metadata: Mapping from expert_id to (gate_bits, up_bits, down_bits)
            use_fp32_acc: Whether to use FP32 accumulation
            enable_ab_testing: Whether to enable A/B testing exploration
            exploration_rate: Exploration rate for A/B testing (0.0-1.0)
        """
        self.lib = lib
        self.expert_bit_metadata = expert_bit_metadata
        self.use_fp32_acc = use_fp32_acc
        self.available_kernels = get_available_kernels(lib)

        # Import at runtime to avoid circular imports
        from .kernel_selection_mixed import set_exploration_rate

        if enable_ab_testing:
            set_exploration_rate(exploration_rate)
        else:
            set_exploration_rate(0.0)

    def dispatch(
        self,
        activations: torch.Tensor,
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
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        cached_buffers=None,
        buffer_pool=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Dispatch MoE computation with intelligent kernel selection.

        Returns:
            Tuple of (output_tensor, metadata_dict)
        """
        # Import at runtime
        from .kernel_selection_mixed import get_mixed_kernel, record_kernel_latency
        from .moe_dispatch import dispatch_moe_trellis_swiglu
        import time

        # Get expert bit-widths
        active_bits = get_expert_bits(expert_ids, self.expert_bit_metadata)

        # Estimate GPU memory pressure
        memory_pressure = estimate_gpu_memory_pressure(activations.device)

        # Get gate/up/down bit widths (use first expert if uniform, or most common)
        if active_bits:
            most_common_bits = max(set(active_bits), key=active_bits.count)
            gate_bits = up_bits = down_bits = most_common_bits
        else:
            gate_bits = up_bits = down_bits = 4  # Default

        # Select optimal kernel
        batch_size = activations.shape[0]
        kernel_name, selection_metadata = get_mixed_kernel(
            batch_size=batch_size,
            active_expert_bits=active_bits,
            gpu_memory_pressure=memory_pressure,
            gate_bits=gate_bits,
            up_bits=up_bits,
            down_bits=down_bits,
            use_fp32_acc=self.use_fp32_acc,
            available_kernels=self.available_kernels,
        )

        # Dispatch kernel
        start_time = time.perf_counter()
        output = dispatch_moe_trellis_swiglu(
            lib=self.lib,
            activations=activations,
            gate_weights=gate_weights,
            gate_scales=gate_scales,
            up_weights=up_weights,
            up_scales=up_scales,
            down_weights=down_weights,
            down_scales=down_scales,
            gate_su=gate_su,
            gate_sv=gate_sv,
            up_su=up_su,
            up_sv=up_sv,
            down_su=down_su,
            down_sv=down_sv,
            grid=grid,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            bits=(gate_bits, up_bits, down_bits),
            cached_buffers=cached_buffers,
            buffer_pool=buffer_pool,
            use_fp32_acc=self.use_fp32_acc,
            kernel_name_override=kernel_name,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Record timing for feedback loop
        record_kernel_latency(kernel_name, batch_size, latency_ms)

        # Analyze activation pattern
        pattern_metrics = analyze_expert_activation_pattern(expert_ids, num_experts)

        # Combine metadata
        metadata = {
            "kernel_name": kernel_name,
            "latency_ms": latency_ms,
            "batch_size": batch_size,
            "active_bits": active_bits,
            "memory_pressure": memory_pressure,
            "pattern_metrics": pattern_metrics,
            "selection_metadata": selection_metadata,
        }

        return output, metadata


# Example usage in a model
def example_usage_in_model():
    """Example showing how to use the dispatcher in a model."""
    # This is pseudocode showing the integration pattern

    # Initialize dispatcher (once per model)
    dispatcher = MixedBpwMoEDispatcher(
        lib=metal_lib,  # Your MetalKernelLibrary instance
        expert_bit_metadata={
            0: (6, 2, 3),  # expert_id: (gate_bits, up_bits, down_bits)
            1: (6, 2, 3),
            2: (6, 3, 4),
            3: (6, 4, 4),
        },
        use_fp32_acc=False,
        enable_ab_testing=True,
        exploration_rate=0.05,
    )

    # In forward pass
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
    ):
        # Dispatch with intelligent kernel selection
        output, metadata = dispatcher.dispatch(
            activations=hidden_states,
            gate_weights=self.gate_weights,
            gate_scales=self.gate_scales,
            up_weights=self.up_weights,
            up_scales=self.up_scales,
            down_weights=self.down_weights,
            down_scales=self.down_scales,
            gate_su=self.gate_su,
            gate_sv=self.gate_sv,
            up_su=self.up_su,
            up_sv=self.up_sv,
            down_su=self.down_su,
            down_sv=self.down_sv,
            grid=self.grid,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            cached_buffers=self.cached_buffers,
            buffer_pool=self.buffer_pool,
        )

        # Optionally log metadata for monitoring
        # logger.info(f"Kernel: {metadata['kernel_name']}, "
        #             f"Latency: {metadata['latency_ms']:.2f}ms, "
        #             f"Selection: {metadata['selection_metadata']['reason']}")

        return output
