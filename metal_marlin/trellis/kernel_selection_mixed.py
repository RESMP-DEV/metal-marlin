"""Intelligent, mixed-precision kernel selection for Metal-Marlin.

This module extends the baseline kernel selection with dynamic, data-driven
heuristics for mixed-bits-per-weight (BPW) models. It aims to select the
optimal kernel by considering not just batch size but also the specific mix of
quantization bit-widths in the active experts.

Key Features:
1.  **Mixed-BPW Heuristics**:
    - Prioritizes a highly optimized `fast_2bit_kernel` when a majority of
      experts use 2-bit quantization, as this is common in memory-constrained
      or high-throughput scenarios.
    - Employs a `mixed_bpw_kernel` designed to handle heterogeneous bit-widths
      (e.g., 2, 3, and 4-bit) efficiently within a single operation.
    - Falls back to the standard, high-performance trellis kernels when the
      expert composition is dominated by 4-bit weights.

2.  **Dynamic Selection Framework (Future Work)**:
    - The architecture is designed to incorporate more dynamic signals for
      kernel selection, including:
      - Real-time batch size and expert activation patterns.
      - GPU memory pressure, to potentially select less memory-intensive
        kernels under load.
      - A feedback loop informed by previous kernel execution timings, allowing
        the selector to adapt to changing workloads.

3.  **A/B Testing and Validation (Future Work)**:
    - Includes a placeholder for an A/B testing framework to empirically
      validate kernel selections. This will enable running candidate kernels
      competitively and choosing the faster one, while logging accuracy to
      continuously refine the selection heuristics.
"""

from __future__ import annotations
import time
import random
from collections import Counter

# --- Existing kernel definitions from kernel_selection.py ---
M4_MAX_THRESHOLDS = {
    "decode_max": 1,
    "prefill4_max": 16,
    "base_max": 32,
    "large_batch_min": 33,
}
TILE_SIZES = {
    "decode": 64,
    "prefill4": 64,
    "base": 64,
    "large_batch": 128,
}
SPECIALIZED_DECODE_KERNELS = {
    (6, 2, 3): "moe_trellis_swiglu_decode_6_2_3",
    (6, 3, 4): "moe_trellis_swiglu_decode_6_3_4",
    (6, 2, 4): "moe_trellis_swiglu_decode_6_2_4",
}

# --- New kernel names for mixed-BPW scenarios ---
FAST_2BIT_KERNEL = "moe_trellis_fast_2bit_swiglu"
MIXED_BPW_KERNEL = "moe_trellis_mixed_bpw_swiglu"


class KernelFeedback:
    """A simple feedback mechanism to track kernel performance."""

    def __init__(self):
        self.timings = {}  # (kernel_name, batch_size) -> [exec_time, ...]
        self.selection_log = []

    def record_timing(self, kernel_name: str, batch_size: int, exec_time: float):
        """Records the execution time for a given kernel and batch size."""
        key = (kernel_name, batch_size)
        if key not in self.timings:
            self.timings[key] = []
        self.timings[key].append(exec_time)

    def get_best_for_batch_size(self, batch_size: int):
        """Gets the historically best-performing kernel for a batch size."""
        best_kernel = None
        min_avg_time = float("inf")
        for (kernel, b_size), times in self.timings.items():
            if b_size == batch_size:
                avg_time = sum(times) / len(times)
                if avg_time < min_avg_time:
                    min_avg_time = avg_time
                    best_kernel = kernel
        return best_kernel


class KernelABTester:
    """A/B testing framework for comparing kernel performance."""

    def __init__(self, feedback_system: KernelFeedback):
        self.feedback = feedback_system
        self.enabled = False  # Disabled by default to avoid overhead

    def run_and_select(self, kernel_a: str, kernel_b: str, *args, **kwargs):
        """
        Runs two kernels with the same inputs, records their performance,
        and returns the result from the faster one.
        
        NOTE: This is a placeholder. A real implementation would need to
              execute the Metal kernels and synchronize properly.
        """
        if not self.enabled:
            # If not enabled, default to the first kernel provided (heuristic choice)
            return kernel_a

        # --- Placeholder for actual kernel execution ---
        start_a = time.perf_counter()
        # execute_metal_kernel(kernel_a, *args, **kwargs)
        time.sleep(random.uniform(0.0001, 0.0005)) # Simulate work
        end_a = time.perf_counter()
        duration_a = end_a - start_a
        
        start_b = time.perf_counter()
        # execute_metal_kernel(kernel_b, *args, **kwargs)
        time.sleep(random.uniform(0.0001, 0.0005)) # Simulate work
        end_b = time.perf_counter()
        duration_b = end_b - start_b

        # Record timings
        batch_size = kwargs.get("batch_size", 1)
        self.feedback.record_timing(kernel_a, batch_size, duration_a)
        self.feedback.record_timing(kernel_b, batch_size, duration_b)

        # Log selection and return the faster kernel
        chosen_kernel = kernel_a if duration_a <= duration_b else kernel_b
        heuristic_choice = kernel_a
        correct_selection = chosen_kernel == heuristic_choice
        self.feedback.selection_log.append({
            "heuristic": heuristic_choice,
            "winner": chosen_kernel,
            "correct": correct_selection,
            "duration_a": duration_a,
            "duration_b": duration_b,
        })
        
        return chosen_kernel


def select_intelligent_kernel(
    batch_size: int,
    expert_bpw: list[int] | None = None,
    use_fp32_acc: bool = False,
    gate_bits: int | None = None,
    up_bits: int | None = None,
    down_bits: int | None = None,
    available_kernels: set[str] | None = None,
    gpu_memory_pressure: float = 0.0, # Placeholder: 0.0 to 1.0
    feedback: KernelFeedback | None = None,
    ab_tester: KernelABTester | None = None,
) -> tuple[str, int]:
    """
    Selects the optimal kernel using mixed-BPW heuristics and dynamic data.
    
    Args:
        batch_size: The number of tokens in the current batch.
        expert_bpw: A list of bit-widths for the activated experts.
        use_fp32_acc: Whether to use FP32 accumulation.
        gate_bits, up_bits, down_bits: Bit-widths for specialized decode.
        available_kernels: Set of available kernel names for validation.
        gpu_memory_pressure: A normalized value indicating current memory load.
        feedback: A feedback system for recording and retrieving kernel timings.
        ab_tester: An A/B testing framework to empirically select the best kernel.

    Returns:
        A tuple of (kernel_name, tile_size).
    """
    # If no expert BPW data, fall back to standard selection
    if not expert_bpw:
        return get_kernel_for_batch_size(
            batch_size, use_fp32_acc, gate_bits, up_bits, down_bits, available_kernels
        )

    # 1. Mixed-BPW Heuristics
    bpw_counts = Counter(expert_bpw)
    num_experts = len(expert_bpw)
    
    # Heuristic 1: If >50% of experts are 2-bit, use the fast 2-bit kernel
    if bpw_counts.get(2, 0) / num_experts > 0.5:
        return FAST_2BIT_KERNEL, TILE_SIZES["base"]

    # Heuristic 2: If there's a mix of 2, 3, and 4 bit experts
    is_mixed_234 = bpw_counts.get(2, 0) > 0 and bpw_counts.get(3, 0) > 0 and bpw_counts.get(4, 0) > 0
    if is_mixed_234:
        return MIXED_BPW_KERNEL, TILE_SIZES["base"]

    # Heuristic 3: If mostly 4-bit, fallback to standard logic.
    # This is the default if no other heuristics match.

    # 2. Dynamic Selection (Future Work Placeholders)
    # TODO: Refine selection based on memory pressure.
    if gpu_memory_pressure > 0.8:
        pass  # e.g., prefer a kernel known to use less memory.
    
    # TODO: Use feedback loop to override heuristics.
    if feedback:
        historical_best = feedback.get_best_for_batch_size(batch_size)
        if historical_best:
            # Find the tile size associated with this historical kernel
            tile_size = TILE_SIZES.get(historical_best.split('_')[-1], TILE_SIZES["base"])
            return historical_best, tile_size
            
    # Fallback to the existing batch-size-based selection logic
    return get_kernel_for_batch_size(
        batch_size, use_fp32_acc, gate_bits, up_bits, down_bits, available_kernels
    )


def get_kernel_for_batch_size(
    batch_size: int,
    use_fp32_acc: bool = False,
    gate_bits: int | None = None,
    up_bits: int | None = None,
    down_bits: int | None = None,
    available_kernels: set[str] | None = None,
) -> tuple[str, int]:
    """
    Select optimal MoE kernel and tile size for given batch size.
    (Based on the implementation in kernel_selection.py)
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    
    thresholds = M4_MAX_THRESHOLDS
    
    # Decode path: batch_size == 1
    if batch_size <= thresholds["decode_max"]:
        if not use_fp32_acc and all(b is not None for b in (gate_bits, up_bits, down_bits)):
            key = (gate_bits, up_bits, down_bits)
            specialized_kernel = SPECIALIZED_DECODE_KERNELS.get(key)
            if (
                specialized_kernel is not None
                and available_kernels is not None
                and specialized_kernel in available_kernels
            ):
                return specialized_kernel, TILE_SIZES["decode"]
        return "moe_trellis_swiglu_decode", TILE_SIZES["decode"]
    
    # Small prefill: 2 <= batch_size <= 16
    if batch_size <= thresholds["prefill4_max"]:
        if use_fp32_acc:
            return "moe_trellis_swiglu_prefill4_fp32acc", TILE_SIZES["prefill4"]
        return "moe_trellis_swiglu_prefill4", TILE_SIZES["prefill4"]
    
    # Medium batch: 17 <= batch_size <= 32
    if batch_size <= thresholds["base_max"]:
        if use_fp32_acc:
            return "moe_trellis_swiglu_fp32acc", TILE_SIZES["base"]
        return "moe_trellis_swiglu", TILE_SIZES["base"]
    
    # Large batch: batch_size >= 33
    if use_fp32_acc:
        return "moe_trellis_swiglu_fp32acc", TILE_SIZES["base"]
    return "moe_trellis_swiglu_large_batch", TILE_SIZES["large_batch"]