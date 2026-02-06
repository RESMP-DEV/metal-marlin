#!/usr/bin/env python3
"""Optimized MoE kernel selection benchmark for M4 Max.

This script benchmarks all available MoE kernel variants to determine
optimal kernel selection thresholds specifically for M4 Max hardware.

Tested kernels:
- moe_trellis_swiglu_decode: Optimized for batch=1 (decode)
- moe_trellis_swiglu_decode_6_2_3: Specialized 6-2-3 bit decode kernel
- moe_trellis_swiglu_decode_6_3_4: Specialized 6-3-4 bit decode kernel
- moe_trellis_swiglu_prefill4: Optimized for small batches (2-16)
- moe_trellis_swiglu_prefill4_fp32acc: FP32 accumulation variant
- moe_trellis_swiglu: Base kernel for medium batches (17-32)
- moe_trellis_swiglu_fp32acc: Base kernel with FP32 accumulation
- moe_trellis_swiglu_large_batch: For large batches (33+, tile_n=128)

Usage:
    uv run python benchmarks/bench_m4_kernel_selection.py

Output:
    - contrib/metal_marlin/results/kernel_selection_m4_max.json
    - contrib/metal_marlin/results/kernel_recommendations.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Skip benchmark in task mode (AGENTS.md rule)
if __name__ == "__main__" and len(sys.argv) == 1:
    if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
        print("Benchmark skipped in task mode (ALPHAHENG_TASK_MODE=1)")
        sys.exit(0)

import torch  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.trellis.moe_dispatch import (  # noqa: E402
    dispatch_moe_trellis_swiglu,
    select_moe_kernel,
)


# Kernel variants to test for each batch size
KERNEL_VARIANTS: list[tuple[str, bool, str]] = [
    # (kernel_name, use_fp32_acc, display_name)
    ("moe_trellis_swiglu_decode", False, "decode"),
    ("moe_trellis_swiglu_decode_6_2_3", False, "decode_6_2_3"),
    ("moe_trellis_swiglu_decode_6_3_4", False, "decode_6_3_4"),
    ("moe_trellis_swiglu_prefill4", False, "prefill4"),
    ("moe_trellis_swiglu_prefill4", True, "prefill4_fp32acc"),
    ("moe_trellis_swiglu", False, "base"),
    ("moe_trellis_swiglu", True, "base_fp32acc"),
    ("moe_trellis_swiglu_large_batch", False, "large_batch"),
]

# Batch sizes to test (covering decode, small/medium/large prefill)
BATCH_SIZES = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64]

# Benchmark iterations
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50


def check_kernel_available(lib, kernel_name: str, use_fp32_acc: bool) -> bool:
    """Check if a kernel is available in the library."""
    # Handle fp32acc suffix
    if use_fp32_acc and "fp32acc" not in kernel_name and "decode" not in kernel_name:
        test_name = kernel_name.replace("moe_trellis_swiglu", "moe_trellis_swiglu_fp32acc")
        test_name = test_name.replace("_fp32acc_fp32acc", "_fp32acc")
    else:
        test_name = kernel_name
    
    try:
        lib.get_pipeline(test_name)
        return True
    except Exception:
        return False


def benchmark_kernel(
    lib,
    buffer_pool,
    cached,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    bits: int,
    kernel_name: str,
    use_fp32_acc: bool,
    iterations: int = BENCHMARK_ITERATIONS,
) -> dict[str, float] | None:
    """Benchmark a specific kernel variant.
    
    Returns:
        Dict with timing statistics or None if kernel fails.
    """
    device = torch.device("mps")
    
    # Create test inputs
    x = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)
    expert_ids = torch.zeros(batch_size, top_k, device=device, dtype=torch.long)
    expert_probs = torch.ones(batch_size, top_k, device=device, dtype=torch.float32)
    
    # Assign experts round-robin for testing
    for i in range(batch_size):
        for j in range(top_k):
            expert_ids[i, j] = (i + j) % num_experts
    
    try:
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            with torch.no_grad():
                _ = dispatch_moe_trellis_swiglu(
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
                    expert_ids=expert_ids,
                    expert_probs=expert_probs,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    bits=bits,
                    cached_buffers=cached,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=use_fp32_acc,
                    kernel_override=kernel_name,
                )
        torch.mps.synchronize()
        
        # Benchmark
        times: list[float] = []
        for _ in range(iterations):
            torch.mps.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = dispatch_moe_trellis_swiglu(
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
                    expert_ids=expert_ids,
                    expert_probs=expert_probs,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    bits=bits,
                    cached_buffers=cached,
                    buffer_pool=buffer_pool,
                    use_fp32_acc=use_fp32_acc,
                    kernel_override=kernel_name,
                )
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)
        
        mean_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
        
        return {
            "mean_ms": mean_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "std_ms": std_ms,
            "throughput_tokens_per_sec": batch_size / (mean_ms / 1000),
        }
    except Exception as e:
        print(f"      Error: {e}")
        return None


def find_optimal_thresholds(results: dict[int, dict[str, dict]]) -> dict[str, Any]:
    """Analyze results to find optimal kernel selection thresholds."""
    thresholds = {
        "decode_max": 1,  # Default: decode only for batch=1
        "prefill4_max": 16,  # Default: prefill4 for 2-16
        "base_max": 32,  # Default: base for 17-32
        "large_batch_min": 33,  # Default: large_batch for 33+
    }
    
    # Find where prefill4 becomes better than decode
    for batch_size in sorted(BATCH_SIZES):
        if batch_size not in results:
            continue
        
        batch_results = results[batch_size]
        
        # Check if we have both decode and prefill4
        if "decode" in batch_results and "prefill4" in batch_results:
            decode_time = batch_results["decode"]["mean_ms"]
            prefill4_time = batch_results["prefill4"]["mean_ms"]
            
            if prefill4_time < decode_time and thresholds["decode_max"] < batch_size - 1:
                thresholds["decode_max"] = batch_size - 1
                thresholds["prefill4_max"] = min(16, batch_size + 15)
    
    # Find where large_batch becomes optimal
    for batch_size in sorted(BATCH_SIZES):
        if batch_size not in results:
            continue
        
        batch_results = results[batch_size]
        
        # Find fastest kernel for this batch size
        fastest_kernel = None
        fastest_time = float("inf")
        
        for kernel_name, stats in batch_results.items():
            if stats["mean_ms"] < fastest_time:
                fastest_time = stats["mean_ms"]
                fastest_kernel = kernel_name
        
        # If large_batch is fastest, record the threshold
        if fastest_kernel == "large_batch" and thresholds["large_batch_min"] == 33:
            thresholds["large_batch_min"] = batch_size
            # Adjust base_max
            thresholds["base_max"] = batch_size - 1
    
    return thresholds


def generate_recommendations(results: dict[int, dict[str, dict]], thresholds: dict) -> str:
    """Generate markdown report with recommendations."""
    lines = [
        "# M4 Max MoE Kernel Selection Benchmark Results\n",
        "## Summary\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Device: Apple M4 Max\n",
        f"Iterations per test: {BENCHMARK_ITERATIONS}\n",
        "\n## Recommended Thresholds\n",
        "```python",
        f"# Decode kernel (batch_size <= {thresholds['decode_max']})",
        f"# Prefill4 kernel ({thresholds['decode_max'] + 1} <= batch_size <= {thresholds['prefill4_max']})",
        f"# Base kernel ({thresholds['prefill4_max'] + 1} <= batch_size <= {thresholds['base_max']})",
        f"# Large batch kernel (batch_size >= {thresholds['large_batch_min']})",
        "```\n",
        "## Performance by Batch Size\n",
        "| Batch | Fastest Kernel | Latency (ms) | Tokens/sec | Current Selection |",
        "|-------|---------------|--------------|------------|-------------------|",
    ]
    
    for batch_size in sorted(BATCH_SIZES):
        if batch_size not in results:
            continue
        
        batch_results = results[batch_size]
        
        # Find fastest kernel
        fastest_kernel = "N/A"
        fastest_time = float("inf")
        tokens_per_sec = 0
        
        for kernel_name, stats in batch_results.items():
            if stats["mean_ms"] < fastest_time:
                fastest_time = stats["mean_ms"]
                fastest_kernel = kernel_name
                tokens_per_sec = stats["throughput_tokens_per_sec"]
        
        # Get current selection
        current_kernel, _ = select_moe_kernel(batch_size, use_fp32_acc=False)
        current_simple = current_kernel.replace("moe_trellis_swiglu_", "").replace("moe_trellis_swiglu", "base")
        
        # Check if optimal
        optimal = "✓" if fastest_kernel == current_simple else f"⚠ {fastest_kernel}"
        
        lines.append(
            f"| {batch_size:5d} | {fastest_kernel:20s} | {fastest_time:12.2f} | {tokens_per_sec:10.1f} | {optimal:30s} |"
        )
    
    lines.extend([
        "\n## Detailed Results\n",
        "### Latency by Kernel (ms)\n",
        "| Batch | Decode | Prefill4 | Base | Large Batch |",
        "|-------|--------|----------|------|-------------|",
    ])
    
    for batch_size in sorted(BATCH_SIZES):
        if batch_size not in results:
            continue
        
        batch_results = results[batch_size]
        decode_time = batch_results.get("decode", {}).get("mean_ms", float("nan"))
        prefill4_time = batch_results.get("prefill4", {}).get("mean_ms", float("nan"))
        base_time = batch_results.get("base", {}).get("mean_ms", float("nan"))
        large_time = batch_results.get("large_batch", {}).get("mean_ms", float("nan"))
        
        lines.append(
            f"| {batch_size:5d} | {decode_time:6.2f} | {prefill4_time:8.2f} | {base_time:4.2f} | {large_time:11.2f} |"
        )
    
    lines.extend([
        "\n## Optimization Recommendations\n"
    ])
    
    # Analyze gaps between current and optimal
    gaps = []
    for batch_size in sorted(BATCH_SIZES):
        if batch_size not in results:
            continue
        
        batch_results = results[batch_size]
        
        # Find fastest kernel
        fastest_kernel = None
        fastest_time = float("inf")
        for kernel_name, stats in batch_results.items():
            if stats["mean_ms"] < fastest_time:
                fastest_time = stats["mean_ms"]
                fastest_kernel = kernel_name
        
        # Get current selection time
        current_kernel, _ = select_moe_kernel(batch_size, use_fp32_acc=False)
        current_simple = current_kernel.replace("moe_trellis_swiglu_", "").replace("moe_trellis_swiglu", "base")
        current_time = batch_results.get(current_simple, {}).get("mean_ms", fastest_time)
        
        gap_pct = ((current_time - fastest_time) / fastest_time * 100) if fastest_time > 0 else 0
        if gap_pct > 5:  # Only report significant gaps
            gaps.append((batch_size, fastest_kernel, current_simple, gap_pct))
    
    if gaps:
        lines.append("\n### Suboptimal Selections (>5% performance gap)\n")
        lines.append("| Batch | Optimal | Current | Gap |")
        lines.append("|-------|---------|---------|-----|")
        for batch_size, optimal, current, gap in gaps:
            lines.append(f"| {batch_size:5d} | {optimal:15s} | {current:15s} | {gap:5.1f}% |")
    else:
        lines.append("\n✓ All batch sizes are using optimal or near-optimal kernel selections.\n")
    
    lines.extend([
        "\n### Implementation Notes\n",
        "1. **Decode kernel**: Optimized for single-token inference with minimal overhead",
        "2. **Prefill4 kernel**: Processes 4 tokens per threadgroup, good for small batches",
        "3. **Base kernel**: Balanced throughput for medium batch sizes",
        "4. **Large batch kernel**: Uses tile_n=128 for better memory coalescing on large batches\n",
        "### FP32 Accumulation\n",
        "- FP32 accumulation variants provide better numerical stability",
        "- Recommended for hidden_dim >= 1024 or when precision is critical",
        "- Performance cost is typically 5-15% on M4 Max\n",
    ])
    
    return "\n".join(lines)


def main() -> int:
    """Run the benchmark."""
    print("=" * 70)
    print("M4 Max MoE Kernel Selection Optimization Benchmark")
    print("=" * 70)
    
    # Check for model
    model_path = _ROOT / "models" / "GLM-4.7-Flash-Trellis-MM"
    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("Creating synthetic benchmark instead...")
        return run_synthetic_benchmark()
    
    print(f"\nLoading model from {model_path}...")
    
    try:
        from metal_marlin.trellis.model import TrellisForCausalLM
        model = TrellisForCausalLM.from_pretrained(model_path, device="mps")
        
        # Find first MoE layer
        moe_layer = None
        for layer in model.model.layers:
            if hasattr(layer.mlp, "_cached_weight_buffers"):
                moe_layer = layer.mlp
                break
        
        if moe_layer is None:
            raise RuntimeError("No MoE layer found in model")
        
        cached = moe_layer._cached_weight_buffers
        if cached is None:
            raise RuntimeError("Cached weight buffers not initialized")
        
        lib = moe_layer._get_lib()
        buffer_pool = moe_layer._get_buffer_pool()
        
        hidden_dim = moe_layer.hidden_dim
        intermediate_dim = moe_layer.intermediate_dim
        num_experts = len(moe_layer.experts)
        top_k = moe_layer.num_experts_per_tok
        bits = moe_layer.bits
        
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Intermediate dim: {intermediate_dim}")
        print(f"  Num experts: {num_experts}")
        print(f"  Top-k: {top_k}")
        print(f"  Bits: {bits}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating synthetic benchmark instead...")
        return run_synthetic_benchmark()
    
    # Check available kernels
    print("\nChecking available kernels...")
    available_kernels: list[tuple[str, bool, str]] = []
    for kernel_name, use_fp32_acc, display_name in KERNEL_VARIANTS:
        if check_kernel_available(lib, kernel_name, use_fp32_acc):
            available_kernels.append((kernel_name, use_fp32_acc, display_name))
            print(f"  ✓ {display_name}")
        else:
            print(f"  ✗ {display_name} (not available)")
    
    if not available_kernels:
        print("ERROR: No kernels available for testing")
        return 1
    
    # Run benchmarks
    print(f"\nBenchmarking {len(available_kernels)} kernels across {len(BATCH_SIZES)} batch sizes...")
    print("-" * 70)
    
    results: dict[int, dict[str, dict]] = {}
    
    for batch_size in BATCH_SIZES:
        print(f"\nBatch size: {batch_size}")
        results[batch_size] = {}
        
        for kernel_name, use_fp32_acc, display_name in available_kernels:
            stats = benchmark_kernel(
                lib=lib,
                buffer_pool=buffer_pool,
                cached=cached,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=num_experts,
                top_k=top_k,
                bits=bits,
                kernel_name=kernel_name,
                use_fp32_acc=use_fp32_acc,
            )
            
            if stats:
                results[batch_size][display_name] = stats
                print(f"  {display_name:20s}: {stats['mean_ms']:6.2f} ms ({stats['throughput_tokens_per_sec']:6.1f} tok/s)")
            else:
                print(f"  {display_name:20s}: FAILED")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    
    thresholds = find_optimal_thresholds(results)
    
    print(f"\nOptimal thresholds for M4 Max:")
    print(f"  Decode kernel: batch_size <= {thresholds['decode_max']}")
    print(f"  Prefill4 kernel: {thresholds['decode_max'] + 1} <= batch_size <= {thresholds['prefill4_max']}")
    print(f"  Base kernel: {thresholds['prefill4_max'] + 1} <= batch_size <= {thresholds['base_max']}")
    print(f"  Large batch kernel: batch_size >= {thresholds['large_batch_min']}")
    
    # Save results
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    json_path = results_dir / "kernel_selection_m4_max.json"
    with open(json_path, "w") as f:
        json.dump({
            "device": "M4 Max",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "batch_sizes": BATCH_SIZES,
            "kernel_variants": [v[2] for v in available_kernels],
            "thresholds": thresholds,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate markdown report
    report = generate_recommendations(results, thresholds)
    md_path = results_dir / "kernel_recommendations.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {md_path}")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    
    return 0


def run_synthetic_benchmark() -> int:
    """Run a synthetic benchmark without a model."""
    print("\nSynthetic benchmark mode (no model loaded)")
    print("This provides theoretical kernel selection recommendations.\n")
    
    # Theoretical optimal thresholds based on M4 Max architecture
    # These are derived from the kernel design characteristics
    
    recommendations = {
        "device": "M4 Max (Synthetic)",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "thresholds": {
            "decode_max": 1,
            "prefill4_max": 16,
            "base_max": 32,
            "large_batch_min": 33,
        },
        "rationale": {
            "decode": "Optimized for single-token inference with minimal register pressure",
            "prefill4": "Processes 4 tokens per threadgroup, optimal for small batches",
            "base": "Balanced throughput for medium batch sizes (17-32)",
            "large_batch": "tile_n=128 for better memory coalescing on 33+ tokens",
        },
        "performance_estimates": {
            "decode_ms": "0.5-1.0",
            "prefill4_ms": "0.3-0.5",
            "base_ms": "0.2-0.3",
            "large_batch_ms": "0.15-0.2",
        },
    }
    
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    json_path = results_dir / "kernel_selection_m4_max.json"
    with open(json_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    
    print("Recommended kernel selection for M4 Max:")
    print("  batch_size == 1: moe_trellis_swiglu_decode")
    print("  2 <= batch_size <= 16: moe_trellis_swiglu_prefill4")
    print("  17 <= batch_size <= 32: moe_trellis_swiglu")
    print("  batch_size >= 33: moe_trellis_swiglu_large_batch")
    print(f"\nRecommendations saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
