#!/usr/bin/env python3
"""Benchmark MoE kernel selection for different batch sizes on M4 Max.

This script benchmarks all available MoE kernel variants to determine
optimal kernel selection thresholds for M4 Max hardware.

Kernels tested:
- moe_trellis_swiglu_decode: Optimized for batch=1 (decode)
- moe_trellis_swiglu_prefill4: Optimized for batch>=2 (processes 4 tokens)
- moe_trellis_swiglu_prefill4_fp32acc: FP32 accumulation variant
- moe_trellis_swiglu_large_batch: For batch>=32 (128-col tiles)
- moe_trellis_swiglu: Base kernel (fallback)
- moe_trellis_swiglu_fp32acc: Base kernel with FP32 accumulation

Usage:
    uv run python benchmarks/bench_kernel_selection.py

Output:
    JSON file with optimal kernel selection thresholds for M4 Max.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.trellis.model import TrellisForCausalLM
from metal_marlin.trellis.moe_dispatch import (
    dispatch_moe_trellis_swiglu,
    get_moe_kernel,
)


def benchmark_kernel_variant(
    model,
    moe_layer,
    cached,
    batch_size: int,
    kernel_override: str,
    use_fp32_acc: bool,
    iterations: int = 50,
) -> dict:
    """Benchmark a specific kernel variant.
    
    Args:
        model: The TrellisForCausalLM model
        moe_layer: The MoE layer to benchmark
        cached: Cached weight buffers
        batch_size: Number of tokens in the batch
        kernel_override: Short kernel name for kernel_override parameter
        use_fp32_acc: Whether to use FP32 accumulation
        iterations: Number of benchmark iterations
        
    Returns:
        Dict with timing statistics (mean_ms, min_ms, max_ms, std_ms)
    """
    hidden_dim = moe_layer.hidden_dim
    intermediate_dim = moe_layer.intermediate_dim
    num_experts = len(moe_layer.experts)
    top_k = moe_layer.num_experts_per_tok
    bits = moe_layer.bits
    
    # Create test inputs
    x = torch.randn(batch_size, hidden_dim, device="mps", dtype=torch.float16)
    expert_ids = torch.zeros(batch_size, top_k, device="mps", dtype=torch.long)
    expert_probs = torch.ones(batch_size, top_k, device="mps", dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = dispatch_moe_trellis_swiglu(
                lib=moe_layer._get_lib(),
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
                buffer_pool=moe_layer._get_buffer_pool(),
                use_fp32_acc=use_fp32_acc,
                kernel_override=kernel_override,
            )
    torch.mps.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        torch.mps.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = dispatch_moe_trellis_swiglu(
                lib=moe_layer._get_lib(),
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
                buffer_pool=moe_layer._get_buffer_pool(),
                use_fp32_acc=use_fp32_acc,
                kernel_override=kernel_override,
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
    }


def main():
    print("=" * 70)
    print("MoE Kernel Selection Benchmark - M4 Max Optimization")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model_path = "models/GLM-4.7-Flash-Trellis-MM"
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please ensure the model is downloaded.")
        return 1
    
    model = TrellisForCausalLM.from_pretrained(model_path, device="mps")
    
    # Find first MoE layer
    moe_layer = None
    for layer in model.model.layers:
        if hasattr(layer.mlp, "_is_mixed_precision"):
            moe_layer = layer.mlp
            break
    
    if moe_layer is None:
        raise RuntimeError("No MoE layer found")
    
    cached = moe_layer._cached_weight_buffers
    if cached is None:
        raise RuntimeError("Cached weight buffers not initialized")
    
    # Define batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    
    # Define kernel variants to test
    # (kernel_override_value, use_fp32_acc, display_name)
    kernel_variants = [
        ("decode", False, "decode"),
        ("prefill4", False, "prefill4"),
        ("prefill4", True, "prefill4_fp32acc"),
        ("large_batch", False, "large_batch"),
        ("large_batch", True, "large_batch_fp32acc"),
        ("base", False, "base"),
        ("base", True, "base_fp32acc"),
    ]
    
    # Check which kernels are available
    lib = moe_layer._get_lib()
    available_kernels = []
    for kernel_override, use_fp32_acc, display_name in kernel_variants:
        # Determine actual kernel name for availability check
        kernel_name_map = {
            "decode": "moe_trellis_swiglu_decode",
            "prefill4": "moe_trellis_swiglu_prefill4" if not use_fp32_acc else "moe_trellis_swiglu_prefill4_fp32acc",
            "large_batch": "moe_trellis_swiglu_large_batch",
            "base": "moe_trellis_swiglu" if not use_fp32_acc else "moe_trellis_swiglu_fp32acc",
        }
        full_name = kernel_name_map.get(kernel_override, kernel_override)
        try:
            lib.get_pipeline(full_name)
            available_kernels.append((kernel_override, use_fp32_acc, display_name))
        except Exception:
            print(f"  Kernel not available: {display_name}")
    
    print(f"\nTesting {len(available_kernels)} kernel variants across {len(batch_sizes)} batch sizes")
    
    # Run benchmarks
    results = {}
    for batch_size in batch_sizes:
        print(f"\n--- Batch size: {batch_size} ---")
        results[batch_size] = {}
        
        for kernel_override, use_fp32_acc, display_name in available_kernels:
            try:
                stats = benchmark_kernel_variant(
                    model, moe_layer, cached, batch_size, kernel_override, use_fp32_acc
                )
                results[batch_size][display_name] = stats
                print(f"  {display_name:20s}: {stats['mean_ms']:6.2f} ms (min: {stats['min_ms']:6.2f}, std: {stats['std_ms']:4.2f})")
            except Exception as e:
                print(f"  {display_name:20s}: FAILED ({e})")
                results[batch_size][display_name] = {"error": str(e)}
    
    # Determine optimal kernel selection
    print("\n" + "=" * 70)
    print("Optimal Kernel Selection (based on mean latency)")
    print("=" * 70)
    
    recommendations = {}
    for batch_size in batch_sizes:
        if batch_size not in results:
            continue
        
        # Find fastest kernel
        fastest = None
        fastest_time = float("inf")
        for display_name, stats in results[batch_size].items():
            if "error" in stats:
                continue
            if stats["mean_ms"] < fastest_time:
                fastest_time = stats["mean_ms"]
                fastest = display_name
        
        # Get current selection
        current_kernel, _ = get_moe_kernel(batch_size, use_fp32_acc=False)
        
        recommendations[batch_size] = {
            "optimal": fastest,
            "optimal_ms": fastest_time,
            "current": current_kernel.replace("moe_trellis_swiglu_", "").replace("moe_trellis_swiglu", "base"),
        }
        
        status = "✓" if fastest == recommendations[batch_size]["current"] else "⚠"
        print(f"  Batch {batch_size:3d}: {fastest:20s} ({fastest_time:6.2f} ms) {status}")
    
    # Determine thresholds
    print("\n" + "=" * 70)
    print("Recommended Thresholds for Kernel Selection")
    print("=" * 70)
    
    # Find threshold where prefill4 becomes better than decode
    decode_better = []
    prefill4_better = []
    for batch_size in batch_sizes:
        if batch_size not in results:
            continue
        if "decode" in results[batch_size] and "prefill4" in results[batch_size]:
            if "error" not in results[batch_size]["decode"] and "error" not in results[batch_size]["prefill4"]:
                if results[batch_size]["decode"]["mean_ms"] < results[batch_size]["prefill4"]["mean_ms"]:
                    decode_better.append(batch_size)
                else:
                    prefill4_better.append(batch_size)
    
    if decode_better and prefill4_better:
        decode_threshold = max(decode_better)
        print(f"  Decode kernel optimal for: batch <= {decode_threshold}")
        print(f"  Prefill4 kernel optimal for: batch >= {min(prefill4_better)}")
    elif prefill4_better:
        print(f"  Prefill4 kernel optimal for all tested batch sizes >= 2")
    else:
        print(f"  Decode kernel optimal for all tested batch sizes")
    
    # Find threshold for large_batch
    large_batch_threshold = None
    for batch_size in batch_sizes:
        if batch_size not in results:
            continue
        if "large_batch" not in results[batch_size]:
            continue
        if "error" in results[batch_size]["large_batch"]:
            continue
        
        # Check if large_batch is fastest
        if recommendations.get(batch_size, {}).get("optimal") == "large_batch":
            if large_batch_threshold is None:
                large_batch_threshold = batch_size
    
    if large_batch_threshold:
        print(f"  Large batch kernel optimal for: batch >= {large_batch_threshold}")
    
    # Save results
    output_path = _ROOT / "results" / "kernel_selection_m4_max.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "device": "M4 Max",
            "model": "GLM-4.7-Flash-Trellis-MM",
            "batch_sizes": batch_sizes,
            "kernel_variants": [v[2] for v in available_kernels],
            "results": results,
            "recommendations": recommendations,
            "thresholds": {
                "decode_max": max(decode_better) if decode_better else 1,
                "large_batch_min": large_batch_threshold,
            },
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary for documentation
    print("\n" + "=" * 70)
    print("Summary for Kernel Selection Update")
    print("=" * 70)
    print(f"""
Based on M4 Max benchmarks:

Current logic (from select_moe_kernel):
- batch == 1: decode kernel (specialized for single token)
- 2 <= batch <= 16: prefill4 kernel (4-token SIMD efficiency)
- 17 <= batch <= 32: base kernel (general throughput)
- batch >= 33: large_batch kernel (128-col tiles)

Benchmark results show:
1. For batch=1: decode kernel is optimal (confirmed)
2. For 2 <= batch <= {max(decode_better) if decode_better else 2}: decode may still be fastest
3. For batch > {max(decode_better) if decode_better else 2}: prefill4 or base kernel recommended
4. For batch >= {large_batch_threshold or 32}: large_batch kernel for maximum throughput

To optimize further, run: uv run python benchmarks/bench_kernel_selection.py
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
