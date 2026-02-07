#!/usr/bin/env python3
"""Comprehensive MoE kernel selection profiler for M4 Max.

This script benchmarks ALL MoE kernel variants across different batch sizes,
bit-width configurations, and FP32 accumulation settings to determine the
optimal kernel selection strategy for M4 Max.

Kernels tested:
- moe_trellis_swiglu_decode: Batch=1 optimized (various bit-width specializations)
- moe_trellis_swiglu_prefill4: Small batch (2-16 tokens)
- moe_trellis_swiglu: Medium batch (base kernel)
- moe_trellis_swiglu_large_batch: Large batch (33+ tokens, tile_n=128)
- All FP32 accumulation variants

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/bench_moe_kernel_selection_profile.py

Output:
    - results/moe_kernel_profile_m4_max.json: Raw benchmark data
    - results/kernel_selection_recommendations.md: Human-readable recommendations
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.mps

# Setup paths
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.trellis.moe_dispatch import (
    dispatch_moe_trellis_swiglu,
    get_moe_kernel,
    select_moe_kernel,
)


@dataclass
class BenchmarkResult:
    """Result of a single kernel benchmark."""
    batch_size: int
    kernel_name: str
    display_name: str
    use_fp32_acc: bool
    gate_bits: int | None
    up_bits: int | None
    down_bits: int | None
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    tokens_per_sec: float


def create_mock_weights(
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    bits: int | tuple[int, int, int],
    device: str = "mps"
) -> dict[str, Any]:
    """Create mock weight tensors for benchmarking.
    
    Creates properly shaped tensors without actual data (zeros)
    since we're only benchmarking kernel dispatch overhead and
    compute throughput, not numerical correctness.
    """
    if isinstance(bits, int):
        gate_bits = up_bits = down_bits = bits
    else:
        gate_bits, up_bits, down_bits = bits
    
    # Calculate packed dimensions for trellis-quantized weights
    # Each weight matrix: (intermediate_dim, hidden_dim) for gate/up
    #                     (hidden_dim, intermediate_dim) for down
    
    def calc_packed_dim(dim: int, bits: int) -> int:
        """Calculate packed dimension for trellis quantization."""
        # Trellis packs 32/bits values per uint32
        values_per_uint32 = 32 // bits
        return (dim + values_per_uint32 - 1) // values_per_uint32
    
    # Gate and Up: (num_experts, intermediate_dim, hidden_dim)
    gate_packed = calc_packed_dim(hidden_dim, gate_bits)
    up_packed = calc_packed_dim(hidden_dim, up_bits)
    
    # Down: (num_experts, hidden_dim, intermediate_dim)
    down_packed = calc_packed_dim(intermediate_dim, down_bits)
    
    weights = {
        # Trellis-packed weights (uint32)
        "gate_weights": torch.zeros(
            num_experts, intermediate_dim, gate_packed,
            dtype=torch.uint32, device=device
        ),
        "up_weights": torch.zeros(
            num_experts, intermediate_dim, up_packed,
            dtype=torch.uint32, device=device
        ),
        "down_weights": torch.zeros(
            num_experts, hidden_dim, down_packed,
            dtype=torch.uint32, device=device
        ),
        # Scales (fp16)
        "gate_scales": torch.zeros(
            num_experts, intermediate_dim, 2,
            dtype=torch.float16, device=device
        ),
        "up_scales": torch.zeros(
            num_experts, intermediate_dim, 2,
            dtype=torch.float16, device=device
        ),
        "down_scales": torch.zeros(
            num_experts, hidden_dim, 2,
            dtype=torch.float16, device=device
        ),
        # SU/SV vectors for Hadamard (fp16)
        "gate_su": torch.ones(
            hidden_dim, dtype=torch.float16, device=device
        ),
        "gate_sv": torch.ones(
            intermediate_dim, dtype=torch.float16, device=device
        ),
        "up_su": torch.ones(
            hidden_dim, dtype=torch.float16, device=device
        ),
        "up_sv": torch.ones(
            intermediate_dim, dtype=torch.float16, device=device
        ),
        "down_su": torch.ones(
            intermediate_dim, dtype=torch.float16, device=device
        ),
        "down_sv": torch.ones(
            hidden_dim, dtype=torch.float16, device=device
        ),
    }
    
    return weights


def benchmark_kernel(
    lib: Any,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    bits: tuple[int, int, int],
    kernel_name: str,
    use_fp32_acc: bool,
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark a single kernel configuration."""
    
    gate_bits, up_bits, down_bits = bits
    device = "mps"
    
    # Create test inputs
    x = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)
    expert_ids = torch.randint(0, num_experts, (batch_size, top_k), device=device)
    expert_probs = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)
    
    # Create mock weights
    weights = create_mock_weights(hidden_dim, intermediate_dim, num_experts, bits, device)
    
    # Warmup
    for _ in range(warmup):
        try:
            _ = dispatch_moe_trellis_swiglu(
                lib=lib,
                activations=x,
                expert_ids=expert_ids,
                expert_probs=expert_probs,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=num_experts,
                top_k=top_k,
                bits=bits,
                use_fp32_acc=use_fp32_acc,
                kernel_name_override=kernel_name,
                **weights,
            )
            torch.mps.synchronize()
        except Exception as e:
            # Kernel not available
            raise RuntimeError(f"Kernel {kernel_name} not available: {e}")
    
    # Benchmark
    times = []
    for _ in range(iterations):
        torch.mps.synchronize()
        start = time.perf_counter()
        
        _ = dispatch_moe_trellis_swiglu(
            lib=lib,
            activations=x,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            bits=bits,
            use_fp32_acc=use_fp32_acc,
            kernel_name_override=kernel_name,
            **weights,
        )
        torch.mps.synchronize()
        
        times.append(time.perf_counter() - start)
    
    mean_s = sum(times) / len(times)
    mean_ms = mean_s * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    tokens_per_sec = batch_size / mean_s
    
    display_name = kernel_name.replace("moe_trellis_swiglu_", "").replace("moe_trellis_swiglu", "base")
    
    return BenchmarkResult(
        batch_size=batch_size,
        kernel_name=kernel_name,
        display_name=display_name,
        use_fp32_acc=use_fp32_acc,
        gate_bits=gate_bits,
        up_bits=up_bits,
        down_bits=down_bits,
        mean_ms=mean_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
        tokens_per_sec=tokens_per_sec,
    )


def get_available_kernels(lib: Any) -> list[tuple[str, str]]:
    """Get list of available kernels from the Metal library."""
    
    kernel_variants = [
        "moe_trellis_swiglu_decode",
        "moe_trellis_swiglu_decode_6_2_3",
        "moe_trellis_swiglu_decode_6_3_4",
        "moe_trellis_swiglu_decode_6_2_4",
        "moe_trellis_swiglu_prefill4",
        "moe_trellis_swiglu_prefill4_fp32acc",
        "moe_trellis_swiglu",
        "moe_trellis_swiglu_fp32acc",
        "moe_trellis_swiglu_large_batch",
        "moe_trellis_swiglu_simd",
    ]
    
    available = []
    for kernel in kernel_variants:
        try:
            lib.get_pipeline(kernel)
            display_name = kernel.replace("moe_trellis_swiglu_", "").replace("moe_trellis_swiglu", "base")
            available.append((kernel, display_name))
        except Exception:
            pass
    
    return available


def run_benchmarks(
    lib: Any,
    batch_sizes: list[int] | None = None,
    hidden_dim: int = 4608,
    intermediate_dim: int = 2304,
    num_experts: int = 256,
    top_k: int = 8,
) -> list[BenchmarkResult]:
    """Run comprehensive benchmarks across all configurations."""
    
    if batch_sizes is None:
        # Test decode (1), small prefill (2-16), medium (17-32), large (33+)
        batch_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    
    # Common bit-width configurations for Trellis
    bit_configs = [
        (6, 2, 3),  # GLM-4.7-Flash default
        (6, 3, 4),
        (6, 2, 4),
        (4, 4, 4),  # Uniform 4-bit
        (8, 8, 8),  # Uniform 8-bit (reference)
    ]
    
    available_kernels = get_available_kernels(lib)
    print(f"Found {len(available_kernels)} available kernels:")
    for kernel, display in available_kernels:
        print(f"  - {display}")
    
    results = []
    total_tests = len(batch_sizes) * len(bit_configs) * len(available_kernels) * 2  # ×2 for fp32acc
    test_num = 0
    
    for batch_size in batch_sizes:
        for bits in bit_configs:
            for kernel_name, display_name in available_kernels:
                for use_fp32_acc in [False, True]:
                    test_num += 1
                    
                    # Skip fp32acc for decode kernels (not available)
                    if "decode" in kernel_name and use_fp32_acc:
                        continue
                    
                    # Skip fp32acc variants for kernels that already have it in name
                    if "fp32acc" in kernel_name and use_fp32_acc:
                        continue
                    
                    print(f"\n[{test_num}/{total_tests}] "
                          f"batch={batch_size}, kernel={display_name}, "
                          f"bits={bits}, fp32acc={use_fp32_acc}")
                    
                    try:
                        result = benchmark_kernel(
                            lib=lib,
                            batch_size=batch_size,
                            hidden_dim=hidden_dim,
                            intermediate_dim=intermediate_dim,
                            num_experts=num_experts,
                            top_k=top_k,
                            bits=bits,
                            kernel_name=kernel_name,
                            use_fp32_acc=use_fp32_acc,
                            iterations=50,
                            warmup=5,
                        )
                        results.append(result)
                        print(f"  ✓ {result.mean_ms:.3f} ms "
                              f"({result.tokens_per_sec:.1f} tok/s)")
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
    
    return results


def analyze_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Analyze benchmark results and generate recommendations."""
    
    # Group by batch size
    by_batch: dict[int, list[BenchmarkResult]] = {}
    for r in results:
        by_batch.setdefault(r.batch_size, []).append(r)
    
    # Find optimal kernel for each batch size
    optimal_selections = {}
    for batch_size, batch_results in sorted(by_batch.items()):
        # Find fastest
        fastest = min(batch_results, key=lambda r: r.mean_ms)
        optimal_selections[batch_size] = {
            "kernel": fastest.kernel_name,
            "display_name": fastest.display_name,
            "fp32acc": fastest.use_fp32_acc,
            "mean_ms": fastest.mean_ms,
            "tokens_per_sec": fastest.tokens_per_sec,
        }
    
    # Determine thresholds
    # Find where prefill4 becomes better than decode
    decode_batches = []
    prefill4_batches = []
    
    for batch_size in sorted(by_batch.keys()):
        batch_results = by_batch[batch_size]
        
        decode_results = [r for r in batch_results if "decode" in r.kernel_name]
        prefill4_results = [r for r in batch_results if "prefill4" in r.kernel_name]
        
        if decode_results and prefill4_results:
            best_decode = min(decode_results, key=lambda r: r.mean_ms)
            best_prefill4 = min(prefill4_results, key=lambda r: r.mean_ms)
            
            if best_decode.mean_ms < best_prefill4.mean_ms:
                decode_batches.append(batch_size)
            else:
                prefill4_batches.append(batch_size)
    
    # Find where large_batch becomes optimal
    large_batch_threshold = None
    for batch_size in sorted(by_batch.keys()):
        optimal = optimal_selections[batch_size]["kernel"]
        if "large_batch" in optimal and large_batch_threshold is None:
            large_batch_threshold = batch_size
    
    # Compare with current logic
    current_logic_comparison = {}
    for batch_size in sorted(by_batch.keys()):
        current_kernel, _ = select_moe_kernel(batch_size, use_fp32_acc=False)
        optimal = optimal_selections[batch_size]["kernel"]
        
        current_results = [r for r in by_batch[batch_size] if r.kernel_name == current_kernel]
        optimal_results = [r for r in by_batch[batch_size] if r.kernel_name == optimal]
        
        if current_results and optimal_results:
            current_best = min(current_results, key=lambda r: r.mean_ms)
            optimal_best = min(optimal_results, key=lambda r: r.mean_ms)
            
            speedup = current_best.mean_ms / optimal_best.mean_ms
            current_logic_comparison[batch_size] = {
                "current_kernel": current_kernel,
                "optimal_kernel": optimal,
                "current_ms": current_best.mean_ms,
                "optimal_ms": optimal_best.mean_ms,
                "speedup": speedup,
                "is_optimal": current_kernel == optimal,
            }
    
    return {
        "optimal_selections": optimal_selections,
        "thresholds": {
            "decode_max": max(decode_batches) if decode_batches else 1,
            "prefill4_min": min(prefill4_batches) if prefill4_batches else 2,
            "large_batch_min": large_batch_threshold,
        },
        "current_logic_comparison": current_logic_comparison,
    }


def generate_recommendations(analysis: dict[str, Any]) -> str:
    """Generate human-readable recommendations."""
    
    lines = [
        "# MoE Kernel Selection Recommendations for M4 Max",
        "",
        "Based on comprehensive profiling of all kernel variants.",
        "",
        "## Optimal Kernel Selection by Batch Size",
        "",
        "| Batch Size | Optimal Kernel | Latency (ms) | Throughput (tok/s) |",
        "|------------|----------------|--------------|-------------------|",
    ]
    
    for batch_size, info in sorted(analysis["optimal_selections"].items()):
        lines.append(
            f"| {batch_size:10d} | {info['display_name']:14s} | "
            f"{info['mean_ms']:12.3f} | {info['tokens_per_sec']:17.1f} |"
        )
    
    lines.extend([
        "",
        "## Recommended Thresholds",
        "",
        f"- **Decode kernel**: batch_size <= {analysis['thresholds']['decode_max']}",
        f"- **Prefill4 kernel**: {analysis['thresholds']['prefill4_min']} <= batch_size < {analysis['thresholds']['large_batch_min'] or 33}",
        f"- **Large batch kernel**: batch_size >= {analysis['thresholds']['large_batch_min'] or 33}",
        "",
        "## Comparison with Current Logic",
        "",
        "| Batch | Current Kernel | Optimal Kernel | Current (ms) | Optimal (ms) | Speedup | Status |",
        "|-------|----------------|----------------|--------------|--------------|---------|--------|",
    ])
    
    for batch_size, info in sorted(analysis["current_logic_comparison"].items()):
        status = "✓" if info["is_optimal"] else f"⚠ {info['speedup']:.2f}x"
        lines.append(
            f"| {batch_size:5d} | {info['current_kernel'].replace('moe_trellis_swiglu_', '').replace('moe_trellis_swiglu', 'base'):14s} | "
            f"{info['optimal_kernel'].replace('moe_trellis_swiglu_', '').replace('moe_trellis_swiglu', 'base'):14s} | "
            f"{info['current_ms']:12.3f} | {info['optimal_ms']:12.3f} | "
            f"{info['speedup']:7.2f}x | {status:6s} |"
        )
    
    lines.extend([
        "",
        "## Implementation Notes",
        "",
        "1. **Decode kernels**: Specialized for batch=1 with compile-time known dequant params",
        "2. **Bit-width specializations**: For (6,2,3), (6,3,4), (6,2,4) configs, use specialized decode kernels",
        "3. **FP32 accumulation**: Use when hidden_dim >= 1024 for better numerical stability",
        "4. **Large batch kernel**: Uses tile_n=128 for better memory coalescing on M4 Max",
        "",
        "## Suggested Code Update",
        "",
        "```python",
        "def select_moe_kernel(",
        "    batch_size: int,",
        "    use_fp32_acc: bool,",
        "    gate_bits: int | None = None,",
        "    up_bits: int | None = None,",
        "    down_bits: int | None = None,",
        ") -> tuple[str, int]:",
        "    # Based on M4 Max profiling",
        f"    if batch_size <= {analysis['thresholds']['decode_max']}:",
        "        # Decode path with bit-width specializations",
        "        if not use_fp32_acc:",
        "            if gate_bits == 6 and up_bits == 2 and down_bits == 3:",
        "                return \"moe_trellis_swiglu_decode_6_2_3\", 64",
        "            # ... other specializations",
        "        return \"moe_trellis_swiglu_decode\", 64",
        f"    elif batch_size < {analysis['thresholds']['large_batch_min'] or 33}:",
        "        # Small to medium batches",
        "        if use_fp32_acc:",
        "            return \"moe_trellis_swiglu_prefill4_fp32acc\", 64",
        "        return \"moe_trellis_swiglu_prefill4\", 64",
        "    else:",
        "        # Large batches",
        f"        if batch_size >= {analysis['thresholds']['large_batch_min'] or 33}:",
        "            if use_fp32_acc:",
        "                return \"moe_trellis_swiglu_fp32acc\", 64",
        "            return \"moe_trellis_swiglu_large_batch\", 128",
        "        if use_fp32_acc:",
        "            return \"moe_trellis_swiglu_fp32acc\", 64",
        "        return \"moe_trellis_swiglu\", 64",
        "```",
    ])
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    
    print("=" * 70)
    print("MoE Kernel Selection Profiler - M4 Max Optimization")
    print("=" * 70)
    
    # Import Metal library
    try:
        from metal_marlin.metal_dispatch import MetalKernelLibrary, HAS_METAL
        if not HAS_METAL:
            print("ERROR: Metal not available on this system")
            return 1
    except ImportError as e:
        print(f"ERROR: Could not import MetalKernelLibrary: {e}")
        return 1
    
    # Initialize library
    print("\nInitializing Metal kernel library...")
    lib = MetalKernelLibrary()
    
    # Run benchmarks
    print("\nRunning comprehensive kernel benchmarks...")
    print("This may take several minutes...")
    
    results = run_benchmarks(
        lib=lib,
        batch_sizes=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64],
        hidden_dim=4608,
        intermediate_dim=2304,
        num_experts=256,
        top_k=8,
    )
    
    if not results:
        print("ERROR: No benchmark results collected")
        return 1
    
    print(f"\n\nCollected {len(results)} benchmark results")
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results)
    
    # Create results directory
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save raw results
    json_path = results_dir / "moe_kernel_profile_m4_max.json"
    with open(json_path, "w") as f:
        json.dump({
            "device": "M4 Max",
            "results": [asdict(r) for r in results],
            "analysis": analysis,
        }, f, indent=2)
    print(f"\nRaw results saved to: {json_path}")
    
    # Generate and save recommendations
    recommendations = generate_recommendations(analysis)
    md_path = results_dir / "kernel_selection_recommendations.md"
    with open(md_path, "w") as f:
        f.write(recommendations)
    print(f"Recommendations saved to: {md_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nOptimal thresholds for M4 Max:")
    print(f"  - Decode (batch <= {analysis['thresholds']['decode_max']})")
    print(f"  - Prefill4 ({analysis['thresholds']['prefill4_min']} <= batch < {analysis['thresholds']['large_batch_min'] or 33})")
    print(f"  - Large batch (batch >= {analysis['thresholds']['large_batch_min'] or 33})")
    
    # Count optimizations
    suboptimal = sum(1 for info in analysis["current_logic_comparison"].values() if not info["is_optimal"])
    total = len(analysis["current_logic_comparison"])
    print(f"\nCurrent logic is optimal for {total - suboptimal}/{total} batch sizes")
    if suboptimal > 0:
        print(f"  {suboptimal} batch sizes could benefit from optimization")
    
    print("\n" + "=" * 70)
    print("Done! Review the recommendations file for detailed analysis.")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
