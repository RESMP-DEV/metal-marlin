#!/usr/bin/env python3
"""Benchmark after MoE optimizations.

Usage:
    python scripts/benchmark_optimized.py          # Quick validation (default)
    python scripts/benchmark_optimized.py --full   # Full benchmark with model loading
"""

import argparse


def run_full_benchmark():
    """Run the full benchmark with model loading and inference."""
    import time

    import torch

    from metal_marlin.trellis.model import TrellisForCausalLM

    print("Loading model...")
    model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')
    model.eval()

    x = torch.tensor([[1]], device='mps')

    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)
    torch.mps.synchronize()

    # Benchmark
    print("Benchmarking...")
    times = []
    for _ in range(10):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000

    print("\nResults:")
    print(f"  Average: {avg_ms:.1f} ms ({1000/avg_ms:.2f} tok/s)")
    print(f"  Min: {min_ms:.1f} ms ({1000/min_ms:.2f} tok/s)")
    print(f"  Max: {max_ms:.1f} ms")

    # Comparison
    baseline_ms = 17249  # From earlier measurement
    print("\nComparison to baseline:")
    print(f"  Baseline: {baseline_ms:.0f} ms (0.058 tok/s)")
    print(f"  Current: {avg_ms:.1f} ms ({1000/avg_ms:.2f} tok/s)")
    print(f"  Speedup: {baseline_ms/avg_ms:.2f}x")

    target_ms = 50  # 20 tok/s
    if avg_ms <= target_ms:
        print(f"\n✓ TARGET ACHIEVED: {1000/avg_ms:.1f} tok/s >= 20 tok/s")
    else:
        print(f"\n✗ Target not met: need {avg_ms/target_ms:.1f}x more speedup")


def run_quick_validation():
    """Quick validation: check imports and basic functionality."""
    print("Running quick validation...")

    # Check core imports
    import torch

    from metal_marlin.trellis.model import TrellisForCausalLM
    from metal_marlin.trellis.moe_dispatch import dispatch_moe_trellis_swiglu

    # Verify MPS availability
    if not torch.backends.mps.is_available():
        print("Warning: MPS not available, benchmark would use CPU")
    else:
        print("  MPS backend: available")

    # Verify model class exists and has expected attributes
    assert hasattr(TrellisForCausalLM, 'from_pretrained'), "Missing from_pretrained method"
    print("  TrellisForCausalLM: OK")

    # Verify dispatch function signature
    import inspect
    sig = inspect.signature(dispatch_moe_trellis_swiglu)
    params = list(sig.parameters.keys())
    assert 'activations' in params, "Missing activations parameter"
    assert 'expert_ids' in params, "Missing expert_ids parameter"
    print("  dispatch_moe_trellis_swiglu: OK")

    print("\n✓ Quick validation passed - all imports and signatures OK")
    print("  Run with --full to execute full benchmark with model loading")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark MoE optimizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run full benchmark with model loading (takes several minutes)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='[Deprecated] Same as default behavior'
    )
    args = parser.parse_args()

    if args.full:
        run_full_benchmark()
    else:
        run_quick_validation()

if __name__ == '__main__':
    main()
