"""
GEMM throughput benchmarks.

Measures kernel-level performance of Metal Marlin FP4 quantized GEMM across
standard LLM problem sizes, comparing against MLX native quantized and FP16
baselines. Uses the framework.Benchmark harness for statistical timing.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from metal_marlin import pack_fp4_weights, quantized_linear  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from framework import Benchmark  # noqa: E402

# Standard problem sizes from real models
SIZES: list[tuple[int, int, int, str]] = [
    # (M, N, K, description)
    (1, 4096, 4096, "Llama-7B q/k/v"),
    (1, 4096, 11008, "Llama-7B up_proj"),
    (1, 11008, 4096, "Llama-7B down_proj"),
    (1, 4096, 14336, "Llama-3-8B up_proj"),
    (1, 14336, 4096, "Llama-3-8B down_proj"),
    (1, 8192, 8192, "Llama-70B q/k/v"),
    (32, 4096, 4096, "Batch32 decode"),
    (128, 4096, 4096, "Prefill 128"),
    (512, 4096, 4096, "Prefill 512"),
    (2048, 4096, 4096, "Prefill 2048"),
]


def _create_fp4_weights(
    K: int, N: int, group_size: int = 128
) -> tuple[mx.array, mx.array]:
    """Create random FP4-packed weights and per-group scales.

    Generates a random FP16 weight matrix, quantizes to FP4 E2M1 via
    pack_fp4_weights, and returns the packed representation ready for
    the quantized GEMM kernel.
    """
    W = mx.random.normal((N, K), dtype=mx.float16)
    mx.eval(W)
    packed, scales = pack_fp4_weights(W, group_size=group_size)
    mx.eval(packed, scales)
    return packed, scales


def bench_marlin_fp4(
    warmup: int = 10,
    iterations: int = 100,
    group_size: int = 128,
) -> Benchmark:
    """Benchmark Marlin FP4 GEMM across standard LLM sizes."""
    bench = Benchmark(warmup=warmup, iterations=iterations)

    for M, N, K, desc in SIZES:
        A = mx.random.normal((M, K), dtype=mx.float16)
        B_packed, scales = _create_fp4_weights(K, N, group_size=group_size)
        mx.eval(A)

        def fn(a=A, b=B_packed, s=scales, gs=group_size):
            return quantized_linear(a, b, s, gs)

        result = bench.run(f"FP4 {desc}", fn, M, N, K)
        print(f"  {desc}: {result.mean_ms:.3f}ms ({result.tflops:.2f} TFLOPS)")

    return bench


def bench_mlx_quantized(
    warmup: int = 10,
    iterations: int = 100,
) -> Benchmark:
    """Benchmark MLX native 4-bit quantized matmul for comparison."""
    bench = Benchmark(warmup=warmup, iterations=iterations)

    for M, N, K, desc in SIZES:
        A = mx.random.normal((M, K), dtype=mx.float16)
        W = mx.random.normal((N, K), dtype=mx.float16)
        mx.eval(A, W)

        w_quant, w_scales, w_biases = mx.quantize(W, bits=4, group_size=64)
        mx.eval(w_quant, w_scales, w_biases)

        def fn(a=A, wq=w_quant, ws=w_scales, wb=w_biases):
            return mx.quantized_matmul(a, wq, ws, wb, bits=4, group_size=64)

        result = bench.run(f"MLX4b {desc}", fn, M, N, K)
        print(f"  {desc}: {result.mean_ms:.3f}ms ({result.tflops:.2f} TFLOPS)")

    return bench


def bench_fp16_baseline(
    warmup: int = 10,
    iterations: int = 100,
) -> Benchmark:
    """Benchmark FP16 matmul as throughput ceiling reference."""
    bench = Benchmark(warmup=warmup, iterations=iterations)

    for M, N, K, desc in SIZES:
        A = mx.random.normal((M, K), dtype=mx.float16)
        B = mx.random.normal((K, N), dtype=mx.float16)
        mx.eval(A, B)

        def fn(a=A, b=B):
            return a @ b

        result = bench.run(f"FP16 {desc}", fn, M, N, K)
        print(f"  {desc}: {result.mean_ms:.3f}ms ({result.tflops:.2f} TFLOPS)")

    return bench


def bench_comparison(
    warmup: int = 10,
    iterations: int = 100,
    group_size: int = 128,
) -> None:
    """Run all backends and print a unified comparison table."""
    bench = Benchmark(warmup=warmup, iterations=iterations)

    for M, N, K, desc in SIZES[:5]:  # Subset for quick comparison
        A = mx.random.normal((M, K), dtype=mx.float16)
        mx.eval(A)

        # Marlin FP4
        B_packed, scales = _create_fp4_weights(K, N, group_size=group_size)

        def marlin_fn(a=A, b=B_packed, s=scales, gs=group_size):
            return quantized_linear(a, b, s, gs)

        bench.run(f"Marlin FP4 | {desc}", marlin_fn, M, N, K)

        # MLX native 4-bit
        W = mx.random.normal((N, K), dtype=mx.float16)
        mx.eval(W)
        w_quant, w_scales, w_biases = mx.quantize(W, bits=4, group_size=64)
        mx.eval(w_quant, w_scales, w_biases)

        def mlx4_fn(a=A, wq=w_quant, ws=w_scales, wb=w_biases):
            return mx.quantized_matmul(a, wq, ws, wb, bits=4, group_size=64)

        bench.run(f"MLX 4bit   | {desc}", mlx4_fn, M, N, K)

        # FP16 baseline
        B_fp16 = mx.random.normal((K, N), dtype=mx.float16)
        mx.eval(B_fp16)

        def fp16_fn(a=A, b=B_fp16):
            return a @ b

        bench.run(f"FP16       | {desc}", fp16_fn, M, N, K)

    bench.print_summary()


def main() -> None:
    """Run all GEMM benchmarks and export results."""
    if not HAS_MLX:
        print("ERROR: Benchmarks require MLX for Metal GPU access.")
        print("Install with: pip install mlx")
        sys.exit(1)

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Metal Marlin FP4 GEMM Throughput Benchmark")
    print("=" * 70)

    print("\n--- Marlin FP4 ---")
    fp4_bench = bench_marlin_fp4()
    fp4_bench.export_json(results_dir / "marlin_fp4.json")
    fp4_bench.export_csv(results_dir / "marlin_fp4.csv")

    print("\n--- MLX Native 4-bit ---")
    mlx_bench = bench_mlx_quantized()
    mlx_bench.export_json(results_dir / "mlx_4bit.json")
    mlx_bench.export_csv(results_dir / "mlx_4bit.csv")

    print("\n--- FP16 Baseline ---")
    fp16_bench = bench_fp16_baseline()
    fp16_bench.export_json(results_dir / "fp16_baseline.json")
    fp16_bench.export_csv(results_dir / "fp16_baseline.csv")

    print("\n--- Head-to-Head Comparison ---")
    bench_comparison()

    print(f"\nResults exported to {results_dir}/")


if __name__ == "__main__":
    main()
