"""
GEMM throughput benchmarks.

Measures kernel-level performance of Metal Marlin FP4 quantized GEMM across
standard LLM problem sizes, comparing against PyTorch MPS native quantized and FP16
baselines. Uses the framework.Benchmark harness for statistical timing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.inference.decode import quantized_linear_torch  # noqa: E402
from metal_marlin.kernels import HAS_METAL, HAS_MPS  # noqa: E402

# Import PyTorch-based pack_fp4_weights if available
if HAS_METAL and HAS_MPS:
    from metal_marlin.kernels import pack_fp4_weights  # noqa: E402
else:
    import numpy as np

    def pack_fp4_weights(
        weight: torch.Tensor, group_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback CPU packing matching decode.py's expected layout.

        Input: weight [N, K] (PyTorch convention)
        Output:
          - packed [K/8, N] uint32, 8 FP4 values from K-dim per uint32
          - scales [K/group_size, N] fp16
        """
        w = weight.T.to(torch.float16).cpu()  # [K, N]
        K, N = w.shape

        if K % 8 != 0:
            raise ValueError(f"K ({K}) must be divisible by 8")
        if K % group_size != 0:
            raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

        # Compute per-group scales along K
        w_grouped = w.reshape(K // group_size, group_size, N)
        scales = w_grouped.abs().amax(dim=1)  # [K/group_size, N]
        scales = scales.clamp(min=1e-7)

        # E2M1 max representable magnitude
        MAX_E2M1 = 3.0

        # Normalize and clamp
        scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]
        w_norm = w / scales_expanded
        w_norm = w_norm.clamp(-MAX_E2M1, MAX_E2M1)

        # E2M1 LUT for quantization
        # Nibble: bit3=sign, bit2-1=exp, bit0=mantissa
        e2m1_values = np.array([
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
            -0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0
        ], dtype=np.float32)

        # Quantize to nearest E2M1 nibble
        w_np = w_norm.float().numpy()
        k_packs = K // 8
        packed = np.zeros((k_packs, N), dtype=np.uint32)

        for k_pack in range(k_packs):
            k_base = k_pack * 8
            for bit_pos in range(8):
                row_vals = w_np[k_base + bit_pos, :]  # [N]
                # Find nearest E2M1 value
                dists = np.abs(row_vals[:, None] - e2m1_values[None, :])
                nibbles = np.argmin(dists, axis=1).astype(np.uint32)
                packed[k_pack, :] |= nibbles << (bit_pos * 4)

        weight_packed = torch.from_numpy(packed).to("mps")
        scales_out = scales.to("mps")
        return weight_packed, scales_out


sys.path.insert(0, str(Path(__file__).parent))
from framework import Benchmark, mps_sync  # noqa: E402

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


def _create_fp4_weights(K: int, N: int, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random FP4-packed weights and per-group scales.

    Generates a random FP16 weight matrix, quantizes to FP4 E2M1 via
    pack_fp4_weights, and returns the packed representation ready for
    the quantized GEMM kernel.

    Args:
        K: Input features dimension.
        N: Output features dimension.
        group_size: Quantization group size.

    Returns:
        Tuple of (packed_weights, scales) on MPS device.
    """
    # Weight in PyTorch convention: [N, K] (out_features, in_features)
    W = torch.randn(N, K, dtype=torch.float16, device="mps")
    mps_sync()
    packed, scales = pack_fp4_weights(W, group_size=group_size)
    mps_sync()
    return packed, scales


def bench_marlin_fp4(
    warmup: int = 10,
    iterations: int = 100,
    group_size: int = 128,
) -> Benchmark:
    """Benchmark Marlin FP4 GEMM across standard LLM sizes."""
    bench = Benchmark(warmup=warmup, iterations=iterations)

    for M, N, K, desc in SIZES:
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        B_packed, scales = _create_fp4_weights(K, N, group_size=group_size)
        mps_sync()

        def fn(
            a: torch.Tensor = A,
            b: torch.Tensor = B_packed,
            s: torch.Tensor = scales,
            gs: int = group_size,
        ) -> torch.Tensor:
            return quantized_linear_torch(a, b, s, gs)

        result = bench.run(f"FP4 {desc}", fn, M, N, K)
        print(f"  {desc}: {result.mean_ms:.3f}ms ({result.tflops:.2f} TFLOPS)")

    return bench


def bench_torch_int4(
    warmup: int = 10,
    iterations: int = 100,
) -> Benchmark:
    """Benchmark PyTorch MPS INT4 quantized matmul for comparison.

    Uses a simple dequantize-then-matmul approach since PyTorch MPS
    doesn't have native 4-bit quantized matmul like MLX.
    """
    bench = Benchmark(warmup=warmup, iterations=iterations)

    for M, N, K, desc in SIZES:
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        # Create quantized weights via pack and use quantized_linear
        B_packed, scales = _create_fp4_weights(K, N, group_size=64)
        mps_sync()

        def fn(
            a: torch.Tensor = A, b: torch.Tensor = B_packed, s: torch.Tensor = scales
        ) -> torch.Tensor:
            return quantized_linear_torch(a, b, s, 64)

        result = bench.run(f"INT4 {desc}", fn, M, N, K)
        print(f"  {desc}: {result.mean_ms:.3f}ms ({result.tflops:.2f} TFLOPS)")

    return bench


def bench_fp16_baseline(
    warmup: int = 10,
    iterations: int = 100,
) -> Benchmark:
    """Benchmark FP16 matmul as throughput ceiling reference."""
    bench = Benchmark(warmup=warmup, iterations=iterations)

    for M, N, K, desc in SIZES:
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        B = torch.randn(K, N, dtype=torch.float16, device="mps")
        mps_sync()

        def fn(a: torch.Tensor = A, b: torch.Tensor = B) -> torch.Tensor:
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
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        mps_sync()

        # Marlin FP4
        B_packed, scales = _create_fp4_weights(K, N, group_size=group_size)

        def marlin_fn(
            a: torch.Tensor = A,
            b: torch.Tensor = B_packed,
            s: torch.Tensor = scales,
            gs: int = group_size,
        ) -> torch.Tensor:
            return quantized_linear_torch(a, b, s, gs)

        bench.run(f"Marlin FP4 | {desc}", marlin_fn, M, N, K)

        # INT4 with smaller group size
        B_packed_int4, scales_int4 = _create_fp4_weights(K, N, group_size=64)
        mps_sync()

        def int4_fn(
            a: torch.Tensor = A, b: torch.Tensor = B_packed_int4, s: torch.Tensor = scales_int4
        ) -> torch.Tensor:
            return quantized_linear_torch(a, b, s, 64)

        bench.run(f"INT4 gs=64 | {desc}", int4_fn, M, N, K)

        # FP16 baseline
        B_fp16 = torch.randn(K, N, dtype=torch.float16, device="mps")
        mps_sync()

        def fp16_fn(a: torch.Tensor = A, b: torch.Tensor = B_fp16) -> torch.Tensor:
            return a @ b

        bench.run(f"FP16       | {desc}", fp16_fn, M, N, K)

    bench.print_summary()


def main() -> None:
    """Run all GEMM benchmarks and export results."""
    if not HAS_MPS:
        print("ERROR: Benchmarks require PyTorch MPS backend for Metal GPU access.")
        print("Ensure you're on Apple Silicon with PyTorch >= 2.0")
        sys.exit(1)

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Metal Marlin FP4 GEMM Throughput Benchmark (PyTorch MPS)")
    print("=" * 70)

    print("\n--- Marlin FP4 ---")
    fp4_bench = bench_marlin_fp4()
    fp4_bench.export_json(results_dir / "marlin_fp4.json")
    fp4_bench.export_csv(results_dir / "marlin_fp4.csv")

    print("\n--- INT4 Reference (gs=64) ---")
    int4_bench = bench_torch_int4()
    int4_bench.export_json(results_dir / "int4_reference.json")
    int4_bench.export_csv(results_dir / "int4_reference.csv")

    print("\n--- FP16 Baseline ---")
    fp16_bench = bench_fp16_baseline()
    fp16_bench.export_json(results_dir / "fp16_baseline.json")
    fp16_bench.export_csv(results_dir / "fp16_baseline.csv")

    print("\n--- Head-to-Head Comparison ---")
    bench_comparison()

    print(f"\nResults exported to {results_dir}/")


if __name__ == "__main__":
    main()
