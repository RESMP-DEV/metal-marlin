"""Benchmark FP4 quantization: NumPy vs Metal."""

import time

import numpy as np


def bench_quantize(shape, group_size, backend, iterations=10):
    """Benchmark a quantization backend."""
    weight = np.random.randn(*shape).astype(np.float32) * 0.1

    # Warmup
    for _ in range(3):
        backend(weight, group_size)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        backend(weight, group_size)
    elapsed = time.perf_counter() - start

    return elapsed / iterations


if __name__ == "__main__":
    from metal_marlin.quantize_fp4_metal import quantize_fp4_metal

    import metal_marlin.quantize_fp4 as mod

    shapes = [
        (256, 512),
        (1024, 4096),
        (4096, 4096),
        (4096, 14336),  # LLaMA-3-8B MLP size
    ]

    print("FP4 Quantization Benchmark")
    print("=" * 60)

    for shape in shapes:
        # Force NumPy
        mod._USE_METAL = False
        time_numpy = bench_quantize(shape, 128, mod.quantize_fp4)

        # Metal
        time_metal = bench_quantize(shape, 128, quantize_fp4_metal)

        speedup = time_numpy / time_metal
        print(f"{shape}: NumPy={time_numpy*1000:.2f}ms, Metal={time_metal*1000:.2f}ms, {speedup:.1f}x speedup")
