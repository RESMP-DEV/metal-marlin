#!/usr/bin/env python3
"""
Benchmark decode-optimized GEMV kernels vs standard tiled GEMM.

This script measures the performance gap for M=1 (decode) workloads:
- marlin_gemm_fp4: 64x64 tiles, ~1.5% utilization for M=1
- decode_gemv_fp4: M=1 optimized, ~100% utilization

The expected improvement is 2-4x for decode workloads.
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Kernel dispatch configuration
DECODE_THREADS = 128
DECODE_TILE_N = 256
WIDE_TILE_N = 512


def load_metal_source():
    """Load the decode GEMV Metal kernel source."""
    metal_path = Path(__file__).parent.parent / "src" / "decode_gemv.metal"
    return metal_path.read_text()


def build_decode_kernels():
    """Build all decode kernel variants."""
    source = load_metal_source()

    kernels = {}

    # Basic decode (2 cols per thread)
    kernels["decode_gemv_fp4"] = mx.fast.metal_kernel(
        name="decode_gemv_fp4_kernel",
        input_names=["A", "B", "scales"],
        output_names=["C"],
        source="""
    uint tgid_x = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    
    const uint DECODE_TILE_N = 256;
    const uint DECODE_COLS_PER_THREAD = 2;
    
    uint tg_col_base = tgid_x * DECODE_TILE_N;
    uint col_offset_0 = tg_col_base + tid * DECODE_COLS_PER_THREAD;
    uint col_offset_1 = col_offset_0 + 1;
    
    float acc_0 = 0.0f;
    float acc_1 = 0.0f;
    
    uint k_packs = (K + 7) / 8;
    
    // FP4 dequant function
    auto dequant_fp4_scaled = [](uint nibble, half scale) -> half {
        uint sign_bit = (nibble >> 3) & 1;
        uint exp_bits = (nibble >> 1) & 0x3;
        uint man_bit  = nibble & 1;
        half magnitude;
        if (exp_bits == 0) {
            magnitude = half(man_bit) * half(0.25h);
        } else {
            half power = half(1u << (exp_bits - 1));
            half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
            magnitude = power * mantissa;
        }
        half raw = sign_bit ? -magnitude : magnitude;
        return raw * scale;
    };
    
    for (uint k_base = 0; k_base < K; k_base += 8) {
        uint pack_idx = k_base / 8;
        uint group_idx = k_base / GROUP_SIZE;
        
        half a_vals[8];
        for (uint i = 0; i < 8; ++i) {
            a_vals[i] = ((k_base + i) < K) ? A[k_base + i] : half(0.0h);
        }
        
        // Column 0
        if (col_offset_0 < N && pack_idx < k_packs) {
            uint packed_0 = B[pack_idx * N + col_offset_0];
            half scale_0 = scales[group_idx * N + col_offset_0];
            for (uint i = 0; i < 8 && (k_base + i) < K; ++i) {
                uint nibble = (packed_0 >> (i * 4)) & 0xF;
                half b_val = dequant_fp4_scaled(nibble, scale_0);
                acc_0 += float(a_vals[i]) * float(b_val);
            }
        }
        
        // Column 1
        if (col_offset_1 < N && pack_idx < k_packs) {
            uint packed_1 = B[pack_idx * N + col_offset_1];
            half scale_1 = scales[group_idx * N + col_offset_1];
            for (uint i = 0; i < 8 && (k_base + i) < K; ++i) {
                uint nibble = (packed_1 >> (i * 4)) & 0xF;
                half b_val = dequant_fp4_scaled(nibble, scale_1);
                acc_1 += float(a_vals[i]) * float(b_val);
            }
        }
    }
    
    if (col_offset_0 < N) C[col_offset_0] = half(acc_0);
    if (col_offset_1 < N) C[col_offset_1] = half(acc_1);
""",
        ensure_row_contiguous=True,
    )

    # Wide decode (4 cols per thread)
    kernels["decode_gemv_fp4_wide"] = mx.fast.metal_kernel(
        name="decode_gemv_fp4_wide_kernel",
        input_names=["A", "B", "scales"],
        output_names=["C"],
        source="""
    uint tgid_x = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    
    const uint WIDE_TILE_N = 512;
    const uint WIDE_COLS_PER_THREAD = 4;
    
    uint tg_col_base = tgid_x * WIDE_TILE_N;
    uint col_base = tg_col_base + tid * WIDE_COLS_PER_THREAD;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    uint k_packs = (K + 7) / 8;
    
    auto dequant_fp4_scaled = [](uint nibble, half scale) -> half {
        uint sign_bit = (nibble >> 3) & 1;
        uint exp_bits = (nibble >> 1) & 0x3;
        uint man_bit  = nibble & 1;
        half magnitude;
        if (exp_bits == 0) {
            magnitude = half(man_bit) * half(0.25h);
        } else {
            half power = half(1u << (exp_bits - 1));
            half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
            magnitude = power * mantissa;
        }
        half raw = sign_bit ? -magnitude : magnitude;
        return raw * scale;
    };
    
    for (uint k_base = 0; k_base < K; k_base += 8) {
        uint pack_idx = k_base / 8;
        uint group_idx = k_base / GROUP_SIZE;
        
        half a_vals[8];
        for (uint i = 0; i < 8; ++i) {
            a_vals[i] = ((k_base + i) < K) ? A[k_base + i] : half(0.0h);
        }
        
        for (uint c = 0; c < 4; ++c) {
            uint col = col_base + c;
            if (col < N && pack_idx < k_packs) {
                uint packed = B[pack_idx * N + col];
                half scale = scales[group_idx * N + col];
                for (uint i = 0; i < 8 && (k_base + i) < K; ++i) {
                    uint nibble = (packed >> (i * 4)) & 0xF;
                    half b_val = dequant_fp4_scaled(nibble, scale);
                    acc[c] += float(a_vals[i]) * float(b_val);
                }
            }
        }
    }
    
    for (uint c = 0; c < 4; ++c) {
        uint col = col_base + c;
        if (col < N) C[col] = half(acc[c]);
    }
""",
        ensure_row_contiguous=True,
    )

    # SIMD optimized (1 col per thread)
    kernels["decode_gemv_fp4_simd"] = mx.fast.metal_kernel(
        name="decode_gemv_fp4_simd_kernel",
        input_names=["A", "B", "scales"],
        output_names=["C"],
        source="""
    uint tgid_x = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    
    uint col = tgid_x * 128 + tid;
    if (col >= N) return;
    
    float acc = 0.0f;
    uint k_packs = (K + 7) / 8;
    
    auto dequant_fp4_scaled = [](uint nibble, half scale) -> half {
        uint sign_bit = (nibble >> 3) & 1;
        uint exp_bits = (nibble >> 1) & 0x3;
        uint man_bit  = nibble & 1;
        half magnitude;
        if (exp_bits == 0) {
            magnitude = half(man_bit) * half(0.25h);
        } else {
            half power = half(1u << (exp_bits - 1));
            half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
            magnitude = power * mantissa;
        }
        half raw = sign_bit ? -magnitude : magnitude;
        return raw * scale;
    };
    
    for (uint k_base = 0; k_base < K; k_base += 8) {
        uint pack_idx = k_base / 8;
        uint group_idx = k_base / GROUP_SIZE;
        
        if (pack_idx >= k_packs) break;
        
        uint packed = B[pack_idx * N + col];
        half scale = scales[group_idx * N + col];
        
        for (uint i = 0; i < 8 && (k_base + i) < K; ++i) {
            half a_val = A[k_base + i];
            uint nibble = (packed >> (i * 4)) & 0xF;
            half b_val = dequant_fp4_scaled(nibble, scale);
            acc += float(a_val) * float(b_val);
        }
    }
    
    C[col] = half(acc);
""",
        ensure_row_contiguous=True,
    )

    return kernels


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def decode_gemv_dispatch(kernel, x, B, scales, N, K, group_size, tile_n):
    """Dispatch a decode kernel with appropriate grid configuration."""
    n_tgs = div_ceil(N, tile_n)

    outputs = kernel(
        inputs=[x, B, scales],
        template=[
            ("N", N),
            ("K", K),
            ("GROUP_SIZE", group_size),
        ],
        grid=(n_tgs, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.float16],
        init_value=0.0,
    )
    return outputs[0]


def pack_fp4_weights_simple(weight: mx.array, group_size: int = 128):
    """
    Simplified FP4 weight packing for testing.

    Pack FP16 weights [K, N] -> uint32 [K/8, N], with scales [K/group_size, N].
    """
    K, N = weight.shape

    # Get E2M1 representable values
    e2m1_values = np.zeros(16, dtype=np.float32)
    for nibble in range(16):
        sign = (nibble >> 3) & 1
        exp_bits = (nibble >> 1) & 3
        mant_bit = nibble & 1

        if exp_bits == 0 and mant_bit == 0:
            val = 0.0
        elif exp_bits == 0 and mant_bit == 1:
            val = 0.5
        else:
            mantissa = 1.0 + mant_bit * 0.5
            exponent = exp_bits - 1
            val = mantissa * (2.0**exponent)

        if sign:
            val = -val
        e2m1_values[nibble] = val

    mx.eval(weight)
    w_np = np.array(weight.astype(mx.float32))

    # Compute scales per group
    num_groups = K // group_size
    w_grouped = w_np.reshape(num_groups, group_size, N)
    absmax = np.abs(w_grouped).max(axis=1)  # [num_groups, N]
    absmax = np.maximum(absmax, 1e-7)
    scales_np = absmax / 6.0  # E2M1 max is 6.0

    # Quantize
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    w_normalized = w_np / scales_expanded  # [-6, 6]
    w_normalized = np.clip(w_normalized, -6.0, 6.0)

    # Map to nearest E2M1 nibble
    k_packs = K // 8
    packed = np.zeros((k_packs, N), dtype=np.uint32)

    for k_pack in range(k_packs):
        k_base = k_pack * 8
        for i in range(8):
            k_idx = k_base + i
            vals = w_normalized[k_idx, :]  # [N]
            # Find nearest E2M1
            dists = np.abs(vals[:, None] - e2m1_values[None, :])  # [N, 16]
            nibbles = np.argmin(dists, axis=1).astype(np.uint32)  # [N]
            packed[k_pack, :] |= nibbles << (i * 4)

    return mx.array(packed), mx.array(scales_np.astype(np.float16))


def benchmark_decode_kernels(
    K: int = 4096,
    N: int = 4096,
    group_size: int = 128,
    warmup: int = 50,
    iterations: int = 200,
):
    """
    Benchmark decode kernels for M=1 workload.

    Tests:
    - decode_gemv_fp4 (256 cols/TG)
    - decode_gemv_fp4_wide (512 cols/TG)
    - decode_gemv_fp4_simd (128 cols/TG)
    - MLX native quantized_matmul (for comparison)
    """
    print(f"\n{'=' * 70}")
    print(f"Decode Benchmark: M=1, K={K}, N={N}, group_size={group_size}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print(f"{'=' * 70}\n")

    # Build kernels
    kernels = build_decode_kernels()

    # Generate test data
    x = mx.random.normal(shape=(K,), dtype=mx.float16)  # [K] for M=1
    w = mx.random.normal(shape=(K, N), dtype=mx.float16)
    mx.eval(x, w)

    # Pack weights
    B_packed, scales = pack_fp4_weights_simple(w, group_size)
    mx.eval(B_packed, scales)

    print(f"Weight packed: {B_packed.shape} uint32 ({B_packed.nbytes / 1e6:.2f} MB)")
    print(f"Scales: {scales.shape} float16 ({scales.nbytes / 1e6:.2f} MB)")
    print()

    results = {}

    # --- decode_gemv_fp4 ---
    kernel = kernels["decode_gemv_fp4"]
    for _ in range(warmup):
        out = decode_gemv_dispatch(kernel, x, B_packed, scales, N, K, group_size, 256)
        mx.eval(out)

    start = time.perf_counter()
    for _ in range(iterations):
        out = decode_gemv_dispatch(kernel, x, B_packed, scales, N, K, group_size, 256)
        mx.eval(out)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["decode_gemv_fp4"] = elapsed
    print(f"decode_gemv_fp4:      {elapsed:.4f} ms")

    # --- decode_gemv_fp4_wide ---
    kernel = kernels["decode_gemv_fp4_wide"]
    for _ in range(warmup):
        out = decode_gemv_dispatch(kernel, x, B_packed, scales, N, K, group_size, 512)
        mx.eval(out)

    start = time.perf_counter()
    for _ in range(iterations):
        out = decode_gemv_dispatch(kernel, x, B_packed, scales, N, K, group_size, 512)
        mx.eval(out)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["decode_gemv_fp4_wide"] = elapsed
    print(f"decode_gemv_fp4_wide: {elapsed:.4f} ms")

    # --- decode_gemv_fp4_simd ---
    kernel = kernels["decode_gemv_fp4_simd"]
    for _ in range(warmup):
        out = decode_gemv_dispatch(kernel, x, B_packed, scales, N, K, group_size, 128)
        mx.eval(out)

    start = time.perf_counter()
    for _ in range(iterations):
        out = decode_gemv_dispatch(kernel, x, B_packed, scales, N, K, group_size, 128)
        mx.eval(out)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["decode_gemv_fp4_simd"] = elapsed
    print(f"decode_gemv_fp4_simd: {elapsed:.4f} ms")

    # --- MLX native matmul (FP16 reference) ---
    x_2d = x.reshape(1, K)
    for _ in range(warmup):
        out_ref = x_2d @ w
        mx.eval(out_ref)

    start = time.perf_counter()
    for _ in range(iterations):
        out_ref = x_2d @ w
        mx.eval(out_ref)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["mlx_fp16"] = elapsed
    print(f"MLX FP16 matmul:      {elapsed:.4f} ms")

    # --- MLX quantized_matmul ---
    w_q, scales_q, biases_q = mx.quantize(w.T, group_size=group_size, bits=4)
    mx.eval(w_q, scales_q, biases_q)

    for _ in range(warmup):
        out_q = mx.quantized_matmul(
            x_2d,
            w_q,
            scales=scales_q,
            biases=biases_q,
            transpose=True,
            group_size=group_size,
            bits=4,
        )
        mx.eval(out_q)

    start = time.perf_counter()
    for _ in range(iterations):
        out_q = mx.quantized_matmul(
            x_2d,
            w_q,
            scales=scales_q,
            biases=biases_q,
            transpose=True,
            group_size=group_size,
            bits=4,
        )
        mx.eval(out_q)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["mlx_q4"] = elapsed
    print(f"MLX quantized_matmul: {elapsed:.4f} ms")

    # Compute speedups
    print(f"\n{'─' * 50}")
    print("Speedups vs MLX FP16:")
    for name, time_ms in results.items():
        if name != "mlx_fp16":
            speedup = results["mlx_fp16"] / time_ms
            print(f"  {name:25s}: {speedup:.2f}x")

    print("\nSpeedups vs MLX quantized_matmul:")
    for name, time_ms in results.items():
        if name not in ["mlx_fp16", "mlx_q4"]:
            speedup = results["mlx_q4"] / time_ms
            print(f"  {name:25s}: {speedup:.2f}x")

    # Compute throughput in TFLOPS
    flops = 2 * K * N  # For M=1 vector-matrix
    print("\nThroughput (TFLOPS):")
    for name, time_ms in results.items():
        tflops = flops / (time_ms / 1000) / 1e12
        print(f"  {name:25s}: {tflops:.2f} TFLOPS")

    # Compute tokens/sec (if this were decode)
    print("\nEquivalent decode tok/s (single layer):")
    for name, time_ms in results.items():
        tps = 1000 / time_ms
        print(f"  {name:25s}: {tps:.0f} tok/s")

    return results


def main():
    """Run decode kernel benchmarks."""
    print("\n" + "=" * 70)
    print("DECODE KERNEL OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print("\nThis benchmark compares decode-optimized kernels against")
    print("tiled GEMM kernels for M=1 workloads.\n")
    print("The optimization addresses the 60% decode gap observed in")
    print("the inference stack benchmark.")
    print("=" * 70)

    # Test matrix sizes matching common LLM hidden dimensions
    for K, N in [(4096, 4096), (4096, 11008), (5120, 5120), (8192, 8192)]:
        benchmark_decode_kernels(K=K, N=N, group_size=128, warmup=50, iterations=200)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Expected results:
- decode_gemv kernels should be 2-4x faster than tiled GEMM for M=1
- This addresses the 60% decode gap by using M=1 optimized tiling
- Key insight: For M=1, we need N-parallelism not M×N tiling

Next steps:
1. Integrate auto-dispatch in the inference pipeline
2. Profile memory bandwidth utilization
3. Test with actual model weights
""")


if __name__ == "__main__":
    main()
