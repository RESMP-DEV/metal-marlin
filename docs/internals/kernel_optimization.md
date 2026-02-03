# Kernel Optimization Guide

This document details the optimization techniques applied to Metal Marlin kernels to achieve high performance on Apple Silicon (specifically M3/M4 Max).

## 1. GEMM Optimizations

The FP4 quantized GEMM kernel (`gemm_fp4_optimized.metal`) implements several critical optimizations to maximize throughput and minimize latency.

### Core Techniques

1.  **LUT-Based Dequantization**
    *   **Mechanism:** Uses a 16-entry constant lookup table (LUT) in constant memory for FP4 E2M1 decoding.
    *   **Impact:** Reduces dequantization cost from ~8 cycles (bitwise ALU ops) to ~3 cycles (memory lookup).
    *   **Implementation:** `dequant_fp4_lut` in `gemm_fp4_optimized.metal`.

2.  **Double-Buffered Tiling**
    *   **Mechanism:** Uses threadgroup memory to double-buffer `A` tiles. While the SIMD unit computes on buffer `i`, the load unit fetches buffer `i+1`.
    *   **Impact:** Hides global memory latency behind arithmetic instructions.

3.  **On-the-Fly Weight Dequantization**
    *   **Mechanism:** Weights are kept in packed FP4 format in global memory and dequantized directly into registers/simdgroup memory immediately before use.
    *   **Impact:** Drastically reduces global memory bandwidth requirements (4 bits vs 16 bits per weight), enabling high arithmetic intensity.

4.  **Simdgroup-Local Staging**
    *   **Mechanism:** Dequantized `B` tile fragments are stored in `threadgroup` memory but partitioned such that each simdgroup only accesses a small 512-byte slice.
    *   **Impact:** Reduces total threadgroup memory usage (512B vs 8KB for full B tile), allowing for higher occupancy (more active threadgroups per core).

5.  **Fused Epilogue**
    *   **Mechanism:** Bias addition and activation functions (GELU, SILU, ReLU) are computed in the same kernel immediately after the GEMM loop.
    *   **Impact:** Eliminates the overhead of launching a separate kernel and reading/writing intermediate results to global memory.

### Specialized Variants

*   **Large M (`gemm_fp4_optimized_large_m`):** Uses 128x64 tiles for batch inference (prefill), prioritizing throughput.
*   **Decode (`gemm_fp4_optimized_decode`):** Uses 32x128 tiles for small M (M=1..16), maximizing parallelism across the N dimension for low-latency decoding.

## 2. MoE Dispatch Optimizations

The Mixture-of-Experts dispatch kernel (`moe_dispatch_optimized.metal`) fuses multiple stages of the MoE pipeline to avoid synchronization overhead.

1.  **Fused Routing & Grouping**
    *   **Mechanism:** Computes routing logits, selects top-k experts, and groups tokens by expert in a single pass.
    *   **Technique:** Uses **simdgroup parallel bitonic sort** (O(log² n)) for top-k selection and warp-level prefix sums for token grouping, avoiding slow atomic operations.

2.  **Coalesced Memory Access**
    *   **Mechanism:** Expert weights are stored in a transposed layout (`[num_experts, K/8, N]`).
    *   **Impact:** Ensures that threads in a simdgroup access contiguous memory addresses when loading weights, maximizing memory transaction efficiency.

3.  **Single-Kernel Pipeline**
    *   **Mechanism:** `routing -> grouping -> GEMM -> combine` are executed within a single kernel launch (or tightly coupled kernels using persistent threadgroups).
    *   **Impact:** Removes the overhead of launching multiple small kernels and synchronizing global memory between stages.

4.  **BF16 Specialized Path**
    *   **Mechanism:** `moe_dispatch_ultra_optimized_bf16` maintains data in `bfloat16` (as `uint16`) throughout the pipeline.
    *   **Impact:** Avoids costly conversion overhead between `float32`/`half` and `bfloat16`.

## 3. Attention Optimizations

Flash Attention implementations (`simdgroup_attention.metal`) leverage the hardware matrix acceleration units.

1.  **Simdgroup Matrix Acceleration**
    *   **Mechanism:** Maps $Q \times K^T$ and $P \times V$ operations directly to Apple's `simdgroup_matrix` (8x8 tile) hardware instructions.
    *   **Impact:** Significantly higher throughput compared to scalar SIMD instructions.

2.  **Online Softmax**
    *   **Mechanism:** Computes softmax statistics (max, sum) on-the-fly during the $Q \times K^T$ computation.
    *   **Impact:** Avoids materializing the $N \times N$ attention matrix, reducing memory complexity from quadratic to linear.

3.  **Instruction Level Parallelism (ILP)**
    *   **Mechanism:** Interleaves computations for multiple keys/values within the main loop.
    *   **Impact:** Hides the latency of special functions (exponential) and memory loads.

## 4. Performance Impact

Measured on Apple M3 Max / M4 Max:

| Optimization | Impact vs Baseline | Notes |
| :--- | :--- | :--- |
| **Fused FP4 GEMM** | **2.4x Speedup** | vs Dequantize + Matmul |
| **LUT Dequantization** | **~15% Speedup** | vs Bitwise Dequantization |
| **Simdgroup Attention** | **~2x Speedup** | vs Naive Attention |
| **Fused MoE Dispatch** | **>10x Speedup** | vs Torch Index Select + GEMM |

## 5. M3/M4 Max Considerations

When optimizing for Apple Silicon GPUs, consider these hardware constraints:

*   **Threadgroup Memory:** Limited to **32 KB** per threadgroup. Exceeding this drastically reduces occupancy.
    *   *Strategy:* Use aggressive tiling (e.g., 64x64x32) that fits exactly within 16KB-32KB double-buffered.
*   **Simdgroup Size:** Fixed at **32 threads**.
    *   *Strategy:* All cross-lane operations (shuffles, reductions) must assume 32-wide groups.
*   **Simdgroup Matrix:** Native support for **8x8** FP16/BF16/FP32 matrix tiles.
    *   *Strategy:* All GEMM tile dimensions ($M, N, K$) should be multiples of 8.
*   **Bandwidth:** ~300-540 GB/s.
    *   *Strategy:* Always use compressed/quantized weights (FP4) to keep kernels compute-bound rather than memory-bound.

## 6. How to Add New Optimizations

1.  **Identify the Bottleneck:** Use Xcode Instruments (Metal System Trace) to determine if a kernel is Compute, Memory, or Occupancy bound.
2.  **Create a Variant:**
    *   Copy an existing kernel in `src/`.
    *   Implement the optimization (e.g., new tiling strategy, prefetch pattern).
3.  **Register in Autotuner:**
    *   Add the new kernel variant macro to `src/kernels_autotune.metal`.
    *   Update `metal_marlin/autotuning/` scripts to include the new variant in the search space.
4.  **Verify:**
    *   Run `scripts/verify_kernels.py` to ensure correctness.
    *   Run `scripts/optimize_kernel.py` to benchmark against existing kernels.
