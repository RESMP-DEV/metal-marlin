// dequant_fp4_fast.metal - Specialized FP4 dequantization for M=1 decode
//
// When M=1 (single token decode), simdgroup matrix operations waste 97%+ of compute
// on zero padding. This file provides dequantization primitives and kernels
// specifically optimized for the M=1 case:
//
//   1. LUT-based dequant with register-resident tables
//   2. Vectorized half4/half8 output for coalesced stores
//   3. Scale factor caching in registers (small working set)
//   4. Optional threadgroup memory staging for A-vector caching
//   5. K-streaming with prefetch for latency hiding
//
// Key insight: For M=1, the operation is C[1,N] = A[1,K] @ dequant(B[K/8,N]).
// This is a vector-matrix multiply where:
//   - A is broadcast across all output columns (same K values for all N)
//   - B dequant is independent per column (perfect parallelism)
//   - No inter-thread communication needed
//
// Performance target: >80% of theoretical memory bandwidth on M4 Max
// (546 GB/s * 0.8 = 437 GB/s effective)
//
// Memory arithmetic for decode:
//   - Read: A[K] + B[K/8,N] + scales[K/gs,N] = K*2 + K*N/2 + K*N/(2*gs) bytes
//   - Write: C[N] = N*2 bytes
//   - Compute: K*N FMAs
//   - For K=4096, N=8192, gs=128: Read=17MB, Write=16KB, 33M FMAs
//   - This is heavily memory-bound (read-dominated)
//
// Strategy: Maximize read bandwidth by:
//   - Processing multiple columns per thread (amortize A reads)
//   - Vectorized B loads (uint4 = 128-bit loads)
//   - Scale caching (same scale reused for group_size K elements)
//
// Kernels provided:
//   - dequant_fp4_decode_bulk: Standalone weight dequantization (no GEMM)
//   - dequant_fp4_decode_gemv: General-purpose M=1 GEMV with safe bounds checking
//   - dequant_fp4_decode_gemv_epilogue: GEMV with fused bias + activation
//   - dequant_fp4_decode_gemv_fast: Maximum bandwidth variant (aligned only)
//   - dequant_fp4_decode_gemv_tg_cached: Threadgroup A-vector caching
//   - dequant_fp4_decode_gemv_transposed_scales: For [N,K/gs] scale layout
//   - dequant_fp4_decode_gemv_prefetch: Software prefetching for large K
//   - dequant_fp4_decode_gemv_aligned: Fast-path with no bounds checking
//   - dequant_fp4_decode_gemv_streamed: Minimal register pressure for very large K

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// FP4 E2M1 Lookup Table
// ============================================================================
//
// Pre-computed values for all 16 FP4 E2M1 codes.
// LUT access is ~3 cycles vs ~8 cycles for bitwise dequant.
// The 16 entries fit in a single register-width constant fetch.
//
// E2M1 encoding: [sign(1)][exp(2)][mantissa(1)]
//   Normal (E>0):    (-1)^S * 2^(E-1) * (1 + M*0.5)
//   Subnormal (E=0): 0 (M=0) or 0.5 (M=1, special case encoding)
//
// Code | Binary | Sign | Exp | Man | Value
//  0   | 0000   |  +   |  0  |  0  |  0.0
//  1   | 0001   |  +   |  0  |  1  |  0.5 (subnormal special)
//  2   | 0010   |  +   |  1  |  0  |  1.0
//  3   | 0011   |  +   |  1  |  1  |  1.5
//  4   | 0100   |  +   |  2  |  0  |  2.0
//  5   | 0101   |  +   |  2  |  1  |  3.0
//  6   | 0110   |  +   |  3  |  0  |  4.0
//  7   | 0111   |  +   |  3  |  1  |  6.0
//  8   | 1000   |  -   |  0  |  0  | -0.0
//  9   | 1001   |  -   |  0  |  1  | -0.5
// 10   | 1010   |  -   |  1  |  0  | -1.0
// 11   | 1011   |  -   |  1  |  1  | -1.5
// 12   | 1100   |  -   |  2  |  0  | -2.0
// 13   | 1101   |  -   |  2  |  1  | -3.0
// 14   | 1110   |  -   |  3  |  0  | -4.0
// 15   | 1111   |  -   |  3  |  1  | -6.0
// ============================================================================

constant half FP4_DECODE_LUT[16] = {
    half( 0.0h),  half( 0.5h),  half( 1.0h),  half( 1.5h),
    half( 2.0h),  half( 3.0h),  half( 4.0h),  half( 6.0h),
    half(-0.0h),  half(-0.5h),  half(-1.0h),  half(-1.5h),
    half(-2.0h),  half(-3.0h),  half(-4.0h),  half(-6.0h)
};

// ============================================================================
// Core Dequantization Primitives for M=1 Decode
// ============================================================================

/// Single nibble dequant via LUT (no scale).
/// ~3 cycles on Apple Silicon GPU.
inline half dequant_fp4_decode(uint nibble) {
    return FP4_DECODE_LUT[nibble & 0xFu];
}

/// Single nibble dequant with scale multiplication.
/// Fused into single FMA where possible.
inline half dequant_fp4_decode_scaled(uint nibble, half scale) {
    return FP4_DECODE_LUT[nibble & 0xFu] * scale;
}

// ============================================================================
// Vectorized Unpacking: 8 FP4 values from uint32
// ============================================================================

/// Unpack and dequant 8 FP4 values with scale, output to half4 pair.
/// This is the workhorse for decode: processes one packed weight word.
///
/// Memory: 4 bytes in (packed) + 2 bytes (scale) -> 16 bytes out (8 halfs)
/// Compute: 8 LUT lookups + 8 multiplies (can fuse to 8 FMAs with accum)
inline void dequant_fp4_decode_x8(uint32_t packed, half scale,
                                   thread half4 &out_lo, thread half4 &out_hi) {
    out_lo = half4(
        FP4_DECODE_LUT[(packed >>  0) & 0xFu],
        FP4_DECODE_LUT[(packed >>  4) & 0xFu],
        FP4_DECODE_LUT[(packed >>  8) & 0xFu],
        FP4_DECODE_LUT[(packed >> 12) & 0xFu]
    ) * scale;
    out_hi = half4(
        FP4_DECODE_LUT[(packed >> 16) & 0xFu],
        FP4_DECODE_LUT[(packed >> 20) & 0xFu],
        FP4_DECODE_LUT[(packed >> 24) & 0xFu],
        FP4_DECODE_LUT[(packed >> 28) & 0xFu]
    ) * scale;
}

/// Unpack 8 FP4 values to flat array (no scale).
/// Use when scale is applied later or when building tiles.
inline void dequant_fp4_decode_x8_raw(uint32_t packed, thread half *out) {
    out[0] = FP4_DECODE_LUT[(packed >>  0) & 0xFu];
    out[1] = FP4_DECODE_LUT[(packed >>  4) & 0xFu];
    out[2] = FP4_DECODE_LUT[(packed >>  8) & 0xFu];
    out[3] = FP4_DECODE_LUT[(packed >> 12) & 0xFu];
    out[4] = FP4_DECODE_LUT[(packed >> 16) & 0xFu];
    out[5] = FP4_DECODE_LUT[(packed >> 20) & 0xFu];
    out[6] = FP4_DECODE_LUT[(packed >> 24) & 0xFu];
    out[7] = FP4_DECODE_LUT[(packed >> 28) & 0xFu];
}

/// Unpack to flat array with scale (more cache-friendly for sequential access).
inline void dequant_fp4_decode_x8_scaled_array(uint32_t packed, half scale, thread half *out) {
    out[0] = FP4_DECODE_LUT[(packed >>  0) & 0xFu] * scale;
    out[1] = FP4_DECODE_LUT[(packed >>  4) & 0xFu] * scale;
    out[2] = FP4_DECODE_LUT[(packed >>  8) & 0xFu] * scale;
    out[3] = FP4_DECODE_LUT[(packed >> 12) & 0xFu] * scale;
    out[4] = FP4_DECODE_LUT[(packed >> 16) & 0xFu] * scale;
    out[5] = FP4_DECODE_LUT[(packed >> 20) & 0xFu] * scale;
    out[6] = FP4_DECODE_LUT[(packed >> 24) & 0xFu] * scale;
    out[7] = FP4_DECODE_LUT[(packed >> 28) & 0xFu] * scale;
}

// ============================================================================
// Fused Dequant + Dot Product for M=1 Decode
// ============================================================================
//
// For vector-matrix multiply, we compute: acc += A[k:k+8] dot dequant(packed)
// Fusing dequant and accumulation minimizes register pressure and enables
// better instruction scheduling.
// ============================================================================

/// Fused dequant + dot product: acc += A[0:8] dot dequant(packed) * scale
/// This is the core inner loop for M=1 decode.
///
/// @param packed   8 packed FP4 values
/// @param a_vals   8 activation values (from A vector)
/// @param scale    Per-group scale factor
/// @param acc      Accumulator (in-out, FP32 for precision)
inline void dequant_fp4_fused_dot8(uint32_t packed,
                                    thread half *a_vals,
                                    half scale,
                                    thread float &acc) {
    // Fully unrolled for maximum ILP
    acc += float(a_vals[0]) * float(FP4_DECODE_LUT[(packed >>  0) & 0xFu] * scale);
    acc += float(a_vals[1]) * float(FP4_DECODE_LUT[(packed >>  4) & 0xFu] * scale);
    acc += float(a_vals[2]) * float(FP4_DECODE_LUT[(packed >>  8) & 0xFu] * scale);
    acc += float(a_vals[3]) * float(FP4_DECODE_LUT[(packed >> 12) & 0xFu] * scale);
    acc += float(a_vals[4]) * float(FP4_DECODE_LUT[(packed >> 16) & 0xFu] * scale);
    acc += float(a_vals[5]) * float(FP4_DECODE_LUT[(packed >> 20) & 0xFu] * scale);
    acc += float(a_vals[6]) * float(FP4_DECODE_LUT[(packed >> 24) & 0xFu] * scale);
    acc += float(a_vals[7]) * float(FP4_DECODE_LUT[(packed >> 28) & 0xFu] * scale);
}

/// Same as above but with half4 input for A (common from vectorized loads).
inline void dequant_fp4_fused_dot8_vec(uint32_t packed,
                                        half4 a_lo, half4 a_hi,
                                        half scale,
                                        thread float &acc) {
    // Process low 4 elements
    acc += float(a_lo.x) * float(FP4_DECODE_LUT[(packed >>  0) & 0xFu] * scale);
    acc += float(a_lo.y) * float(FP4_DECODE_LUT[(packed >>  4) & 0xFu] * scale);
    acc += float(a_lo.z) * float(FP4_DECODE_LUT[(packed >>  8) & 0xFu] * scale);
    acc += float(a_lo.w) * float(FP4_DECODE_LUT[(packed >> 12) & 0xFu] * scale);
    // Process high 4 elements
    acc += float(a_hi.x) * float(FP4_DECODE_LUT[(packed >> 16) & 0xFu] * scale);
    acc += float(a_hi.y) * float(FP4_DECODE_LUT[(packed >> 20) & 0xFu] * scale);
    acc += float(a_hi.z) * float(FP4_DECODE_LUT[(packed >> 24) & 0xFu] * scale);
    acc += float(a_hi.w) * float(FP4_DECODE_LUT[(packed >> 28) & 0xFu] * scale);
}

/// Partial dot product (for K remainder handling).
/// @param valid_count  Number of valid elements (0-8)
inline void dequant_fp4_fused_dot_partial(uint32_t packed,
                                           thread half *a_vals,
                                           half scale,
                                           uint valid_count,
                                           thread float &acc) {
    for (uint i = 0; i < valid_count && i < 8u; ++i) {
        uint nibble = (packed >> (i * 4u)) & 0xFu;
        acc += float(a_vals[i]) * float(FP4_DECODE_LUT[nibble] * scale);
    }
}

// ============================================================================
// Vectorized Load Helpers
// ============================================================================

/// Load 4 packed FP4 words (32 values) with single uint4 load.
/// 128-bit load maximizes memory bandwidth utilization.
inline uint4 load_packed_fp4_x4(device const uint32_t *ptr) {
    return *((device const uint4 *)ptr);
}

/// Load A vector as half4 pairs for vectorized processing.
inline void load_a_vec8(device const half *A, uint k_offset, thread half4 &lo, thread half4 &hi) {
    device const half4 *A_vec = (device const half4 *)(A + k_offset);
    lo = A_vec[0];
    hi = A_vec[1];
}

// ============================================================================
// M=1 Decode Dequant Kernel: Bulk Weight Dequantization
// ============================================================================
//
// Standalone kernel for dequantizing weights without GEMM.
// Useful for: weight inspection, debugging, two-stage pipelines.
//
// Layout:
//   packed: [K/8, N] - each uint32 holds 8 FP4 values along K
//   scales: [K/group_size, N] - per-group scales
//   output: [K, N] - dequantized FP16 weights
//
// Grid: (N, K/8) - each thread dequants one packed word
// ============================================================================

kernel void dequant_fp4_decode_bulk(
    device const uint32_t *packed  [[buffer(0)]],
    device const half *scales      [[buffer(1)]],
    device half *output            [[buffer(2)]],
    constant uint &K               [[buffer(3)]],
    constant uint &N               [[buffer(4)]],
    constant uint &group_size      [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint n_idx = gid.x;      // column
    uint k_block = gid.y;    // K block (8 elements per block)
    uint k_start = k_block * 8u;

    if (n_idx >= N || k_start >= K) return;

    // Load packed weights
    uint packed_idx = k_block * N + n_idx;
    uint32_t word = packed[packed_idx];

    // Load scale for this group
    uint group_idx = k_start / group_size;
    half scale = scales[group_idx * N + n_idx];

    // Dequantize
    half4 lo, hi;
    dequant_fp4_decode_x8(word, scale, lo, hi);

    // Write output [K, N]
    uint out_base = k_start * N + n_idx;
    uint k_remain = min(8u, K - k_start);

    // Unrolled writes for common aligned case
    if (k_remain == 8u) {
        output[out_base + 0u * N] = lo.x;
        output[out_base + 1u * N] = lo.y;
        output[out_base + 2u * N] = lo.z;
        output[out_base + 3u * N] = lo.w;
        output[out_base + 4u * N] = hi.x;
        output[out_base + 5u * N] = hi.y;
        output[out_base + 6u * N] = hi.z;
        output[out_base + 7u * N] = hi.w;
    } else {
        half vals[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        for (uint i = 0; i < k_remain; ++i) {
            output[out_base + i * N] = vals[i];
        }
    }
}

// ============================================================================
// M=1 Decode GEMV: Ultra-Fast Vector-Matrix Multiply
// ============================================================================
//
// C[1,N] = A[1,K] @ dequant(B[K/8,N], scales)
//
// Design optimized for single-token decode:
//   - Each thread owns COLS_PER_THREAD output columns (no reduction needed)
//   - A values loaded once, reused across all output columns
//   - B loads are coalesced across threads (adjacent cols = adjacent memory)
//   - Scales cached in registers (reused within quantization group)
//
// Thread organization:
//   - 128 threads per threadgroup
//   - Each thread handles 4 output columns
//   - Threadgroup covers 512 columns total
//   - Grid: ceil(N / 512) threadgroups
//
// Memory access pattern per K-iteration:
//   - A: 8 halfs broadcast (all threads read same A values)
//   - B: 4 uint32s per thread = 512 coalesced loads per threadgroup
//   - Scale: 4 halfs per thread (cached within group)
// ============================================================================

constant constexpr uint DECODE_FAST_THREADS = 128u;
constant constexpr uint DECODE_FAST_COLS_PER_THREAD = 4u;
constant constexpr uint DECODE_FAST_TILE_N = DECODE_FAST_THREADS * DECODE_FAST_COLS_PER_THREAD;  // 512

kernel void dequant_fp4_decode_gemv(
    device const half *A           [[buffer(0)]],  // [1, K] activation vector
    device const uint32_t *B       [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half *scales      [[buffer(2)]],  // [K/group_size, N] per-group scales
    device half *C                 [[buffer(3)]],  // [1, N] output vector
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    // Column indices for this thread
    const uint col_base = tgid_x * DECODE_FAST_TILE_N + tid * DECODE_FAST_COLS_PER_THREAD;

    // Early exit if all columns are out of bounds
    if (col_base >= N) return;

    // FP32 accumulators for numerical stability
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Precompute valid column mask
    bool valid[4];
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        valid[c] = (col_base + c) < N;
    }

    // Cache for current scale values (reused within quantization group)
    half cached_scales[4];
    uint cached_group_idx = ~0u;  // Invalid sentinel

    const uint k_packs = (K + 7u) / 8u;

    // Main K-reduction loop
    for (uint k_base = 0; k_base < K; k_base += 8u) {
        const uint pack_idx = k_base / 8u;
        const uint group_idx = k_base / group_size;

        // Load A values (same for all columns - broadcast)
        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        // Update scale cache if group changed
        if (group_idx != cached_group_idx) {
            cached_group_idx = group_idx;
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    cached_scales[c] = scales[group_idx * N + col_base + c];
                }
            }
        }

        // Process 4 columns with fused dequant + dot product
        if (pack_idx < k_packs) {
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    uint col = col_base + c;
                    uint32_t packed = B[pack_idx * N + col];

                    // Fused dequant + dot product
                    dequant_fp4_fused_dot8(packed, a_vals, cached_scales[c], acc[c]);
                }
            }
        }
    }

    // Handle K remainder (when K % 8 != 0) - adjust last iteration's contribution
    // The loop above handles this via boundary checks in a_vals loading

    // Store results
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        if (valid[c]) {
            C[col_base + c] = half(acc[c]);
        }
    }
}

// ============================================================================
// M=1 Decode GEMV with Bias + Activation Epilogue
// ============================================================================
//
// Fused operation: C[1,N] = activation(A[1,K] @ dequant(B) + bias[N])
//
// Epilogue modes:
//   0 = None (just GEMV)
//   1 = Bias only
//   2 = GELU (no bias)
//   3 = SiLU (no bias)
//   4 = Bias + GELU
//   5 = Bias + SiLU
//   6 = ReLU (no bias)
//   7 = Bias + ReLU
// ============================================================================

inline half gelu_decode(half x) {
    // GELU approximation: x * sigmoid(1.702 * x)
    half kx = half(1.702h) * x;
    return x / (half(1.0h) + exp(-kx));
}

inline half silu_decode(half x) {
    return x / (half(1.0h) + exp(-x));
}

inline half apply_activation_decode(half val, half bias_val, uint mode) {
    // Add bias if mode requires it (modes 1, 4, 5, 7)
    if (mode == 1 || mode == 4 || mode == 5 || mode == 7) {
        val += bias_val;
    }

    // Apply activation
    switch (mode) {
        case 2: case 4:  // GELU or Bias+GELU
            return gelu_decode(val);
        case 3: case 5:  // SiLU or Bias+SiLU
            return silu_decode(val);
        case 6: case 7:  // ReLU or Bias+ReLU
            return max(val, half(0.0h));
        default:
            return val;
    }
}

kernel void dequant_fp4_decode_gemv_epilogue(
    device const half *A           [[buffer(0)]],
    device const uint32_t *B       [[buffer(1)]],
    device const half *scales      [[buffer(2)]],
    device half *C                 [[buffer(3)]],
    device const half *bias        [[buffer(4)]],  // [N] or nullptr
    constant uint &K               [[buffer(5)]],
    constant uint &N               [[buffer(6)]],
    constant uint &group_size      [[buffer(7)]],
    constant uint &epilogue_mode   [[buffer(8)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    const uint col_base = tgid_x * DECODE_FAST_TILE_N + tid * DECODE_FAST_COLS_PER_THREAD;

    if (col_base >= N) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    bool valid[4];
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        valid[c] = (col_base + c) < N;
    }

    half cached_scales[4];
    uint cached_group_idx = ~0u;

    const uint k_packs = (K + 7u) / 8u;

    for (uint k_base = 0; k_base < K; k_base += 8u) {
        const uint pack_idx = k_base / 8u;
        const uint group_idx = k_base / group_size;

        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        if (group_idx != cached_group_idx) {
            cached_group_idx = group_idx;
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    cached_scales[c] = scales[group_idx * N + col_base + c];
                }
            }
        }

        if (pack_idx < k_packs) {
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    uint col = col_base + c;
                    uint32_t packed = B[pack_idx * N + col];
                    dequant_fp4_fused_dot8(packed, a_vals, cached_scales[c], acc[c]);
                }
            }
        }
    }

    // Apply epilogue and store
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        if (valid[c]) {
            uint col = col_base + c;
            half result = half(acc[c]);
            half bias_val = (bias != nullptr) ? bias[col] : half(0.0h);
            C[col] = apply_activation_decode(result, bias_val, epilogue_mode);
        }
    }
}

// ============================================================================
// M=1 Decode GEMV: Maximum Bandwidth Variant
// ============================================================================
//
// Stripped-down kernel for absolute maximum throughput.
// Assumptions:
//   - N is multiple of 512
//   - K is multiple of 8
//   - group_size is multiple of 8
//   - All buffers are 16-byte aligned
//
// Uses vectorized loads (uint4 for B, half4 for A) and minimal branching.
// ============================================================================

kernel void dequant_fp4_decode_gemv_fast(
    device const half4 *A_vec      [[buffer(0)]],  // [K/4] as half4
    device const uint32_t *B       [[buffer(1)]],  // [K/8, N]
    device const half *scales      [[buffer(2)]],  // [K/gs, N]
    device half4 *C_vec            [[buffer(3)]],  // [N/4] as half4
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    // Each thread handles 4 columns, writes one half4
    const uint col_base = (tgid_x * DECODE_FAST_THREADS + tid) * 4u;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint a_vec_len = K / 4u;

    // Main loop: process K in chunks of 8
    for (uint k_base = 0; k_base < K; k_base += 8u) {
        const uint pack_idx = k_base / 8u;
        const uint group_idx = k_base / group_size;

        // Load A as 2 half4s (8 values total)
        uint a_vec_idx = k_base / 4u;
        half4 a_lo = A_vec[a_vec_idx];
        half4 a_hi = (a_vec_idx + 1u < a_vec_len) ? A_vec[a_vec_idx + 1u] : half4(0.0h);

        // Process 4 columns
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            uint col = col_base + c;
            uint32_t packed = B[pack_idx * N + col];
            half scale = scales[group_idx * N + col];

            dequant_fp4_fused_dot8_vec(packed, a_lo, a_hi, scale, acc[c]);
        }
    }

    // Store as half4
    uint out_idx = (tgid_x * DECODE_FAST_THREADS + tid);
    C_vec[out_idx] = half4(half(acc[0]), half(acc[1]), half(acc[2]), half(acc[3]));
}

// ============================================================================
// Optimized M=1 Decode with Threadgroup A-Vector Caching
// ============================================================================
//
// For M=1, the A vector is loaded K times (once per output column).
// This kernel uses threadgroup memory to cache A, reducing global memory
// bandwidth by ~Nx for the A input.
//
// Layout: A is cached in threadgroup memory, all threads in threadgroup
// read from shared cache instead of global memory.
// ============================================================================

constant constexpr uint DECODE_TG_CACHE_SIZE = 256u;  // Must be multiple of 8

kernel void dequant_fp4_decode_gemv_tg_cached(
    device const half *A           [[buffer(0)]],  // [1, K] activation vector
    device const uint32_t *B       [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half *scales      [[buffer(2)]],  // [K/group_size, N] per-group scales
    device half *C                 [[buffer(3)]],  // [1, N] output vector
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    // Threadgroup cache for A vector
    threadgroup half a_cache[DECODE_TG_CACHE_SIZE];
    
    const uint col_base = tgid_x * DECODE_FAST_TILE_N + tid * DECODE_FAST_COLS_PER_THREAD;
    if (col_base >= N) return;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool valid[4];
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        valid[c] = (col_base + c) < N;
    }
    
    half cached_scales[4];
    uint cached_group_idx = ~0u;
    
    // Process K in chunks that fit in threadgroup cache
    for (uint k_chunk = 0; k_chunk < K; k_chunk += DECODE_TG_CACHE_SIZE) {
        uint k_chunk_end = min(k_chunk + DECODE_TG_CACHE_SIZE, K);
        uint chunk_size = k_chunk_end - k_chunk;
        
        // Cooperative load of A into threadgroup cache
        // Each thread loads multiple elements
        #pragma unroll 4
        for (uint i = tid; i < DECODE_TG_CACHE_SIZE; i += DECODE_FAST_THREADS) {
            uint k_idx = k_chunk + i;
            a_cache[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process this K chunk
        for (uint k_sub = 0; k_sub < chunk_size; k_sub += 8u) {
            uint k_base = k_chunk + k_sub;
            uint pack_idx = k_base / 8u;
            uint group_idx = k_base / group_size;
            
            // Load A from threadgroup cache (fast)
            half a_vals[8];
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                a_vals[i] = a_cache[k_sub + i];
            }
            
            // Update scale cache if group changed
            if (group_idx != cached_group_idx) {
                cached_group_idx = group_idx;
                #pragma unroll
                for (uint c = 0; c < 4; ++c) {
                    if (valid[c]) {
                        cached_scales[c] = scales[group_idx * N + col_base + c];
                    }
                }
            }
            
            // Process 4 columns
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    uint col = col_base + c;
                    uint32_t packed = B[pack_idx * N + col];
                    dequant_fp4_fused_dot8(packed, a_vals, cached_scales[c], acc[c]);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store results
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        if (valid[c]) {
            C[col_base + c] = half(acc[c]);
        }
    }
}

// ============================================================================
// M=1 Decode with Transposed Scale Layout
// ============================================================================
//
// Alternative kernel for when scales are stored as [N, K/group_size] instead
// of [K/group_size, N]. This provides better memory coalescing when each
// thread accesses scales for the same group index across different columns.
//
// Use this when: scales_layout == 'transposed'
// ============================================================================

kernel void dequant_fp4_decode_gemv_transposed_scales(
    device const half *A           [[buffer(0)]],
    device const uint32_t *B       [[buffer(1)]],
    device const half *scales      [[buffer(2)]],  // [N, K/group_size] layout
    device half *C                 [[buffer(3)]],
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    const uint col_base = tgid_x * DECODE_FAST_TILE_N + tid * DECODE_FAST_COLS_PER_THREAD;
    if (col_base >= N) return;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool valid[4];
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        valid[c] = (col_base + c) < N;
    }
    
    const uint n_groups = (K + group_size - 1u) / group_size;
    
    // Main K-reduction loop
    for (uint k_base = 0; k_base < K; k_base += 8u) {
        uint pack_idx = k_base / 8u;
        uint group_idx = k_base / group_size;
        
        // Load A values
        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }
        
        // Load scales for this group (transposed layout: [N, n_groups])
        half local_scales[4];
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            if (valid[c]) {
                uint col = col_base + c;
                local_scales[c] = scales[col * n_groups + group_idx];
            }
        }
        
        // Process columns
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            if (valid[c]) {
                uint col = col_base + c;
                uint32_t packed = B[pack_idx * N + col];
                dequant_fp4_fused_dot8(packed, a_vals, local_scales[c], acc[c]);
            }
        }
    }
    
    // Store results
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        if (valid[c]) {
            C[col_base + c] = half(acc[c]);
        }
    }
}

// ============================================================================
// M=1 Decode GEMV with Prefetch Hints
// ============================================================================
//
// Uses Metal's prefetch hint intrinsics to hide memory latency.
// Optimal for large K where memory bandwidth is the bottleneck.
//
// Prefetch strategy:
//   - Prefetch B for next K iteration while computing current
//   - Scale values prefetched into registers ahead of use
// ============================================================================

kernel void dequant_fp4_decode_gemv_prefetch(
    device const half *A           [[buffer(0)]],
    device const uint32_t *B       [[buffer(1)]],
    device const half *scales      [[buffer(2)]],
    device half *C                 [[buffer(3)]],
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    const uint col_base = tgid_x * DECODE_FAST_TILE_N + tid * DECODE_FAST_COLS_PER_THREAD;
    if (col_base >= N) return;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool valid[4];
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        valid[c] = (col_base + c) < N;
    }
    
    half cached_scales[4];
    uint cached_group_idx = ~0u;
    
    const uint k_packs = (K + 7u) / 8u;
    
    // Prefetch first iteration's data
    half a_vals_prefetch[8];
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        a_vals_prefetch[i] = (i < K) ? A[i] : half(0.0h);
    }
    
    // Main loop with prefetch
    for (uint pack_idx = 0; pack_idx < k_packs; ++pack_idx) {
        uint k_base = pack_idx * 8u;
        uint group_idx = k_base / group_size;
        
        // Use prefetched A values
        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            a_vals[i] = a_vals_prefetch[i];
        }
        
        // Prefetch next A values
        uint next_k_base = (pack_idx + 1u) * 8u;
        if (next_k_base < K) {
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                uint k_idx = next_k_base + i;
                a_vals_prefetch[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
            }
        }
        
        // Update scale cache
        if (group_idx != cached_group_idx) {
            cached_group_idx = group_idx;
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    cached_scales[c] = scales[group_idx * N + col_base + c];
                }
            }
        }
        
        // Process columns
        if (pack_idx < k_packs) {
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    uint col = col_base + c;
                    uint32_t packed = B[pack_idx * N + col];
                    
                    // Inline dequant for better instruction scheduling
                    half4 b_lo = half4(
                        FP4_DECODE_LUT[(packed >>  0) & 0xFu],
                        FP4_DECODE_LUT[(packed >>  4) & 0xFu],
                        FP4_DECODE_LUT[(packed >>  8) & 0xFu],
                        FP4_DECODE_LUT[(packed >> 12) & 0xFu]
                    ) * cached_scales[c];
                    
                    half4 b_hi = half4(
                        FP4_DECODE_LUT[(packed >> 16) & 0xFu],
                        FP4_DECODE_LUT[(packed >> 20) & 0xFu],
                        FP4_DECODE_LUT[(packed >> 24) & 0xFu],
                        FP4_DECODE_LUT[(packed >> 28) & 0xFu]
                    ) * cached_scales[c];
                    
                    acc[c] += float(a_vals[0]) * float(b_lo.x);
                    acc[c] += float(a_vals[1]) * float(b_lo.y);
                    acc[c] += float(a_vals[2]) * float(b_lo.z);
                    acc[c] += float(a_vals[3]) * float(b_lo.w);
                    acc[c] += float(a_vals[4]) * float(b_hi.x);
                    acc[c] += float(a_vals[5]) * float(b_hi.y);
                    acc[c] += float(a_vals[6]) * float(b_hi.z);
                    acc[c] += float(a_vals[7]) * float(b_hi.w);
                }
            }
        }
    }
    
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        if (valid[c]) {
            C[col_base + c] = half(acc[c]);
        }
    }
}

// ============================================================================
// M=1 Decode GEMV Streamed (Minimal Register Pressure)
// ============================================================================
//
// Optimized for very large K (>8192) where register pressure matters.
// Uses streaming accumulation with periodic reduction to minimize register use.
//
// Strategy:
//   - Process K in chunks of 1024 to keep intermediate results in registers
//   - Stream through K dimension with minimal live values
// ============================================================================

constant constexpr uint STREAM_CHUNK_K = 1024u;

kernel void dequant_fp4_decode_gemv_streamed(
    device const half *A           [[buffer(0)]],
    device const uint32_t *B       [[buffer(1)]],
    device const half *scales      [[buffer(2)]],
    device half *C                 [[buffer(3)]],
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    const uint col_base = tgid_x * DECODE_FAST_TILE_N + tid * DECODE_FAST_COLS_PER_THREAD;
    if (col_base >= N) return;
    
    // Track validity of columns
    bool valid[4];
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        valid[c] = (col_base + c) < N;
    }
    
    // Main accumulator - only 4 floats live across entire K dimension
    float total_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Process in chunks to manage register pressure
    for (uint chunk_start = 0; chunk_start < K; chunk_start += STREAM_CHUNK_K) {
        uint chunk_end = min(chunk_start + STREAM_CHUNK_K, K);
        
        // Chunk-local accumulators
        float chunk_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        half cached_scales[4];
        uint cached_group_idx = ~0u;
        
        // Process this chunk
        for (uint k_base = chunk_start; k_base < chunk_end; k_base += 8u) {
            uint pack_idx = k_base / 8u;
            uint group_idx = k_base / group_size;
            
            // Load A values
            half a_vals[8];
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                uint k_idx = k_base + i;
                a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
            }
            
            // Update scales if group changed
            if (group_idx != cached_group_idx) {
                cached_group_idx = group_idx;
                #pragma unroll
                for (uint c = 0; c < 4; ++c) {
                    if (valid[c]) {
                        cached_scales[c] = scales[group_idx * N + col_base + c];
                    }
                }
            }
            
            // Process columns with fused dequant
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                if (valid[c]) {
                    uint col = col_base + c;
                    uint32_t packed = B[pack_idx * N + col];
                    
                    // Unrolled dequant + dot product
                    chunk_acc[c] += float(a_vals[0]) * float(FP4_DECODE_LUT[(packed >>  0) & 0xFu] * cached_scales[c]);
                    chunk_acc[c] += float(a_vals[1]) * float(FP4_DECODE_LUT[(packed >>  4) & 0xFu] * cached_scales[c]);
                    chunk_acc[c] += float(a_vals[2]) * float(FP4_DECODE_LUT[(packed >>  8) & 0xFu] * cached_scales[c]);
                    chunk_acc[c] += float(a_vals[3]) * float(FP4_DECODE_LUT[(packed >> 12) & 0xFu] * cached_scales[c]);
                    chunk_acc[c] += float(a_vals[4]) * float(FP4_DECODE_LUT[(packed >> 16) & 0xFu] * cached_scales[c]);
                    chunk_acc[c] += float(a_vals[5]) * float(FP4_DECODE_LUT[(packed >> 20) & 0xFu] * cached_scales[c]);
                    chunk_acc[c] += float(a_vals[6]) * float(FP4_DECODE_LUT[(packed >> 24) & 0xFu] * cached_scales[c]);
                    chunk_acc[c] += float(a_vals[7]) * float(FP4_DECODE_LUT[(packed >> 28) & 0xFu] * cached_scales[c]);
                }
            }
        }
        
        // Accumulate chunk result
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            total_acc[c] += chunk_acc[c];
        }
    }
    
    // Store final results
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        if (valid[c]) {
            C[col_base + c] = half(total_acc[c]);
        }
    }
}

// ============================================================================
// M=1 Decode GEMV Fast-Path (Assumes aligned dimensions)
// ============================================================================
//
// Maximum performance variant that assumes:
//   - N is multiple of 512
//   - K is multiple of 8
//   - group_size is multiple of 8
//   - All buffers are 16-byte aligned
//
// No bounds checking in inner loops - use only when preconditions are met.
// ============================================================================

kernel void dequant_fp4_decode_gemv_aligned(
    device const half *A           [[buffer(0)]],
    device const uint32_t *B       [[buffer(1)]],
    device const half *scales      [[buffer(2)]],
    device half *C                 [[buffer(3)]],
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tgid_x                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    const uint col_base = tgid_x * DECODE_FAST_TILE_N + tid * DECODE_FAST_COLS_PER_THREAD;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    half cached_scales[4];
    uint cached_group_idx = ~0u;
    
    const uint k_packs = K / 8u;
    
    // Unrolled main loop - no bounds checking
    for (uint pack_idx = 0; pack_idx < k_packs; ++pack_idx) {
        uint k_base = pack_idx * 8u;
        uint group_idx = k_base / group_size;
        
        // Load A values (vectorized)
        half4 a_lo = *((device const half4 *)(A + k_base));
        half4 a_hi = *((device const half4 *)(A + k_base + 4u));
        
        // Update scales if group changed
        if (group_idx != cached_group_idx) {
            cached_group_idx = group_idx;
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                uint col = col_base + c;
                cached_scales[c] = scales[group_idx * N + col];
            }
        }
        
        // Process 4 columns with fused dequant
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            uint col = col_base + c;
            uint32_t packed = B[pack_idx * N + col];
            
            half scale = cached_scales[c];
            acc[c] += float(a_lo.x) * float(FP4_DECODE_LUT[(packed >>  0) & 0xFu] * scale);
            acc[c] += float(a_lo.y) * float(FP4_DECODE_LUT[(packed >>  4) & 0xFu] * scale);
            acc[c] += float(a_lo.z) * float(FP4_DECODE_LUT[(packed >>  8) & 0xFu] * scale);
            acc[c] += float(a_lo.w) * float(FP4_DECODE_LUT[(packed >> 12) & 0xFu] * scale);
            acc[c] += float(a_hi.x) * float(FP4_DECODE_LUT[(packed >> 16) & 0xFu] * scale);
            acc[c] += float(a_hi.y) * float(FP4_DECODE_LUT[(packed >> 20) & 0xFu] * scale);
            acc[c] += float(a_hi.z) * float(FP4_DECODE_LUT[(packed >> 24) & 0xFu] * scale);
            acc[c] += float(a_hi.w) * float(FP4_DECODE_LUT[(packed >> 28) & 0xFu] * scale);
        }
    }
    
    // Store results
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        C[col_base + c] = half(acc[c]);
    }
}

// ============================================================================
// Test Kernels
// ============================================================================

/// Test: Verify LUT values match expected FP4 E2M1 encoding.
kernel void test_fp4_decode_lut(
    device half *output [[buffer(0)]],
    uint tid           [[thread_position_in_grid]]
) {
    if (tid >= 16u) return;
    output[tid] = FP4_DECODE_LUT[tid];
}

/// Test: Verify dequant_fp4_decode_x8 output.
kernel void test_fp4_decode_x8(
    device const uint32_t *packed [[buffer(0)]],
    device const half *scale      [[buffer(1)]],
    device half *output           [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 lo, hi;
    dequant_fp4_decode_x8(packed[0], scale[0], lo, hi);

    output[0] = lo.x; output[1] = lo.y;
    output[2] = lo.z; output[3] = lo.w;
    output[4] = hi.x; output[5] = hi.y;
    output[6] = hi.z; output[7] = hi.w;
}

/// Test: Verify fused dot product accumulation.
kernel void test_fp4_decode_fused_dot(
    device const uint32_t *packed [[buffer(0)]],
    device const half *a_vals     [[buffer(1)]],
    device const half *scale      [[buffer(2)]],
    device float *output          [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    // Copy A values to thread-local
    half a_local[8];
    for (uint i = 0; i < 8; ++i) {
        a_local[i] = a_vals[i];
    }

    float acc = 0.0f;
    dequant_fp4_fused_dot8(packed[0], a_local, scale[0], acc);

    output[0] = acc;
}

/// Test: Full GEMV correctness test with small dimensions.
kernel void test_fp4_decode_gemv_small(
    device const half *A           [[buffer(0)]],  // [1, K]
    device const uint32_t *B       [[buffer(1)]],  // [K/8, N]
    device const half *scales      [[buffer(2)]],  // [K/gs, N]
    device half *C                 [[buffer(3)]],  // [1, N]
    constant uint &K               [[buffer(4)]],
    constant uint &N               [[buffer(5)]],
    constant uint &group_size      [[buffer(6)]],
    uint tid                       [[thread_position_in_grid]]
) {
    // Single-threaded reference implementation for testing
    if (tid > 0u) return;

    for (uint n = 0; n < N; ++n) {
        float acc = 0.0f;

        for (uint k = 0; k < K; k += 8u) {
            uint pack_idx = k / 8u;
            uint group_idx = k / group_size;

            uint32_t packed = B[pack_idx * N + n];
            half scale = scales[group_idx * N + n];

            for (uint i = 0; i < 8u && (k + i) < K; ++i) {
                half a_val = A[k + i];
                uint nibble = (packed >> (i * 4u)) & 0xFu;
                half b_val = FP4_DECODE_LUT[nibble] * scale;
                acc += float(a_val) * float(b_val);
            }
        }

        C[n] = half(acc);
    }
}
