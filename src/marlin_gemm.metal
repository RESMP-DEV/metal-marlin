// marlin_gemm.metal - Quantized GEMM with multi-stage async pipeline
//
// Metal equivalent of Marlin's marlin_template.h.
// Computes C = A @ dequant(B) where B is packed FP4 quantized weights.
//
// Kernel variants:
//   1. marlin_gemm_fp4              - Double-buffered pipelined 2D dispatch
//   2. marlin_gemm_fp4_3stage       - Triple-buffered pipelined 2D dispatch
//   3. marlin_gemm_fp4_single_stage - Non-pipelined baseline (debugging/benchmarking)
//   4. marlin_gemm_fp16_single_stage- FP16 non-pipelined reference (dequant overhead)
//   5. marlin_gemm_fp4_striped      - Pipelined + stripe-partitioned 1D dispatch
//   6. marlin_gemm_fused_fp4        - Register-resident dequant (no B_tile)
//   7. marlin_gemm_fused_u4         - INT4 variant of fused approach
//   8. marlin_gemm_fp4_fp32acc      - Pipelined, FP32 accumulator (precision for K>8192)
//   9. marlin_gemm_fused_fp4_fp32acc- Fused dequant, FP32 accumulator
//  10. marlin_gemm_divergent_fp4    - Simdgroup-divergent load/compute overlap
//  11. marlin_gemm_fp8_e4m3         - Double-buffered W8A16 (FP8 E4M3 weights)
//  12. marlin_gemm_fused_fp8_e4m3   - Fused register-resident W8A16
//
// Pipeline design:
//   Marlin (CUDA) uses 4 stages with cp.async to overlap global->shared loads
//   with tensor core compute. Metal lacks hardware async copy queues, so we
//   use cooperative threadgroup loads with N-buffering (2 or 3 stages).
//   On M4 Max (~16 cycle threadgroup latency, high ALU density), 2 stages
//   is often sufficient. The 3-stage variant adds deeper latency tolerance
//   for large-K workloads where memory system jitter can stall the pipeline.
//
// CUDA -> Metal mapping:
//   mma.sync.aligned.m16n8k16 -> simdgroup_multiply_accumulate (8x8 tiles)
//   cp.async.cg.shared.global -> cooperative threadgroup load + barrier
//   Warp (32 threads)         -> Simdgroup (32 threads)
//   __syncthreads()           -> threadgroup_barrier(mem_flags::mem_threadgroup)
//   shared memory             -> threadgroup memory

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_atomic>
#include "bf16_compat.metal"

// --- MLX Compatibility: Close any open scope from utils.h ---
// This is intentionally empty but ensures clean scope boundary
// Force clean scope boundary
#ifndef MARLIN_GEMM_CLEAN_START
#define MARLIN_GEMM_CLEAN_START
#endif

using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions tuned for M4 Max (40 GPU cores, 32 threads/simdgroup)
//
// M4 Max has 32KB threadgroup memory per threadgroup, 32 threads per simdgroup.
// We target 4 simdgroups per threadgroup for good occupancy.
//
// TILE_M = 64: 8 rows of 8x8 tiles (thread_m_blocks = 8)
// TILE_N = 64: 8 cols of 8x8 tiles (thread_n_blocks = 8)
// TILE_K = 32: 4 depth tiles per mainloop iteration
//
// Double-buffered memory:
//   A_tiles: 2 * 64 * 32 * 2B = 8192 bytes
//   B_tiles: 2 * 32 * 64 * 2B = 8192 bytes
//   Total = 16384 bytes, within 32KB budget
// ---------------------------------------------------------------------------

constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K = 32;

// Number of 8x8 sub-tiles in the K (reduction) dimension
constant constexpr uint K_TILES = TILE_K / 8;  // 4

// Simdgroups per threadgroup (4 simdgroups = 128 threads)
constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 128
constant constexpr uint DIVERGENT_LOADER_SG = 2;
constant constexpr uint DIVERGENT_LOAD_THREADS = DIVERGENT_LOADER_SG * 32;  // 64
constant constexpr uint DIVERGENT_COMPUTE_SG = 2;
constant constexpr uint DIVERGENT_SG_M_TILES = TILE_M / (DIVERGENT_COMPUTE_SG * 8);
constant constexpr uint DIVERGENT_SG_N_TILES = TILE_N / 8;

// Each simdgroup is responsible for a 2x4 block of 8x8 M×N tiles
// (16 rows × 32 cols per simdgroup, 4 simdgroups tile the 64×64 output)
constant constexpr uint SG_M_TILES = 2;  // rows of 8x8 tiles per simdgroup
constant constexpr uint SG_N_TILES = 4;  // cols of 8x8 tiles per simdgroup

// FP4 packing: 8 FP4 values per uint32 (4 bits each)
constant constexpr uint FP4_PER_UINT = 8;

// Pipeline depth
constant constexpr uint NUM_BUFFERS = 2;    // Double-buffered variant
constant constexpr uint NUM_STAGES = 3;     // Triple-buffered variant

// Maximum K-parallel slices: host should allocate reduction_buf for up to 16 slices.
// (Not referenced in-kernel; the actual parallel factor is passed via buffer(10).)

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Numerical stability guards
//
// Guards against NaN/Inf from:
// - Zero scales (div by zero effectively, though we multiply so it's really
//   denormalized scale * normal value -> possible overflow)
// - Extreme zero_points causing overflow in subtraction
// - Denormalized FP16 values causing underflow
// ---------------------------------------------------------------------------

/// Clamp a half value to prevent overflow in INT4 zero-point subtraction.
/// INT4 values are 0-15, so zero_point should reasonably be in [-16, 31].
/// We use a wider range to be conservative.
inline half clamp_zero_point(half zp) {
    return clamp(zp, half(-127.0h), half(127.0h));
}

/// Guard a dequantized half value: replace NaN/Inf with 0.
inline half guard_finite(half val) {
    return select(val, half(0.0h), !isfinite(val));
}

// NOTE: Uses float intermediates to work around Metal compiler bug where
// half parameters in inline functions have fractional parts rounded.
// See docs/metal_half_precision_bug.md for details.
inline half safe_dequant(half raw, half scale) {
    float result = (float)raw * (float)scale;
    // Guard against NaN/Inf from extreme scales
    if (!isfinite(result)) {
        result = 0.0f;
    }
    return (half)result;
}

inline half dequant_fp4_bitwise(uint nibble);

// Safe dequant helpers with guards against NaN/Inf.
// NOTE: Uses float intermediates per Metal compiler bug workaround.
inline half safe_dequant_fp4(uint nibble, half scale) {
    half raw = dequant_fp4_bitwise(nibble);
    return safe_dequant(raw, scale);
}

inline half safe_dequant_u4(uint nibble, half scale, half zero_point) {
    float clamped_zero = (float)clamp_zero_point(zero_point);
    return safe_dequant((half)((float)nibble - clamped_zero), scale);
}

inline half safe_dequant_u4_value(half value, half scale, half zero_point) {
    float clamped_zero = (float)clamp_zero_point(zero_point);
    return safe_dequant((half)((float)value - clamped_zero), scale);
}

// ---------------------------------------------------------------------------
// Bitwise FP4 → FP16 dequantization (no LUT, pure ALU)
//
// FP4 E2M1 format: [sign(1) | exponent(2) | mantissa(1)]
//   bit3 = sign, bit2:1 = exponent, bit0 = mantissa
//
// Subnormal (exp=0): val = (-1)^s * m * 0.25
// Normal (exp>0):    val = (-1)^s * 2^(exp-1) * (1.0 + m*0.5)
//
// E2M1 bias = 1, so exp codes [1,2,3] map to powers [1,2,4].
// ---------------------------------------------------------------------------

inline half dequant_fp4_bitwise(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.5h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }

    return guard_finite(sign_bit ? -magnitude : magnitude);
}

inline half dequant_fp4(uint nibble, half scale) {
    return safe_dequant_fp4(nibble, scale);
}

// Unpack 8 FP4 values from a uint32, dequantize with scale, write to output
inline void unpack_fp4x8(uint packed, half scale, thread half* out) {
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = dequant_fp4(nibble, scale);
    }
}

// ---------------------------------------------------------------------------
// Cooperative tile loaders
//
// All threads in the threadgroup participate in loading tiles from device
// memory into the target threadgroup buffer. For the B tile, dequantization
// is fused into the load to keep the dequantized values in fast TG memory.
// These are factored out so both pipelined and single-stage kernels reuse them.
// ---------------------------------------------------------------------------

inline void load_A_tile(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;  // 16
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = 0.0h;
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_buf[row][col] = val;
    }
}

inline void load_B_tile_dequant(
    device const uint* B,
    device const half* scales,
    threadgroup half (&B_buf)[TILE_K][TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint scale_tiles = div_ceil(K, group_size);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    const uint packed_per_thread = (TILE_K * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);  // 2
    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = 1.0h;
        if (global_n < N && global_k_base < K && scale_k < scale_tiles) {
            s = scales[scale_k * N + global_n];
        }

        uint packed = 0;
        uint b_row = global_k_base / FP4_PER_UINT;
        if (global_n < N && b_row < k_packs && global_k_base < K) {
            packed = B[b_row * N + global_n];
        }

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        half vals[8];
        unpack_fp4x8(packed, s, vals);
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K; ++v) {
            if (n_idx < TILE_N) {
                uint global_k = global_k_base + v;
                B_buf[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : 0.0h;
            }
        }
    }
}

inline void load_A_tile_partial(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint load_thread_idx
) {
    const uint elems_per_thread = (TILE_M * TILE_K) / DIVERGENT_LOAD_THREADS;  // 32
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = load_thread_idx * elems_per_thread + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = 0.0h;
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_buf[row][col] = val;
    }
}

inline void load_B_tile_dequant_partial(
    device const uint* B,
    device const half* scales,
    threadgroup half (&B_buf)[TILE_K][TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint load_thread_idx
) {
    const uint scale_tiles = div_ceil(K, group_size);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    const uint packed_per_thread =
        (TILE_K * TILE_N) / (DIVERGENT_LOAD_THREADS * FP4_PER_UINT);  // 4
    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = load_thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = 1.0h;
        if (global_n < N && global_k_base < K && scale_k < scale_tiles) {
            s = scales[scale_k * N + global_n];
        }

        uint packed = 0;
        uint b_row = global_k_base / FP4_PER_UINT;
        if (global_n < N && b_row < k_packs && global_k_base < K) {
            packed = B[b_row * N + global_n];
        }

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        half vals[8];
        unpack_fp4x8(packed, s, vals);
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K; ++v) {
            if (n_idx < TILE_N) {
                uint global_k = global_k_base + v;
                B_buf[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : 0.0h;
            }
        }
    }
}

// FP16 B tile loader (no dequantization, for reference benchmark)
inline void load_B_tile_fp16(
    device const half* B,
    threadgroup half (&B_buf)[TILE_K][TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (TILE_K * TILE_N) / THREADS_PER_TG;  // 16
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / TILE_N;
        uint col = flat_idx % TILE_N;
        uint global_row = k_block + row;
        uint global_col = tg_col + col;

        half val = 0.0h;
        if (global_row < K && global_col < N) {
            val = B[global_row * N + global_col];
        }
        B_buf[row][col] = val;
    }
}

// ---------------------------------------------------------------------------
// Software prefetch hints: touch a minimal subset of the next tile so the
// hardware prefetcher can start streaming sequential lines before the full
// cooperative load begins.
// ---------------------------------------------------------------------------

inline void prefetch_A_tile_hint(
    device const half* A,
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
    uint flat_idx = thread_idx * elems_per_thread;
    uint row = flat_idx / TILE_K;
    uint col = flat_idx % TILE_K;
    uint global_row = tg_row + row;
    uint global_col = k_block + col;

    if (global_row < M && global_col < K) {
        volatile half sink = A[global_row * K + global_col];
        (void)sink;
    }
}

inline void prefetch_A_tile_hint_bf16(
    device const ushort* A,
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
    uint flat_idx = thread_idx * elems_per_thread;
    uint row = flat_idx / TILE_K;
    uint col = flat_idx % TILE_K;
    uint global_row = tg_row + row;
    uint global_col = k_block + col;

    if (global_row < M && global_col < K) {
        volatile ushort sink = A[global_row * K + global_col];
        (void)sink;
    }
}

inline void prefetch_B_tile_hint(
    device const uint* B,
    device const half* scales,
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint scale_tiles = div_ceil(K, group_size);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    const uint packed_per_thread = (TILE_K * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
    uint flat_packed_idx = thread_idx * packed_per_thread;
    uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
    uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

    uint global_n = tg_col + n_idx;
    uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

    uint scale_k = global_k_base / group_size;
    if (global_n < N && global_k_base < K && scale_k < scale_tiles) {
        volatile half sink_scale = scales[scale_k * N + global_n];
        (void)sink_scale;
    }

    uint b_row = global_k_base / FP4_PER_UINT;
    if (global_n < N && b_row < k_packs && global_k_base < K) {
        volatile uint sink = B[b_row * N + global_n];
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// FP32 tile load/dequant helpers (for BF16 activations + FP32 accumulate)
// ---------------------------------------------------------------------------

inline float safe_dequant_fp4_to_float(uint nibble, float scale) {
    float raw = float(dequant_fp4_bitwise(nibble));
    float result = raw * scale;
    return isfinite(result) ? result : 0.0f;
}

inline void unpack_fp4x8_fp32(uint packed, float scale, thread float* out) {
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = safe_dequant_fp4_to_float(nibble, scale);
    }
}

inline void load_B_tile_dequant_fp32(
    device const uint* B,
    device const half* scales,
    threadgroup float (&B_buf)[TILE_K][TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint scale_tiles = div_ceil(K, group_size);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    const uint packed_per_thread = (TILE_K * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        float s = 1.0f;
        if (global_n < N && global_k_base < K && scale_k < scale_tiles) {
            s = float(scales[scale_k * N + global_n]);
        }

        uint packed = 0;
        uint b_row = global_k_base / FP4_PER_UINT;
        if (global_n < N && b_row < k_packs && global_k_base < K) {
            packed = B[b_row * N + global_n];
        }

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        float vals[8];
        unpack_fp4x8_fp32(packed, s, vals);
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K; ++v) {
            if (n_idx < TILE_N) {
                uint global_k = global_k_base + v;
                B_buf[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : 0.0f;
            }
        }
    }
}

inline void load_A_tile_bf16_fp32(
    device const ushort* A,
    threadgroup float (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint blocks_per_thread = (TILE_M * TILE_K) / (THREADS_PER_TG * 8);
    const uint blocks_per_row = TILE_K / 8;
    for (uint i = 0; i < blocks_per_thread; ++i) {
        uint block_idx = thread_idx * blocks_per_thread + i;
        uint row = block_idx / blocks_per_row;
        uint col_block = block_idx % blocks_per_row;
        uint col = col_block * 8;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        float4 lo = float4(0.0f);
        float4 hi = float4(0.0f);
        if (global_row < M && (global_col + 7) < K) {
            bf16_load_as_float8(A, global_row * K + global_col, lo, hi);
        } else {
            for (uint v = 0; v < 8; ++v) {
                uint gcol = global_col + v;
                float val = 0.0f;
                if (global_row < M && gcol < K) {
                    val = bf16_bits_to_float(A[global_row * K + gcol]);
                }
                A_buf[row][col + v] = val;
            }
            continue;
        }

        A_buf[row][col + 0] = lo.x;
        A_buf[row][col + 1] = lo.y;
        A_buf[row][col + 2] = lo.z;
        A_buf[row][col + 3] = lo.w;
        A_buf[row][col + 4] = hi.x;
        A_buf[row][col + 5] = hi.y;
        A_buf[row][col + 6] = hi.z;
        A_buf[row][col + 7] = hi.w;
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute: multiply A sub-tile by B sub-tile, accumulate
// ---------------------------------------------------------------------------

inline void compute_from_tiles(
    threadgroup const half (&A_buf)[TILE_M][TILE_K],
    threadgroup const half (&B_buf)[TILE_K][TILE_N],
    thread simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    uint sg_row_offset,
    uint sg_col_offset
) {
    for (uint kt = 0; kt < K_TILES; ++kt) {
        for (uint mi = 0; mi < SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &A_buf[sg_row_offset + mi * 8][kt * 8],
                           TILE_K);

            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &B_buf[kt * 8][sg_col_offset + ni * 8],
                               TILE_N);

                simdgroup_multiply_accumulate(acc[mi][ni],
                                              a_frag,
                                              b_frag,
                                              acc[mi][ni]);
            }
        }
    }
}

inline void compute_from_tiles_divergent(
    threadgroup const half (&A_buf)[TILE_M][TILE_K],
    threadgroup const half (&B_buf)[TILE_K][TILE_N],
    thread simdgroup_matrix<half, 8, 8>
        acc[DIVERGENT_SG_M_TILES][DIVERGENT_SG_N_TILES],
    uint compute_id
) {
    const uint sg_row_offset = compute_id * (TILE_M / DIVERGENT_COMPUTE_SG);
    const uint sg_col_offset = 0;

    for (uint kt = 0; kt < K_TILES; ++kt) {
        for (uint mi = 0; mi < DIVERGENT_SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &A_buf[sg_row_offset + mi * 8][kt * 8],
                           TILE_K);

            for (uint ni = 0; ni < DIVERGENT_SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &B_buf[kt * 8][sg_col_offset + ni * 8],
                               TILE_N);

                simdgroup_multiply_accumulate(acc[mi][ni],
                                              a_frag,
                                              b_frag,
                                              acc[mi][ni]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Store accumulated results to global memory
//
// Uses simdgroup_store directly to global memory for full tiles, and
// threadgroup staging for boundary tiles (handles partial writes).
//
// Parameters:
//   staging: threadgroup buffer declared in the calling kernel, used for
//            boundary handling when tiles partially exceed M or N bounds.
// ---------------------------------------------------------------------------

inline void store_results(
    thread simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    device half* C,
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane,
    uint simd_id [[maybe_unused]],
    threadgroup half (&staging)[8][8]
) {
    uint base_row = tg_row + sg_row_offset;
    uint base_col = tg_col + sg_col_offset;

    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = base_row + mi * 8;
        if (out_row >= M) {
            continue;
        }
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = base_col + ni * 8;
            if (out_col >= N) {
                continue;
            }

            if (out_row + 8 <= M && out_col + 8 <= N) {
                // Fast path: entire 8x8 tile fits in output bounds
                // simdgroup_store writes directly to global memory
                simdgroup_store(acc[mi][ni],
                                C + out_row * N + out_col,
                                N);
            } else {
                // Boundary path: store to threadgroup staging, then scatter
                simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = staging[r][c];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

inline void store_results_divergent(
    thread simdgroup_matrix<half, 8, 8>
        acc[DIVERGENT_SG_M_TILES][DIVERGENT_SG_N_TILES],
    device half* C,
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint compute_id,
    uint simd_lane,
    threadgroup half (&staging)[8][8]
) {
    uint base_row = tg_row + compute_id * (TILE_M / DIVERGENT_COMPUTE_SG);
    uint base_col = tg_col;

    for (uint mi = 0; mi < DIVERGENT_SG_M_TILES; ++mi) {
        uint out_row = base_row + mi * 8;
        if (out_row >= M) {
            continue;
        }
        for (uint ni = 0; ni < DIVERGENT_SG_N_TILES; ++ni) {
            uint out_col = base_col + ni * 8;
            if (out_col >= N) {
                continue;
            }

            if (out_row + 8 <= M && out_col + 8 <= N) {
                // Fast path: entire 8x8 tile fits in output bounds
                simdgroup_store(acc[mi][ni],
                                C + out_row * N + out_col,
                                N);
            } else {
                // Boundary path: store to threadgroup staging, then scatter
                simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = staging[r][c];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute (FP32 accumulation)
// ---------------------------------------------------------------------------

inline void compute_from_tiles_fp32(
    threadgroup const half (&A_buf)[TILE_M][TILE_K],
    threadgroup const half (&B_buf)[TILE_K][TILE_N],
    thread simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    uint sg_row_offset,
    uint sg_col_offset
) {
    for (uint kt = 0; kt < K_TILES; ++kt) {
        for (uint mi = 0; mi < SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &A_buf[sg_row_offset + mi * 8][kt * 8],
                           TILE_K);

            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &B_buf[kt * 8][sg_col_offset + ni * 8],
                               TILE_N);

                // simdgroup_multiply_accumulate is the SIMD primitive backing
                // the FP32 accumulator path for BF16 stability and speed.
                simdgroup_multiply_accumulate(acc[mi][ni],
                                              a_frag,
                                              b_frag,
                                              acc[mi][ni]);
            }
        }
    }
}

inline void compute_from_tiles_fp32_full(
    threadgroup const float (&A_buf)[TILE_M][TILE_K],
    threadgroup const float (&B_buf)[TILE_K][TILE_N],
    thread simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    uint sg_row_offset,
    uint sg_col_offset
) {
    for (uint kt = 0; kt < K_TILES; ++kt) {
        for (uint mi = 0; mi < SG_M_TILES; ++mi) {
            simdgroup_matrix<float, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &A_buf[sg_row_offset + mi * 8][kt * 8],
                           TILE_K);

            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                simdgroup_matrix<float, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &B_buf[kt * 8][sg_col_offset + ni * 8],
                               TILE_N);

                simdgroup_multiply_accumulate(acc[mi][ni],
                                              a_frag,
                                              b_frag,
                                              acc[mi][ni]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Store accumulated FP32 results to FP16 output
// ---------------------------------------------------------------------------

inline void store_results_fp32(
    thread simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    device half* C,
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane,
    threadgroup float (&staging)[8][8]
) {
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) {
            continue;
        }
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            if (out_col >= N) {
                continue;
            }

            // FP32 accumulator -> FP16 output with bounds checking
            // simdgroup_store writes to threadgroup staging, then we convert
            // and scatter to global memory with bounds checking.
            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                if (gr < M && gc < N) {
                    // Read FP32 from staging, convert to FP16
                    C[gr * N + gc] = half(staging[r][c]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

inline void store_results_fp32_bf16(
    thread simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    device ushort* C,
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane,
    threadgroup float (&staging)[8][8]
) {
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) {
            continue;
        }
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            if (out_col >= N) {
                continue;
            }

            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (out_row + 8 <= M && out_col + 8 <= N) {
                if (simd_lane < 8) {
                    float4 lo = float4(staging[simd_lane][0],
                                       staging[simd_lane][1],
                                       staging[simd_lane][2],
                                       staging[simd_lane][3]);
                    float4 hi = float4(staging[simd_lane][4],
                                       staging[simd_lane][5],
                                       staging[simd_lane][6],
                                       staging[simd_lane][7]);
                    bf16_store_from_float8(C, (out_row + simd_lane) * N + out_col, lo, hi);
                }
            } else {
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = bf16_from_float_rne(staging[r][c]).bits;
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ===========================================================================
// Kernel: Double-buffered pipelined GEMM (primary kernel, 2D grid dispatch)
//
// Pipeline structure:
//   Prologue: Load tile[0] into buffer[0]
//   Loop k:
//     Load tile[k+1] into buffer[1-current]   (overlapped with compute)
//     Compute on buffer[current]
//     Barrier
//     Swap buffers
//
// This hides threadgroup memory load latency behind the simdgroup MMA
// compute. On M4 Max, the FP4 dequant + 4x K_TILES of 8x8 MMA per
// iteration provides enough compute to fully mask the ~16 cycle TG latency.
//
// C[M,N] = A[M,K] @ dequant(B[K/8,N], scales[K/group_size, N])
//
// Thread organization:
//   - Grid: ceil(M/TILE_M) x ceil(N/TILE_N) threadgroups
//   - Each threadgroup: 128 threads (4 simdgroups of 32)
//   - Each simdgroup handles a SG_M_TILES x SG_N_TILES block of 8x8 tiles
// ---------------------------------------------------------------------------

kernel void marlin_gemm_fp4(
    device const half* A         [[buffer(0)]],  // [M, K] activations (row-major)
    device const uint* B         [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half* scales    [[buffer(2)]],  // [K/group_size, N] per-group scales
    device half* C               [[buffer(3)]],  // [M, N] output (row-major)
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory
    threadgroup half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][TILE_K][TILE_N];
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    uint buf_compute = 0;

    // --- Prologue: Load first K-tile into buffer 0 ---
    load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_dequant(B, scales, B_tiles[0], K, N, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main pipeline loop ---
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * TILE_K;
        uint next_k = k_offset + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Software prefetch hint for the NEXT K-tile, then load into alternate buffer.
        // The barrier at loop end ensures this completes before swap.
        if (next_k < K) {
            prefetch_A_tile_hint(A, M, K, tg_row, next_k, thread_idx);
            prefetch_B_tile_hint(B, scales, K, N, tg_col, next_k, group_size, thread_idx);
            load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant(B, scales, B_tiles[buf_load], K, N, tg_col, next_k, group_size, thread_idx);
        }

        // Compute on current buffer
        compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                           acc, sg_row_offset, sg_col_offset);

        // Barrier: ensures next tile load is visible before swap
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // --- Epilogue: Store accumulated results ---
    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// Kernel: Double-buffered pipelined GEMM (FP32 accumulation)
//
// Identical to marlin_gemm_fp4, but accumulates into FP32 to improve
// numerical stability for large K. Output is still FP16.
// ===========================================================================

kernel void marlin_gemm_fp4_fp32acc(
    device const
#if defined(USE_BF16_INPUTS)
        ushort* A               [[buffer(0)]],  // [M, K] BF16 activations
#else
        half* A                 [[buffer(0)]],  // [M, K] FP16 activations
#endif
    device const uint* B         [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half* scales    [[buffer(2)]],  // [K/group_size, N] per-group scales
    device
#if defined(USE_BF16_OUTPUTS)
        ushort* C               [[buffer(3)]],  // [M, N] BF16 output
#else
        half* C                 [[buffer(3)]],  // [M, N] FP16 output
#endif
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory
    threadgroup
#if defined(USE_BF16_INPUTS)
        float A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup float B_tiles[NUM_BUFFERS][TILE_K][TILE_N];
#else
        half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][TILE_K][TILE_N];
#endif
    threadgroup float staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    uint buf_compute = 0;

    // --- Prologue: Load first K-tile into buffer 0 ---
    #if defined(USE_BF16_INPUTS)
    load_A_tile_bf16_fp32(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_dequant_fp32(B, scales, B_tiles[0], K, N, tg_col, 0, group_size, thread_idx);
    #else
    load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_dequant(B, scales, B_tiles[0], K, N, tg_col, 0, group_size, thread_idx);
    #endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main pipeline loop ---
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * TILE_K;
        uint next_k = k_offset + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Software prefetch hint for the NEXT K-tile, then load into alternate buffer.
        if (next_k < K) {
            #if defined(USE_BF16_INPUTS)
            prefetch_A_tile_hint_bf16(A, M, K, tg_row, next_k, thread_idx);
            #else
            prefetch_A_tile_hint(A, M, K, tg_row, next_k, thread_idx);
            #endif
            prefetch_B_tile_hint(B, scales, K, N, tg_col, next_k, group_size, thread_idx);
            #if defined(USE_BF16_INPUTS)
            load_A_tile_bf16_fp32(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant_fp32(B, scales, B_tiles[buf_load], K, N, tg_col, next_k, group_size, thread_idx);
            #else
            load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant(B, scales, B_tiles[buf_load], K, N, tg_col, next_k, group_size, thread_idx);
            #endif
        }

        // Compute on current buffer (FP32 accumulation)
        #if defined(USE_BF16_INPUTS)
        compute_from_tiles_fp32_full(A_tiles[buf_compute], B_tiles[buf_compute],
                                     acc, sg_row_offset, sg_col_offset);
        #else
        compute_from_tiles_fp32(A_tiles[buf_compute], B_tiles[buf_compute],
                                acc, sg_row_offset, sg_col_offset);
        #endif

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // --- Store ---
    #if defined(USE_BF16_OUTPUTS)
        store_results_fp32_bf16(acc, C, M, N, tg_row, tg_col,
                                sg_row_offset, sg_col_offset, simd_lane, staging[simd_id]);
    #else
        store_results_fp32(acc, C, M, N, tg_row, tg_col,
                           sg_row_offset, sg_col_offset, simd_lane, staging[simd_id]);
    #endif
}

// ===========================================================================
// Kernel: Triple-buffered pipelined GEMM (3-stage, 2D grid dispatch)
//
// Extends the double-buffered approach to 3 stages for deeper latency hiding.
// Metal lacks hardware async copy queues (CUDA cp.async), so load→compute
// overlap relies on the threadgroup_barrier only synchronizing the buffers
// actually in flight. With 3 buffers, the pipeline looks like:
//
//   Prologue: Load tile[0] → buf[0], load tile[1] → buf[1]
//   Loop (k=0..num_k_tiles-1):
//     Load tile[k+2] → buf[stage_load]     (if k+2 < num_k_tiles)
//     Compute on buf[stage_compute]         (tile[k])
//     Barrier
//     Rotate: stage_load, stage_compute advance modulo 3
//   Epilogue: Compute tile[num_k_tiles-1] from buf[stage_compute]
//
// Memory budget: 3 * (64*32*2 + 32*64*2) = 24576 bytes (within 32KB limit).
//
// When does 3-stage help over 2-stage?
//   - Large K with sustained streaming: the extra buffer absorbs memory
//     system jitter (cache misses, bank conflicts) without stalling compute.
//   - When B dequant is lightweight (FP4 ALU-heavy), the load port becomes
//     the bottleneck; a deeper pipeline keeps it saturated.
//   - Diminishing returns when compute is the bottleneck (small K, large M×N).
//
// C[M,N] = A[M,K] @ dequant(B[K/8,N], scales[K/group_size, N])
//
// Dispatch: Grid ceil(N/TILE_N) x ceil(M/TILE_M), threadgroup 128 threads.
// ===========================================================================

kernel void marlin_gemm_fp4_3stage(
    device const half* A         [[buffer(0)]],  // [M, K] activations (row-major)
    device const uint* B         [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half* scales    [[buffer(2)]],  // [K/group_size, N] per-group scales
    device half* C               [[buffer(3)]],  // [M, N] output (row-major)
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // Triple-buffered threadgroup memory (24576 bytes total)
    threadgroup half A_tiles[NUM_STAGES][TILE_M][TILE_K];   // 3 * 4096B = 12288B
    threadgroup half B_tiles[NUM_STAGES][TILE_K][TILE_N];   // 3 * 4096B = 12288B
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(K, TILE_K);

    // Early exit for degenerate case
    if (num_k_tiles == 0) {
        store_results(acc, C, M, N, tg_row, tg_col,
                      sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
        return;
    }

    // --- Prologue: Fill the first min(NUM_STAGES-1, num_k_tiles) buffers ---
    //
    // We load up to 2 tiles ahead. If K is very small (1 tile), we load only
    // one buffer and skip the second prologue load.
    const uint prologue_tiles = min(uint(NUM_STAGES - 1), num_k_tiles);

    for (uint s = 0; s < prologue_tiles; ++s) {
        uint k_offset = s * TILE_K;
        load_A_tile(A, A_tiles[s], M, K, tg_row, k_offset, thread_idx);
        load_B_tile_dequant(B, scales, B_tiles[s], K, N, tg_col, k_offset,
                            group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage pointers: load writes to stage_load, compute reads from stage_compute.
    // After prologue, buffers [0..prologue_tiles-1] are filled.
    // We compute starting from buffer 0.
    uint stage_load    = prologue_tiles % NUM_STAGES;
    uint stage_compute = 0;

    // --- Main pipeline loop ---
    //
    // Iteration k:
    //   1. Initiate load of tile[k + prologue_tiles] into buf[stage_load]
    //   2. Compute on buf[stage_compute] (contains tile[k])
    //   3. Barrier ensures the load completes before we rotate
    //   4. Advance both stage pointers modulo NUM_STAGES
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint prefetch_tile = kt + prologue_tiles;

        // Prefetch: load tile that is prologue_tiles ahead
        if (prefetch_tile < num_k_tiles) {
            uint k_offset = prefetch_tile * TILE_K;
            prefetch_A_tile_hint(A, M, K, tg_row, k_offset, thread_idx);
            prefetch_B_tile_hint(B, scales, K, N, tg_col, k_offset, group_size, thread_idx);
            load_A_tile(A, A_tiles[stage_load], M, K, tg_row, k_offset, thread_idx);
            load_B_tile_dequant(B, scales, B_tiles[stage_load], K, N, tg_col,
                                k_offset, group_size, thread_idx);
        }

        // Compute on the buffer that was loaded prologue_tiles iterations ago
        compute_from_tiles(A_tiles[stage_compute], B_tiles[stage_compute],
                           acc, sg_row_offset, sg_col_offset);

        // Barrier: ensures prefetch store is visible before stage_load is reused
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rotate stages
        stage_load    = (stage_load + 1) % NUM_STAGES;
        stage_compute = (stage_compute + 1) % NUM_STAGES;
    }

    // --- Epilogue: Store accumulated results ---
    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// Simdgroup-divergent pipelined GEMM (load/compute specialization)
//
// Simdgroups 0-1: load tiles into threadgroup memory
// Simdgroups 2-3: compute on loaded tiles
//
// Compute simdgroups cover the full TILE_N range and half of TILE_M each.
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads.
// ===========================================================================

kernel void marlin_gemm_divergent_fp4(
    device const half* A         [[buffer(0)]],  // [M, K] activations (row-major)
    device const uint* B         [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half* scales    [[buffer(2)]],  // [K/group_size, N] per-group scales
    device half* C               [[buffer(3)]],  // [M, N] output (row-major)
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][TILE_K][TILE_N];
    threadgroup half staging[DIVERGENT_COMPUTE_SG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const bool is_loader = simd_id < DIVERGENT_LOADER_SG;
    const bool is_compute = simd_id >= DIVERGENT_LOADER_SG;
    const uint compute_id = simd_id - DIVERGENT_LOADER_SG;
    const uint load_thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(K, TILE_K);

    simdgroup_matrix<half, 8, 8> acc[DIVERGENT_SG_M_TILES][DIVERGENT_SG_N_TILES];
    if (is_compute) {
        for (uint mi = 0; mi < DIVERGENT_SG_M_TILES; ++mi) {
            for (uint ni = 0; ni < DIVERGENT_SG_N_TILES; ++ni) {
                acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }
    }

    if (num_k_tiles == 0) {
        if (is_compute) {
            store_results_divergent(acc, C, M, N, tg_row, tg_col,
                                    compute_id, simd_lane, staging[compute_id]);
        }
        return;
    }

    uint buf_compute = 0;
    uint buf_load = 1;

    // --- Prologue: loaders fill the first buffer ---
    if (is_loader) {
        load_A_tile_partial(A, A_tiles[buf_compute], M, K, tg_row, 0, load_thread_idx);
        load_B_tile_dequant_partial(B, scales, B_tiles[buf_compute], K, N, tg_col, 0,
                                    group_size, load_thread_idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main pipeline loop ---
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint next_k = (kt + 1) * TILE_K;

        if (is_loader) {
            if (next_k < K) {
                load_A_tile_partial(A, A_tiles[buf_load], M, K, tg_row, next_k, load_thread_idx);
                load_B_tile_dequant_partial(B, scales, B_tiles[buf_load], K, N, tg_col, next_k,
                                            group_size, load_thread_idx);
            }
        } else {
            compute_from_tiles_divergent(A_tiles[buf_compute], B_tiles[buf_compute],
                                         acc, compute_id);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute ^= 1;
        buf_load ^= 1;
    }

    if (is_compute) {
        store_results_divergent(acc, C, M, N, tg_row, tg_col,
                                compute_id, simd_lane, staging[compute_id]);
    }
}

// ===========================================================================
// Single-Stage Baseline GEMM (non-pipelined, for debugging & benchmarking)
//
// Load → barrier → compute → barrier → loop
//
// Uses double-buffered tile loads for load/compute overlap without async copy.
// Still useful as:
//   1. Correctness reference for pipelined kernels
//   2. Performance baseline to measure pipelining benefit
//   3. Debugging aid (simpler control flow, deterministic execution order)
//
// Uses double-buffered threadgroup memory to overlap load/compute:
//   A_tiles: 2 * 64 * 32 * 2B = 8192 bytes
//   B_tiles: 2 * 32 * 64 * 2B = 8192 bytes
//   Total = 16384 bytes
//
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads.
// ===========================================================================

kernel void marlin_gemm_fp4_single_stage(
    device const half* A         [[buffer(0)]],  // [M, K] activations (row-major)
    device const uint* B         [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half* scales    [[buffer(2)]],  // [K/group_size, N] per-group scales
    device half* C               [[buffer(3)]],  // [M, N] output (row-major)
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered tiles to overlap loads and compute
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_tiles[2][TILE_K][TILE_N];
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    // Simdgroup layout: 2x2 grid of simdgroups covering 64x64 output tile
    // Each simdgroup handles SG_M_TILES(2) x SG_N_TILES(4) block of 8x8 tiles = 16x32
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);  // 0 or 16
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);  // 0 or 32

    // Initialize accumulators to zero
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;

    const uint k_tiles = div_ceil(K, TILE_K);
    uint buf_idx = 0;

    if (k_tiles > 0) {
        // Initial tile load
        load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
        load_B_tile_dequant(B, scales, B_tiles[0], K, N, tg_col, 0,
                            group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Main K-reduction loop (double-buffered) ---
        for (uint kt = 0; kt < k_tiles; ++kt) {
            uint k_block = kt * TILE_K;
            uint next_k = k_block + TILE_K;
            uint buf_load = 1 - buf_idx;

            if (next_k < K) {
                load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
                load_B_tile_dequant(B, scales, B_tiles[buf_load], K, N, tg_col, next_k,
                                    group_size, thread_idx);
            }

            // Compute on current buffer
            compute_from_tiles(A_tiles[buf_idx], B_tiles[buf_idx],
                               acc, sg_row_offset, sg_col_offset);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_idx = buf_load;
        }
    }

    // --- Store accumulated results ---
    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// FP16 Reference GEMM (no dequantization, for measuring dequant overhead)
//
// Same tiling and single-stage structure as marlin_gemm_fp4_single_stage,
// but B is already in FP16. Isolates the simdgroup MMA throughput from
// the dequantization cost.
//
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads.
// ===========================================================================

kernel void marlin_gemm_fp16_single_stage(
    device const half* A         [[buffer(0)]],  // [M, K] row-major FP16
    device const half* B         [[buffer(1)]],  // [K, N] row-major FP16
    device half* C               [[buffer(2)]],  // [M, N] row-major FP16
    constant uint& M             [[buffer(3)]],
    constant uint& N             [[buffer(4)]],
    constant uint& K             [[buffer(5)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_tiles[2][TILE_K][TILE_N];
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;

    const uint k_tiles = div_ceil(K, TILE_K);
    uint buf_idx = 0;

    if (k_tiles > 0) {
        load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
        load_B_tile_fp16(B, B_tiles[0], K, N, tg_col, 0, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < k_tiles; ++kt) {
            uint k_block = kt * TILE_K;
            uint next_k = k_block + TILE_K;
            uint buf_load = 1 - buf_idx;

            if (next_k < K) {
                load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
                load_B_tile_fp16(B, B_tiles[buf_load], K, N, tg_col, next_k, thread_idx);
            }

            compute_from_tiles(A_tiles[buf_idx], B_tiles[buf_idx],
                               acc, sg_row_offset, sg_col_offset);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_idx = buf_load;
        }
    }

    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// Stripe-Partitioned GEMM Kernel
//
// Marlin's stripe partitioning distributes output tiles across threadgroups
// using a 1D linearized schedule, enabling:
//   1. Load balancing when tile count is not a multiple of SM count
//   2. K-parallel reduction: multiple threadgroups can split the K-dimension
//      work for a single output tile, then atomically reduce partial sums
//
// The "stripe" pattern for a 3x3 tile grid on 5 SMs:
//     0 1 3
//     0 2 3
//     1 2 4
// Each number is the SM (threadgroup) index processing that tile.
//
// With K-parallel factor `parallel`:
//   - Total work units = m_tiles * n_tiles * parallel
//   - Each threadgroup gets ceil(total_work / num_threadgroups) units
//   - When parallel > 1, multiple threadgroups compute partial K-sums for
//     the same output tile, requiring atomic reduction
//
// Dispatch: 1D grid of num_threadgroups, each with THREADS_PER_TG threads.
// The host chooses num_threadgroups to match GPU core count for full occupancy.
//
// Buffer layout:
//   buffer(8): reduction_buf - [parallel * M * N] scratch for partial sums
//   buffer(9): locks         - [m_tiles * n_tiles] atomic counters for ordering
//   buffer(10): parallel     - K-parallel factor (1 = no reduction needed)
//   buffer(11): num_tgs      - total threadgroups dispatched
// ===========================================================================

kernel void marlin_gemm_fp4_striped(
    device const half* A             [[buffer(0)]],   // [M, K] activations
    device const uint* B             [[buffer(1)]],   // [K/8, N] packed FP4 weights
    device const half* scales        [[buffer(2)]],   // [K/group_size, N] scales
    device half* C                   [[buffer(3)]],   // [M, N] output
    constant uint& M                 [[buffer(4)]],
    constant uint& N                 [[buffer(5)]],
    constant uint& K                 [[buffer(6)]],
    constant uint& group_size        [[buffer(7)]],
    device half* reduction_buf       [[buffer(8)]],   // [MAX_PARALLEL, M, N] partial sums
    device atomic_int* locks         [[buffer(9)]],   // [m_tiles * n_tiles] completion counters
    constant uint& parallel          [[buffer(10)]],  // K-parallel factor
    constant uint& num_tgs           [[buffer(11)]],  // total threadgroups dispatched
    uint tgid_x                      [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory (one work unit at a time per TG)
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_tiles[2][TILE_K][TILE_N];

    // -----------------------------------------------------------------------
    // Stripe partitioning: compute which output tiles this threadgroup handles
    //
    // The 2D tile grid (m_tiles x n_tiles) is linearized row-major (N-first),
    // then each linear index is multiplied by the parallel factor to create
    // work units: (tile_row, tile_col, k_slice).
    //
    // Total work = m_tiles * n_tiles * parallel
    // This threadgroup handles work units at strided offsets for load balance:
    //   work_idx = tgid_x, tgid_x + num_tgs, tgid_x + 2*num_tgs, ...
    // -----------------------------------------------------------------------

    const uint m_tiles = div_ceil(M, TILE_M);
    const uint n_tiles = div_ceil(N, TILE_N);
    const uint total_tiles = m_tiles * n_tiles;
    const uint total_work = total_tiles * parallel;

    // K-dimension slicing: total K iterations split among `parallel` slices
    const uint k_iters_total = div_ceil(K, TILE_K);
    const uint k_iters_per_slice = div_ceil(k_iters_total, parallel);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    // -----------------------------------------------------------------------
    // Process each work unit assigned to this threadgroup (strided access)
    // -----------------------------------------------------------------------

    for (uint work_idx = tgid_x; work_idx < total_work; work_idx += num_tgs) {
        // Decode work unit → (tile_row, tile_col, k_slice)
        const uint tile_linear = work_idx / parallel;
        const uint k_slice = work_idx % parallel;

        const uint tile_row = tile_linear / n_tiles;
        const uint tile_col = tile_linear % n_tiles;

        const uint tg_row = tile_row * TILE_M;
        const uint tg_col = tile_col * TILE_N;

        // K range for this slice
        const uint k_start = k_slice * k_iters_per_slice * TILE_K;
        const uint k_end_iter = min((k_slice + 1) * k_iters_per_slice, k_iters_total);
        const uint k_end = min(k_end_iter * TILE_K, K);

        // Initialize accumulators
        simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
        for (uint mi = 0; mi < SG_M_TILES; ++mi) {
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }

        // -------------------------------------------------------------------
        // Main K-loop (only over this slice's range)
        // -------------------------------------------------------------------
        const uint k_iters = div_ceil(k_end - k_start, TILE_K);
        uint buf_idx = 0;

        if (k_iters > 0) {
            // Initial tile load
            {
                const uint k_block = k_start;
                const uint k_remaining = min(TILE_K, k_end - k_block);
                const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_K;
                    uint col = flat_idx % TILE_K;
                    uint global_row = tg_row + row;
                    uint global_col = k_block + col;

                    half val = 0.0h;
                    if (global_row < M && global_col < K && col < k_remaining) {
                        val = A[global_row * K + global_col];
                    }
                    A_tiles[0][row][col] = val;
                }

                const uint packed_per_thread = (TILE_K * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
                for (uint i = 0; i < packed_per_thread; ++i) {
                    uint flat_packed_idx = thread_idx * packed_per_thread + i;
                    uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
                    uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

                    uint global_n = tg_col + n_idx;
                    uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

                    uint scale_k = global_k_base / group_size;
                    half s = 1.0h;
                    if (global_n < N && global_k_base < K && scale_k < div_ceil(K, group_size)) {
                        s = scales[scale_k * N + global_n];
                    }

                    uint packed = 0;
                    uint b_row = global_k_base / FP4_PER_UINT;
                    if (global_n < N && b_row < div_ceil(K, FP4_PER_UINT) && global_k_base < k_end) {
                        packed = B[b_row * N + global_n];
                    }

                    uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
                    half vals[8];
                    unpack_fp4x8(packed, s, vals);
                    for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K; ++v) {
                        if (n_idx < TILE_N) {
                            if (tile_k_base + v >= k_remaining) {
                                B_tiles[0][tile_k_base + v][n_idx] = 0.0h;
                            } else {
                                B_tiles[0][tile_k_base + v][n_idx] = vals[v];
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kt = 0; kt < k_iters; ++kt) {
                uint k_block = k_start + kt * TILE_K;
                uint k_remaining = min(TILE_K, k_end - k_block);
                uint next_k = k_block + TILE_K;
                uint buf_load = 1 - buf_idx;

                if (next_k < k_end) {
                    const uint next_remaining = min(TILE_K, k_end - next_k);
                    const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
                    for (uint i = 0; i < elems_per_thread; ++i) {
                        uint flat_idx = thread_idx * elems_per_thread + i;
                        uint row = flat_idx / TILE_K;
                        uint col = flat_idx % TILE_K;
                        uint global_row = tg_row + row;
                        uint global_col = next_k + col;

                        half val = 0.0h;
                        if (global_row < M && global_col < K && col < next_remaining) {
                            val = A[global_row * K + global_col];
                        }
                        A_tiles[buf_load][row][col] = val;
                    }

                    const uint packed_per_thread = (TILE_K * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
                    for (uint i = 0; i < packed_per_thread; ++i) {
                        uint flat_packed_idx = thread_idx * packed_per_thread + i;
                        uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
                        uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

                        uint global_n = tg_col + n_idx;
                        uint global_k_base = next_k + k_group_in_tile * FP4_PER_UINT;

                        uint scale_k = global_k_base / group_size;
                        half s = 1.0h;
                        if (global_n < N && global_k_base < K && scale_k < div_ceil(K, group_size)) {
                            s = scales[scale_k * N + global_n];
                        }

                        uint packed = 0;
                        uint b_row = global_k_base / FP4_PER_UINT;
                        if (global_n < N && b_row < div_ceil(K, FP4_PER_UINT) && global_k_base < k_end) {
                            packed = B[b_row * N + global_n];
                        }

                        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
                        half vals[8];
                        unpack_fp4x8(packed, s, vals);
                        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K; ++v) {
                            if (n_idx < TILE_N) {
                                if (tile_k_base + v >= next_remaining) {
                                    B_tiles[buf_load][tile_k_base + v][n_idx] = 0.0h;
                                } else {
                                    B_tiles[buf_load][tile_k_base + v][n_idx] = vals[v];
                                }
                            }
                        }
                    }
                }

                // Compute via simdgroup MMA
                uint active_k_tiles = div_ceil(k_remaining, 8u);
                for (uint kt_inner = 0; kt_inner < active_k_tiles; ++kt_inner) {
                    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                        simdgroup_matrix<half, 8, 8> a_frag;
                        simdgroup_load(a_frag,
                                       &A_tiles[buf_idx][sg_row_offset + mi * 8][kt_inner * 8],
                                       TILE_K);

                        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                            simdgroup_matrix<half, 8, 8> b_frag;
                            simdgroup_load(b_frag,
                                           &B_tiles[buf_idx][kt_inner * 8][sg_col_offset + ni * 8],
                                           TILE_N);

                            simdgroup_multiply_accumulate(acc[mi][ni],
                                                          a_frag,
                                                          b_frag,
                                                          acc[mi][ni]);
                        }
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf_idx = buf_load;
            }
        }

        // -------------------------------------------------------------------
        // Store results: direct write or atomic reduction
        //
        // If parallel == 1, each output tile is computed by exactly one
        // threadgroup, so write directly to C.
        //
        // If parallel > 1, multiple threadgroups compute partial K-sums.
        // We use a two-phase reduction:
        //   Phase 1: Each slice writes its partial sum to reduction_buf
        //   Phase 2: The last slice to finish (tracked by atomic counter in
        //            locks[]) reduces all partial sums and writes to C
        //
        // This avoids FP16 atomicAdd (which Metal doesn't support natively)
        // by serializing the final reduction to one threadgroup per tile.
        // -------------------------------------------------------------------

        // Per-simdgroup staging buffer for edge-case stores.
        // Each simdgroup needs its own 8x8 staging area to avoid races when
        // multiple simdgroups simultaneously need the slow path (partial tiles).
        threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

        if (parallel == 1) {
            // Direct store: no reduction needed
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                uint out_row = tg_row + sg_row_offset + mi * 8;
                if (out_row >= M) {
                    continue;
                }
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint out_col = tg_col + sg_col_offset + ni * 8;

                    if (out_row + 8 <= M && out_col + 8 <= N) {
                        simdgroup_store(acc[mi][ni],
                                        C + out_row * N + out_col,
                                        N);
                    } else {
                        simdgroup_store(acc[mi][ni], &staging[simd_id][0][0], 8);
                        threadgroup_barrier(mem_flags::mem_threadgroup);

                        for (uint elem = simd_lane; elem < 64; elem += 32) {
                            uint r = elem / 8;
                            uint c = elem % 8;
                            uint gr = out_row + r;
                            uint gc = out_col + c;
                            if (gr < M && gc < N) {
                                C[gr * N + gc] = staging[simd_id][r][c];
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
            }
        } else {
            // Phase 1: Write partial sum to reduction_buf[k_slice]
            // Layout: reduction_buf[k_slice * M * N + row * N + col]
            device half* slice_buf = reduction_buf + k_slice * M * N;

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                uint out_row = tg_row + sg_row_offset + mi * 8;
                if (out_row >= M) {
                    continue;
                }
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint out_col = tg_col + sg_col_offset + ni * 8;

                    if (out_row + 8 <= M && out_col + 8 <= N) {
                        simdgroup_store(acc[mi][ni],
                                        slice_buf + out_row * N + out_col,
                                        N);
                    } else {
                        simdgroup_store(acc[mi][ni], &staging[simd_id][0][0], 8);
                        threadgroup_barrier(mem_flags::mem_threadgroup);

                        for (uint elem = simd_lane; elem < 64; elem += 32) {
                            uint r = elem / 8;
                            uint c = elem % 8;
                            uint gr = out_row + r;
                            uint gc = out_col + c;
                            if (gr < M && gc < N) {
                                slice_buf[gr * N + gc] = staging[simd_id][r][c];
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
            }

            // Ensure all threads in this threadgroup have completed their
            // device memory stores to reduction_buf before any thread
            // increments the atomic counter. mem_device on the barrier
            // flushes device writes from ALL threads in this threadgroup,
            // which atomic_thread_fence alone cannot do (it only orders
            // the calling thread's writes).
            threadgroup_barrier(mem_flags::mem_device);

            // Phase 2: Atomic counter to determine if we're the last slice.
            // The preceding threadgroup_barrier(mem_device) already flushed
            // this TG's device stores. The atomic uses acq_rel so that the
            // "last" TG (which observes prev == parallel-1) acquires all
            // prior releasers' device writes (other slices' partial sums).
            threadgroup int is_last_slice;
            if (thread_idx == 0) {
                is_last_slice = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_idx == 0) {
                // Note: Metal doesn't support memory_order_acq_rel.
                // We use memory_order_relaxed here because the surrounding
                // threadgroup_barrier(mem_flags::mem_device) calls provide
                // the necessary ordering guarantees for device memory.
                int prev = atomic_fetch_add_explicit(&locks[tile_linear],
                                                    1,
                                                    memory_order_relaxed);
                is_last_slice = (prev == int(parallel) - 1);
            }
            // Broadcast is_last_slice to all threads via threadgroup memory.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (is_last_slice) {
                // Thread 0 acquired via acq_rel above, but other threads in
                // this TG haven't. mem_device barrier ensures all threads
                // observe the device writes from other slices' partial sums.
                threadgroup_barrier(mem_flags::mem_device);

                // Reset the lock for next dispatch
                if (thread_idx == 0) {
                    atomic_store_explicit(&locks[tile_linear], 0, memory_order_relaxed);
                }

                // Ordered reduction: each thread sums partial sums in a FIXED
                // order (s = 0, 1, ..., parallel-1). This guarantees bit-exact
                // determinism regardless of which physical threadgroup arrives
                // last at the atomic counter. The FP addition order is invariant
                // across invocations because:
                //   1. The slice->partial_sum mapping is deterministic (k_slice
                //      derived from work_idx, not from scheduling order)
                //   2. The summation iterates slices in ascending index order
                //   3. No atomic FP adds are used; only the integer lock is atomic
                //
                // Tile covers TILE_M rows x TILE_N cols = 4096 elements
                // 128 threads -> 32 elements per thread
                const uint tile_elems = TILE_M * TILE_N;
                const uint elems_per_thread = tile_elems / THREADS_PER_TG;

                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_N;
                    uint col = flat_idx % TILE_N;
                    uint gr = tg_row + row;
                    uint gc = tg_col + col;

                    if (gr < M && gc < N) {
                        half sum = 0.0h;
                        for (uint s = 0; s < parallel; ++s) {
                            sum += reduction_buf[s * M * N + gr * N + gc];
                        }
                        C[gr * N + gc] = sum;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Helper kernel: Zero-initialize reduction buffers and lock arrays
//
// Must be dispatched before marlin_gemm_fp4_striped when parallel > 1.
// Grid: ceil(total_elements / THREADS_PER_TG) threadgroups, 1D dispatch.
// ===========================================================================

kernel void marlin_zero_reduction(
    device half* reduction_buf       [[buffer(0)]],
    device atomic_int* locks         [[buffer(1)]],
    constant uint& buf_elems         [[buffer(2)]],  // parallel * M * N
    constant uint& num_locks         [[buffer(3)]],  // m_tiles * n_tiles
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid < buf_elems) {
        reduction_buf[tid] = 0.0h;
    }
    if (tid < num_locks) {
        atomic_store_explicit(&locks[tid], 0, memory_order_relaxed);
    }
}

// ===========================================================================
// Fused Dequant-GEMM Kernel
//
// KEY INNOVATION: Dequantize in registers, not in threadgroup memory.
//
// The kernels above (marlin_gemm_fp4, marlin_gemm_fp4_striped) follow a
// "separate" approach: dequantize the full B tile into threadgroup memory
// (B_tile[TILE_K][TILE_N]) before computing. This requires:
//   1. A full threadgroup_barrier between dequant and compute
//   2. 4096 bytes of threadgroup memory for B_tile alone
//   3. All 128 threads cooperating to fill B_tile before any can compute
//
// The fused kernel eliminates this bottleneck. Each simdgroup dequantizes
// only the 8x8 B sub-tile it needs, directly from global memory into a
// 128-byte per-simdgroup staging buffer. This staging buffer is immediately
// consumed by simdgroup_load in the same iteration.
//
// Benefits:
//   - Threadgroup memory: 4096B (A_tile) + 512B (B staging) = 4608B
//     vs 4096B (A) + 4096B (B) = 8192B in the separate approach
//   - No cross-simdgroup barrier for B dequant: each simdgroup works
//     independently on its portion, using only simdgroup_barrier
//   - Dequant compute (ALU) overlaps with A_tile reads from other simdgroups
//   - The 128-byte staging buffer lives in L1 cache (64KB on M4 Max)
//
// Trade-off:
//   - Each packed uint32 from B is loaded K_TILES times per k_block
//     (once per sub-tile, 4x for TILE_K=32). These are L2 hits since
//     the access pattern has high temporal locality and M4 Max has 4MB L2.
//
// Net: reduced threadgroup memory enables higher occupancy on the GPU,
// and the eliminated full-threadgroup barrier removes the dequant→compute
// pipeline stall entirely.
// ===========================================================================

// ---- Fused dequant primitives (inlined for register-resident execution) -----

constant constexpr uint32_t FUSED_MAGIC_BIAS = 0x64006400u;
constant constexpr uint32_t FUSED_LO_MASK    = 0x000F000Fu;

/// INT4 (U4) dequant: magic-bias trick, 8 values from packed uint32.
/// Scale and zero_point applied in registers.
/// Includes numerical stability guards for extreme zero_point values.
inline void fused_dequant_u4x8(uint32_t packed,
                                half scale,
                                half zero_point,
                                thread half *out) {
    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);

    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;

    uint32_t n1 = ((packed >> 4u) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v1 = as_type<half2>(n1) - bias;

    uint32_t n2 = ((packed >> 8u) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v2 = as_type<half2>(n2) - bias;

    uint32_t n3 = ((packed >> 12u) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v3 = as_type<half2>(n3) - bias;

    // v{i}.x = nibble i (lo 16-bit lane), v{i}.y = nibble i+4 (hi lane)
    // Guard each output against NaN/Inf from extreme scales
    out[0] = safe_dequant_u4_value(v0.x, scale, zero_point);
    out[1] = safe_dequant_u4_value(v1.x, scale, zero_point);
    out[2] = safe_dequant_u4_value(v2.x, scale, zero_point);
    out[3] = safe_dequant_u4_value(v3.x, scale, zero_point);
    out[4] = safe_dequant_u4_value(v0.y, scale, zero_point);
    out[5] = safe_dequant_u4_value(v1.y, scale, zero_point);
    out[6] = safe_dequant_u4_value(v2.y, scale, zero_point);
    out[7] = safe_dequant_u4_value(v3.y, scale, zero_point);
}

/// FP4 (E2M1) dequant: branchless scalar, select() for subnormal handling.
inline half fused_dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    // Subnormal (exp=0): 0.0 or 0.25
    half sub_mag = half(man_bit) * half(0.5h);
    // Normal (exp>0): 2^(exp-1) * (1 + mantissa*0.5)
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));

    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

/// FP4 dequant: 8 values from packed uint32, scale applied.
/// Includes numerical stability guards for extreme scale values.
inline void fused_dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    out[0] = safe_dequant(fused_dequant_fp4_scalar((packed >>  0) & 0xF), scale);
    out[1] = safe_dequant(fused_dequant_fp4_scalar((packed >>  4) & 0xF), scale);
    out[2] = safe_dequant(fused_dequant_fp4_scalar((packed >>  8) & 0xF), scale);
    out[3] = safe_dequant(fused_dequant_fp4_scalar((packed >> 12) & 0xF), scale);
    out[4] = safe_dequant(fused_dequant_fp4_scalar((packed >> 16) & 0xF), scale);
    out[5] = safe_dequant(fused_dequant_fp4_scalar((packed >> 20) & 0xF), scale);
    out[6] = safe_dequant(fused_dequant_fp4_scalar((packed >> 24) & 0xF), scale);
    out[7] = safe_dequant(fused_dequant_fp4_scalar((packed >> 28) & 0xF), scale);
}

// ===========================================================================
// Fused FP4 GEMM
//
// C[M,N] = A[M,K] @ dequant(B_packed[K/8,N], scales[K/group_size,N])
//
// Same tile dimensions as marlin_gemm_fp4 (64x64x32) but B_tile is replaced
// with per-simdgroup B_staging[8][8].
//
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads.
// ===========================================================================

kernel void marlin_gemm_fused_fp4(
    device const half* A         [[buffer(0)]],   // [M, K] row-major FP16
    device const uint* B         [[buffer(1)]],   // [K/8, N] packed FP4
    device const half* scales    [[buffer(2)]],   // [K/group_size, N]
    device half* C               [[buffer(3)]],   // [M, N] row-major FP16
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // --- Memory allocation ---
    threadgroup half A_tiles[2][TILE_M][TILE_K];            // 2x4096B shared
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];   // 512B per-sg
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    // --- Tile assignment ---
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);  // 0 or 16
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);  // 0 or 32

    // --- Accumulators ---
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint k_packs = div_ceil(K, FP4_PER_UINT);

    // =========================================================================
    // Main K-reduction loop
    // =========================================================================

    const uint k_tiles = div_ceil(K, TILE_K);
    uint buf_idx = 0;

    if (k_tiles > 0) {
        // Initial A tile load
        {
            const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;  // 16
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tiles[0][row][col] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_tile = 0; k_tile < k_tiles; ++k_tile) {
            uint k_block = k_tile * TILE_K;
            uint next_k = k_block + TILE_K;
            uint buf_load = 1 - buf_idx;

            if (next_k < K) {
                const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_K;
                    uint col = flat_idx % TILE_K;
                    uint global_row = tg_row + row;
                    uint global_col = next_k + col;

                    half val = (global_row < M && global_col < K)
                               ? A[global_row * K + global_col]
                               : half(0.0h);
                    A_tiles[buf_load][row][col] = val;
                }
            }

            // =====================================================================
            // Inner loop: fused dequant + compute
            //
            // For each K sub-tile (kt = 0..3, covering 8 K values each):
            //   For each output tile (mi, ni):
            //     1. simdgroup_load A fragment from A_tiles[buf_idx]
            //     2. Lanes 0-7 load+dequant one column each of B sub-tile
            //     3. Write to B_staging
            //     4. simdgroup_barrier (NOT threadgroup_barrier)
            //     5. simdgroup_load B fragment from B_staging
            //     6. simdgroup_multiply_accumulate
            // =====================================================================

            for (uint kt = 0; kt < K_TILES; ++kt) {
                uint k_sub_base = k_block + kt * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<
#if defined(USE_BF16_INPUTS)
                    float
#else
                    half
#endif
                    , 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_idx][sg_row_offset + mi * 8][kt * 8],
                               TILE_K);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    // --- Fused B dequant: lanes 0-7 each handle one column ---
                    if (simd_lane < 8) {
                        uint b_col = b_col_base + simd_lane;
                        half dequant_vals[8];

                        if (b_col < N && k_pack_idx < k_packs) {
                            // Coalesced global load: adjacent lanes → adjacent cols
                            uint32_t packed = B[k_pack_idx * N + b_col];

                            // Coalesced scale load: same group row, adjacent cols
                            half scale = scales[group_idx * N + b_col];

                            // Dequant in registers: pure ALU, no memory traffic
                            fused_dequant_fp4x8(packed, scale, dequant_vals);
                        } else {
                            for (uint v = 0; v < 8; ++v)
                                dequant_vals[v] = half(0.0h);
                        }

                        // Write column to staging: [row][lane]
                        for (uint row = 0; row < 8; ++row)
                            B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }

                    // Lightweight sync: only 32 threads, not 128
                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    // Load B fragment and compute
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_idx = buf_load;
        }
    }

    // =========================================================================
    // Store results
    // =========================================================================

    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// Fused FP4 GEMM (FP32 accumulation)
//
// Same as marlin_gemm_fused_fp4, but accumulates in FP32 and converts to
// FP16 on store. Useful for very long K reductions.
// ===========================================================================

kernel void marlin_gemm_fused_fp4_fp32acc(
    device const
#if defined(USE_BF16_INPUTS)
        ushort* A               [[buffer(0)]],   // [M, K] row-major BF16
#else
        half* A                 [[buffer(0)]],   // [M, K] row-major FP16
#endif
    device const uint* B         [[buffer(1)]],   // [K/8, N] packed FP4
    device const half* scales    [[buffer(2)]],   // [K/group_size, N]
    device
#if defined(USE_BF16_OUTPUTS)
        ushort* C               [[buffer(3)]],   // [M, N] row-major BF16
#else
        half* C                 [[buffer(3)]],   // [M, N] row-major FP16
#endif
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // --- Memory allocation ---
    threadgroup
#if defined(USE_BF16_INPUTS)
        float A_tiles[2][TILE_M][TILE_K];
    threadgroup float B_staging[SIMDGROUPS_PER_TG][8][8];
#else
        half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
#endif
    threadgroup float staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint k_packs = div_ceil(K, FP4_PER_UINT);

    const uint k_tiles = div_ceil(K, TILE_K);
    uint buf_idx = 0;

    if (k_tiles > 0) {
        // Initial A tile load
        #if defined(USE_BF16_INPUTS)
        load_A_tile_bf16_fp32(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
        #else
        {
            const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tiles[0][row][col] = val;
            }
        }
        #endif

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_tile = 0; k_tile < k_tiles; ++k_tile) {
            uint k_block = k_tile * TILE_K;
            uint next_k = k_block + TILE_K;
            uint buf_load = 1 - buf_idx;

            if (next_k < K) {
                #if defined(USE_BF16_INPUTS)
                load_A_tile_bf16_fp32(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
                #else
                const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_K;
                    uint col = flat_idx % TILE_K;
                    uint global_row = tg_row + row;
                    uint global_col = next_k + col;

                    half val = (global_row < M && global_col < K)
                               ? A[global_row * K + global_col]
                               : half(0.0h);
                    A_tiles[buf_load][row][col] = val;
                }
                #endif
            }

            for (uint kt = 0; kt < K_TILES; ++kt) {
                uint k_sub_base = k_block + kt * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_idx][sg_row_offset + mi * 8][kt * 8],
                               TILE_K);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    if (simd_lane < 8) {
                        uint b_col = b_col_base + simd_lane;
#if defined(USE_BF16_INPUTS)
                        float dequant_vals[8];
                        if (b_col < N && k_pack_idx < k_packs) {
                            uint32_t packed = B[k_pack_idx * N + b_col];
                            float scale = float(scales[group_idx * N + b_col]);
                            for (uint v = 0; v < 8; ++v) {
                                uint nibble = (packed >> (v * 4)) & 0xF;
                                dequant_vals[v] = safe_dequant_fp4_to_float(nibble, scale);
                            }
                        } else {
                            for (uint v = 0; v < 8; ++v)
                                dequant_vals[v] = 0.0f;
                        }
#else
                        half dequant_vals[8];
                        if (b_col < N && k_pack_idx < k_packs) {
                            uint32_t packed = B[k_pack_idx * N + b_col];
                            half scale = scales[group_idx * N + b_col];
                            fused_dequant_fp4x8(packed, scale, dequant_vals);
                        } else {
                            for (uint v = 0; v < 8; ++v)
                                dequant_vals[v] = half(0.0h);
                        }
#endif
                        for (uint row = 0; row < 8; ++row)
                            B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }

                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_matrix<
#if defined(USE_BF16_INPUTS)
                        float
#else
                        half
#endif
                        , 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_idx = buf_load;
        }
    }

    #if defined(USE_BF16_OUTPUTS)
        store_results_fp32_bf16(acc, C, M, N, tg_row, tg_col,
                                sg_row_offset, sg_col_offset, simd_lane, staging[simd_id]);
    #else
        store_results_fp32(acc, C, M, N, tg_row, tg_col,
                           sg_row_offset, sg_col_offset, simd_lane, staging[simd_id]);
    #endif
}

// ===========================================================================
// Fused INT4 (U4) GEMM - with per-group zero points
//
// Same fused architecture as above, but uses INT4 magic-bias dequant with
// asymmetric quantization (scale + zero_point per group).
// ===========================================================================

kernel void marlin_gemm_fused_u4(
    device const half* A         [[buffer(0)]],
    device const uint* B         [[buffer(1)]],   // [K/8, N] packed u4
    device const half* scales    [[buffer(2)]],   // [K/group_size, N]
    device const half* zeros     [[buffer(3)]],   // [K/group_size, N] zero points
    device half* C               [[buffer(4)]],
    constant uint& M             [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& group_size    [[buffer(8)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint k_packs = div_ceil(K, FP4_PER_UINT);

    const uint k_tiles = div_ceil(K, TILE_K);
    uint buf_idx = 0;

    if (k_tiles > 0) {
        {
            const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tiles[0][row][col] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_tile = 0; k_tile < k_tiles; ++k_tile) {
            uint k_block = k_tile * TILE_K;
            uint next_k = k_block + TILE_K;
            uint buf_load = 1 - buf_idx;

            if (next_k < K) {
                const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_K;
                    uint col = flat_idx % TILE_K;
                    uint global_row = tg_row + row;
                    uint global_col = next_k + col;

                    half val = (global_row < M && global_col < K)
                               ? A[global_row * K + global_col]
                               : half(0.0h);
                    A_tiles[buf_load][row][col] = val;
                }
            }

            for (uint kt = 0; kt < K_TILES; ++kt) {
                uint k_sub_base = k_block + kt * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_idx][sg_row_offset + mi * 8][kt * 8],
                               TILE_K);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    if (simd_lane < 8) {
                        uint b_col = b_col_base + simd_lane;
                        half dequant_vals[8];

                        if (b_col < N && k_pack_idx < k_packs) {
                            uint32_t packed = B[k_pack_idx * N + b_col];
                            half scale = scales[group_idx * N + b_col];
                            half zero = zeros[group_idx * N + b_col];

                            fused_dequant_u4x8(packed, scale, zero, dequant_vals);
                        } else {
                            for (uint v = 0; v < 8; ++v)
                                dequant_vals[v] = half(0.0h);
                        }

                        for (uint row = 0; row < 8; ++row)
                            B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }

                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_idx = buf_load;
        }
    }

    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// FP16 Reference GEMM (no dequantization, pure MMA throughput ceiling)
//
// C[M,N] = A[M,K] @ B[K,N]  (both A and B are FP16, row-major)
//
// This kernel uses the exact same tiling (64x64x32), double-buffered pipeline,
// simdgroup MMA compute, and store logic as marlin_gemm_fp4. The ONLY
// difference: B is loaded as raw FP16 via load_B_tile_fp16 instead of packed
// FP4 with dequantization via load_B_tile_dequant.
//
// Purpose: establish the theoretical peak MMA throughput on the target GPU.
// Any gap between this kernel and the quantized kernels is directly
// attributable to dequantization overhead (ALU for nibble extraction +
// FP4/INT4 to FP16 conversion + reduced memory bandwidth efficiency from
// the packed format's irregular access pattern).
//
// Note on memory bandwidth: the FP16 kernel loads 4x more bytes for B
// (16 bits vs 4 bits per weight). This means the FP16 kernel is MORE
// bandwidth-bound than the quantized kernels for large K. The intended
// comparison is compute-bound regime (small M, large N) where MMA
// throughput dominates.
//
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads.
// ===========================================================================

kernel void marlin_gemm_fp16_pipelined(
    device const half* A     [[buffer(0)]],  // [M, K] row-major FP16
    device const half* B     [[buffer(1)]],  // [K, N] row-major FP16
    device half* C           [[buffer(2)]],  // [M, N] row-major FP16
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    uint3 tgid               [[threadgroup_position_in_grid]],
    uint simd_lane           [[thread_index_in_simdgroup]],
    uint simd_id             [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory (identical footprint to FP4 kernel)
    threadgroup half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][TILE_K][TILE_N];
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    uint buf_compute = 0;

    // --- Prologue: Load first K-tile into buffer 0 ---
    load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_fp16(B, B_tiles[0], K, N, tg_col, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main pipeline loop ---
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * TILE_K;
        uint next_k = k_offset + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Load NEXT K-tile into alternate buffer (overlapped with compute)
        if (next_k < K) {
            load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_fp16(B, B_tiles[buf_load], K, N, tg_col, next_k, thread_idx);
        }

        // Compute on current buffer: pure MMA, zero dequant overhead
        compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                           acc, sg_row_offset, sg_col_offset);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // --- Store ---
    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// ASYNC COPY KERNELS (Metal 3.1+, Apple Silicon M3+)
//
// These kernels use simdgroup_async_copy for true asynchronous memory
// transfers that can overlap with compute. On M3+ GPUs, this provides
// better latency hiding than the cooperative load approach.
//
// The key difference from the double-buffered kernels above:
// - Cooperative load: ALL threads participate in loading, then ALL compute
// - Async copy: Load is issued asynchronously, compute can proceed immediately
//
// simdgroup_async_copy signature:
//   void simdgroup_async_copy<T, N>(threadgroup T* dest, const device T* src)
//   void simdgroup_async_copy_fence() - ensures all async copies complete
//
// Note: simdgroup_async_copy requires contiguous memory regions. For the
// quantized B tile, we still need to dequantize, so we use a hybrid approach:
// - A tile: Pure async copy (contiguous FP16 data)
// - B tile: Async copy packed data, then dequantize in-place
// ===========================================================================

#if __METAL_VERSION__ >= 310

// ---------------------------------------------------------------------------
// Async copy helpers for A tile (pure FP16, contiguous)
// ---------------------------------------------------------------------------

/// Issue async copy for one row of A tile.
/// Each simdgroup handles TILE_M/SIMDGROUPS_PER_TG rows.
inline void async_load_A_tile_row(
    device const half* A,
    threadgroup half* A_row_dest,
    uint M, uint K,
    uint global_row, uint k_block,
    uint simd_lane
) {
    // Each simdgroup lane copies TILE_K/32 elements
    const uint elems_per_lane = TILE_K / 32;
    uint base_col = simd_lane * elems_per_lane;

    if (global_row < M) {
        device const half* src = A + global_row * K + k_block + base_col;

        for (uint i = 0; i < elems_per_lane && (k_block + base_col + i) < K; ++i) {
            A_row_dest[base_col + i] = src[i];
        }
    } else {
        for (uint i = 0; i < elems_per_lane; ++i) {
            A_row_dest[base_col + i] = 0.0h;
        }
    }
}

/// Cooperative async load for entire A tile.
/// Uses simdgroup-cooperative pattern: each simdgroup loads multiple rows.
inline void async_load_A_tile_cooperative(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint simd_id,
    uint simd_lane
) {
    // Each of 4 simdgroups handles 16 rows (TILE_M / SIMDGROUPS_PER_TG)
    const uint rows_per_sg = TILE_M / SIMDGROUPS_PER_TG;
    const uint row_start = simd_id * rows_per_sg;

    for (uint r = 0; r < rows_per_sg; ++r) {
        uint local_row = row_start + r;
        uint global_row = tg_row + local_row;
        async_load_A_tile_row(A, &A_buf[local_row][0], M, K, global_row, k_block, simd_lane);
    }
}

// ---------------------------------------------------------------------------
// Async copy + dequant for B tile (packed FP4 -> FP16)
//
// Strategy: Each simdgroup async-loads its portion of packed B data into
// a staging buffer, then dequantizes into the final B_buf.
// ---------------------------------------------------------------------------

/// Load packed FP4 data into staging buffer, then dequantize.
/// This is a two-phase approach: async load packed data, barrier, dequantize.
inline void async_load_B_tile_staged(
    device const uint* B,
    device const half* scales,
    threadgroup uint (&B_packed_buf)[TILE_K / FP4_PER_UINT][TILE_N],
    threadgroup half (&B_buf)[TILE_K][TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint simd_id,
    uint simd_lane,
    uint thread_idx
) {
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    const uint scale_tiles = div_ceil(K, group_size);

    // Phase 1: Load packed data (each simdgroup handles a portion)
    // B layout: [K/8, N] packed uint32
    const uint packed_k_dim = TILE_K / FP4_PER_UINT;  // 4
    const uint total_packed = packed_k_dim * TILE_N;   // 4 * 64 = 256
    const uint packed_per_thread = total_packed / THREADS_PER_TG;  // 2

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_idx / packed_k_dim;
        uint k_pack_idx = flat_idx % packed_k_dim;

        uint global_n = tg_col + n_idx;
        uint global_k_pack = (k_block / FP4_PER_UINT) + k_pack_idx;

        uint packed = 0;
        if (global_n < N && global_k_pack < k_packs) {
            packed = B[global_k_pack * N + global_n];
        }
        B_packed_buf[k_pack_idx][n_idx] = packed;
    }

    // Barrier: ensure packed data is loaded before dequant
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Dequantize packed data to B_buf
    // Each thread handles 2 packed words = 16 FP4 values
    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_idx / packed_k_dim;
        uint k_pack_idx = flat_idx % packed_k_dim;

        if (n_idx >= TILE_N) continue;

        uint32_t packed = B_packed_buf[k_pack_idx][n_idx];

        // Get scale for this group
        uint global_k_base = k_block + k_pack_idx * FP4_PER_UINT;
        uint global_n = tg_col + n_idx;
        uint scale_k = global_k_base / group_size;
        half s = 1.0h;
        if (global_n < N && scale_k < scale_tiles) {
            s = scales[scale_k * N + global_n];
        }

        // Dequantize 8 values
        uint tile_k_base = k_pack_idx * FP4_PER_UINT;
        for (uint v = 0; v < FP4_PER_UINT; ++v) {
            uint nibble = (packed >> (v * 4)) & 0xF;
            uint global_k = global_k_base + v;
            half val = (global_k < K) ? dequant_fp4(nibble, s) : 0.0h;
            B_buf[tile_k_base + v][n_idx] = val;
        }
    }
}

// ===========================================================================
// Kernel: Async copy double-buffered GEMM (Metal 3.1+)
//
// Uses staged async copy for improved memory latency hiding:
// - A tile: Cooperative async load (pure FP16)
// - B tile: Async load packed + dequant (staged approach)
//
// Pipeline:
//   Prologue: Async load tile[0] → buf[0], wait, dequant
//   Loop k:
//     Issue async load tile[k+1] → buf[1-current] (don't wait yet)
//     Compute on buf[current]
//     Wait for async load to complete
//     Dequant B data in buf[1-current]
//     Swap buffers
//
// The compute on buf[current] overlaps with the async load into buf[1-current].
// ===========================================================================

kernel void marlin_gemm_fp4_async(
    device const half* A         [[buffer(0)]],
    device const uint* B         [[buffer(1)]],
    device const half* scales    [[buffer(2)]],
    device half* C               [[buffer(3)]],
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory
    threadgroup half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][TILE_K][TILE_N];

    // Staging buffer for packed B data (reused across iterations)
    threadgroup uint B_packed_staging[TILE_K / FP4_PER_UINT][TILE_N];
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    uint buf_compute = 0;

    // --- Prologue: Load first K-tile into buffer 0 ---
    async_load_A_tile_cooperative(A, A_tiles[0], M, K, tg_row, 0, simd_id, simd_lane);
    async_load_B_tile_staged(B, scales, B_packed_staging, B_tiles[0],
                             K, N, tg_col, 0, group_size, simd_id, simd_lane, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main pipeline loop ---
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * TILE_K;
        uint next_k = k_offset + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Issue async load for NEXT K-tile (will complete during compute)
        bool has_next = (next_k < K);
        if (has_next) {
            // Start loading A tile asynchronously
            async_load_A_tile_cooperative(A, A_tiles[buf_load], M, K, tg_row, next_k, simd_id, simd_lane);
        }

        // Compute on current buffer while load is in flight
        compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                           acc, sg_row_offset, sg_col_offset);

        // Wait for A load and perform staged B load+dequant
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (has_next) {
            async_load_B_tile_staged(B, scales, B_packed_staging, B_tiles[buf_load],
                                     K, N, tg_col, next_k, group_size, simd_id, simd_lane, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        buf_compute = buf_load;
    }

    // --- Epilogue: Store accumulated results ---
    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

// ===========================================================================
// Kernel: Async copy with interleaved dequant (deeper pipeline)
//
// More aggressive pipelining: dequantization of buf[i+1] overlaps with
// compute on buf[i]. This requires careful synchronization but can further
// improve throughput when dequant and compute have similar latencies.
//
// Pipeline structure:
//   Stage 0: Load packed B[k+2] → staging
//   Stage 1: Dequant staging → B_tiles[k+1], Load A[k+1]
//   Stage 2: Compute on A[k], B[k]
//
// Memory: 3 buffers for A, 3 buffers for B (triple buffered)
// ===========================================================================

kernel void marlin_gemm_fp4_async_deep(
    device const half* A         [[buffer(0)]],
    device const uint* B         [[buffer(1)]],
    device const half* scales    [[buffer(2)]],
    device half* C               [[buffer(3)]],
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    // Triple-buffered threadgroup memory for deep pipelining
    threadgroup half A_tiles[NUM_STAGES][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_STAGES][TILE_K][TILE_N];
    threadgroup uint B_packed_staging[NUM_STAGES][TILE_K / FP4_PER_UINT][TILE_N];
    threadgroup half staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // --- Prologue: Prime the pipeline with first 2 tiles ---
    // Load tile 0
    load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_dequant(B, scales, B_tiles[0], K, N, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load tile 1 if available
    if (num_k_tiles > 1) {
        load_A_tile(A, A_tiles[1], M, K, tg_row, TILE_K, thread_idx);
        load_B_tile_dequant(B, scales, B_tiles[1], K, N, tg_col, TILE_K, group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint stage_compute = 0;
    uint stage_load = 2;

    // --- Main pipeline loop ---
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint next_k = (kt + 2) * TILE_K;

        // Stage: Load tile[k+2] if available (overlaps with compute)
        bool has_prefetch = (next_k < K);
        if (has_prefetch) {
            load_A_tile(A, A_tiles[stage_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant(B, scales, B_tiles[stage_load], K, N, tg_col, next_k, group_size, thread_idx);
        }

        // Stage: Compute on tile[k]
        compute_from_tiles(A_tiles[stage_compute], B_tiles[stage_compute],
                           acc, sg_row_offset, sg_col_offset);

        // Rotate stages
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stage_compute = (stage_compute + 1) % NUM_STAGES;
        stage_load = (stage_load + 1) % NUM_STAGES;
    }

    // --- Epilogue: Store accumulated results ---
    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id, staging[simd_id]);
}

#endif // __METAL_VERSION__ >= 310
