// mla_proj.metal - Optimized kernels for MLA (Multi-head Latent Attention) projections
//
// MLA uses compressed KV cache via latent projections:
//   kv_a_proj: [batch, seq, hidden] → [batch, seq, kv_lora_rank]
//   kv_b_proj: [batch, seq, kv_lora_rank] → [batch, seq, n_kv_heads * head_dim]
//
// The latent dimension (kv_lora_rank) is typically small (512-1536), which means:
//   - kv_a_proj has large K (hidden_size, e.g. 4096) but small N (kv_lora_rank)
//   - kv_b_proj has small K (kv_lora_rank) but large N (n_kv_heads * head_dim)
//
// Kernel variants:
//   1. mla_proj_fp4              - Single projection, FP4 quantized weights
//   2. mla_fused_kv_proj_fp4     - Fused kv_a + kv_b (skip intermediate)
//   3. mla_proj_with_rope_fp4    - Projection with fused RoPE on partial output
//   4. mla_decode_proj_fp4       - GEMV for decode phase (single token)
//
// Memory optimizations for small K:
//   - TILE_K_MLA = 16 when kv_lora_rank < 1024, else 32
//   - Use smaller threadgroup for decode (64 threads instead of 128)
//   - Double-buffering may not be needed for small K (single-stage suffices)
//
// Architecture notes:
//   - GLM-4 uses kv_lora_rank=512, head_dim=128
//   - DeepSeek-V2/V3 uses kv_lora_rank=512-1536
//   - RoPE can be applied to decoupled rope_head_dim portion of KV cache

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions for MLA projections
//
// Tuned for small latent dimensions. When K (reduction dim) is small, we
// reduce TILE_K to avoid wasting work and improve memory efficiency.
// ---------------------------------------------------------------------------

constant constexpr uint TILE_M_MLA = 64;     // Same as GEMM for good batch handling
constant constexpr uint TILE_N_MLA = 64;     // Same as GEMM
constant constexpr uint TILE_K_MLA = 16;     // Smaller for small latent dims
constant constexpr uint TILE_K_MLA_LARGE = 32;  // For larger dims (>1024)

constant constexpr uint K_TILES_MLA = TILE_K_MLA / 8;  // 2
constant constexpr uint K_TILES_MLA_LARGE = TILE_K_MLA_LARGE / 8;  // 4

constant constexpr uint SIMDGROUPS_PER_TG_MLA = 4;
constant constexpr uint THREADS_PER_TG_MLA = SIMDGROUPS_PER_TG_MLA * 32;  // 128
constant constexpr uint THREADS_PER_TG_DECODE = 64;  // Smaller for decode GEMV

// Each simdgroup handles 2x4 block of 8x8 tiles (16 rows × 32 cols)
constant constexpr uint SG_M_TILES_MLA = 8;
constant constexpr uint SG_N_TILES_MLA = 2;

constant constexpr uint FP4_PER_UINT = 8;

// ---------------------------------------------------------------------------
// FP4 E2M1 dequantization (branchless, matches main GEMM kernel)
// ---------------------------------------------------------------------------

inline half dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.25h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

inline void dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    out[0] = (half)((float)dequant_fp4_scalar((packed >>  0) & 0xF) * fscale);
    out[1] = (half)((float)dequant_fp4_scalar((packed >>  4) & 0xF) * fscale);
    out[2] = (half)((float)dequant_fp4_scalar((packed >>  8) & 0xF) * fscale);
    out[3] = (half)((float)dequant_fp4_scalar((packed >> 12) & 0xF) * fscale);
    out[4] = (half)((float)dequant_fp4_scalar((packed >> 16) & 0xF) * fscale);
    out[5] = (half)((float)dequant_fp4_scalar((packed >> 20) & 0xF) * fscale);
    out[6] = (half)((float)dequant_fp4_scalar((packed >> 24) & 0xF) * fscale);
    out[7] = (half)((float)dequant_fp4_scalar((packed >> 28) & 0xF) * fscale);
}

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embedding) utilities
//
// RoPE in MLA can be applied to a portion of the latent representation.
// For GLM-4, rope is applied to the first `rope_head_dim` elements.
// ---------------------------------------------------------------------------

// Apply RoPE to a pair of values: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
inline void apply_rope_pair(
    threadgroup half& x,
    threadgroup half& y,
    half cos_val,
    half sin_val
) {
    half x_new = x * cos_val - y * sin_val;
    half y_new = x * sin_val + y * cos_val;
    x = x_new;
    y = y_new;
}

// Apply RoPE to a vector in threadgroup memory
// Assumes rope_dim is even and <= MAX_ROPE_DIM
inline void apply_rope_tg(
    threadgroup half* vec,
    device const half* cos_cache,  // [max_seq, rope_dim/2]
    device const half* sin_cache,  // [max_seq, rope_dim/2]
    uint position,
    uint rope_dim,
    uint tid,
    uint threads_per_tg
) {
    // Each thread handles one or more rotation pairs
    uint pairs = rope_dim / 2;
    for (uint p = tid; p < pairs; p += threads_per_tg) {
        half cos_val = cos_cache[position * pairs + p];
        half sin_val = sin_cache[position * pairs + p];
        apply_rope_pair(vec[p], vec[p + pairs], cos_val, sin_val);
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: Single MLA Projection (kv_a_proj or kv_b_proj)
//
// Optimized for tall-skinny (large M, small N) or short-wide (small K) shapes.
// Uses adaptive TILE_K based on the K dimension.
// ---------------------------------------------------------------------------

// Macro to generate MLA projection kernels with different tile sizes
// TILE_K_VAL: tile size for K dimension, K_TILES_VAL: number of 8x8 tiles in K
#define DEFINE_MLA_PROJ_KERNEL(KERNEL_NAME, TILE_K_VAL, K_TILES_VAL) \
[[kernel]] void KERNEL_NAME( \
    device const half* A            [[buffer(0)]], \
    device const uint* B_packed     [[buffer(1)]], \
    device const half* scales       [[buffer(2)]], \
    device half* out                [[buffer(3)]], \
    constant uint& M                [[buffer(4)]], \
    constant uint& N                [[buffer(5)]], \
    constant uint& K                [[buffer(6)]], \
    constant uint& group_size       [[buffer(7)]], \
    uint3 tgid                      [[threadgroup_position_in_grid]], \
    uint simd_lane                  [[thread_index_in_simdgroup]], \
    uint simd_id                    [[simdgroup_index_in_threadgroup]], \
    uint thread_idx                 [[thread_index_in_threadgroup]] \
) { \
    threadgroup half A_tile[TILE_M_MLA][TILE_K_VAL]; \
    threadgroup half B_tile[TILE_K_VAL][TILE_N_MLA]; \
    \
    uint tg_row = tgid.y * TILE_M_MLA; \
    uint tg_col = tgid.x * TILE_N_MLA; \
    uint sg_row_offset = 0;  // All simdgroups cover all rows \
    uint sg_col_offset = simd_id * (SG_N_TILES_MLA * 8); \
    \
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES_MLA][SG_N_TILES_MLA]; \
    for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) \
        for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) \
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h); \
    \
    uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT; \
    uint scale_tiles = (K + group_size - 1) / group_size; \
    \
    for (uint k_block = 0; k_block < K; k_block += TILE_K_VAL) { \
        { \
            const uint elems_per_thread = (TILE_M_MLA * TILE_K_VAL) / THREADS_PER_TG_MLA; \
            for (uint i = 0; i < elems_per_thread; ++i) { \
                uint flat_idx = thread_idx * elems_per_thread + i; \
                uint row = flat_idx / TILE_K_VAL; \
                uint col = flat_idx % TILE_K_VAL; \
                uint global_row = tg_row + row; \
                uint global_col = k_block + col; \
                half val = (global_row < M && global_col < K) \
                           ? A[global_row * K + global_col] \
                           : half(0.0h); \
                A_tile[row][col] = val; \
            } \
        } \
        { \
            const uint packed_per_thread = (TILE_K_VAL * TILE_N_MLA) / (THREADS_PER_TG_MLA * FP4_PER_UINT); \
            for (uint i = 0; i < packed_per_thread; ++i) { \
                uint flat_packed_idx = thread_idx * packed_per_thread + i; \
                uint n_idx = flat_packed_idx / (TILE_K_VAL / FP4_PER_UINT); \
                uint k_group_in_tile = flat_packed_idx % (TILE_K_VAL / FP4_PER_UINT); \
                uint global_n = tg_col + n_idx; \
                uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT; \
                uint scale_k = global_k_base / group_size; \
                half s = half(1.0h); \
                if (global_n < N && scale_k < scale_tiles) { \
                    s = scales[scale_k * N + global_n]; \
                } \
                uint32_t packed = 0; \
                uint b_row = global_k_base / FP4_PER_UINT; \
                if (global_n < N && b_row < k_packs && global_k_base < K) { \
                    packed = B_packed[b_row * N + global_n]; \
                } \
                uint tile_k_base = k_group_in_tile * FP4_PER_UINT; \
                half vals[8]; \
                dequant_fp4x8(packed, s, vals); \
                for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_VAL; ++v) { \
                    if (n_idx < TILE_N_MLA) { \
                        uint global_k = global_k_base + v; \
                        B_tile[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : half(0.0h); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint kt = 0; kt < K_TILES_VAL; ++kt) { \
            for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) { \
                simdgroup_matrix<half, 8, 8> a_frag; \
                simdgroup_load(a_frag, &A_tile[sg_row_offset + mi * 8][kt * 8], TILE_K_VAL); \
                for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) { \
                    simdgroup_matrix<half, 8, 8> b_frag; \
                    simdgroup_load(b_frag, &B_tile[kt * 8][sg_col_offset + ni * 8], TILE_N_MLA); \
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]); \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) { \
        for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) { \
            uint out_row = tg_row + sg_row_offset + mi * 8; \
            uint out_col = tg_col + sg_col_offset + ni * 8; \
            if (out_row + 8 <= M && out_col + 8 <= N) { \
                simdgroup_store(acc[mi][ni], out + out_row * N + out_col, N); \
            } else if (out_row < M && out_col < N) { \
                threadgroup half out_staging[8][8]; \
                simdgroup_store(acc[mi][ni], &out_staging[0][0], 8); \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
                for (uint elem = simd_lane; elem < 64; elem += 32) { \
                    uint r = elem / 8; \
                    uint c = elem % 8; \
                    if (out_row + r < M && out_col + c < N) { \
                        out[(out_row + r) * N + out_col + c] = out_staging[r][c]; \
                    } \
                } \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
        } \
    } \
}

// Generate kernel variants for different TILE_K values
DEFINE_MLA_PROJ_KERNEL(mla_proj_fp4_k16, TILE_K_MLA, K_TILES_MLA)
DEFINE_MLA_PROJ_KERNEL(mla_proj_fp4_k32, TILE_K_MLA_LARGE, K_TILES_MLA_LARGE)

// ---------------------------------------------------------------------------
// Kernel 2: Fused kv_a + kv_b Projection
//
// Fuses the two-stage projection: hidden → latent → output
// This avoids materializing the intermediate latent tensor.
//
// For each threadgroup handling an M×N_out tile of the final output:
//   1. Load hidden[M, K_hidden] into threadgroup memory
//   2. Compute latent = hidden @ W_a^T (partial K_latent accumulation)
//   3. Immediately multiply latent @ W_b^T and accumulate to output
//
// This is more memory-efficient for inference but requires both weight
// matrices to be accessible. Works best when K_latent is small enough
// to fit in registers (typically <1024).
// ---------------------------------------------------------------------------

[[kernel]] void mla_fused_kv_proj_fp4(
    device const half* hidden           [[buffer(0)]],   // [M, K_hidden]
    device const uint* W_a_packed       [[buffer(1)]],   // [K_hidden/8, K_latent] FP4
    device const half* scales_a         [[buffer(2)]],   // [K_hidden/group_size_a, K_latent]
    device const uint* W_b_packed       [[buffer(3)]],   // [K_latent/8, N_out] FP4
    device const half* scales_b         [[buffer(4)]],   // [K_latent/group_size_b, N_out]
    device half* out                    [[buffer(5)]],   // [M, N_out]
    constant uint& M                    [[buffer(6)]],
    constant uint& K_hidden             [[buffer(7)]],
    constant uint& K_latent             [[buffer(8)]],
    constant uint& N_out                [[buffer(9)]],
    constant uint& group_size_a         [[buffer(10)]],
    constant uint& group_size_b         [[buffer(11)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    // Strategy: Process in M-tiles, computing full latent vector per row
    // then immediately projecting through W_b to final output.
    //
    // Threadgroup handles output tile [TILE_M_MLA, TILE_N_MLA]
    // For each row in the tile:
    //   1. Compute latent[K_latent] = hidden_row @ W_a
    //   2. Compute output[N_out_tile] = latent @ W_b[K_latent, N_out_tile]

    // Threadgroup memory for intermediate latent vectors
    // Each simdgroup computes a portion of rows, stores latent in TG memory
    threadgroup half latent_tile[TILE_M_MLA][TILE_K_MLA_LARGE];  // Max K_latent portion per iteration
    threadgroup half hidden_tile[TILE_M_MLA][TILE_K_MLA_LARGE];
    threadgroup half B_tile[TILE_K_MLA_LARGE][TILE_N_MLA];

    uint tg_row = tgid.y * TILE_M_MLA;
    uint tg_col = tgid.x * TILE_N_MLA;
    uint sg_row_offset = 0;  // All simdgroups cover all rows
    uint sg_col_offset = simd_id * (SG_N_TILES_MLA * 8);

    // Accumulators for final output
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES_MLA][SG_N_TILES_MLA];
    for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi)
        for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint k_packs_a = (K_hidden + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint k_packs_b = (K_latent + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint scale_tiles_a = (K_hidden + group_size_a - 1) / group_size_a;
    uint scale_tiles_b = (K_latent + group_size_b - 1) / group_size_b;

    // Process K_latent in tiles
    for (uint latent_block = 0; latent_block < K_latent; latent_block += TILE_K_MLA_LARGE) {
        // Zero the latent tile accumulator
        for (uint i = thread_idx; i < TILE_M_MLA * TILE_K_MLA_LARGE; i += THREADS_PER_TG_MLA) {
            uint row = i / TILE_K_MLA_LARGE;
            uint col = i % TILE_K_MLA_LARGE;
            latent_tile[row][col] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute latent_tile = hidden @ W_a[:, latent_block:latent_block+TILE_K]
        // Process K_hidden in chunks
        for (uint k_hidden_block = 0; k_hidden_block < K_hidden; k_hidden_block += TILE_K_MLA_LARGE) {
            // Load hidden tile
            {
                const uint elems_per_thread = (TILE_M_MLA * TILE_K_MLA_LARGE) / THREADS_PER_TG_MLA;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_K_MLA_LARGE;
                    uint col = flat_idx % TILE_K_MLA_LARGE;
                    uint global_row = tg_row + row;
                    uint global_col = k_hidden_block + col;

                    half val = (global_row < M && global_col < K_hidden)
                               ? hidden[global_row * K_hidden + global_col]
                               : half(0.0h);
                    hidden_tile[row][col] = val;
                }
            }

            // Load W_a tile and dequantize
            // W_a is [K_hidden, K_latent], we need [k_hidden_block:+TILE_K, latent_block:+TILE_K]
            {
                const uint packed_per_thread = (TILE_K_MLA_LARGE * TILE_K_MLA_LARGE) / (THREADS_PER_TG_MLA * FP4_PER_UINT);
                for (uint i = 0; i < packed_per_thread; ++i) {
                    uint flat_packed_idx = thread_idx * packed_per_thread + i;
                    uint n_idx = flat_packed_idx / (TILE_K_MLA_LARGE / FP4_PER_UINT);
                    uint k_group_in_tile = flat_packed_idx % (TILE_K_MLA_LARGE / FP4_PER_UINT);

                    uint global_n = latent_block + n_idx;  // Column in K_latent
                    uint global_k_base = k_hidden_block + k_group_in_tile * FP4_PER_UINT;

                    uint scale_k = global_k_base / group_size_a;
                    half s = half(1.0h);
                    if (global_n < K_latent && scale_k < scale_tiles_a) {
                        s = scales_a[scale_k * K_latent + global_n];
                    }

                    uint32_t packed = 0;
                    uint b_row = global_k_base / FP4_PER_UINT;
                    if (global_n < K_latent && b_row < k_packs_a && global_k_base < K_hidden) {
                        packed = W_a_packed[b_row * K_latent + global_n];
                    }

                    uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
                    half vals[8];
                    dequant_fp4x8(packed, s, vals);

                    // Store into B_tile (reusing for W_a here)
                    for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_MLA_LARGE; ++v) {
                        if (n_idx < TILE_K_MLA_LARGE) {
                            uint global_k = global_k_base + v;
                            B_tile[tile_k_base + v][n_idx] = (global_k < K_hidden) ? vals[v] : half(0.0h);
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate: latent_tile += hidden_tile @ W_a_tile
            simdgroup_matrix<half, 8, 8> latent_acc[SG_M_TILES_MLA][SG_N_TILES_MLA];
            for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) {
                    // Load current latent accumulator
                    simdgroup_load(latent_acc[mi][ni],
                                   &latent_tile[sg_row_offset + mi * 8][sg_col_offset + ni * 8],
                                   TILE_K_MLA_LARGE);
                }
            }

            for (uint kt = 0; kt < K_TILES_MLA_LARGE; ++kt) {
                for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) {
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag,
                                   &hidden_tile[sg_row_offset + mi * 8][kt * 8],
                                   TILE_K_MLA_LARGE);

                    for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) {
                        simdgroup_matrix<half, 8, 8> b_frag;
                        simdgroup_load(b_frag,
                                       &B_tile[kt * 8][sg_col_offset + ni * 8],
                                       TILE_K_MLA_LARGE);

                        simdgroup_multiply_accumulate(latent_acc[mi][ni],
                                                      a_frag, b_frag, latent_acc[mi][ni]);
                    }
                }
            }

            // Store back to latent_tile
            for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) {
                    simdgroup_store(latent_acc[mi][ni],
                                    &latent_tile[sg_row_offset + mi * 8][sg_col_offset + ni * 8],
                                    TILE_K_MLA_LARGE);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Now latent_tile contains latent[TILE_M, TILE_K_latent_portion]
        // Multiply by W_b to contribute to final output

        // Load W_b tile for this latent_block
        {
            const uint packed_per_thread = (TILE_K_MLA_LARGE * TILE_N_MLA) / (THREADS_PER_TG_MLA * FP4_PER_UINT);
            for (uint i = 0; i < packed_per_thread; ++i) {
                uint flat_packed_idx = thread_idx * packed_per_thread + i;
                uint n_idx = flat_packed_idx / (TILE_K_MLA_LARGE / FP4_PER_UINT);
                uint k_group_in_tile = flat_packed_idx % (TILE_K_MLA_LARGE / FP4_PER_UINT);

                uint global_n = tg_col + n_idx;
                uint global_k_base = latent_block + k_group_in_tile * FP4_PER_UINT;

                uint scale_k = global_k_base / group_size_b;
                half s = half(1.0h);
                if (global_n < N_out && scale_k < scale_tiles_b) {
                    s = scales_b[scale_k * N_out + global_n];
                }

                uint32_t packed = 0;
                uint b_row = global_k_base / FP4_PER_UINT;
                if (global_n < N_out && b_row < k_packs_b && global_k_base < K_latent) {
                    packed = W_b_packed[b_row * N_out + global_n];
                }

                uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
                half vals[8];
                dequant_fp4x8(packed, s, vals);

                for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_MLA_LARGE; ++v) {
                    if (n_idx < TILE_N_MLA) {
                        uint global_k = global_k_base + v;
                        B_tile[tile_k_base + v][n_idx] = (global_k < K_latent) ? vals[v] : half(0.0h);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: out += latent_tile @ W_b_tile
        for (uint kt = 0; kt < K_TILES_MLA_LARGE; ++kt) {
            for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &latent_tile[sg_row_offset + mi * 8][kt * 8],
                               TILE_K_MLA_LARGE);

                for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag,
                                   &B_tile[kt * 8][sg_col_offset + ni * 8],
                                   TILE_N_MLA);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store final output
    for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N_out) {
                simdgroup_store(acc[mi][ni], out + out_row * N_out + out_col, N_out);
            } else if (out_row < M && out_col < N_out) {
                threadgroup half out_staging[8][8];
                simdgroup_store(acc[mi][ni], &out_staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < M && out_col + c < N_out) {
                        out[(out_row + r) * N_out + out_col + c] = out_staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: MLA Projection with RoPE Fusion
//
// Applies RoPE to the first rope_dim elements of the output.
// This is for the decoupled rope head dimension in GLM-4 style models.
//
// Output layout: [M, rope_dim + value_dim]
// RoPE applied to output[:, :rope_dim] only.
// ---------------------------------------------------------------------------

[[kernel]] void mla_proj_with_rope_fp4(
    device const half* A            [[buffer(0)]],   // [M, K]
    device const uint* B_packed     [[buffer(1)]],   // [K/8, N] FP4
    device const half* scales       [[buffer(2)]],   // [K/group_size, N]
    device const half* cos_cache    [[buffer(3)]],   // [max_seq, rope_dim/2]
    device const half* sin_cache    [[buffer(4)]],   // [max_seq, rope_dim/2]
    device const uint* positions    [[buffer(5)]],   // [M] position indices
    device half* out                [[buffer(6)]],   // [M, N]
    constant uint& M                [[buffer(7)]],
    constant uint& N                [[buffer(8)]],
    constant uint& K                [[buffer(9)]],
    constant uint& group_size       [[buffer(10)]],
    constant uint& rope_dim         [[buffer(11)]],  // Elements to apply RoPE to
    constant uint& max_seq          [[buffer(12)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint thread_idx                 [[thread_index_in_threadgroup]]
) {
    // Standard MLA projection...
    threadgroup half A_tile[TILE_M_MLA][TILE_K_MLA_LARGE];
    threadgroup half B_tile[TILE_K_MLA_LARGE][TILE_N_MLA];
    threadgroup half out_tile[TILE_M_MLA][TILE_N_MLA];  // For RoPE application

    uint tg_row = tgid.y * TILE_M_MLA;
    uint tg_col = tgid.x * TILE_N_MLA;
    uint sg_row_offset = 0;  // All simdgroups cover all rows
    uint sg_col_offset = simd_id * (SG_N_TILES_MLA * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES_MLA][SG_N_TILES_MLA];
    for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi)
        for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint scale_tiles = (K + group_size - 1) / group_size;

    // Main K-reduction loop (same as mla_proj_fp4)
    for (uint k_block = 0; k_block < K; k_block += TILE_K_MLA_LARGE) {
        // Load A tile
        {
            const uint elems_per_thread = (TILE_M_MLA * TILE_K_MLA_LARGE) / THREADS_PER_TG_MLA;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / TILE_K_MLA_LARGE;
                uint col = flat_idx % TILE_K_MLA_LARGE;
                uint global_row = tg_row + row;
                uint global_col = k_block + col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tile[row][col] = val;
            }
        }

        // Load and dequant B tile
        {
            const uint packed_per_thread = (TILE_K_MLA_LARGE * TILE_N_MLA) / (THREADS_PER_TG_MLA * FP4_PER_UINT);
            for (uint i = 0; i < packed_per_thread; ++i) {
                uint flat_packed_idx = thread_idx * packed_per_thread + i;
                uint n_idx = flat_packed_idx / (TILE_K_MLA_LARGE / FP4_PER_UINT);
                uint k_group_in_tile = flat_packed_idx % (TILE_K_MLA_LARGE / FP4_PER_UINT);

                uint global_n = tg_col + n_idx;
                uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

                uint scale_k = global_k_base / group_size;
                half s = half(1.0h);
                if (global_n < N && scale_k < scale_tiles) {
                    s = scales[scale_k * N + global_n];
                }

                uint32_t packed = 0;
                uint b_row = global_k_base / FP4_PER_UINT;
                if (global_n < N && b_row < k_packs && global_k_base < K) {
                    packed = B_packed[b_row * N + global_n];
                }

                uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
                half vals[8];
                dequant_fp4x8(packed, s, vals);
                for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_MLA_LARGE; ++v) {
                    if (n_idx < TILE_N_MLA) {
                        uint global_k = global_k_base + v;
                        B_tile[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : half(0.0h);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute
        for (uint kt = 0; kt < K_TILES_MLA_LARGE; ++kt) {
            for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag, &A_tile[sg_row_offset + mi * 8][kt * 8], TILE_K_MLA_LARGE);

                for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_tile[kt * 8][sg_col_offset + ni * 8], TILE_N_MLA);
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results to threadgroup memory for RoPE application
    for (uint mi = 0; mi < SG_M_TILES_MLA; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES_MLA; ++ni) {
            uint tile_row = sg_row_offset + mi * 8;
            uint tile_col = sg_col_offset + ni * 8;
            simdgroup_store(acc[mi][ni], &out_tile[tile_row][tile_col], TILE_N_MLA);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply RoPE to columns [0, rope_dim) if this tile overlaps with rope region
    bool applies_rope = (tg_col < rope_dim);
    uint rope_end_in_tile = min(rope_dim - tg_col, TILE_N_MLA);

    if (applies_rope && rope_end_in_tile > 0) {
        // RoPE is applied per-row based on position
        // Each thread handles some rows
        for (uint row = thread_idx; row < TILE_M_MLA; row += THREADS_PER_TG_MLA) {
            uint global_row = tg_row + row;
            if (global_row >= M) continue;

            uint pos = positions[global_row];

            // For columns in [tg_col, tg_col + rope_end_in_tile], apply rotation
            // RoPE pairs: (col, col + rope_dim/2) for col < rope_dim/2
            // But our tile may only have partial coverage
            uint half_rope = rope_dim / 2;

            // Check if this tile contains any rotation pairs we need to handle
            for (uint c = 0; c < rope_end_in_tile; c += 2) {
                uint global_col = tg_col + c;
                if (global_col >= half_rope) {
                    // This column is in the second half of RoPE - skip
                    continue;
                }

                // Pair is (global_col, global_col + half_rope)
                uint pair_col2 = global_col + half_rope;

                // Check if pair_col2 is in this tile
                if (pair_col2 >= tg_col && pair_col2 < tg_col + TILE_N_MLA) {
                    uint local_c1 = c;
                    uint local_c2 = pair_col2 - tg_col;

                    uint pair_idx = global_col;
                    half cos_val = cos_cache[pos * half_rope + pair_idx];
                    half sin_val = sin_cache[pos * half_rope + pair_idx];

                    apply_rope_pair(out_tile[row][local_c1], out_tile[row][local_c2], cos_val, sin_val);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write out to global memory
    for (uint i = thread_idx; i < TILE_M_MLA * TILE_N_MLA; i += THREADS_PER_TG_MLA) {
        uint row = i / TILE_N_MLA;
        uint col = i % TILE_N_MLA;
        uint global_row = tg_row + row;
        uint global_col = tg_col + col;

        if (global_row < M && global_col < N) {
            out[global_row * N + global_col] = out_tile[row][col];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: MLA Decode GEMV (Single Token Projection)
//
// Optimized for decode phase where we're projecting a single token.
// Uses GEMV (matrix-vector multiply) instead of GEMM for better efficiency.
// ---------------------------------------------------------------------------

[[kernel]] void mla_decode_proj_fp4(
    device const half* x            [[buffer(0)]],   // [1, K] or [K] - single token
    device const uint* W_packed     [[buffer(1)]],   // [K/8, N] FP4
    device const half* scales       [[buffer(2)]],   // [K/group_size, N]
    device half* out                [[buffer(3)]],   // [N]
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint3 grid_size                 [[threads_per_grid]]
) {
    // Each threadgroup handles multiple output columns
    // Each thread accumulates for one output column
    uint tgid_x = tgid.x;
    uint num_tgs = (grid_size.x + THREADS_PER_TG_DECODE - 1) / THREADS_PER_TG_DECODE;
    uint col_stride = num_tgs * THREADS_PER_TG_DECODE;

    for (uint n = tgid_x * THREADS_PER_TG_DECODE + tid; n < N; n += col_stride) {
        float sum = 0.0f;

        uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;

        for (uint k_pack = 0; k_pack < k_packs; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_k = k_base / group_size;

            half s = scales[scale_k * N + n];
            uint32_t packed = W_packed[k_pack * N + n];

            half vals[8];
            dequant_fp4x8(packed, s, vals);

            // Dot product with input
            for (uint v = 0; v < FP4_PER_UINT && k_base + v < K; ++v) {
                sum += (float)x[k_base + v] * (float)vals[v];
            }
        }

        out[n] = (half)sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: Batched Decode (Multiple tokens, still small batch)
//
// For batch_size <= 8, use register tiling per row instead of full GEMM.
// ---------------------------------------------------------------------------

[[kernel]] void mla_decode_batched_fp4(
    device const half* X            [[buffer(0)]],   // [batch, K]
    device const uint* W_packed     [[buffer(1)]],   // [K/8, N] FP4
    device const half* scales       [[buffer(2)]],   // [K/group_size, N]
    device half* out                [[buffer(3)]],   // [batch, N]
    constant uint& batch            [[buffer(4)]],
    constant uint& K                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]]
) {
    // Grid: (N / threads_per_tg, 1, 1)
    // Each threadgroup handles a range of output columns for all batch rows

    uint n_start = tgid.x * THREADS_PER_TG_DECODE;
    uint n = n_start + tid;

    if (n >= N) return;

    // Accumulate for all batch rows
    float sums[8];  // Max batch size for this kernel
    for (uint b = 0; b < batch && b < 8; ++b) {
        sums[b] = 0.0f;
    }

    uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;

    for (uint k_pack = 0; k_pack < k_packs; ++k_pack) {
        uint k_base = k_pack * FP4_PER_UINT;
        uint scale_k = k_base / group_size;

        half s = scales[scale_k * N + n];
        uint32_t packed = W_packed[k_pack * N + n];

        half vals[8];
        dequant_fp4x8(packed, s, vals);

        // Accumulate for each batch row
        for (uint b = 0; b < batch && b < 8; ++b) {
            for (uint v = 0; v < FP4_PER_UINT && k_base + v < K; ++v) {
                sums[b] += (float)X[b * K + k_base + v] * (float)vals[v];
            }
        }
    }

    // Write output
    for (uint b = 0; b < batch && b < 8; ++b) {
        out[b * N + n] = (half)sums[b];
    }
}
