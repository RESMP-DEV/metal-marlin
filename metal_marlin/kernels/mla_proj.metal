// mla_proj.metal - Optimized MLA latent projection kernels
//
// Fused kv_a + kv_b projection avoids materializing the intermediate latent
// tensor, and applies RoPE in latent space when requested.
//
// Assumptions:
// - Activations are FP16 (half)
// - Weights are FP4-quantized with per-group scales
// - kv_lora_rank (K_latent) is small (512-1536)
// - RoPE is applied to the first rope_dim elements of the latent vector
//   and requires rope_dim <= TILE_K_LATENT (default 64)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile configuration tuned for small latent dimensions
// ---------------------------------------------------------------------------

constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K_HIDDEN = 32;
constant constexpr uint TILE_K_LATENT = 64;

constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;

constant constexpr uint SG_M_TILES = 2;
constant constexpr uint SG_N_TILES = 4;

constant constexpr uint FP4_PER_UINT = 8;
constant constexpr uint K_TILES_HIDDEN = TILE_K_HIDDEN / 8;  // 4
constant constexpr uint K_TILES_LATENT = TILE_K_LATENT / 8;  // 8

// ---------------------------------------------------------------------------
// FP4 E2M1 dequantization (branchless)
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
// RoPE helpers
// ---------------------------------------------------------------------------

inline void apply_rope_pair(thread half& x, thread half& y, half cos_val, half sin_val) {
    half x_new = x * cos_val - y * sin_val;
    half y_new = x * sin_val + y * cos_val;
    x = x_new;
    y = y_new;
}

// ---------------------------------------------------------------------------
// Kernel: Fused kv_a + kv_b projection (FP4 weights, FP16 activations)
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
    threadgroup half hidden_tile[TILE_M][TILE_K_HIDDEN];
    threadgroup half latent_tile[TILE_M][TILE_K_LATENT];
    threadgroup half W_a_tile[TILE_K_HIDDEN][TILE_K_LATENT];
    threadgroup half W_b_tile[TILE_K_LATENT][TILE_N];

    uint tg_row = tgid.y * TILE_M;
    uint tg_col = tgid.x * TILE_N;
    uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint k_packs_a = (K_hidden + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint k_packs_b = (K_latent + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint scale_tiles_a = (K_hidden + group_size_a - 1) / group_size_a;
    uint scale_tiles_b = (K_latent + group_size_b - 1) / group_size_b;

    for (uint latent_block = 0; latent_block < K_latent; latent_block += TILE_K_LATENT) {
        // Zero the latent tile accumulator
        for (uint i = thread_idx; i < TILE_M * TILE_K_LATENT; i += THREADS_PER_TG) {
            uint row = i / TILE_K_LATENT;
            uint col = i % TILE_K_LATENT;
            latent_tile[row][col] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // hidden @ W_a (accumulate into latent_tile)
        for (uint k_hidden_block = 0; k_hidden_block < K_hidden; k_hidden_block += TILE_K_HIDDEN) {
            // Load hidden tile
            {
                const uint elems_per_thread = (TILE_M * TILE_K_HIDDEN) / THREADS_PER_TG;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_K_HIDDEN;
                    uint col = flat_idx % TILE_K_HIDDEN;
                    uint global_row = tg_row + row;
                    uint global_col = k_hidden_block + col;

                    half val = (global_row < M && global_col < K_hidden)
                               ? hidden[global_row * K_hidden + global_col]
                               : half(0.0h);
                    hidden_tile[row][col] = val;
                }
            }

            // Load W_a tile and dequantize
            {
                const uint packed_per_thread = (TILE_K_HIDDEN * TILE_K_LATENT) / (THREADS_PER_TG * FP4_PER_UINT);
                for (uint i = 0; i < packed_per_thread; ++i) {
                    uint flat_packed_idx = thread_idx * packed_per_thread + i;
                    uint n_idx = flat_packed_idx / (TILE_K_HIDDEN / FP4_PER_UINT);
                    uint k_group_in_tile = flat_packed_idx % (TILE_K_HIDDEN / FP4_PER_UINT);

                    uint global_n = latent_block + n_idx;
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

                    for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_HIDDEN; ++v) {
                        if (n_idx < TILE_K_LATENT) {
                            uint global_k = global_k_base + v;
                            W_a_tile[tile_k_base + v][n_idx] = (global_k < K_hidden && global_n < K_latent)
                                                                 ? vals[v]
                                                                 : half(0.0h);
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate: latent_tile += hidden_tile @ W_a_tile
            simdgroup_matrix<half, 8, 8> latent_acc[SG_M_TILES][SG_N_TILES];
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_load(latent_acc[mi][ni],
                                   &latent_tile[sg_row_offset + mi * 8][sg_col_offset + ni * 8],
                                   TILE_K_LATENT);
                }
            }

            for (uint kt = 0; kt < K_TILES_HIDDEN; ++kt) {
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag,
                                   &hidden_tile[sg_row_offset + mi * 8][kt * 8],
                                   TILE_K_HIDDEN);

                    for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                        simdgroup_matrix<half, 8, 8> b_frag;
                        simdgroup_load(b_frag,
                                       &W_a_tile[kt * 8][sg_col_offset + ni * 8],
                                       TILE_K_LATENT);

                        simdgroup_multiply_accumulate(latent_acc[mi][ni], a_frag, b_frag, latent_acc[mi][ni]);
                    }
                }
            }

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_store(latent_acc[mi][ni],
                                    &latent_tile[sg_row_offset + mi * 8][sg_col_offset + ni * 8],
                                    TILE_K_LATENT);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Load W_b tile for this latent_block
        {
            const uint packed_per_thread = (TILE_K_LATENT * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
            for (uint i = 0; i < packed_per_thread; ++i) {
                uint flat_packed_idx = thread_idx * packed_per_thread + i;
                uint n_idx = flat_packed_idx / (TILE_K_LATENT / FP4_PER_UINT);
                uint k_group_in_tile = flat_packed_idx % (TILE_K_LATENT / FP4_PER_UINT);

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

                for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_LATENT; ++v) {
                    if (n_idx < TILE_N) {
                        uint global_k = global_k_base + v;
                        W_b_tile[tile_k_base + v][n_idx] = (global_k < K_latent) ? vals[v] : half(0.0h);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: out += latent_tile @ W_b_tile
        for (uint kt = 0; kt < K_TILES_LATENT; ++kt) {
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &latent_tile[sg_row_offset + mi * 8][kt * 8],
                               TILE_K_LATENT);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag,
                                   &W_b_tile[kt * 8][sg_col_offset + ni * 8],
                                   TILE_N);

                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store output tile
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;
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

// ---------------------------------------------------------------------------
// Kernel: Fused kv_a + kv_b with RoPE in latent space
// ---------------------------------------------------------------------------

[[kernel]] void mla_fused_kv_proj_rope_fp4(
    device const half* hidden           [[buffer(0)]],   // [M, K_hidden]
    device const uint* W_a_packed       [[buffer(1)]],   // [K_hidden/8, K_latent] FP4
    device const half* scales_a         [[buffer(2)]],   // [K_hidden/group_size_a, K_latent]
    device const uint* W_b_packed       [[buffer(3)]],   // [K_latent/8, N_out] FP4
    device const half* scales_b         [[buffer(4)]],   // [K_latent/group_size_b, N_out]
    device const half* cos_cache        [[buffer(5)]],   // [max_seq, rope_dim/2]
    device const half* sin_cache        [[buffer(6)]],   // [max_seq, rope_dim/2]
    device const uint* positions        [[buffer(7)]],   // [M]
    device half* out                    [[buffer(8)]],   // [M, N_out]
    constant uint& M                    [[buffer(9)]],
    constant uint& K_hidden             [[buffer(10)]],
    constant uint& K_latent             [[buffer(11)]],
    constant uint& N_out                [[buffer(12)]],
    constant uint& group_size_a         [[buffer(13)]],
    constant uint& group_size_b         [[buffer(14)]],
    constant uint& rope_dim             [[buffer(15)]],
    constant uint& max_seq              [[buffer(16)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    threadgroup half hidden_tile[TILE_M][TILE_K_HIDDEN];
    threadgroup half latent_tile[TILE_M][TILE_K_LATENT];
    threadgroup half W_a_tile[TILE_K_HIDDEN][TILE_K_LATENT];
    threadgroup half W_b_tile[TILE_K_LATENT][TILE_N];

    uint tg_row = tgid.y * TILE_M;
    uint tg_col = tgid.x * TILE_N;
    uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint k_packs_a = (K_hidden + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint k_packs_b = (K_latent + FP4_PER_UINT - 1) / FP4_PER_UINT;
    uint scale_tiles_a = (K_hidden + group_size_a - 1) / group_size_a;
    uint scale_tiles_b = (K_latent + group_size_b - 1) / group_size_b;

    for (uint latent_block = 0; latent_block < K_latent; latent_block += TILE_K_LATENT) {
        // Zero the latent tile accumulator
        for (uint i = thread_idx; i < TILE_M * TILE_K_LATENT; i += THREADS_PER_TG) {
            uint row = i / TILE_K_LATENT;
            uint col = i % TILE_K_LATENT;
            latent_tile[row][col] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_hidden_block = 0; k_hidden_block < K_hidden; k_hidden_block += TILE_K_HIDDEN) {
            // Load hidden tile
            {
                const uint elems_per_thread = (TILE_M * TILE_K_HIDDEN) / THREADS_PER_TG;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint row = flat_idx / TILE_K_HIDDEN;
                    uint col = flat_idx % TILE_K_HIDDEN;
                    uint global_row = tg_row + row;
                    uint global_col = k_hidden_block + col;

                    half val = (global_row < M && global_col < K_hidden)
                               ? hidden[global_row * K_hidden + global_col]
                               : half(0.0h);
                    hidden_tile[row][col] = val;
                }
            }

            // Load W_a tile and dequantize
            {
                const uint packed_per_thread = (TILE_K_HIDDEN * TILE_K_LATENT) / (THREADS_PER_TG * FP4_PER_UINT);
                for (uint i = 0; i < packed_per_thread; ++i) {
                    uint flat_packed_idx = thread_idx * packed_per_thread + i;
                    uint n_idx = flat_packed_idx / (TILE_K_HIDDEN / FP4_PER_UINT);
                    uint k_group_in_tile = flat_packed_idx % (TILE_K_HIDDEN / FP4_PER_UINT);

                    uint global_n = latent_block + n_idx;
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

                    for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_HIDDEN; ++v) {
                        if (n_idx < TILE_K_LATENT) {
                            uint global_k = global_k_base + v;
                            W_a_tile[tile_k_base + v][n_idx] = (global_k < K_hidden && global_n < K_latent)
                                                                 ? vals[v]
                                                                 : half(0.0h);
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate: latent_tile += hidden_tile @ W_a_tile
            simdgroup_matrix<half, 8, 8> latent_acc[SG_M_TILES][SG_N_TILES];
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_load(latent_acc[mi][ni],
                                   &latent_tile[sg_row_offset + mi * 8][sg_col_offset + ni * 8],
                                   TILE_K_LATENT);
                }
            }

            for (uint kt = 0; kt < K_TILES_HIDDEN; ++kt) {
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag,
                                   &hidden_tile[sg_row_offset + mi * 8][kt * 8],
                                   TILE_K_HIDDEN);

                    for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                        simdgroup_matrix<half, 8, 8> b_frag;
                        simdgroup_load(b_frag,
                                       &W_a_tile[kt * 8][sg_col_offset + ni * 8],
                                       TILE_K_LATENT);

                        simdgroup_multiply_accumulate(latent_acc[mi][ni], a_frag, b_frag, latent_acc[mi][ni]);
                    }
                }
            }

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_store(latent_acc[mi][ni],
                                    &latent_tile[sg_row_offset + mi * 8][sg_col_offset + ni * 8],
                                    TILE_K_LATENT);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Apply RoPE to the first rope_dim elements (only when it fits in this block)
        if (rope_dim > 0 && rope_dim <= TILE_K_LATENT && latent_block == 0) {
            uint pairs = rope_dim / 2;
            for (uint row = thread_idx; row < TILE_M; row += THREADS_PER_TG) {
                uint global_row = tg_row + row;
                if (global_row >= M) continue;
                uint pos = positions[global_row];
                if (pos >= max_seq) continue;

                for (uint p = 0; p < pairs; ++p) {
                    half cos_val = cos_cache[pos * pairs + p];
                    half sin_val = sin_cache[pos * pairs + p];
                    apply_rope_pair(latent_tile[row][p], latent_tile[row][p + pairs], cos_val, sin_val);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load W_b tile for this latent_block
        {
            const uint packed_per_thread = (TILE_K_LATENT * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
            for (uint i = 0; i < packed_per_thread; ++i) {
                uint flat_packed_idx = thread_idx * packed_per_thread + i;
                uint n_idx = flat_packed_idx / (TILE_K_LATENT / FP4_PER_UINT);
                uint k_group_in_tile = flat_packed_idx % (TILE_K_LATENT / FP4_PER_UINT);

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

                for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K_LATENT; ++v) {
                    if (n_idx < TILE_N) {
                        uint global_k = global_k_base + v;
                        W_b_tile[tile_k_base + v][n_idx] = (global_k < K_latent) ? vals[v] : half(0.0h);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: out += latent_tile @ W_b_tile
        for (uint kt = 0; kt < K_TILES_LATENT; ++kt) {
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &latent_tile[sg_row_offset + mi * 8][kt * 8],
                               TILE_K_LATENT);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag,
                                   &W_b_tile[kt * 8][sg_col_offset + ni * 8],
                                   TILE_N);

                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store output tile
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;
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
