// debug.metal - Debug kernel for inspecting intermediate GEMM values
//
// Captures:
//  - Dequantized B tile (TILE_K x TILE_N) for a selected K block
//  - Accumulator state (per simdgroup, per 8x8 tile) after that K block

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// Tile dimensions (match marlin_gemm.metal)
constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K = 32;
constant constexpr uint K_TILES = TILE_K / 8;  // 4

constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;
constant constexpr uint SG_M_TILES = 2;
constant constexpr uint SG_N_TILES = 4;
constant constexpr uint FP4_PER_UINT = 8;

// ---------------------------------------------------------------------------
// FP4 dequant + tile loaders (mirrors marlin_gemm.metal)
// NOTE: Uses float intermediates per Metal compiler bug workaround.
// ---------------------------------------------------------------------------

inline half dequant_fp4(uint nibble, half scale) {
    float fscale = (float)scale;
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    float magnitude;
    if (exp_bits == 0) {
        magnitude = (float)man_bit * 0.25f;
    } else {
        float power = (float)(1u << (exp_bits - 1));
        float mantissa = 1.0f + (float)man_bit * 0.5f;
        magnitude = power * mantissa;
    }

    float result = sign_bit ? -magnitude : magnitude;
    return (half)(result * fscale);
}

inline void unpack_fp4x8(uint packed, half scale, thread half* out) {
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = dequant_fp4(nibble, scale);
    }
}

inline void load_A_tile(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (TILE_M * TILE_K) / THREADS_PER_TG;
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
    const uint scale_tiles = (K + group_size - 1) / group_size;
    const uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;
    const uint packed_per_thread = (TILE_K * TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
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
                B_buf[tile_k_base + v][n_idx] = vals[v];
            }
        }
    }
}

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

inline void store_results(
    thread simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    device half* C,
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane
) {
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni],
                                C + out_row * N + out_col,
                                N);
            } else {
                threadgroup half staging[8][8];
                simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Debug kernel
//
// debug_dequant: expects TILE_K * TILE_N halfs
// debug_accum: expects SIMDGROUPS_PER_TG * SG_M_TILES * SG_N_TILES * 64 floats
// debug_k: K-block offset (must be a multiple of TILE_K to capture)
// ---------------------------------------------------------------------------

kernel void marlin_gemm_fp4_debug(
    device const half* A         [[buffer(0)]],
    device const uint* B_packed  [[buffer(1)]],
    device const half* scales    [[buffer(2)]],
    device half* C               [[buffer(3)]],
    device half* debug_dequant   [[buffer(4)]],
    device float* debug_accum    [[buffer(5)]],
    constant uint& M             [[buffer(6)]],
    constant uint& N             [[buffer(7)]],
    constant uint& K             [[buffer(8)]],
    constant uint& group_size    [[buffer(9)]],
    constant uint& debug_k       [[buffer(10)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tile[TILE_M][TILE_K];
    threadgroup half B_tile[TILE_K][TILE_N];
    threadgroup half acc_staging[SIMDGROUPS_PER_TG][SG_M_TILES][SG_N_TILES][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);
    const uint thread_idx = simd_id * 32 + simd_lane;
    const bool capture_tg = (tgid.x == 0 && tgid.y == 0);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    for (uint k_block = 0; k_block < K; k_block += TILE_K) {
        load_A_tile(A, A_tile, M, K, tg_row, k_block, thread_idx);
        load_B_tile_dequant(B_packed, scales, B_tile, K, N, tg_col, k_block,
                            group_size, thread_idx);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (capture_tg && k_block == debug_k && thread_idx == 0) {
            for (uint r = 0; r < TILE_K; ++r) {
                for (uint c = 0; c < TILE_N; ++c) {
                    debug_dequant[r * TILE_N + c] = B_tile[r][c];
                }
            }
        }

        compute_from_tiles(A_tile, B_tile, acc, sg_row_offset, sg_col_offset);

        if (capture_tg && k_block == debug_k) {
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_store(acc[mi][ni],
                                    &acc_staging[simd_id][mi][ni][0][0],
                                    8);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint tiles_per_sg = SG_M_TILES * SG_N_TILES;
            const uint base = simd_id * tiles_per_sg * 64;
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    const uint tile_index = mi * SG_N_TILES + ni;
                    const uint tile_base = base + tile_index * 64;
                    for (uint elem = simd_lane; elem < 64; elem += 32) {
                        uint r = elem / 8;
                        uint c = elem % 8;
                        half h = acc_staging[simd_id][mi][ni][r][c];
                        debug_accum[tile_base + r * 8 + c] = float(h);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    store_results(acc, C, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane);
}
