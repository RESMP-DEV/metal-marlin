// batched_gemm.metal - Strided batched GEMM for FP4 weights
//
// Computes C[b] = A[b] @ dequant(B[b]) for each batch element b.
// Uses a 3D grid: (n_tiles, m_tiles, batch_count).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions (match marlin_gemm.metal)
// ---------------------------------------------------------------------------

constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 32;
constant constexpr uint TILE_K = 48;

constant constexpr uint K_TILES = TILE_K / 8;  // 4
constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;
constant constexpr uint SG_M_TILES = 8;
constant constexpr uint SG_N_TILES = 4;
constant constexpr uint FP4_PER_UINT = 8;
constant constexpr uint NUM_BUFFERS = 4;

// ---------------------------------------------------------------------------
// FP4 dequant helpers
// NOTE: Uses float intermediates to work around Metal compiler bug where
// half parameters in inline functions have fractional parts rounded.
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

// ---------------------------------------------------------------------------
// Cooperative tile loaders
// ---------------------------------------------------------------------------

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

inline void load_B_tile_transposed(
    device const half* B,
    threadgroup half (&B_buf)[TILE_K][TILE_N],
    uint N, uint K,
    uint tg_col, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (TILE_K * TILE_N) / THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint k_row = flat_idx / TILE_N;
        uint n_col = flat_idx % TILE_N;

        uint global_k = k_block + k_row;
        uint global_n = tg_col + n_col;

        half val = 0.0h;
        if (global_k < K && global_n < N) {
            val = B[global_n * K + global_k];
        }
        B_buf[k_row][n_col] = val;
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute and store
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

// Staging buffer dimensions for store_results
constant constexpr uint SG_TILE_ROWS = SG_M_TILES * 8;  // 64
constant constexpr uint SG_TILE_COLS = SG_N_TILES * 8;  // 16

inline void store_results(
    thread simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    device half* C,
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane,
    uint simd_id,
    threadgroup half (&sg_staging)[SIMDGROUPS_PER_TG][SG_TILE_ROWS][SG_TILE_COLS],
    threadgroup half (&edge_staging)[8][8]
) {
    constexpr uint sg_tile_rows = SG_M_TILES * 8;
    constexpr uint sg_tile_cols = SG_N_TILES * 8;
    constexpr uint segments_per_row = sg_tile_cols / 8;
    constexpr uint segments_per_simdgroup = sg_tile_rows * segments_per_row;

    uint base_row = tg_row + sg_row_offset;
    uint base_col = tg_col + sg_col_offset;

    if (base_row + sg_tile_rows <= M && base_col + sg_tile_cols <= N) {
        for (uint mi = 0; mi < SG_M_TILES; ++mi) {
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                simdgroup_store(acc[mi][ni],
                                &sg_staging[simd_id][mi * 8][ni * 8],
                                sg_tile_cols);
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint iter = 0; iter < segments_per_simdgroup / 32; ++iter) {
            uint segment = simd_lane + iter * 32;
            uint row = segment / segments_per_row;
            uint col_segment = segment % segments_per_row;
            uint col = col_segment * 8;

            device half* dst = C + (base_row + row) * N + base_col + col;
            threadgroup half* src = &sg_staging[simd_id][row][col];

            for (uint i = 0; i < 8; ++i) {
                dst[i] = src[i];
            }
        }
        return;
    }

    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = base_row + mi * 8;
            uint out_col = base_col + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni],
                                C + out_row * N + out_col,
                                N);
            } else {
                simdgroup_store(acc[mi][ni], &edge_staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = edge_staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Strided batched GEMM (FP4 weights, 3D grid dispatch)
// ---------------------------------------------------------------------------

kernel void marlin_gemm_batched_fp4(
    device const half* A          [[buffer(0)]],  // [batch, M, K]
    device const uint* B          [[buffer(1)]],  // [batch, K/8, N] packed FP4
    device const half* scales     [[buffer(2)]],  // [K/group_size, N]
    device half* C                [[buffer(3)]],  // [batch, M, N]
    constant uint& M              [[buffer(4)]],
    constant uint& N              [[buffer(5)]],
    constant uint& K              [[buffer(6)]],
    constant uint& group_size     [[buffer(7)]],
    constant uint& batch_count    [[buffer(8)]],
    constant uint& A_batch_stride [[buffer(9)]],
    constant uint& B_batch_stride [[buffer(10)]],
    constant uint& C_batch_stride [[buffer(11)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = tgid.z;
    if (batch_idx >= batch_count) {
        return;
    }

    device const half* A_batch = A + batch_idx * A_batch_stride;
    device const uint* B_batch = B + batch_idx * B_batch_stride;
    device half* C_batch = C + batch_idx * C_batch_stride;

    threadgroup half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][TILE_K][TILE_N];
    threadgroup half sg_staging[SIMDGROUPS_PER_TG][SG_TILE_ROWS][SG_TILE_COLS];
    threadgroup half edge_staging[8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    uint buf_compute = 0;

    load_A_tile(A_batch, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_dequant(B_batch, scales, B_tiles[0], K, N, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * TILE_K;
        uint next_k = k_offset + TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_A_tile(A_batch, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant(B_batch, scales, B_tiles[buf_load], K, N, tg_col, next_k, group_size, thread_idx);
        }

        compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                           acc, sg_row_offset, sg_col_offset);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    store_results(acc, C_batch, M, N, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id,
                  sg_staging, edge_staging);
}

// ---------------------------------------------------------------------------
// Kernel: Grouped GEMM for GQA (Q @ K^T with shared KV heads)
// ---------------------------------------------------------------------------

kernel void marlin_gemm_grouped_attention(
    device const half* Q         [[buffer(0)]],  // [batch, num_q_heads, seq, head_dim]
    device const half* K         [[buffer(1)]],  // [batch, num_kv_heads, seq, head_dim]
    device half* scores          [[buffer(2)]],  // [batch, num_q_heads, seq, seq]
    constant uint& batch         [[buffer(3)]],
    constant uint& num_q_heads   [[buffer(4)]],
    constant uint& num_kv_heads  [[buffer(5)]],
    constant uint& seq_len       [[buffer(6)]],
    constant uint& head_dim      [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint total_heads = batch * num_q_heads;
    if (tgid.z >= total_heads || num_kv_heads == 0) {
        return;
    }

    uint batch_idx = tgid.z / num_q_heads;
    uint q_head = tgid.z % num_q_heads;
    uint group_size = num_q_heads / num_kv_heads;
    uint kv_head = group_size > 0 ? (q_head / group_size) : 0;
    if (kv_head >= num_kv_heads) {
        return;
    }

    device const half* Q_head = Q + (batch_idx * num_q_heads + q_head) * seq_len * head_dim;
    device const half* K_head = K + (batch_idx * num_kv_heads + kv_head) * seq_len * head_dim;
    device half* scores_head = scores + (batch_idx * num_q_heads + q_head) * seq_len * seq_len;

    threadgroup half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][TILE_K][TILE_N];
    threadgroup half sg_staging[SIMDGROUPS_PER_TG][SG_TILE_ROWS][SG_TILE_COLS];
    threadgroup half edge_staging[8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (head_dim + TILE_K - 1) / TILE_K;
    uint buf_compute = 0;

    load_A_tile(Q_head, A_tiles[0], seq_len, head_dim, tg_row, 0, thread_idx);
    load_B_tile_transposed(K_head, B_tiles[0], seq_len, head_dim, tg_col, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * TILE_K;
        uint next_k = k_offset + TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < head_dim) {
            load_A_tile(Q_head, A_tiles[buf_load], seq_len, head_dim, tg_row, next_k, thread_idx);
            load_B_tile_transposed(K_head, B_tiles[buf_load], seq_len, head_dim, tg_col, next_k, thread_idx);
        }

        compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                           acc, sg_row_offset, sg_col_offset);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    store_results(acc, scores_head, seq_len, seq_len, tg_row, tg_col,
                  sg_row_offset, sg_col_offset, simd_lane, simd_id,
                  sg_staging, edge_staging);
}
