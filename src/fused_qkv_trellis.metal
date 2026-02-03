// fused_qkv_trellis.metal - Fused Q/K/V Projection Kernel
// ============================================================================
//
// Fused kernel for computing multiple projections (Q, K, V) from a single input
// using trellis-quantized weights.
//
// Optimizations:
// - Loads input A only once (shared across Q/K/V)
// - Dequantizes and computes Q, K, V in parallel/sequence within the same kernel
// - Outputs to separate buffers
// - Supports disabling outputs by setting N=0
//
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// Common Helpers (Copied from gemm_trellis.metal)
// ============================================================================

constexpr sampler grid_sampler(coord::pixel, address::clamp_to_edge, filter::nearest);

// Tile Dimensions
constant constexpr uint TILE_M = 128;
constant constexpr uint TILE_N = 128;
constant constexpr uint TILE_K = 32;
constant constexpr uint K_TILES = TILE_K / 8;
constant constexpr uint SIMDGROUPS_PER_TG = 8;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;
constant constexpr uint SG_M_TILES = 4;
constant constexpr uint SG_N_TILES = 8;
constant constexpr uint TRELLIS_TILE_DIM = 16;
constant constexpr uint TRELLIS_TILE_SIZE = 256;

// Decode-specific constants
constant constexpr uint DECODE_TILE_M = 32;
constant constexpr uint DECODE_TILE_N = 128;
constant constexpr uint DECODE_K_TILES = TILE_K / 8;
constant constexpr uint DECODE_SG_M_TILES = 4;
constant constexpr uint DECODE_SG_N_TILES = 4;

// --- Unpack Helpers ---

inline uint unpack_2bit_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 2;
    uint bit_offset = (idx_in_tile & 3) << 1;
    return (packed[byte_idx] >> bit_offset) & 0x3;
}

inline uint unpack_3bit_index(device const uchar* packed, uint idx_in_tile) {
    uint bit_offset = idx_in_tile * 3;
    uint byte_idx = bit_offset >> 3;
    uint bit_in_byte = bit_offset & 7;
    uint packed_val = uint(packed[byte_idx]);
    if (bit_in_byte + 3 > 8) {
        packed_val |= uint(packed[byte_idx + 1]) << 8;
    }
    return (packed_val >> bit_in_byte) & 0x7;
}

inline uint unpack_4bit_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 1;
    uint shift = (idx_in_tile & 1) << 2;
    return (packed[byte_idx] >> shift) & 0xF;
}

inline uint unpack_trellis_index(device const uchar* packed, uint idx_in_tile, uint bits) {
    switch (bits) {
        case 2: return unpack_2bit_index(packed, idx_in_tile);
        case 3: return unpack_3bit_index(packed, idx_in_tile);
        case 4: return unpack_4bit_index(packed, idx_in_tile);
        default: return unpack_3bit_index(packed, idx_in_tile);
    }
}

inline uint packed_bytes_per_trellis_tile(uint bits) {
    return (TRELLIS_TILE_SIZE * bits + 7) / 8;
}

// --- Dequant Helpers ---

inline half dequant_trellis_element_fused(
    uint idx,
    float combined_scale,
    device const float* grid
) {
    return half(grid[idx] * combined_scale);
}

// ============================================================================
// Fused Decode Kernel (M <= 16)
// ============================================================================

kernel void fused_qkv_trellis_decode(
    // Input
    device const half* A               [[buffer(0)]],
    
    // Q Projection
    device const uchar* packed_q       [[buffer(1)]],
    device const float* scales_q       [[buffer(2)]],
    device const float* su_q           [[buffer(3)]],
    device const float* sv_q           [[buffer(4)]],
    
    // K Projection
    device const uchar* packed_k       [[buffer(5)]],
    device const float* scales_k       [[buffer(6)]],
    device const float* su_k           [[buffer(7)]],
    device const float* sv_k           [[buffer(8)]],
    
    // V Projection
    device const uchar* packed_v       [[buffer(9)]],
    device const float* scales_v       [[buffer(10)]],
    device const float* su_v           [[buffer(11)]],
    device const float* sv_v           [[buffer(12)]],
    
    // Shared Grid
    device const float* grid           [[buffer(13)]],
    
    // Outputs
    device half* OutQ                  [[buffer(14)]],
    device half* OutK                  [[buffer(15)]],
    device half* OutV                  [[buffer(16)]],
    
    // Dimensions
    constant uint& M                   [[buffer(17)]],
    constant uint& K                   [[buffer(18)]],
    constant uint& Nq                  [[buffer(19)]],
    constant uint& Nk                  [[buffer(20)]],
    constant uint& Nv                  [[buffer(21)]],
    
    // Config
    constant uint& bits                [[buffer(22)]],
    constant uint& n_levels            [[buffer(23)]],
    constant uint& group_size          [[buffer(24)]],
    
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // 32x128x32 tiles: A = 4KB
    threadgroup half A_tiles[2][DECODE_TILE_M][TILE_K];
    // Staging buffer reused for Q, K, V
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    
    const uint tg_row = tgid.y * DECODE_TILE_M;
    const uint tg_col = tgid.x * DECODE_TILE_N; // Global column offset
    
    if (tg_row >= M) return;
    
    // 4 simdgroups tile 32x128 as 1x4
    const uint sg_col_offset = simd_group_id * 32;
    
    // Accumulators for Q, K, V
    simdgroup_matrix<half, 8, 8> acc_q[DECODE_SG_M_TILES][DECODE_SG_N_TILES];
    simdgroup_matrix<half, 8, 8> acc_k[DECODE_SG_M_TILES][DECODE_SG_N_TILES];
    simdgroup_matrix<half, 8, 8> acc_v[DECODE_SG_M_TILES][DECODE_SG_N_TILES];
    
    // Initialize accumulators
    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            if (Nq > 0) acc_q[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
            if (Nk > 0) acc_k[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
            if (Nv > 0) acc_v[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }
    
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n_q = (Nq + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint tiles_n_k = (Nk + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint tiles_n_v = (Nv + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    constexpr uint A_ELEMS = (DECODE_TILE_M * TILE_K) / THREADS_PER_TG;
    
    // Load first A tile
    #pragma unroll
    for (uint i = 0; i < A_ELEMS; ++i) {
        uint flat_idx = thread_idx * A_ELEMS + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        half val = (global_row < M && col < K) ? A[global_row * K + col] : half(0.0h);
        A_tiles[0][row][col] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint buf_compute = 0;
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        
        // Load next A tile
        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < A_ELEMS; ++i) {
                uint flat_idx = thread_idx * A_ELEMS + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = next_k + col;
                half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
                A_tiles[buf_load][row][col] = val;
            }
        }
        
        // Compute Loop
        #pragma unroll
        for (uint kk = 0; kk < DECODE_K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            
            // Load A fragments for this K sub-tile
            simdgroup_matrix<half, 8, 8> a_frag[DECODE_SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi], &A_tiles[buf_compute][mi * 8][kk * 8], TILE_K);
            }
            
            // --- Helper lambda/macro for processing one projection ---
            #define PROCESS_PROJECTION(PACKED, SCALES, SU, SV, N_VAL, TILES_N, ACC) \
            if (N_VAL > 0) { \
                /* Optimization: Prefetch SU for this K-block */ \
                float su_cached[8]; \
                for (uint row = 0; row < 8; ++row) { \
                    uint k_idx = k_sub_base + row; \
                    su_cached[row] = (k_idx < K) ? SU[k_idx] : 0.0f; \
                } \
                 \
                /* Process N sub-tiles */ \
                for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) { \
                    uint b_col_base = tg_col + sg_col_offset + ni * 8; \
                     \
                    if (simd_lane_id < 8) { \
                        uint b_col = b_col_base + simd_lane_id; \
                        half dequant_vals[8]; \
                         \
                        if (b_col < N_VAL) { \
                            uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM; \
                            uint local_k_base = k_sub_base % TRELLIS_TILE_DIM; \
                            uint trellis_tile_n = b_col / TRELLIS_TILE_DIM; \
                            uint local_n = b_col % TRELLIS_TILE_DIM; \
                             \
                            uint tile_offset = (trellis_tile_k * TILES_N + trellis_tile_n) * packed_bytes; \
                            uint scale_idx = group_idx * N_VAL + b_col; \
                            float scale = SCALES[scale_idx]; \
                            float sign_n = SV[b_col]; \
                            float scale_sign_n = scale * sign_n; \
                             \
                            /* Dequantize 8 rows */ \
                            for (uint row = 0; row < 8; ++row) { \
                                uint k_idx = k_sub_base + row; \
                                uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM; \
                                \
                                if (k_idx < K) { \
                                    uint actual_tile_k = k_idx / TRELLIS_TILE_DIM; \
                                    uint actual_tile_offset = (actual_tile_k * TILES_N + trellis_tile_n) * packed_bytes; \
                                    uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k; \
                                     \
                                    uint trellis_idx = unpack_trellis_index(PACKED + actual_tile_offset, idx_in_tile, bits); \
                                    if (trellis_idx >= n_levels) trellis_idx = 0; \
                                     \
                                    float combined = scale_sign_n * su_cached[row]; \
                                    dequant_vals[row] = dequant_trellis_element_fused(trellis_idx, combined, grid); \
                                } else { \
                                    dequant_vals[row] = half(0.0h); \
                                } \
                            } \
                        } else { \
                            for (uint row = 0; row < 8; ++row) dequant_vals[row] = half(0.0h); \
                        } \
                         \
                        /* Write to staging */ \
                        for (uint row = 0; row < 8; ++row) { \
                            B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row]; \
                        } \
                    } \
                     \
                    simdgroup_barrier(mem_flags::mem_threadgroup); \
                     \
                    /* Load B fragment and MMA */ \
                    simdgroup_matrix<half, 8, 8> b_frag; \
                    simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8); \
                     \
                    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) { \
                        simdgroup_multiply_accumulate(ACC[mi][ni], a_frag[mi], b_frag, ACC[mi][ni]); \
                    } \
                    simdgroup_barrier(mem_flags::mem_threadgroup); \
                } \
            }
            
            // Process Q
            PROCESS_PROJECTION(packed_q, scales_q, su_q, sv_q, Nq, tiles_n_q, acc_q);
            
            // Process K
            PROCESS_PROJECTION(packed_k, scales_k, su_k, sv_k, Nk, tiles_n_k, acc_k);
            
            // Process V
            PROCESS_PROJECTION(packed_v, scales_v, su_v, sv_v, Nv, tiles_n_v, acc_v);
            
            #undef PROCESS_PROJECTION
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store outputs
    threadgroup half epilogue_staging[8][8];
    
    #define STORE_OUTPUT(ACC, OUT_BUF, N_VAL) \
    if (N_VAL > 0) { \
        for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) { \
            uint out_row = tg_row + mi * 8; \
            if (out_row >= M) continue; \
             \
            for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) { \
                uint out_col = tg_col + sg_col_offset + ni * 8; \
                 \
                if (out_row + 8 <= M && out_col + 8 <= N_VAL) { \
                    simdgroup_store(ACC[mi][ni], OUT_BUF + out_row * N_VAL + out_col, N_VAL); \
                    continue; \
                } \
                 \
                simdgroup_store(ACC[mi][ni], &epilogue_staging[0][0], 8); \
                simdgroup_barrier(mem_flags::mem_threadgroup); \
                 \
                for (uint elem = simd_lane_id; elem < 64; elem += 32) { \
                    uint r = elem / 8; \
                    uint c = elem % 8; \
                    uint gr = out_row + r; \
                    uint gc = out_col + c; \
                    if (gr < M && gc < N_VAL) { \
                        OUT_BUF[gr * N_VAL + gc] = epilogue_staging[r][c]; \
                    } \
                } \
                simdgroup_barrier(mem_flags::mem_threadgroup); \
            } \
        } \
    }
    
    STORE_OUTPUT(acc_q, OutQ, Nq);
    STORE_OUTPUT(acc_k, OutK, Nk);
    STORE_OUTPUT(acc_v, OutV, Nv);
}