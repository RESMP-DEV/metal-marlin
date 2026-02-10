// gemm_trellis_swiglu.metal - Fused SwiGLU + GEMM Kernel for Trellis-Quantized Weights
// ============================================================================
//
// Fuses the gate and up projections of a SwiGLU MLP into a single kernel:
//   output = silu(x @ W_gate) * (x @ W_up)
//
// This reduces memory bandwidth by:
// 1. Reading input activations 'x' only once for both projections.
// 2. Applying activation and multiplication in registers/threadgroup memory.
// 3. Writing only the activated intermediate result back to DRAM.
//
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// Constants and Helpers (Consistent with gemm_trellis.metal)
// ============================================================================

// Main GEMM tile dimensions
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
constant constexpr uint DECODE_THREADS = 128; // 4 simdgroups

// Fast SiLU approximation using rsqrt
inline half fast_silu(half x) {
    float fx = (float)x;
    float x2 = fx * fx;
    float sigmoid_approx = 0.5f + 0.5f * fx * rsqrt(1.0f + x2);
    return half(fx * sigmoid_approx);
}

inline uint unpack_trellis_index(device const uchar* packed, uint idx_in_tile, uint bits) {
    switch (bits) {
        case 2: {
            uint byte_idx = idx_in_tile >> 2;
            uint bit_offset = (idx_in_tile & 3) << 1;
            return (packed[byte_idx] >> bit_offset) & 0x3;
        }
        case 3: {
            uint bit_offset = idx_in_tile * 3;
            uint byte_idx = bit_offset >> 3;
            uint bit_in_byte = bit_offset & 7;
            uint packed_val = uint(packed[byte_idx]);
            if (bit_in_byte + 3 > 8) {
                packed_val |= uint(packed[byte_idx + 1]) << 8;
            }
            return (packed_val >> bit_in_byte) & 0x7;
        }
        case 4: {
            uint byte_idx = idx_in_tile >> 1;
            uint shift = (idx_in_tile & 1) << 2;
            return (packed[byte_idx] >> shift) & 0xF;
        }
        default: return 0;
    }
}

inline uint packed_bytes_per_trellis_tile(uint bits) {
    return (TRELLIS_TILE_SIZE * bits + 7) / 8;
}

// ============================================================================
// Fused Gate+Up SwiGLU Kernel
// ============================================================================

kernel void gemm_trellis_swiglu_gate_up(
    device const half* A               [[buffer(0)]],   // [M, K]
    device const uchar* packed_gate    [[buffer(1)]],   // [tiles_k, tiles_n, packed]
    device const float* scales_gate    [[buffer(2)]],   // [n_groups, N]
    device const uchar* packed_up      [[buffer(3)]],   // [tiles_k, tiles_n, packed]
    device const float* scales_up      [[buffer(4)]],   // [n_groups, N]
    constant float* grid               [[buffer(5)]],   // [n_levels]
    constant float* su                 [[buffer(6)]],   // [K]
    constant float* sv                 [[buffer(7)]],   // [N]
    device half* C                     [[buffer(8)]],   // [M, N] output intermediate
    constant uint& M                   [[buffer(9)]],
    constant uint& K                   [[buffer(10)]],
    constant uint& N                   [[buffer(11)]],
    constant uint& bits                [[buffer(12)]],
    constant uint& n_levels            [[buffer(13)]],
    constant uint& group_size          [[buffer(14)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_gate_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup half B_up_staging[SIMDGROUPS_PER_TG][8][8];
    
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    
    if (tg_row >= M) return;
    
    // Simdgroup layout: 4x2 grid covering 128x128
    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 64;
    
    // Two sets of accumulators
    simdgroup_matrix<half, 8, 8> acc_gate[SG_M_TILES][SG_N_TILES];
    simdgroup_matrix<half, 8, 8> acc_up[SG_M_TILES][SG_N_TILES];
    
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc_gate[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
            acc_up[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }
    
    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    // Cooperative A tile load
    constexpr uint A_ELEMS = (TILE_M * TILE_K) / THREADS_PER_TG;
    #pragma unroll
    for (uint i = 0; i < A_ELEMS; ++i) {
        uint flat_idx = thread_idx * A_ELEMS + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        uint global_col = col;
        A_tiles[0][row][col] = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint buf_compute = 0;
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < A_ELEMS; ++i) {
                uint flat_idx = thread_idx * A_ELEMS + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = next_k + col;
                A_tiles[buf_load][row][col] = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
            }
        }
        
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi], &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8], TILE_K);
            }
            
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;
                
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_gate[8];
                    half dequant_up[8];
                    
                    if (b_col < N) {
                        uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
                        uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        
                        uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                        uint scale_idx = group_idx * N + b_col;
                        
                        float scale_g = scales_gate[scale_idx];
                        float scale_u = scales_up[scale_idx];
                        float sign_n = sv[b_col];
                        
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM;
                            
                            if (k_idx < K) {
                                uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                                uint actual_tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                                uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;
                                
                                uint idx_g = unpack_trellis_index(packed_gate + actual_tile_offset, idx_in_tile, bits);
                                uint idx_u = unpack_trellis_index(packed_up + actual_tile_offset, idx_in_tile, bits);
                                
                                float combined_sign = sign_n * su[k_idx];
                                dequant_gate[row] = half(grid[idx_g] * scale_g * combined_sign);
                                dequant_up[row] = half(grid[idx_u] * scale_u * combined_sign);
                            } else {
                                dequant_gate[row] = half(0.0h);
                                dequant_up[row] = half(0.0h);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_gate[row] = half(0.0h);
                            dequant_up[row] = half(0.0h);
                        }
                    }
                    
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_gate_staging[simd_group_id][row][simd_lane_id] = dequant_gate[row];
                        B_up_staging[simd_group_id][row][simd_lane_id] = dequant_up[row];
                    }
                }
                
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                simdgroup_matrix<half, 8, 8> b_gate_frag;
                simdgroup_matrix<half, 8, 8> b_up_frag;
                simdgroup_load(b_gate_frag, &B_gate_staging[simd_group_id][0][0], 8);
                simdgroup_load(b_up_frag, &B_up_staging[simd_group_id][0][0], 8);
                
                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc_gate[mi][ni], a_frag[mi], b_gate_frag, acc_gate[mi][ni]);
                    simdgroup_multiply_accumulate(acc_up[mi][ni], a_frag[mi], b_up_frag, acc_up[mi][ni]);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store outputs: silu(gate) * up
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;
        
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            if (out_col >= N) continue;
            
            simdgroup_store(acc_gate[mi][ni], &B_gate_staging[simd_group_id][0][0], 8);
            simdgroup_store(acc_up[mi][ni], &B_up_staging[simd_group_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < M && gc < N) {
                    half g = B_gate_staging[simd_group_id][r][c];
                    half u = B_up_staging[simd_group_id][r][c];
                    C[gr * N + gc] = fast_silu(g) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// Fused Decode Kernel (M <= 16)
// ============================================================================

kernel void gemm_trellis_swiglu_gate_up_decode(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_gate    [[buffer(1)]],
    device const float* scales_gate    [[buffer(2)]],
    device const uchar* packed_up      [[buffer(3)]],
    device const float* scales_up      [[buffer(4)]],
    constant float* grid               [[buffer(5)]],
    constant float* su                 [[buffer(6)]],
    constant float* sv                 [[buffer(7)]],
    device half* C                     [[buffer(8)]],
    constant uint& M                   [[buffer(9)]],
    constant uint& K                   [[buffer(10)]],
    constant uint& N                   [[buffer(11)]],
    constant uint& bits                [[buffer(12)]],
    constant uint& n_levels            [[buffer(13)]],
    constant uint& group_size          [[buffer(14)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    if (simd_group_id >= 4) return; // Only 4 simdgroups for decode

    // 32x128x32 tiles
    threadgroup half A_tiles[2][DECODE_TILE_M][TILE_K];
    threadgroup half B_gate_staging[4][8][8];
    threadgroup half B_up_staging[4][8][8];
    
    const uint tg_row = tgid.y * DECODE_TILE_M;
    const uint tg_col = tgid.x * DECODE_TILE_N;
    
    if (tg_row >= M) return;
    
    // 4 simdgroups tile 32x128 as 1x4 (each handles 32x32)
    const uint sg_row_offset = 0;
    const uint sg_col_offset = simd_group_id * 32;
    
    simdgroup_matrix<half, 8, 8> acc_gate[DECODE_SG_M_TILES][DECODE_SG_N_TILES];
    simdgroup_matrix<half, 8, 8> acc_up[DECODE_SG_M_TILES][DECODE_SG_N_TILES];
    
    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            acc_gate[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
            acc_up[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }
    
    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    constexpr uint A_ELEMS = (DECODE_TILE_M * TILE_K) / DECODE_THREADS;
    #pragma unroll
    for (uint i = 0; i < A_ELEMS; ++i) {
        uint flat_idx = thread_idx * A_ELEMS + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        A_tiles[0][row][col] = (global_row < M && col < K) ? A[global_row * K + col] : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint buf_compute = 0;
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < A_ELEMS; ++i) {
                uint flat_idx = thread_idx * A_ELEMS + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = next_k + col;
                A_tiles[buf_load][row][col] = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
            }
        }
        
        #pragma unroll
        for (uint kk = 0; kk < DECODE_K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            
            simdgroup_matrix<half, 8, 8> a_frag[DECODE_SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi], &A_tiles[buf_compute][mi * 8][kk * 8], TILE_K);
            }
            
            #pragma unroll
            for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;
                
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_gate[8];
                    half dequant_up[8];
                    
                    if (b_col < N) {
                        uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
                        uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        
                        uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                        uint scale_idx = group_idx * N + b_col;
                        
                        float scale_g = scales_gate[scale_idx];
                        float scale_u = scales_up[scale_idx];
                        float sign_n = sv[b_col];
                        
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM;
                            
                            if (k_idx < K) {
                                uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                                uint actual_tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                                uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;
                                
                                uint idx_g = unpack_trellis_index(packed_gate + actual_tile_offset, idx_in_tile, bits);
                                uint idx_u = unpack_trellis_index(packed_up + actual_tile_offset, idx_in_tile, bits);
                                
                                float combined_sign = sign_n * su[k_idx];
                                dequant_gate[row] = half(grid[idx_g] * scale_g * combined_sign);
                                dequant_up[row] = half(grid[idx_u] * scale_u * combined_sign);
                            } else {
                                dequant_gate[row] = half(0.0h);
                                dequant_up[row] = half(0.0h);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_gate[row] = half(0.0h);
                            dequant_up[row] = half(0.0h);
                        }
                    }
                    
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_gate_staging[simd_group_id][row][simd_lane_id] = dequant_gate[row];
                        B_up_staging[simd_group_id][row][simd_lane_id] = dequant_up[row];
                    }
                }
                
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                simdgroup_matrix<half, 8, 8> b_gate_frag;
                simdgroup_matrix<half, 8, 8> b_up_frag;
                simdgroup_load(b_gate_frag, &B_gate_staging[simd_group_id][0][0], 8);
                simdgroup_load(b_up_frag, &B_up_staging[simd_group_id][0][0], 8);
                
                #pragma unroll
                for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc_gate[mi][ni], a_frag[mi], b_gate_frag, acc_gate[mi][ni]);
                    simdgroup_multiply_accumulate(acc_up[mi][ni], a_frag[mi], b_up_frag, acc_up[mi][ni]);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store
    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        uint out_row = tg_row + mi * 8;
        if (out_row >= M) continue;
        
        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            if (out_col >= N) continue;
            
            simdgroup_store(acc_gate[mi][ni], &B_gate_staging[simd_group_id][0][0], 8);
            simdgroup_store(acc_up[mi][ni], &B_up_staging[simd_group_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < M && gc < N) {
                    half g = B_gate_staging[simd_group_id][r][c];
                    half u = B_up_staging[simd_group_id][r][c];
                    C[gr * N + gc] = fast_silu(g) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
