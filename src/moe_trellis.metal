#include <metal_stdlib>
using namespace metal;

constant constexpr uint MOE_TILE_N = 64;
constant constexpr uint MOE_TILE_K = 16;
constant constexpr uint TRELLIS_TILE = 16;
constant constexpr uint PACKED_BYTES_3BIT = 96;
constant constexpr uint MOE_TILE_N_STRIDE = 68;
constant constexpr uint MOE_THREADS = 256;

// ---------------------------------------------------------------------------
// Helpers copied from gemm_trellis_moe.metal
// ---------------------------------------------------------------------------

inline half trellis_dequant_3bit(
    device const uint8_t* packed_weights,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    uint expert_id,
    uint global_k,
    uint global_n,
    uint K_dim,
    uint N_dim,
    uint group_size,
    uint n_levels
) {
    uint num_tiles_n = (N_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint tile_k = global_k / TRELLIS_TILE;
    uint tile_n = global_n / TRELLIS_TILE;
    uint k_in_tile = global_k % TRELLIS_TILE;
    uint n_in_tile = global_n % TRELLIS_TILE;
    uint tile_idx = tile_k * num_tiles_n + tile_n;

    device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_3BIT;

    uint idx_in_tile = n_in_tile * TRELLIS_TILE + k_in_tile;
    uint bit_offset = idx_in_tile * 3;
    uint byte_idx = bit_offset >> 3;
    uint bit_in_byte = bit_offset & 7;

    uint packed_val = uint(tile_packed[byte_idx]);
    if (bit_in_byte + 3 > 8) {
        packed_val |= uint(tile_packed[byte_idx + 1]) << 8;
    }
    uint codebook_idx = (packed_val >> bit_in_byte) & 0x7;

    if (codebook_idx >= n_levels) {
        codebook_idx = 0;
    }

    half dequant = grid[codebook_idx];

    uint n_groups = (K_dim + group_size - 1) / group_size;
    uint group_idx = global_k / group_size;
    half scale = scales[expert_id * N_dim * n_groups + group_idx * N_dim + global_n];
    dequant *= scale;

    dequant *= su[expert_id * K_dim + global_k];
    dequant *= sv[expert_id * N_dim + global_n];

    return dequant;
}

inline void load_trellis_tile(
    device const uint8_t* packed_weights,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N_STRIDE],
    uint k_block,
    uint n_block,
    uint expert_id,
    uint K_dim,
    uint N_dim,
    uint group_size,
    uint n_levels,
    uint thread_idx
) {
    const uint elems_per_thread = (MOE_TILE_K * MOE_TILE_N) / MOE_THREADS;
    const uint vec_elems_per_thread = elems_per_thread / 4;
    const uint remainder = elems_per_thread % 4;

    // Process 4 elements at a time using vectorized loads
    for (uint i = 0; i < vec_elems_per_thread; ++i) {
        uint base_flat_idx = thread_idx * elems_per_thread + i * 4;
        
        for (uint j = 0; j < 4; ++j) {
            uint flat_idx = base_flat_idx + j;
            uint k_local = flat_idx / MOE_TILE_N;
            uint n_local = flat_idx % MOE_TILE_N;
            uint global_k = k_block + k_local;
            uint global_n = n_block + n_local;

            half val = 0.0h;
            if (global_k < K_dim && global_n < N_dim) {
                val = trellis_dequant_3bit(
                    packed_weights, scales, su, sv, grid,
                    expert_id, global_k, global_n,
                    K_dim, N_dim, group_size, n_levels
                );
            }
            B_buf[k_local][n_local] = val;
        }
    }

    // Handle remaining elements
    for (uint i = 0; i < remainder; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + vec_elems_per_thread * 4 + i;
        uint k_local = flat_idx / MOE_TILE_N;
        uint n_local = flat_idx % MOE_TILE_N;
        uint global_k = k_block + k_local;
        uint global_n = n_block + n_local;

        half val = 0.0h;
        if (global_k < K_dim && global_n < N_dim) {
            val = trellis_dequant_3bit(
                packed_weights, scales, su, sv, grid,
                expert_id, global_k, global_n,
                K_dim, N_dim, group_size, n_levels
            );
        }
        B_buf[k_local][n_local] = val;
    }
}

inline void load_activation_tile(
    device const half* activations,
    threadgroup half (&A_buf)[MOE_TILE_K],
    uint token_idx,
    uint k_block,
    uint hidden_dim,
    uint thread_idx
) {
    constexpr uint HALFS_PER_THREAD = 4;
    constexpr uint THREADS_NEEDED = MOE_TILE_K / HALFS_PER_THREAD;

    if (thread_idx < THREADS_NEEDED) {
        uint lane_offset = thread_idx * HALFS_PER_THREAD;
        uint global_k = k_block + lane_offset;
        device const half* row_base = activations + token_idx * hidden_dim;

        if (global_k + HALFS_PER_THREAD <= hidden_dim) {
            half4 chunk = *((device const half4*)(row_base + global_k));
            A_buf[lane_offset + 0] = chunk.x;
            A_buf[lane_offset + 1] = chunk.y;
            A_buf[lane_offset + 2] = chunk.z;
            A_buf[lane_offset + 3] = chunk.w;
        } else {
            for (uint i = 0; i < HALFS_PER_THREAD; ++i) {
                uint k = global_k + i;
                A_buf[lane_offset + i] = (k < hidden_dim) ? row_base[k] : 0.0h;
            }
        }
    }
}

inline half fast_silu_scalar(half x) {
    half x2 = x * x;
    half sigmoid = 0.5h * (1.0h + tanh(0.797885h * x * (1.0h + 0.044715h * x2)));
    return x * sigmoid;
}

inline half4 fast_silu_vec4(half4 x) {
    half4 x2 = x * x;
    half4 sigmoid = 0.5h * (1.0h + tanh(0.797885h * x * (1.0h + 0.044715h * x2)));
    return x * sigmoid;
}

// ---------------------------------------------------------------------------
// Fused MoE Kernel for Single Expert
// ---------------------------------------------------------------------------

kernel void moe_trellis_fused_expert(
    device const half* activations       [[buffer(0)]],   // [batch, hidden] (for this expert)
    device const uint8_t* gate_weights   [[buffer(1)]],   // [num_experts, ...]
    device const half* gate_scales       [[buffer(2)]],   // [num_experts, ...]
    device const uint8_t* up_weights     [[buffer(3)]],   // [num_experts, ...]
    device const half* up_scales         [[buffer(4)]],   // [num_experts, ...]
    device const uint8_t* down_weights   [[buffer(5)]],   // [num_experts, ...]
    device const half* down_scales       [[buffer(6)]],   // [num_experts, ...]
    device const half* gate_su           [[buffer(7)]],   // [num_experts, K]
    device const half* gate_sv           [[buffer(8)]],   // [num_experts, N]
    device const half* up_su             [[buffer(9)]],   // [num_experts, K]
    device const half* up_sv             [[buffer(10)]],  // [num_experts, N]
    device const half* down_su           [[buffer(11)]],  // [num_experts, N]
    device const half* down_sv           [[buffer(12)]],  // [num_experts, K]
    device const half* grid              [[buffer(13)]],  // Codebook grid
    device float* output                 [[buffer(14)]],  // [batch, hidden] (for this expert)
    
    constant uint& batch_size            [[buffer(15)]],
    constant uint& hidden_dim            [[buffer(16)]],
    constant uint& intermediate_dim      [[buffer(17)]],
    constant uint& expert_id             [[buffer(18)]],
    constant uint& group_size            [[buffer(19)]],
    constant uint& n_levels              [[buffer(20)]],

    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half swiglu_result[MOE_TILE_N + 1];
    threadgroup half output_tile[MOE_TILE_N + 1];
    threadgroup half gate_acc_tg[MOE_TILE_N + 1];
    threadgroup half up_acc_tg[MOE_TILE_N + 1];

    const uint n_block = tgid.x * MOE_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) {
        return;
    }

    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        output_tile[i] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;

        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            gate_acc_tg[i] = 0.0h;
            up_acc_tg[i] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;
            load_activation_tile(activations, A_tile, token_idx, k_block, hidden_dim, thread_idx);

            load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                            B_gate, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, group_size, n_levels, thread_idx);

            load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                            B_up, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, group_size, n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each simdgroup (32 threads) handles 16 columns (2 threads each)
            uint lane_id = thread_idx % 32;
            uint sg_id = thread_idx / 32;
            uint col_in_sg = lane_id % 16;
            uint lane_in_col = lane_id / 16;
            uint i = sg_id * 16 + col_in_sg;

            half local_gate = 0.0h;
            half local_up = 0.0h;
            
            // Each thread handles 8 elements of the 16-element K-dim (MOE_TILE_K)
            uint k_start = lane_in_col * 8;
            for (uint k = k_start; k < k_start + 8; ++k) {
                half a = A_tile[k];
                local_gate = fma(a, B_gate[k][i], local_gate);
                local_up = fma(a, B_up[k][i], local_up);
            }

            // Reduce within column using SIMD shuffle (lanes are 16 apart)
            local_gate += simd_shuffle_xor(local_gate, 16);
            local_up += simd_shuffle_xor(local_up, 16);

            if (lane_in_col == 0) {
                gate_acc_tg[i] += local_gate;
                up_acc_tg[i] += local_up;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint i = thread_idx * 4; i < MOE_TILE_N; i += MOE_THREADS * 4) {
            uint global_n = n_chunk_offset + i;
            if (global_n + 3 < intermediate_dim) {
                half4 g = half4(gate_acc_tg[i], gate_acc_tg[i+1], gate_acc_tg[i+2], gate_acc_tg[i+3]);
                half4 u = half4(up_acc_tg[i], up_acc_tg[i+1], up_acc_tg[i+2], up_acc_tg[i+3]);
                half4 silu_g = fast_silu_vec4(g);
                half4 res = silu_g * u;
                swiglu_result[i] = res.x;
                swiglu_result[i+1] = res.y;
                swiglu_result[i+2] = res.z;
                swiglu_result[i+3] = res.w;
            } else {
                for (uint j = 0; j < 4 && i + j < MOE_TILE_N; ++j) {
                    if (global_n + j < intermediate_dim) {
                        half g = gate_acc_tg[i + j];
                        half u = up_acc_tg[i + j];
                        swiglu_result[i + j] = fast_silu_scalar(g) * u;
                    } else {
                        swiglu_result[i + j] = 0.0h;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            load_trellis_tile(down_w, down_scales, down_su, down_sv, grid,
                            B_down, k_down_global, n_block, expert_id,
                            intermediate_dim, hidden_dim, group_size, n_levels, thread_idx);
            
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Optimized down projection: 2 threads per column collab using SIMD shuffle
            uint lane_id = thread_idx % 32;
            uint sg_id = thread_idx / 32;
            uint col_in_sg = lane_id % 16;
            uint lane_in_col = lane_id / 16;
            uint i = sg_id * 16 + col_in_sg;

            uint global_out_n = n_block + i;
            if (global_out_n < hidden_dim) {
                half local_down = 0.0h;
                uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                
                uint k_start = lane_in_col * 8;
                for (uint k = k_start; k < min(k_start + 8, k_end); ++k) {
                    local_down = fma(swiglu_result[k_down_local + k], B_down[k][i], local_down);
                }
                
                // Shuffle to combine partial results from the 2 threads in this column
                local_down += simd_shuffle_xor(local_down, 16);
                
                if (lane_in_col == 0) {
                    output_tile[i] += local_down;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            output[token_idx * hidden_dim + global_n] = float(output_tile[i]);
        }
    }
}
