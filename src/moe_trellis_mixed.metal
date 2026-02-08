#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Kernel Configuration
// ---------------------------------------------------------------------------

constant constexpr uint MOE_TILE_N = 64;
constant constexpr uint MOE_TILE_K = 16;
constant constexpr uint TRELLIS_TILE = 16;
constant constexpr uint MOE_TILE_N_STRIDE = 68;
constant constexpr uint MOE_THREADS = 256;

constant constexpr uint PACKED_BYTES_2BIT = 64;
constant constexpr uint PACKED_BYTES_3BIT = 96;
constant constexpr uint PACKED_BYTES_4BIT = 128;
constant constexpr uint PACKED_BYTES_8BIT = 256;

constant constexpr uint SU_CACHE_SIZE = MOE_TILE_K;
constant constexpr uint SV_CACHE_SIZE = MOE_TILE_N;

struct MoEParams {
    uint batch_size;
    uint hidden_dim;
    uint intermediate_dim;
    uint num_experts;
    uint top_k;
    uint gate_bits;
    uint up_bits;
    uint down_bits;
    uint tile_size;
    uint gate_n_levels;
    uint up_n_levels;
    uint down_n_levels;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline half fast_silu_scalar(half x) {
    return x / (1.0h + exp(-x));
}

inline float fast_silu_scalar_f32(float x) {
    return x / (1.0f + exp(-x));
}

inline uint unpack_2bit_single(device const uint8_t* packed, uint idx) {
    uint byte_idx = idx >> 2;
    uint bit_shift = (idx & 3) << 1;
    return (packed[byte_idx] >> bit_shift) & 0x3;
}

inline uint unpack_3bit_val(device const uint8_t* base, uint bit_offset) {
    uint byte_idx = bit_offset >> 3;
    uint bit_in_byte = bit_offset & 7;
    uint val = uint(base[byte_idx]);
    if (bit_in_byte + 3 > 8) {
        val |= uint(base[byte_idx + 1]) << 8;
    }
    return (val >> bit_in_byte) & 0x7;
}

inline uint unpack_4bit_single(device const uint8_t* packed, uint idx) {
    uint byte_idx = idx >> 1;
    uint bit_shift = (idx & 1) << 2;
    return (packed[byte_idx] >> bit_shift) & 0xF;
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

inline half trellis_dequant_mixed(
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
    uint bit_width
) {
    uint num_tiles_n = (N_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint tile_k = global_k / TRELLIS_TILE;
    uint tile_n = global_n / TRELLIS_TILE;
    uint k_in_tile = global_k % TRELLIS_TILE;
    uint n_in_tile = global_n % TRELLIS_TILE;
    uint tile_idx = tile_k * num_tiles_n + tile_n;

    uint idx_in_tile = n_in_tile * TRELLIS_TILE + k_in_tile;
    
    uint codebook_idx = 0;
    if (bit_width == 2) {
        device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_2BIT;
        codebook_idx = unpack_2bit_single(tile_packed, idx_in_tile);
    } else if (bit_width == 3) {
        device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_3BIT;
        codebook_idx = unpack_3bit_val(tile_packed, idx_in_tile * 3);
    } else if (bit_width == 4) {
        device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_4BIT;
        codebook_idx = unpack_4bit_single(tile_packed, idx_in_tile);
    } else {
        device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_8BIT;
        codebook_idx = uint(tile_packed[idx_in_tile]);
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

// ---------------------------------------------------------------------------
// Tile Loaders
// ---------------------------------------------------------------------------

inline void load_trellis_tile_mixed_decode(
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
    uint bit_width,
    uint thread_idx
) {
    for (uint i = thread_idx; i < MOE_TILE_K * MOE_TILE_N; i += MOE_THREADS) {
        uint k_local = i / MOE_TILE_N;
        uint n_local = i % MOE_TILE_N;
        uint global_k = k_block + k_local;
        uint global_n = n_block + n_local;

        half val = 0.0h;
        if (global_k < K_dim && global_n < N_dim) {
            val = trellis_dequant_mixed(
                packed_weights, scales, su, sv, grid,
                expert_id, global_k, global_n,
                K_dim, N_dim, group_size, bit_width
            );
        }
        B_buf[k_local][n_local] = val;
    }
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

template<bool USE_PREFETCH>
inline void moe_trellis_mixed_impl(
    device const half* activations,
    device const uint8_t* gate_weights,
    device const half* gate_scales,
    device const uint8_t* up_weights,
    device const half* up_scales,
    device const uint8_t* down_weights,
    device const half* down_scales,
    device const half* gate_su,
    device const half* gate_sv,
    device const half* up_su,
    device const half* up_sv,
    device const half* down_su,
    device const half* down_sv,
    device const half* grid,
    device const uint* expert_ids,
    device const half* expert_probs,
    device float* output,
    constant MoEParams& p,
    device const uint8_t* expert_bits,
    uint3 tgid,
    uint thread_idx,
    threadgroup half (&A_tile)[MOE_TILE_K],
    threadgroup half (&B_gate)[MOE_TILE_K][MOE_TILE_N_STRIDE],
    threadgroup half (&B_up)[MOE_TILE_K][MOE_TILE_N_STRIDE],
    threadgroup half (&B_down)[MOE_TILE_K][MOE_TILE_N_STRIDE],
    threadgroup float (&swiglu_result)[MOE_TILE_N + 4],
    threadgroup float (&output_tile)[MOE_TILE_N + 4]
) {
    const uint n_block = tgid.x * MOE_TILE_N;
    const uint token_idx = tgid.y;
    const uint slot = tgid.z;

    if (token_idx >= p.batch_size || slot >= p.top_k) return;

    const uint expert_id = expert_ids[token_idx * p.top_k + slot];
    if (expert_id >= p.num_experts) return;

    const float prob = float(expert_probs[token_idx * p.top_k + slot]);
    const uint bit_width = expert_bits[expert_id];

    // Expert size calculation
    uint packed_bytes_per_tile = 0;
    if (bit_width == 2) packed_bytes_per_tile = PACKED_BYTES_2BIT;
    else if (bit_width == 3) packed_bytes_per_tile = PACKED_BYTES_3BIT;
    else if (bit_width == 4) packed_bytes_per_tile = PACKED_BYTES_4BIT;
    else packed_bytes_per_tile = PACKED_BYTES_8BIT;

    uint num_tiles_k_gate = (p.hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (p.intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * packed_bytes_per_tile;

    uint num_tiles_k_down = (p.intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (p.hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * packed_bytes_per_tile;

    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        output_tile[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_intermediate_chunks = (p.intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (p.hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;
        
        float my_gate_acc = 0.0f;
        float my_up_acc = 0.0f;

        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            for (uint k = thread_idx; k < MOE_TILE_K; k += MOE_THREADS) {
                uint global_k = k_block + k;
                A_tile[k] = (global_k < p.hidden_dim) ? activations[token_idx * p.hidden_dim + global_k] : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (thread_idx < MOE_TILE_N) {
                uint global_n = n_chunk_offset + thread_idx;
                if (global_n < p.intermediate_dim) {
                    for (uint k = 0; k < MOE_TILE_K; ++k) {
                        uint global_k = k_block + k;
                        if (global_k >= p.hidden_dim) break;

                        float act = float(A_tile[k]);
                        float g_val = float(trellis_dequant_mixed(gate_w, gate_scales, gate_su, gate_sv, grid, 
                            expert_id, global_k, global_n, p.hidden_dim, p.intermediate_dim, p.tile_size, bit_width));
                        float u_val = float(trellis_dequant_mixed(up_w, up_scales, up_su, up_sv, grid, 
                            expert_id, global_k, global_n, p.hidden_dim, p.intermediate_dim, p.tile_size, bit_width));

                        my_gate_acc += act * g_val;
                        my_up_acc += act * u_val;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (thread_idx < MOE_TILE_N) {
            uint global_n = n_chunk_offset + thread_idx;
            if (global_n < p.intermediate_dim) {
                swiglu_result[thread_idx] = fast_silu_scalar_f32(my_gate_acc) * my_up_acc;
            } else {
                swiglu_result[thread_idx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down projection
        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, p.intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            load_trellis_tile_mixed_decode(down_w, down_scales, down_su, down_sv, grid, B_down, 
                k_down_global, n_block, expert_id, p.intermediate_dim, p.hidden_dim, p.tile_size, bit_width, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                uint global_out_n = n_block + i;
                if (global_out_n < p.hidden_dim) {
                    float local_down = 0.0f;
                    uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                    for (uint k = 0; k < k_end; ++k) {
                        local_down += swiglu_result[k_down_local + k] * float(B_down[k][i]);
                    }
                    output_tile[i] += local_down;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Atomic write to output
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < p.hidden_dim) {
            float weighted = output_tile[i] * prob;
            uint out_idx = token_idx * p.hidden_dim + global_n;
            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[out_idx]);
            uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_bits;
            bool success;
            do {
                float old_val = as_type<float>(old_bits);
                float new_val = old_val + weighted;
                new_bits = as_type<uint>(new_val);
                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_bits, new_bits,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

#define MOE_TRELLIS_MIXED_ARGS \
    device const half* activations       [[buffer(0)]], \
    device const uint8_t* gate_weights   [[buffer(1)]], \
    device const half* gate_scales       [[buffer(2)]], \
    device const uint8_t* up_weights     [[buffer(3)]], \
    device const half* up_scales         [[buffer(4)]], \
    device const uint8_t* down_weights   [[buffer(5)]], \
    device const half* down_scales       [[buffer(6)]], \
    device const half* gate_su           [[buffer(7)]], \
    device const half* gate_sv           [[buffer(8)]], \
    device const half* up_su             [[buffer(9)]], \
    device const half* up_sv             [[buffer(10)]], \
    device const half* down_su           [[buffer(11)]], \
    device const half* down_sv           [[buffer(12)]], \
    device const half* grid              [[buffer(13)]], \
    device const uint* expert_ids        [[buffer(14)]], \
    device const half* expert_probs      [[buffer(15)]], \
    device float* output                 [[buffer(16)]], \
    constant MoEParams& p                [[buffer(17)]], \
    device const uint8_t* expert_bits    [[buffer(21)]], \
    uint3 tgid                           [[threadgroup_position_in_grid]], \
    uint thread_idx                      [[thread_index_in_threadgroup]]

kernel void moe_trellis_mixed_swiglu_decode(MOE_TRELLIS_MIXED_ARGS) {
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup float swiglu_result[MOE_TILE_N + 4];
    threadgroup float output_tile[MOE_TILE_N + 4];
    
    moe_trellis_mixed_impl<true>(
        activations, gate_weights, gate_scales, up_weights, up_scales, down_weights, down_scales,
        gate_su, gate_sv, up_su, up_sv, down_su, down_sv, grid, expert_ids, expert_probs,
        output, p, expert_bits, tgid, thread_idx,
        A_tile, B_gate, B_up, B_down, swiglu_result, output_tile
    );
}

kernel void moe_trellis_mixed_swiglu_prefill(MOE_TRELLIS_MIXED_ARGS) {
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup float swiglu_result[MOE_TILE_N + 4];
    threadgroup float output_tile[MOE_TILE_N + 4];
    
    moe_trellis_mixed_impl<true>(
        activations, gate_weights, gate_scales, up_weights, up_scales, down_weights, down_scales,
        gate_su, gate_sv, up_su, up_sv, down_su, down_sv, grid, expert_ids, expert_probs,
        output, p, expert_bits, tgid, thread_idx,
        A_tile, B_gate, B_up, B_down, swiglu_result, output_tile
    );
}

kernel void moe_trellis_mixed_swiglu(MOE_TRELLIS_MIXED_ARGS) {
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup float swiglu_result[MOE_TILE_N + 4];
    threadgroup float output_tile[MOE_TILE_N + 4];
    
    moe_trellis_mixed_impl<false>(
        activations, gate_weights, gate_scales, up_weights, up_scales, down_weights, down_scales,
        gate_su, gate_sv, up_su, up_sv, down_su, down_sv, grid, expert_ids, expert_probs,
        output, p, expert_bits, tgid, thread_idx,
        A_tile, B_gate, B_up, B_down, swiglu_result, output_tile
    );
}

kernel void moe_trellis_mixed_swiglu_large(MOE_TRELLIS_MIXED_ARGS) {
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N_STRIDE];
    threadgroup float swiglu_result[MOE_TILE_N + 4];
    threadgroup float output_tile[MOE_TILE_N + 4];
    
    moe_trellis_mixed_impl<false>(
        activations, gate_weights, gate_scales, up_weights, up_scales, down_weights, down_scales,
        gate_su, gate_sv, up_su, up_sv, down_su, down_sv, grid, expert_ids, expert_probs,
        output, p, expert_bits, tgid, thread_idx,
        A_tile, B_gate, B_up, B_down, swiglu_result, output_tile
    );
}