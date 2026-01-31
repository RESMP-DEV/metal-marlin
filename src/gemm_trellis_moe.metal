// gemm_trellis_moe.metal - Fused MoE GEMM with Trellis 3bpw quantization
//
// Single kernel that handles:
//   1. Token routing to top-k experts
//   2. Trellis dequantization (3-bit EXL3) on-the-fly
//   3. SwiGLU activation (gate_proj, up_proj, down_proj)
//   4. Expert probability weighting
//
// STREAMING CHUNK APPROACH for handling intermediate_dim > 64:
// For each chunk of MOE_TILE_N (64) intermediate columns:
//   1. Compute gate_chunk and up_chunk via full K-reduction over hidden_dim
//   2. Apply SwiGLU: swiglu_chunk = silu(gate_chunk) * up_chunk
//   3. Compute partial down contribution: acc_down += swiglu_chunk @ down_weights_chunk
// This avoids materializing the full intermediate result while correctly handling
// arbitrary intermediate dimensions.
//
// Grid layout: (ceil(hidden_dim / MOE_TILE_N), M, top_k)
//   - tgid.x: output column block
//   - tgid.y: token index
//   - tgid.z: expert slot
//
// Memory layout:
//   activations:     [batch, hidden] half
//   gate_weights:    [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   gate_scales:     [num_experts, n_groups, n] half
//   up_weights:      [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   up_scales:       [num_experts, n_groups, n] half
//   down_weights:    [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   down_scales:     [num_experts, n_groups, n] half
//   expert_ids:      [batch, top_k] uint32
//   expert_probs:    [batch, top_k] half
//   su/sv:           [num_experts, ...] half (sign flips)
//   grids:           [bits_to_level] half (codebook lookup)
//   output:          [batch, hidden] half

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant constexpr uint MOE_TILE_N = 64;   // Output/intermediate columns per threadgroup
constant constexpr uint MOE_TILE_K = 16;   // K dimension tile
constant constexpr uint MOE_SIMDGROUPS = 4;
constant constexpr uint MOE_THREADS = MOE_SIMDGROUPS * 32;  // 128 threads

constant constexpr uint TRELLIS_TILE = 16;
constant constexpr uint PACKED_BYTES_3BIT = 96;  // 16*16*3/8

// ---------------------------------------------------------------------------
// Debug NaN stage indices (for MOE_DEBUG_NAN mode)
// ---------------------------------------------------------------------------
#ifdef MOE_DEBUG_NAN
constant constexpr uint NAN_STAGE_NONE   = 255;  // No NaN detected
constant constexpr uint NAN_STAGE_GATE   = 0;    // NaN in gate accumulator
constant constexpr uint NAN_STAGE_UP     = 1;    // NaN in up accumulator
constant constexpr uint NAN_STAGE_SWIGLU = 2;    // NaN after SiLU(gate)*up
constant constexpr uint NAN_STAGE_DOWN   = 3;    // NaN in down accumulator
#endif

// ---------------------------------------------------------------------------
// Parameter structure
// ---------------------------------------------------------------------------

struct TrellisParams {
    uint M;              // Batch size (tokens)
    uint K;              // Input hidden dimension
    uint N;              // Output dimension (intermediate for gate/up, hidden for down)
    uint num_experts;    // Total experts
    uint top_k;          // Experts per token
    uint bits;           // 2, 3, or 4
    uint group_size;     // Quantization group size
    uint n_levels;       // Codebook levels (e.g., 8 for 3-bit)
};

// ---------------------------------------------------------------------------
// 3-bit Trellis dequantization
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
    // Compute tile position
    uint num_tiles_n = (N_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint tile_k = global_k / TRELLIS_TILE;
    uint tile_n = global_n / TRELLIS_TILE;
    uint k_in_tile = global_k % TRELLIS_TILE;
    uint n_in_tile = global_n % TRELLIS_TILE;
    uint tile_idx = tile_k * num_tiles_n + tile_n;

    device const uint8_t* tile_packed = packed_weights + tile_idx * PACKED_BYTES_3BIT;

    // Transposed indexing: idx = n * TILE_DIM + k
    uint idx_in_tile = n_in_tile * TRELLIS_TILE + k_in_tile;

    // Unpack 3-bit index
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

    // Apply scale
    uint n_groups = (K_dim + group_size - 1) / group_size;
    uint group_idx = global_k / group_size;
    half scale = scales[expert_id * N_dim * n_groups + group_idx * N_dim + global_n];
    dequant *= scale;

    // Apply sign flips
    dequant *= su[expert_id * K_dim + global_k];
    dequant *= sv[expert_id * N_dim + global_n];

    return dequant;
}

// ---------------------------------------------------------------------------
// Load Trellis weight tile with dequantization
// ---------------------------------------------------------------------------

inline void load_trellis_tile(
    device const uint8_t* packed_weights,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
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

    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
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

// ---------------------------------------------------------------------------
// Load activation tile (single token)
// ---------------------------------------------------------------------------

inline void load_activation_tile(
    device const half* activations,
    threadgroup half (&A_buf)[MOE_TILE_K],
    uint token_idx,
    uint k_block,
    uint hidden_dim,
    uint thread_idx
) {
    for (uint i = thread_idx; i < MOE_TILE_K; i += MOE_THREADS) {
        uint global_k = k_block + i;
        half val = 0.0h;
        if (global_k < hidden_dim) {
            val = activations[token_idx * hidden_dim + global_k];
        }
        A_buf[i] = val;
    }
}

// ===========================================================================
// Main MoE kernel with SwiGLU activation - Streaming Chunk Approach
//
// For each token, computes:
//   output = sum_{i=0}^{top_k-1} prob[i] * down(silu(gate(x_i)) * up(x_i))
//
// Where x_i is routed to expert_id[i]
//
// Handles intermediate_dim > 64 by processing in chunks:
//   - For each 64-column chunk of intermediate dimension
//   - Compute gate/up projections
//   - Apply SwiGLU
//   - Accumulate partial down projection
// ===========================================================================

kernel void moe_trellis_swiglu(
    device const half* activations       [[buffer(0)]],   // [batch, hidden]
    device const uint8_t* gate_weights   [[buffer(1)]],   // [num_experts, ...] Trellis
    device const half* gate_scales       [[buffer(2)]],   // [num_experts, ...]
    device const uint8_t* up_weights     [[buffer(3)]],   // [num_experts, ...] Trellis
    device const half* up_scales         [[buffer(4)]],   // [num_experts, ...]
    device const uint8_t* down_weights   [[buffer(5)]],   // [num_experts, ...] Trellis
    device const half* down_scales       [[buffer(6)]],   // [num_experts, ...]
    device const half* gate_su           [[buffer(7)]],   // [num_experts, K]
    device const half* gate_sv           [[buffer(8)]],   // [num_experts, N]
    device const half* up_su             [[buffer(9)]],   // [num_experts, K]
    device const half* up_sv             [[buffer(10)]],  // [num_experts, N]
    device const half* down_su           [[buffer(11)]],  // [num_experts, N]
    device const half* down_sv           [[buffer(12)]],  // [num_experts, K]
    device const half* grid              [[buffer(13)]],  // Codebook grid
    device const uint* expert_ids        [[buffer(14)]],  // [batch, top_k]
    device const half* expert_probs      [[buffer(15)]],  // [batch, top_k]
    device half* output                  [[buffer(16)]],  // [batch, hidden]
    constant TrellisParams& p            [[buffer(17)]],
#ifdef MOE_DEBUG_NAN
    device half* debug_gate              [[buffer(18)]],  // [batch, intermediate_dim]
    device half* debug_up                [[buffer(19)]],  // [batch, intermediate_dim]
    device half* debug_swiglu            [[buffer(20)]],  // [batch, intermediate_dim]
    device uint* debug_nan_stage         [[buffer(21)]],  // [batch] stage where NaN first appeared
#endif
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory - sized to fit within 32KB limit
    // A_tile: 16*2 = 32 bytes (single token activation slice)
    // B_gate/up/down: 16*64*2 = 2048 bytes each = 6144 bytes
    // swiglu_result: 64*2 = 128 bytes
    // output_tile: 64*2 = 128 bytes
    // gate_acc/up_acc: 64*2 = 128 bytes each = 256 bytes
    // Total: ~6.7KB
    threadgroup half A_tile[MOE_TILE_K];                    // Activation slice
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N];        // Gate weights
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N];          // Up weights
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N];        // Down weights
    threadgroup half swiglu_result[MOE_TILE_N];             // SwiGLU intermediate (one chunk)
    threadgroup half output_tile[MOE_TILE_N];               // Accumulated output tile
    threadgroup half gate_acc_tg[MOE_TILE_N];               // Gate accumulator (shared)
    threadgroup half up_acc_tg[MOE_TILE_N];                 // Up accumulator (shared)

    // Grid indices
    const uint n_block = tgid.x * MOE_TILE_N;  // Output column block (into hidden_dim)
    const uint token_idx = tgid.y;              // Token index
    const uint slot = tgid.z;                   // Expert slot (0 to top_k-1)

    // Early exit for out-of-bounds
    if (token_idx >= p.M || slot >= p.top_k) {
        return;
    }

    // CRITICAL: Get the expert assigned to THIS SPECIFIC token for this slot
    const uint expert_id = expert_ids[token_idx * p.top_k + slot];
    if (expert_id >= p.num_experts) {
        return;
    }

    // Get probability weight for this expert
    const half prob = expert_probs[token_idx * p.top_k + slot];

    // Dimensions
    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

    // Compute expert weight offsets
    uint num_tiles_k_gate = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_gate = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_k_down = (intermediate_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;
    uint num_tiles_n_down = (hidden_dim + TRELLIS_TILE - 1) / TRELLIS_TILE;

    uint gate_up_expert_size = num_tiles_k_gate * num_tiles_n_gate * PACKED_BYTES_3BIT;
    uint down_expert_size = num_tiles_k_down * num_tiles_n_down * PACKED_BYTES_3BIT;

    // Pointers to this expert's weights
    device const uint8_t* gate_w = gate_weights + expert_id * gate_up_expert_size;
    device const uint8_t* up_w = up_weights + expert_id * gate_up_expert_size;
    device const uint8_t* down_w = down_weights + expert_id * down_expert_size;

    // Initialize output tile to zero (will accumulate partial down projections)
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        output_tile[i] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // STREAMING CHUNK APPROACH
    // Process intermediate_dim in chunks of MOE_TILE_N (64)
    // =========================================================================

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;  // Column offset in intermediate space

        // =================================================================
        // PHASE 1: Gate and Up projections for this chunk
        // =================================================================

        // Initialize accumulators
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            gate_acc_tg[i] = 0.0h;
            up_acc_tg[i] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // K-dimension loop for gate + up projections
        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            // Load activations for this token
            load_activation_tile(activations, A_tile, token_idx, k_block, hidden_dim, thread_idx);

            // Load gate weights for this chunk
            load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                            B_gate, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            // Load up weights for this chunk
            load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                            B_up, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate gate and up: each thread handles a subset of output columns
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                half local_gate = 0.0h;
                half local_up = 0.0h;

                // Dot product over K tile
                for (uint k = 0; k < MOE_TILE_K; ++k) {
                    half act = A_tile[k];
                    local_gate += act * B_gate[k][i];
                    local_up += act * B_up[k][i];
                }

                gate_acc_tg[i] += local_gate;
                up_acc_tg[i] += local_up;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

#ifdef MOE_DEBUG_NAN
        // Check for NaN in gate accumulator and write debug output
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;
            if (global_n < intermediate_dim) {
                half g = gate_acc_tg[i];
                debug_gate[token_idx * intermediate_dim + global_n] = g;
                if (isnan(g) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                    debug_nan_stage[token_idx] = NAN_STAGE_GATE;
                }
            }
        }

        // Check for NaN in up accumulator and write debug output
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;
            if (global_n < intermediate_dim) {
                half u = up_acc_tg[i];
                debug_up[token_idx * intermediate_dim + global_n] = u;
                if (isnan(u) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                    debug_nan_stage[token_idx] = NAN_STAGE_UP;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Early exit if NaN detected in gate or up (preserve problematic values)
        if (debug_nan_stage[token_idx] != NAN_STAGE_NONE) {
            return;
        }
#endif

        // =================================================================
        // PHASE 2: Apply SwiGLU and store to swiglu_result
        // =================================================================

        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;

            if (global_n < intermediate_dim) {
                half g = gate_acc_tg[i];
                half u = up_acc_tg[i];

                // SiLU: x / (1 + exp(-x))
                half silu_g = g / (1.0h + exp(-g));
                swiglu_result[i] = silu_g * u;
            } else {
                swiglu_result[i] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef MOE_DEBUG_NAN
        // Check for NaN after SwiGLU and write debug output
        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;
            if (global_n < intermediate_dim) {
                half s = swiglu_result[i];
                debug_swiglu[token_idx * intermediate_dim + global_n] = s;
                if (isnan(s) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                    debug_nan_stage[token_idx] = NAN_STAGE_SWIGLU;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Early exit if NaN detected in SwiGLU (preserve problematic values)
        if (debug_nan_stage[token_idx] != NAN_STAGE_NONE) {
            return;
        }
#endif

        // =================================================================
        // PHASE 3: Partial down projection for this chunk
        // =================================================================

        // Determine actual chunk size
        uint chunk_end = min(n_chunk_offset + MOE_TILE_N, intermediate_dim);
        uint chunk_size = chunk_end - n_chunk_offset;
        uint num_k_tiles_chunk = (chunk_size + MOE_TILE_K - 1) / MOE_TILE_K;

        for (uint kdt = 0; kdt < num_k_tiles_chunk; ++kdt) {
            uint k_down_local = kdt * MOE_TILE_K;
            uint k_down_global = n_chunk_offset + k_down_local;

            // Load down weights for this K tile
            load_trellis_tile(down_w, down_scales, down_su, down_sv, grid,
                            B_down, k_down_global, n_block, expert_id,
                            intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate: each thread handles a subset of output columns
            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                uint global_out_n = n_block + i;
                if (global_out_n >= hidden_dim) continue;

                half local_down = 0.0h;

                // Dot product over this K tile
                uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                for (uint k = 0; k < k_end; ++k) {
                    half swiglu_k = swiglu_result[k_down_local + k];
                    local_down += swiglu_k * B_down[k][i];
                }

                output_tile[i] += local_down;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }  // End of chunk loop

#ifdef MOE_DEBUG_NAN
    // Check for NaN in down accumulator (output_tile)
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            half d = output_tile[i];
            if (isnan(d) && debug_nan_stage[token_idx] == NAN_STAGE_NONE) {
                debug_nan_stage[token_idx] = NAN_STAGE_DOWN;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Early exit if NaN detected in down (preserve problematic values)
    if (debug_nan_stage[token_idx] != NAN_STAGE_NONE) {
        return;
    }
#endif

    // =========================================================================
    // PHASE 4: Write output with probability weighting using atomic operations
    //
    // CRITICAL FIX: Multiple threadgroups (one per expert slot) write to the
    // same output locations concurrently. Without atomics, there's a race
    // condition where one slot's contribution can be lost.
    //
    // The output buffer must be pre-initialized to zero before kernel launch.
    // We use a CAS loop to atomically add our contribution to the shared output.
    // =========================================================================

    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            half weighted = output_tile[i] * prob;
            uint out_idx = token_idx * hidden_dim + global_n;

            // Atomic add for half precision using CAS on aligned uint32
            // Each uint32 contains two packed half values
            uint aligned_idx = out_idx / 2;
            uint lane = out_idx & 1;  // 0 = low half, 1 = high half

            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[aligned_idx * 2]);

            uint old_val = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_val;
            bool success;
            do {
                // Extract current half value from the appropriate lane
                half current_half;
                if (lane == 0) {
                    current_half = as_type<half>(ushort(old_val & 0xFFFF));
                } else {
                    current_half = as_type<half>(ushort((old_val >> 16) & 0xFFFF));
                }

                // Add our contribution
                half updated_half = current_half + weighted;
                ushort updated_bits = as_type<ushort>(updated_half);

                // Pack back into uint32
                if (lane == 0) {
                    new_val = (old_val & 0xFFFF0000) | uint(updated_bits);
                } else {
                    new_val = (old_val & 0x0000FFFF) | (uint(updated_bits) << 16);
                }

                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_val, new_val,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}

// ===========================================================================
// FP32 Accumulation Variant
//
// Uses FP32 accumulators for numerical stability with large K dimensions.
// Same algorithm as moe_trellis_swiglu but with FP32 intermediate values.
// ===========================================================================

kernel void moe_trellis_swiglu_fp32acc(
    device const half* activations       [[buffer(0)]],
    device const uint8_t* gate_weights   [[buffer(1)]],
    device const half* gate_scales       [[buffer(2)]],
    device const uint8_t* up_weights     [[buffer(3)]],
    device const half* up_scales         [[buffer(4)]],
    device const uint8_t* down_weights   [[buffer(5)]],
    device const half* down_scales       [[buffer(6)]],
    device const half* gate_su           [[buffer(7)]],
    device const half* gate_sv           [[buffer(8)]],
    device const half* up_su             [[buffer(9)]],
    device const half* up_sv             [[buffer(10)]],
    device const half* down_su           [[buffer(11)]],
    device const half* down_sv           [[buffer(12)]],
    device const half* grid              [[buffer(13)]],
    device const uint* expert_ids        [[buffer(14)]],
    device const half* expert_probs      [[buffer(15)]],
    device half* output                  [[buffer(16)]],
    constant TrellisParams& p            [[buffer(17)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory with FP32 accumulators
    threadgroup half A_tile[MOE_TILE_K];
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N];
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N];
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N];
    threadgroup float swiglu_result[MOE_TILE_N];             // FP32
    threadgroup float output_tile[MOE_TILE_N];               // FP32
    threadgroup float gate_acc_tg[MOE_TILE_N];               // FP32
    threadgroup float up_acc_tg[MOE_TILE_N];                 // FP32

    const uint n_block = tgid.x * MOE_TILE_N;
    const uint token_idx = tgid.y;
    const uint slot = tgid.z;

    if (token_idx >= p.M || slot >= p.top_k) {
        return;
    }

    const uint expert_id = expert_ids[token_idx * p.top_k + slot];
    if (expert_id >= p.num_experts) {
        return;
    }

    const float prob = float(expert_probs[token_idx * p.top_k + slot]);

    const uint hidden_dim = p.K;
    const uint intermediate_dim = p.N;

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
        output_tile[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) / MOE_TILE_N;
    uint num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;

    for (uint chunk_idx = 0; chunk_idx < num_intermediate_chunks; ++chunk_idx) {
        uint n_chunk_offset = chunk_idx * MOE_TILE_N;

        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            gate_acc_tg[i] = 0.0f;
            up_acc_tg[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles_hidden; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            load_activation_tile(activations, A_tile, token_idx, k_block, hidden_dim, thread_idx);

            load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                            B_gate, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                            B_up, k_block, n_chunk_offset, expert_id,
                            hidden_dim, intermediate_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                float local_gate = 0.0f;
                float local_up = 0.0f;

                for (uint k = 0; k < MOE_TILE_K; ++k) {
                    float act = float(A_tile[k]);
                    local_gate += act * float(B_gate[k][i]);
                    local_up += act * float(B_up[k][i]);
                }

                gate_acc_tg[i] += local_gate;
                up_acc_tg[i] += local_up;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
            uint global_n = n_chunk_offset + i;

            if (global_n < intermediate_dim) {
                float g = gate_acc_tg[i];
                float u = up_acc_tg[i];
                float silu_g = g / (1.0f + exp(-g));
                swiglu_result[i] = silu_g * u;
            } else {
                swiglu_result[i] = 0.0f;
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
                            intermediate_dim, hidden_dim, p.group_size, p.n_levels, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
                uint global_out_n = n_block + i;
                if (global_out_n >= hidden_dim) continue;

                float local_down = 0.0f;

                uint k_end = min(MOE_TILE_K, chunk_size - k_down_local);
                for (uint k = 0; k < k_end; ++k) {
                    float swiglu_k = swiglu_result[k_down_local + k];
                    local_down += swiglu_k * float(B_down[k][i]);
                }

                output_tile[i] += local_down;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Atomic write to output (same fix as FP16 variant for race condition)
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        uint global_n = n_block + i;
        if (global_n < hidden_dim) {
            half weighted = half(output_tile[i] * prob);
            uint out_idx = token_idx * hidden_dim + global_n;

            // Atomic add for half precision using CAS on aligned uint32
            uint aligned_idx = out_idx / 2;
            uint lane = out_idx & 1;

            device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[aligned_idx * 2]);

            uint old_val = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
            uint new_val;
            bool success;
            do {
                half current_half;
                if (lane == 0) {
                    current_half = as_type<half>(ushort(old_val & 0xFFFF));
                } else {
                    current_half = as_type<half>(ushort((old_val >> 16) & 0xFFFF));
                }

                half updated_half = current_half + weighted;
                ushort updated_bits = as_type<ushort>(updated_half);

                if (lane == 0) {
                    new_val = (old_val & 0xFFFF0000) | uint(updated_bits);
                } else {
                    new_val = (old_val & 0x0000FFFF) | (uint(updated_bits) << 16);
                }

                success = atomic_compare_exchange_weak_explicit(
                    atomic_ptr, &old_val, new_val,
                    memory_order_relaxed, memory_order_relaxed);
            } while (!success);
        }
    }
}
