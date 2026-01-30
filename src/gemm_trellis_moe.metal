// gemm_trellis_moe.metal - Fused MoE GEMM with Trellis 3bpw quantization
//
// Single kernel that handles:
//   1. Token routing to top-k experts
//   2. Trellis dequantization (3-bit EXL3) on-the-fly
//   3. SwiGLU activation (gate_proj, up_proj, down_proj)
//   4. Expert probability weighting
//
// Design: Process tokens in batches of 64, with each expert's weights
// stored contiguously for efficient indexing.
//
// Memory layout:
//   activations:     [batch, hidden] half
//   gate_weights:    [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   gate_scales:     [num_experts, n_groups, n] half
//   up_weights:      [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   up_scales:       [num_experts, n_groups, n] half
//   down_weights:     [num_experts, packed_k, packed_n, packed_bytes] uint8 (Trellis)
//   down_scales:      [num_experts, n_groups, n] half
//   expert_ids:      [batch, top_k] uint32
//   expert_probs:    [batch, top_k] half
//   su/sv:          [num_experts, ...] half (sign flips)
//   grids:          [bits_to_level] half (codebook lookup)
//   output:          [batch, hidden] half

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions
// ---------------------------------------------------------------------------

constant constexpr uint MOE_TILE_M = 64;   // Batch tile (tokens)
constant constexpr uint MOE_TILE_N = 64;   // Output dimension tile
constant constexpr uint MOE_TILE_K = 32;   // Hidden dimension tile

constant constexpr uint MOE_SIMDGROUPS = 4;
constant constexpr uint MOE_THREADS = MOE_SIMDGROUPS * 32;  // 128

constant constexpr uint MOE_SG_M_TILES = 8;  // rows of 8x8 tiles per simdgroup
constant constexpr uint MOE_SG_N_TILES = 2;  // cols of 8x8 tiles per simdgroup

constant constexpr uint TRELLIS_TILE = 16;
constant constexpr uint PACKED_BYTES_3BIT = 96;  // 16*16*3/8

// ---------------------------------------------------------------------------
// 3-bit Trellis dequantization (from dequant_trellis.metal)
// ---------------------------------------------------------------------------

struct TrellisParams {
    uint M;              // Batch size (tokens)
    uint K;              // Input hidden dimension
    uint N;              // Output dimension (intermediate for gate/up, hidden for down)
    uint num_experts;     // Total experts
    uint top_k;           // Experts per token
    uint bits;            // 2, 3, or 4
    uint group_size;       // Quantization group size
    uint n_levels;        // Codebook levels (e.g., 729 for 3-bit)
};

// Fast 3-bit Trellis dequantization using precomputed codebook grid
inline half trellis_dequant_3bit(
    device const uint8_t* packed,
    uint tile_idx,
    uint k_local,
    uint n_local,
    device const half* scales,
    device const half* su,
    device const half* sv,
    device const half* grid,
    const TrellisParams& p,
    uint expert_id
) {
    // Unpack 3 bits from packed bytes
    uint byte_idx = (tile_idx * PACKED_BYTES_3BIT) + (k_local * TRELLIS_TILE + n_local) * 3 / 8;
    uint bit_offset = ((k_local * TRELLIS_TILE + n_local) * 3) % 8;

    uint packed_val = packed[byte_idx];
    uint codebook_idx = (packed_val >> bit_offset) & 0x7;  // 3 bits

    if (codebook_idx >= p.n_levels) {
        codebook_idx = 0;
    }

    half dequant = grid[codebook_idx];

    // Apply scale (per-group along K dimension)
    uint group_idx = (k_local * TRELLIS_TILE) / p.group_size;
    half scale = scales[expert_id * p.N * ((p.K + p.group_size - 1) / p.group_size) +
                        group_idx * p.N + n_local];
    dequant *= scale;

    // Apply sign flips
    dequant *= su[expert_id * p.K + k_local * TRELLIS_TILE];
    dequant *= sv[expert_id * p.N + n_local * TRELLIS_TILE];

    return dequant;
}

// Load Trellis weight tile with dequantization
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
    const TrellisParams& p,
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
        if (global_k < p.K && global_n < p.N) {
            // Compute tile position in packed format
            uint tile_k = global_k / TRELLIS_TILE;
            uint tile_n = global_n / TRELLIS_TILE;
            uint k_in_tile = global_k % TRELLIS_TILE;
            uint n_in_tile = global_n % TRELLIS_TILE;
            uint num_tiles_n = (p.N + TRELLIS_TILE - 1) / TRELLIS_TILE;
            uint tile_idx = tile_k * num_tiles_n + tile_n;

            val = trellis_dequant_3bit(
                packed_weights, tile_idx, k_in_tile, n_in_tile,
                scales, su, sv, grid, p, expert_id
            );
        }
        B_buf[k_local][n_local] = val;
    }
}

// Load activation tile
inline void load_activation_tile(
    device const half* activations,
    threadgroup half (&A_buf)[MOE_TILE_M][MOE_TILE_K],
    uint m_block,
    uint k_block,
    const TrellisParams& p,
    uint thread_idx
) {
    const uint elems_per_thread = (MOE_TILE_M * MOE_TILE_K) / MOE_THREADS;

    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint m_local = flat_idx / MOE_TILE_K;
        uint k_local = flat_idx % MOE_TILE_K;

        uint global_m = m_block + m_local;
        uint global_k = k_block + k_local;

        half val = 0.0h;
        if (global_m < p.M && global_k < p.K) {
            val = activations[global_m * p.K + global_k];
        }
        A_buf[m_local][k_local] = val;
    }
}

// Simdgroup matrix multiply
inline void sgmm(
    threadgroup const half (&A)[MOE_TILE_M][MOE_TILE_K],
    threadgroup const half (&B)[MOE_TILE_K][MOE_TILE_N],
    thread simdgroup_matrix<half, 8, 8> (&acc)[MOE_SG_M_TILES][MOE_SG_N_TILES],
    uint sg_row_offset,
    uint sg_col_offset,
    uint k_tiles
) {
    for (uint kt = 0; kt < k_tiles; ++kt) {
        for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag,
                          &A[sg_row_offset + mi * 8][kt * 8],
                          MOE_TILE_K);

            for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag,
                              &B[kt * 8][sg_col_offset + ni * 8],
                              MOE_TILE_N);

                simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
            }
        }
    }
}

// Store with probability weighting and scatter
inline void store_weighted_scatter(
    thread simdgroup_matrix<half, 8, 8> (&acc)[MOE_SG_M_TILES][MOE_SG_N_TILES],
    device half* output,
    device const uint* expert_ids,
    device const half* expert_probs,
    uint expert_slot,
    const TrellisParams& p,
    uint m_block,
    uint n_block,
    uint sg_row_offset,
    uint sg_col_offset,
    uint simd_lane,
    uint simd_id,
    threadgroup half (&staging)[MOE_SIMDGROUPS][MOE_SG_M_TILES * 8][MOE_SG_N_TILES * 8]
) {
    constexpr uint sg_tile_rows = MOE_SG_M_TILES * 8;
    constexpr uint sg_tile_cols = MOE_SG_N_TILES * 8;

    // Stage results
    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            simdgroup_store(acc[mi][ni],
                          &staging[simd_id][mi * 8][ni * 8],
                          sg_tile_cols);
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Write with probability weighting
    constexpr uint total_elems = sg_tile_rows * sg_tile_cols;
    constexpr uint elems_per_lane = total_elems / 32;

    for (uint iter = 0; iter < elems_per_lane; ++iter) {
        uint elem = simd_lane * elems_per_lane + iter;
        uint local_row = elem / sg_tile_cols;
        uint local_col = elem % sg_tile_cols;

        uint global_m = m_block + sg_row_offset + local_row;
        uint global_n = n_block + sg_col_offset + local_col;

        if (global_m < p.M && global_n < p.N) {
            // Check if this token uses this expert for this slot
            uint token_expert = expert_ids[global_m * p.top_k + expert_slot];

            if (token_expert < p.num_experts) {
                half prob = expert_probs[global_m * p.top_k + expert_slot];
                half val = staging[simd_id][local_row][local_col] * prob;

                // Use atomic for accumulation (simpler approach)
                // For production, use separate buffers per expert then combine
                device atomic<half>* out_ptr = (device atomic<half>*)&output[global_m * p.N + global_n];
                atomic_fetch_add_explicit(out_ptr, val, memory_order_relaxed);
            }
        }
    }
}

// ===========================================================================
// Main MoE kernel with SwiGLU activation
//
// For each token, computes:
//   output = sum_{i=0}^{top_k-1} prob[i] * down(silu(gate(x_i)) * up(x_i))
//
// Where x_i is routed to expert_id[i]
// ===========================================================================

kernel void moe_trellis_swiglu(
    device const half* activations         [[buffer(0)]],  // [batch, hidden]
    device const uint8_t* gate_weights   [[buffer(1)]],  // [num_experts, ...] Trellis
    device const half* gate_scales       [[buffer(2)]],  // [num_experts, ...]
    device const uint8_t* up_weights     [[buffer(3)]],  // [num_experts, ...] Trellis
    device const half* up_scales         [[buffer(4)]],  // [num_experts, ...]
    device const uint8_t* down_weights   [[buffer(5)]],  // [num_experts, ...] Trellis
    device const half* down_scales       [[buffer(6)]],  // [num_experts, ...]
    device const half* gate_su           [[buffer(7)]],  // [num_experts, K]
    device const half* gate_sv           [[buffer(8)]],  // [num_experts, N]
    device const half* up_su             [[buffer(9)]],  // [num_experts, K]
    device const half* up_sv             [[buffer(10)]], // [num_experts, N]
    device const half* down_su           [[buffer(11)]], // [num_experts, N]
    device const half* down_sv           [[buffer(12)]], // [num_experts, K]
    device const half* grid             [[buffer(13)]], // Codebook grid
    device const uint* expert_ids       [[buffer(14)]], // [batch, top_k]
    device const half* expert_probs     [[buffer(15)]], // [batch, top_k]
    device half* output                 [[buffer(16)]], // [batch, hidden]
    constant TrellisParams& p          [[buffer(17)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory
    threadgroup half A_tiles[2][MOE_TILE_M][MOE_TILE_K];      // Activations
    threadgroup half B_gate[MOE_TILE_K][MOE_TILE_N];          // Gate weights
    threadgroup half B_up[MOE_TILE_K][MOE_TILE_N];            // Up weights
    threadgroup half B_down[MOE_TILE_K][MOE_TILE_N];          // Down weights
    threadgroup half gate_up staging[MOE_SIMDGROUPS][MOE_SG_M_TILES * 8][MOE_SG_N_TILES * 8];
    threadgroup half down_staging[MOE_SIMDGROUPS][MOE_SG_M_TILES * 8][MOE_SG_N_TILES * 8];

    const uint tg_m = tgid.y * MOE_TILE_M;
    const uint tg_n = tgid.x * MOE_TILE_N;
    const uint sg_row_offset = simd_id * (MOE_SG_M_TILES * 8);
    const uint sg_col_offset = 0;  // All sg cover columns
    const uint thread_idx = simd_id * 32 + simd_lane;

    // Initialize output to zero
    for (uint i = thread_idx; i < MOE_TILE_M * MOE_TILE_N; i += MOE_THREADS) {
        uint m = i / MOE_TILE_N;
        uint n = i % MOE_TILE_N;
        uint global_m = tg_m + m;
        uint global_n = tg_n + n;
        if (global_m < p.M && global_n < p.N) {
            output[global_m * p.N + global_n] = 0.0h;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each expert slot
    for (uint slot = 0; slot < p.top_k; ++slot) {
        // Use first token's expert for this tile (simplification)
        uint representative_expert = expert_ids[tg_m * p.top_k + slot];
        if (representative_expert >= p.num_experts) {
            continue;
        }

        // Compute strides
        uint num_tiles_k = (p.K + MOE_TILE_K - 1) / MOE_TILE_K;
        uint num_tiles_n = (p.N + MOE_TILE_N - 1) / MOE_TILE_N;
        uint num_tiles_n_inter = (p.N + MOE_TILE_N - 1) / MOE_TILE_N;

        // Pointers to this expert's weights
        device const uint8_t* gate_w = gate_weights + representative_expert * num_tiles_k * num_tiles_n_inter * PACKED_BYTES_3BIT;
        device const uint8_t* up_w = up_weights + representative_expert * num_tiles_k * num_tiles_n_inter * PACKED_BYTES_3BIT;
        device const uint8_t* down_w = down_weights + representative_expert * num_tiles_n_inter * num_tiles_k * PACKED_BYTES_3BIT;

        // K-dimension loop for gate + up projections
        for (uint kt = 0; kt < num_tiles_k; ++kt) {
            uint k_block = kt * MOE_TILE_K;

            // Load activations
            load_activation_tile(activations, A_tiles[0], tg_m, k_block, p, thread_idx);

            // Load gate weights (dequantized)
            load_trellis_tile(gate_w, gate_scales, gate_su, gate_sv, grid,
                            B_gate, k_block, 0, representative_expert, p, thread_idx);

            // Load up weights (dequantized)
            load_trellis_tile(up_w, up_scales, up_su, up_sv, grid,
                            B_up, k_block, 0, representative_expert, p, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute gate and up GEMMs
            simdgroup_matrix<half, 8, 8> acc_gate[MOE_SG_M_TILES][MOE_SG_N_TILES];
            simdgroup_matrix<half, 8, 8> acc_up[MOE_SG_M_TILES][MOE_SG_N_TILES];

            for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                    acc_gate[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                    acc_up[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            sgmm(A_tiles[0], B_gate, acc_gate, sg_row_offset, sg_col_offset, 1);
            sgmm(A_tiles[0], B_up, acc_up, sg_row_offset, sg_col_offset, 1);

            // Apply SwiGLU: silu(gate) * up
            for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                    simdgroup_matrix<half, 8, 8> silu_result;
                    for (uint i = 0; i < 8; ++i) {
                        for (uint j = 0; j < 8; ++j) {
                            half g = acc_gate[mi][ni][i * 8 + j];
                            half u = acc_up[mi][ni][i * 8 + j];
                            silu_result[i * 8 + j] = g / (1.0h + fabs(g)) * u;
                        }
                    }
                    acc_gate[mi][ni] = silu_result;
                }
            }

            // Store intermediate to staging (simplified - should be double-buffered)
            for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                    simdgroup_store(acc_gate[mi][ni],
                                  &staging[simd_id][mi * 8][ni * 8],
                                  MOE_SG_N_TILES * 8);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Now project down: down_proj(silu(gate) * up)
            // Load down weights for this N tile
            load_trellis_tile(down_w, down_scales, down_su, down_sv, grid,
                            B_down, 0, tg_n, representative_expert, p, thread_idx);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute down GEMM
            simdgroup_matrix<half, 8, 8> acc_down[MOE_SG_M_TILES][MOE_SG_N_TILES];
            for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
                for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                    acc_down[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            // A from staging (intermediate), B is down weights
            for (uint kst = 0; kst < MOE_TILE_N / 8; ++kst) {
                for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
                    simdgroup_matrix<half, 8, 8> a_frag;
                    simdgroup_load(a_frag,
                                  &staging[simd_id][mi * 8][kst * 8],
                                  MOE_SG_N_TILES * 8);

                    for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                        simdgroup_matrix<half, 8, 8> b_frag;
                        simdgroup_load(b_frag,
                                      &B_down[kst * 8][sg_col_offset + ni * 8],
                                      MOE_TILE_N);

                        simdgroup_multiply_accumulate(acc_down[mi][ni], a_frag, b_frag, acc_down[mi][ni]);
                    }
                }
            }

            // Store with probability weighting
            store_weighted_scatter(acc_down, output, expert_ids, expert_probs, slot, p,
                                 tg_m, tg_n, sg_row_offset, sg_col_offset,
                                 simd_lane, simd_id, down_staging);
        }
    }
}
