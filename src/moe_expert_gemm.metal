// moe_expert_gemm.metal - Fused MoE expert execution with FP4 quantized weights
//
// Optimized for Apple Unified Memory: all experts are always resident in VRAM,
// so no copying needed - just index into the contiguous expert weights buffer.
//
// Design insight from vLLM's fused_moe: group tokens by expert to maximize
// cache reuse, then dispatch expert GEMMs with weighted accumulation.
//
// Kernel variants:
//   1. moe_expert_gemm_fp4           - Per-expert dispatch, fused prob weighting
//   2. moe_expert_gemm_fp4_grouped   - Token-to-expert grouping + batched dispatch
//   3. moe_expert_gemm_shared        - Shared expert path (always executed)
//
// Key optimizations:
//   - Expert weights stored contiguously [num_experts, out_features, in_features/8]
//   - Per-token expert assignments drive which weights to index
//   - Expert probabilities fused into output accumulation (one write per token)
//   - Shared expert path can be overlapped with routed experts
//
// Memory layout:
//   expert_weights: [num_experts, out/8, in] packed FP4 (8 values per uint32)
//   scales:         [num_experts, out/group_size, in] per-group FP16 scales
//   expert_ids:     [batch, top_k] uint32 expert indices per token
//   expert_probs:   [batch, top_k] half routing probabilities per token

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions - optimized for M4 Max
// ---------------------------------------------------------------------------

constant constexpr uint MOE_TILE_M = 64;   // Batch tile (tokens)
constant constexpr uint MOE_TILE_N = 64;   // Output dimension tile
constant constexpr uint MOE_TILE_K = 32;   // Hidden dimension tile

constant constexpr uint MOE_K_TILES = MOE_TILE_K / 8;  // 4 sub-tiles for simdgroup MMA
constant constexpr uint MOE_SIMDGROUPS = 4;
constant constexpr uint MOE_THREADS = MOE_SIMDGROUPS * 32;  // 128

constant constexpr uint MOE_SG_M_TILES = 8;  // rows of 8x8 tiles per simdgroup
constant constexpr uint MOE_SG_N_TILES = 2;  // cols of 8x8 tiles per simdgroup

constant constexpr uint FP4_PER_UINT = 8;
constant constexpr uint MOE_NUM_BUFFERS = 2;

// ---------------------------------------------------------------------------
// 2:4 Sparsity constants (NVIDIA format)
// Every group of 4 dense K elements has exactly 2 non-zero values.
// ---------------------------------------------------------------------------
constant constexpr uint SPARSE_GROUP = 4;       // Dense elements per sparsity group
constant constexpr uint SPARSE_NNZ = 2;         // Non-zeros per group
constant constexpr uint SPARSE_RATIO = 2;       // Compression ratio (K_dense / K_sparse)

// Metadata encoding: 4 bits per sparsity group (2x 2-bit indices)
constant constexpr uint META_BITS_PER_GROUP = 4;
constant constexpr uint META_GROUPS_PER_UINT = 8;  // 32 bits / 4 bits
constant constexpr uint META_DENSE_K_PER_UINT = META_GROUPS_PER_UINT * SPARSE_GROUP;  // 32

// ---------------------------------------------------------------------------
// MoE parameters struct
// ---------------------------------------------------------------------------

struct MoEParams {
    uint batch_size;      // Number of tokens
    uint hidden_dim;      // Input hidden dimension (K for GEMM)
    uint out_dim;         // Output dimension (N for GEMM)
    uint num_experts;     // Total number of experts
    uint top_k;           // Number of experts per token
    uint group_size;      // Quantization group size for scales
    uint has_shared;      // Whether shared expert exists (1 = yes)
    uint shared_expert_id; // Index of shared expert (if has_shared)
    uint use_sparse;      // Whether to use 2:4 sparsity (1 = yes)
};

// ---------------------------------------------------------------------------
// FP4 E2M1 dequantization - same as marlin_gemm.metal
// ---------------------------------------------------------------------------

inline half moe_guard_finite(half val) {
    return select(val, half(0.0h), !isfinite(val));
}

inline half moe_dequant_fp4_bitwise(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.5h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }
    return moe_guard_finite(sign_bit ? -magnitude : magnitude);
}

inline half moe_dequant_fp4(uint nibble, half scale) {
    half raw = moe_dequant_fp4_bitwise(nibble);
    float result = (float)raw * (float)scale;
    return isfinite(result) ? (half)result : half(0.0h);
}

inline void moe_unpack_fp4x8(uint packed, half scale, thread half* out) {
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = moe_dequant_fp4(nibble, scale);
    }
}

// ---------------------------------------------------------------------------
// Expert weight indexing
//
// Memory layout: expert_weights[expert_id][k_packed][n]
// where k_packed = hidden_dim / 8 (8 FP4 values per uint32)
//
// Expert stride = (hidden_dim/8) * out_dim
// ---------------------------------------------------------------------------

inline uint expert_weight_offset(uint expert_id, uint hidden_dim, uint out_dim) {
    uint k_packed = hidden_dim / FP4_PER_UINT;
    return expert_id * k_packed * out_dim;
}

inline uint expert_scale_offset(uint expert_id, uint hidden_dim, uint out_dim, uint group_size) {
    uint num_groups = (hidden_dim + group_size - 1) / group_size;
    return expert_id * num_groups * out_dim;
}

// ---------------------------------------------------------------------------
// Cooperative tile loaders
// ---------------------------------------------------------------------------

// Load activation tile from [batch, hidden] into A_buf[TILE_M][TILE_K]
inline void moe_load_A_tile(
    device const half* activations,
    threadgroup half (&A_buf)[MOE_TILE_M][MOE_TILE_K],
    uint batch_size, uint hidden_dim,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (MOE_TILE_M * MOE_TILE_K) / MOE_THREADS;  // 16
    constexpr uint VECTOR_SIZE = 4;
    constexpr uint vec_per_thread = elems_per_thread / VECTOR_SIZE;  // 4

    for (uint i = 0; i < vec_per_thread; ++i) {
        uint vec_flat_idx = thread_idx * vec_per_thread + i;
        uint row = vec_flat_idx / (MOE_TILE_K / VECTOR_SIZE);
        uint vec_col = vec_flat_idx % (MOE_TILE_K / VECTOR_SIZE);
        uint col_base = vec_col * VECTOR_SIZE;
        uint global_row = tg_row + row;
        uint global_col_base = k_block + col_base;

        if (row < MOE_TILE_M) {
            if (global_row < batch_size && global_col_base + VECTOR_SIZE <= hidden_dim) {
                half4 vals = *((device const half4*)(activations + global_row * hidden_dim + global_col_base));
                A_buf[row][col_base + 0] = vals.x;
                A_buf[row][col_base + 1] = vals.y;
                A_buf[row][col_base + 2] = vals.z;
                A_buf[row][col_base + 3] = vals.w;
            } else {
                for (uint v = 0; v < VECTOR_SIZE; ++v) {
                    half val = 0.0h;
                    if (global_row < batch_size) {
                        uint global_col = global_col_base + v;
                        if (global_col < hidden_dim) {
                            val = activations[global_row * hidden_dim + global_col];
                        }
                    }
                    A_buf[row][col_base + v] = val;
                }
            }
        }
    }
}

// Load expert weight tile with dequantization
// B is [K/8, N] packed FP4 for ONE expert (offset already applied)
// ---------------------------------------------------------------------------
// Sparse B tile loader with metadata-driven scatter for 2:4 sparsity
//
// Reconstructs dense B tile from compressed sparse representation:
//   - B_sparse[K_sparse/8, N]: packed FP4, only K/2 values stored
//   - B_meta[K/32, N]: 2-bit metadata indices per 4-group
//   - scales[K/group_size, N]: per-group scales
// ---------------------------------------------------------------------------
inline void moe_load_B_tile_sparse_dequant(
    device const uint* B_sparse,     // [K_sparse/8, N] compressed FP4
    device const uint* B_meta,       // [K/32, N] metadata
    device const half* scales,       // [K/group_size, N]
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    uint hidden_dim,                  // Dense K dimension
    uint out_dim,
    uint tg_col,                     // Starting column of this tile
    uint k_block,                    // Starting dense K position of this tile
    uint group_size,
    uint thread_idx
) {
    // Zero the entire B tile first
    const uint elems_per_thread = (MOE_TILE_K * MOE_TILE_N) / MOE_THREADS;  // 16
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / MOE_TILE_N;
        uint col = flat_idx % MOE_TILE_N;
        B_buf[row][col] = half(0.0h);
    }

    const uint groups_in_tile = MOE_TILE_K / SPARSE_GROUP;  // 8
    const uint total_work = groups_in_tile * MOE_TILE_N;     // 512
    const uint work_per_thread = total_work / MOE_THREADS;  // 4
    const uint hidden_dim_sparse = hidden_dim / SPARSE_RATIO;

    for (uint w = 0; w < work_per_thread; ++w) {
        uint work_idx = thread_idx * work_per_thread + w;
        uint n_local = work_idx / groups_in_tile;   // Column within tile [0..63]
        uint g_local = work_idx % groups_in_tile;   // Sparsity group within tile [0..7]

        uint global_n = tg_col + n_local;
        if (global_n >= out_dim) continue;

        uint dense_k_base = k_block + g_local * SPARSE_GROUP;
        if (dense_k_base >= hidden_dim) continue;

        // Read metadata: B_meta[K/32, N]
        uint meta_row = dense_k_base / META_DENSE_K_PER_UINT;
        uint group_in_meta_word = (dense_k_base % META_DENSE_K_PER_UINT) / SPARSE_GROUP;
        uint meta_word = B_meta[meta_row * out_dim + global_n];
        uint meta_bits = (meta_word >> (group_in_meta_word * META_BITS_PER_GROUP)) & 0xF;

        uint idx0 = meta_bits & 0x3;         // First non-zero position [0..3]
        uint idx1 = (meta_bits >> 2) & 0x3;  // Second non-zero position [0..3]

        // Read compressed FP4 values: B_sparse[K_sparse/8, N]
        uint sparse_k_base = (dense_k_base / SPARSE_GROUP) * SPARSE_NNZ;
        uint scale_group = dense_k_base / group_size;
        half s = scales[scale_group * out_dim + global_n];

        uint pack_idx = sparse_k_base / FP4_PER_UINT;
        uint nibble_offset = sparse_k_base % FP4_PER_UINT;

        uint packed = 0;
        if (pack_idx < (hidden_dim_sparse / FP4_PER_UINT)) {
            packed = B_sparse[pack_idx * out_dim + global_n];
        }

        uint nibble0 = (packed >> (nibble_offset * 4)) & 0xF;
        uint nibble1 = (packed >> ((nibble_offset + 1) * 4)) & 0xF;

        half val0 = moe_dequant_fp4(nibble0, s);
        half val1 = moe_dequant_fp4(nibble1, s);

        uint tile_k0 = g_local * SPARSE_GROUP + idx0;
        uint tile_k1 = g_local * SPARSE_GROUP + idx1;

        if (tile_k0 < MOE_TILE_K) {
            B_buf[tile_k0][n_local] = val0;
        }
        if (tile_k1 < MOE_TILE_K) {
            B_buf[tile_k1][n_local] = val1;
        }
    }
}

// Load expert weight tile with dequantization
// B is [K/8, N] packed FP4 for ONE expert (offset already applied)
inline void moe_load_B_tile_dequant(
    device const uint* expert_B,
    device const half* expert_scales,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    uint hidden_dim, uint out_dim,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint k_packs = (hidden_dim + FP4_PER_UINT - 1) / FP4_PER_UINT;
    const uint num_groups = (hidden_dim + group_size - 1) / group_size;
    const uint packed_per_thread = (MOE_TILE_K * MOE_TILE_N) / (MOE_THREADS * FP4_PER_UINT);  // 2

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (MOE_TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (MOE_TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        // Read scale for this group
        uint scale_k = global_k_base / group_size;
        half s = 1.0h;
        if (global_n < out_dim && global_k_base < hidden_dim && scale_k < num_groups) {
            s = expert_scales[scale_k * out_dim + global_n];
        }

        // Read packed FP4 weights
        uint packed = 0;
        uint b_row = global_k_base / FP4_PER_UINT;
        if (global_n < out_dim && b_row < k_packs && global_k_base < hidden_dim) {
            packed = expert_B[b_row * out_dim + global_n];
        }

        // Dequantize and store to tile
        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        half vals[8];
        moe_unpack_fp4x8(packed, s, vals);
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < MOE_TILE_K; ++v) {
            if (n_idx < MOE_TILE_N) {
                uint global_k = global_k_base + v;
                B_buf[tile_k_base + v][n_idx] = (global_k < hidden_dim) ? vals[v] : 0.0h;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute and store
// ---------------------------------------------------------------------------

inline void moe_compute_from_tiles(
    threadgroup const half (&A_buf)[MOE_TILE_M][MOE_TILE_K],
    threadgroup const half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    thread simdgroup_matrix<half, 8, 8> acc[MOE_SG_M_TILES][MOE_SG_N_TILES],
    uint sg_row_offset,
    uint sg_col_offset
) {
    for (uint kt = 0; kt < MOE_K_TILES; ++kt) {
        for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &A_buf[sg_row_offset + mi * 8][kt * 8],
                           MOE_TILE_K);

            for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &B_buf[kt * 8][sg_col_offset + ni * 8],
                               MOE_TILE_N);

                simdgroup_multiply_accumulate(acc[mi][ni],
                                              a_frag, b_frag, acc[mi][ni]);
            }
        }
    }
}

// Store results with expert probability weighting (fused multiply-add)
// Note: staging must be passed from the kernel since threadgroup vars cannot be
// declared in non-kernel functions.
inline void moe_store_results_weighted(
    thread simdgroup_matrix<half, 8, 8> acc[MOE_SG_M_TILES][MOE_SG_N_TILES],
    device half* output,
    device const half* expert_probs,  // [batch, top_k]
    threadgroup half (&staging)[MOE_SIMDGROUPS][MOE_SG_M_TILES * 8][MOE_SG_N_TILES * 8],
    uint batch_size, uint out_dim, uint top_k,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint expert_slot,  // Which slot in top_k (0, 1, ..., top_k-1)
    bool is_first_expert,  // First expert for this token writes, others accumulate
    uint simd_lane,
    uint simd_id
) {
    constexpr uint sg_tile_rows = MOE_SG_M_TILES * 8;  // 16
    constexpr uint sg_tile_cols = MOE_SG_N_TILES * 8;  // 32

    uint base_row = tg_row + sg_row_offset;
    uint base_col = tg_col + sg_col_offset;

    // Store all acc tiles to staging (passed from kernel)
    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            simdgroup_store(acc[mi][ni],
                            &staging[simd_id][mi * 8][ni * 8],
                            sg_tile_cols);
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane writes a portion of the staging buffer with probability weighting
    constexpr uint total_elems = sg_tile_rows * sg_tile_cols;  // 512
    constexpr uint elems_per_lane = total_elems / 32;  // 16

    for (uint iter = 0; iter < elems_per_lane; ++iter) {
        uint elem = simd_lane * elems_per_lane + iter;
        uint row = elem / sg_tile_cols;
        uint col = elem % sg_tile_cols;

        uint out_row = base_row + row;
        uint out_col = base_col + col;

        if (out_row < batch_size && out_col < out_dim) {
            // Get expert probability for this token
            half prob = expert_probs[out_row * top_k + expert_slot];
            half val = staging[simd_id][row][col] * prob;

            // First expert writes, subsequent experts accumulate
            if (is_first_expert) {
                output[out_row * out_dim + out_col] = val;
            } else {
                output[out_row * out_dim + out_col] += val;
            }
        }
    }
}

// ===========================================================================
// Kernel 1: Per-token MoE expert GEMM with fused probability weighting
//
// Each token has top_k expert assignments. This kernel computes:
//   output[token] = sum_{i=0}^{top_k-1} prob[token,i] * expert[id[token,i]](input[token])
//
// Strategy:
//   - Grid over (output_tiles, batch_tiles)
//   - Each threadgroup processes one output tile for a batch of tokens
//   - Loop over top_k expert slots
//   - For each slot, execute the expert GEMM and accumulate weighted results
//
// This is a simple approach that works well when top_k is small (2-4) and
// tokens have diverse expert assignments. For models where many tokens share
// the same expert, use the grouped kernel below.
//
// Dispatch: Grid [ceil(out_dim/64), ceil(batch/64)]
// ===========================================================================

kernel void moe_expert_gemm_fp4(
    device const half* activations     [[buffer(0)]],   // [batch, hidden]
    device const uint* expert_weights  [[buffer(1)]],   // [num_experts, hidden/8, out] packed FP4
    device const half* scales          [[buffer(2)]],   // [num_experts, hidden/group, out]
    device const uint* expert_ids      [[buffer(3)]],   // [batch, top_k]
    device const half* expert_probs    [[buffer(4)]],   // [batch, top_k]
    device half* output                [[buffer(5)]],   // [batch, out]
    constant MoEParams& params         [[buffer(6)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[MOE_NUM_BUFFERS][MOE_TILE_M][MOE_TILE_K];
    threadgroup half B_tiles[MOE_NUM_BUFFERS][MOE_TILE_K][MOE_TILE_N];
    // Staging buffer for store_results_weighted (must be in kernel, not helper)
    threadgroup half result_staging[MOE_SIMDGROUPS][MOE_SG_M_TILES * 8][MOE_SG_N_TILES * 8];

    const uint tg_row = tgid.y * MOE_TILE_M;  // Token batch offset
    const uint tg_col = tgid.x * MOE_TILE_N;  // Output dimension offset

    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (MOE_SG_N_TILES * 8);

    const uint thread_idx = simd_id * 32 + simd_lane;

    // Precompute expert strides
    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint expert_weight_stride = k_packed * params.out_dim;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;
    const uint expert_scale_stride = num_scale_groups * params.out_dim;

    // Loop over top_k expert slots
    for (uint slot = 0; slot < params.top_k; ++slot) {

        // Initialize accumulators for this expert
        simdgroup_matrix<half, 8, 8> acc[MOE_SG_M_TILES][MOE_SG_N_TILES];
        for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
            for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }

        // For this tile of tokens, we need to find which expert each is assigned to.
        // For simplicity, we assume all tokens in the tile use the same expert
        // (determined by the first valid token in the tile). This is an approximation
        // that works well when batch size is large and experts are well-distributed.
        //
        // TODO: For small batches or skewed distributions, use the grouped kernel.

        uint representative_token = tg_row;
        if (representative_token >= params.batch_size) {
            continue;  // Tile is entirely out of bounds
        }

        uint expert_id = expert_ids[representative_token * params.top_k + slot];
        if (expert_id >= params.num_experts) {
            continue;  // Invalid expert ID
        }

        // Compute pointers to this expert's weights and scales
        device const uint* B = expert_weights + expert_id * expert_weight_stride;
        device const half* S = scales + expert_id * expert_scale_stride;

        // Double-buffered K-dimension loop
        const uint num_k_tiles = (params.hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;
        uint buf_compute = 0;

        // Prologue: load first K-tile
        moe_load_A_tile(activations, A_tiles[0], params.batch_size, params.hidden_dim,
                        tg_row, 0, thread_idx);
        moe_load_B_tile_dequant(B, S, B_tiles[0], params.hidden_dim, params.out_dim,
                                tg_col, 0, params.group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Main loop
        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint k_offset = kt * MOE_TILE_K;
            uint next_k = k_offset + MOE_TILE_K;
            uint buf_load = 1 - buf_compute;

            // Prefetch next K-tile
            if (next_k < params.hidden_dim) {
                moe_load_A_tile(activations, A_tiles[buf_load], params.batch_size, params.hidden_dim,
                                tg_row, next_k, thread_idx);
                moe_load_B_tile_dequant(B, S, B_tiles[buf_load], params.hidden_dim, params.out_dim,
                                        tg_col, next_k, params.group_size, thread_idx);
            }

            // Compute on current buffer
            moe_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                                   acc, sg_row_offset, sg_col_offset);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_compute = buf_load;
        }

        // Store results with expert probability weighting
        moe_store_results_weighted(acc, output, expert_probs, result_staging,
                                   params.batch_size, params.out_dim, params.top_k,
                                   tg_row, tg_col,
                                   sg_row_offset, sg_col_offset,
                                   slot, (slot == 0), simd_lane, simd_id);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// Kernel 2: Token-grouped MoE expert GEMM
//
// For better cache utilization when many tokens share experts, group tokens
// by expert and process each expert's assigned tokens together.
//
// Requires pre-computed grouping:
//   sorted_token_ids: [total_assignments] - token indices sorted by expert
//   expert_offsets:   [num_experts + 1] - start/end offsets in sorted_token_ids
//   expert_probs_sorted: [total_assignments] - probabilities in sorted order
//
// Strategy:
//   - Grid over (output_tiles, num_experts)
//   - Each threadgroup processes one expert for all its assigned tokens
//   - Token batching within expert to fill M-tiles
//
// This achieves better weight reuse: each expert's weights are loaded once
// and applied to all assigned tokens, versus loading weights num_tokens times.
//
// Dispatch: Grid [ceil(out_dim/64), num_experts]
// ===========================================================================

kernel void moe_expert_gemm_fp4_grouped(
    device const half* activations        [[buffer(0)]],   // [batch, hidden]
    device const uint* expert_weights     [[buffer(1)]],   // [num_experts, hidden/8, out]
    device const half* scales             [[buffer(2)]],   // [num_experts, hidden/group, out]
    device const uint* sorted_token_ids   [[buffer(3)]],   // [total_assignments]
    device const uint* expert_offsets     [[buffer(4)]],   // [num_experts + 1]
    device const half* expert_probs_sorted [[buffer(5)]],  // [total_assignments]
    device half* output                   [[buffer(6)]],   // [batch, out]
    constant MoEParams& params            [[buffer(7)]],
    uint3 tgid                            [[threadgroup_position_in_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]],
    uint simd_id                          [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory for tiles
    threadgroup half A_tiles[MOE_NUM_BUFFERS][MOE_TILE_M][MOE_TILE_K];
    threadgroup half B_tiles[MOE_NUM_BUFFERS][MOE_TILE_K][MOE_TILE_N];
    // Token IDs for this batch
    threadgroup uint token_batch[MOE_TILE_M];
    threadgroup half prob_batch[MOE_TILE_M];

    const uint expert_id = tgid.y;
    const uint tg_col = tgid.x * MOE_TILE_N;

    if (expert_id >= params.num_experts) {
        return;
    }

    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (MOE_SG_N_TILES * 8);
    const uint thread_idx = simd_id * 32 + simd_lane;

    // Get token range for this expert
    uint token_start = expert_offsets[expert_id];
    uint token_end = expert_offsets[expert_id + 1];
    uint num_tokens = token_end - token_start;

    if (num_tokens == 0) {
        return;  // No tokens assigned to this expert
    }

    // Precompute expert strides
    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint expert_weight_stride = k_packed * params.out_dim;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;
    const uint expert_scale_stride = num_scale_groups * params.out_dim;

    // Pointers to this expert's weights
    device const uint* B = expert_weights + expert_id * expert_weight_stride;
    device const half* S = scales + expert_id * expert_scale_stride;

    // Process tokens in batches of MOE_TILE_M
    for (uint token_batch_start = 0; token_batch_start < num_tokens; token_batch_start += MOE_TILE_M) {
        uint batch_end = min(token_batch_start + MOE_TILE_M, num_tokens);
        uint batch_count = batch_end - token_batch_start;

        // Load token IDs and probabilities into threadgroup memory
        for (uint i = thread_idx; i < MOE_TILE_M; i += MOE_THREADS) {
            if (i < batch_count) {
                uint sorted_idx = token_start + token_batch_start + i;
                token_batch[i] = sorted_token_ids[sorted_idx];
                prob_batch[i] = expert_probs_sorted[sorted_idx];
            } else {
                token_batch[i] = 0;  // Padding
                prob_batch[i] = 0.0h;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Initialize accumulators
        simdgroup_matrix<half, 8, 8> acc[MOE_SG_M_TILES][MOE_SG_N_TILES];
        for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
            for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }

        // Double-buffered K loop
        const uint num_k_tiles = (params.hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;
        uint buf_compute = 0;

        // Load activation tile (gather from scattered token positions)
        // A_tiles[0][row][col] = activations[token_batch[row] * hidden + col]
        {
            constexpr uint VECTOR_SIZE = 4;
            const uint elems_per_thread = (MOE_TILE_M * MOE_TILE_K) / MOE_THREADS;
            const uint vec_per_thread = elems_per_thread / VECTOR_SIZE;

            for (uint i = 0; i < vec_per_thread; ++i) {
                uint vec_flat_idx = thread_idx * vec_per_thread + i;
                uint row = vec_flat_idx / (MOE_TILE_K / VECTOR_SIZE);
                uint vec_col = vec_flat_idx % (MOE_TILE_K / VECTOR_SIZE);
                uint col_base = vec_col * VECTOR_SIZE;

                half vals[VECTOR_SIZE] = {0.0h, 0.0h, 0.0h, 0.0h};
                if (row < batch_count) {
                    uint token_id = token_batch[row];
                    if (token_id < params.batch_size && col_base + VECTOR_SIZE <= params.hidden_dim) {
                        half4 loaded = *((device const half4*)(activations + token_id * params.hidden_dim + col_base));
                        vals[0] = loaded.x;
                        vals[1] = loaded.y;
                        vals[2] = loaded.z;
                        vals[3] = loaded.w;
                    } else {
                        for (uint v = 0; v < VECTOR_SIZE; ++v) {
                            if (token_id < params.batch_size) {
                                uint col = col_base + v;
                                if (col < params.hidden_dim) {
                                    vals[v] = activations[token_id * params.hidden_dim + col];
                                }
                            }
                        }
                    }
                }
                A_tiles[0][row][col_base + 0] = vals[0];
                A_tiles[0][row][col_base + 1] = vals[1];
                A_tiles[0][row][col_base + 2] = vals[2];
                A_tiles[0][row][col_base + 3] = vals[3];
            }
        }

        moe_load_B_tile_dequant(B, S, B_tiles[0], params.hidden_dim, params.out_dim,
                                tg_col, 0, params.group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Main K loop
        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint k_offset = kt * MOE_TILE_K;
            uint next_k = k_offset + MOE_TILE_K;
            uint buf_load = 1 - buf_compute;

            if (next_k < params.hidden_dim) {
                // Load next A tile (gathered)
                constexpr uint VECTOR_SIZE = 4;
                const uint elems_per_thread = (MOE_TILE_M * MOE_TILE_K) / MOE_THREADS;
                const uint vec_per_thread = elems_per_thread / VECTOR_SIZE;

                for (uint i = 0; i < vec_per_thread; ++i) {
                    uint vec_flat_idx = thread_idx * vec_per_thread + i;
                    uint row = vec_flat_idx / (MOE_TILE_K / VECTOR_SIZE);
                    uint vec_col = vec_flat_idx % (MOE_TILE_K / VECTOR_SIZE);
                    uint col_base = vec_col * VECTOR_SIZE;
                    uint global_col_base = next_k + col_base;

                    half vals[VECTOR_SIZE] = {0.0h, 0.0h, 0.0h, 0.0h};
                    if (row < batch_count) {
                        uint token_id = token_batch[row];
                        if (token_id < params.batch_size && global_col_base + VECTOR_SIZE <= params.hidden_dim) {
                            half4 loaded = *((device const half4*)(activations + token_id * params.hidden_dim + global_col_base));
                            vals[0] = loaded.x;
                            vals[1] = loaded.y;
                            vals[2] = loaded.z;
                            vals[3] = loaded.w;
                        } else {
                            for (uint v = 0; v < VECTOR_SIZE; ++v) {
                                uint global_col = global_col_base + v;
                                if (token_id < params.batch_size && global_col < params.hidden_dim) {
                                    vals[v] = activations[token_id * params.hidden_dim + global_col];
                                }
                            }
                        }
                    }
                    A_tiles[buf_load][row][col_base + 0] = vals[0];
                    A_tiles[buf_load][row][col_base + 1] = vals[1];
                    A_tiles[buf_load][row][col_base + 2] = vals[2];
                    A_tiles[buf_load][row][col_base + 3] = vals[3];
                }

                moe_load_B_tile_dequant(B, S, B_tiles[buf_load], params.hidden_dim, params.out_dim,
                                        tg_col, next_k, params.group_size, thread_idx);
            }

            moe_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                                   acc, sg_row_offset, sg_col_offset);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_compute = buf_load;
        }

        // Store results with probability weighting and scatter to output
        constexpr uint sg_tile_rows = MOE_SG_M_TILES * 8;
        constexpr uint sg_tile_cols = MOE_SG_N_TILES * 8;

        uint base_row = sg_row_offset;
        uint base_col = tg_col + sg_col_offset;

        // Stage results
        threadgroup half staging[MOE_SIMDGROUPS][sg_tile_rows][sg_tile_cols];
        for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
            for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                simdgroup_store(acc[mi][ni],
                                &staging[simd_id][mi * 8][ni * 8],
                                sg_tile_cols);
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter write with probability weighting
        constexpr uint total_elems = sg_tile_rows * sg_tile_cols;
        constexpr uint elems_per_lane = total_elems / 32;

        for (uint iter = 0; iter < elems_per_lane; ++iter) {
            uint elem = simd_lane * elems_per_lane + iter;
            uint row = elem / sg_tile_cols;
            uint col = elem % sg_tile_cols;

            uint local_row = base_row + row;
            uint out_col = base_col + col;

            if (local_row < batch_count && out_col < params.out_dim) {
                uint token_id = token_batch[local_row];
                half prob = prob_batch[local_row];
                half val = staging[simd_id][row][col] * prob;

                // Atomic add to handle multiple experts writing to same token
                // Note: on Metal 3.0+, we could use atomics. For now, we assume
                // the host handles accumulation across expert outputs, or we
                // allocate separate output buffers per expert.
                output[token_id * params.out_dim + out_col] += val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// Kernel 3: Shared expert execution (always runs for all tokens)
//
// Many MoE architectures (DeepSeek, Mixtral variants) have a "shared" expert
// that runs for all tokens regardless of routing. This is a simple dense GEMM.
//
// output += shared_prob * shared_expert(activations)
//
// Can be overlapped with routed expert dispatch if scheduled on separate queue.
//
// Dispatch: Grid [ceil(out_dim/64), ceil(batch/64)]
// ===========================================================================

kernel void moe_expert_gemm_shared_fp4(
    device const half* activations       [[buffer(0)]],   // [batch, hidden]
    device const uint* shared_weights    [[buffer(1)]],   // [hidden/8, out] packed FP4
    device const half* shared_scales     [[buffer(2)]],   // [hidden/group, out]
    device half* output                  [[buffer(3)]],   // [batch, out]
    constant uint& batch_size            [[buffer(4)]],
    constant uint& hidden_dim            [[buffer(5)]],
    constant uint& out_dim               [[buffer(6)]],
    constant uint& group_size            [[buffer(7)]],
    constant half& shared_prob           [[buffer(8)]],   // Global shared expert weight
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[MOE_NUM_BUFFERS][MOE_TILE_M][MOE_TILE_K];
    threadgroup half B_tiles[MOE_NUM_BUFFERS][MOE_TILE_K][MOE_TILE_N];

    const uint tg_row = tgid.y * MOE_TILE_M;
    const uint tg_col = tgid.x * MOE_TILE_N;

    const uint sg_row_offset = 0;  // All simdgroups cover all rows
    const uint sg_col_offset = simd_id * (MOE_SG_N_TILES * 8);

    const uint thread_idx = simd_id * 32 + simd_lane;

    simdgroup_matrix<half, 8, 8> acc[MOE_SG_M_TILES][MOE_SG_N_TILES];
    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint num_k_tiles = (hidden_dim + MOE_TILE_K - 1) / MOE_TILE_K;
    uint buf_compute = 0;

    // Prologue
    moe_load_A_tile(activations, A_tiles[0], batch_size, hidden_dim, tg_row, 0, thread_idx);
    moe_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[0],
                            hidden_dim, out_dim, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * MOE_TILE_K;
        uint next_k = k_offset + MOE_TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < hidden_dim) {
            moe_load_A_tile(activations, A_tiles[buf_load], batch_size, hidden_dim,
                            tg_row, next_k, thread_idx);
            moe_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[buf_load],
                                    hidden_dim, out_dim, tg_col, next_k, group_size, thread_idx);
        }

        moe_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                               acc, sg_row_offset, sg_col_offset);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store results with shared_prob weighting (accumulate into output)
    constexpr uint sg_tile_rows = MOE_SG_M_TILES * 8;
    constexpr uint sg_tile_cols = MOE_SG_N_TILES * 8;

    uint base_row = tg_row + sg_row_offset;
    uint base_col = tg_col + sg_col_offset;

    threadgroup half staging[MOE_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            simdgroup_store(acc[mi][ni],
                            &staging[simd_id][mi * 8][ni * 8],
                            sg_tile_cols);
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint total_elems = sg_tile_rows * sg_tile_cols;
    constexpr uint elems_per_lane = total_elems / 32;

    for (uint iter = 0; iter < elems_per_lane; ++iter) {
        uint elem = simd_lane * elems_per_lane + iter;
        uint row = elem / sg_tile_cols;
        uint col = elem % sg_tile_cols;

        uint out_row = base_row + row;
        uint out_col = base_col + col;

        if (out_row < batch_size && out_col < out_dim) {
            half val = staging[simd_id][row][col] * shared_prob;
            output[out_row * out_dim + out_col] += val;
        }
    }
}
