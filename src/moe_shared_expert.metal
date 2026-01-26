// moe_shared_expert.metal - Shared Expert Fusion for MoE Models
//
// Implements fused shared expert + routed expert aggregation for MoE
// architectures like GLM-4, Qwen3-MoE, and DeepSeek-V2.
//
// Architecture pattern:
//   output = shared_expert(x) + sum(prob_i * routed_expert_i(x))
//
// The shared expert always runs on every token. Routed experts are
// selected per-token by a gating mechanism (typically top-k softmax).
// This kernel fuses the aggregation to avoid a separate kernel launch
// for the shared expert addition.
//
// Kernel variants:
//   1. moe_shared_expert_fused         - Fused shared + routed aggregation
//   2. moe_shared_expert_fused_fp4     - With FP4 dequantization for experts
//   3. moe_shared_expert_scatter       - Scatter-style per-token routing
//
// Design notes:
//   - Shared expert output computed first (always needed)
//   - Routed expert outputs accumulated with probability weights
//   - Simdgroup reductions for weighted sum across experts
//   - Token-parallel dispatch (one threadgroup per token for small M)
//   - Expert-parallel within threadgroup for large models
//
// Memory layout:
//   - hidden_states: [num_tokens, hidden_dim]
//   - shared_expert_weights: [hidden_dim, intermediate_dim] (or quantized)
//   - routed_expert_weights: [num_experts, hidden_dim, intermediate_dim]
//   - router_probs: [num_tokens, top_k] - gating probabilities
//   - router_indices: [num_tokens, top_k] - selected expert indices
//
// References:
//   - GLM-4: https://arxiv.org/abs/2406.12793
//   - Qwen2-MoE: https://arxiv.org/abs/2407.10671
//   - DeepSeek-V2: https://arxiv.org/abs/2405.04434

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Configuration constants
//
// Tuned for M4 Max. The shared expert runs on all tokens, so we optimize
// for the common case of single-token decode (M=1) while supporting
// batched prefill.
// ---------------------------------------------------------------------------

// Tile dimensions for expert GEMM
constant constexpr uint MOE_TILE_M = 64;
constant constexpr uint MOE_TILE_N = 64;
constant constexpr uint MOE_TILE_K = 32;

constant constexpr uint MOE_K_TILES = MOE_TILE_K / 8;  // 4
constant constexpr uint MOE_SIMDGROUPS_PER_TG = 4;
constant constexpr uint MOE_THREADS_PER_TG = MOE_SIMDGROUPS_PER_TG * 32;  // 128

// Each simdgroup handles 2x4 block of 8x8 tiles (16x32 output region)
constant constexpr uint MOE_SG_M_TILES = 2;
constant constexpr uint MOE_SG_N_TILES = 4;

// FP4 packing: 8 FP4 values per uint32
constant constexpr uint MOE_FP4_PER_UINT = 8;

// Maximum experts to aggregate (top-k routing)
constant constexpr uint MAX_TOP_K = 8;

// Double-buffering for pipelining
constant constexpr uint MOE_NUM_BUFFERS = 2;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

inline uint moe_div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

// Guard against NaN/Inf in accumulation
inline half guard_finite_moe(half val) {
    return select(val, half(0.0h), !isfinite(val));
}

// Safe multiply-add with NaN/Inf guard (for weighted expert sum)
inline half safe_fma(half a, half b, half c) {
    float result = fma((float)a, (float)b, (float)c);
    return isfinite(result) ? (half)result : c;
}

// ---------------------------------------------------------------------------
// FP4 E2M1 dequantization
//
// Same format as marlin_gemm.metal. Inline to avoid function call overhead.
// Uses float intermediates for Metal half precision bug workaround.
// ---------------------------------------------------------------------------

inline half moe_dequant_fp4(uint nibble, half scale) {
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

inline void moe_unpack_fp4x8(uint packed, half scale, thread half* out) {
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = moe_dequant_fp4(nibble, scale);
    }
}

// ---------------------------------------------------------------------------
// Simdgroup reduction utilities for weighted expert aggregation
//
// These implement fast cross-lane reductions using simdgroup shuffle.
// Used to sum weighted expert outputs across the simdgroup.
// ---------------------------------------------------------------------------

// Simdgroup-wide sum of a half value
inline half simd_sum_half(half val, uint simd_lane [[thread_index_in_simdgroup]]) {
    // Tree reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    val += simd_shuffle_xor(val, 1);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 16);
    return val;
}

// Simdgroup-wide weighted sum: sum(weight[i] * value[i]) for lanes with data
// Each lane holds one expert's contribution; we sum across lanes.
inline half simd_weighted_sum(
    half weight,
    half value,
    uint num_active,  // How many lanes have valid data
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Mask out inactive lanes
    half contribution = (simd_lane < num_active) ? (weight * value) : half(0.0h);
    return simd_sum_half(contribution, simd_lane);
}

// ---------------------------------------------------------------------------
// Cooperative tile loaders (adapted from marlin_gemm.metal)
// ---------------------------------------------------------------------------

inline void moe_load_A_tile(
    device const half* A,
    threadgroup half (&A_buf)[MOE_TILE_M][MOE_TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (MOE_TILE_M * MOE_TILE_K) / MOE_THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / MOE_TILE_K;
        uint col = flat_idx % MOE_TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = 0.0h;
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_buf[row][col] = val;
    }
}

inline void moe_load_B_tile_dequant(
    device const uint* B,
    device const half* scales,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint scale_tiles = moe_div_ceil(K, group_size);
    const uint k_packs = moe_div_ceil(K, MOE_FP4_PER_UINT);
    const uint packed_per_thread = (MOE_TILE_K * MOE_TILE_N) / (MOE_THREADS_PER_TG * MOE_FP4_PER_UINT);

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (MOE_TILE_K / MOE_FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (MOE_TILE_K / MOE_FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * MOE_FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = 1.0h;
        if (global_n < N && global_k_base < K && scale_k < scale_tiles) {
            s = scales[scale_k * N + global_n];
        }

        uint packed = 0;
        uint b_row = global_k_base / MOE_FP4_PER_UINT;
        if (global_n < N && b_row < k_packs && global_k_base < K) {
            packed = B[b_row * N + global_n];
        }

        uint tile_k_base = k_group_in_tile * MOE_FP4_PER_UINT;
        half vals[8];
        moe_unpack_fp4x8(packed, s, vals);
        for (uint v = 0; v < MOE_FP4_PER_UINT && (tile_k_base + v) < MOE_TILE_K; ++v) {
            if (n_idx < MOE_TILE_N) {
                B_buf[tile_k_base + v][n_idx] = vals[v];
            }
        }
    }
}

inline void moe_load_B_tile_fp16(
    device const half* B,
    threadgroup half (&B_buf)[MOE_TILE_K][MOE_TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (MOE_TILE_K * MOE_TILE_N) / MOE_THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / MOE_TILE_N;
        uint col = flat_idx % MOE_TILE_N;
        uint global_row = k_block + row;
        uint global_col = tg_col + col;

        half val = 0.0h;
        if (global_row < K && global_col < N) {
            val = B[global_row * N + global_col];
        }
        B_buf[row][col] = val;
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute: multiply A sub-tile by B sub-tile, accumulate
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
                                              a_frag,
                                              b_frag,
                                              acc[mi][ni]);
            }
        }
    }
}

// ===========================================================================
// Kernel 1: moe_shared_expert_fused
//
// Fused shared expert + routed expert aggregation (FP16 weights).
//
// Computes: output[t,d] = shared_out[t,d] + sum_k(prob[t,k] * expert_out[t,k,d])
//
// For each token:
//   1. Compute shared_expert(hidden_states[t]) -> shared_out[t]
//   2. For each selected expert k in top-K:
//      - Compute expert_k(hidden_states[t]) -> expert_out[t,k]
//      - Accumulate: prob[t,k] * expert_out[t,k]
//   3. output[t] = shared_out[t] + accumulated routed output
//
// This version dispatches one threadgroup per (M_tile, N_tile, token) triplet.
// For single-token decode (M=1), this becomes one TG per N_tile.
//
// Buffers:
//   0: hidden_states [num_tokens, hidden_dim] - input activations
//   1: shared_expert_w [hidden_dim, intermediate_dim] - shared expert weights
//   2: routed_expert_w [num_experts, hidden_dim, intermediate_dim] - expert weights
//   3: router_probs [num_tokens, top_k] - gating probabilities
//   4: router_indices [num_tokens, top_k] - selected expert indices (uint32)
//   5: output [num_tokens, intermediate_dim] - output buffer
//   6: num_tokens (uint)
//   7: hidden_dim (uint)
//   8: intermediate_dim (uint)
//   9: top_k (uint) - number of experts per token
//   10: num_experts (uint) - total number of routed experts
//
// Dispatch: Grid(ceil(intermediate_dim/64), ceil(num_tokens/64), 1)
//           Threadgroup: 128 threads (4 simdgroups)
// ===========================================================================

kernel void moe_shared_expert_fused(
    device const half* hidden_states        [[buffer(0)]],   // [num_tokens, hidden_dim]
    device const half* shared_expert_w      [[buffer(1)]],   // [hidden_dim, intermediate_dim]
    device const half* routed_expert_w      [[buffer(2)]],   // [num_experts, hidden_dim, intermediate_dim]
    device const half* router_probs         [[buffer(3)]],   // [num_tokens, top_k]
    device const uint* router_indices       [[buffer(4)]],   // [num_tokens, top_k]
    device half* output                     [[buffer(5)]],   // [num_tokens, intermediate_dim]
    constant uint& num_tokens               [[buffer(6)]],
    constant uint& hidden_dim               [[buffer(7)]],
    constant uint& intermediate_dim         [[buffer(8)]],
    constant uint& top_k                    [[buffer(9)]],
    constant uint& num_experts              [[buffer(10)]],
    uint3 tgid                              [[threadgroup_position_in_grid]],
    uint simd_lane                          [[thread_index_in_simdgroup]],
    uint simd_id                            [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory for hidden_states (A) and weights (B)
    threadgroup half A_tiles[MOE_NUM_BUFFERS][MOE_TILE_M][MOE_TILE_K];
    threadgroup half B_tiles[MOE_NUM_BUFFERS][MOE_TILE_K][MOE_TILE_N];

    // Per-simdgroup staging for output
    threadgroup half sg_staging[MOE_SIMDGROUPS_PER_TG][MOE_SG_M_TILES * 8][MOE_SG_N_TILES * 8];

    const uint tg_row = tgid.y * MOE_TILE_M;  // Token block start
    const uint tg_col = tgid.x * MOE_TILE_N;  // Output dim block start

    if (tg_row >= num_tokens) return;

    const uint sg_row_offset = (simd_id / 2) * (MOE_SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (MOE_SG_N_TILES * 8);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = moe_div_ceil(hidden_dim, MOE_TILE_K);

    // =========================================================================
    // Phase 1: Compute shared expert output
    //
    // shared_out = hidden_states @ shared_expert_w
    // This runs on every token (the "shared" in shared expert).
    // =========================================================================

    simdgroup_matrix<half, 8, 8> shared_acc[MOE_SG_M_TILES][MOE_SG_N_TILES];
    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi)
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni)
            shared_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint buf_compute = 0;

    // Prologue: Load first K-tile
    moe_load_A_tile(hidden_states, A_tiles[0], num_tokens, hidden_dim,
                    tg_row, 0, thread_idx);
    moe_load_B_tile_fp16(shared_expert_w, B_tiles[0], hidden_dim, intermediate_dim,
                         tg_col, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pipelined K-reduction loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * MOE_TILE_K;
        uint next_k = k_offset + MOE_TILE_K;
        uint buf_load = 1 - buf_compute;

        // Load next K-tile (overlapped with compute)
        if (next_k < hidden_dim) {
            moe_load_A_tile(hidden_states, A_tiles[buf_load], num_tokens, hidden_dim,
                            tg_row, next_k, thread_idx);
            moe_load_B_tile_fp16(shared_expert_w, B_tiles[buf_load], hidden_dim, intermediate_dim,
                                 tg_col, next_k, thread_idx);
        }

        // Compute on current buffer
        moe_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                               shared_acc, sg_row_offset, sg_col_offset);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store shared expert result to staging (will add routed outputs below)
    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            simdgroup_store(shared_acc[mi][ni],
                            &sg_staging[simd_id][mi * 8][ni * 8],
                            MOE_SG_N_TILES * 8);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 2: Accumulate routed expert outputs
    //
    // For each token in our tile, iterate over its top-k selected experts,
    // compute the expert output, and add prob[k] * expert_out to the accumulator.
    //
    // This fuses the routing aggregation: no separate kernel for the addition.
    // =========================================================================

    // Expert stride: each expert's weights are [hidden_dim, intermediate_dim]
    const uint expert_stride = hidden_dim * intermediate_dim;

    // Process each token in our M-tile that needs routed experts
    uint actual_m = min(MOE_TILE_M, num_tokens - tg_row);

    for (uint token_in_tile = 0; token_in_tile < actual_m; ++token_in_tile) {
        uint token_idx = tg_row + token_in_tile;

        // Load routing info for this token
        device const half* probs = router_probs + token_idx * top_k;
        device const uint* indices = router_indices + token_idx * top_k;

        // For each selected expert
        for (uint k = 0; k < top_k; ++k) {
            half prob = probs[k];
            uint expert_idx = indices[k];

            // Skip if probability is too small (gating sparsity)
            if (prob < half(1e-6h) || expert_idx >= num_experts) continue;

            // Get pointer to this expert's weights
            device const half* expert_w = routed_expert_w + expert_idx * expert_stride;

            // Compute expert output for this token
            // We compute one row of the output at a time (token_in_tile, :)
            simdgroup_matrix<half, 8, 8> expert_acc[MOE_SG_M_TILES][MOE_SG_N_TILES];
            for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi)
                for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni)
                    expert_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

            buf_compute = 0;

            // Load first K-tile for this expert
            // A is the single token's hidden state, B is expert weights
            moe_load_A_tile(hidden_states, A_tiles[0], num_tokens, hidden_dim,
                            token_idx, 0, thread_idx);
            moe_load_B_tile_fp16(expert_w, B_tiles[0], hidden_dim, intermediate_dim,
                                 tg_col, 0, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kt_inner = 0; kt_inner < num_k_tiles; ++kt_inner) {
                uint k_off = kt_inner * MOE_TILE_K;
                uint next_k_inner = k_off + MOE_TILE_K;
                uint buf_load_inner = 1 - buf_compute;

                if (next_k_inner < hidden_dim) {
                    moe_load_A_tile(hidden_states, A_tiles[buf_load_inner],
                                    num_tokens, hidden_dim,
                                    token_idx, next_k_inner, thread_idx);
                    moe_load_B_tile_fp16(expert_w, B_tiles[buf_load_inner],
                                         hidden_dim, intermediate_dim,
                                         tg_col, next_k_inner, thread_idx);
                }

                // For single-token expert compute, we only care about row 0
                // of the A_tile (the token's hidden state)
                moe_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                                       expert_acc, 0, sg_col_offset);

                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf_compute = buf_load_inner;
            }

            // Add weighted expert output to staging
            // Only update the row corresponding to token_in_tile
            // expert_acc contains results for row 0 (the single token)
            for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                // Store expert result to temporary, then accumulate
                threadgroup half temp_out[8][8];
                simdgroup_store(expert_acc[0][ni], &temp_out[0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Each lane accumulates part of the 8x8 tile
                // We only care about row 0 (single token)
                if (simd_lane < 8) {
                    uint row_in_staging = token_in_tile % (MOE_SG_M_TILES * 8);
                    uint col = sg_col_offset + ni * 8 + simd_lane;
                    if (col < MOE_SG_N_TILES * 8) {
                        half weighted = prob * temp_out[0][simd_lane];
                        sg_staging[simd_id][row_in_staging][col] =
                            safe_fma(prob, temp_out[0][simd_lane],
                                     sg_staging[simd_id][row_in_staging][col]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 3: Store final output
    //
    // sg_staging now contains shared_out + sum(prob_k * expert_k_out)
    // =========================================================================

    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            // Reload from staging for proper simdgroup store
            simdgroup_matrix<half, 8, 8> out_mat;
            simdgroup_load(out_mat,
                           &sg_staging[simd_id][mi * 8][ni * 8],
                           MOE_SG_N_TILES * 8);

            if (out_row + 8 <= num_tokens && out_col + 8 <= intermediate_dim) {
                simdgroup_store(out_mat, output + out_row * intermediate_dim + out_col,
                                intermediate_dim);
            } else if (out_row < num_tokens && out_col < intermediate_dim) {
                // Boundary case: element-wise store
                threadgroup half temp[8][8];
                simdgroup_store(out_mat, &temp[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < num_tokens && out_col + c < intermediate_dim) {
                        output[(out_row + r) * intermediate_dim + out_col + c] = temp[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ===========================================================================
// Kernel 2: moe_shared_expert_fused_fp4
//
// Same as moe_shared_expert_fused but with FP4-quantized expert weights.
// Shared expert can optionally be at higher precision (FP16) for quality.
//
// Buffers:
//   0: hidden_states [num_tokens, hidden_dim]
//   1: shared_expert_w [hidden_dim/8, intermediate_dim] - packed FP4
//   2: shared_expert_scales [hidden_dim/group_size, intermediate_dim]
//   3: routed_expert_w [num_experts, hidden_dim/8, intermediate_dim] - packed FP4
//   4: routed_expert_scales [num_experts, hidden_dim/group_size, intermediate_dim]
//   5: router_probs [num_tokens, top_k]
//   6: router_indices [num_tokens, top_k]
//   7: output [num_tokens, intermediate_dim]
//   8: num_tokens, 9: hidden_dim, 10: intermediate_dim, 11: top_k, 12: num_experts
//   13: group_size
// ===========================================================================

kernel void moe_shared_expert_fused_fp4(
    device const half* hidden_states          [[buffer(0)]],
    device const uint* shared_expert_w        [[buffer(1)]],   // Packed FP4
    device const half* shared_expert_scales   [[buffer(2)]],
    device const uint* routed_expert_w        [[buffer(3)]],   // Packed FP4
    device const half* routed_expert_scales   [[buffer(4)]],
    device const half* router_probs           [[buffer(5)]],
    device const uint* router_indices         [[buffer(6)]],
    device half* output                       [[buffer(7)]],
    constant uint& num_tokens                 [[buffer(8)]],
    constant uint& hidden_dim                 [[buffer(9)]],
    constant uint& intermediate_dim           [[buffer(10)]],
    constant uint& top_k                      [[buffer(11)]],
    constant uint& num_experts                [[buffer(12)]],
    constant uint& group_size                 [[buffer(13)]],
    uint3 tgid                                [[threadgroup_position_in_grid]],
    uint simd_lane                            [[thread_index_in_simdgroup]],
    uint simd_id                              [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[MOE_NUM_BUFFERS][MOE_TILE_M][MOE_TILE_K];
    threadgroup half B_tiles[MOE_NUM_BUFFERS][MOE_TILE_K][MOE_TILE_N];
    threadgroup half sg_staging[MOE_SIMDGROUPS_PER_TG][MOE_SG_M_TILES * 8][MOE_SG_N_TILES * 8];

    const uint tg_row = tgid.y * MOE_TILE_M;
    const uint tg_col = tgid.x * MOE_TILE_N;

    if (tg_row >= num_tokens) return;

    const uint sg_row_offset = (simd_id / 2) * (MOE_SG_M_TILES * 8);
    const uint sg_col_offset = (simd_id % 2) * (MOE_SG_N_TILES * 8);

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = moe_div_ceil(hidden_dim, MOE_TILE_K);

    // =========================================================================
    // Phase 1: Compute shared expert output (FP4 dequant)
    // =========================================================================

    simdgroup_matrix<half, 8, 8> shared_acc[MOE_SG_M_TILES][MOE_SG_N_TILES];
    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi)
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni)
            shared_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint buf_compute = 0;

    moe_load_A_tile(hidden_states, A_tiles[0], num_tokens, hidden_dim,
                    tg_row, 0, thread_idx);
    moe_load_B_tile_dequant(shared_expert_w, shared_expert_scales, B_tiles[0],
                            hidden_dim, intermediate_dim, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * MOE_TILE_K;
        uint next_k = k_offset + MOE_TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < hidden_dim) {
            moe_load_A_tile(hidden_states, A_tiles[buf_load], num_tokens, hidden_dim,
                            tg_row, next_k, thread_idx);
            moe_load_B_tile_dequant(shared_expert_w, shared_expert_scales, B_tiles[buf_load],
                                    hidden_dim, intermediate_dim, tg_col, next_k, group_size, thread_idx);
        }

        moe_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                               shared_acc, sg_row_offset, sg_col_offset);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store shared expert result
    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            simdgroup_store(shared_acc[mi][ni],
                            &sg_staging[simd_id][mi * 8][ni * 8],
                            MOE_SG_N_TILES * 8);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 2: Accumulate routed expert outputs (FP4)
    // =========================================================================

    // Expert strides for packed FP4
    const uint expert_w_stride = (hidden_dim / MOE_FP4_PER_UINT) * intermediate_dim;
    const uint expert_s_stride = (hidden_dim / group_size) * intermediate_dim;

    uint actual_m = min(MOE_TILE_M, num_tokens - tg_row);

    for (uint token_in_tile = 0; token_in_tile < actual_m; ++token_in_tile) {
        uint token_idx = tg_row + token_in_tile;

        device const half* probs = router_probs + token_idx * top_k;
        device const uint* indices = router_indices + token_idx * top_k;

        for (uint k = 0; k < top_k; ++k) {
            half prob = probs[k];
            uint expert_idx = indices[k];

            if (prob < half(1e-6h) || expert_idx >= num_experts) continue;

            device const uint* expert_w = routed_expert_w + expert_idx * expert_w_stride;
            device const half* expert_s = routed_expert_scales + expert_idx * expert_s_stride;

            simdgroup_matrix<half, 8, 8> expert_acc[MOE_SG_M_TILES][MOE_SG_N_TILES];
            for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi)
                for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni)
                    expert_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

            buf_compute = 0;

            moe_load_A_tile(hidden_states, A_tiles[0], num_tokens, hidden_dim,
                            token_idx, 0, thread_idx);
            moe_load_B_tile_dequant(expert_w, expert_s, B_tiles[0],
                                    hidden_dim, intermediate_dim, tg_col, 0, group_size, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kt_inner = 0; kt_inner < num_k_tiles; ++kt_inner) {
                uint k_off = kt_inner * MOE_TILE_K;
                uint next_k_inner = k_off + MOE_TILE_K;
                uint buf_load_inner = 1 - buf_compute;

                if (next_k_inner < hidden_dim) {
                    moe_load_A_tile(hidden_states, A_tiles[buf_load_inner],
                                    num_tokens, hidden_dim,
                                    token_idx, next_k_inner, thread_idx);
                    moe_load_B_tile_dequant(expert_w, expert_s, B_tiles[buf_load_inner],
                                            hidden_dim, intermediate_dim,
                                            tg_col, next_k_inner, group_size, thread_idx);
                }

                moe_compute_from_tiles(A_tiles[buf_compute], B_tiles[buf_compute],
                                       expert_acc, 0, sg_col_offset);

                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf_compute = buf_load_inner;
            }

            // Accumulate weighted expert output
            for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
                threadgroup half temp_out[8][8];
                simdgroup_store(expert_acc[0][ni], &temp_out[0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                if (simd_lane < 8) {
                    uint row_in_staging = token_in_tile % (MOE_SG_M_TILES * 8);
                    uint col = sg_col_offset + ni * 8 + simd_lane;
                    if (col < MOE_SG_N_TILES * 8) {
                        sg_staging[simd_id][row_in_staging][col] =
                            safe_fma(prob, temp_out[0][simd_lane],
                                     sg_staging[simd_id][row_in_staging][col]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 3: Store final output
    // =========================================================================

    for (uint mi = 0; mi < MOE_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < MOE_SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            simdgroup_matrix<half, 8, 8> out_mat;
            simdgroup_load(out_mat,
                           &sg_staging[simd_id][mi * 8][ni * 8],
                           MOE_SG_N_TILES * 8);

            if (out_row + 8 <= num_tokens && out_col + 8 <= intermediate_dim) {
                simdgroup_store(out_mat, output + out_row * intermediate_dim + out_col,
                                intermediate_dim);
            } else if (out_row < num_tokens && out_col < intermediate_dim) {
                threadgroup half temp[8][8];
                simdgroup_store(out_mat, &temp[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < num_tokens && out_col + c < intermediate_dim) {
                        output[(out_row + r) * intermediate_dim + out_col + c] = temp[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ===========================================================================
// Kernel 3: moe_shared_expert_scatter
//
// Optimized for the decode case (M=1 or small M).
// Uses a different dispatch pattern: one threadgroup per token, each TG
// handles all output dimensions for that token using a loop.
//
// This avoids launching O(num_tokens * N_tiles) threadgroups when we have
// very few tokens but many output dimensions.
//
// Buffers: Same as moe_shared_expert_fused.
// Dispatch: Grid(num_tokens, 1, 1), Threadgroup: 128 threads
// ===========================================================================

kernel void moe_shared_expert_scatter(
    device const half* hidden_states        [[buffer(0)]],
    device const half* shared_expert_w      [[buffer(1)]],
    device const half* routed_expert_w      [[buffer(2)]],
    device const half* router_probs         [[buffer(3)]],
    device const uint* router_indices       [[buffer(4)]],
    device half* output                     [[buffer(5)]],
    constant uint& num_tokens               [[buffer(6)]],
    constant uint& hidden_dim               [[buffer(7)]],
    constant uint& intermediate_dim         [[buffer(8)]],
    constant uint& top_k                    [[buffer(9)]],
    constant uint& num_experts              [[buffer(10)]],
    uint3 tgid                              [[threadgroup_position_in_grid]],
    uint simd_lane                          [[thread_index_in_simdgroup]],
    uint simd_id                            [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= num_tokens) return;

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint expert_stride = hidden_dim * intermediate_dim;

    // Each simdgroup processes a portion of the output dimension
    // 4 simdgroups * 32 threads = 128 threads
    // Each thread processes output dims in a strided pattern

    // Load routing info for this token
    device const half* probs = router_probs + token_idx * top_k;
    device const uint* indices = router_indices + token_idx * top_k;

    // Process output dimension in chunks
    for (uint out_d = thread_idx; out_d < intermediate_dim; out_d += MOE_THREADS_PER_TG) {
        // Compute shared expert output for this dimension
        float acc = 0.0f;

        // Dot product: hidden_states[token, :] @ shared_expert_w[:, out_d]
        for (uint h = 0; h < hidden_dim; ++h) {
            float x = (float)hidden_states[token_idx * hidden_dim + h];
            float w = (float)shared_expert_w[h * intermediate_dim + out_d];
            acc = fma(x, w, acc);
        }

        // Add routed expert contributions
        for (uint k = 0; k < top_k; ++k) {
            half prob = probs[k];
            uint expert_idx = indices[k];

            if (prob < half(1e-6h) || expert_idx >= num_experts) continue;

            device const half* expert_w = routed_expert_w + expert_idx * expert_stride;

            // Dot product: hidden_states[token, :] @ expert_w[:, out_d]
            float expert_out = 0.0f;
            for (uint h = 0; h < hidden_dim; ++h) {
                float x = (float)hidden_states[token_idx * hidden_dim + h];
                float w = (float)expert_w[h * intermediate_dim + out_d];
                expert_out = fma(x, w, expert_out);
            }

            acc = fma((float)prob, expert_out, acc);
        }

        output[token_idx * intermediate_dim + out_d] = (half)acc;
    }
}

// ===========================================================================
// Kernel 4: moe_aggregate_weighted
//
// Lightweight kernel for just the weighted aggregation step when shared
// and routed expert outputs are already computed separately.
//
// Computes: output[t,d] = shared_out[t,d] + sum_k(prob[t,k] * expert_out[t,k,d])
//
// This is useful when the expert GEMMs are already done and we just need
// to combine the results. Uses simdgroup reductions for efficiency.
//
// Buffers:
//   0: shared_out [num_tokens, output_dim]
//   1: expert_out [num_tokens, top_k, output_dim]
//   2: router_probs [num_tokens, top_k]
//   3: output [num_tokens, output_dim]
//   4: num_tokens, 5: output_dim, 6: top_k
//
// Dispatch: Grid(ceil(output_dim/128), num_tokens, 1), Threadgroup: 128
// ===========================================================================

kernel void moe_aggregate_weighted(
    device const half* shared_out           [[buffer(0)]],
    device const half* expert_out           [[buffer(1)]],
    device const half* router_probs         [[buffer(2)]],
    device half* output                     [[buffer(3)]],
    constant uint& num_tokens               [[buffer(4)]],
    constant uint& output_dim               [[buffer(5)]],
    constant uint& top_k                    [[buffer(6)]],
    uint3 tgid                              [[threadgroup_position_in_grid]],
    uint simd_lane                          [[thread_index_in_simdgroup]],
    uint simd_id                            [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.y;
    const uint d_base = tgid.x * MOE_THREADS_PER_TG;

    if (token_idx >= num_tokens) return;

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint d = d_base + thread_idx;

    if (d >= output_dim) return;

    // Load shared expert output
    float acc = (float)shared_out[token_idx * output_dim + d];

    // Load routing probs for this token
    device const half* probs = router_probs + token_idx * top_k;

    // expert_out layout: [num_tokens, top_k, output_dim]
    device const half* token_experts = expert_out + token_idx * top_k * output_dim;

    // Accumulate weighted expert outputs
    for (uint k = 0; k < top_k; ++k) {
        half prob = probs[k];
        if (prob >= half(1e-6h)) {
            half expert_val = token_experts[k * output_dim + d];
            acc = fma((float)prob, (float)expert_val, acc);
        }
    }

    output[token_idx * output_dim + d] = (half)acc;
}
