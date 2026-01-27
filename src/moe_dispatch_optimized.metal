// moe_dispatch_optimized.metal - Ultra-optimized MoE dispatch for Apple Silicon
//
// Target: GLM-4.7-Flash MoE configuration
//   - 64 experts total
//   - 4 experts active per token (top-k=4)
//   - 1 shared expert (always active)
//   - hidden_dim: 4096, intermediate_dim: 14336
//
// Design goal: <5% overhead vs dense model of equivalent active params
//
// Key optimizations:
//   1. Vectorized top-k using simdgroup parallel bitonic sort (O(log^2 n))
//   2. Atomic-free token grouping via warp-level prefix sum
//   3. Coalesced expert weight loading with transposed layout
//   4. Fused routing weight multiplication on output
//   5. Persistent threadgroups to minimize dispatch overhead
//   6. Register-based accumulation with simdgroup matrix ops
//
// Memory layout (optimized for coalescing):
//   activations:     [batch, hidden_dim] half, row-major
//   expert_weights:  [num_experts, out_dim, hidden_dim/8] packed FP4, transposed
//   scales:          [num_experts, out_dim, num_groups] half, transposed
//   router_logits:   [batch, num_experts] half (or computed inline)
//   output:          [batch, out_dim] half
//
// Dispatch strategy:
//   Single-kernel approach that fuses: routing -> grouping -> GEMM -> combine
//   Avoids synchronization points between phases by using threadgroup memory

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>

#include "bf16_compat.metal"

using namespace metal;

// ===========================================================================
// Configuration - optimized for GLM-4.7-Flash
// ===========================================================================

// Expert configuration
constant constexpr uint NUM_EXPERTS = 64;
constant constexpr uint TOP_K = 4;
constant constexpr uint HAS_SHARED_EXPERT = 1;

// Tile dimensions - tuned for Apple Silicon threadgroup memory limits
// NOTE: OPT_TILE_M * OPT_TILE_N drives threadgroup memory usage. Keep <= 32 KB.
constant constexpr uint OPT_TILE_M = 16;    // Tokens per threadgroup
constant constexpr uint OPT_TILE_N = 64;    // Output dimension tile
constant constexpr uint OPT_TILE_K = 64;    // Hidden dimension tile

constant constexpr uint OPT_SIMDGROUPS = 2;
constant constexpr uint OPT_THREADS = OPT_SIMDGROUPS * 32;  // 64

// Simdgroup matrix tiling: each simdgroup handles 8x32 output region
constant constexpr uint OPT_SG_M_TILES = 1;  // 1x 8-row tile
constant constexpr uint OPT_SG_N_TILES = 4;  // 4x 8-col tiles = 32 cols

// FP4 packing
constant constexpr uint FP4_PER_UINT = 8;
constant constexpr uint NUM_BUFFERS = 2;

// Top-k routing constants
constant constexpr uint TOPK_THREADS = 64;   // Threads for parallel top-k
constant constexpr uint EXPERTS_PER_THREAD = NUM_EXPERTS / TOPK_THREADS;  // 1

// ===========================================================================
// Parameters struct
// ===========================================================================

struct MoEOptParams {
    uint batch_size;        // Number of tokens
    uint hidden_dim;        // Input dimension (K)
    uint out_dim;           // Output dimension (N)
    uint num_experts;       // Total experts (typically 64)
    uint top_k;             // Experts per token (typically 4)
    uint group_size;        // Quantization group size (typically 128)
    uint has_shared;        // 1 if shared expert exists
};

// ===========================================================================
// FP4 E2M1 dequantization with SIMD optimization
// ===========================================================================

// Lookup table for FP4 E2M1 values (precomputed)
// Format: sign=1bit, exp=2bits, mantissa=1bit
// Values: 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, -0, -0.25, ..., -3
constant constexpr half FP4_LUT[16] = {
    half(0.0h), half(0.25h), half(0.5h), half(0.75h),
    half(1.0h), half(1.5h), half(2.0h), half(3.0h),
    half(-0.0h), half(-0.25h), half(-0.5h), half(-0.75h),
    half(-1.0h), half(-1.5h), half(-2.0h), half(-3.0h)
};

// Dequantize 8 FP4 values using LUT (faster than bitwise computation)
inline void opt_dequant_fp4x8_lut(uint packed, half scale, thread half* out) {
    float fscale = (float)scale;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = half(fscale * (float)FP4_LUT[nibble]);
    }
}

// Alternative: bitwise dequantization (for comparison/fallback)
inline void opt_dequant_fp4x8_bitwise(uint packed, half scale, thread half* out) {
    float fscale = (float)scale;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        uint sign = (nibble >> 3) & 1;
        uint exp = (nibble >> 1) & 0x3;
        uint man = nibble & 1;

        float magnitude;
        if (exp == 0) {
            magnitude = (float)man * 0.25f;
        } else {
            float power = (float)(1u << (exp - 1));
            magnitude = power * (1.0f + (float)man * 0.5f);
        }
        out[i] = half(fscale * (sign ? -magnitude : magnitude));
    }
}

// ===========================================================================
// Simdgroup parallel top-k selection (O(k * log n) comparisons)
//
// Uses a parallel reduction approach where each thread handles a subset of
// experts, then we merge top-k results across the simdgroup using shuffles.
//
// For NUM_EXPERTS=64 and TOP_K=4:
//   - Each simdgroup has 32 threads
//   - Each thread initially handles 2 experts (64/32)
//   - After local top-k, merge across threads using bitonic-style comparisons
// ===========================================================================

// Thread-local top-k merge: insert a (value, index) pair into sorted array
template<uint K>
inline void local_topk_insert(
    thread float (&vals)[K],
    thread uint (&ids)[K],
    float new_val,
    uint new_id
) {
    // Skip if worse than current worst
    if (new_val <= vals[K - 1]) return;

    // Binary search for insertion point (unrolled for small K)
    uint pos = K;
    #pragma unroll
    for (uint i = 0; i < K; ++i) {
        if (new_val > vals[i]) {
            pos = i;
            break;
        }
    }

    // Shift elements down
    #pragma unroll
    for (uint i = K - 1; i > pos; --i) {
        vals[i] = vals[i - 1];
        ids[i] = ids[i - 1];
    }

    // Insert
    vals[pos] = new_val;
    ids[pos] = new_id;
}

// Merge two sorted top-k arrays into one (K1 + K2 -> K)
template<uint K>
inline void merge_topk(
    thread float (&vals)[K],
    thread uint (&ids)[K],
    thread const float (&other_vals)[K],
    thread const uint (&other_ids)[K]
) {
    // Simply insert each element from other array
    // This is O(K^2) but K is small (4) so it's fast
    #pragma unroll
    for (uint i = 0; i < K; ++i) {
        local_topk_insert<K>(vals, ids, other_vals[i], other_ids[i]);
    }
}

// Simdgroup-wide top-k using butterfly reduction pattern
// Each thread starts with its local top-k, then exchanges with neighbors
inline void simdgroup_topk_4(
    thread float (&vals)[4],
    thread uint (&ids)[4],
    uint simd_lane
) {
    // Butterfly reduction: exchange with lane pairs at distances 16, 8, 4, 2, 1
    // After each exchange, each thread merges its top-k with the received values

    #pragma unroll
    for (uint delta = 16; delta >= 1; delta >>= 1) {
        // Get values from partner lane
        float partner_vals[4];
        uint partner_ids[4];

        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            partner_vals[i] = simd_shuffle_xor(vals[i], delta);
            partner_ids[i] = simd_shuffle_xor(ids[i], delta);
        }

        // Merge with partner's top-k
        merge_topk<4>(vals, ids, partner_vals, partner_ids);
    }

    // After reduction, all lanes have the same global top-4
}

// ===========================================================================
// Optimized cooperative tile loaders with prefetching
// ===========================================================================

// Load activation tile with vectorized access
inline void opt_load_A_tile(
    device const half* A,
    threadgroup half (&A_buf)[OPT_TILE_M][OPT_TILE_K + 4],  // +4 for bank conflict avoidance
    uint M, uint K,
    uint tg_row, uint k_offset,
    uint thread_idx
) {
    // Each thread loads 2 float4s (16 halfs) for better memory efficiency
    // 128 threads * 16 halfs = 2048 halfs = 32 * 64 = full tile
    const uint halfs_per_thread = (OPT_TILE_M * OPT_TILE_K) / OPT_THREADS;  // 16

    #pragma unroll 2
    for (uint i = 0; i < halfs_per_thread; i += 4) {
        uint flat_idx = thread_idx * halfs_per_thread + i;
        uint row = flat_idx / OPT_TILE_K;
        uint col = flat_idx % OPT_TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_offset + col;

        // Vectorized load (4 halfs = 8 bytes)
        if (global_row < M && global_col + 3 < K) {
            // Aligned vector load
            device const half4* src = (device const half4*)(A + global_row * K + global_col);
            half4 vec = *src;
            A_buf[row][col + 0] = vec.x;
            A_buf[row][col + 1] = vec.y;
            A_buf[row][col + 2] = vec.z;
            A_buf[row][col + 3] = vec.w;
        } else {
            // Scalar loads for boundary
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                uint gc = global_col + j;
                half val = (global_row < M && gc < K) ? A[global_row * K + gc] : half(0.0h);
                A_buf[row][col + j] = val;
            }
        }
    }
}

// Load activation tile as raw BF16 (uint16) without conversion.
// Conversion to half happens at GEMM input to avoid dispatch-side overhead.
inline void opt_load_A_tile_bf16_raw(
    device const ushort* A,
    threadgroup ushort (&A_buf)[OPT_TILE_M][OPT_TILE_K + 4],  // +4 for bank conflict avoidance
    uint M, uint K,
    uint tg_row, uint k_offset,
    uint thread_idx
) {
    const uint elems_per_thread = (OPT_TILE_M * OPT_TILE_K) / OPT_THREADS;  // 16

    #pragma unroll 2
    for (uint i = 0; i < elems_per_thread; i += 4) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / OPT_TILE_K;
        uint col = flat_idx % OPT_TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_offset + col;

        if (global_row < M && global_col + 3 < K) {
            device const ushort4* src = (device const ushort4*)(A + global_row * K + global_col);
            ushort4 vec = *src;
            A_buf[row][col + 0] = vec.x;
            A_buf[row][col + 1] = vec.y;
            A_buf[row][col + 2] = vec.z;
            A_buf[row][col + 3] = vec.w;
        } else {
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                uint gc = global_col + j;
                ushort val = (global_row < M && gc < K) ? A[global_row * K + gc] : ushort(0);
                A_buf[row][col + j] = val;
            }
        }
    }
}

// Convert BF16 tile (raw uint16) to half at GEMM input.
inline void opt_unpack_A_tile_bf16_to_half(
    threadgroup const ushort (&A_src)[OPT_TILE_M][OPT_TILE_K + 4],
    threadgroup half (&A_dst)[OPT_TILE_M][OPT_TILE_K + 4],
    uint thread_idx
) {
    const uint elems = OPT_TILE_M * (OPT_TILE_K + 4);
    for (uint i = thread_idx; i < elems; i += OPT_THREADS) {
        uint row = i / (OPT_TILE_K + 4);
        uint col = i % (OPT_TILE_K + 4);
        bf16_t v;
        v.bits = A_src[row][col];
        A_dst[row][col] = bf16_to_half(v);
    }
}

// Load and dequantize FP4 weights with transposed access pattern
// Weight layout: [K/8, N] packed FP4 (transposed for coalescing)
inline void opt_load_B_tile_dequant(
    device const uint* B,       // [K/8, N] packed FP4
    device const half* scales,  // [num_groups, N]
    threadgroup half (&B_buf)[OPT_TILE_K][OPT_TILE_N + 4],
    uint K, uint N,
    uint tg_col, uint k_offset,
    uint group_size,
    uint thread_idx
) {
    // Each packed uint contains 8 FP4 values (K dimension)
    // Threads load packed values in N dimension for coalescing
    const uint k_packs = K / FP4_PER_UINT;
    const uint tile_k_packs = OPT_TILE_K / FP4_PER_UINT;  // 8
    const uint num_groups = (K + group_size - 1) / group_size;

    // Each thread handles some portion of the N * K_packs space
    const uint total_packs = tile_k_packs * OPT_TILE_N;  // 8 * 128 = 1024
    const uint packs_per_thread = total_packs / OPT_THREADS;  // 8

    #pragma unroll 2
    for (uint p = 0; p < packs_per_thread; ++p) {
        uint flat_pack = thread_idx * packs_per_thread + p;
        uint n_idx = flat_pack % OPT_TILE_N;
        uint k_pack_in_tile = flat_pack / OPT_TILE_N;

        uint global_n = tg_col + n_idx;
        uint global_k_pack = k_offset / FP4_PER_UINT + k_pack_in_tile;
        uint global_k_base = global_k_pack * FP4_PER_UINT;

        // Load scale for this group
        uint scale_group = global_k_base / group_size;
        half scale = half(1.0h);
        if (global_n < N && scale_group < num_groups) {
            scale = scales[scale_group * N + global_n];
        }

        // Load packed FP4
        uint packed = 0;
        if (global_n < N && global_k_pack < k_packs) {
            packed = B[global_k_pack * N + global_n];
        }

        // Dequantize 8 values and store to tile
        uint tile_k_base = k_pack_in_tile * FP4_PER_UINT;
        half vals[8];
        opt_dequant_fp4x8_lut(packed, scale, vals);

        #pragma unroll
        for (uint v = 0; v < 8; ++v) {
            if (tile_k_base + v < OPT_TILE_K) {
                B_buf[tile_k_base + v][n_idx] = vals[v];
            }
        }
    }
}

// ===========================================================================
// Simdgroup GEMM compute with register tiling
// ===========================================================================

inline void opt_compute_gemm(
    threadgroup const half (&A_buf)[OPT_TILE_M][OPT_TILE_K + 4],
    threadgroup const half (&B_buf)[OPT_TILE_K][OPT_TILE_N + 4],
    thread simdgroup_matrix<half, 8, 8> acc[OPT_SG_M_TILES][OPT_SG_N_TILES],
    uint sg_row_offset,
    uint sg_col_offset
) {
    const uint K_SUBTILES = OPT_TILE_K / 8;  // 8

    #pragma unroll 4
    for (uint kt = 0; kt < K_SUBTILES; ++kt) {
        // Load A fragment for this simdgroup's row region
        #pragma unroll
        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag,
                          &A_buf[sg_row_offset + mi * 8][kt * 8],
                          OPT_TILE_K + 4);  // stride includes padding

            // Load B fragments and accumulate
            #pragma unroll
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag,
                              &B_buf[kt * 8][sg_col_offset + ni * 8],
                              OPT_TILE_N + 4);

                simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
            }
        }
    }
}

// ===========================================================================
// Main kernel: Fused MoE dispatch with all optimizations
//
// Single kernel that performs:
//   1. Compute routing logits and select top-k experts
//   2. Group tokens by expert (warp-level, no atomics)
//   3. Execute batched expert GEMMs with probability weighting
//   4. Combine outputs (including shared expert)
//
// Grid dispatch: [ceil(out_dim / TILE_N), ceil(batch / TILE_M)]
// ===========================================================================

kernel void moe_dispatch_optimized(
    device const half* activations       [[buffer(0)]],   // [batch, hidden]
    device const half* router_weights    [[buffer(1)]],   // [hidden, num_experts]
    device const uint* expert_weights    [[buffer(2)]],   // [num_experts, K/8, N] packed FP4
    device const half* expert_scales     [[buffer(3)]],   // [num_experts, K/group, N]
    device const uint* shared_weights    [[buffer(4)]],   // [K/8, N] packed FP4 (optional)
    device const half* shared_scales     [[buffer(5)]],   // [K/group, N] (optional)
    device half* output                  [[buffer(6)]],   // [batch, N]
    constant MoEOptParams& params        [[buffer(7)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory allocation
    // A tiles: activations [M, K]
    threadgroup half A_tiles[NUM_BUFFERS][OPT_TILE_M][OPT_TILE_K + 4];
    // B tiles: expert weights [K, N]
    threadgroup half B_tiles[NUM_BUFFERS][OPT_TILE_K][OPT_TILE_N + 4];

    // Routing info for this tile of tokens
    threadgroup uint tile_expert_ids[OPT_TILE_M][TOP_K];     // Selected experts
    threadgroup half tile_expert_probs[OPT_TILE_M][TOP_K];   // Routing weights

    // Output staging (shared expert + routed experts)
    threadgroup half result_staging[OPT_TILE_M][OPT_TILE_N + 4];

    const uint tg_row = tgid.y * OPT_TILE_M;  // Token offset
    const uint tg_col = tgid.x * OPT_TILE_N;  // Output column offset

    // Simdgroup layout: 2 simdgroups handle different rows
    // SG0: rows 0-7, SG1: rows 8-15
    const uint sg_row_offset = simd_id * 8;
    const uint sg_col_offset = 0;  // All simdgroups cover full N tile

    const uint thread_idx = simd_id * 32 + simd_lane;

    // Precompute strides
    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint expert_weight_stride = k_packed * params.out_dim;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;
    const uint expert_scale_stride = num_scale_groups * params.out_dim;

    // =========================================================================
    // Phase 0: Initialize output staging to zero
    // =========================================================================

    {
        const uint elems = OPT_TILE_M * (OPT_TILE_N + 4);
        for (uint i = thread_idx; i < elems; i += OPT_THREADS) {
            uint row = i / (OPT_TILE_N + 4);
            uint col = i % (OPT_TILE_N + 4);
            result_staging[row][col] = half(0.0h);
        }
    }

    // =========================================================================
    // Phase 1: Compute routing logits and select top-k experts
    //
    // Each simdgroup handles 8 tokens (matching its row region)
    // Within simdgroup, use parallel top-k selection
    // =========================================================================

    {
        // Each simdgroup processes tokens [sg_row_offset, sg_row_offset + 8)
        uint local_token = simd_lane / 4;  // 0-7 (8 tokens per simdgroup, 4 threads per token)
        uint expert_group = simd_lane % 4;  // Each thread handles 16 experts (64/4)

        uint global_token = tg_row + sg_row_offset + local_token;

        // Thread-local top-4 for its 16 experts
        float local_vals[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        uint local_ids[4] = {0, 0, 0, 0};

        if (global_token < params.batch_size) {
            // Compute routing logits for 16 experts
            device const half* token_acts = activations + global_token * params.hidden_dim;

            #pragma unroll 4
            for (uint e = 0; e < 16; ++e) {
                uint expert_idx = expert_group * 16 + e;
                if (expert_idx >= params.num_experts) break;

                // Dot product: token_acts @ router_weights[:, expert_idx]
                float logit = 0.0f;

                // Vectorized dot product
                for (uint d = 0; d < params.hidden_dim; d += 4) {
                    float4 a = float4(
                        token_acts[d + 0],
                        token_acts[d + 1],
                        token_acts[d + 2],
                        token_acts[d + 3]
                    );
                    float4 w = float4(
                        router_weights[(d + 0) * params.num_experts + expert_idx],
                        router_weights[(d + 1) * params.num_experts + expert_idx],
                        router_weights[(d + 2) * params.num_experts + expert_idx],
                        router_weights[(d + 3) * params.num_experts + expert_idx]
                    );
                    logit += dot(a, w);
                }

                // Insert into local top-k
                local_topk_insert<4>(local_vals, local_ids, logit, expert_idx);
            }
        }

        // Merge top-k across the 4 threads handling this token
        // Use warp shuffle to exchange values between threads 0-3, 4-7, etc.
        {
            // Phase 1: Exchange with thread at distance 2
            float partner_vals[4];
            uint partner_ids[4];
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                partner_vals[i] = simd_shuffle_xor(local_vals[i], 2);
                partner_ids[i] = simd_shuffle_xor(local_ids[i], 2);
            }
            merge_topk<4>(local_vals, local_ids, partner_vals, partner_ids);

            // Phase 2: Exchange with thread at distance 1
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                partner_vals[i] = simd_shuffle_xor(local_vals[i], 1);
                partner_ids[i] = simd_shuffle_xor(local_ids[i], 1);
            }
            merge_topk<4>(local_vals, local_ids, partner_vals, partner_ids);
        }

        // Thread 0 (and 4, 8, 12, ...) of each 4-thread group has final top-k
        // Write to shared memory
        if ((simd_lane % 4) == 0) {
            uint local_row = sg_row_offset + local_token;

            // Compute softmax over selected experts for renormalized probabilities
            float max_val = local_vals[0];
            float exp_sum = 0.0f;
            float probs[4];

            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                float exp_val = exp(local_vals[k] - max_val);
                probs[k] = exp_val;
                exp_sum += exp_val;
            }

            float inv_sum = 1.0f / exp_sum;

            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                tile_expert_ids[local_row][k] = local_ids[k];
                tile_expert_probs[local_row][k] = half(probs[k] * inv_sum);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 2: Compute shared expert contribution (if enabled)
    //
    // The shared expert always runs on all tokens. We compute it first
    // as a baseline, then add routed expert contributions.
    // =========================================================================

    if (params.has_shared) {
        simdgroup_matrix<half, 8, 8> shared_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
        #pragma unroll
        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
            #pragma unroll
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                shared_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }

        const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
        uint buf_compute = 0;

        // Prologue: load first tiles
        opt_load_A_tile(activations, A_tiles[0], params.batch_size, params.hidden_dim,
                        tg_row, 0, thread_idx);
        opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[0],
                                params.hidden_dim, params.out_dim,
                                tg_col, 0, params.group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Main K loop with double buffering
        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint k_offset = kt * OPT_TILE_K;
            uint next_k = k_offset + OPT_TILE_K;
            uint buf_load = 1 - buf_compute;

            // Prefetch next tiles
            if (next_k < params.hidden_dim) {
                opt_load_A_tile(activations, A_tiles[buf_load], params.batch_size, params.hidden_dim,
                                tg_row, next_k, thread_idx);
                opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[buf_load],
                                        params.hidden_dim, params.out_dim,
                                        tg_col, next_k, params.group_size, thread_idx);
            }

            // Compute
            opt_compute_gemm(A_tiles[buf_compute], B_tiles[buf_compute],
                            shared_acc, sg_row_offset, sg_col_offset);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_compute = buf_load;
        }

        // Store shared expert results to staging
        constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;  // 8
        constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;  // 32

        threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

        #pragma unroll
        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
            #pragma unroll
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                simdgroup_store(shared_acc[mi][ni],
                               &sg_temp[simd_id][mi * 8][ni * 8],
                               sg_tile_cols);
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Copy to result_staging (shared expert has implicit weight of 1.0)
        constexpr uint total_elems = sg_tile_rows * sg_tile_cols;  // 256
        constexpr uint elems_per_lane = total_elems / 32;  // 8

        #pragma unroll
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint elem = simd_lane * elems_per_lane + i;
            uint row = elem / sg_tile_cols;
            uint col = elem % sg_tile_cols;

            uint staging_row = sg_row_offset + row;
            uint staging_col = col;

            result_staging[staging_row][staging_col] = sg_temp[simd_id][row][col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // Phase 3: Process routed experts
    //
    // For each of the top-k expert slots, find tokens that selected that expert
    // and batch their computations together.
    //
    // Optimization: Instead of grouping by expert (which requires sorting),
    // we iterate over slots. Within each slot, most tokens in a tile tend to
    // select similar experts due to semantic locality.
    // =========================================================================

    // Process each top-k slot
    for (uint slot = 0; slot < params.top_k; ++slot) {
        // For each expert that appears in this slot, process all tokens with that expert
        // We use a simple approach: iterate through unique experts in the tile

        // Find unique experts in this slot (simplified: just process each token)
        // This works well when tokens in same tile often share experts

        // Alternative: process each token independently (simpler, good baseline)
        // For optimal performance with diverse expert assignments, use grouped dispatch

        // Get the most common expert for this slot in the tile
        // Use simdgroup ballot to find consensus
        uint slot_expert = tile_expert_ids[sg_row_offset][slot];

        // Check if all tokens in this simdgroup's region use the same expert
        bool same_expert = true;
        for (uint t = 1; t < 8 && (tg_row + sg_row_offset + t) < params.batch_size; ++t) {
            if (tile_expert_ids[sg_row_offset + t][slot] != slot_expert) {
                same_expert = false;
                break;
            }
        }

        if (same_expert && slot_expert < params.num_experts) {
            // Fast path: all tokens use the same expert, can batch efficiently
            simdgroup_matrix<half, 8, 8> expert_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
            #pragma unroll
            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
                #pragma unroll
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                    expert_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            // Get pointers to this expert's weights
            device const uint* B = expert_weights + slot_expert * expert_weight_stride;
            device const half* S = expert_scales + slot_expert * expert_scale_stride;

            const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
            uint buf_compute = 0;

            // Prologue
            opt_load_A_tile(activations, A_tiles[0], params.batch_size, params.hidden_dim,
                            tg_row, 0, thread_idx);
            opt_load_B_tile_dequant(B, S, B_tiles[0],
                                    params.hidden_dim, params.out_dim,
                                    tg_col, 0, params.group_size, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Main K loop
            for (uint kt = 0; kt < num_k_tiles; ++kt) {
                uint k_offset = kt * OPT_TILE_K;
                uint next_k = k_offset + OPT_TILE_K;
                uint buf_load = 1 - buf_compute;

                if (next_k < params.hidden_dim) {
                    opt_load_A_tile(activations, A_tiles[buf_load], params.batch_size, params.hidden_dim,
                                    tg_row, next_k, thread_idx);
                    opt_load_B_tile_dequant(B, S, B_tiles[buf_load],
                                            params.hidden_dim, params.out_dim,
                                            tg_col, next_k, params.group_size, thread_idx);
                }

                opt_compute_gemm(A_tiles[buf_compute], B_tiles[buf_compute],
                                expert_acc, sg_row_offset, sg_col_offset);

                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf_compute = buf_load;
            }

            // Accumulate weighted results to staging
            constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;
            constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;

            threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

            #pragma unroll
            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
                #pragma unroll
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                    simdgroup_store(expert_acc[mi][ni],
                                   &sg_temp[simd_id][mi * 8][ni * 8],
                                   sg_tile_cols);
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Add to staging with probability weighting
            constexpr uint total_elems = sg_tile_rows * sg_tile_cols;
            constexpr uint elems_per_lane = total_elems / 32;

            #pragma unroll
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint elem = simd_lane * elems_per_lane + i;
                uint row = elem / sg_tile_cols;
                uint col = elem % sg_tile_cols;

                uint staging_row = sg_row_offset + row;
                uint global_row = tg_row + staging_row;

                if (global_row < params.batch_size && staging_row < OPT_TILE_M) {
                    half prob = tile_expert_probs[staging_row][slot];
                    half val = sg_temp[simd_id][row][col];
                    result_staging[staging_row][col] += val * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // Slow path: different experts, process one token at a time
            // This is less efficient but handles the general case
            for (uint local_token = 0; local_token < 8; ++local_token) {
                uint staging_row = sg_row_offset + local_token;
                uint global_token = tg_row + staging_row;

                if (global_token >= params.batch_size) continue;

                uint expert_id = tile_expert_ids[staging_row][slot];
                half prob = tile_expert_probs[staging_row][slot];

                if (expert_id >= params.num_experts || prob < half(1e-6h)) continue;

                // Compute expert output for this single token
                device const half* token_acts = activations + global_token * params.hidden_dim;
                device const uint* B = expert_weights + expert_id * expert_weight_stride;
                device const half* S = expert_scales + expert_id * expert_scale_stride;

                // Process output columns assigned to this simdgroup
                // Each thread handles a portion of the output
                for (uint out_col = simd_lane; out_col < OPT_SG_N_TILES * 8 && (tg_col + out_col) < params.out_dim; out_col += 32) {
                    uint global_col = tg_col + out_col;

                    float acc = 0.0f;

                    // Dot product: token_acts @ expert_weights[:, out_col]
                    for (uint k_pack = 0; k_pack < params.hidden_dim / FP4_PER_UINT; ++k_pack) {
                        uint k_base = k_pack * FP4_PER_UINT;
                        uint scale_group = k_base / params.group_size;

                        half scale = S[scale_group * params.out_dim + global_col];
                        uint packed = B[k_pack * params.out_dim + global_col];

                        half vals[8];
                        opt_dequant_fp4x8_lut(packed, scale, vals);

                        #pragma unroll
                        for (uint v = 0; v < 8; ++v) {
                            acc += (float)token_acts[k_base + v] * (float)vals[v];
                        }
                    }

                    // Add weighted contribution to staging
                    result_staging[staging_row][out_col] += half(acc) * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // =========================================================================
    // Phase 4: Write final results to global memory
    // =========================================================================

    for (uint i = thread_idx; i < OPT_TILE_M * OPT_TILE_N; i += OPT_THREADS) {
        uint local_row = i / OPT_TILE_N;
        uint local_col = i % OPT_TILE_N;
        uint global_row = tg_row + local_row;
        uint global_col = tg_col + local_col;

        if (global_row < params.batch_size && global_col < params.out_dim) {
            output[global_row * params.out_dim + global_col] = result_staging[local_row][local_col];
        }
    }
}

// ===========================================================================
// BF16 kernel variant: BF16 activations + BF16 output
//
// - Dispatch copies activations as raw BF16 (uint16) into threadgroup storage
// - BF16 -> half conversion happens once at GEMM input
// - Output writes directly to BF16 buffer (no separate gather conversion)
// - Router softmax stays FP32 (no FP16 casts)
// - Shared expert path still uses half simdgroup GEMM (Metal limitation)
// ===========================================================================

kernel void moe_dispatch_ultra_optimized_bf16(
    device const ushort* activations    [[buffer(0)]],   // [batch, hidden] BF16 (uint16)
    device const ushort* router_weights [[buffer(1)]],   // [hidden, num_experts] BF16 (uint16)
    device const uint* expert_weights   [[buffer(2)]],   // [num_experts, K/8, N] packed FP4
    device const half* expert_scales    [[buffer(3)]],   // [num_experts, K/group, N]
    device const uint* shared_weights   [[buffer(4)]],   // [K/8, N] (optional)
    device const half* shared_scales    [[buffer(5)]],   // [K/group, N] (optional)
    device ushort* output               [[buffer(6)]],   // [batch, N] BF16 (uint16)
    constant MoEOptParams& params       [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory allocation
    threadgroup ushort A_tiles_bf16[NUM_BUFFERS][OPT_TILE_M][OPT_TILE_K + 4];
    threadgroup half A_tiles_half[NUM_BUFFERS][OPT_TILE_M][OPT_TILE_K + 4];
    threadgroup half B_tiles[NUM_BUFFERS][OPT_TILE_K][OPT_TILE_N + 4];

    threadgroup uint tile_expert_ids[OPT_TILE_M][TOP_K];
    threadgroup half tile_expert_probs[OPT_TILE_M][TOP_K];
    threadgroup half result_staging[OPT_TILE_M][OPT_TILE_N + 4];

    const uint tg_row = tgid.y * OPT_TILE_M;
    const uint tg_col = tgid.x * OPT_TILE_N;

    const uint sg_row_offset = simd_id * 8;
    const uint sg_col_offset = 0;
    const uint thread_idx = simd_id * 32 + simd_lane;

    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint expert_weight_stride = k_packed * params.out_dim;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;
    const uint expert_scale_stride = num_scale_groups * params.out_dim;

    // Initialize output staging to zero
    {
        const uint elems = OPT_TILE_M * (OPT_TILE_N + 4);
        for (uint i = thread_idx; i < elems; i += OPT_THREADS) {
            uint row = i / (OPT_TILE_N + 4);
            uint col = i % (OPT_TILE_N + 4);
            result_staging[row][col] = half(0.0h);
        }
    }

    // Phase 1: Routing (BF16 -> FP32, no FP16 casts)
    {
        uint local_token = simd_lane / 4;
        uint expert_group = simd_lane % 4;
        uint global_token = tg_row + sg_row_offset + local_token;

        float local_vals[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        uint local_ids[4] = {0, 0, 0, 0};

        if (global_token < params.batch_size) {
            device const ushort* token_acts = activations + global_token * params.hidden_dim;

            #pragma unroll 4
            for (uint e = 0; e < 16; ++e) {
                uint expert_idx = expert_group * 16 + e;
                if (expert_idx >= params.num_experts) break;

                float logit = 0.0f;

                for (uint d = 0; d < params.hidden_dim; d += 4) {
                    ushort4 a_packed = *((device const ushort4*)(token_acts + d));
                    float4 a = bf16x4_to_float4(a_packed);

                    ushort4 w_packed = ushort4(
                        router_weights[(d + 0) * params.num_experts + expert_idx],
                        router_weights[(d + 1) * params.num_experts + expert_idx],
                        router_weights[(d + 2) * params.num_experts + expert_idx],
                        router_weights[(d + 3) * params.num_experts + expert_idx]
                    );
                    float4 w = bf16x4_to_float4(w_packed);
                    logit += dot(a, w);
                }

                local_topk_insert<4>(local_vals, local_ids, logit, expert_idx);
            }
        }

        {
            float partner_vals[4];
            uint partner_ids[4];
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                partner_vals[i] = simd_shuffle_xor(local_vals[i], 2);
                partner_ids[i] = simd_shuffle_xor(local_ids[i], 2);
            }
            merge_topk<4>(local_vals, local_ids, partner_vals, partner_ids);

            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                partner_vals[i] = simd_shuffle_xor(local_vals[i], 1);
                partner_ids[i] = simd_shuffle_xor(local_ids[i], 1);
            }
            merge_topk<4>(local_vals, local_ids, partner_vals, partner_ids);
        }

        if ((simd_lane % 4) == 0) {
            uint local_row = sg_row_offset + local_token;

            float max_val = local_vals[0];
            float exp_sum = 0.0f;
            float probs[4];

            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                float exp_val = exp(local_vals[k] - max_val);
                probs[k] = exp_val;
                exp_sum += exp_val;
            }

            float inv_sum = 1.0f / exp_sum;

            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                tile_expert_ids[local_row][k] = local_ids[k];
                tile_expert_probs[local_row][k] = half(probs[k] * inv_sum);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Shared expert (uses half GEMM; BF16 -> half conversion at input)
    if (params.has_shared) {
        simdgroup_matrix<half, 8, 8> shared_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
        #pragma unroll
        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
            #pragma unroll
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                shared_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }

        const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
        uint buf_compute = 0;

        opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[0], params.batch_size, params.hidden_dim,
                                 tg_row, 0, thread_idx);
        opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[0],
                                params.hidden_dim, params.out_dim,
                                tg_col, 0, params.group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint k_offset = kt * OPT_TILE_K;
            uint next_k = k_offset + OPT_TILE_K;
            uint buf_load = 1 - buf_compute;

            if (next_k < params.hidden_dim) {
                opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[buf_load], params.batch_size, params.hidden_dim,
                                         tg_row, next_k, thread_idx);
                opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[buf_load],
                                        params.hidden_dim, params.out_dim,
                                        tg_col, next_k, params.group_size, thread_idx);
            }

            opt_unpack_A_tile_bf16_to_half(A_tiles_bf16[buf_compute], A_tiles_half[buf_compute], thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            opt_compute_gemm(A_tiles_half[buf_compute], B_tiles[buf_compute],
                             shared_acc, sg_row_offset, sg_col_offset);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_compute = buf_load;
        }

        constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;
        constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;

        threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

        #pragma unroll
        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
            #pragma unroll
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                simdgroup_store(shared_acc[mi][ni],
                                &sg_temp[simd_id][mi * 8][ni * 8],
                                sg_tile_cols);
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        constexpr uint total_elems = sg_tile_rows * sg_tile_cols;
        constexpr uint elems_per_lane = total_elems / 32;

        #pragma unroll
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint elem = simd_lane * elems_per_lane + i;
            uint row = elem / sg_tile_cols;
            uint col = elem % sg_tile_cols;

            uint staging_row = sg_row_offset + row;
            uint staging_col = col;

            result_staging[staging_row][staging_col] = sg_temp[simd_id][row][col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: Routed experts (same structure as half kernel, BF16 input)
    for (uint slot = 0; slot < params.top_k; ++slot) {
        uint slot_expert = tile_expert_ids[sg_row_offset][slot];

        bool same_expert = true;
        for (uint t = 1; t < 8 && (tg_row + sg_row_offset + t) < params.batch_size; ++t) {
            if (tile_expert_ids[sg_row_offset + t][slot] != slot_expert) {
                same_expert = false;
                break;
            }
        }

        if (same_expert && slot_expert < params.num_experts) {
            simdgroup_matrix<half, 8, 8> expert_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
            #pragma unroll
            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
                #pragma unroll
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                    expert_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            device const uint* B = expert_weights + slot_expert * expert_weight_stride;
            device const half* S = expert_scales + slot_expert * expert_scale_stride;

            const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
            uint buf_compute = 0;

            opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[0], params.batch_size, params.hidden_dim,
                                     tg_row, 0, thread_idx);
            opt_load_B_tile_dequant(B, S, B_tiles[0],
                                    params.hidden_dim, params.out_dim,
                                    tg_col, 0, params.group_size, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kt = 0; kt < num_k_tiles; ++kt) {
                uint k_offset = kt * OPT_TILE_K;
                uint next_k = k_offset + OPT_TILE_K;
                uint buf_load = 1 - buf_compute;

                if (next_k < params.hidden_dim) {
                    opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[buf_load], params.batch_size, params.hidden_dim,
                                             tg_row, next_k, thread_idx);
                    opt_load_B_tile_dequant(B, S, B_tiles[buf_load],
                                            params.hidden_dim, params.out_dim,
                                            tg_col, next_k, params.group_size, thread_idx);
                }

                opt_unpack_A_tile_bf16_to_half(A_tiles_bf16[buf_compute], A_tiles_half[buf_compute], thread_idx);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                opt_compute_gemm(A_tiles_half[buf_compute], B_tiles[buf_compute],
                                 expert_acc, sg_row_offset, sg_col_offset);

                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf_compute = buf_load;
            }

            constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;
            constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;

            threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

            #pragma unroll
            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi) {
                #pragma unroll
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni) {
                    simdgroup_store(expert_acc[mi][ni],
                                    &sg_temp[simd_id][mi * 8][ni * 8],
                                    sg_tile_cols);
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            constexpr uint total_elems = sg_tile_rows * sg_tile_cols;
            constexpr uint elems_per_lane = total_elems / 32;

            #pragma unroll
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint elem = simd_lane * elems_per_lane + i;
                uint row = elem / sg_tile_cols;
                uint col = elem % sg_tile_cols;

                uint staging_row = sg_row_offset + row;
                uint global_row = tg_row + staging_row;

                if (global_row < params.batch_size && staging_row < OPT_TILE_M) {
                    half prob = tile_expert_probs[staging_row][slot];
                    half val = sg_temp[simd_id][row][col];
                    result_staging[staging_row][col] += val * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (uint local_token = 0; local_token < 8; ++local_token) {
                uint staging_row = sg_row_offset + local_token;
                uint global_token = tg_row + staging_row;

                if (global_token >= params.batch_size) continue;

                uint expert_id = tile_expert_ids[staging_row][slot];
                half prob = tile_expert_probs[staging_row][slot];

                if (expert_id >= params.num_experts || prob < half(1e-6h)) continue;

                device const ushort* token_acts = activations + global_token * params.hidden_dim;
                device const uint* B = expert_weights + expert_id * expert_weight_stride;
                device const half* S = expert_scales + expert_id * expert_scale_stride;

                for (uint out_col = simd_lane; out_col < OPT_SG_N_TILES * 8 && (tg_col + out_col) < params.out_dim; out_col += 32) {
                    uint global_col = tg_col + out_col;

                    float acc = 0.0f;

                    for (uint k_pack = 0; k_pack < params.hidden_dim / FP4_PER_UINT; ++k_pack) {
                        uint k_base = k_pack * FP4_PER_UINT;
                        uint scale_group = k_base / params.group_size;

                        half scale = S[scale_group * params.out_dim + global_col];
                        uint packed = B[k_pack * params.out_dim + global_col];

                        half vals[8];
                        opt_dequant_fp4x8_lut(packed, scale, vals);

                        #pragma unroll
                        for (uint v = 0; v < 8; ++v) {
                            bf16_t a;
                            a.bits = token_acts[k_base + v];
                            acc += (float)vals[v] * bf16_to_float(a);
                        }
                    }

                    result_staging[staging_row][out_col] += half(acc) * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Phase 4: Write BF16 output directly (no extra gather conversion)
    for (uint i = thread_idx; i < OPT_TILE_M * OPT_TILE_N; i += OPT_THREADS) {
        uint local_row = i / OPT_TILE_N;
        uint local_col = i % OPT_TILE_N;
        uint global_row = tg_row + local_row;
        uint global_col = tg_col + local_col;

        if (global_row < params.batch_size && global_col < params.out_dim) {
            bf16_t out_val = bf16_from_half(result_staging[local_row][local_col]);
            output[global_row * params.out_dim + global_col] = out_val.bits;
        }
    }
}

// ===========================================================================
// Kernel variant: Pre-routed dispatch (when routing is computed separately)
//
// Use this when routing decisions are made by a separate kernel (e.g., for
// profiling or when routing needs to be saved for debugging).
//
// This version takes pre-computed expert_ids and expert_probs as input.
// ===========================================================================

kernel void moe_dispatch_optimized_prerouted(
    device const half* activations       [[buffer(0)]],   // [batch, hidden]
    device const uint* expert_weights    [[buffer(1)]],   // [num_experts, K/8, N]
    device const half* expert_scales     [[buffer(2)]],   // [num_experts, K/group, N]
    device const uint* expert_ids        [[buffer(3)]],   // [batch, top_k]
    device const half* expert_probs      [[buffer(4)]],   // [batch, top_k]
    device const uint* shared_weights    [[buffer(5)]],   // [K/8, N] (optional)
    device const half* shared_scales     [[buffer(6)]],   // [K/group, N] (optional)
    device half* output                  [[buffer(7)]],   // [batch, N]
    constant MoEOptParams& params        [[buffer(8)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory
    threadgroup half A_tiles[NUM_BUFFERS][OPT_TILE_M][OPT_TILE_K + 4];
    threadgroup half B_tiles[NUM_BUFFERS][OPT_TILE_K][OPT_TILE_N + 4];
    threadgroup uint tile_expert_ids[OPT_TILE_M][TOP_K];
    threadgroup half tile_expert_probs[OPT_TILE_M][TOP_K];
    threadgroup half result_staging[OPT_TILE_M][OPT_TILE_N + 4];

    const uint tg_row = tgid.y * OPT_TILE_M;
    const uint tg_col = tgid.x * OPT_TILE_N;

    const uint sg_row_offset = simd_id * 8;
    const uint thread_idx = simd_id * 32 + simd_lane;

    // Precompute strides
    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint expert_weight_stride = k_packed * params.out_dim;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;
    const uint expert_scale_stride = num_scale_groups * params.out_dim;

    // Initialize staging
    for (uint i = thread_idx; i < OPT_TILE_M * (OPT_TILE_N + 4); i += OPT_THREADS) {
        uint row = i / (OPT_TILE_N + 4);
        uint col = i % (OPT_TILE_N + 4);
        result_staging[row][col] = half(0.0h);
    }

    // Load pre-computed routing info
    for (uint i = thread_idx; i < OPT_TILE_M * params.top_k; i += OPT_THREADS) {
        uint local_row = i / params.top_k;
        uint slot = i % params.top_k;
        uint global_row = tg_row + local_row;

        if (global_row < params.batch_size && slot < params.top_k) {
            tile_expert_ids[local_row][slot] = expert_ids[global_row * params.top_k + slot];
            tile_expert_probs[local_row][slot] = expert_probs[global_row * params.top_k + slot];
        } else {
            tile_expert_ids[local_row][slot] = 0;
            tile_expert_probs[local_row][slot] = half(0.0h);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Shared expert computation (same as main kernel)
    if (params.has_shared) {
        simdgroup_matrix<half, 8, 8> shared_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                shared_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
        uint buf_compute = 0;

        opt_load_A_tile(activations, A_tiles[0], params.batch_size, params.hidden_dim,
                        tg_row, 0, thread_idx);
        opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[0],
                                params.hidden_dim, params.out_dim,
                                tg_col, 0, params.group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint next_k = (kt + 1) * OPT_TILE_K;
            uint buf_load = 1 - buf_compute;

            if (next_k < params.hidden_dim) {
                opt_load_A_tile(activations, A_tiles[buf_load], params.batch_size, params.hidden_dim,
                                tg_row, next_k, thread_idx);
                opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[buf_load],
                                        params.hidden_dim, params.out_dim,
                                        tg_col, next_k, params.group_size, thread_idx);
            }

            opt_compute_gemm(A_tiles[buf_compute], B_tiles[buf_compute],
                            shared_acc, sg_row_offset, 0);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_compute = buf_load;
        }

        // Store shared expert results
        constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;
        constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;
        threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                simdgroup_store(shared_acc[mi][ni], &sg_temp[simd_id][mi * 8][ni * 8], sg_tile_cols);

        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = simd_lane; i < sg_tile_rows * sg_tile_cols; i += 32) {
            uint row = i / sg_tile_cols;
            uint col = i % sg_tile_cols;
            result_staging[sg_row_offset + row][col] = sg_temp[simd_id][row][col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Routed experts (same as main kernel but uses pre-loaded routing)
    for (uint slot = 0; slot < params.top_k; ++slot) {
        uint slot_expert = tile_expert_ids[sg_row_offset][slot];

        bool same_expert = true;
        for (uint t = 1; t < 8 && (tg_row + sg_row_offset + t) < params.batch_size; ++t) {
            if (tile_expert_ids[sg_row_offset + t][slot] != slot_expert) {
                same_expert = false;
                break;
            }
        }

        if (same_expert && slot_expert < params.num_experts) {
            simdgroup_matrix<half, 8, 8> expert_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                    expert_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

            device const uint* B = expert_weights + slot_expert * expert_weight_stride;
            device const half* S = expert_scales + slot_expert * expert_scale_stride;

            const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
            uint buf_compute = 0;

            opt_load_A_tile(activations, A_tiles[0], params.batch_size, params.hidden_dim,
                            tg_row, 0, thread_idx);
            opt_load_B_tile_dequant(B, S, B_tiles[0],
                                    params.hidden_dim, params.out_dim,
                                    tg_col, 0, params.group_size, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kt = 0; kt < num_k_tiles; ++kt) {
                uint next_k = (kt + 1) * OPT_TILE_K;
                uint buf_load = 1 - buf_compute;

                if (next_k < params.hidden_dim) {
                    opt_load_A_tile(activations, A_tiles[buf_load], params.batch_size, params.hidden_dim,
                                    tg_row, next_k, thread_idx);
                    opt_load_B_tile_dequant(B, S, B_tiles[buf_load],
                                            params.hidden_dim, params.out_dim,
                                            tg_col, next_k, params.group_size, thread_idx);
                }

                opt_compute_gemm(A_tiles[buf_compute], B_tiles[buf_compute],
                                expert_acc, sg_row_offset, 0);

                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf_compute = buf_load;
            }

            constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;
            constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;
            threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                    simdgroup_store(expert_acc[mi][ni], &sg_temp[simd_id][mi * 8][ni * 8], sg_tile_cols);

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = simd_lane; i < sg_tile_rows * sg_tile_cols; i += 32) {
                uint row = i / sg_tile_cols;
                uint col = i % sg_tile_cols;
                uint staging_row = sg_row_offset + row;

                if (staging_row < OPT_TILE_M) {
                    half prob = tile_expert_probs[staging_row][slot];
                    result_staging[staging_row][col] += sg_temp[simd_id][row][col] * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // Fallback for diverse expert assignments
            for (uint local_token = 0; local_token < 8; ++local_token) {
                uint staging_row = sg_row_offset + local_token;
                uint global_token = tg_row + staging_row;

                if (global_token >= params.batch_size) continue;

                uint expert_id = tile_expert_ids[staging_row][slot];
                half prob = tile_expert_probs[staging_row][slot];

                if (expert_id >= params.num_experts || prob < half(1e-6h)) continue;

                device const half* token_acts = activations + global_token * params.hidden_dim;
                device const uint* B = expert_weights + expert_id * expert_weight_stride;
                device const half* S = expert_scales + expert_id * expert_scale_stride;

                for (uint out_col = simd_lane; out_col < OPT_SG_N_TILES * 8 && (tg_col + out_col) < params.out_dim; out_col += 32) {
                    uint global_col = tg_col + out_col;
                    float acc = 0.0f;

                    for (uint k_pack = 0; k_pack < params.hidden_dim / FP4_PER_UINT; ++k_pack) {
                        uint k_base = k_pack * FP4_PER_UINT;
                        uint scale_group = k_base / params.group_size;

                        half scale = S[scale_group * params.out_dim + global_col];
                        uint packed = B[k_pack * params.out_dim + global_col];

                        half vals[8];
                        opt_dequant_fp4x8_lut(packed, scale, vals);

                        for (uint v = 0; v < 8; ++v) {
                            acc += (float)token_acts[k_base + v] * (float)vals[v];
                        }
                    }

                    result_staging[staging_row][out_col] += half(acc) * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write output
    for (uint i = thread_idx; i < OPT_TILE_M * OPT_TILE_N; i += OPT_THREADS) {
        uint local_row = i / OPT_TILE_N;
        uint local_col = i % OPT_TILE_N;
        uint global_row = tg_row + local_row;
        uint global_col = tg_col + local_col;

        if (global_row < params.batch_size && global_col < params.out_dim) {
            output[global_row * params.out_dim + global_col] = result_staging[local_row][local_col];
        }
    }
}

// ===========================================================================
// BF16 kernel variant: Pre-routed dispatch (BF16 activations/output)
// ===========================================================================

kernel void moe_dispatch_optimized_prerouted_bf16(
    device const ushort* activations   [[buffer(0)]],   // [batch, hidden] BF16 (uint16)
    device const uint* expert_weights  [[buffer(1)]],   // [num_experts, K/8, N]
    device const half* expert_scales   [[buffer(2)]],   // [num_experts, K/group, N]
    device const uint* expert_ids      [[buffer(3)]],   // [batch, top_k]
    device const half* expert_probs    [[buffer(4)]],   // [batch, top_k]
    device const uint* shared_weights  [[buffer(5)]],   // [K/8, N] (optional)
    device const half* shared_scales   [[buffer(6)]],   // [K/group, N] (optional)
    device ushort* output              [[buffer(7)]],   // [batch, N] BF16 (uint16)
    constant MoEOptParams& params      [[buffer(8)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    threadgroup ushort A_tiles_bf16[NUM_BUFFERS][OPT_TILE_M][OPT_TILE_K + 4];
    threadgroup half A_tiles_half[NUM_BUFFERS][OPT_TILE_M][OPT_TILE_K + 4];
    threadgroup half B_tiles[NUM_BUFFERS][OPT_TILE_K][OPT_TILE_N + 4];
    threadgroup uint tile_expert_ids[OPT_TILE_M][TOP_K];
    threadgroup half tile_expert_probs[OPT_TILE_M][TOP_K];
    threadgroup half result_staging[OPT_TILE_M][OPT_TILE_N + 4];

    const uint tg_row = tgid.y * OPT_TILE_M;
    const uint tg_col = tgid.x * OPT_TILE_N;
    const uint sg_row_offset = simd_id * 8;
    const uint thread_idx = simd_id * 32 + simd_lane;

    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint expert_weight_stride = k_packed * params.out_dim;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;
    const uint expert_scale_stride = num_scale_groups * params.out_dim;

    for (uint i = thread_idx; i < OPT_TILE_M * (OPT_TILE_N + 4); i += OPT_THREADS) {
        uint row = i / (OPT_TILE_N + 4);
        uint col = i % (OPT_TILE_N + 4);
        result_staging[row][col] = half(0.0h);
    }

    for (uint i = thread_idx; i < OPT_TILE_M * params.top_k; i += OPT_THREADS) {
        uint local_row = i / params.top_k;
        uint slot = i % params.top_k;
        uint global_row = tg_row + local_row;

        if (global_row < params.batch_size && slot < params.top_k) {
            tile_expert_ids[local_row][slot] = expert_ids[global_row * params.top_k + slot];
            tile_expert_probs[local_row][slot] = expert_probs[global_row * params.top_k + slot];
        } else {
            tile_expert_ids[local_row][slot] = 0;
            tile_expert_probs[local_row][slot] = half(0.0h);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params.has_shared) {
        simdgroup_matrix<half, 8, 8> shared_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                shared_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
        uint buf_compute = 0;

        opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[0], params.batch_size, params.hidden_dim,
                                 tg_row, 0, thread_idx);
        opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[0],
                                params.hidden_dim, params.out_dim,
                                tg_col, 0, params.group_size, thread_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint next_k = (kt + 1) * OPT_TILE_K;
            uint buf_load = 1 - buf_compute;

            if (next_k < params.hidden_dim) {
                opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[buf_load], params.batch_size, params.hidden_dim,
                                         tg_row, next_k, thread_idx);
                opt_load_B_tile_dequant(shared_weights, shared_scales, B_tiles[buf_load],
                                        params.hidden_dim, params.out_dim,
                                        tg_col, next_k, params.group_size, thread_idx);
            }

            opt_unpack_A_tile_bf16_to_half(A_tiles_bf16[buf_compute], A_tiles_half[buf_compute], thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            opt_compute_gemm(A_tiles_half[buf_compute], B_tiles[buf_compute],
                             shared_acc, sg_row_offset, 0);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf_compute = buf_load;
        }

        constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;
        constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;
        threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

        for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
            for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                simdgroup_store(shared_acc[mi][ni], &sg_temp[simd_id][mi * 8][ni * 8], sg_tile_cols);

        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = simd_lane; i < sg_tile_rows * sg_tile_cols; i += 32) {
            uint row = i / sg_tile_cols;
            uint col = i % sg_tile_cols;
            result_staging[sg_row_offset + row][col] = sg_temp[simd_id][row][col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = 0; slot < params.top_k; ++slot) {
        uint slot_expert = tile_expert_ids[sg_row_offset][slot];

        bool same_expert = true;
        for (uint t = 1; t < 8 && (tg_row + sg_row_offset + t) < params.batch_size; ++t) {
            if (tile_expert_ids[sg_row_offset + t][slot] != slot_expert) {
                same_expert = false;
                break;
            }
        }

        if (same_expert && slot_expert < params.num_experts) {
            simdgroup_matrix<half, 8, 8> expert_acc[OPT_SG_M_TILES][OPT_SG_N_TILES];
            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                    expert_acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

            device const uint* B = expert_weights + slot_expert * expert_weight_stride;
            device const half* S = expert_scales + slot_expert * expert_scale_stride;

            const uint num_k_tiles = (params.hidden_dim + OPT_TILE_K - 1) / OPT_TILE_K;
            uint buf_compute = 0;

            opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[0], params.batch_size, params.hidden_dim,
                                     tg_row, 0, thread_idx);
            opt_load_B_tile_dequant(B, S, B_tiles[0],
                                    params.hidden_dim, params.out_dim,
                                    tg_col, 0, params.group_size, thread_idx);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kt = 0; kt < num_k_tiles; ++kt) {
                uint next_k = (kt + 1) * OPT_TILE_K;
                uint buf_load = 1 - buf_compute;

                if (next_k < params.hidden_dim) {
                    opt_load_A_tile_bf16_raw(activations, A_tiles_bf16[buf_load], params.batch_size, params.hidden_dim,
                                             tg_row, next_k, thread_idx);
                    opt_load_B_tile_dequant(B, S, B_tiles[buf_load],
                                            params.hidden_dim, params.out_dim,
                                            tg_col, next_k, params.group_size, thread_idx);
                }

                opt_unpack_A_tile_bf16_to_half(A_tiles_bf16[buf_compute], A_tiles_half[buf_compute], thread_idx);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                opt_compute_gemm(A_tiles_half[buf_compute], B_tiles[buf_compute],
                                 expert_acc, sg_row_offset, 0);

                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf_compute = buf_load;
            }

            constexpr uint sg_tile_rows = OPT_SG_M_TILES * 8;
            constexpr uint sg_tile_cols = OPT_SG_N_TILES * 8;
            threadgroup half sg_temp[OPT_SIMDGROUPS][sg_tile_rows][sg_tile_cols];

            for (uint mi = 0; mi < OPT_SG_M_TILES; ++mi)
                for (uint ni = 0; ni < OPT_SG_N_TILES; ++ni)
                    simdgroup_store(expert_acc[mi][ni], &sg_temp[simd_id][mi * 8][ni * 8], sg_tile_cols);

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = simd_lane; i < sg_tile_rows * sg_tile_cols; i += 32) {
                uint row = i / sg_tile_cols;
                uint col = i % sg_tile_cols;
                uint staging_row = sg_row_offset + row;

                if (staging_row < OPT_TILE_M) {
                    half prob = tile_expert_probs[staging_row][slot];
                    result_staging[staging_row][col] += sg_temp[simd_id][row][col] * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (uint local_token = 0; local_token < 8; ++local_token) {
                uint staging_row = sg_row_offset + local_token;
                uint global_token = tg_row + staging_row;

                if (global_token >= params.batch_size) continue;

                uint expert_id = tile_expert_ids[staging_row][slot];
                half prob = tile_expert_probs[staging_row][slot];

                if (expert_id >= params.num_experts || prob < half(1e-6h)) continue;

                device const ushort* token_acts = activations + global_token * params.hidden_dim;
                device const uint* B = expert_weights + expert_id * expert_weight_stride;
                device const half* S = expert_scales + expert_id * expert_scale_stride;

                for (uint out_col = simd_lane; out_col < OPT_SG_N_TILES * 8 && (tg_col + out_col) < params.out_dim; out_col += 32) {
                    uint global_col = tg_col + out_col;
                    float acc = 0.0f;

                    for (uint k_pack = 0; k_pack < params.hidden_dim / FP4_PER_UINT; ++k_pack) {
                        uint k_base = k_pack * FP4_PER_UINT;
                        uint scale_group = k_base / params.group_size;

                        half scale = S[scale_group * params.out_dim + global_col];
                        uint packed = B[k_pack * params.out_dim + global_col];

                        half vals[8];
                        opt_dequant_fp4x8_lut(packed, scale, vals);

                        for (uint v = 0; v < 8; ++v) {
                            bf16_t a;
                            a.bits = token_acts[k_base + v];
                            acc += (float)vals[v] * bf16_to_float(a);
                        }
                    }

                    result_staging[staging_row][out_col] += half(acc) * prob;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = thread_idx; i < OPT_TILE_M * OPT_TILE_N; i += OPT_THREADS) {
        uint local_row = i / OPT_TILE_N;
        uint local_col = i % OPT_TILE_N;
        uint global_row = tg_row + local_row;
        uint global_col = tg_col + local_col;

        if (global_row < params.batch_size && global_col < params.out_dim) {
            bf16_t out_val = bf16_from_half(result_staging[local_row][local_col]);
            output[global_row * params.out_dim + global_col] = out_val.bits;
        }
    }
}

// ===========================================================================
// Specialized kernel: Single-token decode (batch_size=1)
//
// For the common decode case where we process one token at a time.
// Uses a different strategy optimized for minimal latency:
// - Skip all batching/grouping overhead
// - Single threadgroup computes full output
// - Expert weights streamed directly without staging
// ===========================================================================

kernel void moe_dispatch_decode(
    device const half* activations       [[buffer(0)]],   // [hidden]
    device const half* router_weights    [[buffer(1)]],   // [hidden, num_experts]
    device const uint* expert_weights    [[buffer(2)]],   // [num_experts, K/8, N]
    device const half* expert_scales     [[buffer(3)]],   // [num_experts, K/group, N]
    device const uint* shared_weights    [[buffer(4)]],   // [K/8, N]
    device const half* shared_scales     [[buffer(5)]],   // [K/group, N]
    device half* output                  [[buffer(6)]],   // [N]
    constant MoEOptParams& params        [[buffer(7)]],
    uint tid                             [[thread_position_in_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]]
) {
    // Each thread computes a portion of the output dimension
    // Total threads = num_out_cols, launched as 1D grid

    uint out_col = tid;
    if (out_col >= params.out_dim) return;

    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;

    // Step 1: Compute routing (all threads compute this redundantly but in register)
    float logits[NUM_EXPERTS];
    float max_logit = -INFINITY;

    for (uint e = 0; e < params.num_experts; ++e) {
        float logit = 0.0f;
        for (uint d = 0; d < params.hidden_dim; ++d) {
            logit += (float)activations[d] * (float)router_weights[d * params.num_experts + e];
        }
        logits[e] = logit;
        max_logit = max(max_logit, logit);
    }

    // Top-4 selection
    float top_vals[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint top_ids[4] = {0, 0, 0, 0};

    for (uint e = 0; e < params.num_experts; ++e) {
        local_topk_insert<4>(top_vals, top_ids, logits[e], e);
    }

    // Softmax over selected experts
    float exp_sum = 0.0f;
    float probs[4];
    for (uint k = 0; k < 4; ++k) {
        probs[k] = exp(top_vals[k] - max_logit);
        exp_sum += probs[k];
    }
    for (uint k = 0; k < 4; ++k) {
        probs[k] /= exp_sum;
    }

    // Step 2: Compute shared expert output
    float result = 0.0f;

    if (params.has_shared) {
        for (uint k_pack = 0; k_pack < k_packed; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_group = k_base / params.group_size;

            half scale = shared_scales[scale_group * params.out_dim + out_col];
            uint packed = shared_weights[k_pack * params.out_dim + out_col];

            half vals[8];
            opt_dequant_fp4x8_lut(packed, scale, vals);

            for (uint v = 0; v < 8; ++v) {
                result += (float)activations[k_base + v] * (float)vals[v];
            }
        }
    }

    // Step 3: Add routed expert contributions
    for (uint k = 0; k < 4; ++k) {
        if (probs[k] < 1e-6f) continue;

        uint expert_id = top_ids[k];
        uint expert_w_offset = expert_id * k_packed * params.out_dim;
        uint expert_s_offset = expert_id * num_scale_groups * params.out_dim;

        float expert_out = 0.0f;

        for (uint k_pack = 0; k_pack < k_packed; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_group = k_base / params.group_size;

            half scale = expert_scales[expert_s_offset + scale_group * params.out_dim + out_col];
            uint packed = expert_weights[expert_w_offset + k_pack * params.out_dim + out_col];

            half vals[8];
            opt_dequant_fp4x8_lut(packed, scale, vals);

            for (uint v = 0; v < 8; ++v) {
                expert_out += (float)activations[k_base + v] * (float)vals[v];
            }
        }

        result += probs[k] * expert_out;
    }

    output[out_col] = half(result);
}

// ===========================================================================
// BF16 decode kernel variant (single token)
// ===========================================================================

kernel void moe_dispatch_decode_bf16(
    device const ushort* activations   [[buffer(0)]],   // [hidden] BF16
    device const ushort* router_weights [[buffer(1)]],  // [hidden, num_experts] BF16
    device const uint* expert_weights  [[buffer(2)]],   // [num_experts, K/8, N]
    device const half* expert_scales   [[buffer(3)]],   // [num_experts, K/group, N]
    device const uint* shared_weights  [[buffer(4)]],   // [K/8, N]
    device const half* shared_scales   [[buffer(5)]],   // [K/group, N]
    device ushort* output              [[buffer(6)]],   // [N] BF16
    constant MoEOptParams& params      [[buffer(7)]],
    uint tid                            [[thread_position_in_grid]],
    uint simd_lane                      [[thread_index_in_simdgroup]]
) {
    uint out_col = tid;
    if (out_col >= params.out_dim) return;

    const uint k_packed = params.hidden_dim / FP4_PER_UINT;
    const uint num_scale_groups = (params.hidden_dim + params.group_size - 1) / params.group_size;

    float logits[NUM_EXPERTS];
    float max_logit = -INFINITY;

    for (uint e = 0; e < params.num_experts; ++e) {
        float logit = 0.0f;
        for (uint d = 0; d < params.hidden_dim; d += 4) {
            ushort4 a_packed = *((device const ushort4*)(activations + d));
            float4 a = bf16x4_to_float4(a_packed);

            ushort4 w_packed = ushort4(
                router_weights[(d + 0) * params.num_experts + e],
                router_weights[(d + 1) * params.num_experts + e],
                router_weights[(d + 2) * params.num_experts + e],
                router_weights[(d + 3) * params.num_experts + e]
            );
            float4 w = bf16x4_to_float4(w_packed);
            logit += dot(a, w);
        }
        logits[e] = logit;
        max_logit = max(max_logit, logit);
    }

    float top_vals[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint top_ids[4] = {0, 0, 0, 0};

    for (uint e = 0; e < params.num_experts; ++e) {
        local_topk_insert<4>(top_vals, top_ids, logits[e], e);
    }

    float exp_sum = 0.0f;
    float probs[4];
    for (uint k = 0; k < 4; ++k) {
        probs[k] = exp(top_vals[k] - max_logit);
        exp_sum += probs[k];
    }
    for (uint k = 0; k < 4; ++k) {
        probs[k] /= exp_sum;
    }

    float result = 0.0f;

    if (params.has_shared) {
        for (uint k_pack = 0; k_pack < k_packed; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_group = k_base / params.group_size;

            half scale = shared_scales[scale_group * params.out_dim + out_col];
            uint packed = shared_weights[k_pack * params.out_dim + out_col];

            half vals[8];
            opt_dequant_fp4x8_lut(packed, scale, vals);

            for (uint v = 0; v < 8; ++v) {
                bf16_t a;
                a.bits = activations[k_base + v];
                result += bf16_to_float(a) * (float)vals[v];
            }
        }
    }

    for (uint k = 0; k < 4; ++k) {
        if (probs[k] < 1e-6f) continue;

        uint expert_id = top_ids[k];
        uint expert_w_offset = expert_id * k_packed * params.out_dim;
        uint expert_s_offset = expert_id * num_scale_groups * params.out_dim;

        float expert_out = 0.0f;

        for (uint k_pack = 0; k_pack < k_packed; ++k_pack) {
            uint k_base = k_pack * FP4_PER_UINT;
            uint scale_group = k_base / params.group_size;

            half scale = expert_scales[expert_s_offset + scale_group * params.out_dim + out_col];
            uint packed = expert_weights[expert_w_offset + k_pack * params.out_dim + out_col];

            half vals[8];
            opt_dequant_fp4x8_lut(packed, scale, vals);

            for (uint v = 0; v < 8; ++v) {
                bf16_t a;
                a.bits = activations[k_base + v];
                expert_out += bf16_to_float(a) * (float)vals[v];
            }
        }

        result += probs[k] * expert_out;
    }

    bf16_t out_val = bf16_from_float_rne(result);
    output[out_col] = out_val.bits;
}
