// moe_fused_shared_expert.metal - Fused Router + MoE dispatch + shared expert kernel
//
// Combines three stages into unified kernels:
//   1. Router: GEMV + softmax + top-k selection
//   2. Routed MoE: dispatch and compute selected experts
//   3. Shared expert: always active, adds to final output
//
// This kernel provides multiple fusion levels:
//   - Router + Shared: Independent ops computed concurrently
//   - Full Fusion: Router + Routed + Shared in one kernel
//
// Memory savings:
//   - Eliminates intermediate router output buffers when fused
//   - Shared expert computed concurrently with router (overlapped compute)
//
// Current flow (separate):
//   moe_out = dispatch_moe(x)           // Kernel 1: write to global
//   shared_out = shared_expert(x)       // Kernel 2: write to global
//   result = moe_out + shared_out       // Kernel 3: read both, write result
//
// Fused flow (this kernel):
//   Compute both simultaneously, add results before write:
//   result = dispatch_moe(x) + shared_expert(x)  // Single kernel
//
// Memory savings:
//   - Eliminates 2x hidden_dim writes (moe_out, shared_out intermediates)
//   - Eliminates 2x hidden_dim reads (for the addition)
//   - For batch=1, hidden=7168: saves 57KB memory traffic per layer
//
// Architecture:
//   - Grid: (ceil(hidden_dim / TILE_N), batch_size, top_k)
//   - Each threadgroup handles one output tile [TILE_N] for one token
//   - Threadgroup 0 in z-dimension computes shared expert; others do routed
//   - Uses atomic_add to accumulate: shared writes first, routed experts add
//
// Alternative design (used here):
//   - Dedicated shared expert threads within each threadgroup
//   - Shared expert result stored in threadgroup memory
//   - Routed expert threads add shared result before global write
//   - Avoids atomics entirely by synchronizing via barriers
//
// This version implements the non-atomic design for better performance.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ---------------------------------------------------------------------------
// Constants (matching gemm_trellis_moe.metal)
// ---------------------------------------------------------------------------

constant constexpr uint FUSED_SIMDGROUPS = 4;
constant constexpr uint FUSED_THREADS = FUSED_SIMDGROUPS * 32;  // 128
constant constexpr uint FUSED_TILE_N = FUSED_THREADS;           // Output columns per threadgroup (must be >= FUSED_THREADS)
constant constexpr uint FUSED_TILE_K = 16;                      // K dimension tile

constant constexpr uint FUSED_TILE_N_PAD = 4;
constant constexpr uint FUSED_TILE_N_STRIDE = FUSED_TILE_N + FUSED_TILE_N_PAD;

constant constexpr uint TRELLIS_TILE = 16;
constant constexpr uint PACKED_BYTES_3BIT = 96;  // 16*16*3/8

// ---------------------------------------------------------------------------
// Parameter structure for fused kernel
// ---------------------------------------------------------------------------

struct FusedMoEParams {
    uint batch_size;         // Number of tokens
    uint hidden_dim;         // Input/output dimension
    uint intermediate_dim;   // FFN intermediate dimension
    uint num_experts;        // Total routed experts
    uint top_k;              // Experts per token
    uint bits;               // Quantization bits
    uint group_size;         // Quantization group size
    uint n_levels;           // Codebook levels
    uint has_shared_expert;  // 1 if shared expert present, 0 otherwise
};

// ---------------------------------------------------------------------------
// Forward declarations for Trellis dequantization (shared with gemm_trellis_moe)
// ---------------------------------------------------------------------------

inline half trellis_dequant_3bit_fused(
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

    // Transposed indexing
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

// Fast SiLU approximation
inline half fast_silu_fused(half x) {
    half x2 = x * x;
    half sigmoid_approx = 0.5h + 0.5h * x * rsqrt(1.0h + x2);
    return x * sigmoid_approx;
}

inline half4 fast_silu_vec4_fused(half4 x) {
    half4 x2 = x * x;
    half4 sigmoid_approx = 0.5h + 0.5h * x * rsqrt(1.0h + x2);
    return x * sigmoid_approx;
}

// FP32 atomic add helper
inline void atomic_add_fp32_fused(device float* output, uint idx, float value) {
    device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[idx]);
    uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
    uint new_bits;
    bool success;
    do {
        float old_val = as_type<float>(old_bits);
        float new_val = old_val + value;
        new_bits = as_type<uint>(new_val);
        success = atomic_compare_exchange_weak_explicit(
            atomic_ptr, &old_bits, new_bits,
            memory_order_relaxed, memory_order_relaxed);
    } while (!success);
}

// ---------------------------------------------------------------------------
// Kernel: moe_fused_shared_expert_decode
//
// Fused decode kernel for batch=1 that computes both routed experts and
// shared expert in a coordinated manner.
//
// Grid: (ceil(hidden_dim / TILE_N), 1, top_k + 1)
//   - z=0: shared expert contribution
//   - z=1..top_k: routed expert contributions
//
// The shared expert threadgroups write first (no atomics needed for z=0),
// then routed expert threadgroups atomically add their contributions.
//
// Buffer layout:
//   0: activations [batch, hidden_dim] half
//   1: gate_weights [num_experts, packed] uint8 (routed)
//   2: gate_scales [num_experts, ...] half
//   3: up_weights [num_experts, packed] uint8
//   4: up_scales [num_experts, ...] half
//   5: down_weights [num_experts, packed] uint8
//   6: down_scales [num_experts, ...] half
//   7: gate_su/sv, up_su/sv, down_su/sv [num_experts, ...] half
//   ... (similar to gemm_trellis_moe)
//   14: shared_gate_weights [packed] uint8
//   15: shared_gate_scales [...] half
//   16: shared_up_weights [packed] uint8
//   17: shared_up_scales [...] half
//   18: shared_down_weights [packed] uint8
//   19: shared_down_scales [...] half
//   20: shared_gate_su/sv, up_su/sv, down_su/sv [...] half
//   27: expert_ids [batch, top_k] uint32
//   28: expert_probs [batch, top_k] half
//   29: output [batch, hidden_dim] float32
//   30: params (FusedMoEParams)
//   31: grid [n_levels] half
// ---------------------------------------------------------------------------

kernel void moe_fused_shared_expert_decode(
    // Routed expert weights
    device const half* activations               [[buffer(0)]],
    device const uint8_t* routed_gate_weights    [[buffer(1)]],
    device const half* routed_gate_scales        [[buffer(2)]],
    device const uint8_t* routed_up_weights      [[buffer(3)]],
    device const half* routed_up_scales          [[buffer(4)]],
    device const uint8_t* routed_down_weights    [[buffer(5)]],
    device const half* routed_down_scales        [[buffer(6)]],
    device const half* routed_gate_su            [[buffer(7)]],
    device const half* routed_gate_sv            [[buffer(8)]],
    device const half* routed_up_su              [[buffer(9)]],
    device const half* routed_up_sv              [[buffer(10)]],
    device const half* routed_down_su            [[buffer(11)]],
    device const half* routed_down_sv            [[buffer(12)]],
    // Shared expert weights
    device const uint8_t* shared_gate_weights    [[buffer(13)]],
    device const half* shared_gate_scales        [[buffer(14)]],
    device const uint8_t* shared_up_weights      [[buffer(15)]],
    device const half* shared_up_scales          [[buffer(16)]],
    device const uint8_t* shared_down_weights    [[buffer(17)]],
    device const half* shared_down_scales        [[buffer(18)]],
    device const half* shared_gate_su            [[buffer(19)]],
    device const half* shared_gate_sv            [[buffer(20)]],
    device const half* shared_up_su              [[buffer(21)]],
    device const half* shared_up_sv              [[buffer(22)]],
    device const half* shared_down_su            [[buffer(23)]],
    device const half* shared_down_sv            [[buffer(24)]],
    // Routing
    device const uint* expert_ids                [[buffer(25)]],
    device const half* expert_probs              [[buffer(26)]],
    // Output
    device float* output                         [[buffer(27)]],
    // Parameters
    constant FusedMoEParams& params              [[buffer(28)]],
    device const half* grid                      [[buffer(29)]],
    // Thread identification
    uint3 tgid                                   [[threadgroup_position_in_grid]],
    uint simd_lane                               [[thread_index_in_simdgroup]],
    uint simd_id                                 [[simdgroup_index_in_threadgroup]],
    uint thread_idx                              [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * FUSED_TILE_N;
    const uint token_idx = tgid.y;  // Always 0 for decode
    const uint slot = tgid.z;

    // Shared/threadgroup memory
    threadgroup half A_buf[FUSED_TILE_K];
    threadgroup half B_gate[FUSED_TILE_K][FUSED_TILE_N_STRIDE];
    threadgroup half B_up[FUSED_TILE_K][FUSED_TILE_N_STRIDE];
    threadgroup half B_down[FUSED_TILE_K][FUSED_TILE_N_STRIDE];

    // Accumulators (per-thread registers)
    float acc_down[FUSED_TILE_N / FUSED_THREADS];
    for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
        acc_down[i] = 0.0f;
    }

    // Determine if this is shared expert (slot 0) or routed expert (slot 1..top_k)
    const bool is_shared = (slot == 0);

    uint expert_id = 0;
    half expert_prob = 1.0h;  // Shared expert always has weight 1.0

    if (!is_shared) {
        // Routed expert: get routing info
        uint routed_slot = slot - 1;  // Convert to 0-indexed for routing lookup
        expert_id = expert_ids[token_idx * params.top_k + routed_slot];
        expert_prob = expert_probs[token_idx * params.top_k + routed_slot];

        // Skip if expert probability is negligible
        if (expert_prob < 1e-6h) {
            return;
        }
    }

    // Select weight pointers based on expert type
    device const uint8_t* gate_w = is_shared ? shared_gate_weights : routed_gate_weights;
    device const half* gate_s = is_shared ? shared_gate_scales : routed_gate_scales;
    device const uint8_t* up_w = is_shared ? shared_up_weights : routed_up_weights;
    device const half* up_s = is_shared ? shared_up_scales : routed_up_scales;
    device const uint8_t* down_w = is_shared ? shared_down_weights : routed_down_weights;
    device const half* down_s = is_shared ? shared_down_scales : routed_down_scales;
    device const half* gate_su = is_shared ? shared_gate_su : routed_gate_su;
    device const half* gate_sv = is_shared ? shared_gate_sv : routed_gate_sv;
    device const half* up_su = is_shared ? shared_up_su : routed_up_su;
    device const half* up_sv = is_shared ? shared_up_sv : routed_up_sv;
    device const half* down_su = is_shared ? shared_down_su : routed_down_su;
    device const half* down_sv = is_shared ? shared_down_sv : routed_down_sv;

    // For shared expert, expert_id = 0 in the shared weight arrays
    // For routed expert, expert_id indexes into the routed weight arrays
    uint weight_expert_id = is_shared ? 0 : expert_id;

    // Stream through intermediate dimension in chunks
    const uint num_inter_chunks = (params.intermediate_dim + FUSED_TILE_N - 1) / FUSED_TILE_N;

    for (uint inter_chunk = 0; inter_chunk < num_inter_chunks; ++inter_chunk) {
        uint inter_block = inter_chunk * FUSED_TILE_N;

        // Accumulators for this intermediate chunk
        float gate_acc[FUSED_TILE_N / FUSED_THREADS];
        float up_acc[FUSED_TILE_N / FUSED_THREADS];
        for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
            gate_acc[i] = 0.0f;
            up_acc[i] = 0.0f;
        }

        // K-reduction loop for gate and up projections
        const uint num_k_tiles = (params.hidden_dim + FUSED_TILE_K - 1) / FUSED_TILE_K;

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint k_block = kt * FUSED_TILE_K;

            // Load activation tile cooperatively
            if (thread_idx < FUSED_TILE_K) {
                uint global_k = k_block + thread_idx;
                A_buf[thread_idx] = (global_k < params.hidden_dim)
                    ? activations[token_idx * params.hidden_dim + global_k]
                    : 0.0h;
            }

            // Load gate weight tile
            const uint elems_per_thread = (FUSED_TILE_K * FUSED_TILE_N) / FUSED_THREADS;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint k_local = flat_idx / FUSED_TILE_N;
                uint n_local = flat_idx % FUSED_TILE_N;
                uint global_k = k_block + k_local;
                uint global_n = inter_block + n_local;

                half val = 0.0h;
                if (global_k < params.hidden_dim && global_n < params.intermediate_dim) {
                    val = trellis_dequant_3bit_fused(
                        gate_w, gate_s, gate_su, gate_sv, grid,
                        weight_expert_id, global_k, global_n,
                        params.hidden_dim, params.intermediate_dim,
                        params.group_size, params.n_levels
                    );
                }
                B_gate[k_local][n_local] = val;
            }

            // Load up weight tile
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint k_local = flat_idx / FUSED_TILE_N;
                uint n_local = flat_idx % FUSED_TILE_N;
                uint global_k = k_block + k_local;
                uint global_n = inter_block + n_local;

                half val = 0.0h;
                if (global_k < params.hidden_dim && global_n < params.intermediate_dim) {
                    val = trellis_dequant_3bit_fused(
                        up_w, up_s, up_su, up_sv, grid,
                        weight_expert_id, global_k, global_n,
                        params.hidden_dim, params.intermediate_dim,
                        params.group_size, params.n_levels
                    );
                }
                B_up[k_local][n_local] = val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute gate and up partial sums
            for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                uint n_idx = thread_idx + i * FUSED_THREADS;
                if (inter_block + n_idx >= params.intermediate_dim) continue;

                float gate_sum = 0.0f;
                float up_sum = 0.0f;

                for (uint k = 0; k < FUSED_TILE_K && k_block + k < params.hidden_dim; ++k) {
                    float a = (float)A_buf[k];
                    gate_sum += a * (float)B_gate[k][n_idx];
                    up_sum += a * (float)B_up[k][n_idx];
                }

                gate_acc[i] += gate_sum;
                up_acc[i] += up_sum;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Apply SwiGLU: silu(gate) * up, then multiply by down weights
        // Store swiglu result temporarily in B_gate
        for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
            uint n_idx = thread_idx + i * FUSED_THREADS;
            half swiglu = 0.0h;
            if (inter_block + n_idx < params.intermediate_dim) {
                half gate_h = (half)gate_acc[i];
                half up_h = (half)up_acc[i];
                swiglu = fast_silu_fused(gate_h) * up_h;
            }
            // Store in first row of B_gate for the down projection
            if (n_idx < FUSED_TILE_N) {
                B_gate[0][n_idx] = swiglu;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down projection: intermediate -> hidden
        // For this chunk of intermediate, compute contribution to output
        // This is a [1 x TILE_N] @ [TILE_N x hidden] -> [1 x hidden]
        // But we're streaming, so each chunk contributes to all hidden outputs

        const uint num_out_tiles = (params.hidden_dim + FUSED_TILE_N - 1) / FUSED_TILE_N;

        for (uint out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
            uint out_block = out_tile * FUSED_TILE_N;
            if (out_block != n_block) continue;  // Only process our assigned output tile

            // Load down weight tile for this (inter_block, out_block) region
            const uint elems_per_thread = (FUSED_TILE_K * FUSED_TILE_N) / FUSED_THREADS;

            // Down weights are [intermediate_dim x hidden_dim]
            // We need [TILE_N x TILE_N] tile at (inter_block, out_block)
            // But TILE_K != TILE_N, so we iterate

            const uint down_k_tiles = (FUSED_TILE_N + FUSED_TILE_K - 1) / FUSED_TILE_K;

            for (uint dkt = 0; dkt < down_k_tiles; ++dkt) {
                uint down_k_block = inter_block + dkt * FUSED_TILE_K;

                // Load down weight tile
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = thread_idx * elems_per_thread + i;
                    uint k_local = flat_idx / FUSED_TILE_N;
                    uint n_local = flat_idx % FUSED_TILE_N;
                    uint global_k = down_k_block + k_local;  // intermediate dim
                    uint global_n = out_block + n_local;      // hidden dim (output)

                    half val = 0.0h;
                    if (global_k < params.intermediate_dim && global_n < params.hidden_dim) {
                        val = trellis_dequant_3bit_fused(
                            down_w, down_s, down_su, down_sv, grid,
                            weight_expert_id, global_k, global_n,
                            params.intermediate_dim, params.hidden_dim,
                            params.group_size, params.n_levels
                        );
                    }
                    B_down[k_local][n_local] = val;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Accumulate down projection contribution
                for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                    uint n_idx = thread_idx + i * FUSED_THREADS;
                    if (out_block + n_idx >= params.hidden_dim) continue;

                    float down_sum = 0.0f;

                    for (uint k = 0; k < FUSED_TILE_K; ++k) {
                        uint global_k = down_k_block + k - inter_block;  // Index into swiglu result
                        if (global_k < FUSED_TILE_N) {
                            float swiglu = (float)B_gate[0][global_k];
                            float w = (float)B_down[k][n_idx];
                            down_sum += swiglu * w;
                        }
                    }

                    acc_down[i] += down_sum;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Write output with expert probability weighting
    // Shared expert writes directly (probability = 1.0)
    // Routed experts use atomic add
    for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
        uint n_idx = thread_idx + i * FUSED_THREADS;
        uint global_n = n_block + n_idx;

        if (global_n < params.hidden_dim) {
            uint out_idx = token_idx * params.hidden_dim + global_n;
            float weighted_val = acc_down[i] * (float)expert_prob;

            if (is_shared) {
                // Shared expert writes first (no atomic needed if guaranteed first)
                output[out_idx] = weighted_val;
            } else {
                // Routed experts atomically add
                atomic_add_fp32_fused(output, out_idx, weighted_val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Router + Shared Expert Kernel
//
// Single kernel that computes both router decisions and shared expert output.
// These are independent operations (both read from same activations), so they
// can be computed concurrently within separate threadgroups of the same launch.
//
// Grid dispatch: 2D grid with z-dimension separation
//   z=0: Router threadgroups (one per token)
//         - Output: expert_ids[batch, top_k], expert_probs[batch, top_k]
//   z=1: Shared expert threadgroups (one per token)
//         - Output: adds shared expert contribution to output buffer
//
// Benefits over sequential execution:
//   - 1 kernel launch instead of 2
//   - Router and shared expert execute in parallel on same GPU
//   - No intermediate buffers needed for router output
//   - Shared expert output added directly to final buffer via atomic_add
//
// Buffer layout (shared expert only):
//   0: activations [batch, hidden_dim] half
//   1-12: shared expert weights (gate, up, down with scales and su/sv)
//   13: output [batch, hidden_dim] float32 (atomic add for shared expert)
//   14: params (FusedMoEParams)
//   15: grid [n_levels] half
//
// Buffer layout (router output buffers):
//   16: expert_ids [batch, top_k] uint32
//   17: expert_probs [batch, top_k] half
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Router constants (shared with moe_fused_router.metal)
// ---------------------------------------------------------------------------

constant constexpr uint ROUTER_MAX_EXPERTS = 256;
constant constexpr uint ROUTER_MAX_TOP_K = 16;
constant constexpr uint ROUTER_THREADS = 256;
constant constexpr uint ROUTER_SIMD_WIDTH = 32;

// ---------------------------------------------------------------------------
// Router parameters (subset of RouterParams)
// ---------------------------------------------------------------------------

struct RouterSharedParams {
    uint batch_size;      // Number of tokens
    uint hidden_dim;      // Hidden dimension
    uint num_experts;     // Total routed experts
    uint top_k;           // Experts per token
};

// ---------------------------------------------------------------------------
// Numerically stable softmax (from moe_fused_router.metal)
// ---------------------------------------------------------------------------

inline float safe_exp_router(float x, float max_val) {
    float shifted = x - max_val;
    shifted = clamp(shifted, -88.0f, 88.0f);
    return exp(shifted);
}

// ---------------------------------------------------------------------------
// Simdgroup reductions for router
// ---------------------------------------------------------------------------

inline float simd_max_router(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_router(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Top-k insertion (from moe_fused_router.metal)
// ---------------------------------------------------------------------------

template <uint K>
inline bool insert_topk_router(
    thread float (&values)[K],
    thread uint (&indices)[K],
    float val,
    uint idx
) {
    if (val <= values[K - 1]) return false;

    uint insert_pos = K;
    for (uint i = 0; i < K; ++i) {
        if (val > values[i]) {
            insert_pos = i;
            break;
        }
    }

    for (uint i = K - 1; i > insert_pos; --i) {
        values[i] = values[i - 1];
        indices[i] = indices[i - 1];
    }

    values[insert_pos] = val;
    indices[insert_pos] = idx;
    return true;
}

// ---------------------------------------------------------------------------
// Fused Router + Shared Expert Kernel
// ---------------------------------------------------------------------------
// Single kernel that computes router decisions AND shared expert output concurrently.
//
// Grid dispatch:
//   z = 0: Router threadgroups (one per token)
//   z = 1: Shared expert threadgroups (one per token)
//
// These execute concurrently on the GPU, providing significant speedup
// over sequential kernel launches.
//
// Benefits:
//   - 1 kernel launch instead of 2
//   - Router and shared expert compute in parallel
//   - No intermediate router output buffer needed for shared expert path
//   - Shared expert result added atomically to output while router computes
//
// Buffer layout:
//   Common inputs:
//     0: activations [batch, hidden_dim] half
//
//   Shared expert weights:
//     1-12: shared expert weights (gate, up, down with scales and su/sv)
//
//   Router weights (router_z=0 only):
//     13: router_weights [hidden_dim, num_experts] half
//
//   Outputs:
//     14: output [batch, hidden_dim] float32 (atomic add for shared expert)
//     15: expert_ids [batch, top_k] uint32 (router output)
//     16: expert_probs [batch, top_k] half (router output)
//
//   Parameters:
//     17: router_params (RouterSharedParams)
//     18: shared_params (FusedMoEParams)
//     19: grid [n_levels] half
// ---------------------------------------------------------------------------

kernel void fused_router_shared_expert(
    // Common input
    device const half* activations               [[buffer(0)]],

    // Shared expert weights (shared_expert_z=1)
    device const uint8_t* shared_gate_weights    [[buffer(1)]],
    device const half* shared_gate_scales        [[buffer(2)]],
    device const uint8_t* shared_up_weights      [[buffer(3)]],
    device const half* shared_up_scales          [[buffer(4)]],
    device const uint8_t* shared_down_weights    [[buffer(5)]],
    device const half* shared_down_scales        [[buffer(6)]],
    device const half* shared_gate_su            [[buffer(7)]],
    device const half* shared_gate_sv            [[buffer(8)]],
    device const half* shared_up_su              [[buffer(9)]],
    device const half* shared_up_sv              [[buffer(10)]],
    device const half* shared_down_su            [[buffer(11)]],
    device const half* shared_down_sv            [[buffer(12)]],

    // Router weights (router_z=0 only)
    device const half* router_weights             [[buffer(13)]],

    // Outputs
    device float* output                         [[buffer(14)]],  // Shared expert writes here
    device uint* expert_ids                      [[buffer(15)]],  // Router writes here
    device half* expert_probs                    [[buffer(16)]],  // Router writes here

    // Parameters
    constant RouterSharedParams& router_params      [[buffer(17)]],
    constant FusedMoEParams& shared_params        [[buffer(18)]],
    device const half* grid                       [[buffer(19)]],

    // Thread identification
    uint3 tgid                                   [[threadgroup_position_in_grid]],
    uint tid                                     [[thread_index_in_threadgroup]],
    uint simd_lane                               [[thread_index_in_simdgroup]]
) {
    // z=0: Router computation
    // z=1: Shared expert computation
    const uint is_router = (tgid.z == 0);

    if (is_router) {
        // =====================================================================
        // Router path (z=0): Compute top-k expert selection
        // =====================================================================

        const uint batch_idx = tgid.y;
        if (batch_idx >= router_params.batch_size) return;

        const uint num_simdgroups = ROUTER_THREADS / ROUTER_SIMD_WIDTH;
        const uint simd_id = tid / ROUTER_SIMD_WIDTH;

        // Threadgroup memory for router
        threadgroup float tg_logits[ROUTER_MAX_EXPERTS];
        threadgroup float tg_max[8];  // Up to 8 simdgroups
        threadgroup float tg_sum[8];
        threadgroup float tg_topk_vals[ROUTER_MAX_TOP_K];
        threadgroup uint tg_topk_ids[ROUTER_MAX_TOP_K];

        device const half* h = activations + batch_idx * router_params.hidden_dim;

        // --------------------------------------------------------------------
        // Step 1: Router GEMV - compute logits
        // --------------------------------------------------------------------

        uint experts_per_thread = (router_params.num_experts + ROUTER_THREADS - 1) / ROUTER_THREADS;

        for (uint e_iter = 0; e_iter < experts_per_thread; ++e_iter) {
            uint expert_idx = tid + e_iter * ROUTER_THREADS;

            if (expert_idx < router_params.num_experts) {
                float acc = 0.0f;
                device const half* w_col = router_weights + expert_idx;

                // Vectorized accumulation (8 at a time)
                uint d = 0;
                uint hidden_vec = router_params.hidden_dim & ~7u;

                for (; d < hidden_vec; d += 8) {
                    acc += float(h[d + 0]) * float(w_col[(d + 0) * router_params.num_experts]);
                    acc += float(h[d + 1]) * float(w_col[(d + 1) * router_params.num_experts]);
                    acc += float(h[d + 2]) * float(w_col[(d + 2) * router_params.num_experts]);
                    acc += float(h[d + 3]) * float(w_col[(d + 3) * router_params.num_experts]);
                    acc += float(h[d + 4]) * float(w_col[(d + 4) * router_params.num_experts]);
                    acc += float(h[d + 5]) * float(w_col[(d + 5) * router_params.num_experts]);
                    acc += float(h[d + 6]) * float(w_col[(d + 6) * router_params.num_experts]);
                    acc += float(h[d + 7]) * float(w_col[(d + 7) * router_params.num_experts]);
                }

                // Handle remainder
                for (; d < router_params.hidden_dim; ++d) {
                    acc += float(h[d]) * float(w_col[d * router_params.num_experts]);
                }

                tg_logits[expert_idx] = acc;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --------------------------------------------------------------------
        // Step 2: Softmax (numerically stable)
        // --------------------------------------------------------------------

        // Find max
        float local_max = -INFINITY;
        for (uint e = tid; e < router_params.num_experts; e += ROUTER_THREADS) {
            local_max = max(local_max, tg_logits[e]);
        }

        local_max = simd_max_router(local_max);

        if (simd_lane == 0) {
            tg_max[simd_id] = local_max;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float global_max = tg_max[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_max = max(global_max, tg_max[s]);
        }

        if (simd_id == 0 && simd_lane == 0) {
            tg_max[0] = global_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        global_max = tg_max[0];

        // Compute exp and sum
        float local_sum = 0.0f;
        for (uint e = tid; e < router_params.num_experts; e += ROUTER_THREADS) {
            float exp_val = safe_exp_router(tg_logits[e], global_max);
            tg_logits[e] = exp_val;
            local_sum += exp_val;
        }

        local_sum = simd_sum_router(local_sum);

        if (simd_lane == 0) {
            tg_sum[simd_id] = local_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float global_sum = tg_sum[0];
        for (uint s = 1; s < num_simdgroups; ++s) {
            global_sum += tg_sum[s];
        }

        if (simd_id == 0 && simd_lane == 0) {
            tg_sum[0] = global_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        global_sum = tg_sum[0];

        // Normalize
        float inv_sum = 1.0f / global_sum;
        for (uint e = tid; e < router_params.num_experts; e += ROUTER_THREADS) {
            tg_logits[e] *= inv_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --------------------------------------------------------------------
        // Step 3: Top-k selection (thread 0)
        // --------------------------------------------------------------------

        if (tid == 0) {
            float local_topk_vals[ROUTER_MAX_TOP_K];
            uint local_topk_ids[ROUTER_MAX_TOP_K];

            for (uint i = 0; i < router_params.top_k; ++i) {
                local_topk_vals[i] = -INFINITY;
                local_topk_ids[i] = 0;
            }

            for (uint e = 0; e < router_params.num_experts; ++e) {
                float prob = tg_logits[e];
                insert_topk_router<ROUTER_MAX_TOP_K>(local_topk_vals, local_topk_ids, prob, e);
            }

            // Store for renormalization
            for (uint i = 0; i < router_params.top_k; ++i) {
                tg_topk_vals[i] = local_topk_vals[i];
                tg_topk_ids[i] = local_topk_ids[i];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --------------------------------------------------------------------
        // Step 4: Renormalize and write router outputs
        // --------------------------------------------------------------------

        if (tid == 0) {
            float selected_sum = 0.0f;
            for (uint i = 0; i < router_params.top_k; ++i) {
                selected_sum += tg_topk_vals[i];
            }

            float inv_selected_sum = 1.0f / max(selected_sum, 1e-8f);

            device uint* out_ids = expert_ids + batch_idx * router_params.top_k;
            device half* out_probs = expert_probs + batch_idx * router_params.top_k;

            for (uint i = 0; i < router_params.top_k; ++i) {
                out_ids[i] = tg_topk_ids[i];
                out_probs[i] = half(tg_topk_vals[i] * inv_selected_sum);
            }
        }

    } else {
        // =====================================================================
        // Shared expert path (z=1): Compute shared expert contribution
        // =====================================================================

        const uint batch_idx = tgid.y;
        if (batch_idx >= shared_params.batch_size) return;

        const uint n_block = tgid.x * FUSED_TILE_N;
        const uint token_idx = batch_idx;

        // Threadgroup memory for shared expert
        threadgroup half A_buf[FUSED_TILE_K];
        threadgroup half B_gate[FUSED_TILE_K][FUSED_TILE_N_STRIDE];
        threadgroup half B_up[FUSED_TILE_K][FUSED_TILE_N_STRIDE];

        float acc_out[FUSED_TILE_N / FUSED_THREADS];
        for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
            acc_out[i] = 0.0f;
        }

        // Stream through intermediate dimension
        const uint num_inter_chunks = (shared_params.intermediate_dim + FUSED_TILE_N - 1) / FUSED_TILE_N;

        for (uint inter_chunk = 0; inter_chunk < num_inter_chunks; ++inter_chunk) {
            uint inter_block = inter_chunk * FUSED_TILE_N;

            float gate_acc[FUSED_TILE_N / FUSED_THREADS];
            float up_acc[FUSED_TILE_N / FUSED_THREADS];
            for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                gate_acc[i] = 0.0f;
                up_acc[i] = 0.0f;
            }

            const uint num_k_tiles = (shared_params.hidden_dim + FUSED_TILE_K - 1) / FUSED_TILE_K;

            for (uint kt = 0; kt < num_k_tiles; ++kt) {
                uint k_block = kt * FUSED_TILE_K;

                // Load activation tile
                if (tid < FUSED_TILE_K) {
                    uint global_k = k_block + tid;
                    A_buf[tid] = (global_k < shared_params.hidden_dim)
                        ? activations[token_idx * shared_params.hidden_dim + global_k]
                        : 0.0h;
                }

                // Load gate and up weight tiles
                const uint elems_per_thread = (FUSED_TILE_K * FUSED_TILE_N) / FUSED_THREADS;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = tid * elems_per_thread + i;
                    uint k_local = flat_idx / FUSED_TILE_N;
                    uint n_local = flat_idx % FUSED_TILE_N;
                    uint global_k = k_block + k_local;
                    uint global_n = inter_block + n_local;

                    half gate_val = 0.0h;
                    half up_val = 0.0h;

                    if (global_k < shared_params.hidden_dim && global_n < shared_params.intermediate_dim) {
                        gate_val = trellis_dequant_3bit_fused(
                            shared_gate_weights, shared_gate_scales,
                            shared_gate_su, shared_gate_sv, grid,
                            0, global_k, global_n,
                            shared_params.hidden_dim, shared_params.intermediate_dim,
                            shared_params.group_size, shared_params.n_levels
                        );
                        up_val = trellis_dequant_3bit_fused(
                            shared_up_weights, shared_up_scales,
                            shared_up_su, shared_up_sv, grid,
                            0, global_k, global_n,
                            shared_params.hidden_dim, shared_params.intermediate_dim,
                            shared_params.group_size, shared_params.n_levels
                        );
                    }
                    B_gate[k_local][n_local] = gate_val;
                    B_up[k_local][n_local] = up_val;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Compute partial sums
                for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                    uint n_idx = tid + i * FUSED_THREADS;
                    if (inter_block + n_idx >= shared_params.intermediate_dim) continue;

                    float gate_sum = 0.0f;
                    float up_sum = 0.0f;

                    for (uint k = 0; k < FUSED_TILE_K && k_block + k < shared_params.hidden_dim; ++k) {
                        float a = (float)A_buf[k];
                        gate_sum += a * (float)B_gate[k][n_idx];
                        up_sum += a * (float)B_up[k][n_idx];
                    }

                    gate_acc[i] += gate_sum;
                    up_acc[i] += up_sum;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // SwiGLU activation
            threadgroup half swiglu_buf[FUSED_TILE_N];
            for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                uint n_idx = tid + i * FUSED_THREADS;
                half swiglu = 0.0h;
                if (inter_block + n_idx < shared_params.intermediate_dim) {
                    half gate_h = (half)gate_acc[i];
                    half up_h = (half)up_acc[i];
                    swiglu = fast_silu_fused(gate_h) * up_h;
                }
                if (n_idx < FUSED_TILE_N) {
                    swiglu_buf[n_idx] = swiglu;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Down projection
            const uint down_k_tiles = (FUSED_TILE_N + FUSED_TILE_K - 1) / FUSED_TILE_K;

            for (uint dkt = 0; dkt < down_k_tiles; ++dkt) {
                uint down_k_offset = dkt * FUSED_TILE_K;
                uint down_k_block = inter_block + down_k_offset;

                const uint elems_per_thread = (FUSED_TILE_K * FUSED_TILE_N) / FUSED_THREADS;
                for (uint i = 0; i < elems_per_thread; ++i) {
                    uint flat_idx = tid * elems_per_thread + i;
                    uint k_local = flat_idx / FUSED_TILE_N;
                    uint n_local = flat_idx % FUSED_TILE_N;
                    uint global_k = down_k_block + k_local;
                    uint global_n = n_block + n_local;

                    half val = 0.0h;
                    if (global_k < shared_params.intermediate_dim && global_n < shared_params.hidden_dim) {
                        val = trellis_dequant_3bit_fused(
                            shared_down_weights, shared_down_scales,
                            shared_down_su, shared_down_sv, grid,
                            0, global_k, global_n,
                            shared_params.intermediate_dim, shared_params.hidden_dim,
                            shared_params.group_size, shared_params.n_levels
                        );
                    }
                    B_gate[k_local][n_local] = val;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Accumulate down projection
                for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                    uint n_idx = tid + i * FUSED_THREADS;
                    if (n_block + n_idx >= shared_params.hidden_dim) continue;

                    float down_sum = 0.0f;

                    for (uint k = 0; k < FUSED_TILE_K; ++k) {
                        uint swiglu_idx = down_k_offset + k;
                        if (swiglu_idx < FUSED_TILE_N && inter_block + swiglu_idx < shared_params.intermediate_dim) {
                            float s = (float)swiglu_buf[swiglu_idx];
                            float w = (float)B_gate[k][n_idx];
                            down_sum += s * w;
                        }
                    }

                    acc_out[i] += down_sum;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Write shared expert output with atomic add (allows concurrent execution with routed experts)
        for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
            uint n_idx = tid + i * FUSED_THREADS;
            uint global_n = n_block + n_idx;

            if (global_n < shared_params.hidden_dim) {
                uint out_idx = token_idx * shared_params.hidden_dim + global_n;
                atomic_add_fp32_fused(output, out_idx, acc_out[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simplified fused kernel for the common case where shared expert is added
// after MoE dispatch. This kernel assumes:
//   1. MoE output is already computed in output buffer
//   2. We just need to add shared expert contribution
//
// This is more practical for integration since it doesn't require changing
// the existing MoE kernel infrastructure.
//
// Grid: (ceil(hidden_dim / TILE_N), batch_size, 1)
// ---------------------------------------------------------------------------

kernel void add_shared_expert_decode(
    // Input
    device const half* activations               [[buffer(0)]],
    // Shared expert weights (Trellis quantized)
    device const uint8_t* shared_gate_weights    [[buffer(1)]],
    device const half* shared_gate_scales        [[buffer(2)]],
    device const uint8_t* shared_up_weights      [[buffer(3)]],
    device const half* shared_up_scales          [[buffer(4)]],
    device const uint8_t* shared_down_weights    [[buffer(5)]],
    device const half* shared_down_scales        [[buffer(6)]],
    device const half* shared_gate_su            [[buffer(7)]],
    device const half* shared_gate_sv            [[buffer(8)]],
    device const half* shared_up_su              [[buffer(9)]],
    device const half* shared_up_sv              [[buffer(10)]],
    device const half* shared_down_su            [[buffer(11)]],
    device const half* shared_down_sv            [[buffer(12)]],
    // Output (MoE output already present, we add to it)
    device float* output                         [[buffer(13)]],
    // Parameters
    constant FusedMoEParams& params              [[buffer(14)]],
    device const half* grid                      [[buffer(15)]],
    // Thread identification
    uint3 tgid                                   [[threadgroup_position_in_grid]],
    uint thread_idx                              [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * FUSED_TILE_N;
    const uint token_idx = tgid.y;

    // Threadgroup memory
    threadgroup half A_buf[FUSED_TILE_K];
    threadgroup half B_gate[FUSED_TILE_K][FUSED_TILE_N_STRIDE];
    threadgroup half B_up[FUSED_TILE_K][FUSED_TILE_N_STRIDE];

    // Output accumulator
    float acc_out[FUSED_TILE_N / FUSED_THREADS];
    for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
        acc_out[i] = 0.0f;
    }

    // Stream through intermediate dimension
    const uint num_inter_chunks = (params.intermediate_dim + FUSED_TILE_N - 1) / FUSED_TILE_N;

    for (uint inter_chunk = 0; inter_chunk < num_inter_chunks; ++inter_chunk) {
        uint inter_block = inter_chunk * FUSED_TILE_N;

        float gate_acc[FUSED_TILE_N / FUSED_THREADS];
        float up_acc[FUSED_TILE_N / FUSED_THREADS];
        for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
            gate_acc[i] = 0.0f;
            up_acc[i] = 0.0f;
        }

        // K-reduction for gate and up
        const uint num_k_tiles = (params.hidden_dim + FUSED_TILE_K - 1) / FUSED_TILE_K;

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            uint k_block = kt * FUSED_TILE_K;

            // Load activation tile
            if (thread_idx < FUSED_TILE_K) {
                uint global_k = k_block + thread_idx;
                A_buf[thread_idx] = (global_k < params.hidden_dim)
                    ? activations[token_idx * params.hidden_dim + global_k]
                    : 0.0h;
            }

            // Load gate and up weight tiles
            const uint elems_per_thread = (FUSED_TILE_K * FUSED_TILE_N) / FUSED_THREADS;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint k_local = flat_idx / FUSED_TILE_N;
                uint n_local = flat_idx % FUSED_TILE_N;
                uint global_k = k_block + k_local;
                uint global_n = inter_block + n_local;

                half gate_val = 0.0h;
                half up_val = 0.0h;

                if (global_k < params.hidden_dim && global_n < params.intermediate_dim) {
                    gate_val = trellis_dequant_3bit_fused(
                        shared_gate_weights, shared_gate_scales,
                        shared_gate_su, shared_gate_sv, grid,
                        0, global_k, global_n,
                        params.hidden_dim, params.intermediate_dim,
                        params.group_size, params.n_levels
                    );
                    up_val = trellis_dequant_3bit_fused(
                        shared_up_weights, shared_up_scales,
                        shared_up_su, shared_up_sv, grid,
                        0, global_k, global_n,
                        params.hidden_dim, params.intermediate_dim,
                        params.group_size, params.n_levels
                    );
                }
                B_gate[k_local][n_local] = gate_val;
                B_up[k_local][n_local] = up_val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute partial sums
            for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                uint n_idx = thread_idx + i * FUSED_THREADS;
                if (inter_block + n_idx >= params.intermediate_dim) continue;

                float gate_sum = 0.0f;
                float up_sum = 0.0f;

                for (uint k = 0; k < FUSED_TILE_K && k_block + k < params.hidden_dim; ++k) {
                    float a = (float)A_buf[k];
                    gate_sum += a * (float)B_gate[k][n_idx];
                    up_sum += a * (float)B_up[k][n_idx];
                }

                gate_acc[i] += gate_sum;
                up_acc[i] += up_sum;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // SwiGLU activation
        threadgroup half swiglu_buf[FUSED_TILE_N];
        for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
            uint n_idx = thread_idx + i * FUSED_THREADS;
            half swiglu = 0.0h;
            if (inter_block + n_idx < params.intermediate_dim) {
                half gate_h = (half)gate_acc[i];
                half up_h = (half)up_acc[i];
                swiglu = fast_silu_fused(gate_h) * up_h;
            }
            if (n_idx < FUSED_TILE_N) {
                swiglu_buf[n_idx] = swiglu;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down projection - stream through output tiles
        // Only process our assigned output tile (n_block)
        const uint down_k_tiles = (FUSED_TILE_N + FUSED_TILE_K - 1) / FUSED_TILE_K;

        for (uint dkt = 0; dkt < down_k_tiles; ++dkt) {
            uint down_k_offset = dkt * FUSED_TILE_K;
            uint down_k_block = inter_block + down_k_offset;

            // Load down weight tile [TILE_K x TILE_N] from [inter_block, n_block]
            const uint elems_per_thread = (FUSED_TILE_K * FUSED_TILE_N) / FUSED_THREADS;
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint k_local = flat_idx / FUSED_TILE_N;
                uint n_local = flat_idx % FUSED_TILE_N;
                uint global_k = down_k_block + k_local;  // intermediate dim
                uint global_n = n_block + n_local;        // output (hidden) dim

                half val = 0.0h;
                if (global_k < params.intermediate_dim && global_n < params.hidden_dim) {
                    val = trellis_dequant_3bit_fused(
                        shared_down_weights, shared_down_scales,
                        shared_down_su, shared_down_sv, grid,
                        0, global_k, global_n,
                        params.intermediate_dim, params.hidden_dim,
                        params.group_size, params.n_levels
                    );
                }
                B_gate[k_local][n_local] = val;  // Reuse B_gate buffer for down weights
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate down projection
            for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
                uint n_idx = thread_idx + i * FUSED_THREADS;
                if (n_block + n_idx >= params.hidden_dim) continue;

                float down_sum = 0.0f;

                for (uint k = 0; k < FUSED_TILE_K; ++k) {
                    uint swiglu_idx = down_k_offset + k;
                    if (swiglu_idx < FUSED_TILE_N && inter_block + swiglu_idx < params.intermediate_dim) {
                        float s = (float)swiglu_buf[swiglu_idx];
                        float w = (float)B_gate[k][n_idx];
                        down_sum += s * w;
                    }
                }

                acc_out[i] += down_sum;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Add shared expert output to existing MoE output (atomic add)
    for (uint i = 0; i < FUSED_TILE_N / FUSED_THREADS; ++i) {
        uint n_idx = thread_idx + i * FUSED_THREADS;
        uint global_n = n_block + n_idx;

        if (global_n < params.hidden_dim) {
            uint out_idx = token_idx * params.hidden_dim + global_n;
            atomic_add_fp32_fused(output, out_idx, acc_out[i]);
        }
    }
}
