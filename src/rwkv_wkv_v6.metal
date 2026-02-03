#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RWKV WKV v6 (Weighted Key-Value) Kernel for Apple Metal
// ============================================================================
//
// RWKV v6 ("Eagle/Finch") introduces matrix-valued states for improved
// long-context modeling and better GPU utilization. Key differences from v5:
//
// 1. **Matrix-valued state**: State is now [head_dim, head_dim] per head,
//    enabling richer temporal dependencies and better tensor core utilization.
//
// 2. **Dynamic recurrence**: The state update uses matrix multiplication
//    patterns that map efficiently to GPU hardware (vs. v5's vector ops).
//
// 3. **Enhanced time-mixing**: Improved gate mechanism and time decay
//    for better long-range reasoning.
//
// Mathematical formulation (per head):
//   state_t = exp(-w) ⊙ state_{t-1} + exp(k_t) ⊗ v_t^T
//   output_t = r_t ⊙ (state_t @ g_t)
//
// Where:
//   ⊙ = elementwise multiply
//   ⊗ = outer product
//   @ = matrix multiply
//
// Reference: "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic
// Recurrence" (arXiv:2404.05892)
//
// ============================================================================

// ============================================================================
// Constants
// ============================================================================

constant constexpr uint MAX_HEAD_DIM = 128;
constant constexpr uint THREADS_PER_TG = 128;

// Warp size for SIMD operations (Apple Silicon uses 32-wide SIMD)
constant constexpr uint SIMD_WIDTH = 32;

// Tile sizes for matrix operations
constant constexpr uint TILE_M = 16;
constant constexpr uint TILE_N = 16;
constant constexpr uint TILE_K = 16;

// ============================================================================
// Helper functions
// ============================================================================

/// Numerically stable exponential with clamping
inline float safe_exp(float x) {
    return exp(clamp(x, -88.0f, 88.0f));
}

/// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + safe_exp(-x));
}

/// Swish/SiLU activation (used in v6 gating)
inline float swish(float x) {
    return x * sigmoid(x);
}

/// Log-sum-exp for numerical stability
inline float log_add_exp(float a, float b) {
    float max_val = max(a, b);
    return max_val + log(safe_exp(a - max_val) + safe_exp(b - max_val));
}

// ============================================================================
// WKV v6 Single-token kernel
// ============================================================================
//
// Processes one token for autoregressive generation.
// Each threadgroup handles one (batch, head) pair.
//
// Inputs:
//   r: Receptance [batch, heads, head_dim]
//   k: Key [batch, heads, head_dim]
//   v: Value [batch, heads, head_dim]
//   g: Gate (v6 addition) [batch, heads, head_dim]
//   w: Time decay [heads, head_dim]
//   u: Time first bonus [heads, head_dim]
//   state: Matrix state [batch, heads, head_dim, head_dim]
//
// Outputs:
//   output: WKV output [batch, heads, head_dim]
//   new_state: Updated state [batch, heads, head_dim, head_dim]
//
kernel void rwkv_wkv_v6_single_token(
    device const half* r          [[buffer(0)]],
    device const half* k          [[buffer(1)]],
    device const half* v          [[buffer(2)]],
    device const half* g          [[buffer(3)]],  // Gate (v6)
    device const float* w         [[buffer(4)]],  // Time decay (FP32)
    device const float* u         [[buffer(5)]],  // Time first (FP32)
    device const float* state     [[buffer(6)]],  // [batch, heads, head_dim, head_dim]
    device half* output           [[buffer(7)]],
    device float* new_state       [[buffer(8)]],
    constant uint& batch_size     [[buffer(9)]],
    constant uint& num_heads      [[buffer(10)]],
    constant uint& head_dim       [[buffer(11)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tg_id.x;
    uint head_idx = tg_id.y;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Base indices
    uint rkv_base = (batch_idx * num_heads + head_idx) * head_dim;
    uint wu_base = head_idx * head_dim;
    uint state_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim;

    // Threadgroup-local storage for state matrix
    threadgroup float tg_state[MAX_HEAD_DIM * MAX_HEAD_DIM];
    threadgroup float tg_k[MAX_HEAD_DIM];
    threadgroup float tg_v[MAX_HEAD_DIM];
    threadgroup float tg_g[MAX_HEAD_DIM];
    threadgroup float tg_output[MAX_HEAD_DIM];

    // Load current state to threadgroup memory
    for (uint i = tid_in_tg; i < head_dim * head_dim; i += THREADS_PER_TG) {
        tg_state[i] = state[state_base + i];
    }

    // Load k, v, g vectors
    for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
        tg_k[i] = float(k[rkv_base + i]);
        tg_v[i] = float(v[rkv_base + i]);
        tg_g[i] = float(g[rkv_base + i]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1: Apply time decay to state: state_decayed = exp(-w) ⊙ state
    // Each thread handles a row of the state matrix
    for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
        float wi = w[wu_base + i];
        float decay = safe_exp(-wi);
        
        uint row_base = i * head_dim;
        for (uint j = 0; j < head_dim; j++) {
            tg_state[row_base + j] *= decay;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Add outer product: state += exp(k) ⊗ v^T
    // Outer product creates [head_dim, head_dim] matrix from two [head_dim] vectors
    for (uint idx = tid_in_tg; idx < head_dim * head_dim; idx += THREADS_PER_TG) {
        uint i = idx / head_dim;  // row
        uint j = idx % head_dim;  // col
        
        float kj = tg_k[j];
        float vi = tg_v[i];
        float exp_kj = safe_exp(kj);
        
        tg_state[i * head_dim + j] += vi * exp_kj;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute output = r ⊙ (state @ g)
    // Matrix-vector multiply: state @ g gives [head_dim] vector
    for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
        float sum = 0.0f;
        uint row_base = i * head_dim;
        
        // Dot product of state row with g vector
        for (uint j = 0; j < head_dim; j++) {
            sum += tg_state[row_base + j] * tg_g[j];
        }
        
        tg_output[i] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Apply receptance gating and time-first bonus
    for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
        float ri = float(r[rkv_base + i]);
        float ui = u[wu_base + i];
        float ki = tg_k[i];
        float vi = tg_v[i];
        
        // Time-first contribution: current token bonus
        float exp_uk = safe_exp(ui + ki);
        float current_contrib = exp_uk * vi;
        
        // Combine with state output
        float combined = tg_output[i] + current_contrib;
        
        // Apply receptance gating (v6 uses swish activation)
        float gated = swish(ri) * combined;
        
        output[rkv_base + i] = half(gated);
    }

    // Step 5: Write updated state back to device memory
    for (uint i = tid_in_tg; i < head_dim * head_dim; i += THREADS_PER_TG) {
        new_state[state_base + i] = tg_state[i];
    }
}

// ============================================================================
// WKV v6 Batched kernel for prefill
// ============================================================================
//
// Processes multiple tokens in sequence for prefill phase.
// More efficient than multiple single-token calls.
//
// Inputs:
//   r: Receptance [batch, seq_len, heads, head_dim]
//   k: Key [batch, seq_len, heads, head_dim]
//   v: Value [batch, seq_len, heads, head_dim]
//   g: Gate [batch, seq_len, heads, head_dim]
//   w: Time decay [heads, head_dim]
//   u: Time first [heads, head_dim]
//   initial_state: Initial state [batch, heads, head_dim, head_dim]
//
// Outputs:
//   output: WKV output [batch, seq_len, heads, head_dim]
//   final_state: Final state [batch, heads, head_dim, head_dim]
//
kernel void rwkv_wkv_v6_batched(
    device const half* r          [[buffer(0)]],
    device const half* k          [[buffer(1)]],
    device const half* v          [[buffer(2)]],
    device const half* g          [[buffer(3)]],
    device const float* w         [[buffer(4)]],
    device const float* u         [[buffer(5)]],
    device const float* initial_state [[buffer(6)]],
    device half* output           [[buffer(7)]],
    device float* final_state     [[buffer(8)]],
    constant uint& batch_size     [[buffer(9)]],
    constant uint& seq_len        [[buffer(10)]],
    constant uint& num_heads      [[buffer(11)]],
    constant uint& head_dim       [[buffer(12)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]]
) {
    uint batch_idx = tg_id.x;
    uint head_idx = tg_id.y;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Threadgroup-local state matrix
    threadgroup float tg_state[MAX_HEAD_DIM * MAX_HEAD_DIM];
    threadgroup float tg_k[MAX_HEAD_DIM];
    threadgroup float tg_v[MAX_HEAD_DIM];
    threadgroup float tg_g[MAX_HEAD_DIM];
    threadgroup float tg_output[MAX_HEAD_DIM];

    uint wu_base = head_idx * head_dim;
    uint state_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim;

    // Initialize state from initial_state
    for (uint i = tid_in_tg; i < head_dim * head_dim; i += THREADS_PER_TG) {
        tg_state[i] = initial_state[state_base + i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each token in sequence
    for (uint t = 0; t < seq_len; t++) {
        uint rkv_base = ((batch_idx * seq_len + t) * num_heads + head_idx) * head_dim;
        uint out_base = rkv_base;

        // Load current token's k, v, g
        for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
            tg_k[i] = float(k[rkv_base + i]);
            tg_v[i] = float(v[rkv_base + i]);
            tg_g[i] = float(g[rkv_base + i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply time decay
        for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
            float wi = w[wu_base + i];
            float decay = safe_exp(-wi);
            
            uint row_base = i * head_dim;
            for (uint j = 0; j < head_dim; j++) {
                tg_state[row_base + j] *= decay;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Add outer product: state += exp(k) ⊗ v^T
        for (uint idx = tid_in_tg; idx < head_dim * head_dim; idx += THREADS_PER_TG) {
            uint i = idx / head_dim;
            uint j = idx % head_dim;
            
            float kj = tg_k[j];
            float vi = tg_v[i];
            float exp_kj = safe_exp(kj);
            
            tg_state[i * head_dim + j] += vi * exp_kj;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute state @ g
        for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
            float sum = 0.0f;
            uint row_base = i * head_dim;
            
            for (uint j = 0; j < head_dim; j++) {
                sum += tg_state[row_base + j] * tg_g[j];
            }
            
            tg_output[i] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply receptance and time-first bonus
        for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
            float ri = float(r[rkv_base + i]);
            float ui = u[wu_base + i];
            float ki = tg_k[i];
            float vi = tg_v[i];
            
            float exp_uk = safe_exp(ui + ki);
            float current_contrib = exp_uk * vi;
            
            float combined = tg_output[i] + current_contrib;
            float gated = swish(ri) * combined;
            
            output[out_base + i] = half(gated);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final state
    for (uint i = tid_in_tg; i < head_dim * head_dim; i += THREADS_PER_TG) {
        final_state[state_base + i] = tg_state[i];
    }
}

// ============================================================================
// WKV v6 Optimized kernel with tiling (for larger head_dim)
// ============================================================================
//
// Uses matrix tiling for better memory access patterns and SIMD utilization.
// Best for head_dim > 64.
//
kernel void rwkv_wkv_v6_tiled(
    device const half* r          [[buffer(0)]],
    device const half* k          [[buffer(1)]],
    device const half* v          [[buffer(2)]],
    device const half* g          [[buffer(3)]],
    device const float* w         [[buffer(4)]],
    device const float* u         [[buffer(5)]],
    device const float* state     [[buffer(6)]],
    device half* output           [[buffer(7)]],
    device float* new_state       [[buffer(8)]],
    constant uint& batch_size     [[buffer(9)]],
    constant uint& num_heads      [[buffer(10)]],
    constant uint& head_dim       [[buffer(11)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg               [[thread_position_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tg_id.x;
    uint head_idx = tg_id.y;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Tile this computation for better cache behavior
    // Each threadgroup processes tiles of the state matrix
    
    uint rkv_base = (batch_idx * num_heads + head_idx) * head_dim;
    uint wu_base = head_idx * head_dim;
    uint state_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim;

    threadgroup float tg_tile_state[TILE_M * TILE_N];
    threadgroup float tg_k[MAX_HEAD_DIM];
    threadgroup float tg_v[MAX_HEAD_DIM];
    threadgroup float tg_g[MAX_HEAD_DIM];

    uint num_tiles_per_dim = (head_dim + TILE_M - 1) / TILE_M;

    // Load k, v, g (collaborative load)
    for (uint i = tid_in_tg.x + tid_in_tg.y * TILE_M; i < head_dim; i += TILE_M * TILE_N) {
        if (i < head_dim) {
            tg_k[i] = float(k[rkv_base + i]);
            tg_v[i] = float(v[rkv_base + i]);
            tg_g[i] = float(g[rkv_base + i]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process state in tiles
    for (uint tile_i = 0; tile_i < num_tiles_per_dim; tile_i++) {
        for (uint tile_j = 0; tile_j < num_tiles_per_dim; tile_j++) {
            // Load tile from global state
            uint global_i = tile_i * TILE_M + tid_in_tg.y;
            uint global_j = tile_j * TILE_N + tid_in_tg.x;
            
            if (global_i < head_dim && global_j < head_dim) {
                uint state_idx = state_base + global_i * head_dim + global_j;
                tg_tile_state[tid_in_tg.y * TILE_N + tid_in_tg.x] = state[state_idx];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Apply decay
            if (global_i < head_dim && global_j < head_dim) {
                float wi = w[wu_base + global_i];
                float decay = safe_exp(-wi);
                tg_tile_state[tid_in_tg.y * TILE_N + tid_in_tg.x] *= decay;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Add outer product contribution
            if (global_i < head_dim && global_j < head_dim) {
                float kj = tg_k[global_j];
                float vi = tg_v[global_i];
                float exp_kj = safe_exp(kj);
                tg_tile_state[tid_in_tg.y * TILE_N + tid_in_tg.x] += vi * exp_kj;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Write tile back to global state
            if (global_i < head_dim && global_j < head_dim) {
                uint state_idx = state_base + global_i * head_dim + global_j;
                new_state[state_idx] = tg_tile_state[tid_in_tg.y * TILE_N + tid_in_tg.x];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Compute output (state @ g) using tiled reduction
    threadgroup float tg_partial_sums[TILE_M];
    
    for (uint i = tid_in_tg.y; i < head_dim; i += TILE_M) {
        float sum = 0.0f;
        
        // Each thread computes partial sum for its row
        for (uint j = tid_in_tg.x; j < head_dim; j += TILE_N) {
            uint state_idx = state_base + i * head_dim + j;
            sum += new_state[state_idx] * tg_g[j];
        }
        
        // Reduce within SIMD group
        sum += simd_shuffle_down(sum, 16);
        sum += simd_shuffle_down(sum, 8);
        sum += simd_shuffle_down(sum, 4);
        sum += simd_shuffle_down(sum, 2);
        sum += simd_shuffle_down(sum, 1);
        
        if (tid_in_tg.x == 0) {
            tg_partial_sums[tid_in_tg.y] = sum;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply receptance gating
    for (uint i = tid_in_tg.y; i < head_dim; i += TILE_M) {
        if (tid_in_tg.x == 0) {
            float ri = float(r[rkv_base + i]);
            float ui = u[wu_base + i];
            float ki = tg_k[i];
            float vi = tg_v[i];
            
            float exp_uk = safe_exp(ui + ki);
            float current_contrib = exp_uk * vi;
            
            float combined = tg_partial_sums[tid_in_tg.y] + current_contrib;
            float gated = swish(ri) * combined;
            
            output[rkv_base + i] = half(gated);
        }
    }
}

// ============================================================================
// Utility kernels
// ============================================================================

/// Initialize v6 state to zeros
kernel void rwkv_wkv_v6_init_state(
    device float* state           [[buffer(0)]],  // [batch, heads, head_dim, head_dim]
    constant uint& batch_size     [[buffer(1)]],
    constant uint& num_heads      [[buffer(2)]],
    constant uint& head_dim       [[buffer(3)]],
    uint3 tid                     [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint elem_idx = tid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || elem_idx >= head_dim * head_dim) {
        return;
    }

    uint state_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim;
    state[state_base + elem_idx] = 0.0f;
}

/// Apply time decay to state (standalone operation for debugging)
kernel void rwkv_wkv_v6_apply_decay(
    device float* state           [[buffer(0)]],  // [batch, heads, head_dim, head_dim]
    device const float* w         [[buffer(1)]],  // [heads, head_dim]
    constant uint& batch_size     [[buffer(2)]],
    constant uint& num_heads      [[buffer(3)]],
    constant uint& head_dim       [[buffer(4)]],
    uint3 tid                     [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint row_idx = tid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || row_idx >= head_dim) {
        return;
    }

    float wi = w[head_idx * head_dim + row_idx];
    float decay = safe_exp(-wi);

    uint state_row_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim
                         + row_idx * head_dim;

    for (uint j = 0; j < head_dim; j++) {
        state[state_row_base + j] *= decay;
    }
}

/// Test kernel: Compute outer product k ⊗ v^T
kernel void test_outer_product(
    device const half* k          [[buffer(0)]],  // [head_dim]
    device const half* v          [[buffer(1)]],  // [head_dim]
    device float* output          [[buffer(2)]],  // [head_dim, head_dim]
    constant uint& head_dim       [[buffer(3)]],
    uint2 tid                     [[thread_position_in_grid]]
) {
    uint i = tid.y;  // row
    uint j = tid.x;  // col

    if (i >= head_dim || j >= head_dim) return;

    float kj = float(k[j]);
    float vi = float(v[i]);
    
    output[i * head_dim + j] = vi * safe_exp(kj);
}

/// Test kernel: Matrix-vector multiply
kernel void test_matvec_multiply(
    device const float* matrix    [[buffer(0)]],  // [head_dim, head_dim]
    device const half* vector     [[buffer(1)]],  // [head_dim]
    device half* output           [[buffer(2)]],  // [head_dim]
    constant uint& head_dim       [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid >= head_dim) return;

    float sum = 0.0f;
    uint row_base = tid * head_dim;
    
    for (uint j = 0; j < head_dim; j++) {
        sum += matrix[row_base + j] * float(vector[j]);
    }
    
    output[tid] = half(sum);
}
