#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RWKV WKV (Weighted Key-Value) Kernel for Apple Metal
// ============================================================================
//
// The WKV operator is the core of RWKV's linear attention mechanism.
// It computes a weighted sum of values where weights decay exponentially
// over time:
//
//   wkv_t = Σ_{j<t} exp(-(t-j)*w + k_j) * v_j  /  Σ_{j<t} exp(-(t-j)*w + k_j)
//
// This can be computed efficiently with O(1) state update per token:
//   state_t = decay * state_{t-1} + exp(k_t) * v_t
//   output_t = state_t / normalizer_t
//
// The kernel supports:
// - Multi-head operation (independent state per head)
// - FP32 state accumulation for numerical stability
// - FP16/BF16 input/output for memory efficiency
//
// ============================================================================

// ============================================================================
// Constants
// ============================================================================

// Maximum head dimension for threadgroup memory allocation
constant constexpr uint MAX_HEAD_DIM = 128;

// Threads per threadgroup - tuned for Apple Silicon
constant constexpr uint THREADS_PER_TG = 128;

// ============================================================================
// Helper functions
// ============================================================================

/// Numerically stable log-sum-exp for two values
inline float log_add_exp(float a, float b) {
    float max_val = max(a, b);
    return max_val + log(exp(a - max_val) + exp(b - max_val));
}

/// Clamp to prevent exp overflow
inline float safe_exp(float x) {
    return exp(clamp(x, -88.0f, 88.0f));
}

// ============================================================================
// Single-token WKV update kernel
// ============================================================================
//
// This kernel processes one token at a time for autoregressive generation.
// Each threadgroup handles one (batch, head) pair.
//
// Inputs:
//   r: Receptance [batch, heads, head_dim]
//   k: Key [batch, heads, head_dim]
//   v: Value [batch, heads, head_dim]
//   w: Time decay [heads, head_dim]
//   u: Time first bonus [heads, head_dim]
//   state: Running state [batch, heads, head_dim, head_dim]
//   state_denom: Running denominator [batch, heads, head_dim]
//
// Outputs:
//   output: WKV output [batch, heads, head_dim]
//   new_state: Updated state [batch, heads, head_dim, head_dim]
//   new_state_denom: Updated denominator [batch, heads, head_dim]
//
kernel void rwkv_wkv_single_token(
    device const half* r          [[buffer(0)]],
    device const half* k          [[buffer(1)]],
    device const half* v          [[buffer(2)]],
    device const float* w         [[buffer(3)]],  // Time decay (FP32 for precision)
    device const float* u         [[buffer(4)]],  // Time first (FP32 for precision)
    device const float* state     [[buffer(5)]],  // [batch, heads, head_dim, head_dim]
    device const float* state_denom [[buffer(6)]],  // [batch, heads, head_dim]
    device half* output           [[buffer(7)]],
    device float* new_state       [[buffer(8)]],
    device float* new_state_denom [[buffer(9)]],
    constant uint& batch_size     [[buffer(10)]],
    constant uint& num_heads      [[buffer(11)]],
    constant uint& head_dim       [[buffer(12)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]]
) {
    // tg_id.x = batch index, tg_id.y = head index
    uint batch_idx = tg_id.x;
    uint head_idx = tg_id.y;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Base indices for this (batch, head)
    uint rkv_base = (batch_idx * num_heads + head_idx) * head_dim;
    uint wu_base = head_idx * head_dim;
    uint state_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim;
    uint denom_base = (batch_idx * num_heads + head_idx) * head_dim;

    // Each thread handles one or more head_dim elements
    for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
        // Load current token's values
        float ri = float(r[rkv_base + i]);
        float ki = float(k[rkv_base + i]);
        float vi = float(v[rkv_base + i]);
        float wi = w[wu_base + i];
        float ui = u[wu_base + i];

        // Compute decay factor
        float decay = safe_exp(-wi);

        // Compute current token's contribution bonus
        float exp_uk = safe_exp(ui + ki);

        // Load and update state for this output dimension
        float output_val = 0.0f;
        float denom_val = 0.0f;

        // State is [head_dim, head_dim]: sum over source dimension j
        for (uint j = 0; j < head_dim; j++) {
            uint state_idx = state_base + i * head_dim + j;
            float old_state = state[state_idx];

            // Load v_j and k_j for the outer product update
            float kj = float(k[rkv_base + j]);
            float vj = float(v[rkv_base + j]);
            float exp_kj = safe_exp(kj);

            // Accumulate for output: weighted sum from state
            output_val += old_state;

            // Update state: decay old + add new contribution
            float new_state_val = decay * old_state + vi * exp_kj;
            new_state[state_idx] = new_state_val;
        }

        // Handle denominator
        float old_denom = state_denom[denom_base + i];
        float exp_ki = safe_exp(ki);

        // Output with current token bonus
        output_val += exp_uk * vi;
        denom_val = old_denom + exp_uk;

        // Normalize and apply receptance
        float normalized = output_val / max(denom_val, 1e-8f);
        float gated = ri * (1.0f / (1.0f + safe_exp(-ri))) * normalized;

        output[rkv_base + i] = half(gated);

        // Update denominator
        new_state_denom[denom_base + i] = decay * old_denom + exp_ki;
    }
}

// ============================================================================
// Batched WKV kernel for prefill
// ============================================================================
//
// Processes multiple tokens in sequence. More efficient for prefill since
// we can pipeline state updates across the sequence.
//
// Inputs:
//   r: Receptance [batch, seq_len, heads, head_dim]
//   k: Key [batch, seq_len, heads, head_dim]
//   v: Value [batch, seq_len, heads, head_dim]
//   w: Time decay [heads, head_dim]
//   u: Time first bonus [heads, head_dim]
//   initial_state: Initial state [batch, heads, head_dim, head_dim]
//   initial_denom: Initial denominator [batch, heads, head_dim]
//
// Outputs:
//   output: WKV output [batch, seq_len, heads, head_dim]
//   final_state: Final state [batch, heads, head_dim, head_dim]
//   final_denom: Final denominator [batch, heads, head_dim]
//
kernel void rwkv_wkv_batched(
    device const half* r          [[buffer(0)]],
    device const half* k          [[buffer(1)]],
    device const half* v          [[buffer(2)]],
    device const float* w         [[buffer(3)]],
    device const float* u         [[buffer(4)]],
    device const float* initial_state [[buffer(5)]],
    device const float* initial_denom [[buffer(6)]],
    device half* output           [[buffer(7)]],
    device float* final_state     [[buffer(8)]],
    device float* final_denom     [[buffer(9)]],
    constant uint& batch_size     [[buffer(10)]],
    constant uint& seq_len        [[buffer(11)]],
    constant uint& num_heads      [[buffer(12)]],
    constant uint& head_dim       [[buffer(13)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]]
) {
    // tg_id.x = batch index, tg_id.y = head index
    uint batch_idx = tg_id.x;
    uint head_idx = tg_id.y;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Threadgroup-local state (one per head dimension element)
    // We use float for accumulation precision
    threadgroup float tg_state[MAX_HEAD_DIM * MAX_HEAD_DIM];
    threadgroup float tg_denom[MAX_HEAD_DIM];

    uint wu_base = head_idx * head_dim;
    uint state_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim;
    uint denom_base = (batch_idx * num_heads + head_idx) * head_dim;

    // Initialize threadgroup state from initial state
    for (uint i = tid_in_tg; i < head_dim * head_dim; i += THREADS_PER_TG) {
        tg_state[i] = initial_state[state_base + i];
    }
    for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
        tg_denom[i] = initial_denom[denom_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process sequence tokens
    for (uint t = 0; t < seq_len; t++) {
        uint rkv_base = ((batch_idx * seq_len + t) * num_heads + head_idx) * head_dim;
        uint out_base = rkv_base;  // Same layout for output

        // Load w and u (same for all tokens)
        // Each thread handles subset of head_dim
        for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
            float ri = float(r[rkv_base + i]);
            float ki = float(k[rkv_base + i]);
            float vi = float(v[rkv_base + i]);
            float wi = w[wu_base + i];
            float ui = u[wu_base + i];

            float decay = safe_exp(-wi);
            float exp_uk = safe_exp(ui + ki);
            float exp_ki = safe_exp(ki);

            // Compute output: sum over state + current bonus
            float output_val = 0.0f;
            for (uint j = 0; j < head_dim; j++) {
                output_val += tg_state[i * head_dim + j];
            }
            output_val += exp_uk * vi;

            float denom_val = tg_denom[i] + exp_uk;
            float normalized = output_val / max(denom_val, 1e-8f);
            float gated = ri * (1.0f / (1.0f + safe_exp(-ri))) * normalized;

            output[out_base + i] = half(gated);

            // Update state for next token
            for (uint j = 0; j < head_dim; j++) {
                float kj = float(k[rkv_base + j]);
                float exp_kj = safe_exp(kj);
                uint state_idx = i * head_dim + j;
                tg_state[state_idx] = decay * tg_state[state_idx] + vi * exp_kj;
            }
            tg_denom[i] = decay * tg_denom[i] + exp_ki;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final state back to device memory
    for (uint i = tid_in_tg; i < head_dim * head_dim; i += THREADS_PER_TG) {
        final_state[state_base + i] = tg_state[i];
    }
    for (uint i = tid_in_tg; i < head_dim; i += THREADS_PER_TG) {
        final_denom[denom_base + i] = tg_denom[i];
    }
}

// ============================================================================
// Time decay application kernel
// ============================================================================
//
// Applies exponential time decay to state. Can be fused with other operations
// or used standalone for debugging.
//
kernel void rwkv_apply_time_decay(
    device float* state           [[buffer(0)]],  // [batch, heads, head_dim, head_dim]
    device float* state_denom     [[buffer(1)]],  // [batch, heads, head_dim]
    device const float* w         [[buffer(2)]],  // [heads, head_dim]
    constant uint& batch_size     [[buffer(3)]],
    constant uint& num_heads      [[buffer(4)]],
    constant uint& head_dim       [[buffer(5)]],
    uint3 tid                     [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint dim_idx = tid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }

    float wi = w[head_idx * head_dim + dim_idx];
    float decay = safe_exp(-wi);

    // Decay state row
    uint state_row_base = (batch_idx * num_heads + head_idx) * head_dim * head_dim
                         + dim_idx * head_dim;
    for (uint j = 0; j < head_dim; j++) {
        state[state_row_base + j] *= decay;
    }

    // Decay denominator
    uint denom_idx = (batch_idx * num_heads + head_idx) * head_dim + dim_idx;
    state_denom[denom_idx] *= decay;
}

// ============================================================================
// Token shift kernel
// ============================================================================
//
// Mixes current token with previous token using learned mixing ratios.
// This is a key RWKV operation that provides temporal context efficiently.
//
// output[i] = x[i] * mix + x_prev[i] * (1 - mix)
//
// For sequence processing:
//   output[0] = x[0] * mix + x_prev * (1 - mix)
//   output[t] = x[t] * mix + x[t-1] * (1 - mix), for t > 0
//
kernel void rwkv_token_shift(
    device const half* x          [[buffer(0)]],  // [batch, seq_len, hidden]
    device const half* x_prev     [[buffer(1)]],  // [batch, hidden]
    device const half* mix_ratio  [[buffer(2)]],  // [hidden]
    device half* output           [[buffer(3)]],  // [batch, seq_len, hidden]
    constant uint& batch_size     [[buffer(4)]],
    constant uint& seq_len        [[buffer(5)]],
    constant uint& hidden_size    [[buffer(6)]],
    uint3 tid                     [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint seq_idx = tid.y;
    uint hidden_idx = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }

    float mix = float(mix_ratio[hidden_idx]);
    float one_minus_mix = 1.0f - mix;

    uint curr_idx = (batch_idx * seq_len + seq_idx) * hidden_size + hidden_idx;
    float x_curr = float(x[curr_idx]);

    float x_shifted;
    if (seq_idx == 0) {
        // Use x_prev for first token
        uint prev_idx = batch_idx * hidden_size + hidden_idx;
        x_shifted = float(x_prev[prev_idx]);
    } else {
        // Use previous token in sequence
        uint prev_idx = (batch_idx * seq_len + seq_idx - 1) * hidden_size + hidden_idx;
        x_shifted = float(x[prev_idx]);
    }

    output[curr_idx] = half(x_curr * mix + x_shifted * one_minus_mix);
}

// ============================================================================
// Squared ReLU kernel (for channel mixing)
// ============================================================================
//
// RWKV uses squared ReLU in its FFN: relu(x)^2
// This provides stronger feature selection than standard activations.
//
kernel void rwkv_squared_relu(
    device const half* input      [[buffer(0)]],
    device half* output           [[buffer(1)]],
    constant uint& num_elements   [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;

    float x = float(input[tid]);
    float relu_x = max(x, 0.0f);
    output[tid] = half(relu_x * relu_x);
}

// ============================================================================
// Test kernels
// ============================================================================

/// Test kernel: verify time decay computation
kernel void test_time_decay(
    device const float* w         [[buffer(0)]],
    device float* decay_output    [[buffer(1)]],
    constant uint& num_elements   [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;
    decay_output[tid] = safe_exp(-w[tid]);
}

/// Test kernel: verify token shift for single token
kernel void test_token_shift_single(
    device const half* x_curr     [[buffer(0)]],
    device const half* x_prev     [[buffer(1)]],
    device const half* mix        [[buffer(2)]],
    device half* output           [[buffer(3)]],
    constant uint& hidden_size    [[buffer(4)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid >= hidden_size) return;

    float m = float(mix[tid]);
    float curr = float(x_curr[tid]);
    float prev = float(x_prev[tid]);

    output[tid] = half(curr * m + prev * (1.0f - m));
}

/// Test kernel: verify WKV state update for a single element
kernel void test_wkv_state_update(
    device const float* old_state [[buffer(0)]],  // Single state value
    device const float* decay     [[buffer(1)]],  // Single decay value
    device const float* k         [[buffer(2)]],  // Single key value
    device const float* v         [[buffer(3)]],  // Single value
    device float* new_state       [[buffer(4)]],  // Output state
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid > 0) return;

    float d = decay[0];
    float exp_k = safe_exp(k[0]);

    new_state[0] = d * old_state[0] + v[0] * exp_k;
}
