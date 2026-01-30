// layernorm.metal - Layer Normalization kernels for LLMs
//
// Implements both RMSNorm (used by Llama/GLM) and standard LayerNorm (used by GPT).
// Uses simdgroup reduction for efficient mean/variance computation.
//
// Supported hidden dimensions: up to 16384 (GLM-4)
//
// Kernels:
//   1. rmsnorm                 - Root Mean Square Layer Normalization
//   2. layernorm               - Standard Layer Normalization  
//   3. rmsnorm_fused_residual  - RMSNorm with fused residual add
//
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Simdgroup reduction utilities
// ============================================================================

inline float simd_sum_f32(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// ============================================================================
// RMSNorm (Root Mean Square Layer Normalization)
//
// Formula: y = x / rms(x) * gamma
// where rms(x) = sqrt(mean(x^2) + eps)
//
// Used by: Llama, GLM, Mistral, and other modern LLMs
//
// Each threadgroup processes one token (row) cooperatively.
// Uses simdgroup reduction for efficient sum-of-squares computation.
//
// Arguments:
//   input   - [num_tokens, hidden_dim] row-major
//   gamma   - [hidden_dim] scale weights (no bias in RMSNorm)
//   output  - [num_tokens, hidden_dim] normalized output
//   num_tokens    - batch size * sequence length
//   hidden_dim    - hidden dimension (up to 16384)
//   eps           - small constant for numerical stability
// ============================================================================

kernel void rmsnorm(
    device const half* input        [[buffer(0)]],
    device const half* gamma        [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& num_tokens       [[buffer(3)]],
    constant uint& hidden_dim       [[buffer(4)]],
    constant float& eps             [[buffer(5)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]],
    uint lane_id                  [[thread_index_in_simdgroup]],
    uint sg_id                    [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= num_tokens) return;
    
    const uint num_simdgroups = 8;  // Assumes 256 threads = 8 simdgroups
    const uint SIMDGROUP_SIZE = 32;
    
    // Each simdgroup handles a portion of the hidden_dim
    // First pass: compute sum of squares for this simdgroup's elements
    float sum_sq = 0.0f;
    const uint elems_per_lane = (hidden_dim + num_simdgroups * SIMDGROUP_SIZE - 1) / (num_simdgroups * SIMDGROUP_SIZE);
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        if (k_idx < hidden_dim) {
            float val = float(input[token_idx * hidden_dim + k_idx]);
            sum_sq += val * val;
        }
    }
    
    // Reduce sum of squares across all threads in the simdgroup
    sum_sq = simd_sum_f32(sum_sq);
    
    // Store simdgroup sums in threadgroup memory for final reduction
    threadgroup float sg_sums[8];
    if (lane_id == 0) {
        sg_sums[sg_id] = sum_sq;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction: simdgroup 0 sums up all simdgroup results
    float rms_inv = 1.0f;
    if (sg_id == 0) {
        float total_sum = 0.0f;
        if (lane_id < num_simdgroups) {
            total_sum = sg_sums[lane_id];
        }
        total_sum = simd_sum_f32(total_sum);
        
        if (lane_id == 0) {
            float variance = total_sum / float(hidden_dim);
            rms_inv = rsqrt(variance + eps);
            sg_sums[0] = rms_inv;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    rms_inv = sg_sums[0];
    
    // Second pass: normalize and write output
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        if (k_idx < hidden_dim) {
            float x = float(input[token_idx * hidden_dim + k_idx]);
            float g = float(gamma[k_idx]);
            output[token_idx * hidden_dim + k_idx] = half(x * rms_inv * g);
        }
    }
}

// ============================================================================
// Standard Layer Normalization
//
// Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
// where mean = mean(x)
//       var  = mean((x - mean)^2)
//
// Used by: GPT, BERT, and other transformer models
//
// Two-pass algorithm:
//   Pass 1: Compute mean
//   Pass 2: Compute variance
//   Pass 3: Normalize with gamma and beta
//
// Arguments:
//   input   - [num_tokens, hidden_dim] row-major
//   gamma   - [hidden_dim] scale weights
//   beta    - [hidden_dim] bias weights
//   output  - [num_tokens, hidden_dim] normalized output
// ============================================================================

kernel void layernorm(
    device const half* input        [[buffer(0)]],
    device const half* gamma        [[buffer(1)]],
    device const half* beta         [[buffer(2)]],
    device half* output             [[buffer(3)]],
    constant uint& num_tokens       [[buffer(4)]],
    constant uint& hidden_dim       [[buffer(5)]],
    constant float& eps             [[buffer(6)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]],
    uint lane_id                  [[thread_index_in_simdgroup]],
    uint sg_id                    [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= num_tokens) return;
    
    const uint num_simdgroups = 8;
    const uint SIMDGROUP_SIZE = 32;
    
    threadgroup float shared_stats[2];  // [mean, inv_std]
    threadgroup float sg_sums[8];
    
    const uint elems_per_lane = (hidden_dim + num_simdgroups * SIMDGROUP_SIZE - 1) / (num_simdgroups * SIMDGROUP_SIZE);
    
    // Pass 1: Compute mean
    float local_sum = 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        if (k_idx < hidden_dim) {
            local_sum += float(input[token_idx * hidden_dim + k_idx]);
        }
    }
    
    local_sum = simd_sum_f32(local_sum);
    
    if (lane_id == 0) {
        sg_sums[sg_id] = local_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float mean = 0.0f;
    if (sg_id == 0) {
        float total = 0.0f;
        if (lane_id < num_simdgroups) {
            total = sg_sums[lane_id];
        }
        total = simd_sum_f32(total);
        
        if (lane_id == 0) {
            mean = total / float(hidden_dim);
            shared_stats[0] = mean;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = shared_stats[0];
    
    // Pass 2: Compute variance
    float local_sq_diff = 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        if (k_idx < hidden_dim) {
            float diff = float(input[token_idx * hidden_dim + k_idx]) - mean;
            local_sq_diff += diff * diff;
        }
    }
    
    local_sq_diff = simd_sum_f32(local_sq_diff);
    
    if (lane_id == 0) {
        sg_sums[sg_id] = local_sq_diff;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_std = 1.0f;
    if (sg_id == 0) {
        float total_var = 0.0f;
        if (lane_id < num_simdgroups) {
            total_var = sg_sums[lane_id];
        }
        total_var = simd_sum_f32(total_var);
        
        if (lane_id == 0) {
            float variance = total_var / float(hidden_dim);
            inv_std = rsqrt(variance + eps);
            shared_stats[1] = inv_std;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_std = shared_stats[1];
    
    // Pass 3: Normalize, scale, and shift
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        if (k_idx < hidden_dim) {
            float x = float(input[token_idx * hidden_dim + k_idx]);
            float g = float(gamma[k_idx]);
            float b = float(beta[k_idx]);
            float normalized = (x - mean) * inv_std;
            output[token_idx * hidden_dim + k_idx] = half(normalized * g + b);
        }
    }
}

// ============================================================================
// RMSNorm with Fused Residual Connection
//
// Fused kernel that performs:
//   1. new_residual = x + residual
//   2. normed = RMSNorm(new_residual)
//
// This eliminates an intermediate memory write for the residual add,
// reducing memory bandwidth by ~33% for this operation sequence.
//
// Common in transformer layers:
//   - Pre-norm:  x = x + Attention(RMSNorm(x))
//   - Post-norm: x = RMSNorm(x + Attention(x))
//
// Arguments:
//   x           - [num_tokens, hidden_dim] input activation
//   residual    - [num_tokens, hidden_dim] residual connection
//   gamma       - [hidden_dim] RMSNorm scale weights
//   normed_out  - [num_tokens, hidden_dim] normalized output
//   residual_out- [num_tokens, hidden_dim] x + residual (for next layer)
// ============================================================================

kernel void rmsnorm_fused_residual(
    device const half* x            [[buffer(0)]],
    device const half* residual     [[buffer(1)]],
    device const half* gamma        [[buffer(2)]],
    device half* normed_out         [[buffer(3)]],
    device half* residual_out       [[buffer(4)]],
    constant uint& num_tokens       [[buffer(5)]],
    constant uint& hidden_dim       [[buffer(6)]],
    constant float& eps             [[buffer(7)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]],
    uint lane_id                  [[thread_index_in_simdgroup]],
    uint sg_id                    [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= num_tokens) return;
    
    const uint num_simdgroups = 8;
    const uint SIMDGROUP_SIZE = 32;
    
    threadgroup float sg_sums[8];
    const uint elems_per_lane = (hidden_dim + num_simdgroups * SIMDGROUP_SIZE - 1) / (num_simdgroups * SIMDGROUP_SIZE);
    
    // First pass: compute residual add and sum of squares in one go
    float sum_sq = 0.0f;
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        if (k_idx < hidden_dim) {
            uint offset = token_idx * hidden_dim + k_idx;
            
            // Compute residual add
            float x_val = float(x[offset]);
            float res_val = float(residual[offset]);
            float new_res = x_val + res_val;
            
            // Store residual output
            residual_out[offset] = half(new_res);
            
            // Accumulate sum of squares for RMS computation
            sum_sq += new_res * new_res;
        }
    }
    
    // Reduce sum of squares across simdgroup
    sum_sq = simd_sum_f32(sum_sq);
    
    if (lane_id == 0) {
        sg_sums[sg_id] = sum_sq;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction to compute RMS inverse
    float rms_inv = 1.0f;
    if (sg_id == 0) {
        float total_sum = 0.0f;
        if (lane_id < num_simdgroups) {
            total_sum = sg_sums[lane_id];
        }
        total_sum = simd_sum_f32(total_sum);
        
        if (lane_id == 0) {
            float variance = total_sum / float(hidden_dim);
            rms_inv = rsqrt(variance + eps);
            sg_sums[0] = rms_inv;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    rms_inv = sg_sums[0];
    
    // Second pass: normalize and write output
    // Note: we read from residual_out since it contains x + residual
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        if (k_idx < hidden_dim) {
            uint offset = token_idx * hidden_dim + k_idx;
            float val = float(residual_out[offset]);
            float g = float(gamma[k_idx]);
            normed_out[offset] = half(val * rms_inv * g);
        }
    }
}

// ============================================================================
// Vectorized variants for improved memory throughput
//
// These variants use half4 loads/stores when hidden_dim is divisible by 4,
// improving memory bandwidth utilization.
// ============================================================================

kernel void rmsnorm_vec4(
    device const half* input        [[buffer(0)]],
    device const half* gamma        [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& num_tokens       [[buffer(3)]],
    constant uint& hidden_dim       [[buffer(4)]],
    constant float& eps             [[buffer(5)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint tid_in_tg                [[thread_index_in_threadgroup]],
    uint lane_id                  [[thread_index_in_simdgroup]],
    uint sg_id                    [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= num_tokens) return;
    
    const uint num_simdgroups = 8;
    const uint SIMDGROUP_SIZE = 32;
    
    threadgroup float sg_sums[8];
    // Process 4 elements at a time
    const uint vec_hidden_dim = hidden_dim / 4;
    const uint elems_per_lane = (vec_hidden_dim + num_simdgroups * SIMDGROUP_SIZE - 1) / (num_simdgroups * SIMDGROUP_SIZE);
    
    // First pass: compute sum of squares using half4
    float sum_sq = 0.0f;
    device const half4* input_vec = (device const half4*)input;
    device const half4* gamma_vec = (device const half4*)gamma;
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint vec_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        uint base_k = vec_idx * 4;
        if (base_k + 3 < hidden_dim) {
            half4 x = input_vec[token_idx * vec_hidden_dim + vec_idx];
            sum_sq += float(x.x) * float(x.x);
            sum_sq += float(x.y) * float(x.y);
            sum_sq += float(x.z) * float(x.z);
            sum_sq += float(x.w) * float(x.w);
        }
    }
    
    sum_sq = simd_sum_f32(sum_sq);
    
    if (lane_id == 0) {
        sg_sums[sg_id] = sum_sq;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float rms_inv = 1.0f;
    if (sg_id == 0) {
        float total_sum = 0.0f;
        if (lane_id < num_simdgroups) {
            total_sum = sg_sums[lane_id];
        }
        total_sum = simd_sum_f32(total_sum);
        
        if (lane_id == 0) {
            float variance = total_sum / float(hidden_dim);
            rms_inv = rsqrt(variance + eps);
            sg_sums[0] = rms_inv;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    rms_inv = sg_sums[0];
    
    // Second pass: normalize and write using half4
    device half4* output_vec = (device half4*)output;
    
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint vec_idx = (sg_id * SIMDGROUP_SIZE + lane_id) + i * (num_simdgroups * SIMDGROUP_SIZE);
        uint base_k = vec_idx * 4;
        if (base_k + 3 < hidden_dim) {
            half4 x = input_vec[token_idx * vec_hidden_dim + vec_idx];
            half4 g = gamma_vec[vec_idx];
            half4 y;
            y.x = half(float(x.x) * rms_inv * float(g.x));
            y.y = half(float(x.y) * rms_inv * float(g.y));
            y.z = half(float(x.z) * rms_inv * float(g.z));
            y.w = half(float(x.w) * rms_inv * float(g.w));
            output_vec[token_idx * vec_hidden_dim + vec_idx] = y;
        }
    }
}

// ============================================================================
// Optimized RMSNorm for small hidden dimensions (<= 2048)
//
// Uses a single simdgroup per token for lower latency when hidden_dim
// is small enough that multiple simdgroups aren't needed for parallelism.
// ============================================================================

kernel void rmsnorm_small(
    device const half* input        [[buffer(0)]],
    device const half* gamma        [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& num_tokens       [[buffer(3)]],
    constant uint& hidden_dim       [[buffer(4)]],
    constant float& eps             [[buffer(5)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint lane_id                  [[thread_index_in_simdgroup]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= num_tokens) return;
    
    // Each token is processed by one simdgroup (32 threads)
    const uint elems_per_lane = (hidden_dim + 31) / 32;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = lane_id + i * 32;
        if (k_idx < hidden_dim) {
            float val = float(input[token_idx * hidden_dim + k_idx]);
            sum_sq += val * val;
        }
    }
    
    // Reduce within simdgroup
    sum_sq = simd_sum_f32(sum_sq);
    
    // Compute RMS
    float rms_inv = rsqrt(sum_sq / float(hidden_dim) + eps);
    
    // Normalize and write
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint k_idx = lane_id + i * 32;
        if (k_idx < hidden_dim) {
            float x = float(input[token_idx * hidden_dim + k_idx]);
            float g = float(gamma[k_idx]);
            output[token_idx * hidden_dim + k_idx] = half(x * rms_inv * g);
        }
    }
}
