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

constant uint VEC_ELEMS_PER_LANE = 16;
constant uint TILE_SIZE = 8;
constant uint CHUNK_SIZE = 1024;

// ============================================================================
// Simdgroup reduction utilities
//
// Using Metal's built-in simd_sum intrinsic for hardware-accelerated reduction.
// This is significantly faster than manual simd_shuffle_xor chains because:
// 1. Single instruction vs 5 dependent instructions
// 2. Dedicated hardware support on Apple Silicon
// 3. Lower latency and energy consumption
// ============================================================================

inline float simd_sum_f32(float val) {
    return simd_sum(val);
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

// ============================================================================
// Multi-pass RMSNorm for large hidden dimensions (>= 8192)
//
// For large hidden dimensions (GLM-4: 12288, 16384), this kernel:
// 1. Processes data in multiple passes to avoid register pressure
// 2. Uses vectorized (half4) loads for better memory throughput
// 3. Distributes work evenly across all threads in the threadgroup
// 4. Accumulates partial sums across passes before final reduction
//
// Each pass processes a tile of elements per thread (TILE_SIZE = 16 half4 = 64 elems).
// This ensures:
// - Fits comfortably in registers for all passes
// - Coalesced memory access patterns (contiguous loads)
// - Better L1/L2 cache utilization through locality
// - Work is evenly distributed across all 256 threads
// ============================================================================
// Tile and chunk sizes for multi-pass reduction

kernel void rmsnorm_multipass(
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

    const uint SIMDGROUP_SIZE = 32;
    const uint num_simdgroups = 8;
    const uint TOTAL_THREADS = num_simdgroups * SIMDGROUP_SIZE;
    const uint tid = sg_id * SIMDGROUP_SIZE + lane_id;

    // Threadgroup memory for inter-simdgroup reduction (8 simdgroups)
    threadgroup float sg_partial_sums[8];

    // Compute number of half4 elements (4 float per half4)
    const uint vec_hidden_dim = (hidden_dim + 3) / 4;

    // Each thread processes TILE_SIZE half4 elements per pass
    // Total elements covered per pass = TOTAL_THREADS * TILE_SIZE * 4
    const uint vec_elements_per_pass = TOTAL_THREADS * TILE_SIZE;
    const uint num_passes = (vec_hidden_dim + vec_elements_per_pass - 1) / vec_elements_per_pass;

    device const half4* input_vec = reinterpret_cast<device const half4*>(input + token_idx * hidden_dim);
    device const half4* gamma_vec = reinterpret_cast<device const half4*>(gamma);

    // Pass 1: Compute sum of squares across multiple passes
    float sum_sq = 0.0f;

    for (uint pass = 0; pass < num_passes; ++pass) {
        // Each thread processes TILE_SIZE half4 elements starting at pass offset
        const uint vec_base = pass * vec_elements_per_pass + tid * TILE_SIZE;

        #pragma unroll
        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint vec_idx = vec_base + i;
            if (vec_idx < vec_hidden_dim) {
                const uint elem_base = vec_idx * 4;
                if (elem_base + 3 < hidden_dim) {
                    half4 x = input_vec[vec_idx];
                    sum_sq += float(x.x) * float(x.x);
                    sum_sq += float(x.y) * float(x.y);
                    sum_sq += float(x.z) * float(x.z);
                    sum_sq += float(x.w) * float(x.w);
                } else {
                    // Handle remaining elements that don't form a full half4
                    for (uint j = 0; elem_base + j < hidden_dim; ++j) {
                        float val = float(input[token_idx * hidden_dim + elem_base + j]);
                        sum_sq += val * val;
                    }
                }
            }
        }
    }

    // Reduce within simdgroup
    sum_sq = simd_sum_f32(sum_sq);

    // First-level reduction across simdgroups
    if (lane_id == 0) {
        sg_partial_sums[sg_id] = sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second-level: simdgroup 0 performs final reduction
    float rms_inv = 1.0f;
    if (sg_id == 0) {
        float total_sum = 0.0f;
        if (lane_id < num_simdgroups) {
            total_sum = sg_partial_sums[lane_id];
        }
        total_sum = simd_sum_f32(total_sum);

        if (lane_id == 0) {
            float variance = total_sum / float(hidden_dim);
            rms_inv = rsqrt(variance + eps);
            sg_partial_sums[0] = rms_inv;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    rms_inv = sg_partial_sums[0];

    // Pass 2+: Normalize and write output (reuse same pass strategy)
    device half4* output_vec = reinterpret_cast<device half4*>(output + token_idx * hidden_dim);

    for (uint pass = 0; pass < num_passes; ++pass) {
        const uint vec_base = pass * vec_elements_per_pass + tid * TILE_SIZE;

        #pragma unroll
        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint vec_idx = vec_base + i;
            if (vec_idx < vec_hidden_dim) {
                const uint elem_base = vec_idx * 4;
                if (elem_base + 3 < hidden_dim) {
                    half4 x = input_vec[vec_idx];
                    half4 g = gamma_vec[vec_idx];
                    half4 y;
                    y.x = half(float(x.x) * rms_inv * float(g.x));
                    y.y = half(float(x.y) * rms_inv * float(g.y));
                    y.z = half(float(x.z) * rms_inv * float(g.z));
                    y.w = half(float(x.w) * rms_inv * float(g.w));
                    output_vec[vec_idx] = y;
                } else {
                    // Handle remaining elements
                    for (uint j = 0; elem_base + j < hidden_dim; ++j) {
                        float x = float(input[token_idx * hidden_dim + elem_base + j]);
                        float g = float(gamma[elem_base + j]);
                        output[token_idx * hidden_dim + elem_base + j] = half(x * rms_inv * g);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Multi-pass LayerNorm for large hidden dimensions
//
// Similar to rmsnorm_multipass but for standard LayerNorm with mean and variance.
// Uses the same tile-based pass strategy to handle large dimensions efficiently.
//
// Three-pass algorithm:
//   Pass 1: Compute mean across all elements
//   Pass 2: Compute variance using mean from Pass 1
//   Pass 3: Normalize with gamma and beta
// ============================================================================

kernel void layernorm_multipass(
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

    const uint SIMDGROUP_SIZE = 32;
    const uint num_simdgroups = 8;
    const uint TOTAL_THREADS = num_simdgroups * SIMDGROUP_SIZE;
    const uint tid = sg_id * SIMDGROUP_SIZE + lane_id;

    threadgroup float sg_partial_sums[8];
    threadgroup float final_stats[2];  // [mean, inv_std]

    // Compute number of half4 elements (4 float per half4)
    const uint vec_hidden_dim = (hidden_dim + 3) / 4;
    const uint vec_elements_per_pass = TOTAL_THREADS * TILE_SIZE;
    const uint num_passes = (vec_hidden_dim + vec_elements_per_pass - 1) / vec_elements_per_pass;

    device const half4* input_vec = reinterpret_cast<device const half4*>(input + token_idx * hidden_dim);

    // Pass 1: Compute mean
    float sum = 0.0f;
    for (uint pass = 0; pass < num_passes; ++pass) {
        const uint vec_base = pass * vec_elements_per_pass + tid * TILE_SIZE;

        #pragma unroll
        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint vec_idx = vec_base + i;
            if (vec_idx < vec_hidden_dim) {
                const uint elem_base = vec_idx * 4;
                if (elem_base + 3 < hidden_dim) {
                    half4 x = input_vec[vec_idx];
                    sum += float(x.x) + float(x.y) + float(x.z) + float(x.w);
                } else {
                    for (uint j = 0; elem_base + j < hidden_dim; ++j) {
                        sum += float(input[token_idx * hidden_dim + elem_base + j]);
                    }
                }
            }
        }
    }

    sum = simd_sum_f32(sum);

    if (lane_id == 0) {
        sg_partial_sums[sg_id] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = 0.0f;
    if (sg_id == 0) {
        float total = 0.0f;
        if (lane_id < num_simdgroups) {
            total = sg_partial_sums[lane_id];
        }
        total = simd_sum_f32(total);

        if (lane_id == 0) {
            mean = total / float(hidden_dim);
            final_stats[0] = mean;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = final_stats[0];

    // Pass 2: Compute variance (using mean from Pass 1)
    float sum_sq_diff = 0.0f;
    for (uint pass = 0; pass < num_passes; ++pass) {
        const uint vec_base = pass * vec_elements_per_pass + tid * TILE_SIZE;

        #pragma unroll
        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint vec_idx = vec_base + i;
            if (vec_idx < vec_hidden_dim) {
                const uint elem_base = vec_idx * 4;
                if (elem_base + 3 < hidden_dim) {
                    half4 x = input_vec[vec_idx];
                    sum_sq_diff += (float(x.x) - mean) * (float(x.x) - mean);
                    sum_sq_diff += (float(x.y) - mean) * (float(x.y) - mean);
                    sum_sq_diff += (float(x.z) - mean) * (float(x.z) - mean);
                    sum_sq_diff += (float(x.w) - mean) * (float(x.w) - mean);
                } else {
                    for (uint j = 0; elem_base + j < hidden_dim; ++j) {
                        float diff = float(input[token_idx * hidden_dim + elem_base + j]) - mean;
                        sum_sq_diff += diff * diff;
                    }
                }
            }
        }
    }

    sum_sq_diff = simd_sum_f32(sum_sq_diff);

    if (lane_id == 0) {
        sg_partial_sums[sg_id] = sum_sq_diff;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_std = 1.0f;
    if (sg_id == 0) {
        float total_var = 0.0f;
        if (lane_id < num_simdgroups) {
            total_var = sg_partial_sums[lane_id];
        }
        total_var = simd_sum_f32(total_var);

        if (lane_id == 0) {
            float variance = total_var / float(hidden_dim);
            inv_std = rsqrt(variance + eps);
            final_stats[1] = inv_std;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_std = final_stats[1];

    // Pass 3: Normalize and write
    device half4* output_vec = reinterpret_cast<device half4*>(output + token_idx * hidden_dim);
    device const half4* gamma_vec = reinterpret_cast<device const half4*>(gamma);
    device const half4* beta_vec = reinterpret_cast<device const half4*>(beta);

    for (uint pass = 0; pass < num_passes; ++pass) {
        const uint vec_base = pass * vec_elements_per_pass + tid * TILE_SIZE;

        #pragma unroll
        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint vec_idx = vec_base + i;
            if (vec_idx < vec_hidden_dim) {
                const uint elem_base = vec_idx * 4;
                if (elem_base + 3 < hidden_dim) {
                    half4 x = input_vec[vec_idx];
                    half4 g = gamma_vec[vec_idx];
                    half4 b = beta_vec[vec_idx];
                    half4 y;
                    y.x = half((float(x.x) - mean) * inv_std * float(g.x) + float(b.x));
                    y.y = half((float(x.y) - mean) * inv_std * float(g.y) + float(b.y));
                    y.z = half((float(x.z) - mean) * inv_std * float(g.z) + float(b.z));
                    y.w = half((float(x.w) - mean) * inv_std * float(g.w) + float(b.w));
                    output_vec[vec_idx] = y;
                } else {
                    for (uint j = 0; elem_base + j < hidden_dim; ++j) {
                        float x = float(input[token_idx * hidden_dim + elem_base + j]);
                        float g = float(gamma[elem_base + j]);
                        float b = float(beta[elem_base + j]);
                        output[token_idx * hidden_dim + elem_base + j] = half((x - mean) * inv_std * g + b);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Multi-pass RMSNorm with fused residual for large hidden dimensions
//
// Combines the multi-pass strategy with residual addition fusion.
// ============================================================================

kernel void rmsnorm_fused_residual_multipass(
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

    const uint SIMDGROUP_SIZE = 32;
    const uint num_simdgroups = 8;
    const uint TOTAL_THREADS = num_simdgroups * SIMDGROUP_SIZE;

    threadgroup float sg_partial_sums[16];

    const uint num_chunks = hidden_dim / CHUNK_SIZE;
    const uint remainder = hidden_dim % CHUNK_SIZE;
    const uint vec_chunks = num_chunks / (TOTAL_THREADS * 4);

    device const half4* x_vec = reinterpret_cast<device const half4*>(x + token_idx * hidden_dim);
    device const half4* res_vec = reinterpret_cast<device const half4*>(residual + token_idx * hidden_dim);

    // Pass 1: Compute residual add and sum of squares
    float sum_sq = 0.0f;

    for (uint chunk = 0; chunk < vec_chunks; ++chunk) {
        const uint base_idx = chunk * TOTAL_THREADS * 4 + sg_id * SIMDGROUP_SIZE * 4 + lane_id * 4;

        #pragma unroll
        for (uint v = 0; v < VEC_ELEMS_PER_LANE && (base_idx + v * TOTAL_THREADS + 3) < hidden_dim / 4; ++v) {
            const uint vec_idx = base_idx + v * TOTAL_THREADS;
            if (vec_idx * 4 + 3 < hidden_dim) {
                half4 xv = x_vec[vec_idx];
                half4 rv = res_vec[vec_idx];

                // Compute and store residual
                device half4* out_res = reinterpret_cast<device half4*>(residual_out + token_idx * hidden_dim);
                half4 new_res;
                new_res.x = half(float(xv.x) + float(rv.x));
                new_res.y = half(float(xv.y) + float(rv.y));
                new_res.z = half(float(xv.z) + float(rv.z));
                new_res.w = half(float(xv.w) + float(rv.w));
                out_res[vec_idx] = new_res;

                // Accumulate sum of squares
                sum_sq += float(new_res.x) * float(new_res.x);
                sum_sq += float(new_res.y) * float(new_res.y);
                sum_sq += float(new_res.z) * float(new_res.z);
                sum_sq += float(new_res.w) * float(new_res.w);
            }
        }
    }

    const uint remaining_start = num_chunks * CHUNK_SIZE;
    const uint base_offset = token_idx * hidden_dim + remaining_start;

    for (uint i = lane_id + sg_id * SIMDGROUP_SIZE; i < remainder; i += TOTAL_THREADS) {
        float xv = float(x[base_offset + i]);
        float rv = float(residual[base_offset + i]);
        float new_res = xv + rv;
        residual_out[base_offset + i] = half(new_res);
        sum_sq += new_res * new_res;
    }

    sum_sq = simd_sum_f32(sum_sq);

    if (lane_id == 0) {
        sg_partial_sums[sg_id] = sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = 1.0f;
    if (sg_id == 0) {
        float total = 0.0f;
        if (lane_id < num_simdgroups) {
            total = sg_partial_sums[lane_id];
        }
        total = simd_sum_f32(total);

        if (lane_id == 0) {
            float variance = total / float(hidden_dim);
            rms_inv = rsqrt(variance + eps);
            sg_partial_sums[0] = rms_inv;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    rms_inv = sg_partial_sums[0];

    // Pass 2: Normalize
    device half4* normed_vec = reinterpret_cast<device half4*>(normed_out + token_idx * hidden_dim);
    device const half4* residual_out_vec = reinterpret_cast<device half4*>(residual_out + token_idx * hidden_dim);
    device const half4* gamma_vec = reinterpret_cast<device const half4*>(gamma);

    for (uint chunk = 0; chunk < vec_chunks; ++chunk) {
        const uint base_idx = chunk * TOTAL_THREADS * 4 + sg_id * SIMDGROUP_SIZE * 4 + lane_id * 4;

        #pragma unroll
        for (uint v = 0; v < VEC_ELEMS_PER_LANE && (base_idx + v * TOTAL_THREADS + 3) < hidden_dim / 4; ++v) {
            const uint vec_idx = base_idx + v * TOTAL_THREADS;
            if (vec_idx * 4 + 3 < hidden_dim) {
                half4 r = residual_out_vec[vec_idx];
                half4 g = gamma_vec[vec_idx];
                half4 y;
                y.x = half(float(r.x) * rms_inv * float(g.x));
                y.y = half(float(r.y) * rms_inv * float(g.y));
                y.z = half(float(r.z) * rms_inv * float(g.z));
                y.w = half(float(r.w) * rms_inv * float(g.w));
                normed_vec[vec_idx] = y;
            }
        }
    }

    for (uint i = lane_id + sg_id * SIMDGROUP_SIZE; i < remainder; i += TOTAL_THREADS) {
        float r = float(residual_out[base_offset + i]);
        float g = float(gamma[remaining_start + i]);
        normed_out[base_offset + i] = half(r * rms_inv * g);
    }
}
