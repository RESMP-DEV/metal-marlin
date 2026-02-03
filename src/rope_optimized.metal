// ============================================================================
// Optimized RoPE (Rotary Position Embedding) for Apple Metal
//
// Features:
// 1. Texture-based sin/cos lookup for hardware-accelerated cache access
// 2. Fused Q/K projection + RoPE kernels
// 3. Precomputed sin/cos tables with optimized memory layout
// 4. Simdgroup-optimized kernels for small dimensions
//
// Texture Benefits:
// - Dedicated texture cache (separate from L1/L2)
// - Hardware filtering capabilities
// - Optimized for 2D spatial locality patterns
//
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================
constant constexpr uint ROPE_WARP_SIZE = 32;
constant constexpr uint ROPE_SIMDGROUP_SIZE = 32;
constant constexpr float ROPE_DEFAULT_BASE = 10000.0f;

// ============================================================================
// Texture-Based Sin/Cos Lookup
// ============================================================================

/**
 * Lookup sin/cos values from 2D texture.
 *
 * Texture format: [position][dim_pair/2] with (sin, cos) packed
 * Using half2 for compact storage and better cache utilization.
 *
 * @param tex_sin: 2D texture with sin values
 * @param tex_cos: 2D texture with cos values (or packed texture)
 * @param position: Sequence position
 * @param pair_idx: Dimension pair index
 * @param half_dim: Half of head dimension
 */
inline half2 lookup_sincos_texture(
    texture2d<half, access::sample> tex_sincos,
    uint position,
    uint pair_idx,
    uint half_dim
) {
    // Use normalized coordinates for texture lookup
    float u = (float)(pair_idx) / (float)half_dim;
    float v = (float)position;
    
    // Sample from texture (nearest neighbor, no interpolation)
    constexpr sampler texture_sampler(coord::normalized,
                                      address::clamp_to_edge,
                                      filter::nearest);
    
    // Texture stores (sin, cos) as half2
    return tex_sincos.sample(texture_sampler, float2(u, v)).rg;
}

/**
 * 1D texture buffer lookup for linear cache.
 * Better for smaller head dimensions.
 */
inline half2 lookup_sincos_buffer(
    const device half2* sincos_buffer,
    uint position,
    uint pair_idx,
    uint half_dim
) {
    uint idx = position * half_dim + pair_idx;
    return sincos_buffer[idx];
}

// ============================================================================
// Optimized RoPE Kernel with Texture Lookup
// ============================================================================

/**
 * RoPE forward with texture-based sin/cos lookup.
 *
 * Uses Metal's texture cache for fast access to precomputed tables.
 * Processes one rotation pair per thread.
 *
 * @param input: Input tensor [batch, seq_len, num_heads, head_dim]
 * @param tex_sincos: 2D texture with packed (sin, cos) values
 * @param output: Output tensor
 * @param batch_size: Batch dimension
 * @param seq_len: Sequence length
 * @param num_heads: Number of heads
 * @param head_dim: Dimension per head
 * @param position_offset: Offset for KV cache
 */
kernel void rope_forward_texture(
    device const half* input       [[buffer(0)]],
    texture2d<half, access::sample> tex_sincos [[texture(0)]],
    device half* output            [[buffer(1)]],
    constant uint& batch_size      [[buffer(2)]],
    constant uint& seq_len         [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& head_dim        [[buffer(5)]],
    constant uint& position_offset [[buffer(6)]],
    constant uint& max_seq_len     [[buffer(7)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint pair_idx = gid.x;
    uint head_idx = gid.y;
    uint batch_seq = gid.z;
    
    uint half_head_dim = head_dim >> 1;
    if (pair_idx >= half_head_dim) return;
    
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    uint position = seq_idx + position_offset;
    
    // Lookup sin/cos from texture
    half2 sincos = lookup_sincos_texture(tex_sincos, position, pair_idx, half_head_dim);
    half sin_val = sincos.x;
    half cos_val = sincos.y;
    
    // Compute tensor indices
    uint input_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    uint idx_x = input_base + (pair_idx << 1);
    uint idx_y = idx_x + 1;
    
    half x = input[idx_x];
    half y = input[idx_y];
    
    // Apply rotation
    half x_rot = x * cos_val - y * sin_val;
    half y_rot = x * sin_val + y * cos_val;
    
    output[idx_x] = x_rot;
    output[idx_y] = y_rot;
}

/**
 * Vectorized RoPE with texture lookup (processes 4 elements per thread).
 */
kernel void rope_forward_texture_x4(
    device const half* input       [[buffer(0)]],
    texture2d<half, access::sample> tex_sincos [[texture(0)]],
    device half* output            [[buffer(1)]],
    constant uint& batch_size      [[buffer(2)]],
    constant uint& seq_len         [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& head_dim        [[buffer(5)]],
    constant uint& position_offset [[buffer(6)]],
    constant uint& max_seq_len     [[buffer(7)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint quad_idx = gid.x;
    uint head_idx = gid.y;
    uint batch_seq = gid.z;
    
    uint quarter_head_dim = head_dim >> 2;
    if (quad_idx >= quarter_head_dim) return;
    
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    uint position = seq_idx + position_offset;
    uint half_head_dim = head_dim >> 1;
    
    // Two consecutive pairs
    uint pair_idx_0 = quad_idx << 1;
    uint pair_idx_1 = pair_idx_0 + 1;
    
    // Lookup sin/cos for both pairs
    half2 sincos0 = lookup_sincos_texture(tex_sincos, position, pair_idx_0, half_head_dim);
    half2 sincos1 = lookup_sincos_texture(tex_sincos, position, pair_idx_1, half_head_dim);
    
    // Pack cos/sin: (cos0, sin0, cos1, sin1)
    half4 cos_sin(sincos0.y, sincos0.x, sincos1.y, sincos1.x);
    
    // Load 4 values
    uint input_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    uint idx = input_base + (quad_idx << 2);
    
    half4 xy = *reinterpret_cast<device const half4*>(input + idx);
    
    // Apply vectorized rotation
    half4 rotated;
    rotated.x = xy.x * cos_sin.x - xy.y * cos_sin.y;
    rotated.y = xy.x * cos_sin.y + xy.y * cos_sin.x;
    rotated.z = xy.z * cos_sin.z - xy.w * cos_sin.w;
    rotated.w = xy.z * cos_sin.w + xy.w * cos_sin.z;
    
    *reinterpret_cast<device half4*>(output + idx) = rotated;
}

// ============================================================================
// Simdgroup-Optimized RoPE with Texture
// ============================================================================

/**
 * Simdgroup-optimized RoPE using texture lookup.
 *
 * Each simdgroup processes one (batch, seq, head) combination.
 * Uses simdgroup shuffle for efficient sin/cos broadcast.
 */
kernel void rope_forward_simdgroup_texture(
    device const half* input       [[buffer(0)]],
    texture2d<half, access::sample> tex_sincos [[texture(0)]],
    device half* output            [[buffer(1)]],
    constant uint& batch_size      [[buffer(2)]],
    constant uint& seq_len         [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& head_dim        [[buffer(5)]],
    constant uint& position_offset [[buffer(6)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    uint simdgroup_idx = tg_idx;
    uint total_simdgroups = batch_size * seq_len * num_heads;
    
    if (simdgroup_idx >= total_simdgroups) return;
    
    // Decompose simdgroup index
    uint head_idx = simdgroup_idx % num_heads;
    uint temp = simdgroup_idx / num_heads;
    uint seq_idx = temp % seq_len;
    uint batch_idx = temp / seq_len;
    
    uint position = seq_idx + position_offset;
    uint half_head_dim = head_dim >> 1;
    
    uint input_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    
    // Each lane processes multiple pairs
    for (uint pair = lane_id; pair < half_head_dim; pair += ROPE_SIMDGROUP_SIZE) {
        // Lookup sin/cos
        half2 sincos = lookup_sincos_texture(tex_sincos, position, pair, half_head_dim);
        half sin_val = sincos.x;
        half cos_val = sincos.y;
        
        uint idx_x = input_base + (pair << 1);
        uint idx_y = idx_x + 1;
        
        half x = input[idx_x];
        half y = input[idx_y];
        
        half x_rot = x * cos_val - y * sin_val;
        half y_rot = x * sin_val + y * cos_val;
        
        output[idx_x] = x_rot;
        output[idx_y] = y_rot;
    }
}

// ============================================================================
// Fused Q/K Projection + RoPE Kernels
// ============================================================================

/**
 * Fused Q projection + RoPE.
 *
 * Performs: output = RoPE(x @ Wq.T)
 *
 * Optimized for small head_dim that fits in simdgroup registers.
 * Uses simdgroup matrix multiply for the projection.
 */
kernel void rope_fused_q_projection(
    device const half* x           [[buffer(0)]],
    device const half* Wq          [[buffer(1)]],
    texture2d<half, access::sample> tex_sincos [[texture(0)]],
    device half* output            [[buffer(2)]],
    constant uint& batch_size      [[buffer(3)]],
    constant uint& seq_len         [[buffer(4)]],
    constant uint& num_heads       [[buffer(5)]],
    constant uint& head_dim        [[buffer(6)]],
    constant uint& hidden_dim      [[buffer(7)]],
    constant uint& position_offset [[buffer(8)]],
    uint3 gid                      [[thread_position_in_grid]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    uint pair_idx = gid.x;
    uint head_idx = gid.y;
    uint batch_seq = gid.z;
    
    uint half_dim = head_dim >> 1;
    if (pair_idx >= half_dim) return;
    
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    uint position = seq_idx + position_offset;
    
    // Compute projection for this pair (2 elements of Q)
    // Simplified: each thread computes 2 output elements
    uint x_base = (batch_idx * seq_len + seq_idx) * hidden_dim;
    uint w_base = (head_idx * head_dim + (pair_idx << 1)) * hidden_dim;
    
    half q0 = 0.0h, q1 = 0.0h;
    
    // Dot product with co-operative loading
    for (uint h = lane_id; h < hidden_dim; h += ROPE_SIMDGROUP_SIZE) {
        half x_val = x[x_base + h];
        q0 += x_val * Wq[w_base + h];
        q1 += x_val * Wq[w_base + hidden_dim + h];
    }
    
    // Reduce within simdgroup using shuffle
    for (uint offset = ROPE_SIMDGROUP_SIZE >> 1; offset > 0; offset >>= 1) {
        q0 += simd_shuffle_down(q0, offset);
        q1 += simd_shuffle_down(q1, offset);
    }
    
    // Lane 0 has the final result
    if (lane_id == 0) {
        // Lookup sin/cos
        half2 sincos = lookup_sincos_texture(tex_sincos, position, pair_idx, half_dim);
        half sin_val = sincos.x;
        half cos_val = sincos.y;
        
        // Apply RoPE
        half q0_rot = q0 * cos_val - q1 * sin_val;
        half q1_rot = q0 * sin_val + q1 * cos_val;
        
        uint out_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
        output[out_base + (pair_idx << 1)] = q0_rot;
        output[out_base + (pair_idx << 1) + 1] = q1_rot;
    }
}

/**
 * Fused Q and K projection + RoPE.
 *
 * Processes both Q and K in a single kernel dispatch.
 * Shares the same position encoding for both.
 */
kernel void rope_fused_qk_projection(
    device const half* x           [[buffer(0)]],
    device const half* Wq          [[buffer(1)]],
    device const half* Wk          [[buffer(2)]],
    texture2d<half, access::sample> tex_sincos [[texture(0)]],
    device half* q_out             [[buffer(3)]],
    device half* k_out             [[buffer(4)]],
    constant uint& batch_size      [[buffer(5)]],
    constant uint& seq_len         [[buffer(6)]],
    constant uint& num_heads       [[buffer(7)]],
    constant uint& num_kv_heads    [[buffer(8)]],
    constant uint& head_dim        [[buffer(9)]],
    constant uint& hidden_dim      [[buffer(10)]],
    constant uint& position_offset [[buffer(11)]],
    uint3 gid                      [[thread_position_in_grid]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    uint pair_idx = gid.x;
    uint head_idx = gid.y;
    uint batch_seq = gid.z;
    
    uint half_dim = head_dim >> 1;
    uint total_heads = num_heads + num_kv_heads;
    if (pair_idx >= half_dim || head_idx >= total_heads) return;
    
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    bool is_q = head_idx < num_heads;
    uint actual_head_idx = is_q ? head_idx : head_idx - num_heads;
    
    uint position = seq_idx + position_offset;
    
    // Compute projection
    uint x_base = (batch_idx * seq_len + seq_idx) * hidden_dim;
    uint w_base = (actual_head_idx * head_dim + (pair_idx << 1)) * hidden_dim;
    const device half* W = is_q ? Wq : Wk;
    
    half val0 = 0.0h, val1 = 0.0h;
    
    for (uint h = lane_id; h < hidden_dim; h += ROPE_SIMDGROUP_SIZE) {
        half x_val = x[x_base + h];
        val0 += x_val * W[w_base + h];
        val1 += x_val * W[w_base + hidden_dim + h];
    }
    
    // Reduce
    for (uint offset = ROPE_SIMDGROUP_SIZE >> 1; offset > 0; offset >>= 1) {
        val0 += simd_shuffle_down(val0, offset);
        val1 += simd_shuffle_down(val1, offset);
    }
    
    if (lane_id == 0) {
        half2 sincos = lookup_sincos_texture(tex_sincos, position, pair_idx, half_dim);
        half sin_val = sincos.x;
        half cos_val = sincos.y;
        
        half rot0 = val0 * cos_val - val1 * sin_val;
        half rot1 = val0 * sin_val + val1 * cos_val;
        
        if (is_q) {
            uint out_base = ((batch_idx * seq_len + seq_idx) * num_heads + actual_head_idx) * head_dim;
            q_out[out_base + (pair_idx << 1)] = rot0;
            q_out[out_base + (pair_idx << 1) + 1] = rot1;
        } else {
            uint out_base = ((batch_idx * seq_len + seq_idx) * num_kv_heads + actual_head_idx) * head_dim;
            k_out[out_base + (pair_idx << 1)] = rot0;
            k_out[out_base + (pair_idx << 1) + 1] = rot1;
        }
    }
}

// ============================================================================
// Precomputation Kernels
// ============================================================================

/**
 * Precompute sin/cos tables with interleaved layout.
 *
 * Output format: half2[seq_len][head_dim/2] where each element is (sin, cos)
 * This layout is optimal for texture/buffer lookup.
 */
kernel void rope_precompute_sincos(
    device half2* sincos_out       [[buffer(0)]],
    constant uint& max_seq_len     [[buffer(1)]],
    constant uint& head_dim        [[buffer(2)]],
    constant float& base           [[buffer(3)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint dim_idx = gid.x;
    uint pos_idx = gid.y;
    
    uint half_dim = head_dim >> 1;
    if (dim_idx >= half_dim || pos_idx >= max_seq_len) return;
    
    // Compute inverse frequency
    float exp = (2.0f * (float)dim_idx) / (float)head_dim;
    float inv_freq = 1.0f / pow(base, exp);
    
    // Compute angle
    float theta = (float)pos_idx * inv_freq;
    
    half sin_val = half(sin(theta));
    half cos_val = half(cos(theta));
    
    uint idx = pos_idx * half_dim + dim_idx;
    sincos_out[idx] = half2(sin_val, cos_val);
}

/**
 * Precompute sin/cos with YaRN scaling.
 */
kernel void rope_precompute_sincos_yarn(
    device half2* sincos_out       [[buffer(0)]],
    constant uint& max_seq_len     [[buffer(1)]],
    constant uint& head_dim        [[buffer(2)]],
    constant float& base           [[buffer(3)]],
    constant float& scale_factor   [[buffer(4)]],
    constant float& beta_fast      [[buffer(5)]],
    constant float& beta_slow      [[buffer(6)]],
    constant uint& original_max_pos [[buffer(7)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint dim_idx = gid.x;
    uint pos_idx = gid.y;
    
    uint half_dim = head_dim >> 1;
    if (dim_idx >= half_dim || pos_idx >= max_seq_len) return;
    
    // Base inverse frequency
    float exp = (2.0f * (float)dim_idx) / (float)head_dim;
    float inv_freq_base = 1.0f / pow(base, exp);
    
    // YaRN interpolation
    float freq = 1.0f / inv_freq_base;
    float rotations = (float)original_max_pos * freq / (2.0f * M_PI_F);
    
    float interp_factor;
    if (rotations >= beta_fast) {
        interp_factor = 1.0f / scale_factor;
    } else if (rotations <= beta_slow) {
        interp_factor = 1.0f;
    } else {
        interp_factor = 1.0f - (rotations - beta_slow) / (beta_fast - beta_slow) * (1.0f - 1.0f/scale_factor);
    }
    
    float inv_freq = inv_freq_base * interp_factor;
    float theta = (float)pos_idx * inv_freq;
    
    // Apply mscale
    float mscale = 0.1f * log(scale_factor) + 1.0f;
    
    half sin_val = half(sin(theta) * mscale);
    half cos_val = half(cos(theta) * mscale);
    
    uint idx = pos_idx * half_dim + dim_idx;
    sincos_out[idx] = half2(sin_val, cos_val);
}

// ============================================================================
// Test/Verification Kernels
// ============================================================================

/**
 * Verify texture lookup produces correct results.
 */
kernel void rope_verify_texture_lookup(
    texture2d<half, access::sample> tex_sincos [[texture(0)]],
    device half2* output           [[buffer(0)]],
    constant uint& max_seq_len     [[buffer(1)]],
    constant uint& head_dim        [[buffer(2)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint dim_idx = gid.x;
    uint pos_idx = gid.y;
    
    uint half_dim = head_dim >> 1;
    if (dim_idx >= half_dim || pos_idx >= max_seq_len) return;
    
    half2 sincos = lookup_sincos_texture(tex_sincos, pos_idx, dim_idx, half_dim);
    
    uint idx = pos_idx * half_dim + dim_idx;
    output[idx] = sincos;
}
