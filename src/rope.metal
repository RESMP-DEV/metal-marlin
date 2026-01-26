#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Rotary Position Embedding (RoPE) for Apple Metal
// ============================================================================
//
// Implements RoPE for Multi-head Latent Attention (MLA) with:
// 1. Standard RoPE (Llama-style interleaved)
// 2. Small latent dimension support (32-512)
// 3. GLM rope_ratio scaling
// 4. Fused projection + RoPE variant (see mla_proj.metal)
//
// RoPE encoding:
//   x_rotated[2i]   = x[2i]   * cos(θ_i) - x[2i+1] * sin(θ_i)
//   x_rotated[2i+1] = x[2i]   * sin(θ_i) + x[2i+1] * cos(θ_i)
//
// Where θ_i = position / (base^(2i/dim))
//
// For MLA, RoPE can be applied in two locations:
// 1. Standard: After full K/V projections (large head_dim)
// 2. Optimized: In compressed latent space (small kv_lora_rank, uses rope_ratio)
//
// Mathematical equivalence for fused RoPE:
//   If RoPE(kv_b_proj(x)) == kv_b_proj(RoPE(x)) for appropriately scaled RoPE
//   This holds when kv_b_proj is orthogonal or when using decoupled RoPE dimensions.
//
// GLM-4.7-Flash uses a `qk_rope_head_dim` separate from `kv_lora_rank`:
//   kv_a_proj output: [kv_lora_rank + qk_rope_head_dim]
//   The qk_rope_head_dim portion receives RoPE, kv_lora_rank does not.
//
// ============================================================================

// ============================================================================
// Constants and inline helpers
// ============================================================================

// Default RoPE base frequency (Llama)
constant constexpr float DEFAULT_ROPE_BASE = 10000.0f;

/// Compute inverse frequency for position i: 1 / (base^(2i/dim))
/// This is precomputed on CPU/Python side for efficiency.
inline float compute_inv_freq(uint dim_idx, uint rope_dim, float base) {
    float exp = float(dim_idx * 2) / float(rope_dim);
    return 1.0f / powr(base, exp);
}

/// Compute cos/sin for a given pair with rope_ratio scaling (on-the-fly).
inline half2 compute_rope_cos_sin_scaled(uint pair_idx, uint rope_dim, uint position,
                                         float base, float rope_ratio) {
    float exp = float(pair_idx * 2) / float(rope_dim);
    float inv_freq = rope_ratio / powr(base, exp);
    float theta = float(position) * inv_freq;
    return half2(half(cos(theta)), half(sin(theta)));
}

/// Apply rotation to a single (x, y) pair
/// Returns (x*cos - y*sin, x*sin + y*cos)
inline half2 apply_rope_rotation(half x, half y, half cos_val, half sin_val) {
    return half2(
        x * cos_val - y * sin_val,
        x * sin_val + y * cos_val
    );
}

/// Apply rotation with rope_ratio scaling (GLM-style)
/// The rope_ratio adjusts frequency: theta *= rope_ratio
inline half2 apply_rope_rotation_scaled(half x, half y, half cos_val, half sin_val,
                                         half rope_ratio) {
    // When rope_ratio != 1, the cos/sin values should already incorporate it
    // This function is kept for clarity; actual scaling happens in freq computation
    return half2(
        x * cos_val - y * sin_val,
        x * sin_val + y * cos_val
    );
}

// ============================================================================
// RoPE kernel: Standard Llama-style (interleaved pairs)
// ============================================================================
//
// Layout: [batch, seq_len, num_heads, head_dim]
// RoPE applied to pairs: (x[0], x[1]), (x[2], x[3]), ...
//
// Grid: (num_heads, seq_len, batch)
// Each thread processes one pair within head_dim.
//
// ============================================================================

/// Standard RoPE kernel for query/key tensors.
///
/// @param input       Input tensor [batch, seq_len, num_heads, head_dim]
/// @param cos_cache   Precomputed cos values [max_seq, head_dim/2]
/// @param sin_cache   Precomputed sin values [max_seq, head_dim/2]
/// @param output      Output tensor (same shape as input)
/// @param batch_size  Batch dimension
/// @param seq_len     Current sequence length
/// @param num_heads   Number of attention heads
/// @param head_dim    Dimension per head (must be even)
/// @param position_offset  Offset for KV cache continuation
kernel void rope_forward(
    device const half* input       [[buffer(0)]],
    device const half* cos_cache   [[buffer(1)]],
    device const half* sin_cache   [[buffer(2)]],
    device half* output            [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& seq_len         [[buffer(5)]],
    constant uint& num_heads       [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant uint& position_offset [[buffer(8)]],
    constant uint& max_seq_len     [[buffer(9)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint pair_idx = gid.x;      // Which pair within head_dim (0 to head_dim/2 - 1)
    uint head_idx = gid.y;      // Which head
    uint batch_seq = gid.z;     // Combined batch * seq index

    uint half_head_dim = head_dim / 2;
    if (pair_idx >= half_head_dim) return;

    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    // Compute position with offset (for KV cache continuation)
    uint position = seq_idx + position_offset;

    // Load precomputed cos/sin for this position and dimension
    uint cache_idx = position * half_head_dim + pair_idx;
    half cos_val = cos_cache[cache_idx];
    half sin_val = sin_cache[cache_idx];

    // Input index: [batch, seq, heads, head_dim]
    // Stride: batch * seq_len * num_heads * head_dim
    uint input_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    uint idx_x = input_base + pair_idx * 2;
    uint idx_y = input_base + pair_idx * 2 + 1;

    half x = input[idx_x];
    half y = input[idx_y];

    half2 rotated = apply_rope_rotation(x, y, cos_val, sin_val);

    output[idx_x] = rotated.x;
    output[idx_y] = rotated.y;
}

// ============================================================================
// RoPE kernel: Small latent dimensions (MLA optimized)
// ============================================================================
//
// For MLA models, the qk_rope_head_dim is typically small (32-128).
// This kernel is optimized for:
// - Small rope_dim that fits in simdgroup registers
// - Processing multiple positions per threadgroup
// - Fused with latent extraction from kv_a_proj output
//
// Layout: kv_a_proj output is [batch, seq_len, kv_lora_rank + qk_rope_head_dim]
// We apply RoPE only to the qk_rope_head_dim portion.
//
// ============================================================================

/// RoPE for MLA latent representations.
///
/// Applies RoPE only to the qk_rope_head_dim portion of the kv_a_proj output.
/// The kv_lora_rank portion passes through unchanged.
///
/// @param input         kv_a_proj output [batch, seq_len, kv_lora_rank + rope_dim]
/// @param cos_cache     Precomputed cos [max_seq, rope_dim/2]
/// @param sin_cache     Precomputed sin [max_seq, rope_dim/2]
/// @param output        Output tensor (same shape)
/// @param batch_size    Batch dimension
/// @param seq_len       Current sequence length
/// @param kv_lora_rank  Latent dimension (passes through unchanged)
/// @param rope_dim      RoPE dimension (qk_rope_head_dim)
/// @param position_offset  For KV cache continuation
kernel void rope_mla_latent(
    device const half* input       [[buffer(0)]],
    device const half* cos_cache   [[buffer(1)]],
    device const half* sin_cache   [[buffer(2)]],
    device half* output            [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& seq_len         [[buffer(5)]],
    constant uint& kv_lora_rank    [[buffer(6)]],
    constant uint& rope_dim        [[buffer(7)]],
    constant uint& position_offset [[buffer(8)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint elem_idx = gid.x;      // Element index within the total dimension
    uint batch_seq = gid.y;     // Combined batch * seq index

    uint total_dim = kv_lora_rank + rope_dim;
    if (elem_idx >= total_dim) return;

    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint position = seq_idx + position_offset;
    uint input_idx = (batch_idx * seq_len + seq_idx) * total_dim + elem_idx;

    if (elem_idx < kv_lora_rank) {
        // Pass through latent portion unchanged
        output[input_idx] = input[input_idx];
    } else {
        // Apply RoPE to the rope_dim portion
        uint rope_idx = elem_idx - kv_lora_rank;
        uint pair_idx = rope_idx / 2;
        uint half_rope_dim = rope_dim / 2;

        // Load cos/sin for this position and dimension
        uint cache_idx = position * half_rope_dim + pair_idx;
        half cos_val = cos_cache[cache_idx];
        half sin_val = sin_cache[cache_idx];

        // Load the pair
        uint base_idx = (batch_idx * seq_len + seq_idx) * total_dim + kv_lora_rank;
        half x = input[base_idx + pair_idx * 2];
        half y = input[base_idx + pair_idx * 2 + 1];

        half2 rotated = apply_rope_rotation(x, y, cos_val, sin_val);

        // Only write the element this thread is responsible for
        if (rope_idx % 2 == 0) {
            output[input_idx] = rotated.x;
        } else {
            output[input_idx] = rotated.y;
        }
    }
}

// ============================================================================
// RoPE kernel: Small latent dimensions with simdgroup
// ============================================================================
//
// For rope_dim <= 64, use a single simdgroup per position to rotate the
// positional portion while copying the latent portion unchanged.
//
// This kernel expects threadgroup size == simdgroup size (32).
//
// Layout: kv_a_proj output is [batch, seq_len, kv_lora_rank + rope_dim]
// We apply RoPE only to the rope_dim portion.
//
// ============================================================================

/// Simdgroup-optimized RoPE for MLA latents (small rope_dim).
kernel void rope_mla_latent_small(
    device const half* input       [[buffer(0)]],
    device const half* cos_cache   [[buffer(1)]],
    device const half* sin_cache   [[buffer(2)]],
    device half* output            [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& seq_len         [[buffer(5)]],
    constant uint& kv_lora_rank    [[buffer(6)]],
    constant uint& rope_dim        [[buffer(7)]],
    constant uint& position_offset [[buffer(8)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    uint batch_seq = tg_idx;
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint position = seq_idx + position_offset;
    uint total_dim = kv_lora_rank + rope_dim;
    uint input_base = (batch_idx * seq_len + seq_idx) * total_dim;

    // Copy latent portion unchanged.
    for (uint i = lane_id; i < kv_lora_rank; i += 32) {
        output[input_base + i] = input[input_base + i];
    }

    // Apply RoPE to positional portion.
    uint half_rope = rope_dim / 2;
    uint rope_base = input_base + kv_lora_rank;
    for (uint pair_idx = lane_id; pair_idx < half_rope; pair_idx += 32) {
        uint cache_idx = position * half_rope + pair_idx;
        half cos_val = cos_cache[cache_idx];
        half sin_val = sin_cache[cache_idx];

        half x = input[rope_base + pair_idx * 2];
        half y = input[rope_base + pair_idx * 2 + 1];
        half2 rotated = apply_rope_rotation(x, y, cos_val, sin_val);

        output[rope_base + pair_idx * 2] = rotated.x;
        output[rope_base + pair_idx * 2 + 1] = rotated.y;
    }
}

/// Simdgroup-optimized RoPE for MLA latents with rope_ratio scaling.
/// Computes cos/sin on-the-fly for GLM-style scaling (small rope_dim).
kernel void rope_mla_latent_small_scaled(
    device const half* input       [[buffer(0)]],
    device half* output            [[buffer(1)]],
    constant uint& batch_size      [[buffer(2)]],
    constant uint& seq_len         [[buffer(3)]],
    constant uint& kv_lora_rank    [[buffer(4)]],
    constant uint& rope_dim        [[buffer(5)]],
    constant uint& position_offset [[buffer(6)]],
    constant float& base           [[buffer(7)]],
    constant float& rope_ratio     [[buffer(8)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    uint batch_seq = tg_idx;
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint position = seq_idx + position_offset;
    uint total_dim = kv_lora_rank + rope_dim;
    uint input_base = (batch_idx * seq_len + seq_idx) * total_dim;

    // Copy latent portion unchanged.
    for (uint i = lane_id; i < kv_lora_rank; i += 32) {
        output[input_base + i] = input[input_base + i];
    }

    // Apply scaled RoPE to positional portion.
    uint half_rope = rope_dim / 2;
    uint rope_base = input_base + kv_lora_rank;
    for (uint pair_idx = lane_id; pair_idx < half_rope; pair_idx += 32) {
        half2 cs = compute_rope_cos_sin_scaled(pair_idx, rope_dim, position, base, rope_ratio);
        half x = input[rope_base + pair_idx * 2];
        half y = input[rope_base + pair_idx * 2 + 1];
        half2 rotated = apply_rope_rotation(x, y, cs.x, cs.y);

        output[rope_base + pair_idx * 2] = rotated.x;
        output[rope_base + pair_idx * 2 + 1] = rotated.y;
    }
}

// ============================================================================
// RoPE kernel: GLM rope_ratio support
// ============================================================================
//
// GLM-4.7-Flash uses rope_ratio to scale frequencies:
//   inv_freq *= rope_ratio
//
// This is handled by using pre-scaled cos/sin caches.
// The kernel below generates the cache on-the-fly for dynamic rope_ratio.
//
// ============================================================================

/// Generate RoPE cos/sin cache on-the-fly with rope_ratio scaling.
///
/// @param cos_out      Output cos cache [seq_len, rope_dim/2]
/// @param sin_out      Output sin cache [seq_len, rope_dim/2]
/// @param seq_len      Number of positions to generate
/// @param rope_dim     RoPE dimension
/// @param base         Base frequency (default 10000)
/// @param rope_ratio   GLM-style frequency scaling factor
kernel void rope_generate_cache(
    device half* cos_out           [[buffer(0)]],
    device half* sin_out           [[buffer(1)]],
    constant uint& seq_len         [[buffer(2)]],
    constant uint& rope_dim        [[buffer(3)]],
    constant float& base           [[buffer(4)]],
    constant float& rope_ratio     [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint dim_idx = gid.x;       // Dimension pair index (0 to rope_dim/2 - 1)
    uint pos_idx = gid.y;       // Position index

    uint half_rope_dim = rope_dim / 2;
    if (dim_idx >= half_rope_dim || pos_idx >= seq_len) return;

    // Compute inverse frequency with rope_ratio scaling
    // inv_freq = 1 / (base^(2*dim_idx/rope_dim)) * rope_ratio
    float exp = float(dim_idx * 2) / float(rope_dim);
    float inv_freq = rope_ratio / powr(base, exp);

    // Compute theta = position * inv_freq
    float theta = float(pos_idx) * inv_freq;

    // Write to cache
    uint cache_idx = pos_idx * half_rope_dim + dim_idx;
    cos_out[cache_idx] = half(cos(theta));
    sin_out[cache_idx] = half(sin(theta));
}

// ============================================================================
// Fused RoPE + Linear Projection kernel
// ============================================================================
//
// For MLA, we can fuse RoPE application with the latent projection.
// This kernel applies RoPE to the input and then performs a linear projection.
//
// Mathematical basis:
//   output = W @ RoPE(x)
//
// For decoupled RoPE (GLM-style where qk_rope_head_dim is separate):
//   kv_compressed = kv_a_proj(hidden)
//   c_kv = kv_compressed[:kv_lora_rank]  # No RoPE
//   k_pe = RoPE(kv_compressed[kv_lora_rank:])  # RoPE on position encoding
//
// This separation means RoPE is applied after kv_a_proj but only to part of output.
// The fusion opportunity is limited - we fuse RoPE with the split operation here.
// For full projection fusion (kv_a_proj + RoPE), use mla_proj_with_rope_fp4 in mla_proj.metal.
//
// ============================================================================

/// Fused split + RoPE for MLA kv_a_proj output.
///
/// Takes kv_a_proj output and:
/// 1. Splits into latent (c_kv) and positional (k_pe) portions
/// 2. Applies RoPE to k_pe
/// 3. Outputs both in separate buffers (ready for cache)
///
/// @param kv_a_output   kv_a_proj output [batch, seq_len, kv_lora_rank + rope_dim]
/// @param cos_cache     Precomputed cos [max_seq, rope_dim/2]
/// @param sin_cache     Precomputed sin [max_seq, rope_dim/2]
/// @param c_kv_out      Latent output [batch, seq_len, kv_lora_rank]
/// @param k_pe_out      Positional output with RoPE [batch, seq_len, rope_dim]
/// @param batch_size    Batch dimension
/// @param seq_len       Current sequence length
/// @param kv_lora_rank  Latent dimension
/// @param rope_dim      Position encoding dimension
/// @param position_offset  For KV cache continuation
kernel void rope_mla_split_fused(
    device const half* kv_a_output [[buffer(0)]],
    device const half* cos_cache   [[buffer(1)]],
    device const half* sin_cache   [[buffer(2)]],
    device half* c_kv_out          [[buffer(3)]],
    device half* k_pe_out          [[buffer(4)]],
    constant uint& batch_size      [[buffer(5)]],
    constant uint& seq_len         [[buffer(6)]],
    constant uint& kv_lora_rank    [[buffer(7)]],
    constant uint& rope_dim        [[buffer(8)]],
    constant uint& position_offset [[buffer(9)]],
    uint2 gid                      [[thread_position_in_grid]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    uint batch_seq = gid.y;
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint position = seq_idx + position_offset;
    uint total_dim = kv_lora_rank + rope_dim;
    uint input_base = (batch_idx * seq_len + seq_idx) * total_dim;

    // Part 1: Copy latent portion (no RoPE)
    // Each thread copies multiple elements for efficiency
    uint latent_base = (batch_idx * seq_len + seq_idx) * kv_lora_rank;
    for (uint i = gid.x; i < kv_lora_rank; i += 256) {  // Assuming 256 threads
        if (i < kv_lora_rank) {
            c_kv_out[latent_base + i] = kv_a_output[input_base + i];
        }
    }

    // Part 2: RoPE on positional portion
    uint half_rope_dim = rope_dim / 2;
    uint k_pe_base = (batch_idx * seq_len + seq_idx) * rope_dim;

    for (uint pair_idx = gid.x; pair_idx < half_rope_dim; pair_idx += 256) {
        if (pair_idx < half_rope_dim) {
            // Load cos/sin
            uint cache_idx = position * half_rope_dim + pair_idx;
            half cos_val = cos_cache[cache_idx];
            half sin_val = sin_cache[cache_idx];

            // Load input pair from positional portion
            uint src_idx = input_base + kv_lora_rank + pair_idx * 2;
            half x = kv_a_output[src_idx];
            half y = kv_a_output[src_idx + 1];

            // Apply rotation
            half2 rotated = apply_rope_rotation(x, y, cos_val, sin_val);

            // Write to k_pe output
            k_pe_out[k_pe_base + pair_idx * 2] = rotated.x;
            k_pe_out[k_pe_base + pair_idx * 2 + 1] = rotated.y;
        }
    }
}

// ============================================================================
// Optimized RoPE for small dimensions using simdgroup
// ============================================================================
//
// For rope_dim <= 64, we can process entire positions within a simdgroup.
// This eliminates threadgroup synchronization and enables register-only compute.
//
// ============================================================================

/// Simdgroup-optimized RoPE for small dimensions (rope_dim <= 64).
///
/// Each simdgroup processes one position. All 32 lanes cooperatively
/// handle the rotation, with each lane processing 2 elements (one pair).
///
/// @param input       Input tensor [batch, seq_len, dim]
/// @param cos_cache   Precomputed cos [max_seq, dim/2]
/// @param sin_cache   Precomputed sin [max_seq, dim/2]
/// @param output      Output tensor
/// @param batch_size  Batch dimension
/// @param seq_len     Current sequence length
/// @param dim         RoPE dimension (must be <= 64)
/// @param position_offset  For KV cache continuation
kernel void rope_small_dim(
    device const half* input       [[buffer(0)]],
    device const half* cos_cache   [[buffer(1)]],
    device const half* sin_cache   [[buffer(2)]],
    device half* output            [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& seq_len         [[buffer(5)]],
    constant uint& dim             [[buffer(6)]],
    constant uint& position_offset [[buffer(7)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint lane_id                   [[thread_index_in_simdgroup]]
) {
    // Each threadgroup is one simdgroup, processing one position
    uint batch_seq = tg_idx;
    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint position = seq_idx + position_offset;
    uint half_dim = dim / 2;

    // Each lane handles one pair (lane_id < half_dim)
    if (lane_id < half_dim) {
        // Load cos/sin
        uint cache_idx = position * half_dim + lane_id;
        half cos_val = cos_cache[cache_idx];
        half sin_val = sin_cache[cache_idx];

        // Load input pair
        uint input_base = (batch_idx * seq_len + seq_idx) * dim;
        half x = input[input_base + lane_id * 2];
        half y = input[input_base + lane_id * 2 + 1];

        // Apply rotation
        half2 rotated = apply_rope_rotation(x, y, cos_val, sin_val);

        // Write output
        output[input_base + lane_id * 2] = rotated.x;
        output[input_base + lane_id * 2 + 1] = rotated.y;
    }
}

// ============================================================================
// In-place RoPE (overwrites input)
// ============================================================================

/// In-place RoPE application (saves memory by reusing input buffer).
///
/// Same algorithm as rope_forward but writes back to input buffer.
/// Use when input tensor can be safely modified.
kernel void rope_inplace(
    device half* data              [[buffer(0)]],
    device const half* cos_cache   [[buffer(1)]],
    device const half* sin_cache   [[buffer(2)]],
    constant uint& batch_size      [[buffer(3)]],
    constant uint& seq_len         [[buffer(4)]],
    constant uint& num_heads       [[buffer(5)]],
    constant uint& head_dim        [[buffer(6)]],
    constant uint& position_offset [[buffer(7)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint pair_idx = gid.x;
    uint head_idx = gid.y;
    uint batch_seq = gid.z;

    uint half_head_dim = head_dim / 2;
    if (pair_idx >= half_head_dim) return;

    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint position = seq_idx + position_offset;
    uint cache_idx = position * half_head_dim + pair_idx;
    half cos_val = cos_cache[cache_idx];
    half sin_val = sin_cache[cache_idx];

    uint data_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    uint idx_x = data_base + pair_idx * 2;
    uint idx_y = data_base + pair_idx * 2 + 1;

    half x = data[idx_x];
    half y = data[idx_y];

    half2 rotated = apply_rope_rotation(x, y, cos_val, sin_val);

    data[idx_x] = rotated.x;
    data[idx_y] = rotated.y;
}

// ============================================================================
// Batched RoPE for Q and K simultaneously
// ============================================================================
//
// In attention, both Q and K need RoPE with the same position encoding.
// Fusing them into one kernel reduces memory traffic for cos/sin cache.
//
// ============================================================================

/// Batched RoPE for Q and K tensors simultaneously.
///
/// @param q_input     Query input [batch, seq_len, num_heads, head_dim]
/// @param k_input     Key input [batch, seq_len, num_kv_heads, head_dim]
/// @param cos_cache   Precomputed cos [max_seq, head_dim/2]
/// @param sin_cache   Precomputed sin [max_seq, head_dim/2]
/// @param q_output    Query output
/// @param k_output    Key output
/// @param batch_size  Batch dimension
/// @param seq_len     Current sequence length
/// @param num_heads   Number of query heads
/// @param num_kv_heads Number of KV heads (for GQA)
/// @param head_dim    Dimension per head
/// @param position_offset  For KV cache continuation
kernel void rope_qk_fused(
    device const half* q_input     [[buffer(0)]],
    device const half* k_input     [[buffer(1)]],
    device const half* cos_cache   [[buffer(2)]],
    device const half* sin_cache   [[buffer(3)]],
    device half* q_output          [[buffer(4)]],
    device half* k_output          [[buffer(5)]],
    constant uint& batch_size      [[buffer(6)]],
    constant uint& seq_len         [[buffer(7)]],
    constant uint& num_heads       [[buffer(8)]],
    constant uint& num_kv_heads    [[buffer(9)]],
    constant uint& head_dim        [[buffer(10)]],
    constant uint& position_offset [[buffer(11)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint pair_idx = gid.x;      // Pair within head_dim
    uint head_idx = gid.y;      // Head index (max of num_heads, num_kv_heads)
    uint batch_seq = gid.z;     // Combined batch * seq

    uint half_head_dim = head_dim / 2;
    if (pair_idx >= half_head_dim) return;

    uint batch_idx = batch_seq / seq_len;
    uint seq_idx = batch_seq % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint position = seq_idx + position_offset;
    uint cache_idx = position * half_head_dim + pair_idx;
    half cos_val = cos_cache[cache_idx];
    half sin_val = sin_cache[cache_idx];

    // Process Q if head_idx < num_heads
    if (head_idx < num_heads) {
        uint q_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
        uint q_idx_x = q_base + pair_idx * 2;
        uint q_idx_y = q_base + pair_idx * 2 + 1;

        half q_x = q_input[q_idx_x];
        half q_y = q_input[q_idx_y];
        half2 q_rot = apply_rope_rotation(q_x, q_y, cos_val, sin_val);

        q_output[q_idx_x] = q_rot.x;
        q_output[q_idx_y] = q_rot.y;
    }

    // Process K if head_idx < num_kv_heads
    if (head_idx < num_kv_heads) {
        uint k_base = ((batch_idx * seq_len + seq_idx) * num_kv_heads + head_idx) * head_dim;
        uint k_idx_x = k_base + pair_idx * 2;
        uint k_idx_y = k_base + pair_idx * 2 + 1;

        half k_x = k_input[k_idx_x];
        half k_y = k_input[k_idx_y];
        half2 k_rot = apply_rope_rotation(k_x, k_y, cos_val, sin_val);

        k_output[k_idx_x] = k_rot.x;
        k_output[k_idx_y] = k_rot.y;
    }
}

// ============================================================================
// Test/Verification kernels
// ============================================================================

/// Test kernel: verify RoPE rotation properties.
/// Applies RoPE twice with negated sin (simulating position -p) to verify
/// that the result recovers the original input.
///
/// RoPE(θ) @ RoPE(-θ) = I (identity)
kernel void test_rope_roundtrip(
    device const half* input       [[buffer(0)]],
    device half* output            [[buffer(1)]],
    constant uint& dim             [[buffer(2)]],
    constant uint& n_elements      [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    if (tid >= n_elements) return;

    uint pair_idx = tid % (dim / 2);

    // Use arbitrary angle for testing
    float theta = float(pair_idx) * 0.1f;
    half cos_val = half(cos(theta));
    half sin_val = half(sin(theta));

    // Load pair
    uint base_idx = (tid / (dim / 2)) * dim;
    half x = input[base_idx + pair_idx * 2];
    half y = input[base_idx + pair_idx * 2 + 1];

    // Forward rotation
    half2 fwd = apply_rope_rotation(x, y, cos_val, sin_val);

    // Backward rotation (negate sin)
    half2 bwd = apply_rope_rotation(fwd.x, fwd.y, cos_val, -sin_val);

    // Output should match input (within FP16 precision)
    output[base_idx + pair_idx * 2] = bwd.x;
    output[base_idx + pair_idx * 2 + 1] = bwd.y;
}
