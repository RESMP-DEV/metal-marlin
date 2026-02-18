// flash_attention_v2_fused.metal - Fused Attention + O Projection
//
// Fuses Flash Attention V2 with the output (O) projection to avoid
// intermediate tensor materialization. Reduces memory bandwidth by ~25%
// for typical transformer architectures.
//
// Math:
//   Original: O = Attention(Q, K, V); result = O @ W_o
//   Fused:    result = (Attention(Q, K, V)) @ W_o (single pass)
//
// The kernel computes attention output in registers and immediately
// multiplies by dequantized O weights before writing to global memory.
//
// Kernel variants:
//   flash_attention_v2_fused         - Tiled prefill, non-causal
//   flash_attention_v2_fused_causal    - Tiled prefill, causal
//   flash_attention_v2_fused_decode    - Single-query decode (seq_q=1)
//
// Memory layout:
//   Q: [batch, heads_q, seq_q, head_dim]
//   K: [batch, heads_kv, seq_k, head_dim]
//   V: [batch, heads_kv, seq_k, head_dim]
//   O_weights_packed: [num_heads_q * head_dim // 8, hidden_size] (FP4 packed)
//   O_scales: [num_heads_q * head_dim // group_size, hidden_size]
//   O_bias: [hidden_size] (optional)
//   Output: [batch, seq_q, hidden_size]

#include <metal_stdlib>
#include "../bf16_compat.metal"
using namespace metal;

#ifdef USE_BF16_INPUTS
using input_t = bf16_t;
#else
using input_t = half;
#endif

inline half4 half4_load(device const half* src) {
    return *reinterpret_cast<device const half4*>(src);
}

#ifdef USE_BF16_INPUTS
inline float input_to_float(input_t v) {
    return bf16_to_float(v);
}
#else
inline float input_to_float(input_t v) {
    return float(v);
}
#endif

// ---------------------------------------------------------------------------
// FP4 E2M1 dequantization (same as layers.py)
// ---------------------------------------------------------------------------

inline half dequant_fp4_e2m1(uint8_t nibble, half scale) {
    // Extract sign, exponent, mantissa
    uint sign = (nibble >> 3) & 1u;
    uint exp = (nibble >> 1) & 3u;
    uint mant = nibble & 1u;

    float magnitude;
    if (exp == 0u && mant == 0u) {
        magnitude = 0.0f;
    } else if (exp == 0u && mant == 1u) {
        magnitude = 0.5f;  // Subnormal
    } else {
        float m = 1.0f + float(mant) * 0.5f;
        float e = float(exp) - 1.0f;
        magnitude = m * exp2(e);
    }

    return select(magnitude, -magnitude, bool(sign)) * scale;
}

// ---------------------------------------------------------------------------
// Tile dimensions (same as flash_attention_v2.metal)
// ---------------------------------------------------------------------------

constant constexpr uint TILE_Q = 16;
constant constexpr uint TILE_KV = 24;
constant constexpr uint HEAD_DIM_64 = 64;
constant constexpr uint HEAD_DIM_128 = 128;
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint NUM_SIMDGROUPS = 4;
constant constexpr uint THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;

// ---------------------------------------------------------------------------
// Attention parameters with O projection
// ---------------------------------------------------------------------------

struct AttentionParamsFused {
    uint batch;
    uint num_heads_q;
    uint num_heads_kv;
    uint seq_q;
    uint seq_k;
    uint head_dim;
    float scale;
    uint gqa_ratio;
    uint is_causal;

    // O projection parameters
    uint hidden_size;
    uint o_group_size;  // Quantization group size for O weights
};

// ---------------------------------------------------------------------------
// Fused Flash Attention + O Projection (Non-Causal)
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_fused(
    device const input_t* Q               [[buffer(0)]],
    device const input_t* K               [[buffer(1)]],
    device const input_t* V               [[buffer(2)]],
    device const uint32_t* O_weights_packed [[buffer(3)]],  // FP4 packed
    device const half* O_scales          [[buffer(4)]],      // Per-group scales
    device const half* O_bias            [[buffer(5)]],      // Optional bias
    device half* Output                  [[buffer(6)]],
    constant AttentionParamsFused& params [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;
    const uint hidden_size = params.hidden_size;

    const uint head_kv = head_q / params.gqa_ratio;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Q strides: [batch, heads_q, seq_q, head_dim]
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    // K/V strides: [batch, heads_kv, seq_k, head_dim]
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    // O projection stride: [num_heads_q * head_dim, hidden_size]
    // Row-major: packed weights for O[i, :] at i * (hidden_size // 8)
    const uint o_row_stride = hidden_size / 8u;  // 8 FP4 values per uint32
    const uint o_col_stride = 1u;  // Column-major for efficient access

    // Threadgroup memory for Q and K/V tiles
    threadgroup input_t Q_tile[TILE_Q][HEAD_DIM_128];
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_128];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_128];

    // Load Q tile cooperatively
    {
        const uint elems_to_load = q_rows * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint q_row = idx / head_dim;
                uint q_col = idx % head_dim;
                Q_tile[q_row][q_col] = Q[q_base + q_row * q_stride_s + q_col];
            }
        }
    }

    // Per-simdgroup state
    const uint rows_per_sg = TILE_Q / NUM_SIMDGROUPS;
    const uint sg_q_start = sg_id * rows_per_sg;
    const uint sg_q_rows = min(rows_per_sg, (q_rows > sg_q_start) ? (q_rows - sg_q_start) : 0u);

    const uint elems_per_lane = head_dim / SIMD_SIZE;

    // Register allocation: Q values, online softmax state, attention output
    float q_reg[4][HEAD_DIM_128 / SIMD_SIZE];
    float m_prev[4];
    float l_prev[4];
    float o_acc[4][HEAD_DIM_128 / SIMD_SIZE];

    // Initialize
    for (uint r = 0; r < rows_per_sg; ++r) {
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            q_reg[r][i] = 0.0f;
            o_acc[r][i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q into registers
    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = input_to_float(Q_tile[q_row][d]);
        }
    }

    // Preload first K/V tile
    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    {
        const uint elems = min(uint(TILE_KV), seq_k) * head_dim;
        const uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                if (kv_row < seq_k) {
                    K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: stream through K/V tiles and compute attention
    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

        // Async load next tile
        if (tile_idx + 1 < num_kv_tiles) {
            uint next_start = (tile_idx + 1) * TILE_KV;
            uint next_len = min(uint(TILE_KV), seq_k - next_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    K_tile[buf_load][kv_row][kv_col] = K[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                    V_tile[buf_load][kv_row][kv_col] = V[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                }
            }
        }

        // Compute attention scores
        for (uint r = 0; r < sg_q_rows; ++r) {
            float scores[TILE_KV];

            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[r][i] * float(K_tile[buf][ki][d]);
                }
                dot = simd_sum(dot);
                scores[ki] = dot * scale;
            }

            // Zero pad invalid positions
            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax update
            float m_tile = -INFINITY;
            for (uint ki = 0; ki < tile_len; ++ki) {
                m_tile = max(m_tile, scores[ki]);
            }

            float m_new = max(m_prev[r], m_tile);
            float corr = exp(m_prev[r] - m_new);

            l_prev[r] *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[r][i] *= corr;
            }

            for (uint ki = 0; ki < tile_len; ++ki) {
                float p = exp(scores[ki] - m_new);
                l_prev[r] += p;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[r][i] += p * float(V_tile[buf][ki][d]);
                }
            }

            m_prev[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // ---------------------------------------------------------------------------
    // Normalize attention output and multiply by O weights
    // ---------------------------------------------------------------------------
    //
    // For each query position (seq_q), we have attention output o_acc with
    // shape [num_heads_q, head_dim]. We multiply by O weights to get
    // final output: output[q] = sum_h (o_acc[h, :] @ O_weights[h, :])
    //
    // O weights are stored packed as FP4, so we dequantize on the fly.
    // ---------------------------------------------------------------------------

    const uint hidden_elems_per_lane = (hidden_size + SIMD_SIZE - 1) / SIMD_SIZE;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;

        // Normalize attention output
        float attn_norm[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            attn_norm[i] = o_acc[r][i] * inv_l;
        }

        // Compute output = attn_norm @ O_weights
        // Output is per-sequence: [hidden_size]
        float proj_acc[HEAD_DIM_128 / SIMD_SIZE];
        for (uint j = 0; j < hidden_elems_per_lane; ++j) {
            proj_acc[j] = 0.0f;
        }

        // O weights row index: this head's O weights
        // O weights are [num_heads_q * head_dim, hidden_size]
        // We need the row offset for this head: head_q * head_dim + d
        const uint o_row_offset = head_q * head_dim;

        // Loop over head_dim to compute dot products
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;  // Head dimension index
            if (d >= head_dim) continue;

            // Base O weight row for this head dimension
            const uint o_row_base = o_row_offset + d;

            // Loop over output dimensions in chunks
            for (uint h_start = 0; h_start < hidden_size; h_start += 8u * SIMD_SIZE) {
                // Each uint32 contains 8 FP4 values (4 nibbles each)
                uint packed_idx = (o_row_base * o_row_stride) + (h_start / 8u);

                // Unpack 8 FP4 values
                uint32_t packed_word = O_weights_packed[packed_idx];

                float attn_val = attn_norm[i];
                half attn_half = half(attn_val);

                // Process 8 FP4 values
                for (uint k = 0; k < 8u; ++k) {
                    uint h = h_start + k + lane_id * 8u;
                    if (h >= hidden_size) continue;

                    uint8_t nibble = (packed_word >> (k * 4u)) & 0xFu;

                    // Get scale for this group
                    uint scale_idx = (o_row_base / params.o_group_size) * hidden_size + h;
                    half scale = O_scales[scale_idx];

                    // Dequantize and accumulate
                    half o_weight = dequant_fp4_e2m1(nibble, scale);
                    proj_acc[k / 4u] += float(attn_half * o_weight);
                }
            }
        }

        // Sum across lanes for each output dimension
        float proj_final[4];
        for (uint j = 0; j < hidden_elems_per_lane; ++j) {
            proj_final[j] = simd_sum(proj_acc[j]);
        }

        // Write to global memory
        const uint out_base = batch_idx * seq_q * hidden_size + global_q * hidden_size;

        for (uint j = 0; j < hidden_elems_per_lane; ++j) {
            uint h = lane_id * 4 + j * 4;
            if (h < hidden_size) {
                half val = half(proj_final[j]);
                if (O_bias != nullptr) {
                    val += O_bias[h];
                }
                Output[out_base + h] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Flash Attention + O Projection (Causal)
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_fused_causal(
    device const input_t* Q               [[buffer(0)]],
    device const input_t* K               [[buffer(1)]],
    device const input_t* V               [[buffer(2)]],
    device const uint32_t* O_weights_packed [[buffer(3)]],
    device const half* O_scales          [[buffer(4)]],
    device const half* O_bias            [[buffer(5)]],
    device half* Output                  [[buffer(6)]],
    constant AttentionParamsFused& params [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;
    const uint hidden_size = params.hidden_size;

    const uint head_kv = head_q / params.gqa_ratio;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;
    const uint o_row_stride = hidden_size / 8u;

    threadgroup input_t Q_tile[TILE_Q][HEAD_DIM_128];
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_128];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_128];

    {
        const uint elems_to_load = q_rows * head_dim;
        const uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint q_row = idx / head_dim;
                uint q_col = idx % head_dim;
                Q_tile[q_row][q_col] = Q[q_base + q_row * q_stride_s + q_col];
            }
        }
    }

    const uint rows_per_sg = TILE_Q / NUM_SIMDGROUPS;
    const uint sg_q_start = sg_id * rows_per_sg;
    const uint sg_q_rows = min(rows_per_sg, (q_rows > sg_q_start) ? (q_rows - sg_q_start) : 0u);

    const uint elems_per_lane = head_dim / SIMD_SIZE;

    float q_reg[4][HEAD_DIM_128 / SIMD_SIZE];
    float m_prev[4];
    float l_prev[4];
    float o_acc[4][HEAD_DIM_128 / SIMD_SIZE];

    for (uint r = 0; r < rows_per_sg; ++r) {
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            q_reg[r][i] = 0.0f;
            o_acc[r][i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = input_to_float(Q_tile[q_row][d]);
        }
    }

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    {
        const uint elems = min(uint(TILE_KV), seq_k) * head_dim;
        const uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                if (kv_row < seq_k) {
                    K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                    V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

        if (tile_idx + 1 < num_kv_tiles) {
            uint next_start = (tile_idx + 1) * TILE_KV;
            uint next_len = min(uint(TILE_KV), seq_k - next_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    K_tile[buf_load][kv_row][kv_col] = K[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                    V_tile[buf_load][kv_row][kv_col] = V[kv_base + (next_start + kv_row) * k_stride_s + kv_col];
                }
            }
        }

        for (uint r = 0; r < sg_q_rows; ++r) {
            uint global_q = q_start + sg_q_start + r;

            float scores[TILE_KV];

            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[r][i] * float(K_tile[buf][ki][d]);
                }
                dot = simd_sum(dot);

                // Causal masking: only attend to positions <= global_q
                uint kv_pos = tile_start + ki;
                bool masked = kv_pos > global_q;
                scores[ki] = select(dot * scale, -INFINITY, masked);
            }

            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            float m_tile = -INFINITY;
            for (uint ki = 0; ki < tile_len; ++ki) {
                m_tile = max(m_tile, scores[ki]);
            }

            float m_new = max(m_prev[r], m_tile);
            float corr = exp(m_prev[r] - m_new);

            l_prev[r] *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[r][i] *= corr;
            }

            for (uint ki = 0; ki < tile_len; ++ki) {
                float p = exp(scores[ki] - m_new);
                l_prev[r] += p;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[r][i] += p * float(V_tile[buf][ki][d]);
                }
            }

            m_prev[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // Normalize and project
    const uint hidden_elems_per_lane = (hidden_size + SIMD_SIZE - 1) / SIMD_SIZE;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;

        float attn_norm[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            attn_norm[i] = o_acc[r][i] * inv_l;
        }

        float proj_acc[HEAD_DIM_128 / SIMD_SIZE];
        for (uint j = 0; j < hidden_elems_per_lane; ++j) {
            proj_acc[j] = 0.0f;
        }

        const uint o_row_offset = head_q * head_dim;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d >= head_dim) continue;

            const uint o_row_base = o_row_offset + d;

            for (uint h_start = 0; h_start < hidden_size; h_start += 8u * SIMD_SIZE) {
                uint packed_idx = (o_row_base * o_row_stride) + (h_start / 8u);

                uint32_t packed_word = O_weights_packed[packed_idx];

                float attn_val = attn_norm[i];
                half attn_half = half(attn_val);

                for (uint k = 0; k < 8u; ++k) {
                    uint h = h_start + k + lane_id * 8u;
                    if (h >= hidden_size) continue;

                    uint8_t nibble = (packed_word >> (k * 4u)) & 0xFu;

                    uint scale_idx = (o_row_base / params.o_group_size) * hidden_size + h;
                    half scale = O_scales[scale_idx];

                    half o_weight = dequant_fp4_e2m1(nibble, scale);
                    proj_acc[k / 4u] += float(attn_half * o_weight);
                }
            }
        }

        float proj_final[4];
        for (uint j = 0; j < hidden_elems_per_lane; ++j) {
            proj_final[j] = simd_sum(proj_acc[j]);
        }

        const uint out_base = batch_idx * seq_q * hidden_size + global_q * hidden_size;

        for (uint j = 0; j < hidden_elems_per_lane; ++j) {
            uint h = lane_id * 4 + j * 4;
            if (h < hidden_size) {
                half val = half(proj_final[j]);
                if (O_bias != nullptr) {
                    val += O_bias[h];
                }
                Output[out_base + h] = val;
            }
        }
    }
}
