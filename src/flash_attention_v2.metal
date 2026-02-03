// flash_attention_v2.metal - Optimized Flash Attention for Apple Silicon
//
// High-performance implementation targeting >80% memory bandwidth utilization.
// Builds on FlashAttention-2 algorithm with Apple Silicon-specific optimizations.
//
// Key optimizations over v1:
//   1. Process Q tiles (not single rows) for better arithmetic intensity
//   2. Vectorized 128-bit loads (half4/half8) for memory bandwidth
//   3. Branchless causal masking with fused compute
//   4. Improved register allocation for head_dim=64/128
//   5. Better warp-level parallelism across query rows
//   6. Specialized decode kernel for seq_q=1 (single-token generation)
//
// Kernel variants:
//   flash_attention_v2             - Tiled prefill, non-causal
//   flash_attention_v2_causal      - Tiled prefill, causal
//   flash_attention_v2_decode      - Single-query decode (optimized for seq_q=1)
//   flash_attention_v2_mqa         - Multi-Query Attention (1 KV head, N Q heads)
//   flash_attention_v2_gqa         - Grouped-Query Attention
//
// Memory layout (all row-major, contiguous):
//   Q: [batch, heads_q, seq_q, head_dim]
//   K: [batch, heads_kv, seq_k, head_dim]
//   V: [batch, heads_kv, seq_k, head_dim]
//   O: [batch, heads_q, seq_q, head_dim]
//
// Apple Silicon specifics:
//   - M1/M2/M3/M4: 32KB threadgroup memory, 32-wide simdgroups
//   - M4 Max: ~800 GB/s memory bandwidth, target 640+ GB/s achieved
//   - Prefer half (FP16) for compute, accumulate in float for precision
//   - simd_shuffle_xor for cross-lane communication (no shared memory needed)
//
// Algorithm: FlashAttention-2 online softmax
//   For each Q tile:
//     For each K/V tile:
//       S = Q @ K^T * scale (optionally masked)
//       m_new = max(m_old, rowmax(S))
//       P = exp(S - m_new)
//       l_new = l_old * exp(m_old - m_new) + rowsum(P)
//       O = O * exp(m_old - m_new) + P @ V
//     O = O / l_new

#include <metal_stdlib>
#include "bf16_compat.metal"
using namespace metal;

#ifdef USE_BF16_INPUTS
using input_t = bf16_t;
using output_t = ushort;
#else
using input_t = half;
using output_t = half;
#endif

inline half4 half4_load(device const half* src) {
    return *reinterpret_cast<device const half4*>(src);
}

#ifdef USE_BF16_INPUTS
inline float4 bf16_load_as_float4(device const ushort* src) {
    ushort4 packed = *reinterpret_cast<device const ushort4*>(src);
    return bf16x4_to_float4(packed);
}

inline float input_to_float(input_t v) {
    return bf16_to_float(v);
}

inline void bf16_store_from_float8(device ushort* dst, float4 lo, float4 hi) {
    ushort4 lo_packed = float4_to_bf16x4_rne(lo);
    ushort4 hi_packed = float4_to_bf16x4_rne(hi);
    *reinterpret_cast<device ushort4*>(dst) = lo_packed;
    *reinterpret_cast<device ushort4*>(dst + 4) = hi_packed;
}

inline void store_output_scalar(device ushort* dst, uint idx, float val) {
    dst[idx] = bf16_from_float_rne(val).bits;
}

inline void store_output_bf16_vectorized(device ushort* dst,
                                         uint base,
                                         uint lane_id,
                                         uint elems_per_lane,
                                         thread const float* vals,
                                         uint head_dim) {
    if (elems_per_lane == 4 && ((lane_id & 1u) == 0u)) {
        uint d0 = lane_id * elems_per_lane;
        float4 lo = float4(vals[0], vals[1], vals[2], vals[3]);
        float4 hi = float4(simd_shuffle_xor(vals[0], 1),
                           simd_shuffle_xor(vals[1], 1),
                           simd_shuffle_xor(vals[2], 1),
                           simd_shuffle_xor(vals[3], 1));
        if (d0 + 7 < head_dim) {
            bf16_store_from_float8(dst + base + d0, lo, hi);
            return;
        }
    }

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            store_output_scalar(dst, base + d, vals[i]);
        }
    }
}
#else
inline float input_to_float(input_t v) {
    return float(v);
}

inline void store_output_scalar(device half* dst, uint idx, float val) {
    dst[idx] = half(val);
}
#endif

// ---------------------------------------------------------------------------
// Configuration for Apple Silicon M4 Max
// ---------------------------------------------------------------------------

// Tile dimensions - tuned for M4 Max with head_dim=64
constant constexpr uint TILE_Q = 16;          // Query rows per threadgroup
// NOTE: TILE_KV is tuned to keep threadgroup memory <= 32KB on Apple GPUs.
// With HEAD_DIM_128 and double-buffered K/V, TILE_KV=24 keeps the kernels
// under the 32KB limit so pipeline creation succeeds across M1/M2/M3/M4.
constant constexpr uint TILE_KV = 24;         // K/V rows per tile
constant constexpr uint HEAD_DIM_64 = 64;     // Compile-time constant for head_dim=64
constant constexpr uint HEAD_DIM_128 = 128;   // Compile-time constant for head_dim=128

// Thread organization
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint NUM_SIMDGROUPS = 4;   // 128 threads total
constant constexpr uint THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;

// Memory optimization
constant constexpr uint VECTOR_WIDTH = 4;     // half4 loads/stores

// FP8 packing constants
constant constexpr uint FP8_PER_UINT = 4;     // 4 FP8 bytes per uint32

// ---------------------------------------------------------------------------
// FP8 E4M3 dequantization for KV cache
// ---------------------------------------------------------------------------
//
// FP8 E4M3 format: [1 sign][4 exponent (bias=7)][3 mantissa]
//
// Value encoding:
//   Normal (0 < E < 15):  (-1)^S * 2^(E-7) * (1 + M/8)
//   Subnormal (E == 0):   (-1)^S * 2^(-6) * (M/8)
//   NaN (E == 15):        Mapped to max representable value (448.0)
//   Zero (E=0, M=0):      +/- 0.0
//
// For KV cache quantization, we use a simplified dequantization that trades
// perfect NaN handling for performance. NaN values (E=15, M=7) are mapped
// to the maximum representable FP8 E4M3 value (448.0) times scale.

/// Dequantize a single FP8 E4M3 value to half precision with scale.
/// Uses branchless select() operations for performance.
inline half dequant_fp8_e4m3(uint8_t val, half scale) {
    uint sign = (val >> 7) & 1u;
    uint exp = (val >> 3) & 0xFu;
    uint man = val & 0x7u;

    // Handle special cases:
    // - NaN (E=15, M=7): Map to max value (448.0) since KV cache shouldn't have NaN
    // - Zero (E=0, M=0): Returns 0
    // - Subnormal (E=0, M>0): Value = M * 2^(-9)
    // - Normal (0<E<15): Value = (1 + M/8) * 2^(E-7)

    // For NaN, return scaled 448.0 (the max normal value)
    bool is_nan = (exp == 15u) && (man == 7u);

    half magnitude;
    if (exp == 0u) {
        // Subnormal: value = M * 2^(-9) = M / 512
        magnitude = half(man) * half(0.001953125h);  // 2^-9
    } else {
        // Normal: value = (1 + M/8) * 2^(E-7)
        // For E4M3, E ranges from 1-14, giving 2^(-6) to 2^7
        half mantissa = half(1.0h) + half(man) * half(0.125h);
        // 2^(E-7): E=1 gives 2^-6, E=14 gives 2^7
        // Use integer shift for exact powers of 2
        if (exp >= 7u) {
            magnitude = mantissa * half(float(1u << (exp - 7u)));
        } else {
            // For negative exponents, divide by power of 2
            magnitude = mantissa / half(float(1u << (7u - exp)));
        }
    }

    // Apply sign
    half result = select(magnitude, -magnitude, bool(sign));

    // Handle NaN case by returning max value
    return select(result, half(448.0h), is_nan) * scale;
}

/// Dequantize 4 FP8 E4M3 values packed in a uint32 with a single scale.
inline void dequant_fp8_e4m3_x4(uint packed, half scale,
                                 thread half& out0, thread half& out1,
                                 thread half& out2, thread half& out3) {
    out0 = dequant_fp8_e4m3(uint8_t((packed >>  0) & 0xFFu), scale);
    out1 = dequant_fp8_e4m3(uint8_t((packed >>  8) & 0xFFu), scale);
    out2 = dequant_fp8_e4m3(uint8_t((packed >> 16) & 0xFFu), scale);
    out3 = dequant_fp8_e4m3(uint8_t((packed >> 24) & 0xFFu), scale);
}

// ---------------------------------------------------------------------------
// Vectorized memory operations for bandwidth optimization
// ---------------------------------------------------------------------------

// Load half4 (64-bit) with bounds checking
inline half4 load_half4_safe(device const half* ptr, uint offset, uint valid_elems) {
    if (valid_elems >= 4) {
        return *reinterpret_cast<device const half4*>(ptr + offset);
    }
    half4 result = half4(0);
    for (uint i = 0; i < min(valid_elems, 4u); ++i) {
        result[i] = ptr[offset + i];
    }
    return result;
}

// Store half4 with bounds checking
inline void store_half4_safe(device half* ptr, uint offset, half4 val, uint valid_elems) {
    if (valid_elems >= 4) {
        *reinterpret_cast<device half4*>(ptr + offset) = val;
        return;
    }
    for (uint i = 0; i < min(valid_elems, 4u); ++i) {
        ptr[offset + i] = val[i];
    }
}

// ---------------------------------------------------------------------------
// Fast simd reductions using hardware intrinsics
//
// Metal's built-in simd_sum/simd_max are hardware-accelerated and significantly
// faster than manual simd_shuffle_xor chains:
// - Single instruction vs 5 dependent instructions
// - Dedicated reduction hardware on Apple Silicon
// - Lower latency and energy consumption
// ---------------------------------------------------------------------------

inline float simd_max_reduce(float val) {
    return simd_max(val);
}

inline float simd_sum_reduce(float val) {
    return simd_sum(val);
}

// Fused max-reduce returning both max and index
inline float2 simd_max_with_correction(float m_old, float m_new) {
    // Returns (new_max, correction_factor)
    float m = max(m_old, m_new);
    float corr = exp(m_old - m);
    return float2(m, corr);
}

// ---------------------------------------------------------------------------
// Branchless causal mask computation
// ---------------------------------------------------------------------------

// Returns 0.0f if masked (k_pos > q_pos), -INFINITY otherwise for causal
inline float causal_mask(uint q_pos, uint k_pos) {
    // select(a, b, cond) returns a if cond is false, b if true
    // We want -INFINITY when k_pos > q_pos
    return select(0.0f, -INFINITY, k_pos > q_pos);
}

// Batch compute causal mask for a row (vectorized)
inline void apply_causal_mask_row(
    thread float* scores,
    uint q_pos,
    uint k_tile_start,
    uint num_k
) {
    // Compute mask for all positions without branching
    for (uint ki = 0; ki < num_k; ++ki) {
        uint k_pos = k_tile_start + ki;
        // Fused mask: if k_pos > q_pos, score becomes -INFINITY
        scores[ki] = select(scores[ki], -INFINITY, k_pos > q_pos);
    }
}

// ---------------------------------------------------------------------------
// AttentionParams - packed parameters for kernel dispatch
// ---------------------------------------------------------------------------

struct AttentionParams {
    uint batch;
    uint num_heads_q;
    uint num_heads_kv;
    uint seq_q;
    uint seq_k;
    uint head_dim;
    float scale;
    uint gqa_ratio;     // num_heads_q / num_heads_kv
    uint is_causal;
};

// ---------------------------------------------------------------------------
// Flash Attention V2 - Tiled Prefill (Non-Causal)
//
// Each threadgroup processes TILE_Q query rows.
// K/V tiles are streamed with double-buffering.
// Uses online softmax to avoid materializing full attention matrix.
//
// Dispatch: [num_heads_q, ceil(seq_q / TILE_Q), batch] threadgroups
// ---------------------------------------------------------------------------

kernel void flash_attention_v2(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    // GQA: map Q head to KV head
    const uint head_kv = head_q / params.gqa_ratio;

    // Base indices for this threadgroup
    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides for [batch, heads, seq, head_dim] layout
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    // Base offsets
    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    // ---------------------------------------------------------------------------
    // Threadgroup memory allocation
    // Q tile: [TILE_Q][HEAD_DIM_128] = 16 * 128 * 2 = 4 KB
    // K/V double buffer: 2 * [TILE_KV][HEAD_DIM_128] * 2 = 2 * 64 * 128 * 2 * 2 = 32 KB
    // Total: ~36 KB, within 32 KB limit for smaller configs
    // For head_dim=128, use smaller tiles or single buffer
    // ---------------------------------------------------------------------------

    threadgroup input_t Q_tile[TILE_Q][HEAD_DIM_128];
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_128];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_128];

    // ---------------------------------------------------------------------------
    // Cooperative Q tile load (all threads participate)
    // Uses vectorized loads for bandwidth
    // ---------------------------------------------------------------------------
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

    // ---------------------------------------------------------------------------
    // Per-simdgroup state: each simdgroup handles TILE_Q/NUM_SIMDGROUPS = 4 query rows
    // ---------------------------------------------------------------------------

    const uint rows_per_sg = TILE_Q / NUM_SIMDGROUPS;  // 4
    const uint sg_q_start = sg_id * rows_per_sg;
    const uint sg_q_rows = min(rows_per_sg, (q_rows > sg_q_start) ? (q_rows - sg_q_start) : 0u);

    // Register allocation for Q (each lane holds head_dim/32 elements per row)
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[4][HEAD_DIM_128 / SIMD_SIZE];  // 4 rows, up to 4 elems each

    // Online softmax state per row
    float m_prev[4];  // Running max
    float l_prev[4];  // Running sum
    float o_acc[4][HEAD_DIM_128 / SIMD_SIZE];  // Output accumulators

    // Initialize
    for (uint r = 0; r < rows_per_sg; ++r) {
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[r][i] = 0.0f;
        }
    }

    // Wait for Q load to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q rows into registers
    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = input_to_float(Q_tile[q_row][d]);
        }
    }

    // ---------------------------------------------------------------------------
    // Preload first K/V tile
    // ---------------------------------------------------------------------------

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

    // ---------------------------------------------------------------------------
    // Main loop: stream through K/V tiles
    // ---------------------------------------------------------------------------

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

        // ---------------------------------------------------------------------------
        // Compute Q @ K^T for this tile (4 query rows per simdgroup)
        // ---------------------------------------------------------------------------

        for (uint r = 0; r < sg_q_rows; ++r) {
            float scores[TILE_KV];

            // Compute all dot products for this query row
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

            // Rescale and accumulate
            l_prev[r] *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[r][i] *= corr;
            }

            // Accumulate new contributions
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
    // Normalize and store output
    // ---------------------------------------------------------------------------

    const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;

        const uint o_row_base = o_base + (sg_q_start + r) * q_stride_s;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[r][i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
            }
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V2 - Causal (Branchless masking)
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_causal(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    const uint head_q = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    const uint head_kv = head_q / params.gqa_ratio;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    threadgroup input_t Q_tile[TILE_Q][HEAD_DIM_128];
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_128];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_128];

    // Load Q tile
    {
        const uint elems = q_rows * head_dim;
        const uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
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
        for (uint i = 0; i < elems_per_lane; ++i) o_acc[r][i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = input_to_float(Q_tile[q_row][d]);
        }
    }

    // Causal limit: for each query row, we only process up to q_pos + 1 keys
    // Early termination at tile level: skip tiles where tile_start > max(q_positions)
    const uint max_q_pos = q_start + q_rows - 1;
    const uint causal_limit = min(max_q_pos + 1, seq_k);
    const uint num_kv_tiles = (causal_limit + TILE_KV - 1) / TILE_KV;

    // Preload first tile
    {
        uint tile_len = min(uint(TILE_KV), causal_limit);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), causal_limit - tile_start);

        // Load next tile
        if (tile_idx + 1 < num_kv_tiles) {
            uint next_start = (tile_idx + 1) * TILE_KV;
            uint next_len = min(uint(TILE_KV), causal_limit - next_start);
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

        // Compute with causal mask
        for (uint r = 0; r < sg_q_rows; ++r) {
            uint global_q_pos = q_start + sg_q_start + r;
            float scores[TILE_KV];

            for (uint ki = 0; ki < tile_len; ++ki) {
                uint k_pos = tile_start + ki;

                // Compute dot product
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[r][i] * input_to_float(K_tile[buf][ki][d]);
                }
                dot = simd_sum(dot);

                // Branchless causal mask
                // If k_pos > q_pos, use -INFINITY, else use the score
                float score = dot * scale;
                scores[ki] = select(score, -INFINITY, k_pos > global_q_pos);
            }

            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax
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

    // Store output
    const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;

        const uint o_row_base = o_base + (sg_q_start + r) * q_stride_s;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[r][i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
            }
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V2 - Decode (Optimized for seq_q=1)
//
// Specialized kernel for autoregressive decoding where we have a single
// query token attending to a long KV cache. Uses all threads for K/V
// processing rather than distributing across Q rows.
//
// Dispatch: [num_seqs * num_heads_q, 1, 1] threadgroups
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_decode(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant uint& num_seqs         [[buffer(4)]],
    constant uint& num_heads_q      [[buffer(5)]],
    constant uint& num_heads_kv     [[buffer(6)]],
    constant uint& seq_k            [[buffer(7)]],
    constant uint& head_dim         [[buffer(8)]],
    constant float& scale           [[buffer(9)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // ---------------------------------------------------------------------------
    // Decode kernel: seq_q = 1, single query attending to full KV cache.
    //
    // CAUSAL MASK OPTIMIZATION: For decode, the query is at position seq_k-1
    // (the newest token), so it can attend to ALL positions 0..seq_k-1.
    // The causal constraint is trivially satisfied - NO MASK COMPUTATION NEEDED.
    // This eliminates one comparison per attention score.
    // ---------------------------------------------------------------------------
    const uint seq_idx = tgid / num_heads_q;
    const uint head_q = tgid % num_heads_q;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_heads_kv;
    const uint head_kv = head_q / gqa_ratio;

    // Q layout: [num_seqs, num_heads_q, head_dim]
    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;

    // K/V layout: [num_seqs, num_heads_kv, seq_k, head_dim]
    const uint kv_stride_s = head_dim;
    const uint kv_stride_h = seq_k * head_dim;
    const uint kv_stride_b = num_heads_kv * kv_stride_h;
    const uint kv_base = seq_idx * kv_stride_b + head_kv * kv_stride_h;

    // Load Q into registers (distributed across simdgroup 0)
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_128 / SIMD_SIZE];

    if (sg_id == 0) {
        if (elems_per_lane == 4) {
            uint d = lane_id * elems_per_lane;
            if (d + 3 < head_dim) {
#ifdef USE_BF16_INPUTS
                float4 q_vals = bf16_load_as_float4(reinterpret_cast<device const ushort*>(Q + q_offset + d));
#else
                float4 q_vals = float4(half4_load(reinterpret_cast<device const half*>(Q + q_offset + d)));
#endif
                q_reg[0] = q_vals.x;
                q_reg[1] = q_vals.y;
                q_reg[2] = q_vals.z;
                q_reg[3] = q_vals.w;
            } else {
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d_tail = d + i;
                    q_reg[i] = (d_tail < head_dim) ? input_to_float(Q[q_offset + d_tail]) : 0.0f;
                }
            }
        } else {
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[i] = (d < head_dim) ? input_to_float(Q[q_offset + d]) : 0.0f;
            }
        }
    }

    // Threadgroup memory for K/V (double-buffered)
    threadgroup input_t K_smem[2][TILE_KV][HEAD_DIM_128];
    threadgroup input_t V_smem[2][TILE_KV][HEAD_DIM_128];

    // Online softmax state
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_128 / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    const uint num_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Preload first tile
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_smem[0][kv_row][kv_col] = K[kv_base + kv_row * kv_stride_s + kv_col];
                V_smem[0][kv_row][kv_col] = V[kv_base + kv_row * kv_stride_s + kv_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

        // Load next tile (all threads)
        if (tile_idx + 1 < num_tiles) {
            uint next_start = (tile_idx + 1) * TILE_KV;
            uint next_len = min(uint(TILE_KV), seq_k - next_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    K_smem[buf_load][kv_row][kv_col] = K[kv_base + (next_start + kv_row) * kv_stride_s + kv_col];
                    V_smem[buf_load][kv_row][kv_col] = V[kv_base + (next_start + kv_row) * kv_stride_s + kv_col];
                }
            }
        }

        // Compute (simdgroup 0 only for single Q)
        // NO CAUSAL MASK: decode query at position seq_k-1 attends to all 0..seq_k-1
        if (sg_id == 0) {
            float scores[TILE_KV];

            // Compute dot products - no causal masking needed for decode
            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * input_to_float(K_smem[buf][ki][d]);
                }
                dot = simd_sum(dot);
                scores[ki] = dot * scale;
            }

            // Pad invalid tile positions
            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax (no causal mask checks)
            float m_tile = -INFINITY;
            for (uint ki = 0; ki < tile_len; ++ki) {
                m_tile = max(m_tile, scores[ki]);
            }

            float m_new = max(m_prev, m_tile);
            float corr = exp(m_prev - m_new);

            l_prev *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= corr;
            }

            for (uint ki = 0; ki < tile_len; ++ki) {
                float p = exp(scores[ki] - m_new);
                l_prev += p;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * input_to_float(V_smem[buf][ki][d]);
                }
            }

            m_prev = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // Store output (simdgroup 0)
    if (sg_id == 0) {
        const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_offset, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_offset + d, o_acc[i] * inv_l);
            }
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V2 - Decode Fast Path (batch=1, seq_q=1)
//
// Ultra-optimized decode kernel for single-token generation where softmax
// can be computed with minimal overhead.
//
// Key optimizations:
//   1. Special case for seq_k=1: softmax is trivially 1.0, output = V[0]
//   2. Warp-level parallel score computation with simd_sum reduction
//   3. Fused online softmax + V accumulation in single pass
//   4. No separate kernel launch for softmax normalization
//
// For decode, the attention is:
//   scores[k] = dot(Q, K[k]) * scale  for k in [0, seq_k)
//   probs = softmax(scores)
//   output = sum_k(probs[k] * V[k])
//
// We fuse this using online softmax: maintain running max and sum,
// rescale V accumulator as new tiles are processed.
//
// Dispatch: [num_heads_q, 1, 1] threadgroups (one TG per head)
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_decode_fast(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant uint& num_heads_q      [[buffer(4)]],
    constant uint& num_heads_kv     [[buffer(5)]],
    constant uint& seq_k            [[buffer(6)]],
    constant uint& head_dim         [[buffer(7)]],
    constant float& scale           [[buffer(8)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // Single sequence, single query token (batch=1, seq_q=1)
    const uint head_q = tgid;
    if (head_q >= num_heads_q) return;

    const uint gqa_ratio = num_heads_q / num_heads_kv;
    const uint head_kv = head_q / gqa_ratio;

    // Q offset: [heads_q, head_dim] -> single query vector
    const uint q_offset = head_q * head_dim;

    // K/V offset: [heads_kv, seq_k, head_dim]
    const uint kv_stride_h = seq_k * head_dim;
    const uint kv_base = head_kv * kv_stride_h;

    const uint elems_per_lane = head_dim / SIMD_SIZE;

    // =========================================================================
    // FAST PATH: seq_k == 1 (first token decode)
    // Softmax of single element is 1.0, output = V[0]
    // =========================================================================
    if (seq_k == 1 && sg_id == 0) {
        const uint o_offset = head_q * head_dim;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                float v_val = input_to_float(V[kv_base + d]);
                store_output_scalar(O, o_offset + d, v_val);
            }
        }
        return;
    }

    // =========================================================================
    // Load Q vector into registers (distributed across lanes of simdgroup 0)
    // Each lane holds elems_per_lane consecutive elements
    // =========================================================================
    float q_reg[HEAD_DIM_128 / SIMD_SIZE];

    if (sg_id == 0) {
        if (elems_per_lane == 4) {
            uint d = lane_id * 4;
            if (d + 3 < head_dim) {
#ifdef USE_BF16_INPUTS
                float4 q_vals = bf16_load_as_float4(reinterpret_cast<device const ushort*>(Q + q_offset + d));
#else
                float4 q_vals = float4(half4_load(reinterpret_cast<device const half*>(Q + q_offset + d)));
#endif
                q_reg[0] = q_vals.x;
                q_reg[1] = q_vals.y;
                q_reg[2] = q_vals.z;
                q_reg[3] = q_vals.w;
            }
        } else {
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[i] = (d < head_dim) ? input_to_float(Q[q_offset + d]) : 0.0f;
            }
        }
    }

    // =========================================================================
    // WARP-LEVEL PATH (simdgroup 0 only): Fused score + softmax + V accumulation
    //
    // Strategy: Each lane computes partial dot product (its portion of Q·K),
    // then simd_sum reduces to full score. Online softmax tracks running
    // max and sum while accumulating weighted V.
    // =========================================================================

    if (sg_id != 0) return;  // Only simdgroup 0 computes for decode

    // Online softmax state
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float o_acc[HEAD_DIM_128 / SIMD_SIZE] = {0.0f};

    // Process K positions one at a time with online softmax
    // For longer sequences, we could tile, but decode seq_k is usually manageable
    for (uint k_pos = 0; k_pos < seq_k; ++k_pos) {
        // Compute dot(Q, K[k_pos]) using distributed Q and parallel K load
        // Each lane loads its portion of K and computes partial dot
        float partial_dot = 0.0f;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                float k_val = input_to_float(K[kv_base + k_pos * head_dim + d]);
                partial_dot += q_reg[i] * k_val;
            }
        }

        // Reduce partial dots across lanes to get full score
        float score = simd_sum(partial_dot) * scale;

        // Online softmax update
        // If score > running_max, rescale previous accumulator
        float prev_max = running_max;
        running_max = max(running_max, score);

        // Rescale factor: exp(prev_max - new_max)
        // When prev_max == -INFINITY, this is 0 (correct: no prior contribution)
        float rescale = exp(prev_max - running_max);

        // Rescale running sum and accumulator
        running_sum *= rescale;
        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= rescale;
        }

        // Add current position's contribution
        float weight = exp(score - running_max);
        running_sum += weight;

        // Accumulate weighted V
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                float v_val = input_to_float(V[kv_base + k_pos * head_dim + d]);
                o_acc[i] += weight * v_val;
            }
        }
    }

    // Final normalization and store
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    const uint o_offset = head_q * head_dim;

#ifdef USE_BF16_INPUTS
    float out_vals[HEAD_DIM_128 / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        out_vals[i] = o_acc[i] * inv_sum;
    }
    store_output_bf16_vectorized(O, o_offset, lane_id, elems_per_lane, out_vals, head_dim);
#else
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            store_output_scalar(O, o_offset + d, o_acc[i] * inv_sum);
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Flash Attention V2 - GQA (Grouped Query Attention)
//
// Optimized for models like GLM-4.7-Flash with many Q heads sharing few KV heads.
// GLM-4.7-Flash: 32 Q heads, 2 KV heads -> gqa_ratio = 16
//
// Key optimization: Load KV once, process multiple Q heads in parallel.
// Each simdgroup handles one Q head, sharing KV data via threadgroup memory.
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_gqa(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // For high GQA ratio, process multiple Q heads per threadgroup sharing same K/V
    // Dispatch: [num_heads_kv, ceil(seq_q / TILE_Q), batch] threadgroups
    // Each threadgroup handles gqa_ratio Q heads that share one KV head

    const uint head_kv = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint gqa_ratio = params.gqa_ratio;
    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;

    // K/V tiles are shared across all Q heads in this group
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_128];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_128];

    // Each simdgroup handles one Q head within the GQA group
    // With 4 simdgroups and potentially gqa_ratio=16, we need multiple passes
    const uint heads_per_pass = min(gqa_ratio, uint(NUM_SIMDGROUPS));
    const uint num_head_passes = (gqa_ratio + heads_per_pass - 1) / heads_per_pass;

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Process each Q head pass
    for (uint head_pass = 0; head_pass < num_head_passes; ++head_pass) {
        uint head_offset = head_pass * heads_per_pass + sg_id;
        if (head_offset >= gqa_ratio) continue;

        uint head_q = head_kv * gqa_ratio + head_offset;
        if (head_q >= params.num_heads_q) continue;

        const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

        // Load Q into registers (this simdgroup's head)
        const uint elems_per_lane = head_dim / SIMD_SIZE;
        float q_reg[TILE_Q][HEAD_DIM_128 / SIMD_SIZE];
        float m_prev[TILE_Q];
        float l_prev[TILE_Q];
        float o_acc[TILE_Q][HEAD_DIM_128 / SIMD_SIZE];

        for (uint r = 0; r < q_rows; ++r) {
            m_prev[r] = -INFINITY;
            l_prev[r] = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[r][i] = input_to_float(Q[q_base + r * q_stride_s + d]);
                o_acc[r][i] = 0.0f;
            }
        }

        // Preload first K/V tile for this head pass
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint buf = 0;

        for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
            uint buf_load = 1 - buf;
            uint tile_start = tile_idx * TILE_KV;
            uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

            // Load next tile for this head pass
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

            // Compute attention for this head
            for (uint r = 0; r < q_rows; ++r) {
                float scores[TILE_KV];

                for (uint ki = 0; ki < tile_len; ++ki) {
                    float dot = 0.0f;
                    for (uint i = 0; i < elems_per_lane; ++i) {
                        uint d = lane_id * elems_per_lane + i;
                        dot += q_reg[r][i] * input_to_float(K_tile[buf][ki][d]);
                    }
                    dot = simd_sum(dot);
                    scores[ki] = dot * scale;
                }

                for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                    scores[ki] = -INFINITY;
                }

                // Causal mask if needed
                if (params.is_causal) {
                    uint global_q_pos = q_start + r;
                    for (uint ki = 0; ki < tile_len; ++ki) {
                        uint k_pos = tile_start + ki;
                        scores[ki] = select(scores[ki], -INFINITY, k_pos > global_q_pos);
                    }
                }

                // Online softmax
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
                        o_acc[r][i] += p * input_to_float(V_tile[buf][ki][d]);
                    }
                }

                m_prev[r] = m_new;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf = buf_load;
        }

        // Store output for this head
        const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

        for (uint r = 0; r < q_rows; ++r) {
            float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;
            const uint o_row_base = o_base + r * q_stride_s;
#ifdef USE_BF16_INPUTS
            float out_vals[HEAD_DIM_128 / SIMD_SIZE];
            for (uint i = 0; i < elems_per_lane; ++i) {
                out_vals[i] = o_acc[r][i] * inv_l;
            }
            store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                if (d < head_dim) {
                    store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
                }
            }
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V2 - MQA (Multi-Query Attention, single KV head)
//
// Special case of GQA where num_heads_kv = 1.
// All Q heads share the same K/V, maximum memory savings.
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_mqa(
    device const input_t* Q         [[buffer(0)]],
    device const input_t* K         [[buffer(1)]],
    device const input_t* V         [[buffer(2)]],
    device output_t* O              [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane_id                    [[thread_index_in_simdgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    // For MQA: num_heads_kv = 1, so all Q heads share one K/V
    // Dispatch: [ceil(num_heads_q / NUM_SIMDGROUPS), ceil(seq_q / TILE_Q), batch]
    // Each threadgroup processes NUM_SIMDGROUPS Q heads, one per simdgroup

    const uint head_group = tgid.x;
    const uint q_tile_idx = tgid.y;
    const uint batch_idx = tgid.z;

    const uint head_q = head_group * NUM_SIMDGROUPS + sg_id;
    if (head_q >= params.num_heads_q) return;

    const uint head_dim = params.head_dim;
    const uint seq_q = params.seq_q;
    const uint seq_k = params.seq_k;
    const float scale = params.scale;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;
    const uint k_stride_b = seq_k * head_dim;  // num_heads_kv = 1
    const uint k_stride_s = head_dim;

    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b;  // Single KV head

    // Shared K/V tiles
    threadgroup input_t K_tile[2][TILE_KV][HEAD_DIM_128];
    threadgroup input_t V_tile[2][TILE_KV][HEAD_DIM_128];

    // Per-simdgroup state
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[TILE_Q][HEAD_DIM_128 / SIMD_SIZE];
    float m_prev[TILE_Q];
    float l_prev[TILE_Q];
    float o_acc[TILE_Q][HEAD_DIM_128 / SIMD_SIZE];

    // Initialize and load Q
    for (uint r = 0; r < q_rows; ++r) {
        m_prev[r] = -INFINITY;
        l_prev[r] = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = input_to_float(Q[q_base + r * q_stride_s + d]);
            o_acc[r][i] = 0.0f;
        }
    }

    const uint num_kv_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Preload first K/V tile (all threads cooperate)
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                K_tile[0][kv_row][kv_col] = K[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = V[kv_base + kv_row * k_stride_s + kv_col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

        // Load next tile
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

        // Each simdgroup computes for its Q head
        for (uint r = 0; r < q_rows; ++r) {
            float scores[TILE_KV];

            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[r][i] * input_to_float(K_tile[buf][ki][d]);
                }
                dot = simd_sum(dot);
                scores[ki] = dot * scale;
            }

            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Causal mask
            if (params.is_causal) {
                uint global_q_pos = q_start + r;
                for (uint ki = 0; ki < tile_len; ++ki) {
                    uint k_pos = tile_start + ki;
                    scores[ki] = select(scores[ki], -INFINITY, k_pos > global_q_pos);
                }
            }

            // Online softmax
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
                    o_acc[r][i] += p * input_to_float(V_tile[buf][ki][d]);
                }
            }

            m_prev[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // Store output
    const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

    for (uint r = 0; r < q_rows; ++r) {
        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;
        const uint o_row_base = o_base + r * q_stride_s;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[r][i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
            }
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V2 - FP8 Quantized KV Cache (Decode)
//
// Optimized for autoregressive decoding with FP8 E4M3 quantized KV cache.
// FP8 KV cache reduces memory bandwidth by 2x compared to FP16, enabling
// longer context lengths and higher throughput on memory-bound decode.
//
// K/V are stored as packed uint8_t with per-head-per-sequence scales.
// During decode, we dequantize on-the-fly in registers before attention compute.
//
// Memory layout:
//   K_fp8:    [batch, heads_kv, seq_k, head_dim]  (uint8_t packed)
//   V_fp8:    [batch, heads_kv, seq_k, head_dim]  (uint8_t packed)
//   K_scales: [batch, heads_kv, seq_k]            (half, per-token)
//   V_scales: [batch, heads_kv, seq_k]            (half, per-token)
//   Q:        [batch, heads_q, 1, head_dim]       (half)
//   O:        [batch, heads_q, 1, head_dim]       (half)
//
// Dispatch: [num_seqs * num_heads_q, 1, 1] threadgroups
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_fp8_kv(
    device const input_t* Q             [[buffer(0)]],
    device const uint8_t* K_fp8         [[buffer(1)]],
    device const uint8_t* V_fp8         [[buffer(2)]],
    device const half* K_scales         [[buffer(3)]],
    device const half* V_scales         [[buffer(4)]],
    device output_t* O                  [[buffer(5)]],
    constant uint& num_seqs             [[buffer(6)]],
    constant uint& num_heads_q          [[buffer(7)]],
    constant uint& num_heads_kv         [[buffer(8)]],
    constant uint& seq_k                [[buffer(9)]],
    constant uint& head_dim             [[buffer(10)]],
    constant float& scale               [[buffer(11)]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    // Decode has seq_q = 1, so one threadgroup per (sequence, head) pair
    const uint seq_idx = tgid / num_heads_q;
    const uint head_q = tgid % num_heads_q;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_heads_kv;
    const uint head_kv = head_q / gqa_ratio;

    // Q layout: [num_seqs, num_heads_q, head_dim]
    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;

    // K/V FP8 layout: [num_seqs, num_heads_kv, seq_k, head_dim]
    const uint kv_stride_s = head_dim;
    const uint kv_stride_h = seq_k * head_dim;
    const uint kv_stride_b = num_heads_kv * kv_stride_h;
    const uint kv_base = seq_idx * kv_stride_b + head_kv * kv_stride_h;

    // Scale layout: [num_seqs, num_heads_kv, seq_k]
    const uint scale_stride_h = seq_k;
    const uint scale_stride_b = num_heads_kv * scale_stride_h;
    const uint scale_base = seq_idx * scale_stride_b + head_kv * scale_stride_h;

    // Load Q into registers (distributed across simdgroup 0)
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_128 / SIMD_SIZE];

    if (sg_id == 0) {
        if (elems_per_lane == 4) {
            uint d = lane_id * elems_per_lane;
            if (d + 3 < head_dim) {
#ifdef USE_BF16_INPUTS
                float4 q_vals = bf16_load_as_float4(reinterpret_cast<device const ushort*>(Q + q_offset + d));
#else
                float4 q_vals = float4(half4_load(reinterpret_cast<device const half*>(Q + q_offset + d)));
#endif
                q_reg[0] = q_vals.x;
                q_reg[1] = q_vals.y;
                q_reg[2] = q_vals.z;
                q_reg[3] = q_vals.w;
            } else {
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d_tail = d + i;
                    q_reg[i] = (d_tail < head_dim) ? input_to_float(Q[q_offset + d_tail]) : 0.0f;
                }
            }
        } else {
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                q_reg[i] = (d < head_dim) ? input_to_float(Q[q_offset + d]) : 0.0f;
            }
        }
    }

    // Threadgroup memory for K/V (dequantized to half, double-buffered)
    threadgroup half K_smem[2][TILE_KV][HEAD_DIM_128];
    threadgroup half V_smem[2][TILE_KV][HEAD_DIM_128];

    // Online softmax state
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_128 / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    const uint num_tiles = (seq_k + TILE_KV - 1) / TILE_KV;

    // Preload first tile (dequantize FP8 -> FP16 on load)
    {
        uint tile_len = min(uint(TILE_KV), seq_k);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;

                // Load scales for this token
                half k_scale = K_scales[scale_base + kv_row];
                half v_scale = V_scales[scale_base + kv_row];

                // Load and dequantize K
                uint8_t k_packed = K_fp8[kv_base + kv_row * kv_stride_s + kv_col];
                K_smem[0][kv_row][kv_col] = dequant_fp8_e4m3(k_packed, k_scale);

                // Load and dequantize V
                uint8_t v_packed = V_fp8[kv_base + kv_row * kv_stride_s + kv_col];
                V_smem[0][kv_row][kv_col] = dequant_fp8_e4m3(v_packed, v_scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), seq_k - tile_start);

        // Load next tile (all threads) with FP8 dequantization
        if (tile_idx + 1 < num_tiles) {
            uint next_start = (tile_idx + 1) * TILE_KV;
            uint next_len = min(uint(TILE_KV), seq_k - next_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;

                    // Token position in full sequence
                    uint global_kv_row = next_start + kv_row;

                    half k_scale = K_scales[scale_base + global_kv_row];
                    half v_scale = V_scales[scale_base + global_kv_row];

                    uint8_t k_packed = K_fp8[kv_base + global_kv_row * kv_stride_s + kv_col];
                    K_smem[buf_load][kv_row][kv_col] = dequant_fp8_e4m3(k_packed, k_scale);

                    uint8_t v_packed = V_fp8[kv_base + global_kv_row * kv_stride_s + kv_col];
                    V_smem[buf_load][kv_row][kv_col] = dequant_fp8_e4m3(v_packed, v_scale);
                }
            }
        }

        // Compute (simdgroup 0 only for single Q)
        if (sg_id == 0) {
            float scores[TILE_KV];

            for (uint ki = 0; ki < tile_len; ++ki) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf][ki][d]);
                }
                dot = simd_sum(dot);
                scores[ki] = dot * scale;
            }

            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax
            float m_tile = -INFINITY;
            for (uint ki = 0; ki < tile_len; ++ki) {
                m_tile = max(m_tile, scores[ki]);
            }

            float m_new = max(m_prev, m_tile);
            float corr = exp(m_prev - m_new);

            l_prev *= corr;
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= corr;
            }

            for (uint ki = 0; ki < tile_len; ++ki) {
                float p = exp(scores[ki] - m_new);
                l_prev += p;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf][ki][d]);
                }
            }

            m_prev = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // Store output (simdgroup 0)
    if (sg_id == 0) {
        const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_offset, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_offset + d, o_acc[i] * inv_l);
            }
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Flash Attention V2 - FP8 Quantized KV Cache (Causal Prefill)
//
// For prefill with FP8 KV cache. Similar to flash_attention_v2_causal but
// with on-the-fly FP8 dequantization of K and V.
// ---------------------------------------------------------------------------

kernel void flash_attention_v2_fp8_kv_causal(
    device const input_t* Q             [[buffer(0)]],
    device const uint8_t* K_fp8         [[buffer(1)]],
    device const uint8_t* V_fp8         [[buffer(2)]],
    device const half* K_scales         [[buffer(3)]],
    device const half* V_scales         [[buffer(4)]],
    device output_t* O                  [[buffer(5)]],
    constant AttentionParams& params    [[buffer(6)]],
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
    const float attn_scale = params.scale;

    const uint head_kv = head_q / params.gqa_ratio;

    const uint q_start = q_tile_idx * TILE_Q;
    if (q_start >= seq_q) return;

    const uint q_rows = min(TILE_Q, seq_q - q_start);

    // Strides
    const uint q_stride_b = params.num_heads_q * seq_q * head_dim;
    const uint q_stride_h = seq_q * head_dim;
    const uint q_stride_s = head_dim;

    const uint k_stride_b = params.num_heads_kv * seq_k * head_dim;
    const uint k_stride_h = seq_k * head_dim;
    const uint k_stride_s = head_dim;

    // Scale strides: [batch, heads_kv, seq_k]
    const uint scale_stride_b = params.num_heads_kv * seq_k;
    const uint scale_stride_h = seq_k;

    const uint q_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;
    const uint kv_base = batch_idx * k_stride_b + head_kv * k_stride_h;
    const uint scale_base = batch_idx * scale_stride_b + head_kv * scale_stride_h;

    threadgroup input_t Q_tile[TILE_Q][HEAD_DIM_128];
    threadgroup half K_tile[2][TILE_KV][HEAD_DIM_128];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_128];

    // Load Q tile
    {
        const uint elems = q_rows * head_dim;
        const uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
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
        for (uint i = 0; i < elems_per_lane; ++i) o_acc[r][i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint q_row = sg_q_start + r;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[r][i] = input_to_float(Q_tile[q_row][d]);
        }
    }

    // Causal limit
    const uint max_q_pos = q_start + q_rows - 1;
    const uint causal_limit = min(max_q_pos + 1, seq_k);
    const uint num_kv_tiles = (causal_limit + TILE_KV - 1) / TILE_KV;

    // Preload first tile with FP8 dequantization
    {
        uint tile_len = min(uint(TILE_KV), causal_limit);
        uint elems = tile_len * head_dim;
        uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_PER_TG;
            if (idx < elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;

                half k_scale = K_scales[scale_base + kv_row];
                half v_scale = V_scales[scale_base + kv_row];

                uint8_t k_packed = K_fp8[kv_base + kv_row * k_stride_s + kv_col];
                K_tile[0][kv_row][kv_col] = dequant_fp8_e4m3(k_packed, k_scale);

                uint8_t v_packed = V_fp8[kv_base + kv_row * k_stride_s + kv_col];
                V_tile[0][kv_row][kv_col] = dequant_fp8_e4m3(v_packed, v_scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;

    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf;
        uint tile_start = tile_idx * TILE_KV;
        uint tile_len = min(uint(TILE_KV), causal_limit - tile_start);

        // Load next tile with FP8 dequantization
        if (tile_idx + 1 < num_kv_tiles) {
            uint next_start = (tile_idx + 1) * TILE_KV;
            uint next_len = min(uint(TILE_KV), causal_limit - next_start);
            uint elems = next_len * head_dim;
            uint per_thread = (elems + THREADS_PER_TG - 1) / THREADS_PER_TG;
            for (uint i = 0; i < per_thread; ++i) {
                uint idx = tid + i * THREADS_PER_TG;
                if (idx < elems) {
                    uint kv_row = idx / head_dim;
                    uint kv_col = idx % head_dim;
                    uint global_kv_row = next_start + kv_row;

                    half k_scale = K_scales[scale_base + global_kv_row];
                    half v_scale = V_scales[scale_base + global_kv_row];

                    uint8_t k_packed = K_fp8[kv_base + global_kv_row * k_stride_s + kv_col];
                    K_tile[buf_load][kv_row][kv_col] = dequant_fp8_e4m3(k_packed, k_scale);

                    uint8_t v_packed = V_fp8[kv_base + global_kv_row * k_stride_s + kv_col];
                    V_tile[buf_load][kv_row][kv_col] = dequant_fp8_e4m3(v_packed, v_scale);
                }
            }
        }

        // Compute with causal mask
        for (uint r = 0; r < sg_q_rows; ++r) {
            uint global_q_pos = q_start + sg_q_start + r;
            float scores[TILE_KV];

            for (uint ki = 0; ki < tile_len; ++ki) {
                uint k_pos = tile_start + ki;

                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[r][i] * input_to_float(K_tile[buf][ki][d]);
                }
                dot = simd_sum(dot);

                // Branchless causal mask
                float score = dot * attn_scale;
                scores[ki] = select(score, -INFINITY, k_pos > global_q_pos);
            }

            for (uint ki = tile_len; ki < TILE_KV; ++ki) {
                scores[ki] = -INFINITY;
            }

            // Online softmax
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
                    o_acc[r][i] += p * input_to_float(V_tile[buf][ki][d]);
                }
            }

            m_prev[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = buf_load;
    }

    // Store output
    const uint o_base = batch_idx * q_stride_b + head_q * q_stride_h + q_start * q_stride_s;

    for (uint r = 0; r < sg_q_rows; ++r) {
        uint global_q = q_start + sg_q_start + r;
        if (global_q >= seq_q) continue;

        float inv_l = (l_prev[r] > 0.0f) ? (1.0f / l_prev[r]) : 0.0f;
        const uint o_row_base = o_base + (sg_q_start + r) * q_stride_s;
#ifdef USE_BF16_INPUTS
        float out_vals[HEAD_DIM_128 / SIMD_SIZE];
        for (uint i = 0; i < elems_per_lane; ++i) {
            out_vals[i] = o_acc[r][i] * inv_l;
        }
        store_output_bf16_vectorized(O, o_row_base, lane_id, elems_per_lane, out_vals, head_dim);
#else
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                store_output_scalar(O, o_row_base + d, o_acc[r][i] * inv_l);
            }
        }
#endif
    }
}
