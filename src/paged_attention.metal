// paged_attention.metal - Paged Attention for Apple Silicon
//
// Implements vLLM-style paged attention adapted for Metal simdgroup architecture.
// Paged attention decouples logical token positions from physical memory layout,
// enabling efficient batch serving with variable-length sequences.
//
// Key differences from flash_attention.metal:
//   - KV cache is organized in fixed-size blocks (pages)
//   - Block tables map logical block indices to physical block addresses
//   - Each sequence can have a different context length
//   - Designed for decode phase: Q has exactly 1 token per sequence
//   - Supports batch serving (multiple sequences with different lengths)
//
// Kernel variants:
//   1. paged_attention_v1              - Single-pass (short-medium contexts)
//   2. paged_attention_v2              - Multi-partition (long contexts, >1024 tokens)
//   3. paged_attention_v1_fp4          - FP4-quantized KV blocks
//   4. paged_attention_v1_int4         - INT4-quantized KV blocks
//
// KV block layout (physical):
//   [num_blocks, num_kv_heads, block_size, head_dim]
//
// Block table:
//   [num_seqs, max_blocks_per_seq]  - maps (seq, logical_block) -> physical_block
//
// Dispatch:
//   paged_attention_v1: [num_seqs, num_heads_q, 1] threadgroups, 128 threads/tg
//   paged_attention_v2: [num_seqs, num_heads_q, num_partitions] threadgroups
//
// CUDA -> Metal mapping:
//   vLLM uses 1 warp per head (32 threads), multiple warps for parallel reduce.
//   Metal uses 1 simdgroup per head, 4 simdgroups per threadgroup for occupancy.
//   Since decode Q is always 1 token, we assign 1 simdgroup per (seq, head) pair.
//   The other 3 simdgroups in the threadgroup cooperate on KV block loading.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Tokens per KV block. Must match the page size used by the Python scheduler.
// 16 is the standard vLLM block size; powers of 2 enable shift-based indexing.
constant constexpr uint BLOCK_SIZE = 16;

// Maximum supported head dimension
constant constexpr uint HEAD_DIM_MAX = 128;

// Threads per simdgroup (Apple Silicon fixed at 32)
constant constexpr uint SIMD_SIZE = 32;

// Simdgroups per threadgroup. 4 gives 128 threads total.
// All 4 simdgroups cooperate on loading KV blocks; simdgroup 0 does the
// attention computation. This hides load latency behind compute.
constant constexpr uint NUM_SIMDGROUPS = 4;
constant constexpr uint THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;  // 128

// Number of KV blocks loaded into threadgroup memory simultaneously.
// Double-buffering: compute on one while loading the next.
constant constexpr uint KV_TILES = 2;

// For v2 partitioning: maximum tokens processed per partition
constant constexpr uint PARTITION_SIZE = 256;

// FP4 packing
constant constexpr uint FP4_PER_UINT = 8;

// FP8 packing
constant constexpr uint FP8_PER_UINT = 4;

// ---------------------------------------------------------------------------
// Utility functions using hardware-accelerated simd reductions
//
// Metal's built-in simd_sum/simd_max are significantly faster than manual
// simd_shuffle_xor chains: single instruction vs 5 dependent instructions.
// ---------------------------------------------------------------------------

inline float simd_reduce_sum(float val) {
    return simd_sum(val);
}

inline float simd_reduce_max(float val) {
    return simd_max(val);
}

// ---------------------------------------------------------------------------
// FP4/INT4 dequantization helpers (same as flash_attention.metal)
// ---------------------------------------------------------------------------

inline half dequant_fp4(uint nibble, half scale) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.5h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }

    half result = sign_bit ? -magnitude : magnitude;
    return result * scale;
}

inline half dequant_int4(uint nibble, half scale) {
    int signed_val = int(nibble & 0xFu) - 8;
    return half(signed_val) * scale;
}

// ---------------------------------------------------------------------------
// FP8 E4M3 dequantization for paged KV cache
// ---------------------------------------------------------------------------
//
// FP8 E4M3 format: [1 sign][4 exponent (bias=7)][3 mantissa]
//
// Value encoding:
//   Normal (0 < E < 15):  (-1)^S * 2^(E-7) * (1 + M/8)
//   Subnormal (E == 0):   (-1)^S * 2^(-6) * (M/8)
//   NaN (E == 15, M == 7): Mapped to max value (448.0)
//   Zero (E=0, M=0):      +/- 0.0

/// Dequantize a single FP8 E4M3 value to half precision with scale.
inline half dequant_fp8_e4m3(uint8_t val, half scale) {
    uint sign = (val >> 7) & 1u;
    uint exp = (val >> 3) & 0xFu;
    uint man = val & 0x7u;

    // Handle NaN: E=15, M=7 -> max value 448.0
    bool is_nan = (exp == 15u) && (man == 7u);

    half magnitude;
    if (exp == 0u) {
        // Subnormal: value = M * 2^(-9)
        magnitude = half(man) * half(0.001953125h);  // 2^-9
    } else {
        // Normal: value = (1 + M/8) * 2^(E-7)
        half mantissa = half(1.0h) + half(man) * half(0.125h);
        if (exp >= 7u) {
            magnitude = mantissa * half(float(1u << (exp - 7u)));
        } else {
            magnitude = mantissa / half(float(1u << (7u - exp)));
        }
    }

    half result = select(magnitude, -magnitude, bool(sign));
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
// Paged Attention V1 - Single-pass decode attention
//
// For each sequence in the batch, computes:
//   output[seq][head] = softmax(Q[seq][head] @ K_cache^T / sqrt(d)) @ V_cache
//
// where K_cache and V_cache are scattered across physical blocks according
// to the block table.
//
// Dispatch: [num_seqs, num_heads_q, 1] threadgroups
//           THREADS_PER_TG (128) threads per threadgroup
//
// Each threadgroup handles one (sequence, Q-head) pair.
// The Q vector is loaded into registers distributed across simdgroup 0.
// All 4 simdgroups cooperate to load KV blocks into threadgroup memory.
// Simdgroup 0 computes the dot products and online softmax.
// ---------------------------------------------------------------------------

kernel void paged_attention_v1(
    device const half* Q                [[buffer(0)]],   // [num_seqs, num_heads_q, head_dim]
    device const half* K_cache          [[buffer(1)]],   // [num_blocks, num_kv_heads, block_size, head_dim]
    device const half* V_cache          [[buffer(2)]],   // [num_blocks, num_kv_heads, block_size, head_dim]
    device const int* block_tables      [[buffer(3)]],   // [num_seqs, max_blocks_per_seq]
    device const int* context_lens      [[buffer(4)]],   // [num_seqs]
    device half* output                 [[buffer(5)]],   // [num_seqs, num_heads_q, head_dim]
    constant uint& num_seqs             [[buffer(6)]],
    constant uint& num_heads_q          [[buffer(7)]],
    constant uint& num_kv_heads         [[buffer(8)]],
    constant uint& head_dim             [[buffer(9)]],
    constant uint& max_blocks_per_seq   [[buffer(10)]],
    constant float& scale               [[buffer(11)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                      [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint seq_idx = tgid.x;
    const uint head_q = tgid.y;

    if (seq_idx >= num_seqs) return;

    // GQA: map Q head to KV head
    const uint gqa_ratio = num_heads_q / num_kv_heads;
    const uint head_kv = head_q / gqa_ratio;

    // Context length for this sequence
    const int ctx_len = context_lens[seq_idx];
    if (ctx_len <= 0) return;

    const uint context_len = uint(ctx_len);
    const uint num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Q layout: [num_seqs, num_heads_q, head_dim]
    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;

    // Load Q vector into registers (distributed across lanes of simdgroup 0)
    // Each lane holds head_dim/32 consecutive elements.
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_MAX / SIMD_SIZE];
    if (sg_id == 0) {
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
        }
    }

    // Threadgroup memory for one KV block at a time (double-buffered)
    // Each block has BLOCK_SIZE tokens, each with head_dim elements.
    // Memory: 2 * BLOCK_SIZE * HEAD_DIM_MAX * sizeof(half) = 2 * 16 * 128 * 2 = 8 KB
    // Well within the 32 KB threadgroup memory budget.
    threadgroup half K_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];
    threadgroup half V_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];

    // Online softmax state (simdgroup 0 only, but all lanes participate)
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    // KV block stride: [num_blocks, num_kv_heads, block_size, head_dim]
    const uint kv_head_stride = BLOCK_SIZE * head_dim;
    const uint kv_block_stride = num_kv_heads * kv_head_stride;

    // Block table for this sequence
    device const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Preload first block into buffer 0
    {
        int phys_block = seq_block_table[0];
        uint kv_base = uint(phys_block) * kv_block_stride + head_kv * kv_head_stride;
        uint elems_to_load = BLOCK_SIZE * head_dim;
        uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint token_in_block = idx / head_dim;
                uint d = idx % head_dim;
                K_smem[0][token_in_block][d] = K_cache[kv_base + token_in_block * head_dim + d];
                V_smem[0][token_in_block][d] = V_cache[kv_base + token_in_block * head_dim + d];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------------------------------------------------------------------
    // Main loop: iterate over KV blocks with double-buffering
    // ---------------------------------------------------------------------------
    uint buf_compute = 0;

    for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint buf_load = 1 - buf_compute;

        // Async load next block (all threads participate)
        if (block_idx + 1 < num_blocks) {
            int next_phys_block = seq_block_table[block_idx + 1];
            uint kv_base = uint(next_phys_block) * kv_block_stride + head_kv * kv_head_stride;
            uint elems_to_load = BLOCK_SIZE * head_dim;
            uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint token_in_block = idx / head_dim;
                    uint d = idx % head_dim;
                    K_smem[buf_load][token_in_block][d] = K_cache[kv_base + token_in_block * head_dim + d];
                    V_smem[buf_load][token_in_block][d] = V_cache[kv_base + token_in_block * head_dim + d];
                }
            }
        }

        // Compute attention for current block (simdgroup 0)
        // Determine valid tokens in this block
        uint block_start_token = block_idx * BLOCK_SIZE;
        uint block_tokens = min(uint(BLOCK_SIZE), context_len - block_start_token);

        if (sg_id == 0) {
            // Compute QK^T for each valid token in this block
            float scores[BLOCK_SIZE];
            for (uint t = 0; t < block_tokens; ++t) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf_compute][t][d]);
                }
                dot = simd_reduce_sum(dot);
                scores[t] = dot * scale;
            }
            // Mask invalid positions
            for (uint t = block_tokens; t < BLOCK_SIZE; ++t) {
                scores[t] = -INFINITY;
            }

            // Online softmax update for this block
            float m_block = -INFINITY;
            for (uint t = 0; t < block_tokens; ++t) {
                m_block = max(m_block, scores[t]);
            }

            float m_new = max(m_prev, m_block);
            float correction = exp(m_prev - m_new);

            // Rescale running sum and add new exponentials
            float l_new = l_prev * correction;
            for (uint t = 0; t < block_tokens; ++t) {
                l_new += exp(scores[t] - m_new);
            }

            // Rescale output accumulator
            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= correction;
            }

            // Accumulate weighted V
            for (uint t = 0; t < block_tokens; ++t) {
                float p = exp(scores[t] - m_new);
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf_compute][t][d]);
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }

        // Barrier before swapping buffers (all threads must finish loading)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // ---------------------------------------------------------------------------
    // Store output (simdgroup 0 only)
    // ---------------------------------------------------------------------------
    if (sg_id == 0) {
        const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                output[o_offset + d] = half(o_acc[i] * inv_l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Paged Attention V2 - Multi-partition for long contexts
//
// For sequences with many KV blocks, v1 serializes over all blocks in one
// threadgroup, limiting parallelism. V2 partitions the blocks across multiple
// threadgroups along the z-axis, each computing a partial softmax result.
// A final reduction combines partitions using the log-sum-exp trick.
//
// Phase 1 (this kernel): Each partition computes:
//   - partial output accumulator (unnormalized)
//   - partial max logit (m)
//   - partial sum of exponentials (l)
//
// Phase 2 (paged_attention_v2_reduce): Combines partitions.
//
// Dispatch: [num_seqs, num_heads_q, num_partitions] threadgroups
// ---------------------------------------------------------------------------

kernel void paged_attention_v2(
    device const half* Q                [[buffer(0)]],
    device const half* K_cache          [[buffer(1)]],
    device const half* V_cache          [[buffer(2)]],
    device const int* block_tables      [[buffer(3)]],
    device const int* context_lens      [[buffer(4)]],
    device float* partial_out           [[buffer(5)]],   // [num_seqs, num_heads_q, max_partitions, head_dim]
    device float* partial_m             [[buffer(6)]],   // [num_seqs, num_heads_q, max_partitions]
    device float* partial_l             [[buffer(7)]],   // [num_seqs, num_heads_q, max_partitions]
    constant uint& num_seqs             [[buffer(8)]],
    constant uint& num_heads_q          [[buffer(9)]],
    constant uint& num_kv_heads         [[buffer(10)]],
    constant uint& head_dim             [[buffer(11)]],
    constant uint& max_blocks_per_seq   [[buffer(12)]],
    constant uint& max_partitions       [[buffer(13)]],
    constant float& scale               [[buffer(14)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                      [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint seq_idx = tgid.x;
    const uint head_q = tgid.y;
    const uint partition_idx = tgid.z;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_kv_heads;
    const uint head_kv = head_q / gqa_ratio;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len <= 0) return;
    const uint context_len = uint(ctx_len);

    // Determine which blocks this partition handles
    // Each partition handles PARTITION_SIZE / BLOCK_SIZE blocks
    const uint blocks_per_partition = PARTITION_SIZE / BLOCK_SIZE;  // 16
    const uint total_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const uint partition_start_block = partition_idx * blocks_per_partition;
    if (partition_start_block >= total_blocks) return;  // This partition has no work

    const uint partition_end_block = min(partition_start_block + blocks_per_partition, total_blocks);
    const uint partition_num_blocks = partition_end_block - partition_start_block;

    // Load Q
    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_MAX / SIMD_SIZE];
    if (sg_id == 0) {
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
        }
    }

    threadgroup half K_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];
    threadgroup half V_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    const uint kv_head_stride = BLOCK_SIZE * head_dim;
    const uint kv_block_stride = num_kv_heads * kv_head_stride;
    device const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Preload first block of this partition
    {
        uint blk = partition_start_block;
        int phys_block = seq_block_table[blk];
        uint kv_base = uint(phys_block) * kv_block_stride + head_kv * kv_head_stride;
        uint elems_to_load = BLOCK_SIZE * head_dim;
        uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint token_in_block = idx / head_dim;
                uint d = idx % head_dim;
                K_smem[0][token_in_block][d] = K_cache[kv_base + token_in_block * head_dim + d];
                V_smem[0][token_in_block][d] = V_cache[kv_base + token_in_block * head_dim + d];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint blk_offset = 0; blk_offset < partition_num_blocks; ++blk_offset) {
        uint buf_load = 1 - buf_compute;
        uint abs_block = partition_start_block + blk_offset;

        // Load next block
        if (blk_offset + 1 < partition_num_blocks) {
            uint next_abs_block = partition_start_block + blk_offset + 1;
            int next_phys_block = seq_block_table[next_abs_block];
            uint kv_base = uint(next_phys_block) * kv_block_stride + head_kv * kv_head_stride;
            uint elems_to_load = BLOCK_SIZE * head_dim;
            uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint token_in_block = idx / head_dim;
                    uint d = idx % head_dim;
                    K_smem[buf_load][token_in_block][d] = K_cache[kv_base + token_in_block * head_dim + d];
                    V_smem[buf_load][token_in_block][d] = V_cache[kv_base + token_in_block * head_dim + d];
                }
            }
        }

        // Compute on current block
        uint block_start_token = abs_block * BLOCK_SIZE;
        uint block_tokens = min(uint(BLOCK_SIZE), context_len - block_start_token);

        if (sg_id == 0) {
            float scores[BLOCK_SIZE];
            for (uint t = 0; t < block_tokens; ++t) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf_compute][t][d]);
                }
                dot = simd_reduce_sum(dot);
                scores[t] = dot * scale;
            }
            for (uint t = block_tokens; t < BLOCK_SIZE; ++t) {
                scores[t] = -INFINITY;
            }

            float m_block = -INFINITY;
            for (uint t = 0; t < block_tokens; ++t) {
                m_block = max(m_block, scores[t]);
            }

            float m_new = max(m_prev, m_block);
            float correction = exp(m_prev - m_new);
            float l_new = l_prev * correction;
            for (uint t = 0; t < block_tokens; ++t) {
                l_new += exp(scores[t] - m_new);
            }

            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= correction;
            }
            for (uint t = 0; t < block_tokens; ++t) {
                float p = exp(scores[t] - m_new);
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf_compute][t][d]);
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store partial results (simdgroup 0)
    if (sg_id == 0) {
        // partial_out: [num_seqs, num_heads_q, max_partitions, head_dim]
        const uint po_offset = ((seq_idx * num_heads_q + head_q) * max_partitions + partition_idx) * head_dim;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                partial_out[po_offset + d] = o_acc[i];
            }
        }

        // partial_m/l: [num_seqs, num_heads_q, max_partitions]
        if (lane_id == 0) {
            const uint pm_offset = (seq_idx * num_heads_q + head_q) * max_partitions + partition_idx;
            partial_m[pm_offset] = m_prev;
            partial_l[pm_offset] = l_prev;
        }
    }
}

// ---------------------------------------------------------------------------
// Paged Attention V2 - Reduction kernel
//
// Combines partial results from paged_attention_v2 using the log-sum-exp trick:
//   m_global = max(m_0, m_1, ..., m_P)
//   l_global = sum_p(l_p * exp(m_p - m_global))
//   O = sum_p(O_p * exp(m_p - m_global)) / l_global
//
// Dispatch: [num_seqs, num_heads_q, 1] threadgroups, SIMD_SIZE threads
// ---------------------------------------------------------------------------

kernel void paged_attention_v2_reduce(
    device const float* partial_out     [[buffer(0)]],   // [num_seqs, num_heads_q, max_partitions, head_dim]
    device const float* partial_m       [[buffer(1)]],   // [num_seqs, num_heads_q, max_partitions]
    device const float* partial_l       [[buffer(2)]],   // [num_seqs, num_heads_q, max_partitions]
    device const int* context_lens      [[buffer(3)]],   // [num_seqs]
    device half* output                 [[buffer(4)]],   // [num_seqs, num_heads_q, head_dim]
    constant uint& num_seqs             [[buffer(5)]],
    constant uint& num_heads_q          [[buffer(6)]],
    constant uint& head_dim             [[buffer(7)]],
    constant uint& max_partitions       [[buffer(8)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                      [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]]
) {
    const uint seq_idx = tgid.x;
    const uint head_q = tgid.y;

    if (seq_idx >= num_seqs) return;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len <= 0) return;
    const uint context_len = uint(ctx_len);

    // How many partitions were actually used for this sequence?
    const uint total_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint blocks_per_partition = PARTITION_SIZE / BLOCK_SIZE;
    const uint num_parts = (total_blocks + blocks_per_partition - 1) / blocks_per_partition;

    // If only 1 partition, the v2 kernel result is already final (just copy + normalize)
    const uint pm_base = (seq_idx * num_heads_q + head_q) * max_partitions;
    const uint po_base = ((seq_idx * num_heads_q + head_q) * max_partitions) * head_dim;

    // Find global max across partitions
    float m_global = -INFINITY;
    for (uint p = 0; p < num_parts; ++p) {
        float mp = partial_m[pm_base + p];
        m_global = max(m_global, mp);
    }

    // Compute global l and accumulate output
    float l_global = 0.0f;
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float o_final[HEAD_DIM_MAX / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_final[i] = 0.0f;
    }

    for (uint p = 0; p < num_parts; ++p) {
        float mp = partial_m[pm_base + p];
        float lp = partial_l[pm_base + p];
        float w = exp(mp - m_global) * lp;
        l_global += w;

        // Weight this partition's output contribution
        float scale_p = exp(mp - m_global);
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                o_final[i] += partial_out[po_base + p * head_dim + d] * scale_p;
            }
        }
    }

    // Normalize and store
    const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < head_dim) {
            output[o_offset + d] = half(o_final[i] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Paged Attention V1 - FP4 quantized KV cache
//
// KV blocks stored as packed FP4 E2M1 with per-token-per-head scales.
//
// K_cache_packed: [num_blocks, num_kv_heads, block_size, head_dim/8]  (uint32)
// K_scales:       [num_blocks, num_kv_heads, block_size]              (half)
// ---------------------------------------------------------------------------

kernel void paged_attention_v1_fp4(
    device const half* Q                [[buffer(0)]],
    device const uint* K_cache_packed   [[buffer(1)]],
    device const uint* V_cache_packed   [[buffer(2)]],
    device const half* K_scales         [[buffer(3)]],
    device const half* V_scales         [[buffer(4)]],
    device const int* block_tables      [[buffer(5)]],
    device const int* context_lens      [[buffer(6)]],
    device half* output                 [[buffer(7)]],
    constant uint& num_seqs             [[buffer(8)]],
    constant uint& num_heads_q          [[buffer(9)]],
    constant uint& num_kv_heads         [[buffer(10)]],
    constant uint& head_dim             [[buffer(11)]],
    constant uint& max_blocks_per_seq   [[buffer(12)]],
    constant float& scale               [[buffer(13)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                      [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint seq_idx = tgid.x;
    const uint head_q = tgid.y;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_kv_heads;
    const uint head_kv = head_q / gqa_ratio;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len <= 0) return;
    const uint context_len = uint(ctx_len);
    const uint num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_MAX / SIMD_SIZE];
    if (sg_id == 0) {
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
        }
    }

    // Packed dimensions
    const uint packed_head_dim = head_dim / FP4_PER_UINT;

    // KV packed layout: [num_blocks, num_kv_heads, block_size, packed_head_dim]
    const uint kv_packed_head_stride = BLOCK_SIZE * packed_head_dim;
    const uint kv_packed_block_stride = num_kv_heads * kv_packed_head_stride;

    // Scale layout: [num_blocks, num_kv_heads, block_size]
    const uint scale_head_stride = BLOCK_SIZE;
    const uint scale_block_stride = num_kv_heads * scale_head_stride;

    threadgroup half K_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];
    threadgroup half V_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    device const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Preload first block (dequantize FP4 -> FP16 into threadgroup memory)
    {
        int phys_block = seq_block_table[0];
        uint k_packed_base = uint(phys_block) * kv_packed_block_stride + head_kv * kv_packed_head_stride;
        uint k_scale_base = uint(phys_block) * scale_block_stride + head_kv * scale_head_stride;

        uint packed_elems = BLOCK_SIZE * packed_head_dim;
        uint loads_per_thread = (packed_elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        // Load and dequant K
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < packed_elems) {
                uint token_in_block = idx / packed_head_dim;
                uint packed_col = idx % packed_head_dim;
                half s = K_scales[k_scale_base + token_in_block];
                uint word = K_cache_packed[k_packed_base + token_in_block * packed_head_dim + packed_col];
                uint base_d = packed_col * FP4_PER_UINT;
                for (uint j = 0; j < FP4_PER_UINT && base_d + j < head_dim; ++j) {
                    uint nibble = (word >> (j * 4)) & 0xFu;
                    K_smem[0][token_in_block][base_d + j] = dequant_fp4(nibble, s);
                }
            }
        }
        // Load and dequant V
        uint v_packed_base = uint(phys_block) * kv_packed_block_stride + head_kv * kv_packed_head_stride;
        uint v_scale_base = uint(phys_block) * scale_block_stride + head_kv * scale_head_stride;
        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < packed_elems) {
                uint token_in_block = idx / packed_head_dim;
                uint packed_col = idx % packed_head_dim;
                half s = V_scales[v_scale_base + token_in_block];
                uint word = V_cache_packed[v_packed_base + token_in_block * packed_head_dim + packed_col];
                uint base_d = packed_col * FP4_PER_UINT;
                for (uint j = 0; j < FP4_PER_UINT && base_d + j < head_dim; ++j) {
                    uint nibble = (word >> (j * 4)) & 0xFu;
                    V_smem[0][token_in_block][base_d + j] = dequant_fp4(nibble, s);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint buf_load = 1 - buf_compute;

        // Load next block with FP4 dequant
        if (block_idx + 1 < num_blocks) {
            int next_phys_block = seq_block_table[block_idx + 1];
            uint k_packed_base = uint(next_phys_block) * kv_packed_block_stride + head_kv * kv_packed_head_stride;
            uint k_scale_base = uint(next_phys_block) * scale_block_stride + head_kv * scale_head_stride;
            uint v_packed_base = k_packed_base;  // Same layout for V
            uint v_scale_base = uint(next_phys_block) * scale_block_stride + head_kv * scale_head_stride;

            uint packed_elems = BLOCK_SIZE * packed_head_dim;
            uint loads_per_thread = (packed_elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < packed_elems) {
                    uint token_in_block = idx / packed_head_dim;
                    uint packed_col = idx % packed_head_dim;

                    half ks = K_scales[k_scale_base + token_in_block];
                    uint k_word = K_cache_packed[k_packed_base + token_in_block * packed_head_dim + packed_col];
                    half vs = V_scales[v_scale_base + token_in_block];
                    uint v_word = V_cache_packed[v_packed_base + token_in_block * packed_head_dim + packed_col];

                    uint base_d = packed_col * FP4_PER_UINT;
                    for (uint j = 0; j < FP4_PER_UINT && base_d + j < head_dim; ++j) {
                        uint k_nibble = (k_word >> (j * 4)) & 0xFu;
                        uint v_nibble = (v_word >> (j * 4)) & 0xFu;
                        K_smem[buf_load][token_in_block][base_d + j] = dequant_fp4(k_nibble, ks);
                        V_smem[buf_load][token_in_block][base_d + j] = dequant_fp4(v_nibble, vs);
                    }
                }
            }
        }

        uint block_start_token = block_idx * BLOCK_SIZE;
        uint block_tokens = min(uint(BLOCK_SIZE), context_len - block_start_token);

        if (sg_id == 0) {
            float scores[BLOCK_SIZE];
            for (uint t = 0; t < block_tokens; ++t) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf_compute][t][d]);
                }
                dot = simd_reduce_sum(dot);
                scores[t] = dot * scale;
            }
            for (uint t = block_tokens; t < BLOCK_SIZE; ++t) {
                scores[t] = -INFINITY;
            }

            float m_block = -INFINITY;
            for (uint t = 0; t < block_tokens; ++t) {
                m_block = max(m_block, scores[t]);
            }

            float m_new = max(m_prev, m_block);
            float correction = exp(m_prev - m_new);
            float l_new = l_prev * correction;
            for (uint t = 0; t < block_tokens; ++t) {
                l_new += exp(scores[t] - m_new);
            }

            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= correction;
            }
            for (uint t = 0; t < block_tokens; ++t) {
                float p = exp(scores[t] - m_new);
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf_compute][t][d]);
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    if (sg_id == 0) {
        const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                output[o_offset + d] = half(o_acc[i] * inv_l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Paged Attention V1 - INT4 quantized KV cache
//
// Same structure as FP4 variant but uses signed 4-bit integer dequantization.
// ---------------------------------------------------------------------------

kernel void paged_attention_v1_int4(
    device const half* Q                [[buffer(0)]],
    device const uint* K_cache_packed   [[buffer(1)]],
    device const uint* V_cache_packed   [[buffer(2)]],
    device const half* K_scales         [[buffer(3)]],
    device const half* V_scales         [[buffer(4)]],
    device const int* block_tables      [[buffer(5)]],
    device const int* context_lens      [[buffer(6)]],
    device half* output                 [[buffer(7)]],
    constant uint& num_seqs             [[buffer(8)]],
    constant uint& num_heads_q          [[buffer(9)]],
    constant uint& num_kv_heads         [[buffer(10)]],
    constant uint& head_dim             [[buffer(11)]],
    constant uint& max_blocks_per_seq   [[buffer(12)]],
    constant float& scale               [[buffer(13)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                      [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint seq_idx = tgid.x;
    const uint head_q = tgid.y;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_kv_heads;
    const uint head_kv = head_q / gqa_ratio;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len <= 0) return;
    const uint context_len = uint(ctx_len);
    const uint num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_MAX / SIMD_SIZE];
    if (sg_id == 0) {
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
        }
    }

    const uint packed_head_dim = head_dim / FP4_PER_UINT;
    const uint kv_packed_head_stride = BLOCK_SIZE * packed_head_dim;
    const uint kv_packed_block_stride = num_kv_heads * kv_packed_head_stride;
    const uint scale_head_stride = BLOCK_SIZE;
    const uint scale_block_stride = num_kv_heads * scale_head_stride;

    threadgroup half K_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];
    threadgroup half V_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    device const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Preload first block with INT4 dequant
    {
        int phys_block = seq_block_table[0];
        uint k_packed_base = uint(phys_block) * kv_packed_block_stride + head_kv * kv_packed_head_stride;
        uint k_scale_base = uint(phys_block) * scale_block_stride + head_kv * scale_head_stride;
        uint v_packed_base = k_packed_base;
        uint v_scale_base = k_scale_base;

        uint packed_elems = BLOCK_SIZE * packed_head_dim;
        uint loads_per_thread = (packed_elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < packed_elems) {
                uint token_in_block = idx / packed_head_dim;
                uint packed_col = idx % packed_head_dim;

                half ks = K_scales[k_scale_base + token_in_block];
                uint k_word = K_cache_packed[k_packed_base + token_in_block * packed_head_dim + packed_col];
                half vs = V_scales[v_scale_base + token_in_block];
                uint v_word = V_cache_packed[v_packed_base + token_in_block * packed_head_dim + packed_col];

                uint base_d = packed_col * FP4_PER_UINT;
                for (uint j = 0; j < FP4_PER_UINT && base_d + j < head_dim; ++j) {
                    uint k_nibble = (k_word >> (j * 4)) & 0xFu;
                    uint v_nibble = (v_word >> (j * 4)) & 0xFu;
                    K_smem[0][token_in_block][base_d + j] = dequant_int4(k_nibble, ks);
                    V_smem[0][token_in_block][base_d + j] = dequant_int4(v_nibble, vs);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint buf_load = 1 - buf_compute;

        if (block_idx + 1 < num_blocks) {
            int next_phys_block = seq_block_table[block_idx + 1];
            uint k_packed_base = uint(next_phys_block) * kv_packed_block_stride + head_kv * kv_packed_head_stride;
            uint k_scale_base = uint(next_phys_block) * scale_block_stride + head_kv * scale_head_stride;
            uint v_packed_base = k_packed_base;
            uint v_scale_base = k_scale_base;

            uint packed_elems = BLOCK_SIZE * packed_head_dim;
            uint loads_per_thread = (packed_elems + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < packed_elems) {
                    uint token_in_block = idx / packed_head_dim;
                    uint packed_col = idx % packed_head_dim;

                    half ks = K_scales[k_scale_base + token_in_block];
                    uint k_word = K_cache_packed[k_packed_base + token_in_block * packed_head_dim + packed_col];
                    half vs = V_scales[v_scale_base + token_in_block];
                    uint v_word = V_cache_packed[v_packed_base + token_in_block * packed_head_dim + packed_col];

                    uint base_d = packed_col * FP4_PER_UINT;
                    for (uint j = 0; j < FP4_PER_UINT && base_d + j < head_dim; ++j) {
                        uint k_nibble = (k_word >> (j * 4)) & 0xFu;
                        uint v_nibble = (v_word >> (j * 4)) & 0xFu;
                        K_smem[buf_load][token_in_block][base_d + j] = dequant_int4(k_nibble, ks);
                        V_smem[buf_load][token_in_block][base_d + j] = dequant_int4(v_nibble, vs);
                    }
                }
            }
        }

        uint block_start_token = block_idx * BLOCK_SIZE;
        uint block_tokens = min(uint(BLOCK_SIZE), context_len - block_start_token);

        if (sg_id == 0) {
            float scores[BLOCK_SIZE];
            for (uint t = 0; t < block_tokens; ++t) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf_compute][t][d]);
                }
                dot = simd_reduce_sum(dot);
                scores[t] = dot * scale;
            }
            for (uint t = block_tokens; t < BLOCK_SIZE; ++t) {
                scores[t] = -INFINITY;
            }

            float m_block = -INFINITY;
            for (uint t = 0; t < block_tokens; ++t) {
                m_block = max(m_block, scores[t]);
            }

            float m_new = max(m_prev, m_block);
            float correction = exp(m_prev - m_new);
            float l_new = l_prev * correction;
            for (uint t = 0; t < block_tokens; ++t) {
                l_new += exp(scores[t] - m_new);
            }

            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= correction;
            }
            for (uint t = 0; t < block_tokens; ++t) {
                float p = exp(scores[t] - m_new);
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf_compute][t][d]);
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    if (sg_id == 0) {
        const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                output[o_offset + d] = half(o_acc[i] * inv_l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Paged Attention V1 - FP8 E4M3 quantized KV cache
//
// KV blocks stored as packed FP8 E4M3 (8 bits per element) with per-token-per-head scales.
// FP8 provides 2x memory reduction compared to FP16 with minimal accuracy loss.
//
// K_cache_fp8:   [num_blocks, num_kv_heads, block_size, head_dim]  (uint8_t)
// V_cache_fp8:   [num_blocks, num_kv_heads, block_size, head_dim]  (uint8_t)
// K_scales:      [num_blocks, num_kv_heads, block_size]            (half)
// V_scales:      [num_blocks, num_kv_heads, block_size]            (half)
// ---------------------------------------------------------------------------

kernel void paged_attention_v1_fp8(
    device const half* Q                [[buffer(0)]],
    device const uint8_t* K_cache_fp8   [[buffer(1)]],
    device const uint8_t* V_cache_fp8   [[buffer(2)]],
    device const half* K_scales         [[buffer(3)]],
    device const half* V_scales         [[buffer(4)]],
    device const int* block_tables      [[buffer(5)]],
    device const int* context_lens      [[buffer(6)]],
    device half* output                 [[buffer(7)]],
    constant uint& num_seqs             [[buffer(8)]],
    constant uint& num_heads_q          [[buffer(9)]],
    constant uint& num_kv_heads         [[buffer(10)]],
    constant uint& head_dim             [[buffer(11)]],
    constant uint& max_blocks_per_seq   [[buffer(12)]],
    constant float& scale               [[buffer(13)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                      [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint seq_idx = tgid.x;
    const uint head_q = tgid.y;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_kv_heads;
    const uint head_kv = head_q / gqa_ratio;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len <= 0) return;
    const uint context_len = uint(ctx_len);
    const uint num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_MAX / SIMD_SIZE];
    if (sg_id == 0) {
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
        }
    }

    // FP8 KV cache layout: [num_blocks, num_kv_heads, block_size, head_dim]
    const uint kv_head_stride = BLOCK_SIZE * head_dim;
    const uint kv_block_stride = num_kv_heads * kv_head_stride;

    // Scale layout: [num_blocks, num_kv_heads, block_size]
    const uint scale_head_stride = BLOCK_SIZE;
    const uint scale_block_stride = num_kv_heads * scale_head_stride;

    threadgroup half K_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];
    threadgroup half V_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    device const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Preload first block (dequantize FP8 -> FP16 into threadgroup memory)
    {
        int phys_block = seq_block_table[0];
        uint kv_base = uint(phys_block) * kv_block_stride + head_kv * kv_head_stride;
        uint scale_base = uint(phys_block) * scale_block_stride + head_kv * scale_head_stride;

        uint elems_to_load = BLOCK_SIZE * head_dim;
        uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint token_in_block = idx / head_dim;
                uint d = idx % head_dim;

                half k_scale = K_scales[scale_base + token_in_block];
                half v_scale = V_scales[scale_base + token_in_block];

                uint8_t k_val = K_cache_fp8[kv_base + token_in_block * head_dim + d];
                uint8_t v_val = V_cache_fp8[kv_base + token_in_block * head_dim + d];

                K_smem[0][token_in_block][d] = dequant_fp8_e4m3(k_val, k_scale);
                V_smem[0][token_in_block][d] = dequant_fp8_e4m3(v_val, v_scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint buf_load = 1 - buf_compute;

        // Load next block with FP8 dequant
        if (block_idx + 1 < num_blocks) {
            int next_phys_block = seq_block_table[block_idx + 1];
            uint kv_base = uint(next_phys_block) * kv_block_stride + head_kv * kv_head_stride;
            uint scale_base = uint(next_phys_block) * scale_block_stride + head_kv * scale_head_stride;

            uint elems_to_load = BLOCK_SIZE * head_dim;
            uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint token_in_block = idx / head_dim;
                    uint d = idx % head_dim;

                    half k_scale = K_scales[scale_base + token_in_block];
                    half v_scale = V_scales[scale_base + token_in_block];

                    uint8_t k_val = K_cache_fp8[kv_base + token_in_block * head_dim + d];
                    uint8_t v_val = V_cache_fp8[kv_base + token_in_block * head_dim + d];

                    K_smem[buf_load][token_in_block][d] = dequant_fp8_e4m3(k_val, k_scale);
                    V_smem[buf_load][token_in_block][d] = dequant_fp8_e4m3(v_val, v_scale);
                }
            }
        }

        uint block_start_token = block_idx * BLOCK_SIZE;
        uint block_tokens = min(uint(BLOCK_SIZE), context_len - block_start_token);

        if (sg_id == 0) {
            float scores[BLOCK_SIZE];
            for (uint t = 0; t < block_tokens; ++t) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf_compute][t][d]);
                }
                dot = simd_reduce_sum(dot);
                scores[t] = dot * scale;
            }
            for (uint t = block_tokens; t < BLOCK_SIZE; ++t) {
                scores[t] = -INFINITY;
            }

            float m_block = -INFINITY;
            for (uint t = 0; t < block_tokens; ++t) {
                m_block = max(m_block, scores[t]);
            }

            float m_new = max(m_prev, m_block);
            float correction = exp(m_prev - m_new);
            float l_new = l_prev * correction;
            for (uint t = 0; t < block_tokens; ++t) {
                l_new += exp(scores[t] - m_new);
            }

            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= correction;
            }
            for (uint t = 0; t < block_tokens; ++t) {
                float p = exp(scores[t] - m_new);
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf_compute][t][d]);
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    if (sg_id == 0) {
        const uint o_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                output[o_offset + d] = half(o_acc[i] * inv_l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Paged Attention V2 - FP8 E4M3 quantized KV cache (Multi-partition)
//
// Long context version that partitions blocks across threadgroups.
// Uses the same FP8 dequantization approach as paged_attention_v1_fp8.
// ---------------------------------------------------------------------------

kernel void paged_attention_v2_fp8(
    device const half* Q                [[buffer(0)]],
    device const uint8_t* K_cache_fp8   [[buffer(1)]],
    device const uint8_t* V_cache_fp8   [[buffer(2)]],
    device const half* K_scales         [[buffer(3)]],
    device const half* V_scales         [[buffer(4)]],
    device const int* block_tables      [[buffer(5)]],
    device const int* context_lens      [[buffer(6)]],
    device float* partial_out           [[buffer(7)]],   // [num_seqs, num_heads_q, max_partitions, head_dim]
    device float* partial_m             [[buffer(8)]],   // [num_seqs, num_heads_q, max_partitions]
    device float* partial_l             [[buffer(9)]],   // [num_seqs, num_heads_q, max_partitions]
    constant uint& num_seqs             [[buffer(10)]],
    constant uint& num_heads_q          [[buffer(11)]],
    constant uint& num_kv_heads         [[buffer(12)]],
    constant uint& head_dim             [[buffer(13)]],
    constant uint& max_blocks_per_seq   [[buffer(14)]],
    constant uint& max_partitions       [[buffer(15)]],
    constant float& scale               [[buffer(16)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint tid_in_tg                      [[thread_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]]
) {
    const uint seq_idx = tgid.x;
    const uint head_q = tgid.y;
    const uint partition_idx = tgid.z;

    if (seq_idx >= num_seqs) return;

    const uint gqa_ratio = num_heads_q / num_kv_heads;
    const uint head_kv = head_q / gqa_ratio;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len <= 0) return;
    const uint context_len = uint(ctx_len);

    // Determine which blocks this partition handles
    const uint blocks_per_partition = PARTITION_SIZE / BLOCK_SIZE;
    const uint total_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const uint partition_start_block = partition_idx * blocks_per_partition;
    if (partition_start_block >= total_blocks) return;

    const uint partition_end_block = min(partition_start_block + blocks_per_partition, total_blocks);
    const uint partition_num_blocks = partition_end_block - partition_start_block;

    // Load Q
    const uint q_offset = seq_idx * num_heads_q * head_dim + head_q * head_dim;
    const uint elems_per_lane = head_dim / SIMD_SIZE;
    float q_reg[HEAD_DIM_MAX / SIMD_SIZE];
    if (sg_id == 0) {
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            q_reg[i] = (d < head_dim) ? float(Q[q_offset + d]) : 0.0f;
        }
    }

    // FP8 KV cache strides
    const uint kv_head_stride = BLOCK_SIZE * head_dim;
    const uint kv_block_stride = num_kv_heads * kv_head_stride;
    const uint scale_head_stride = BLOCK_SIZE;
    const uint scale_block_stride = num_kv_heads * scale_head_stride;

    threadgroup half K_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];
    threadgroup half V_smem[KV_TILES][BLOCK_SIZE][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / SIMD_SIZE];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    device const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Preload first block of this partition
    {
        uint blk = partition_start_block;
        int phys_block = seq_block_table[blk];
        uint kv_base = uint(phys_block) * kv_block_stride + head_kv * kv_head_stride;
        uint scale_base = uint(phys_block) * scale_block_stride + head_kv * scale_head_stride;

        uint elems_to_load = BLOCK_SIZE * head_dim;
        uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

        for (uint i = 0; i < loads_per_thread; ++i) {
            uint idx = tid_in_tg + i * THREADS_PER_TG;
            if (idx < elems_to_load) {
                uint token_in_block = idx / head_dim;
                uint d = idx % head_dim;

                half k_scale = K_scales[scale_base + token_in_block];
                half v_scale = V_scales[scale_base + token_in_block];

                uint8_t k_val = K_cache_fp8[kv_base + token_in_block * head_dim + d];
                uint8_t v_val = V_cache_fp8[kv_base + token_in_block * head_dim + d];

                K_smem[0][token_in_block][d] = dequant_fp8_e4m3(k_val, k_scale);
                V_smem[0][token_in_block][d] = dequant_fp8_e4m3(v_val, v_scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint blk_offset = 0; blk_offset < partition_num_blocks; ++blk_offset) {
        uint buf_load = 1 - buf_compute;
        uint abs_block = partition_start_block + blk_offset;

        // Load next block
        if (blk_offset + 1 < partition_num_blocks) {
            uint next_abs_block = partition_start_block + blk_offset + 1;
            int next_phys_block = seq_block_table[next_abs_block];
            uint kv_base = uint(next_phys_block) * kv_block_stride + head_kv * kv_head_stride;
            uint scale_base = uint(next_phys_block) * scale_block_stride + head_kv * scale_head_stride;

            uint elems_to_load = BLOCK_SIZE * head_dim;
            uint loads_per_thread = (elems_to_load + THREADS_PER_TG - 1) / THREADS_PER_TG;

            for (uint i = 0; i < loads_per_thread; ++i) {
                uint idx = tid_in_tg + i * THREADS_PER_TG;
                if (idx < elems_to_load) {
                    uint token_in_block = idx / head_dim;
                    uint d = idx % head_dim;

                    half k_scale = K_scales[scale_base + token_in_block];
                    half v_scale = V_scales[scale_base + token_in_block];

                    uint8_t k_val = K_cache_fp8[kv_base + token_in_block * head_dim + d];
                    uint8_t v_val = V_cache_fp8[kv_base + token_in_block * head_dim + d];

                    K_smem[buf_load][token_in_block][d] = dequant_fp8_e4m3(k_val, k_scale);
                    V_smem[buf_load][token_in_block][d] = dequant_fp8_e4m3(v_val, v_scale);
                }
            }
        }

        // Compute on current block
        uint block_start_token = abs_block * BLOCK_SIZE;
        uint block_tokens = min(uint(BLOCK_SIZE), context_len - block_start_token);

        if (sg_id == 0) {
            float scores[BLOCK_SIZE];
            for (uint t = 0; t < block_tokens; ++t) {
                float dot = 0.0f;
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    dot += q_reg[i] * float(K_smem[buf_compute][t][d]);
                }
                dot = simd_reduce_sum(dot);
                scores[t] = dot * scale;
            }
            for (uint t = block_tokens; t < BLOCK_SIZE; ++t) {
                scores[t] = -INFINITY;
            }

            float m_block = -INFINITY;
            for (uint t = 0; t < block_tokens; ++t) {
                m_block = max(m_block, scores[t]);
            }

            float m_new = max(m_prev, m_block);
            float correction = exp(m_prev - m_new);
            float l_new = l_prev * correction;
            for (uint t = 0; t < block_tokens; ++t) {
                l_new += exp(scores[t] - m_new);
            }

            for (uint i = 0; i < elems_per_lane; ++i) {
                o_acc[i] *= correction;
            }
            for (uint t = 0; t < block_tokens; ++t) {
                float p = exp(scores[t] - m_new);
                for (uint i = 0; i < elems_per_lane; ++i) {
                    uint d = lane_id * elems_per_lane + i;
                    o_acc[i] += p * float(V_smem[buf_compute][t][d]);
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store partial results (simdgroup 0)
    if (sg_id == 0) {
        const uint po_offset = ((seq_idx * num_heads_q + head_q) * max_partitions + partition_idx) * head_dim;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = lane_id * elems_per_lane + i;
            if (d < head_dim) {
                partial_out[po_offset + d] = o_acc[i];
            }
        }

        if (lane_id == 0) {
            const uint pm_offset = (seq_idx * num_heads_q + head_q) * max_partitions + partition_idx;
            partial_m[pm_offset] = m_prev;
            partial_l[pm_offset] = l_prev;
        }
    }
}
