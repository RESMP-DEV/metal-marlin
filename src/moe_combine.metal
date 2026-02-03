// moe_combine.metal - SIMD-vectorized MoE expert output combination
//
// Computes: result = Σ(weight_i * expert_i_output)
//
// This kernel processes multiple expert outputs in parallel using SIMD
// operations for efficient weighted summation. Designed for post-hoc
// combination when expert outputs are computed separately.
//
// Key optimizations:
//   - Process 4-8 experts per SIMD lane using FMA
//   - Vectorized loads (half4/half8) for expert outputs
//   - Coalesced memory access patterns
//   - Threadgroup-level reduction for multiple tokens
//
// Use cases:
//   - Separate expert GEMM + combine (vs. fused approach)
//   - Multi-GPU expert parallelism (combine after allgather)
//   - Debug/profiling (measure combine cost separately)

#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// Configuration
// ===========================================================================

constant constexpr uint COMBINE_THREADS = 256;      // Threads per threadgroup
constant constexpr uint COMBINE_TILE_N = 256;       // Output columns per threadgroup
constant constexpr uint MAX_EXPERTS_PARALLEL = 8;   // Max experts to process in parallel

// ===========================================================================
// Kernel: moe_combine_weighted
//
// Combines up to 8 expert outputs with probability weights using SIMD FMA.
// Each thread processes multiple output columns with vectorized loads.
//
// Grid: [ceil(hidden_dim / TILE_N), batch_size]
//
// Memory layout:
//   expert_outputs: [batch, top_k, hidden_dim] half - stored contiguously
//   weights:        [batch, top_k] half - routing probabilities (normalized)
//   output:         [batch, hidden_dim] half
// ===========================================================================

kernel void moe_combine_weighted(
    device const half* expert_outputs    [[buffer(0)]],  // [batch, top_k, hidden]
    device const half* weights           [[buffer(1)]],  // [batch, top_k]
    device half* output                  [[buffer(2)]],  // [batch, hidden]
    constant uint& batch_size            [[buffer(3)]],
    constant uint& hidden_dim            [[buffer(4)]],
    constant uint& top_k                 [[buffer(5)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * COMBINE_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Load weights for this token into registers (up to 8 experts)
    half w[MAX_EXPERTS_PARALLEL];
    for (uint k = 0; k < top_k && k < MAX_EXPERTS_PARALLEL; ++k) {
        w[k] = weights[token_idx * top_k + k];
    }
    for (uint k = top_k; k < MAX_EXPERTS_PARALLEL; ++k) {
        w[k] = 0.0h;
    }

    // Base pointers for expert outputs
    device const half* expert_base = expert_outputs + token_idx * top_k * hidden_dim;
    device half* out_base = output + token_idx * hidden_dim;

    // Process output columns, 4 at a time using SIMD
    for (uint col = n_block + thread_idx * 4; col < n_block + COMBINE_TILE_N && col + 3 < hidden_dim; col += COMBINE_THREADS * 4) {
        // Initialize accumulator with vectorized loads
        half4 acc = half4(0.0h);

        // Unrolled loop for up to 8 experts
        // Load 4 consecutive values from each expert and FMA
        if (top_k >= 1) {
            half4 e0 = *(device const half4*)(expert_base + 0 * hidden_dim + col);
            acc = fma(half4(w[0]), e0, acc);
        }
        if (top_k >= 2) {
            half4 e1 = *(device const half4*)(expert_base + 1 * hidden_dim + col);
            acc = fma(half4(w[1]), e1, acc);
        }
        if (top_k >= 3) {
            half4 e2 = *(device const half4*)(expert_base + 2 * hidden_dim + col);
            acc = fma(half4(w[2]), e2, acc);
        }
        if (top_k >= 4) {
            half4 e3 = *(device const half4*)(expert_base + 3 * hidden_dim + col);
            acc = fma(half4(w[3]), e3, acc);
        }
        if (top_k >= 5) {
            half4 e4 = *(device const half4*)(expert_base + 4 * hidden_dim + col);
            acc = fma(half4(w[4]), e4, acc);
        }
        if (top_k >= 6) {
            half4 e5 = *(device const half4*)(expert_base + 5 * hidden_dim + col);
            acc = fma(half4(w[5]), e5, acc);
        }
        if (top_k >= 7) {
            half4 e6 = *(device const half4*)(expert_base + 6 * hidden_dim + col);
            acc = fma(half4(w[6]), e6, acc);
        }
        if (top_k >= 8) {
            half4 e7 = *(device const half4*)(expert_base + 7 * hidden_dim + col);
            acc = fma(half4(w[7]), e7, acc);
        }

        // Store result
        *(device half4*)(out_base + col) = acc;
    }

    // Handle boundary columns (not multiple of 4)
    uint remaining_start = n_block + ((COMBINE_TILE_N / 4) * 4);
    for (uint col = remaining_start + thread_idx; col < n_block + COMBINE_TILE_N && col < hidden_dim; col += COMBINE_THREADS) {
        half acc = 0.0h;
        for (uint k = 0; k < top_k; ++k) {
            acc = fma(w[k], expert_base[k * hidden_dim + col], acc);
        }
        out_base[col] = acc;
    }
}

// ===========================================================================
// Kernel: moe_combine_weighted_simd8
//
// Optimized for exactly 8 experts (top_k=8) using 8-wide SIMD operations.
// Uses SIMD shuffle for efficient weight broadcast.
//
// Grid: [ceil(hidden_dim / TILE_N), batch_size]
// ===========================================================================

kernel void moe_combine_weighted_simd8(
    device const half* expert_outputs    [[buffer(0)]],  // [batch, 8, hidden]
    device const half* weights           [[buffer(1)]],  // [batch, 8]
    device half* output                  [[buffer(2)]],  // [batch, hidden]
    constant uint& batch_size            [[buffer(3)]],
    constant uint& hidden_dim            [[buffer(4)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * COMBINE_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Load all 8 weights into first 8 lanes of each simdgroup
    half w;
    if (simd_lane < 8) {
        w = weights[token_idx * 8 + simd_lane];
    }

    // Broadcast weights to all lanes using simd_shuffle
    half w0 = simd_shuffle(w, 0);
    half w1 = simd_shuffle(w, 1);
    half w2 = simd_shuffle(w, 2);
    half w3 = simd_shuffle(w, 3);
    half w4 = simd_shuffle(w, 4);
    half w5 = simd_shuffle(w, 5);
    half w6 = simd_shuffle(w, 6);
    half w7 = simd_shuffle(w, 7);

    device const half* expert_base = expert_outputs + token_idx * 8 * hidden_dim;
    device half* out_base = output + token_idx * hidden_dim;

    // Each thread processes 8 columns using two half4 vectors (Metal doesn't support half8)
    for (uint col = n_block + thread_idx * 8; col + 7 < hidden_dim; col += COMBINE_THREADS * 8) {
        // Load first 4 values from each expert
        half4 e0_lo = *(device const half4*)(expert_base + 0 * hidden_dim + col);
        half4 e1_lo = *(device const half4*)(expert_base + 1 * hidden_dim + col);
        half4 e2_lo = *(device const half4*)(expert_base + 2 * hidden_dim + col);
        half4 e3_lo = *(device const half4*)(expert_base + 3 * hidden_dim + col);
        half4 e4_lo = *(device const half4*)(expert_base + 4 * hidden_dim + col);
        half4 e5_lo = *(device const half4*)(expert_base + 5 * hidden_dim + col);
        half4 e6_lo = *(device const half4*)(expert_base + 6 * hidden_dim + col);
        half4 e7_lo = *(device const half4*)(expert_base + 7 * hidden_dim + col);

        // Load second 4 values from each expert
        half4 e0_hi = *(device const half4*)(expert_base + 0 * hidden_dim + col + 4);
        half4 e1_hi = *(device const half4*)(expert_base + 1 * hidden_dim + col + 4);
        half4 e2_hi = *(device const half4*)(expert_base + 2 * hidden_dim + col + 4);
        half4 e3_hi = *(device const half4*)(expert_base + 3 * hidden_dim + col + 4);
        half4 e4_hi = *(device const half4*)(expert_base + 4 * hidden_dim + col + 4);
        half4 e5_hi = *(device const half4*)(expert_base + 5 * hidden_dim + col + 4);
        half4 e6_hi = *(device const half4*)(expert_base + 6 * hidden_dim + col + 4);
        half4 e7_hi = *(device const half4*)(expert_base + 7 * hidden_dim + col + 4);

        // Weighted sum using FMA chain - low part
        half4 acc_lo = half4(w0) * e0_lo;
        acc_lo = fma(half4(w1), e1_lo, acc_lo);
        acc_lo = fma(half4(w2), e2_lo, acc_lo);
        acc_lo = fma(half4(w3), e3_lo, acc_lo);
        acc_lo = fma(half4(w4), e4_lo, acc_lo);
        acc_lo = fma(half4(w5), e5_lo, acc_lo);
        acc_lo = fma(half4(w6), e6_lo, acc_lo);
        acc_lo = fma(half4(w7), e7_lo, acc_lo);

        // Weighted sum using FMA chain - high part
        half4 acc_hi = half4(w0) * e0_hi;
        acc_hi = fma(half4(w1), e1_hi, acc_hi);
        acc_hi = fma(half4(w2), e2_hi, acc_hi);
        acc_hi = fma(half4(w3), e3_hi, acc_hi);
        acc_hi = fma(half4(w4), e4_hi, acc_hi);
        acc_hi = fma(half4(w5), e5_hi, acc_hi);
        acc_hi = fma(half4(w6), e6_hi, acc_hi);
        acc_hi = fma(half4(w7), e7_hi, acc_hi);

        *(device half4*)(out_base + col) = acc_lo;
        *(device half4*)(out_base + col + 4) = acc_hi;
    }

    // Boundary handling
    uint col_start = n_block + COMBINE_TILE_N - (COMBINE_TILE_N % 8);
    for (uint col = col_start + thread_idx; col < min(n_block + COMBINE_TILE_N, hidden_dim); col += COMBINE_THREADS) {
        half acc = 0.0h;
        acc = fma(w0, expert_base[0 * hidden_dim + col], acc);
        acc = fma(w1, expert_base[1 * hidden_dim + col], acc);
        acc = fma(w2, expert_base[2 * hidden_dim + col], acc);
        acc = fma(w3, expert_base[3 * hidden_dim + col], acc);
        acc = fma(w4, expert_base[4 * hidden_dim + col], acc);
        acc = fma(w5, expert_base[5 * hidden_dim + col], acc);
        acc = fma(w6, expert_base[6 * hidden_dim + col], acc);
        acc = fma(w7, expert_base[7 * hidden_dim + col], acc);
        out_base[col] = acc;
    }
}

// ===========================================================================
// Kernel: moe_combine_scattered
//
// Combines expert outputs when they are NOT stored contiguously.
// Uses expert_ids to gather outputs from sparse expert locations.
//
// Memory layout:
//   all_expert_outputs: [num_experts, batch, hidden_dim] half
//   expert_ids:         [batch, top_k] uint - indices into first dimension
//   weights:            [batch, top_k] half
//   output:             [batch, hidden_dim] half
// ===========================================================================

kernel void moe_combine_scattered(
    device const half* all_expert_outputs [[buffer(0)]],  // [num_experts, batch, hidden]
    device const uint* expert_ids         [[buffer(1)]],  // [batch, top_k]
    device const half* weights            [[buffer(2)]],  // [batch, top_k]
    device half* output                   [[buffer(3)]],  // [batch, hidden]
    constant uint& batch_size             [[buffer(4)]],
    constant uint& hidden_dim             [[buffer(5)]],
    constant uint& num_experts            [[buffer(6)]],
    constant uint& top_k                  [[buffer(7)]],
    uint3 tgid                            [[threadgroup_position_in_grid]],
    uint thread_idx                       [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * COMBINE_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Expert stride in the all_expert_outputs buffer
    const uint expert_stride = batch_size * hidden_dim;

    // Load weights and expert IDs for this token
    half w[MAX_EXPERTS_PARALLEL];
    uint e_ids[MAX_EXPERTS_PARALLEL];
    for (uint k = 0; k < top_k && k < MAX_EXPERTS_PARALLEL; ++k) {
        w[k] = weights[token_idx * top_k + k];
        e_ids[k] = expert_ids[token_idx * top_k + k];
    }

    device half* out_base = output + token_idx * hidden_dim;

    // Process columns with vectorized access where possible
    for (uint col = n_block + thread_idx * 4; col + 3 < n_block + COMBINE_TILE_N && col + 3 < hidden_dim; col += COMBINE_THREADS * 4) {
        half4 acc = half4(0.0h);

        for (uint k = 0; k < top_k && k < MAX_EXPERTS_PARALLEL; ++k) {
            // Gather: load from scattered expert location
            device const half* expert_ptr = all_expert_outputs + e_ids[k] * expert_stride + token_idx * hidden_dim + col;
            half4 e_val = *(device const half4*)expert_ptr;
            acc = fma(half4(w[k]), e_val, acc);
        }

        *(device half4*)(out_base + col) = acc;
    }

    // Boundary columns
    uint remaining = (hidden_dim - n_block) % 4;
    if (remaining > 0) {
        uint col_start = hidden_dim - remaining;
        for (uint col = col_start + thread_idx; col < hidden_dim; col += COMBINE_THREADS) {
            half acc = 0.0h;
            for (uint k = 0; k < top_k && k < MAX_EXPERTS_PARALLEL; ++k) {
                half e_val = all_expert_outputs[e_ids[k] * expert_stride + token_idx * hidden_dim + col];
                acc = fma(w[k], e_val, acc);
            }
            out_base[col] = acc;
        }
    }
}

// ===========================================================================
// Kernel: moe_combine_fp32_acc
//
// FP32 accumulation version for numerical stability.
// Useful when combining many experts or when inputs have large magnitude.
//
// Grid: [ceil(hidden_dim / TILE_N), batch_size]
// ===========================================================================

kernel void moe_combine_fp32_acc(
    device const half* expert_outputs    [[buffer(0)]],  // [batch, top_k, hidden]
    device const half* weights           [[buffer(1)]],  // [batch, top_k]
    device half* output                  [[buffer(2)]],  // [batch, hidden]
    constant uint& batch_size            [[buffer(3)]],
    constant uint& hidden_dim            [[buffer(4)]],
    constant uint& top_k                 [[buffer(5)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * COMBINE_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Load weights as FP32
    float w[MAX_EXPERTS_PARALLEL];
    for (uint k = 0; k < top_k && k < MAX_EXPERTS_PARALLEL; ++k) {
        w[k] = float(weights[token_idx * top_k + k]);
    }

    device const half* expert_base = expert_outputs + token_idx * top_k * hidden_dim;
    device half* out_base = output + token_idx * hidden_dim;

    // Process columns using float4 accumulator
    for (uint col = n_block + thread_idx * 4; col + 3 < n_block + COMBINE_TILE_N && col + 3 < hidden_dim; col += COMBINE_THREADS * 4) {
        float4 acc = float4(0.0f);

        for (uint k = 0; k < top_k && k < MAX_EXPERTS_PARALLEL; ++k) {
            half4 e_half = *(device const half4*)(expert_base + k * hidden_dim + col);
            float4 e_float = float4(e_half);
            acc = fma(float4(w[k]), e_float, acc);
        }

        // Convert back to half for storage
        half4 result = half4(acc);
        *(device half4*)(out_base + col) = result;
    }

    // Boundary columns
    for (uint col = n_block + ((COMBINE_TILE_N / 4) * 4) + thread_idx; col < min(n_block + COMBINE_TILE_N, hidden_dim); col += COMBINE_THREADS) {
        float acc = 0.0f;
        for (uint k = 0; k < top_k && k < MAX_EXPERTS_PARALLEL; ++k) {
            acc = fma(w[k], float(expert_base[k * hidden_dim + col]), acc);
        }
        out_base[col] = half(acc);
    }
}

// ===========================================================================
// Kernel: moe_combine_atomic_add
//
// Atomic add version for when expert outputs are written by separate kernels.
// Uses FP32 atomics (via CAS) for combining contributions.
//
// Note: This is slower than the direct combine kernels above, but allows
// each expert kernel to write directly to the output buffer.
//
// Grid: [ceil(hidden_dim / TILE_N), batch_size, top_k]
// ===========================================================================

kernel void moe_combine_atomic_add(
    device const half* expert_output     [[buffer(0)]],  // [batch, hidden] - single expert's output
    device const half* weight            [[buffer(1)]],  // [batch] - weight for this expert
    device float* output                 [[buffer(2)]],  // [batch, hidden] - FP32 accumulator
    constant uint& batch_size            [[buffer(3)]],
    constant uint& hidden_dim            [[buffer(4)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * COMBINE_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    float w = float(weight[token_idx]);
    device const half* in_base = expert_output + token_idx * hidden_dim;
    device float* out_base = output + token_idx * hidden_dim;

    // Atomic add using CAS loop
    for (uint col = n_block + thread_idx; col < min(n_block + COMBINE_TILE_N, hidden_dim); col += COMBINE_THREADS) {
        float weighted = float(in_base[col]) * w;

        device atomic_uint* atomic_ptr = (device atomic_uint*)(&out_base[col]);
        uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
        uint new_bits;
        bool success;
        do {
            float old_val = as_type<float>(old_bits);
            float new_val = old_val + weighted;
            new_bits = as_type<uint>(new_val);
            success = atomic_compare_exchange_weak_explicit(
                atomic_ptr, &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed);
        } while (!success);
    }
}

// ===========================================================================
// Kernel: moe_combine_finalize
//
// Converts FP32 accumulator to FP16 output after atomic combine.
// Call this after all expert contributions have been atomically added.
//
// Grid: [ceil(hidden_dim / TILE_N), batch_size]
// ===========================================================================

kernel void moe_combine_finalize(
    device const float* fp32_output      [[buffer(0)]],  // [batch, hidden] FP32 accumulator
    device half* fp16_output             [[buffer(1)]],  // [batch, hidden] FP16 final output
    constant uint& batch_size            [[buffer(2)]],
    constant uint& hidden_dim            [[buffer(3)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * COMBINE_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    device const float* in_base = fp32_output + token_idx * hidden_dim;
    device half* out_base = fp16_output + token_idx * hidden_dim;

    // Vectorized conversion using float4 -> half4
    for (uint col = n_block + thread_idx * 4; col + 3 < min(n_block + COMBINE_TILE_N, hidden_dim); col += COMBINE_THREADS * 4) {
        float4 f4 = *(device const float4*)(in_base + col);
        *(device half4*)(out_base + col) = half4(f4);
    }

    // Boundary
    for (uint col = n_block + ((COMBINE_TILE_N / 4) * 4) + thread_idx; col < min(n_block + COMBINE_TILE_N, hidden_dim); col += COMBINE_THREADS) {
        out_base[col] = half(in_base[col]);
    }
}
