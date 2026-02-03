// expert_scatter_add.metal - Expert output combination via weighted scatter-add
//
// Combines expert outputs by scattering and adding weighted contributions
// to the output tensor without Python-side indexing operations.
//
// Input layout:
//   expert_outputs:  [batch, top_k, out_dim] half - per-expert outputs
//   expert_indices:  [batch, top_k] uint - which expert produced each output
//   routing_weights: [batch, top_k] half - normalized routing probabilities
//
// Output layout:
//   output: [batch, out_dim] half - combined result
//
// Computation:
//   output[b, d] = Σ_k (routing_weights[b, k] * expert_outputs[b, k, d])
//
// This kernel is designed for the common MoE pattern where expert outputs
// are already arranged per-token (not per-expert), avoiding the need for
// Python-side gather/scatter operations.
//
// Key optimizations:
//   - Vectorized half4 loads for expert outputs (4x throughput)
//   - Register tiling for weight broadcast (no re-reads)
//   - Coalesced memory access via output-dimension blocking
//   - Specialized fast paths for top_k=2,4,8
//   - SIMD shuffle for weight distribution within simdgroups
//
// Performance notes:
//   - Output-dimension parallelism: threads process consecutive out_dim columns
//   - Batch parallelism: each threadgroup handles one token
//   - Expert loop unrolled for common top_k values

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ===========================================================================
// Configuration Constants
// ===========================================================================

constant constexpr uint SCATTER_ADD_THREADS = 256;     // Threads per threadgroup
constant constexpr uint SCATTER_ADD_TILE_N = 256;      // Output columns per threadgroup
constant constexpr uint MAX_TOP_K_SCATTER = 8;         // Maximum supported top_k

// ===========================================================================
// Kernel: expert_scatter_add
//
// Combines expert outputs using routing weights. Each threadgroup processes
// one token (batch element) and a tile of output dimensions.
//
// Grid dispatch: [ceil(out_dim / TILE_N), batch_size]
//
// This is the workhorse kernel for MoE output combination when expert
// outputs are arranged as [batch, top_k, out_dim].
// ===========================================================================

kernel void expert_scatter_add(
    device const half* expert_outputs    [[buffer(0)]],  // [batch, top_k, out_dim]
    device const uint* expert_indices    [[buffer(1)]],  // [batch, top_k] (unused in basic combine)
    device const half* routing_weights   [[buffer(2)]],  // [batch, top_k]
    device half* output                  [[buffer(3)]],  // [batch, out_dim]
    constant uint& batch_size            [[buffer(4)]],
    constant uint& top_k                 [[buffer(5)]],
    constant uint& out_dim               [[buffer(6)]],
    uint3 gid                            [[thread_position_in_grid]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * SCATTER_ADD_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Load routing weights for this token into registers
    half w[MAX_TOP_K_SCATTER];
    for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
        w[k] = routing_weights[token_idx * top_k + k];
    }
    for (uint k = top_k; k < MAX_TOP_K_SCATTER; ++k) {
        w[k] = 0.0h;
    }

    // Base pointers for this token
    // expert_outputs layout: [batch][top_k][out_dim]
    device const half* expert_base = expert_outputs + token_idx * top_k * out_dim;
    device half* out_base = output + token_idx * out_dim;

    // Process output columns, 4 at a time using vectorized loads
    // Each thread handles 4 consecutive columns
    for (uint col = n_block + thread_idx * 4;
         col < n_block + SCATTER_ADD_TILE_N && col + 3 < out_dim;
         col += SCATTER_ADD_THREADS * 4) {

        half4 acc = half4(0.0h);

        // Unrolled loop for common top_k values
        // Specialized paths reduce branch overhead
        if (top_k == 2) {
            // top_k=2: Most common for efficient MoE
            half4 e0 = *(device const half4*)(expert_base + 0 * out_dim + col);
            half4 e1 = *(device const half4*)(expert_base + 1 * out_dim + col);
            acc = fma(half4(w[0]), e0, acc);
            acc = fma(half4(w[1]), e1, acc);
        } else if (top_k == 4) {
            // top_k=4: Common for larger capacity
            half4 e0 = *(device const half4*)(expert_base + 0 * out_dim + col);
            half4 e1 = *(device const half4*)(expert_base + 1 * out_dim + col);
            half4 e2 = *(device const half4*)(expert_base + 2 * out_dim + col);
            half4 e3 = *(device const half4*)(expert_base + 3 * out_dim + col);
            acc = fma(half4(w[0]), e0, acc);
            acc = fma(half4(w[1]), e1, acc);
            acc = fma(half4(w[2]), e2, acc);
            acc = fma(half4(w[3]), e3, acc);
        } else if (top_k == 8) {
            // top_k=8: High capacity models
            #pragma unroll
            for (uint k = 0; k < 8; ++k) {
                half4 ek = *(device const half4*)(expert_base + k * out_dim + col);
                acc = fma(half4(w[k]), ek, acc);
            }
        } else {
            // Generic path for arbitrary top_k
            for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
                half4 ek = *(device const half4*)(expert_base + k * out_dim + col);
                acc = fma(half4(w[k]), ek, acc);
            }
        }

        // Store result
        *(device half4*)(out_base + col) = acc;
    }

    // Handle boundary columns (not multiple of 4)
    uint remaining_start = n_block + SCATTER_ADD_TILE_N - (SCATTER_ADD_TILE_N % 4);
    if (remaining_start > out_dim) {
        remaining_start = (out_dim / 4) * 4;
    }
    for (uint col = remaining_start + thread_idx;
         col < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS) {
        half acc = 0.0h;
        for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
            acc = fma(w[k], expert_base[k * out_dim + col], acc);
        }
        out_base[col] = acc;
    }
}

// ===========================================================================
// Kernel: expert_scatter_add_simd
//
// SIMD-optimized version using shuffle for weight broadcast.
// Best for small top_k (≤8) where weights fit in a single SIMD width.
//
// Grid dispatch: [ceil(out_dim / TILE_N), batch_size]
// ===========================================================================

kernel void expert_scatter_add_simd(
    device const half* expert_outputs    [[buffer(0)]],  // [batch, top_k, out_dim]
    device const uint* expert_indices    [[buffer(1)]],  // [batch, top_k]
    device const half* routing_weights   [[buffer(2)]],  // [batch, top_k]
    device half* output                  [[buffer(3)]],  // [batch, out_dim]
    constant uint& batch_size            [[buffer(4)]],
    constant uint& top_k                 [[buffer(5)]],
    constant uint& out_dim               [[buffer(6)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * SCATTER_ADD_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Load weight into first top_k lanes of each simdgroup
    half w = 0.0h;
    if (simd_lane < top_k) {
        w = routing_weights[token_idx * top_k + simd_lane];
    }

    // Broadcast weights using SIMD shuffle
    half w0 = simd_shuffle(w, 0);
    half w1 = top_k >= 2 ? simd_shuffle(w, 1) : 0.0h;
    half w2 = top_k >= 3 ? simd_shuffle(w, 2) : 0.0h;
    half w3 = top_k >= 4 ? simd_shuffle(w, 3) : 0.0h;
    half w4 = top_k >= 5 ? simd_shuffle(w, 4) : 0.0h;
    half w5 = top_k >= 6 ? simd_shuffle(w, 5) : 0.0h;
    half w6 = top_k >= 7 ? simd_shuffle(w, 6) : 0.0h;
    half w7 = top_k >= 8 ? simd_shuffle(w, 7) : 0.0h;

    device const half* expert_base = expert_outputs + token_idx * top_k * out_dim;
    device half* out_base = output + token_idx * out_dim;

    // Process 8 columns per thread using two half4 vectors
    for (uint col = n_block + thread_idx * 8;
         col + 7 < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS * 8) {

        half4 acc_lo = half4(0.0h);
        half4 acc_hi = half4(0.0h);

        // Expert 0
        if (top_k >= 1) {
            half4 e0_lo = *(device const half4*)(expert_base + 0 * out_dim + col);
            half4 e0_hi = *(device const half4*)(expert_base + 0 * out_dim + col + 4);
            acc_lo = fma(half4(w0), e0_lo, acc_lo);
            acc_hi = fma(half4(w0), e0_hi, acc_hi);
        }
        // Expert 1
        if (top_k >= 2) {
            half4 e1_lo = *(device const half4*)(expert_base + 1 * out_dim + col);
            half4 e1_hi = *(device const half4*)(expert_base + 1 * out_dim + col + 4);
            acc_lo = fma(half4(w1), e1_lo, acc_lo);
            acc_hi = fma(half4(w1), e1_hi, acc_hi);
        }
        // Expert 2
        if (top_k >= 3) {
            half4 e2_lo = *(device const half4*)(expert_base + 2 * out_dim + col);
            half4 e2_hi = *(device const half4*)(expert_base + 2 * out_dim + col + 4);
            acc_lo = fma(half4(w2), e2_lo, acc_lo);
            acc_hi = fma(half4(w2), e2_hi, acc_hi);
        }
        // Expert 3
        if (top_k >= 4) {
            half4 e3_lo = *(device const half4*)(expert_base + 3 * out_dim + col);
            half4 e3_hi = *(device const half4*)(expert_base + 3 * out_dim + col + 4);
            acc_lo = fma(half4(w3), e3_lo, acc_lo);
            acc_hi = fma(half4(w3), e3_hi, acc_hi);
        }
        // Expert 4
        if (top_k >= 5) {
            half4 e4_lo = *(device const half4*)(expert_base + 4 * out_dim + col);
            half4 e4_hi = *(device const half4*)(expert_base + 4 * out_dim + col + 4);
            acc_lo = fma(half4(w4), e4_lo, acc_lo);
            acc_hi = fma(half4(w4), e4_hi, acc_hi);
        }
        // Expert 5
        if (top_k >= 6) {
            half4 e5_lo = *(device const half4*)(expert_base + 5 * out_dim + col);
            half4 e5_hi = *(device const half4*)(expert_base + 5 * out_dim + col + 4);
            acc_lo = fma(half4(w5), e5_lo, acc_lo);
            acc_hi = fma(half4(w5), e5_hi, acc_hi);
        }
        // Expert 6
        if (top_k >= 7) {
            half4 e6_lo = *(device const half4*)(expert_base + 6 * out_dim + col);
            half4 e6_hi = *(device const half4*)(expert_base + 6 * out_dim + col + 4);
            acc_lo = fma(half4(w6), e6_lo, acc_lo);
            acc_hi = fma(half4(w6), e6_hi, acc_hi);
        }
        // Expert 7
        if (top_k >= 8) {
            half4 e7_lo = *(device const half4*)(expert_base + 7 * out_dim + col);
            half4 e7_hi = *(device const half4*)(expert_base + 7 * out_dim + col + 4);
            acc_lo = fma(half4(w7), e7_lo, acc_lo);
            acc_hi = fma(half4(w7), e7_hi, acc_hi);
        }

        *(device half4*)(out_base + col) = acc_lo;
        *(device half4*)(out_base + col + 4) = acc_hi;
    }

    // Boundary handling
    uint col_start = n_block + SCATTER_ADD_TILE_N - (SCATTER_ADD_TILE_N % 8);
    for (uint col = col_start + thread_idx;
         col < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS) {
        half acc = 0.0h;
        if (top_k >= 1) acc = fma(w0, expert_base[0 * out_dim + col], acc);
        if (top_k >= 2) acc = fma(w1, expert_base[1 * out_dim + col], acc);
        if (top_k >= 3) acc = fma(w2, expert_base[2 * out_dim + col], acc);
        if (top_k >= 4) acc = fma(w3, expert_base[3 * out_dim + col], acc);
        if (top_k >= 5) acc = fma(w4, expert_base[4 * out_dim + col], acc);
        if (top_k >= 6) acc = fma(w5, expert_base[5 * out_dim + col], acc);
        if (top_k >= 7) acc = fma(w6, expert_base[6 * out_dim + col], acc);
        if (top_k >= 8) acc = fma(w7, expert_base[7 * out_dim + col], acc);
        out_base[col] = acc;
    }
}

// ===========================================================================
// Kernel: expert_scatter_add_fp32_acc
//
// FP32 accumulation version for numerical stability.
// Useful when combining many experts or when expert outputs have large variance.
//
// Grid dispatch: [ceil(out_dim / TILE_N), batch_size]
// ===========================================================================

kernel void expert_scatter_add_fp32_acc(
    device const half* expert_outputs    [[buffer(0)]],  // [batch, top_k, out_dim]
    device const uint* expert_indices    [[buffer(1)]],  // [batch, top_k]
    device const half* routing_weights   [[buffer(2)]],  // [batch, top_k]
    device half* output                  [[buffer(3)]],  // [batch, out_dim]
    constant uint& batch_size            [[buffer(4)]],
    constant uint& top_k                 [[buffer(5)]],
    constant uint& out_dim               [[buffer(6)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * SCATTER_ADD_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Load weights as FP32 for numerical stability
    float w[MAX_TOP_K_SCATTER];
    for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
        w[k] = float(routing_weights[token_idx * top_k + k]);
    }
    for (uint k = top_k; k < MAX_TOP_K_SCATTER; ++k) {
        w[k] = 0.0f;
    }

    device const half* expert_base = expert_outputs + token_idx * top_k * out_dim;
    device half* out_base = output + token_idx * out_dim;

    // Process columns using float4 accumulator
    for (uint col = n_block + thread_idx * 4;
         col + 3 < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS * 4) {

        float4 acc = float4(0.0f);

        for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
            half4 e_half = *(device const half4*)(expert_base + k * out_dim + col);
            float4 e_float = float4(e_half);
            acc = fma(float4(w[k]), e_float, acc);
        }

        // Convert back to half for storage
        *(device half4*)(out_base + col) = half4(acc);
    }

    // Boundary columns
    uint remaining_start = (out_dim / 4) * 4;
    for (uint col = remaining_start + thread_idx;
         col < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS) {
        float acc = 0.0f;
        for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
            acc = fma(w[k], float(expert_base[k * out_dim + col]), acc);
        }
        out_base[col] = half(acc);
    }
}

// ===========================================================================
// Kernel: expert_scatter_add_atomic
//
// Atomic scatter-add for when expert outputs are written by independent kernels.
// Uses FP32 CAS for thread-safe accumulation.
//
// Use case: Parallel expert GEMM where each expert writes to shared output.
//
// Grid dispatch: [ceil(out_dim / TILE_N), batch_size, top_k]
// ===========================================================================

kernel void expert_scatter_add_atomic(
    device const half* expert_output     [[buffer(0)]],  // [batch, out_dim] single expert
    device const half* weight            [[buffer(1)]],  // [batch] weight for this expert
    device float* output_accumulator     [[buffer(2)]],  // [batch, out_dim] FP32
    constant uint& batch_size            [[buffer(3)]],
    constant uint& out_dim               [[buffer(4)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * SCATTER_ADD_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    float w = float(weight[token_idx]);
    device const half* in_base = expert_output + token_idx * out_dim;
    device float* out_base = output_accumulator + token_idx * out_dim;

    // Atomic add using CAS loop
    for (uint col = n_block + thread_idx;
         col < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS) {

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
// Kernel: expert_scatter_add_finalize
//
// Converts FP32 accumulator to FP16 output after atomic scatter-add.
// Uses vectorized conversion for efficiency.
//
// Grid dispatch: [ceil(out_dim / TILE_N), batch_size]
// ===========================================================================

kernel void expert_scatter_add_finalize(
    device const float* fp32_output      [[buffer(0)]],  // [batch, out_dim] FP32
    device half* fp16_output             [[buffer(1)]],  // [batch, out_dim] FP16
    constant uint& batch_size            [[buffer(2)]],
    constant uint& out_dim               [[buffer(3)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint thread_idx                      [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * SCATTER_ADD_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    device const float* in_base = fp32_output + token_idx * out_dim;
    device half* out_base = fp16_output + token_idx * out_dim;

    // Vectorized conversion using float4 -> half4
    for (uint col = n_block + thread_idx * 4;
         col + 3 < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS * 4) {
        float4 f4 = *(device const float4*)(in_base + col);
        *(device half4*)(out_base + col) = half4(f4);
    }

    // Boundary
    uint remaining_start = (out_dim / 4) * 4;
    for (uint col = remaining_start + thread_idx;
         col < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS) {
        out_base[col] = half(in_base[col]);
    }
}

// ===========================================================================
// Kernel: expert_scatter_add_indexed
//
// Scatter-add with explicit expert index lookup.
// Handles the case where expert outputs are stored per-expert rather than
// per-token, requiring gather before combine.
//
// Memory layout:
//   all_expert_outputs: [num_experts, batch, out_dim] half
//   expert_indices:     [batch, top_k] uint - which expert for each slot
//   routing_weights:    [batch, top_k] half
//   output:             [batch, out_dim] half
//
// Grid dispatch: [ceil(out_dim / TILE_N), batch_size]
// ===========================================================================

kernel void expert_scatter_add_indexed(
    device const half* all_expert_outputs [[buffer(0)]],  // [num_experts, batch, out_dim]
    device const uint* expert_indices     [[buffer(1)]],  // [batch, top_k]
    device const half* routing_weights    [[buffer(2)]],  // [batch, top_k]
    device half* output                   [[buffer(3)]],  // [batch, out_dim]
    constant uint& batch_size             [[buffer(4)]],
    constant uint& out_dim                [[buffer(5)]],
    constant uint& num_experts            [[buffer(6)]],
    constant uint& top_k                  [[buffer(7)]],
    uint3 tgid                            [[threadgroup_position_in_grid]],
    uint thread_idx                       [[thread_index_in_threadgroup]]
) {
    const uint n_block = tgid.x * SCATTER_ADD_TILE_N;
    const uint token_idx = tgid.y;

    if (token_idx >= batch_size) return;

    // Expert stride: each expert has [batch, out_dim] worth of data
    const uint expert_stride = batch_size * out_dim;

    // Load weights and expert IDs for this token
    half w[MAX_TOP_K_SCATTER];
    uint e_ids[MAX_TOP_K_SCATTER];
    for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
        w[k] = routing_weights[token_idx * top_k + k];
        e_ids[k] = expert_indices[token_idx * top_k + k];
    }

    device half* out_base = output + token_idx * out_dim;

    // Process columns with vectorized gather-combine
    for (uint col = n_block + thread_idx * 4;
         col + 3 < min(n_block + SCATTER_ADD_TILE_N, out_dim);
         col += SCATTER_ADD_THREADS * 4) {

        half4 acc = half4(0.0h);

        for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
            // Gather from scattered expert location
            device const half* expert_ptr = all_expert_outputs +
                e_ids[k] * expert_stride + token_idx * out_dim + col;
            half4 e_val = *(device const half4*)expert_ptr;
            acc = fma(half4(w[k]), e_val, acc);
        }

        *(device half4*)(out_base + col) = acc;
    }

    // Boundary columns
    uint remaining = out_dim % 4;
    if (remaining > 0) {
        uint col_start = out_dim - remaining;
        for (uint col = col_start + thread_idx;
             col < min(n_block + SCATTER_ADD_TILE_N, out_dim);
             col += SCATTER_ADD_THREADS) {
            half acc = 0.0h;
            for (uint k = 0; k < top_k && k < MAX_TOP_K_SCATTER; ++k) {
                half e_val = all_expert_outputs[e_ids[k] * expert_stride +
                                                token_idx * out_dim + col];
                acc = fma(w[k], e_val, acc);
            }
            out_base[col] = acc;
        }
    }
}
