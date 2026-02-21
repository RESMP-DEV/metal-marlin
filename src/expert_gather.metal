// expert_gather.metal - GPU-side expert weight gathering to avoid CPU sync
//
// MoE models require gathering expert weights based on routing decisions.
// Naive PyTorch advanced indexing (expert_weights[expert_indices]) triggers
// CPU-GPU synchronization because indices must be materialized on CPU.
//
// This kernel performs the gather entirely on GPU:
//   output[batch, k, :] = expert_weights[expert_indices[batch, k], :]
//
// Key benefits:
//   - Zero CPU sync: indices remain on GPU throughout
//   - Overlaps with routing computation: gather can start while routing finishes
//   - Memory bandwidth optimized: vectorized half4/half8 loads and stores
//   - Supports both contiguous and strided weight layouts
//
// Memory layout assumptions:
//   expert_weights: [num_experts, hidden_dim, out_dim] half, row-major
//   expert_indices: [batch, top_k] uint32
//   output:         [batch, top_k, out_dim] half
//
// For quantized weights (FP4), use expert_gather_fp4 which includes dequantization.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ===========================================================================
// Configuration Constants
// ===========================================================================

constant constexpr uint GATHER_TILE_BATCH = 32;    // Batch elements per threadgroup
constant constexpr uint GATHER_TILE_OUT = 256;     // Output dimension elements per threadgroup
constant constexpr uint GATHER_THREADS = 256;      // Threads per threadgroup

// ===========================================================================
// Kernel: expert_gather
// ===========================================================================
//
// Gathers expert weight slices based on per-token expert assignments.
// Each token selects top_k experts, and this kernel gathers the corresponding
// weight rows for subsequent GEMM operations.
//
// Grid: [ceil(out_dim / TILE_OUT), ceil(batch * top_k / TILE_BATCH)]
//
// This is the entry point for GPU-side expert selection, eliminating the
// CPU sync that occurs with torch.index_select or advanced indexing.

kernel void expert_gather(
    device const half* expert_weights   [[buffer(0)]],  // [num_experts, hidden, out]
    device const uint* expert_indices   [[buffer(1)]],  // [batch, top_k]
    device half* output                 [[buffer(2)]],  // [batch, top_k, out]
    constant uint& num_experts          [[buffer(3)]],
    constant uint& hidden_dim           [[buffer(4)]],
    constant uint& out_dim              [[buffer(5)]],
    constant uint& batch_size           [[buffer(6)]],
    constant uint& top_k                [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    // Compute which batch/expert slot and output column this thread handles
    const uint out_block = tgid.x * GATHER_TILE_OUT;
    const uint batch_block = tgid.y * GATHER_TILE_BATCH;

    const uint total_slots = batch_size * top_k;
    const uint expert_weight_stride = hidden_dim * out_dim;

    // Each thread processes multiple output elements across multiple batch slots
    // to maximize memory coalescing and hide latency
    const uint slots_per_wave = GATHER_THREADS / (GATHER_TILE_OUT / 4);  // Threads process 4 outputs each
    const uint slot_in_tile = thread_idx / (GATHER_TILE_OUT / 4);
    const uint out_offset_in_tile = (thread_idx % (GATHER_TILE_OUT / 4)) * 4;

    // Process multiple slots per thread for better occupancy
    for (uint s = slot_in_tile; s < GATHER_TILE_BATCH; s += slots_per_wave) {
        uint global_slot = batch_block + s;
        if (global_slot >= total_slots) break;

        // Get the expert index for this slot
        uint expert_id = expert_indices[global_slot];

        // Bounds check expert_id
        if (expert_id >= num_experts) continue;

        // Compute source offset in expert weights
        // We're gathering one row (out_dim elements) from the expert's weight matrix
        device const half* src_row = expert_weights + expert_id * expert_weight_stride;
        device half* dst_row = output + global_slot * out_dim;

        uint global_out = out_block + out_offset_in_tile;

        // Vectorized gather using half4
        if (global_out + 3 < out_dim) {
            half4 vals = *(device const half4*)(src_row + global_out);
            *(device half4*)(dst_row + global_out) = vals;
        } else if (global_out < out_dim) {
            // Scalar fallback for boundary
            for (uint i = 0; i < 4 && global_out + i < out_dim; ++i) {
                dst_row[global_out + i] = src_row[global_out + i];
            }
        }
    }
}

// ===========================================================================
// Kernel: expert_gather_2d
// ===========================================================================
//
// Gathers 2D slices from expert weights: output[b,k,h,:] = weights[indices[b,k],h,:]
//
// This variant gathers along the hidden dimension for cases where we need
// a specific hidden slice from each expert.
//
// Grid: [ceil(out_dim / TILE_OUT), ceil(batch * top_k / TILE_BATCH)]

kernel void expert_gather_2d(
    device const half* expert_weights   [[buffer(0)]],  // [num_experts, hidden, out]
    device const uint* expert_indices   [[buffer(1)]],  // [batch, top_k]
    device half* output                 [[buffer(2)]],  // [batch * top_k, hidden, out]
    constant uint& num_experts          [[buffer(3)]],
    constant uint& hidden_dim           [[buffer(4)]],
    constant uint& out_dim              [[buffer(5)]],
    constant uint& batch_size           [[buffer(6)]],
    constant uint& top_k                [[buffer(7)]],
    constant uint& hidden_slice         [[buffer(8)]],  // Which hidden row to gather
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    const uint out_block = tgid.x * GATHER_TILE_OUT;
    const uint batch_block = tgid.y * GATHER_TILE_BATCH;

    const uint total_slots = batch_size * top_k;
    const uint expert_weight_stride = hidden_dim * out_dim;

    const uint slots_per_wave = GATHER_THREADS / (GATHER_TILE_OUT / 4);
    const uint slot_in_tile = thread_idx / (GATHER_TILE_OUT / 4);
    const uint out_offset_in_tile = (thread_idx % (GATHER_TILE_OUT / 4)) * 4;

    for (uint s = slot_in_tile; s < GATHER_TILE_BATCH; s += slots_per_wave) {
        uint global_slot = batch_block + s;
        if (global_slot >= total_slots) break;

        uint expert_id = expert_indices[global_slot];
        if (expert_id >= num_experts || hidden_slice >= hidden_dim) continue;

        // Source: expert_weights[expert_id, hidden_slice, :]
        device const half* src_row = expert_weights +
            expert_id * expert_weight_stride +
            hidden_slice * out_dim;

        // Destination: output[global_slot, :]
        device half* dst_row = output + global_slot * out_dim;

        uint global_out = out_block + out_offset_in_tile;

        if (global_out + 3 < out_dim) {
            half4 vals = *(device const half4*)(src_row + global_out);
            *(device half4*)(dst_row + global_out) = vals;
        } else if (global_out < out_dim) {
            for (uint i = 0; i < 4 && global_out + i < out_dim; ++i) {
                dst_row[global_out + i] = src_row[global_out + i];
            }
        }
    }
}

// ===========================================================================
// Kernel: expert_gather_full
// ===========================================================================
//
// Gathers complete expert weight matrices for selected experts.
// Output contains the full [hidden, out] weight matrix for each selected expert.
//
// output[b * top_k + k, h, o] = expert_weights[indices[b, k], h, o]
//
// This is used when the full expert weight matrix is needed, not just a slice.
// More memory intensive but avoids repeated gathers during the forward pass.
//
// Grid: [ceil(out_dim / TILE_OUT), ceil(hidden_dim / 8), batch * top_k]

kernel void expert_gather_full(
    device const half* expert_weights   [[buffer(0)]],  // [num_experts, hidden, out]
    device const uint* expert_indices   [[buffer(1)]],  // [batch, top_k]
    device half* output                 [[buffer(2)]],  // [batch * top_k, hidden, out]
    constant uint& num_experts          [[buffer(3)]],
    constant uint& hidden_dim           [[buffer(4)]],
    constant uint& out_dim              [[buffer(5)]],
    constant uint& batch_size           [[buffer(6)]],
    constant uint& top_k                [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    const uint out_block = tgid.x * GATHER_TILE_OUT;
    const uint hidden_block = tgid.y * 8;  // Process 8 hidden rows per threadgroup
    const uint global_slot = tgid.z;

    const uint total_slots = batch_size * top_k;
    if (global_slot >= total_slots) return;

    uint expert_id = expert_indices[global_slot];
    if (expert_id >= num_experts) return;

    const uint expert_weight_stride = hidden_dim * out_dim;

    // Each thread handles one output column across 8 hidden rows
    for (uint o = out_block + thread_idx; o < out_dim && o < out_block + GATHER_TILE_OUT; o += GATHER_THREADS) {
        // Process 8 hidden rows
        #pragma unroll
        for (uint h = 0; h < 8; ++h) {
            uint global_h = hidden_block + h;
            if (global_h >= hidden_dim) break;

            // Source: expert_weights[expert_id, global_h, o]
            half val = expert_weights[expert_id * expert_weight_stride + global_h * out_dim + o];

            // Destination: output[global_slot, global_h, o]
            output[global_slot * expert_weight_stride + global_h * out_dim + o] = val;
        }
    }
}

// ===========================================================================
// Kernel: expert_gather_vec8
// ===========================================================================
//
// Optimized vectorized gather using half8 loads when out_dim is divisible by 8.
// Achieves higher memory bandwidth utilization through wider vector operations.
//
// Grid: [ceil(out_dim / (TILE_OUT * 2)), ceil(batch * top_k / TILE_BATCH)]

kernel void expert_gather_vec8(
    device const half* expert_weights   [[buffer(0)]],  // [num_experts, hidden, out]
    device const uint* expert_indices   [[buffer(1)]],  // [batch, top_k]
    device half* output                 [[buffer(2)]],  // [batch, top_k, out]
    constant uint& num_experts          [[buffer(3)]],
    constant uint& hidden_dim           [[buffer(4)]],
    constant uint& out_dim              [[buffer(5)]],
    constant uint& batch_size           [[buffer(6)]],
    constant uint& top_k                [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    const uint out_block = tgid.x * GATHER_TILE_OUT * 2;  // Wider tile for vec8
    const uint batch_block = tgid.y * GATHER_TILE_BATCH;

    const uint total_slots = batch_size * top_k;
    const uint expert_weight_stride = hidden_dim * out_dim;

    // With half8, each thread processes 8 output elements
    const uint slots_per_wave = GATHER_THREADS / (GATHER_TILE_OUT * 2 / 8);
    const uint slot_in_tile = thread_idx / (GATHER_TILE_OUT * 2 / 8);
    const uint out_offset_in_tile = (thread_idx % (GATHER_TILE_OUT * 2 / 8)) * 8;

    for (uint s = slot_in_tile; s < GATHER_TILE_BATCH; s += slots_per_wave) {
        uint global_slot = batch_block + s;
        if (global_slot >= total_slots) break;

        uint expert_id = expert_indices[global_slot];
        if (expert_id >= num_experts) continue;

        device const half* src_row = expert_weights + expert_id * expert_weight_stride;
        device half* dst_row = output + global_slot * out_dim;

        uint global_out = out_block + out_offset_in_tile;

        // Use two half4 loads/stores to simulate half8
        if (global_out + 7 < out_dim) {
            half4 vals0 = *(device const half4*)(src_row + global_out);
            half4 vals1 = *(device const half4*)(src_row + global_out + 4);
            *(device half4*)(dst_row + global_out) = vals0;
            *(device half4*)(dst_row + global_out + 4) = vals1;
        } else if (global_out < out_dim) {
            // Scalar fallback for boundary
            for (uint i = 0; i < 8 && global_out + i < out_dim; ++i) {
                dst_row[global_out + i] = src_row[global_out + i];
            }
        }
    }
}

// ===========================================================================
// Kernel: expert_gather_with_hidden
// ===========================================================================
//
// Gathers expert weights and applies them to hidden states in a single kernel.
// This is a fused gather-multiply operation for the first stage of MoE forward.
//
// output[b,k,:] = input[b,:] @ expert_weights[indices[b,k],:,:]
//
// For small batch sizes, this avoids materializing the gathered weights.

kernel void expert_gather_with_hidden(
    device const half* input            [[buffer(0)]],  // [batch, hidden]
    device const half* expert_weights   [[buffer(1)]],  // [num_experts, hidden, out]
    device const uint* expert_indices   [[buffer(2)]],  // [batch, top_k]
    device half* output                 [[buffer(3)]],  // [batch, top_k, out]
    constant uint& num_experts          [[buffer(4)]],
    constant uint& hidden_dim           [[buffer(5)]],
    constant uint& out_dim              [[buffer(6)]],
    constant uint& batch_size           [[buffer(7)]],
    constant uint& top_k                [[buffer(8)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint thread_idx                     [[thread_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory for input tile
    threadgroup half shared_input[32][32];  // [batch_tile, hidden_tile]

    const uint out_block = tgid.x * 64;  // Output tile size
    const uint batch_slot = tgid.y;      // batch * top_k index

    const uint total_slots = batch_size * top_k;
    if (batch_slot >= total_slots) return;

    const uint batch_idx = batch_slot / top_k;
    const uint k_idx = batch_slot % top_k;

    uint expert_id = expert_indices[batch_idx * top_k + k_idx];
    if (expert_id >= num_experts) return;

    const uint expert_weight_stride = hidden_dim * out_dim;

    // Initialize accumulator
    half acc[4] = {0.0h, 0.0h, 0.0h, 0.0h};

    // Process hidden dimension in tiles
    for (uint h_block = 0; h_block < hidden_dim; h_block += 32) {
        // Load input tile cooperatively
        if (thread_idx < 32) {
            uint h = h_block + thread_idx;
            shared_input[0][thread_idx] = (h < hidden_dim) ? input[batch_idx * hidden_dim + h] : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread computes 4 output elements
        uint out_base = out_block + thread_idx * 4;
        if (out_base >= out_dim) continue;

        // Accumulate: acc += input[batch, h_block:h_block+32] @ weights[expert, h_block:h_block+32, out_base:out_base+4]
        for (uint h = 0; h < 32 && h_block + h < hidden_dim; ++h) {
            half in_val = shared_input[0][h];

            #pragma unroll
            for (uint o = 0; o < 4; ++o) {
                uint global_out = out_base + o;
                if (global_out < out_dim) {
                    half w = expert_weights[expert_id * expert_weight_stride + (h_block + h) * out_dim + global_out];
                    acc[o] += in_val * w;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    uint out_base = out_block + thread_idx * 4;
    if (out_base < out_dim) {
        device half* dst = output + batch_slot * out_dim + out_base;

        if (out_base + 3 < out_dim) {
            *(device half4*)dst = half4(acc[0], acc[1], acc[2], acc[3]);
        } else {
            for (uint o = 0; o < 4 && out_base + o < out_dim; ++o) {
                dst[o] = acc[o];
            }
        }
    }
}

// ===========================================================================
// Kernel: expert_gather_transposed
// ===========================================================================
//
// Gathers expert weights with transposed output layout for GEMM optimization.
// Output is laid out for efficient subsequent matrix multiplication.
//
// output[k, batch, :] = expert_weights[indices[batch, k], :]
//
// This layout allows contiguous memory access when processing all tokens
// for a single expert.

kernel void expert_gather_transposed(
    device const half* expert_weights   [[buffer(0)]],  // [num_experts, hidden, out]
    device const uint* expert_indices   [[buffer(1)]],  // [batch, top_k]
    device half* output                 [[buffer(2)]],  // [top_k, batch, out]
    constant uint& num_experts          [[buffer(3)]],
    constant uint& hidden_dim           [[buffer(4)]],
    constant uint& out_dim              [[buffer(5)]],
    constant uint& batch_size           [[buffer(6)]],
    constant uint& top_k                [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    const uint out_block = tgid.x * GATHER_TILE_OUT;
    const uint batch_block = tgid.y * GATHER_TILE_BATCH;
    const uint k_idx = tgid.z;  // top_k index

    if (k_idx >= top_k) return;

    const uint expert_weight_stride = hidden_dim * out_dim;
    const uint output_batch_stride = batch_size * out_dim;  // Transposed layout

    // Each thread processes 4 output elements
    const uint elems_per_thread = 4;
    const uint threads_per_batch = GATHER_TILE_OUT / elems_per_thread;  // 64
    const uint batch_in_tile = thread_idx / threads_per_batch;
    const uint out_offset = (thread_idx % threads_per_batch) * elems_per_thread;

    for (uint b = batch_in_tile; b < GATHER_TILE_BATCH; b += GATHER_THREADS / threads_per_batch) {
        uint global_batch = batch_block + b;
        if (global_batch >= batch_size) break;

        uint expert_id = expert_indices[global_batch * top_k + k_idx];
        if (expert_id >= num_experts) continue;

        device const half* src_row = expert_weights + expert_id * expert_weight_stride;
        // Transposed output: [k_idx, global_batch, :]
        device half* dst_row = output + k_idx * output_batch_stride + global_batch * out_dim;

        uint global_out = out_block + out_offset;

        if (global_out + 3 < out_dim) {
            half4 vals = *(device const half4*)(src_row + global_out);
            *(device half4*)(dst_row + global_out) = vals;
        } else if (global_out < out_dim) {
            for (uint i = 0; i < 4 && global_out + i < out_dim; ++i) {
                dst_row[global_out + i] = src_row[global_out + i];
            }
        }
    }
}

// ===========================================================================
// Kernel: expert_gather_fp4
// ===========================================================================
//
// Gathers FP4-quantized expert weights with on-the-fly dequantization.
// This avoids materializing dequantized weights, saving memory bandwidth.
//
// The weights are stored as packed FP4 (8 values per uint32), and this kernel
// dequantizes them during the gather operation.

// FP4 E2M1 dequantization helper
inline half dequant_fp4_scalar(uint nibble) {
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
    return select(magnitude, -magnitude, bool(sign_bit));
}

kernel void expert_gather_fp4(
    device const uint* expert_weights   [[buffer(0)]],  // [num_experts, hidden/8, out] packed FP4
    device const half* scales           [[buffer(1)]],  // [num_experts, hidden/group_size, out]
    device const uint* expert_indices   [[buffer(2)]],  // [batch, top_k]
    device half* output                 [[buffer(3)]],  // [batch * top_k, hidden, out] dequantized
    constant uint& num_experts          [[buffer(4)]],
    constant uint& hidden_dim           [[buffer(5)]],
    constant uint& out_dim              [[buffer(6)]],
    constant uint& batch_size           [[buffer(7)]],
    constant uint& top_k                [[buffer(8)]],
    constant uint& group_size           [[buffer(9)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint thread_idx                     [[thread_index_in_threadgroup]]
) {
    const uint out_block = tgid.x * 64;           // Output columns per threadgroup
    const uint hidden_block = tgid.y * 8;         // Hidden rows (8 FP4 values = 1 packed uint)
    const uint global_slot = tgid.z;              // batch * top_k index

    const uint total_slots = batch_size * top_k;
    if (global_slot >= total_slots) return;

    uint expert_id = expert_indices[global_slot];
    if (expert_id >= num_experts) return;

    const uint hidden_packed = (hidden_dim + 7) / 8;
    const uint num_groups = (hidden_dim + group_size - 1) / group_size;

    const uint expert_weight_stride = hidden_packed * out_dim;
    const uint expert_scale_stride = num_groups * out_dim;
    const uint output_stride = hidden_dim * out_dim;

    // Each thread handles one output column and unpacks 8 hidden values
    for (uint o = out_block + thread_idx; o < out_dim && o < out_block + 64; o += GATHER_THREADS) {
        // Skip if out of bounds
        if (hidden_block >= hidden_dim) continue;

        // Read packed FP4 value (contains 8 hidden values)
        uint packed_h = hidden_block / 8;
        uint packed = expert_weights[expert_id * expert_weight_stride + packed_h * out_dim + o];

        // Read scale for this group
        uint scale_group = hidden_block / group_size;
        half s = scales[expert_id * expert_scale_stride + scale_group * out_dim + o];

        // Dequantize and store each of the 8 values
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint global_h = hidden_block + i;
            if (global_h >= hidden_dim) break;

            uint nibble = (packed >> (i * 4)) & 0xF;
            half raw = dequant_fp4_scalar(nibble);
            half val = raw * s;

            output[global_slot * output_stride + global_h * out_dim + o] = val;
        }
    }
}

// ===========================================================================
// Kernel: expert_indices_validate
// ===========================================================================
//
// Validates expert indices are within bounds and clamps or replaces invalid ones.
// Run this before gather operations to avoid out-of-bounds memory access.
//
// This kernel is optional but recommended for safety when expert indices
// come from learned routing that might produce out-of-range values.

kernel void expert_indices_validate(
    device uint* expert_indices         [[buffer(0)]],  // [batch, top_k] in-place
    constant uint& num_experts          [[buffer(1)]],
    constant uint& batch_size           [[buffer(2)]],
    constant uint& top_k                [[buffer(3)]],
    constant uint& replacement_expert   [[buffer(4)]],  // Index to use for invalid experts
    uint tid                            [[thread_position_in_grid]]
) {
    uint total = batch_size * top_k;
    if (tid >= total) return;

    uint expert_id = expert_indices[tid];
    if (expert_id >= num_experts) {
        expert_indices[tid] = replacement_expert;
    }
}
