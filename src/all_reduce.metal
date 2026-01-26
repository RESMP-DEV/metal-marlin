/**
 * All-reduce kernel for Metal.
 *
 * This kernel implements in-device reduction operations for tensor parallelism.
 * For multi-GPU scenarios on Apple Silicon (future M-series multi-die chips),
 * this kernel handles local reduction before inter-device communication.
 *
 * Supported operations:
 * - sum: Element-wise sum reduction
 * - mean: Element-wise mean reduction
 * - max: Element-wise maximum
 * - min: Element-wise minimum
 *
 * For unified memory (current Apple Silicon), this kernel can also serve as
 * an efficient way to reduce tensors from multiple threads/threadgroups.
 *
 * Memory layout:
 * - Input: [num_shards, ...tensor_shape] - stacked tensor shards
 * - Output: [...tensor_shape] - reduced result
 */

#include <metal_stdlib>
using namespace metal;

// Reduction operation types
enum class ReduceOp : uint8_t {
    SUM = 0,
    MEAN = 1,
    MAX = 2,
    MIN = 3
};

/**
 * All-reduce kernel for fp16/fp32 tensors.
 *
 * Reduces num_shards tensors into a single output tensor using the
 * specified reduction operation.
 *
 * Note: Metal does not have native bfloat16 support. Use half (fp16)
 * or float (fp32) instead.
 *
 * Template parameters:
 *   T - element type (half or float)
 *
 * Arguments:
 *   input: Input tensors stacked along first dimension [num_shards, numel]
 *   output: Output tensor [numel]
 *   num_shards: Number of tensor shards to reduce
 *   numel: Number of elements per tensor
 *   op: Reduction operation (0=sum, 1=mean, 2=max, 3=min)
 */
template <typename T>
[[kernel]] void all_reduce_kernel(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& numel [[buffer(3)]],
    constant uint8_t& op [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) return;

    float acc;
    ReduceOp reduce_op = static_cast<ReduceOp>(op);

    // Initialize accumulator based on operation
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc = 0.0f;
            break;
        case ReduceOp::MAX:
            acc = -INFINITY;
            break;
        case ReduceOp::MIN:
            acc = INFINITY;
            break;
    }

    // Reduce across shards
    for (uint32_t s = 0; s < num_shards; s++) {
        float val = float(input[s * numel + gid]);

        switch (reduce_op) {
            case ReduceOp::SUM:
            case ReduceOp::MEAN:
                acc += val;
                break;
            case ReduceOp::MAX:
                acc = max(acc, val);
                break;
            case ReduceOp::MIN:
                acc = min(acc, val);
                break;
        }
    }

    // Finalize
    if (reduce_op == ReduceOp::MEAN) {
        acc /= float(num_shards);
    }

    output[gid] = T(acc);
}

// Explicit instantiations
template [[host_name("all_reduce_half")]]
[[kernel]] void all_reduce_kernel<half>(
    device const half*,
    device half*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint8_t&,
    uint
);

template [[host_name("all_reduce_float")]]
[[kernel]] void all_reduce_kernel<float>(
    device const float*,
    device float*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint8_t&,
    uint
);

/**
 * Simdgroup-optimized all-reduce for small reductions.
 *
 * Uses simdgroup shuffle operations for efficient reduction within
 * a single simdgroup (32 threads on Apple Silicon).
 *
 * Best for: Small tensor reductions where num_shards <= 32
 */
template <typename T>
[[kernel]] void all_reduce_simd_kernel(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& numel [[buffer(3)]],
    constant uint8_t& op [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (gid >= numel) return;

    float acc;
    ReduceOp reduce_op = static_cast<ReduceOp>(op);

    // Each thread loads one shard's element
    float val = 0.0f;
    if (simd_lane < num_shards) {
        val = float(input[simd_lane * numel + gid]);
    }

    // Initialize accumulator
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc = val;
            break;
        case ReduceOp::MAX:
            acc = simd_lane < num_shards ? val : -INFINITY;
            break;
        case ReduceOp::MIN:
            acc = simd_lane < num_shards ? val : INFINITY;
            break;
    }

    // Simdgroup reduction
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc = simd_sum(acc);
            break;
        case ReduceOp::MAX:
            acc = simd_max(acc);
            break;
        case ReduceOp::MIN:
            acc = simd_min(acc);
            break;
    }

    // Finalize and write (only lane 0 writes)
    if (simd_lane == 0) {
        if (reduce_op == ReduceOp::MEAN) {
            acc /= float(num_shards);
        }
        output[gid] = T(acc);
    }
}

template [[host_name("all_reduce_simd_half")]]
[[kernel]] void all_reduce_simd_kernel<half>(
    device const half*,
    device half*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint8_t&,
    uint,
    uint
);

template [[host_name("all_reduce_simd_float")]]
[[kernel]] void all_reduce_simd_kernel<float>(
    device const float*,
    device float*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint8_t&,
    uint,
    uint
);

/**
 * Scatter kernel: Split tensor along a dimension.
 *
 * Takes a single tensor and scatters it to multiple output buffers,
 * splitting along the first dimension.
 *
 * Input: [total_size, ...rest]
 * Outputs: num_shards buffers, each [shard_size, ...rest]
 */
template <typename T>
[[kernel]] void scatter_kernel(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],  // Flat buffer for all shards
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& shard_size [[buffer(3)]],
    constant uint32_t& stride [[buffer(4)]],  // Elements per row
    uint2 gid [[thread_position_in_grid]]  // (shard_idx, element_idx)
) {
    uint32_t shard_idx = gid.x;
    uint32_t elem_idx = gid.y;

    if (shard_idx >= num_shards || elem_idx >= shard_size * stride) return;

    uint32_t input_idx = shard_idx * shard_size * stride + elem_idx;
    uint32_t output_idx = shard_idx * shard_size * stride + elem_idx;

    output[output_idx] = input[input_idx];
}

template [[host_name("scatter_half")]]
[[kernel]] void scatter_kernel<half>(
    device const half*,
    device half*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    uint2
);

template [[host_name("scatter_float")]]
[[kernel]] void scatter_kernel<float>(
    device const float*,
    device float*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    uint2
);

/**
 * All-gather kernel: Concatenate shards along a dimension.
 *
 * Takes multiple input shards and concatenates them into a single
 * output tensor along the first dimension.
 *
 * Inputs: num_shards buffers, each [shard_size, ...rest]
 * Output: [total_size, ...rest] where total_size = num_shards * shard_size
 */
template <typename T>
[[kernel]] void all_gather_kernel(
    device const T* input [[buffer(0)]],  // Flat buffer with all shards
    device T* output [[buffer(1)]],
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& shard_size [[buffer(3)]],
    constant uint32_t& stride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]  // (shard_idx, element_idx)
) {
    uint32_t shard_idx = gid.x;
    uint32_t elem_idx = gid.y;

    if (shard_idx >= num_shards || elem_idx >= shard_size * stride) return;

    uint32_t input_idx = shard_idx * shard_size * stride + elem_idx;
    uint32_t output_idx = shard_idx * shard_size * stride + elem_idx;

    output[output_idx] = input[input_idx];
}

template [[host_name("all_gather_half")]]
[[kernel]] void all_gather_kernel<half>(
    device const half*,
    device half*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    uint2
);

template [[host_name("all_gather_float")]]
[[kernel]] void all_gather_kernel<float>(
    device const float*,
    device float*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    uint2
);

/**
 * Reduce-scatter: Reduce then scatter in a single pass.
 *
 * This is more efficient than separate reduce + scatter operations
 * as it avoids materializing the full reduced tensor.
 *
 * Input: [num_shards, total_size]
 * Output: [num_shards, shard_size] - each shard gets a portion
 */
template <typename T>
[[kernel]] void reduce_scatter_kernel(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& total_size [[buffer(3)]],
    constant uint8_t& op [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]  // (shard_idx, local_elem_idx)
) {
    uint32_t shard_idx = gid.x;
    uint32_t local_idx = gid.y;
    uint32_t shard_size = total_size / num_shards;

    if (shard_idx >= num_shards || local_idx >= shard_size) return;

    // Global index for this element
    uint32_t global_idx = shard_idx * shard_size + local_idx;

    float acc;
    ReduceOp reduce_op = static_cast<ReduceOp>(op);

    // Initialize
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc = 0.0f;
            break;
        case ReduceOp::MAX:
            acc = -INFINITY;
            break;
        case ReduceOp::MIN:
            acc = INFINITY;
            break;
    }

    // Reduce across all shards for this element
    for (uint32_t s = 0; s < num_shards; s++) {
        float val = float(input[s * total_size + global_idx]);

        switch (reduce_op) {
            case ReduceOp::SUM:
            case ReduceOp::MEAN:
                acc += val;
                break;
            case ReduceOp::MAX:
                acc = max(acc, val);
                break;
            case ReduceOp::MIN:
                acc = min(acc, val);
                break;
        }
    }

    // Finalize
    if (reduce_op == ReduceOp::MEAN) {
        acc /= float(num_shards);
    }

    // Write to this shard's output buffer
    output[shard_idx * shard_size + local_idx] = T(acc);
}

template [[host_name("reduce_scatter_half")]]
[[kernel]] void reduce_scatter_kernel<half>(
    device const half*,
    device half*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint8_t&,
    uint2
);

template [[host_name("reduce_scatter_float")]]
[[kernel]] void reduce_scatter_kernel<float>(
    device const float*,
    device float*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint8_t&,
    uint2
);
