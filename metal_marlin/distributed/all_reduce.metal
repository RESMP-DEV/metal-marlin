/**
 * All-reduce kernel for Metal.
 *
 * This file mirrors the kernel in contrib/metal_marlin/src/all_reduce.metal
 * so distributed code can reference a local kernel definition when needed.
 *
 * The kernel implements in-device reduction operations for tensor parallelism.
 * For multi-GPU scenarios on Apple Silicon (future M-series multi-die chips),
 * this kernel handles local reduction before inter-device communication.
 *
 * Supported operations:
 * - sum: Element-wise sum reduction
 * - mean: Element-wise mean reduction
 * - max: Element-wise maximum
 * - min: Element-wise minimum
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
 * All-reduce kernel for bf16/fp16 tensors.
 *
 * Reduces num_shards tensors into a single output tensor using the
 * specified reduction operation.
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
void all_reduce_kernel(
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

/**
 * Simdgroup-optimized all-reduce for small reductions.
 *
 * Uses simdgroup shuffle operations for efficient reduction within
 * a single simdgroup (32 threads on Apple Silicon).
 *
 * Best for: Small tensor reductions where num_shards <= 32
 */
template <typename T>
void all_reduce_simd_kernel(
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

// Kernel Entry Points
// ---------------------------------------------------------------------------

[[kernel]] void all_reduce_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& numel [[buffer(3)]],
    constant uint8_t& op [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    all_reduce_kernel<half>(input, output, num_shards, numel, op, gid);
}

[[kernel]] void all_reduce_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& numel [[buffer(3)]],
    constant uint8_t& op [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    all_reduce_kernel<float>(input, output, num_shards, numel, op, gid);
}

[[kernel]] void all_reduce_simd_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint32_t& num_shards [[buffer(2)]],
    constant uint32_t& numel [[buffer(3)]],
    constant uint8_t& op [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    all_reduce_simd_kernel<half>(input, output, num_shards, numel, op, gid, simd_lane);
}


// Note: bfloat16 not supported in Metal 3.0. Using half instead.
// For bfloat16 support, use Metal 3.1+ (macOS 14+) with bfloat16_t.
