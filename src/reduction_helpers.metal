#ifndef REDUCTION_HELPERS_METAL
#define REDUCTION_HELPERS_METAL

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// SIMDgroup-level reduction intrinsics
//
// These use Metal's simd_sum/simd_max intrinsics for fast within-SIMD
// reductions. Each simdgroup (32 threads on Apple Silicon) performs
// a hardware-accelerated reduction in a single instruction.
//
// Performance: O(1) within simdgroup vs O(log N) tree reduction
// ---------------------------------------------------------------------------

// Fast simdgroup sum using hardware intrinsics
inline float simd_sum_reduction(float val) {
    return simd_sum(val);
}

// Fast simdgroup max using hardware intrinsics
inline float simd_max_reduction(float val) {
    return simd_max(val);
}

// ---------------------------------------------------------------------------
// Threadgroup-wide reductions using SIMDgroup intrinsics
//
// Two-phase approach:
// 1. Each simdgroup reduces locally using simd_sum/simd_max
// 2. Simdgroup leaders reduce across simdgroups
//
// For 128-thread threadgroups (4 simdgroups), this requires:
// - Phase 1: 4 parallel simd_sum/simd_max operations (hardware accelerated)
// - Phase 2: 1 simd_sum/simd_max over 4 values
//
// vs old tree reduction:
// - 5 barrier-synchronized steps with conditionals
// ---------------------------------------------------------------------------

// Number of simdgroups per threadgroup (assumes 128 threads = 4 simdgroups)
constant constexpr uint SIMDGROUPS_PER_TG = 4;

// Threadgroup reduction: max (SIMDgroup-optimized)
inline float threadgroup_reduce_max(
    float thread_value,
    uint tid,
    threadgroup float* scratch
) {
    // Phase 1: SIMDgroup-local max (hardware accelerated)
    float sg_max = simd_max(thread_value);

    uint sg_id = tid / 32;
    uint lane = tid % 32;

    // Leader writes simdgroup result
    if (lane == 0) {
        scratch[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Reduce across simdgroups (only first simdgroup participates)
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_PER_TG) ? scratch[lane] : -INFINITY;
        float result = simd_max(v);
        if (lane == 0) {
            scratch[0] = result;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return scratch[0];
}

// Threadgroup reduction: sum (SIMDgroup-optimized)
inline float threadgroup_reduce_sum(
    float thread_value,
    uint tid,
    threadgroup float* scratch
) {
    // Phase 1: SIMDgroup-local sum (hardware accelerated)
    float sg_sum = simd_sum(thread_value);

    uint sg_id = tid / 32;
    uint lane = tid % 32;

    // Leader writes simdgroup result
    if (lane == 0) {
        scratch[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Reduce across simdgroups (only first simdgroup participates)
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_PER_TG) ? scratch[lane] : 0.0f;
        float result = simd_sum(v);
        if (lane == 0) {
            scratch[0] = result;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return scratch[0];
}

// Tree attention variant (for tree_attention.metal)
inline float threadgroup_reduce_max_tree(
    float thread_value,
    uint tid,
    threadgroup float* scratch
) {
    // Now uses SIMDgroup-optimized implementation
    return threadgroup_reduce_max(thread_value, tid, scratch);
}

// ---------------------------------------------------------------------------
// Macro versions (SIMDgroup-optimized for compatibility)
// ---------------------------------------------------------------------------

#define THREADGROUP_REDUCE_MAX(value, tid, scratch, result) \
    do { \
        float _sg_max = simd_max(value); \
        uint _sg_id = (tid) / 32; \
        uint _lane = (tid) % 32; \
        if (_lane == 0) { \
            scratch[_sg_id] = _sg_max; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (_sg_id == 0) { \
            float _v = (_lane < SIMDGROUPS_PER_TG) ? scratch[_lane] : -INFINITY; \
            float _r = simd_max(_v); \
            if (_lane == 0) { \
                scratch[0] = _r; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        result = scratch[0]; \
    } while(0)

#define THREADGROUP_REDUCE_SUM(value, tid, scratch, result) \
    do { \
        float _sg_sum = simd_sum(value); \
        uint _sg_id = (tid) / 32; \
        uint _lane = (tid) % 32; \
        if (_lane == 0) { \
            scratch[_sg_id] = _sg_sum; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (_sg_id == 0) { \
            float _v = (_lane < SIMDGROUPS_PER_TG) ? scratch[_lane] : 0.0f; \
            float _r = simd_sum(_v); \
            if (_lane == 0) { \
                scratch[0] = _r; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        result = scratch[0]; \
    } while(0)

#endif // REDUCTION_HELPERS_METAL
