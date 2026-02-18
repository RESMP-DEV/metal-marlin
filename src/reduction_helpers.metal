#ifndef REDUCTION_HELPERS_METAL
#define REDUCTION_HELPERS_METAL

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Bit manipulation helpers
// ---------------------------------------------------------------------------

// Count trailing zeros - returns the number of consecutive 0 bits starting
// from the least significant bit (position 0).
// Used by block-sparse attention kernels to iterate over set bits in masks.
inline uint ctz(uint64_t x) {
    // Metal 3.0+ has built-in ctz, but for compatibility we provide a fallback
    // implementation using bit manipulation.
    if (x == 0) return 64;
    
    // Count trailing zeros using bit manipulation
    // This is a classic algorithm that works on all Metal versions
    uint count = 0;
    
    // Check lower 32 bits first
    uint32_t low = uint32_t(x);
    if (low == 0) {
        count += 32;
        low = uint32_t(x >> 32);
    }
    
    // Now count trailing zeros in the 32-bit value
    // Use binary search approach
    if ((low & 0x0000FFFF) == 0) { count += 16; low >>= 16; }
    if ((low & 0x000000FF) == 0) { count += 8;  low >>= 8;  }
    if ((low & 0x0000000F) == 0) { count += 4;  low >>= 4;  }
    if ((low & 0x00000003) == 0) { count += 2;  low >>= 2;  }
    if ((low & 0x00000001) == 0) { count += 1; }
    
    return count;
}

// ---------------------------------------------------------------------------
// SIMDgroup-level reduction intrinsics
//
// These use Metal's simd_sum/simd_max intrinsics for fast within-SIMD
// reductions. Each simdgroup (32 threads on Apple Silicon) performs
// a hardware-accelerated reduction in a single instruction.
//
// Performance: O(1) within simdgroup vs O(log N) tree reduction
// ---------------------------------------------------------------------------

// Fast simdgroup sum using hardware intrinsics (single instruction)
inline float simd_sum_reduction(float val) {
    return simd_sum(val);
}

// Fast simdgroup max using hardware intrinsics (single instruction)
inline float simd_max_reduction(float val) {
    return simd_max(val);
}

// ---------------------------------------------------------------------------
// SIMDgroup-level normalization helpers for Attention
//
// These functions provide fast normalization for attention softmax,
// using hardware-accelerated simd_sum for the reduction.
//
// Usage in attention kernels:
//   float max_score = simd_max_reduction(thread_max);
//   float sum_exp = simd_sum_reduction(exp(score - max_score));
//   float prob = exp(score - max_score) / sum_exp;
// ---------------------------------------------------------------------------

// Fast simdgroup softmax normalization
// Returns normalized probability for this thread's score
inline float simdgroup_softmax_normalize(float score, float max_score) {
    float exp_val = exp(score - max_score);
    float sum_exp = simd_sum(exp_val);
    return exp_val / sum_exp;
}

// Online softmax update with simd_sum for faster reduction
// Updates running max and sum with new score values
inline void simdgroup_online_softmax_update(
    float score,
    thread float& running_max,
    thread float& running_sum
) {
    float new_max = max(running_max, score);
    running_sum = running_sum * exp(running_max - new_max) + exp(score - new_max);
    running_max = new_max;
}

// Reduce sum across all threads in a simdgroup using simd_sum
inline float simdgroup_reduce_sum(float val) {
    return simd_sum(val);
}

// Reduce max across all threads in a simdgroup using simd_max  
inline float simdgroup_reduce_max(float val) {
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
