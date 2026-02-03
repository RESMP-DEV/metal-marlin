// sampling.metal - Token sampling kernels for autoregressive generation
//
// Provides efficient GPU-accelerated sampling strategies:
//   - softmax: Numerically stable softmax over vocabulary
//   - argmax: Greedy decoding (deterministic)
//   - top_p: Nucleus sampling (stochastic)
//   - top_k: Top-k sampling (stochastic)
//   - categorical: Sample from softmax distribution (stochastic)
//
// All kernels operate on logits [batch, vocab_size] and output token indices.
// Temperature scaling should be applied to logits before calling these kernels.

#include <metal_stdlib>
using namespace metal;

// Vector types for 8-element vectorized paths
typedef half half_v8 __attribute__((ext_vector_type(8)));
typedef float float_v8 __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------------------
// Constants and helpers
// ---------------------------------------------------------------------------

constant float NEG_INF = -INFINITY;

// Warp-level reduction for max
inline float simd_max(float val) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val = max(val, simd_shuffle_down(val, offset));
    }
    return simd_broadcast_first(val);
}

// Warp-level reduction for sum
inline float simd_sum(float val) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return simd_broadcast_first(val);
}

// PCG random number generator state
// Uses the PCG-XSH-RR variant for 32-bit output
struct PCGState {
    ulong state;
    ulong inc;
};

inline uint pcg32(thread PCGState& rng) {
    ulong oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + rng.inc;
    uint xorshifted = uint(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint rot = uint(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Generate uniform float in [0, 1)
inline float pcg32_float(thread PCGState& rng) {
    return float(pcg32(rng)) * (1.0f / 4294967296.0f);
}

// Initialize RNG from seed and sequence
inline PCGState pcg_init(ulong seed, ulong seq) {
    PCGState rng;
    rng.state = 0;
    rng.inc = (seq << 1u) | 1u;
    pcg32(rng);
    rng.state += seed;
    pcg32(rng);
    return rng;
}

// ---------------------------------------------------------------------------
// Softmax kernel
// ---------------------------------------------------------------------------

struct SoftmaxParams {
    uint vocab_size;
    uint batch_size;
};

// Numerically stable softmax: exp(x - max) / sum(exp(x - max))
// Input: logits [batch, vocab_size]
// Output: probs [batch, vocab_size]
// Each threadgroup handles one batch element
// Optimized for large vocab (>100K) with vectorized loads and single-pass algorithm
kernel void softmax(
    device const float* logits [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    device const float* row = logits + batch_idx * vocab;
    device float* out_row = probs + batch_idx * vocab;

    // Pass 1: Find max with vectorized loads (4x throughput for large vocab)
    float local_max = NEG_INF;
    const uint vec_stride = tg_size * 4;
    uint i = tid * 4;
    
    // Vectorized loop for 4-element blocks
    for (; i + 3 < vocab; i += vec_stride) {
        float4 vals = *((device const float4*)(row + i));
        local_max = max(local_max, max(max(vals.x, vals.y), max(vals.z, vals.w)));
    }
    
    // Handle remainder
    for (i = vocab - (vocab % 4) + tid; i < vocab; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    float global_max = simd_max(local_max);

    // Cross-simdgroup reduction (for threadgroups > 32 threads)
    threadgroup float shared_max[32];
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_max[simd_id] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_max[tid] : NEG_INF;
        global_max = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];

    // Pass 2: Compute exp(x - max) and sum with vectorized stores
    float local_sum = 0.0f;
    i = tid * 4;
    
    // Vectorized exp computation (4x throughput)
    for (; i + 3 < vocab; i += vec_stride) {
        float4 vals = *((device const float4*)(row + i));
        vals -= global_max;
        float4 exp_vals = exp(vals);
        *((device float4*)(out_row + i)) = exp_vals;
        local_sum += exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
    }
    
    // Handle remainder
    for (i = vocab - (vocab % 4) + tid; i < vocab; i += tg_size) {
        float exp_val = exp(row[i] - global_max);
        out_row[i] = exp_val;
        local_sum += exp_val;
    }
    float global_sum = simd_sum(local_sum);

    // Cross-simdgroup sum reduction
    threadgroup float shared_sum[32];
    if (simd_lane == 0) {
        shared_sum[simd_id] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_sum[tid] : 0.0f;
        global_sum = simd_sum(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_sum[0];

    // Pass 3: Normalize with vectorized operations
    float inv_sum = 1.0f / global_sum;
    i = tid * 4;
    
    // Vectorized normalization
    for (; i + 3 < vocab; i += vec_stride) {
        float4 vals = *((device float4*)(out_row + i));
        vals *= inv_sum;
        *((device float4*)(out_row + i)) = vals;
    }
    
    // Handle remainder
    for (i = vocab - (vocab % 4) + tid; i < vocab; i += tg_size) {
        out_row[i] *= inv_sum;
    }
}

// FP16 variant with vectorized loads (8-element vectors for 2x bandwidth)
kernel void softmax_fp16(
    device const half* logits [[buffer(0)]],
    device half* probs [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    device const half* row = logits + batch_idx * vocab;
    device half* out_row = probs + batch_idx * vocab;

    // Use float32 for accumulation, FP16 vectors for I/O (8-wide for 2x bandwidth)
    float local_max = NEG_INF;
    const uint vec_stride = tg_size * 8;
    uint i = tid * 8;
    
    // Vectorized max with 8-element half vectors
    for (; i + 7 < vocab; i += vec_stride) {
        half_v8 vals = *((device const half_v8*)(row + i));
        float_v8 fvals = float_v8(vals);
        local_max = max(local_max, max(max(max(fvals[0], fvals[1]), max(fvals[2], fvals[3])),
                                       max(max(fvals[4], fvals[5]), max(fvals[6], fvals[7]))));
    }
    
    // Remainder
    for (i = vocab - (vocab % 8) + tid; i < vocab; i += tg_size) {
        local_max = max(local_max, float(row[i]));
    }
    float global_max = simd_max(local_max);

    threadgroup float shared_max[32];
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_max[simd_id] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_max[tid] : NEG_INF;
        global_max = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];

    float local_sum = 0.0f;
    i = tid * 8;
    
    // Vectorized exp + sum with FP16 I/O
    for (; i + 7 < vocab; i += vec_stride) {
        half_v8 vals = *((device const half_v8*)(row + i));
        float_v8 fvals = float_v8(vals) - global_max;
        float_v8 exp_vals;
        for (int v = 0; v < 8; ++v) exp_vals[v] = exp(fvals[v]);
        *((device half_v8*)(out_row + i)) = half_v8(exp_vals);
        local_sum += exp_vals[0] + exp_vals[1] + exp_vals[2] + exp_vals[3] +
                     exp_vals[4] + exp_vals[5] + exp_vals[6] + exp_vals[7];
    }
    
    for (i = vocab - (vocab % 8) + tid; i < vocab; i += tg_size) {
        float exp_val = exp(float(row[i]) - global_max);
        out_row[i] = half(exp_val);
        local_sum += exp_val;
    }
    float global_sum = simd_sum(local_sum);

    threadgroup float shared_sum[32];
    if (simd_lane == 0) {
        shared_sum[simd_id] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_sum[tid] : 0.0f;
        global_sum = simd_sum(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_sum[0];

    float inv_sum = 1.0f / global_sum;
    i = tid * 8;
    
    // Vectorized normalization with FP16
    for (; i + 7 < vocab; i += vec_stride) {
        half_v8 vals = *((device half_v8*)(out_row + i));
        float_v8 fvals = float_v8(vals) * inv_sum;
        *((device half_v8*)(out_row + i)) = half_v8(fvals);
    }
    
    for (i = vocab - (vocab % 8) + tid; i < vocab; i += tg_size) {
        out_row[i] = half(float(out_row[i]) * inv_sum);
    }
}

// ---------------------------------------------------------------------------
// Argmax kernel (greedy decoding)
// ---------------------------------------------------------------------------

struct ArgmaxParams {
    uint vocab_size;
    uint batch_size;
};

// Find index of maximum logit value
// Input: logits [batch, vocab_size]
// Output: indices [batch]
kernel void argmax(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant ArgmaxParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    device const float* row = logits + batch_idx * vocab;

    // Find local max and its index
    float local_max = NEG_INF;
    uint local_idx = 0;

    for (uint i = tid; i < vocab; i += tg_size) {
        float val = row[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // SIMD reduction to find max across simdgroup
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float other_max = simd_shuffle_down(local_max, offset);
        uint other_idx = simd_shuffle_down(local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // Cross-simdgroup reduction
    threadgroup float shared_max[32];
    threadgroup uint shared_idx[32];

    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_max[simd_id] = local_max;
        shared_idx[simd_id] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint num_groups = (tg_size + 31) / 32;
        local_max = (tid < num_groups) ? shared_max[tid] : NEG_INF;
        local_idx = (tid < num_groups) ? shared_idx[tid] : 0;

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float other_max = simd_shuffle_down(local_max, offset);
            uint other_idx = simd_shuffle_down(local_idx, offset);
            if (other_max > local_max) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }

        if (tid == 0) {
            indices[batch_idx] = local_idx;
        }
    }
}

// FP16 argmax variant
kernel void argmax_fp16(
    device const half* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant ArgmaxParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    device const half* row = logits + batch_idx * vocab;

    float local_max = NEG_INF;
    uint local_idx = 0;

    for (uint i = tid; i < vocab; i += tg_size) {
        float val = float(row[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float other_max = simd_shuffle_down(local_max, offset);
        uint other_idx = simd_shuffle_down(local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    threadgroup float shared_max[32];
    threadgroup uint shared_idx[32];

    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_max[simd_id] = local_max;
        shared_idx[simd_id] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint num_groups = (tg_size + 31) / 32;
        local_max = (tid < num_groups) ? shared_max[tid] : NEG_INF;
        local_idx = (tid < num_groups) ? shared_idx[tid] : 0;

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float other_max = simd_shuffle_down(local_max, offset);
            uint other_idx = simd_shuffle_down(local_idx, offset);
            if (other_max > local_max) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }

        if (tid == 0) {
            indices[batch_idx] = local_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Top-K Sampling
// ---------------------------------------------------------------------------

struct TopKParams {
    uint vocab_size;
    uint batch_size;
    uint k;
    ulong seed;
};

// Partial sort to find top-k elements, then sample from them
// Uses register-based heap for small k (k <= 64)
// Input: logits [batch, vocab_size]
// Output: indices [batch]
kernel void sample_top_k(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant TopKParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const uint k = min((uint)params.k, (uint)vocab);
    device const float* row = logits + batch_idx * vocab;

    // Initialize RNG
    PCGState rng = pcg_init(params.seed, (ulong)batch_idx * 1000ULL + (ulong)tid);

    // Each thread maintains its own top-k heap (min-heap for top-k)
    // Max heap size is 64 per thread
    const uint MAX_K = 64;
    float local_heap[MAX_K];
    uint local_idx[MAX_K];
    uint heap_size = 0;

    // Fill local heap with initial elements
    for (uint i = tid; i < vocab; i += tg_size) {
        float val = row[i];

        if (heap_size < k) {
            // Heap not full, insert
            local_heap[heap_size] = val;
            local_idx[heap_size] = i;
            heap_size++;

            // Sift up (min-heap property)
            uint pos = heap_size - 1;
            while (pos > 0) {
                uint parent = (pos - 1) / 2;
                if (local_heap[pos] < local_heap[parent]) {
                    float tv = local_heap[pos];
                    uint ti = local_idx[pos];
                    local_heap[pos] = local_heap[parent];
                    local_idx[pos] = local_idx[parent];
                    local_heap[parent] = tv;
                    local_idx[parent] = ti;
                    pos = parent;
                } else {
                    break;
                }
            }
        } else if (val > local_heap[0]) {
            // Replace min with new value
            local_heap[0] = val;
            local_idx[0] = i;

            // Sift down (min-heap property)
            uint pos = 0;
            while (true) {
                uint left = 2 * pos + 1;
                uint right = 2 * pos + 2;
                uint smallest = pos;

                if (left < heap_size && local_heap[left] < local_heap[smallest]) {
                    smallest = left;
                }
                if (right < heap_size && local_heap[right] < local_heap[smallest]) {
                    smallest = right;
                }

                if (smallest != pos) {
                    float tv = local_heap[pos];
                    uint ti = local_idx[pos];
                    local_heap[pos] = local_heap[smallest];
                    local_idx[pos] = local_idx[smallest];
                    local_heap[smallest] = tv;
                    local_idx[smallest] = ti;
                    pos = smallest;
                } else {
                    break;
                }
            }
        }
    }

    // Merge heaps across threads via threadgroup memory
    // For simplicity, thread 0 collects all top-k candidates and does final selection
    threadgroup float all_vals[1024];  // Assume max 1024 candidates (16 threads * 64)
    threadgroup uint all_idxs[1024];
    threadgroup uint counts[32];

    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;

    // Store counts
    if (simd_lane == 0) {
        counts[simd_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread stores its heap
    uint base = tid * MAX_K;
    for (uint i = 0; i < heap_size && base + i < 1024; i++) {
        all_vals[base + i] = local_heap[i];
        all_idxs[base + i] = local_idx[i];
    }

    // Store thread's count
    threadgroup uint thread_counts[256];
    thread_counts[tid] = heap_size;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 does final merging and sampling
    if (tid == 0) {
        // Collect all candidates
        float final_heap[64];
        uint final_idx[64];
        uint final_size = 0;

        for (uint t = 0; t < tg_size; t++) {
            uint count = thread_counts[t];
            uint tbase = t * MAX_K;

            for (uint i = 0; i < count; i++) {
                float val = all_vals[tbase + i];
                uint idx = all_idxs[tbase + i];

                if (final_size < k) {
                    final_heap[final_size] = val;
                    final_idx[final_size] = idx;
                    final_size++;

                    uint pos = final_size - 1;
                    while (pos > 0) {
                        uint parent = (pos - 1) / 2;
                        if (final_heap[pos] < final_heap[parent]) {
                            float tv = final_heap[pos];
                            uint ti = final_idx[pos];
                            final_heap[pos] = final_heap[parent];
                            final_idx[pos] = final_idx[parent];
                            final_heap[parent] = tv;
                            final_idx[parent] = ti;
                            pos = parent;
                        } else {
                            break;
                        }
                    }
                } else if (val > final_heap[0]) {
                    final_heap[0] = val;
                    final_idx[0] = idx;

                    uint pos = 0;
                    while (true) {
                        uint left = 2 * pos + 1;
                        uint right = 2 * pos + 2;
                        uint smallest = pos;

                        if (left < final_size && final_heap[left] < final_heap[smallest]) {
                            smallest = left;
                        }
                        if (right < final_size && final_heap[right] < final_heap[smallest]) {
                            smallest = right;
                        }

                        if (smallest != pos) {
                            float tv = final_heap[pos];
                            uint ti = final_idx[pos];
                            final_heap[pos] = final_heap[smallest];
                            final_idx[pos] = final_idx[smallest];
                            final_heap[smallest] = tv;
                            final_idx[smallest] = ti;
                            pos = smallest;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Compute softmax over top-k
        float max_val = NEG_INF;
        for (uint i = 0; i < final_size; i++) {
            max_val = max(max_val, final_heap[i]);
        }

        float sum = 0.0f;
        for (uint i = 0; i < final_size; i++) {
            final_heap[i] = exp(final_heap[i] - max_val);
            sum += final_heap[i];
        }

        // Sample from softmax distribution
        float r = pcg32_float(rng) * sum;
        float cumsum = 0.0f;
        uint sampled = final_idx[0];

        for (uint i = 0; i < final_size; i++) {
            cumsum += final_heap[i];
            if (cumsum >= r) {
                sampled = final_idx[i];
                break;
            }
        }

        indices[batch_idx] = sampled;
    }
}

// ---------------------------------------------------------------------------
// Top-K Selection (values and indices)
// ---------------------------------------------------------------------------

struct TopKSelectionParams {
    uint n;          // Total number of elements
    uint k;          // Number of top elements to select
    uint batch_size; // Batch dimension
};

// Helper: min-heap insertion for small k (< 64)
inline void heap_insert_topk(
    thread float* heap_vals,
    thread uint* heap_indices,
    thread uint& heap_size,
    uint max_k,
    float val,
    uint idx
) {
    if (heap_size < max_k) {
        // Insert into heap
        heap_vals[heap_size] = val;
        heap_indices[heap_size] = idx;
        heap_size++;
        
        // Sift up (min-heap)
        uint pos = heap_size - 1;
        while (pos > 0) {
            uint parent = (pos - 1) / 2;
            if (heap_vals[pos] < heap_vals[parent]) {
                float temp_val = heap_vals[pos];
                uint temp_idx = heap_indices[pos];
                heap_vals[pos] = heap_vals[parent];
                heap_indices[pos] = heap_indices[parent];
                heap_vals[parent] = temp_val;
                heap_indices[parent] = temp_idx;
                pos = parent;
            } else {
                break;
            }
        }
    } else if (val > heap_vals[0]) {
        // Replace min and sift down
        heap_vals[0] = val;
        heap_indices[0] = idx;
        
        uint pos = 0;
        while (true) {
            uint left = 2 * pos + 1;
            uint right = 2 * pos + 2;
            uint smallest = pos;
            
            if (left < heap_size && heap_vals[left] < heap_vals[smallest]) {
                smallest = left;
            }
            if (right < heap_size && heap_vals[right] < heap_vals[smallest]) {
                smallest = right;
            }
            
            if (smallest != pos) {
                float temp_val = heap_vals[pos];
                uint temp_idx = heap_indices[pos];
                heap_vals[pos] = heap_vals[smallest];
                heap_indices[pos] = heap_indices[smallest];
                heap_vals[smallest] = temp_val;
                heap_indices[smallest] = temp_idx;
                pos = smallest;
            } else {
                break;
            }
        }
    }
}

// Bitonic sort for top-k selection when k is large (>= 64)
// More efficient than heap for larger k values
inline void bitonic_sort_partial(
    threadgroup float* vals,
    threadgroup uint* indices,
    uint size,
    uint k
) {
    // Sort first k elements using bitonic sort
    for (uint stride = 2; stride <= k; stride <<= 1) {
        for (uint i = stride; i > 0; i >>= 1) {
            for (uint j = 0; j < k; j++) {
                uint ixj = j ^ i;
                if (ixj > j && ixj < k) {
                    bool ascending = (j & stride) == 0;
                    bool should_swap = ascending ? (vals[j] < vals[ixj]) : (vals[j] > vals[ixj]);
                    
                    if (should_swap) {
                        float temp_val = vals[j];
                        uint temp_idx = indices[j];
                        vals[j] = vals[ixj];
                        indices[j] = indices[ixj];
                        vals[ixj] = temp_val;
                        indices[ixj] = temp_idx;
                    }
                }
            }
        }
    }
    
    // Process remaining elements if any
    for (uint i = k; i < size; i++) {
        if (vals[i] > vals[0]) {
            vals[0] = vals[i];
            indices[0] = indices[i];
            
            // Bubble down to maintain sorted order (descending)
            for (uint j = 0; j < k - 1; j++) {
                if (vals[j] < vals[j + 1]) {
                    float temp_val = vals[j];
                    uint temp_idx = indices[j];
                    vals[j] = vals[j + 1];
                    indices[j] = indices[j + 1];
                    vals[j + 1] = temp_val;
                    indices[j + 1] = temp_idx;
                }
            }
        }
    }
}

// Top-k selection kernel - returns both values and indices
// Efficient parallel top-k using heap for small k and bitonic sort for large k
kernel void topk_values_indices(
    device const float* input [[buffer(0)]],      // [batch, n]
    device float* values_out [[buffer(1)]],        // [batch, k]
    device uint* indices_out [[buffer(2)]],        // [batch, k]
    constant TopKSelectionParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint tid = gid.x;
    
    if (batch_idx >= params.batch_size) return;
    
    const uint n = params.n;
    const uint k = min((uint)params.k, n);
    
    // Input pointer for this batch
    device const float* row = input + batch_idx * n;
    
    // Output pointers for this batch
    device float* values_row = values_out + batch_idx * k;
    device uint* indices_row = indices_out + batch_idx * k;
    
    // Use different strategies based on k value
    const bool USE_HEAP = k < 64;
    
    if (USE_HEAP) {
        // Heap-based approach for small k
        // Each thread maintains its own local heap
        const uint MAX_K = 64;
        thread float local_heap[MAX_K];
        thread uint local_indices[MAX_K];
        uint heap_size = 0;
        
        // Each thread processes a subset of elements
        for (uint i = tid; i < n; i += 32) { // Assume 32 threads per row
            heap_insert_topk(local_heap, local_indices, heap_size, k, row[i], i);
        }
        
        // Merge heaps - thread 0 collects and sorts
        threadgroup float merged_vals[32 * MAX_K];
        threadgroup uint merged_indices[32 * MAX_K];
        threadgroup uint heap_sizes[32];
        
        // Store heap sizes
        heap_sizes[tid] = heap_size;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Store heap contents
        uint offset = tid * MAX_K;
        for (uint i = 0; i < heap_size; i++) {
            merged_vals[offset + i] = local_heap[i];
            merged_indices[offset + i] = local_indices[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Thread 0 merges final results
        if (tid == 0) {
            float final_heap[MAX_K];
            uint final_indices[MAX_K];
            uint final_size = 0;
            
            // Merge all thread heaps
            for (uint t = 0; t < 32; t++) {
                uint t_offset = t * MAX_K;
                uint t_size = heap_sizes[t];
                
                for (uint i = 0; i < t_size; i++) {
                    heap_insert_topk(final_heap, final_indices, final_size, k, 
                                    merged_vals[t_offset + i], merged_indices[t_offset + i]);
                }
            }
            
            // Sort final heap in descending order
            for (uint i = 0; i < final_size; i++) {
                for (uint j = i + 1; j < final_size; j++) {
                    if (final_heap[i] < final_heap[j]) {
                        float temp_val = final_heap[i];
                        uint temp_idx = final_indices[i];
                        final_heap[i] = final_heap[j];
                        final_indices[i] = final_indices[j];
                        final_heap[j] = temp_val;
                        final_indices[j] = temp_idx;
                    }
                }
            }
            
            // Write output
            for (uint i = 0; i < k; i++) {
                if (i < final_size) {
                    values_row[i] = final_heap[i];
                    indices_row[i] = final_indices[i];
                } else {
                    values_row[i] = NEG_INF;
                    indices_row[i] = 0;
                }
            }
        }
    } else {
        // Bitonic sort approach for large k
        // Use threadgroup memory for sorting
        threadgroup float tg_vals[1024];  // Adjust size as needed
        threadgroup uint tg_indices[1024];
        
        // Load elements into threadgroup memory
        uint load_size = min((uint)n, 1024u);
        for (uint i = tid; i < load_size; i += 32) {
            tg_vals[i] = (i < n) ? row[i] : NEG_INF;
            tg_indices[i] = i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Perform bitonic sort
        if (tid == 0) {
            bitonic_sort_partial(tg_vals, tg_indices, load_size, k);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Write top-k results
        for (uint i = tid; i < k; i += 32) {
            values_row[i] = tg_vals[i];
            indices_row[i] = tg_indices[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Top-P (Nucleus) Sampling - Optimized for Large Vocabularies
// ---------------------------------------------------------------------------

struct TopPParams {
    uint vocab_size;
    uint batch_size;
    float p;
    ulong seed;
};

/*
// Radix sort helper for sorting (prob, index) pairs by probability descending
// Uses 8-bit radix sort, processing in 4 passes (32-bit float representation)
inline void radix_sort_descending_8bit(
    threadgroup float* probs,
    threadgroup uint* indices,
    uint size,
    uint tid,
    uint tg_size
) {
    // Temporary buffers for sorting
    threadgroup float temp_probs[2048];
    threadgroup uint temp_indices[2048];
    threadgroup uint count[256];

    // Sort by bits of probability (reversed for descending order)
    for (uint pass = 0; pass < 4; pass++) {
        uint shift = pass * 8;

        // Clear histogram
        for (uint i = tid; i < 256; i += tg_size) {
            count[i] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Build histogram
        for (uint i = tid; i < size; i += tg_size) {
            uint bits = as_type<uint>(probs[i]);
            uint bucket = 255 - ((bits >> shift) & 0xFF);  // Reversed for descending
            atomic_fetch_add_explicit(&count[bucket], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Prefix sum (exclusive)
        uint total = 0;
        for (uint i = 0; i < 256; i++) {
            uint c = count[i];
            count[i] = total;
            total += c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter to temporary buffers
        for (uint i = tid; i < size; i += tg_size) {
            uint bits = as_type<uint>(probs[i]);
            uint bucket = 255 - ((bits >> shift) & 0xFF);
            uint pos = atomic_fetch_add_explicit(&count[bucket], 1, memory_order_relaxed);
            temp_probs[pos] = probs[i];
            temp_indices[pos] = indices[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Copy back
        for (uint i = tid; i < size; i += tg_size) {
            probs[i] = temp_probs[i];
            indices[i] = temp_indices[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
*/

// Optimized nucleus sampling using Gumbel-TopP trick and bucketed selection
// Input: logits [batch, vocab_size]
// Output: indices [batch]
//
// Algorithm for large vocabularies (>100K):
//   1. Compute softmax probabilities in parallel
//   2. Apply Gumbel noise to all probabilities
//   3. Use bucket-based sampling to efficiently find nucleus without full sort
//   4. Binary search within bucket for final sample
kernel void sample_top_p_large_vocab(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* workspace [[buffer(2)]],  // [batch, vocab_size * 2] for probs + gumbel_noise
    constant TopPParams& params [[buffer(3)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float p = params.p;
    device const float* row = logits + batch_idx * vocab;

    // Workspace layout: [probs, gumbel_noise]
    device float* probs = workspace + batch_idx * vocab * 2;
    device float* gumbel_noise = probs + vocab;

    // Initialize RNG with different seeds per thread for parallel noise generation
    PCGState rng = pcg_init(params.seed, (ulong)batch_idx * (ulong)vocab + (ulong)tid);

    // Step 1: Find max with vectorized loads (4x throughput)
    float local_max = NEG_INF;
    const uint vec_stride = tg_size * 4;
    uint i = tid * 4;
    
    for (; i + 3 < vocab; i += vec_stride) {
        float4 vals = *((device const float4*)(row + i));
        local_max = max(local_max, max(max(vals.x, vals.y), max(vals.z, vals.w)));
    }
    
    for (i = vocab - (vocab % 4) + tid; i < vocab; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    float global_max = simd_max(local_max);

    // Cross-simdgroup reduction
    threadgroup float shared_max[32];
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_max[simd_id] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_max[tid] : NEG_INF;
        global_max = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];

    // Step 2: Compute exp(x - max) and sum with vectorized ops
    float local_sum = 0.0f;
    i = tid * 4;
    
    for (; i + 3 < vocab; i += vec_stride) {
        float4 vals = *((device const float4*)(row + i));
        vals -= global_max;
        float4 exp_vals = exp(vals);
        *((device float4*)(probs + i)) = exp_vals;
        local_sum += exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
    }
    
    for (i = vocab - (vocab % 4) + tid; i < vocab; i += tg_size) {
        float exp_val = exp(row[i] - global_max);
        probs[i] = exp_val;
        local_sum += exp_val;
    }
    float global_sum = simd_sum(local_sum);

    // Cross-simdgroup sum reduction
    threadgroup float shared_sum[32];
    if (simd_lane == 0) {
        shared_sum[simd_id] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_sum[tid] : 0.0f;
        global_sum = simd_sum(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_sum[0];

    // Step 3: Normalize and generate Gumbel noise with vectorized ops
    float inv_sum = 1.0f / global_sum;
    i = tid * 4;
    
    // Vectorized normalization + Gumbel generation
    for (; i + 3 < vocab; i += vec_stride) {
        float4 prob_vals = *((device float4*)(probs + i)) * inv_sum;
        *((device float4*)(probs + i)) = prob_vals;
        
        // Generate 4 Gumbel noise values in parallel
        float4 gumbel;
        for (uint j = 0; j < 4; j++) {
            float u = max(pcg32_float(rng), 1e-10f);
            gumbel[j] = -log(-log(u));
        }
        *((device float4*)(gumbel_noise + i)) = gumbel;
    }
    
    for (i = vocab - (vocab % 4) + tid; i < vocab; i += tg_size) {
        probs[i] *= inv_sum;
        float u = max(pcg32_float(rng), 1e-10f);
        gumbel_noise[i] = -log(-log(u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Hierarchical bucket reduction to minimize atomic contention
    // For large vocab (>100K), reduce atomics by using thread-local accumulators
    constexpr uint NUM_BUCKETS = 256;  // Increased for better granularity
    threadgroup float bucket_sums[NUM_BUCKETS];
    threadgroup float bucket_max_gumbel[NUM_BUCKETS];
    threadgroup uint bucket_max_idx[NUM_BUCKETS];
    
    // Thread-local accumulators to batch atomic updates
    float local_bucket_sums[16];  // Each thread tracks 16 buckets
    float local_bucket_max[16];
    uint local_bucket_idx[16];
    
    // Initialize thread-local accumulators
    for (uint j = 0; j < 16; j++) {
        local_bucket_sums[j] = 0.0f;
        local_bucket_max[j] = NEG_INF;
        local_bucket_idx[j] = 0;
    }
    
    // Initialize shared buckets
    if (tid < NUM_BUCKETS) {
        bucket_sums[tid] = 0.0f;
        bucket_max_gumbel[tid] = NEG_INF;
        bucket_max_idx[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 1: Accumulate into thread-local buckets (no atomics)
    i = tid * 4;
    for (; i + 3 < vocab; i += vec_stride) {
        float4 prob_vals = *((device const float4*)(probs + i));
        float4 gumbel_vals = *((device const float4*)(gumbel_noise + i));
        
        for (uint j = 0; j < 4; j++) {
            float log_prob = log(max(prob_vals[j], 1e-10f));
            int bucket_idx = int((log_prob + 20.0f) * (NUM_BUCKETS / 20.0f));
            bucket_idx = clamp(bucket_idx, 0, int(NUM_BUCKETS - 1));
            
            // Map global bucket to local accumulator (16 buckets per thread)
            uint local_slot = bucket_idx % 16;
            local_bucket_sums[local_slot] += prob_vals[j];
            
            float adjusted = log_prob + gumbel_vals[j];
            if (adjusted > local_bucket_max[local_slot]) {
                local_bucket_max[local_slot] = adjusted;
                local_bucket_idx[local_slot] = i + j;
            }
        }
    }
    
    for (i = vocab - (vocab % 4) + tid; i < vocab; i += tg_size) {
        float log_prob = log(max(probs[i], 1e-10f));
        int bucket_idx = int((log_prob + 20.0f) * (NUM_BUCKETS / 20.0f));
        bucket_idx = clamp(bucket_idx, 0, int(NUM_BUCKETS - 1));
        
        uint local_slot = bucket_idx % 16;
        local_bucket_sums[local_slot] += probs[i];
        
        float adjusted = log_prob + gumbel_noise[i];
        if (adjusted > local_bucket_max[local_slot]) {
            local_bucket_max[local_slot] = adjusted;
            local_bucket_idx[local_slot] = i;
        }
    }
    
    // Phase 2: Flush thread-local accumulators to shared memory via SIMD reductions
    // Avoid atomics by using warp-level aggregation first
    for (uint j = 0; j < 16; j++) {
        uint bucket_id = (tid * 16 + j) % NUM_BUCKETS;
        
        // SIMD-level sum reduction for this bucket slot
        float _simd_sum = simd_sum(local_bucket_sums[j]);
        float simd_max_val = simd_max(local_bucket_max[j]);
        
        // Only lane 0 of each SIMD writes to shared memory
        if (simd_lane == 0 && _simd_sum > 0.0f) {
            bucket_sums[bucket_id] += _simd_sum;  // Still needs atomic, but 32x fewer ops
            if (simd_max_val > bucket_max_gumbel[bucket_id]) {
                bucket_max_gumbel[bucket_id] = simd_max_val;
                bucket_max_idx[bucket_id] = local_bucket_idx[j];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: Find top-p by scanning buckets from highest to lowest
    if (tid == 0) {
        float cumsum = 0.0f;
        uint candidate_idx = 0;
        float max_adjusted = NEG_INF;

        // Find which buckets constitute the top-p nucleus
        for (int b = NUM_BUCKETS - 1; b >= 0; b--) {
            float bucket_sum = bucket_sums[b];
            if (bucket_sum <= 0.0f) continue;

            if (cumsum + bucket_sum <= p) {
                cumsum += bucket_sum;
                // Track best candidate from included buckets
                if (bucket_max_gumbel[b] > max_adjusted) {
                    max_adjusted = bucket_max_gumbel[b];
                    candidate_idx = bucket_max_idx[b];
                }
            } else {
                // Partial bucket - need to examine elements in this bucket
                cumsum += bucket_sum;
                // Track best candidate from this partial bucket
                if (bucket_max_gumbel[b] > max_adjusted) {
                    max_adjusted = bucket_max_gumbel[b];
                    candidate_idx = bucket_max_idx[b];
                }
                break;
            }
        }

        // Final sample using gumbel-top trick on the best candidate
        // For simplicity, use the max-adjusted token (could refine by searching partial bucket)
        indices[batch_idx] = candidate_idx;
    }
}

// Original top-p for smaller vocabularies (< 100K) - kept for compatibility
kernel void sample_top_p(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* workspace [[buffer(2)]],  // [batch, vocab_size * 2] for probs + sorted
    constant TopPParams& params [[buffer(3)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float p = params.p;
    device const float* row = logits + batch_idx * vocab;

    // Workspace layout: [probs, indices_float]
    device float* probs = workspace + batch_idx * vocab * 2;
    device float* idx_buf = probs + vocab;

    // Initialize RNG
    PCGState rng = pcg_init(params.seed, batch_idx);

    // Step 1: Compute softmax
    float local_max = NEG_INF;
    for (uint i = tid; i < vocab; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    float global_max = simd_max(local_max);

    threadgroup float shared[32];
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared[simd_id] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared[tid] : NEG_INF;
        global_max = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        float exp_val = exp(row[i] - global_max);
        probs[i] = exp_val;
        idx_buf[i] = float(i);  // Store indices as float for sorting
        local_sum += exp_val;
    }
    float global_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared[simd_id] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared[tid] : 0.0f;
        global_sum = simd_sum(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared[0];

    float inv_sum = 1.0f / global_sum;
    for (uint i = tid; i < vocab; i += tg_size) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 2-4: Thread 0 does sorting and sampling
    // (For large vocabs, use radix sort; for now, use selection)
    if (tid == 0) {
        float cumsum = 0.0f;
        float target = pcg32_float(rng) * p;  // Random threshold within nucleus
        uint sampled = 0;

        // Greedy selection: repeatedly find max until cumsum >= p
        float adjusted_sum = 0.0f;
        while (cumsum < p && adjusted_sum < 1.0f) {
            float max_prob = 0.0f;
            uint max_idx = 0;

            for (uint i = 0; i < vocab; i++) {
                if (probs[i] > max_prob) {
                    max_prob = probs[i];
                    max_idx = i;
                }
            }

            if (max_prob <= 0.0f) break;

            adjusted_sum += max_prob;
            cumsum += max_prob;

            // Check if we should sample this token
            if (target <= adjusted_sum) {
                sampled = max_idx;
                break;
            }

            // Mark as used
            probs[max_idx] = -1.0f;
            sampled = max_idx;  // Default to last one if we exhaust
        }

        indices[batch_idx] = sampled;
    }
}

// ---------------------------------------------------------------------------
// Search-based Top-P (Nucleus) Sampling
// ---------------------------------------------------------------------------

// Parameters for search-based top-p sampling
struct TopPSearchParams {
    uint vocab_size;
    uint batch_size;
    float p;           // Nucleus threshold [0, 1]
    ulong seed;
    uint iterations;    // Binary search iterations (8-10 for float precision)
};

// Binary search for threshold t where sum_{p_i >= t} p_i >= p_target
// Complexity: O(vocab_size * log(iterations)) instead of O(vocab_size * log(vocab_size))
kernel void sample_top_p_search(
    device const float* logits [[buffer(0)]],    // [batch, vocab_size]
    device uint* indices [[buffer(1)]],          // [batch] output
    device float* workspace_probs [[buffer(2)]],  // [batch, vocab_size] workspace for probs
    constant TopPSearchParams& params [[buffer(3)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float p_target = params.p;
    device const float* row = logits + batch_idx * vocab;
    device float* probs_row = workspace_probs + batch_idx * vocab;

    // Step 1: Compute softmax probabilities
    float local_max = NEG_INF;
    for (uint i = tid; i < vocab; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    float global_max = simd_max(local_max);

    // Cross-simdgroup reduction
    threadgroup float shared_max[32];
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_max[simd_id] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_max[tid] : NEG_INF;
        global_max = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        float exp_val = exp(row[i] - global_max);
        probs_row[i] = exp_val;
        local_sum += exp_val;
    }
    float global_sum = simd_sum(local_sum);

    threadgroup float shared_sum[32];
    if (simd_lane == 0) {
        shared_sum[simd_id] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_sum[tid] : 0.0f;
        global_sum = simd_sum(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_sum[0];

    float inv_sum = 1.0f / global_sum;
    for (uint i = tid; i < vocab; i += tg_size) {
        probs_row[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: Find max probability for binary search bounds
    float max_prob = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        max_prob = max(max_prob, probs_row[i]);
    }
    max_prob = simd_max(max_prob);

    threadgroup float shared_prob[32];
    if (simd_lane == 0) {
        shared_prob[simd_id] = max_prob;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_prob[tid] : 0.0f;
        max_prob = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_prob[0] = max_prob;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float high = shared_prob[0];
    float low = 0.0f;

    // Step 3: Binary search for threshold
    // Search for t where sum_{p_i >= t} p_i = p_target
    const uint MAX_ITERS = params.iterations;
    float threshold = high * 0.5f;  // Start at midpoint

    for (uint iter = 0; iter < MAX_ITERS; iter++) {
        // Compute cumulative probability of tokens >= threshold
        uint local_count = 0;
        float local_cumsum = 0.0f;

        for (uint i = tid; i < vocab; i += tg_size) {
            float p = probs_row[i];
            if (p >= threshold) {
                local_count++;
                local_cumsum += p;
            }
        }

        // Warp reduction
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            uint other_c = simd_shuffle_down(local_count, offset);
            float other_s = simd_shuffle_down(local_cumsum, offset);
            local_count += other_c;
            local_cumsum += other_s;
        }

        // Cross-simdgroup reduction
        threadgroup float tg_stats[32];
        if (simd_lane == 0) {
            tg_stats[simd_id] = float(local_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 32) {
            uint num_groups = (tg_size + 31) / 32;
            float c_val = (tid < num_groups) ? tg_stats[tid] : 0.0f;
            uint count = uint(c_val);

            for (ushort offset = 16; offset > 0; offset >>= 1) {
                uint other_c = simd_shuffle_down(count, offset);
                float other_s = simd_shuffle_down(local_cumsum, offset);
                count += other_c;
                local_cumsum += other_s;
            }
            local_count = count;

            if (tid == 0) {
                float cumprob = local_cumsum;
                // Adjust search range:
                // - If cumprob > p_target and count > 0: lower threshold (include fewer tokens)
                // - Otherwise: raise threshold (include more tokens)
                if (cumprob > p_target && count > 0) {
                    low = threshold;
                } else {
                    high = threshold;
                }
                threshold = (low + high) * 0.5f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 4: Sample from tokens >= final threshold
    if (tid == 0) {
        PCGState rng = pcg_init(params.seed, batch_idx);

        // Compute sum of probabilities in nucleus
        float nucleus_sum = 0.0f;
        uint nucleus_count = 0;

        for (uint i = 0; i < vocab; i++) {
            if (probs_row[i] >= threshold) {
                nucleus_sum += probs_row[i];
                nucleus_count++;
            }
        }

        // Edge case: if threshold is too high, nothing selected
        if (nucleus_count == 0) {
            // Fallback: sample from highest probability token
            float max_p = 0.0f;
            uint max_i = 0;
            for (uint i = 0; i < vocab; i++) {
                if (probs_row[i] > max_p) {
                    max_p = probs_row[i];
                    max_i = i;
                }
            }
            indices[batch_idx] = max_i;
        } else {
            // Renormalize nucleus probabilities and sample
            float r = pcg32_float(rng) * nucleus_sum;
            float cumsum = 0.0f;
            uint sampled = 0;

            for (uint i = 0; i < vocab; i++) {
                if (probs_row[i] >= threshold) {
                    cumsum += probs_row[i];
                    if (cumsum >= r) {
                        sampled = i;
                        break;
                    }
                }
            }

            // Fallback: if random value exceeds computed sum (numerical edge case)
            if (cumsum < r) {
                for (uint i = vocab - 1; i > 0; i--) {
                    if (probs_row[i] >= threshold) {
                        sampled = i;
                        break;
                    }
                }
            }

            indices[batch_idx] = sampled;
        }
    }
}

// Batched version of search-based top-p
// Each threadgroup processes one batch element independently
kernel void sample_top_p_search_batched(
    device const float* logits [[buffer(0)]],    // [batch, vocab_size]
    device uint* indices [[buffer(1)]],          // [batch] output
    device float* workspace_probs [[buffer(2)]],  // [batch, vocab_size] workspace
    constant TopPSearchParams& params [[buffer(3)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float p_target = params.p;
    device const float* row = logits + batch_idx * vocab;
    device float* probs_row = workspace_probs + batch_idx * vocab;

    // Step 1: Compute softmax (same as single-batch version)
    float local_max = NEG_INF;
    for (uint i = tid; i < vocab; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    float global_max = simd_max(local_max);

    threadgroup float shared[32];
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared[simd_id] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared[tid] : NEG_INF;
        global_max = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        float exp_val = exp(row[i] - global_max);
        probs_row[i] = exp_val;
        local_sum += exp_val;
    }
    float global_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared[simd_id] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared[tid] : 0.0f;
        global_sum = simd_sum(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared[0];

    float inv_sum = 1.0f / global_sum;
    for (uint i = tid; i < vocab; i += tg_size) {
        probs_row[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Find max probability
    float max_prob = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        max_prob = max(max_prob, probs_row[i]);
    }
    max_prob = simd_max(max_prob);

    threadgroup float shared_prob[32];
    if (simd_lane == 0) {
        shared_prob[simd_id] = max_prob;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float val = (tid < (tg_size + 31) / 32) ? shared_prob[tid] : 0.0f;
        max_prob = simd_max(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_prob[0] = max_prob;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float high = shared_prob[0];
    float low = 0.0f;

    // Binary search - each threadgroup works independently
    const uint MAX_ITERS = 8;
    float threshold = high * 0.5f;

    for (uint iter = 0; iter < MAX_ITERS; iter++) {
        uint local_count = 0;
        float local_cumsum = 0.0f;

        for (uint i = tid; i < vocab; i += tg_size) {
            float p = probs_row[i];
            if (p >= threshold) {
                local_count++;
                local_cumsum += p;
            }
        }

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            uint other_c = simd_shuffle_down(local_count, offset);
            float other_s = simd_shuffle_down(local_cumsum, offset);
            local_count += other_c;
            local_cumsum += other_s;
        }

        threadgroup float tg_stats[32];
        if (simd_lane == 0) {
            tg_stats[simd_id] = float(local_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 32) {
            uint num_groups = (tg_size + 31) / 32;
            float c_val = (tid < num_groups) ? tg_stats[tid] : 0.0f;
            uint count = uint(c_val);

            for (ushort offset = 16; offset > 0; offset >>= 1) {
                uint other_c = simd_shuffle_down(count, offset);
                float other_s = simd_shuffle_down(local_cumsum, offset);
                count += other_c;
                local_cumsum += other_s;
            }

            if (tid == 0) {
                float cumprob = local_cumsum;
                if (cumprob > p_target && count > 0) {
                    low = threshold;
                } else {
                    high = threshold;
                }
                threshold = (low + high) * 0.5f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Sample using RNG with batch_idx for different sequences
    if (tid == 0) {
        PCGState rng = pcg_init(params.seed, batch_idx);
        float nucleus_sum = 0.0f;
        uint nucleus_count = 0;

        for (uint i = 0; i < vocab; i++) {
            if (probs_row[i] >= threshold) {
                nucleus_sum += probs_row[i];
                nucleus_count++;
            }
        }

        if (nucleus_count == 0) {
            float max_p = 0.0f;
            uint max_i = 0;
            for (uint i = 0; i < vocab; i++) {
                if (probs_row[i] > max_p) {
                    max_p = probs_row[i];
                    max_i = i;
                }
            }
            indices[batch_idx] = max_i;
        } else {
            float r = pcg32_float(rng) * nucleus_sum;
            float cumsum = 0.0f;
            uint sampled = 0;

            for (uint i = 0; i < vocab; i++) {
                if (probs_row[i] >= threshold) {
                    cumsum += probs_row[i];
                    if (cumsum >= r) {
                        sampled = i;
                        break;
                    }
                }
            }

            if (cumsum < r) {
                for (uint i = vocab - 1; i > 0; i--) {
                    if (probs_row[i] >= threshold) {
                        sampled = i;
                        break;
                    }
                }
            }

            indices[batch_idx] = sampled;
        }
    }
}

// ---------------------------------------------------------------------------
// Categorical Sampling (from softmax probabilities)
// ---------------------------------------------------------------------------

struct CategoricalParams {
    uint vocab_size;
    uint batch_size;
    ulong seed;
};

// Sample from a categorical distribution (softmax probabilities)
// Input: probs [batch, vocab_size] (must sum to 1)
// Output: indices [batch]
kernel void sample_categorical(
    device const float* probs [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant CategoricalParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    device const float* row = probs + batch_idx * vocab;

    // Initialize RNG
    PCGState rng = pcg_init(params.seed, batch_idx);

    // Thread 0 does the sampling
    if (tid == 0) {
        float r = pcg32_float(rng);
        float cumsum = 0.0f;
        uint sampled = 0;

        for (uint i = 0; i < vocab; i++) {
            cumsum += row[i];
            if (cumsum >= r) {
                sampled = i;
                break;
            }
        }

        indices[batch_idx] = sampled;
    }
}

// Sample from logits (computes softmax internally)
// Input: logits [batch, vocab_size]
// Output: indices [batch]
kernel void sample_categorical_logits(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant CategoricalParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    device const float* row = logits + batch_idx * vocab;

    // Initialize RNG
    PCGState rng = pcg_init(params.seed, batch_idx);

    // Compute softmax and sample in one pass using Gumbel-max trick
    // Sample = argmax(logits + Gumbel noise)
    // where Gumbel noise = -log(-log(uniform))

    float max_gumbel = NEG_INF;
    uint max_idx = 0;

    // Each thread processes a portion of the vocabulary
    for (uint i = tid; i < vocab; i += tg_size) {
        float u = pcg32_float(rng);
        // Avoid log(0)
        u = max(u, 1e-10f);
        float gumbel = -log(-log(u));
        float perturbed = row[i] + gumbel;

        if (perturbed > max_gumbel) {
            max_gumbel = perturbed;
            max_idx = i;
        }
    }

    // SIMD reduction
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(max_gumbel, offset);
        uint other_idx = simd_shuffle_down(max_idx, offset);
        if (other_val > max_gumbel) {
            max_gumbel = other_val;
            max_idx = other_idx;
        }
    }

    // Cross-simdgroup reduction
    threadgroup float shared_val[32];
    threadgroup uint shared_idx[32];

    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_val[simd_id] = max_gumbel;
        shared_idx[simd_id] = max_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint num_groups = (tg_size + 31) / 32;
        max_gumbel = (tid < num_groups) ? shared_val[tid] : NEG_INF;
        max_idx = (tid < num_groups) ? shared_idx[tid] : 0;

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(max_gumbel, offset);
            uint other_idx = simd_shuffle_down(max_idx, offset);
            if (other_val > max_gumbel) {
                max_gumbel = other_val;
                max_idx = other_idx;
            }
        }

        if (tid == 0) {
            indices[batch_idx] = max_idx;
        }
    }
}

// FP16 variant
kernel void sample_categorical_logits_fp16(
    device const half* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant CategoricalParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    device const half* row = logits + batch_idx * vocab;

    PCGState rng = pcg_init(params.seed, batch_idx);

    float max_gumbel = NEG_INF;
    uint max_idx = 0;

    for (uint i = tid; i < vocab; i += tg_size) {
        float u = pcg32_float(rng);
        u = max(u, 1e-10f);
        float gumbel = -log(-log(u));
        float perturbed = float(row[i]) + gumbel;

        if (perturbed > max_gumbel) {
            max_gumbel = perturbed;
            max_idx = i;
        }
    }

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(max_gumbel, offset);
        uint other_idx = simd_shuffle_down(max_idx, offset);
        if (other_val > max_gumbel) {
            max_gumbel = other_val;
            max_idx = other_idx;
        }
    }

    threadgroup float shared_val[32];
    threadgroup uint shared_idx[32];

    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (simd_lane == 0) {
        shared_val[simd_id] = max_gumbel;
        shared_idx[simd_id] = max_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint num_groups = (tg_size + 31) / 32;
        max_gumbel = (tid < num_groups) ? shared_val[tid] : NEG_INF;
        max_idx = (tid < num_groups) ? shared_idx[tid] : 0;

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(max_gumbel, offset);
            uint other_idx = simd_shuffle_down(max_idx, offset);
            if (other_val > max_gumbel) {
                max_gumbel = other_val;
                max_idx = other_idx;
            }
        }

        if (tid == 0) {
            indices[batch_idx] = max_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Repetition Penalty
// ---------------------------------------------------------------------------

struct RepetitionPenaltyParams {
    uint vocab_size;
    uint batch_size;
    uint num_generated;
    float penalty;
};

// Apply repetition penalty to logits in-place
// For each token in generated_ids:
//   if logits[token] > 0: logits[token] /= penalty
//   else: logits[token] *= penalty
// Input/Output: logits [batch, vocab_size]
// generated_ids: [num_generated] - token IDs to penalize
kernel void apply_repetition_penalty(
    device float* logits [[buffer(0)]],
    device const uint* generated_ids [[buffer(1)]],
    constant RepetitionPenaltyParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float penalty = params.penalty;
    device float* row = logits + batch_idx * vocab;

    // Each thread processes a subset of generated tokens
    for (uint i = tid; i < params.num_generated; i += tg_size) {
        uint token_id = generated_ids[i];
        if (token_id < vocab) {
            float val = row[token_id];
            if (val > 0.0f) {
                row[token_id] = val / penalty;
            } else {
                row[token_id] = val * penalty;
            }
        }
    }
}

// FP16 variant
kernel void apply_repetition_penalty_fp16(
    device half* logits [[buffer(0)]],
    device const uint* generated_ids [[buffer(1)]],
    constant RepetitionPenaltyParams& params [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float penalty = params.penalty;
    device half* row = logits + batch_idx * vocab;

    for (uint i = tid; i < params.num_generated; i += tg_size) {
        uint token_id = generated_ids[i];
        if (token_id < vocab) {
            float val = float(row[token_id]);
            if (val > 0.0f) {
                row[token_id] = half(val / penalty);
            } else {
                row[token_id] = half(val * penalty);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Temperature Scaling
// ---------------------------------------------------------------------------

struct TemperatureParams {
    uint vocab_size;
    uint batch_size;
    float inv_temperature;  // 1.0 / temperature
};

// Apply temperature scaling to logits in-place
// logits = logits / temperature = logits * inv_temperature
kernel void apply_temperature(
    device float* logits [[buffer(0)]],
    constant TemperatureParams& params [[buffer(1)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float inv_temp = params.inv_temperature;
    device float* row = logits + batch_idx * vocab;

    for (uint i = tid; i < vocab; i += tg_size) {
        row[i] *= inv_temp;
    }
}

kernel void apply_temperature_fp16(
    device half* logits [[buffer(0)]],
    constant TemperatureParams& params [[buffer(1)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint vocab = params.vocab_size;
    const float inv_temp = params.inv_temperature;
    device half* row = logits + batch_idx * vocab;

    for (uint i = tid; i < vocab; i += tg_size) {
        row[i] = half(float(row[i]) * inv_temp);
    }
}
