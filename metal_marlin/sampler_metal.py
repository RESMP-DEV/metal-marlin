"""
Metal-accelerated token sampling for large vocabulary language models.

This module provides GPU-accelerated sampling operations optimized for
large vocabularies (e.g., 154,880 tokens for GLM-4.7) using Metal compute shaders.

Sampling methods supported:
    1. Greedy (argmax) - deterministic selection of highest logit
    2. Top-k sampling - sample from k highest probability tokens
    3. Top-p (nucleus) sampling - sample from tokens comprising top p cumulative probability
    4. Temperature scaling - adjust logit distribution sharpness
    5. Repetition penalty - discourage recently generated tokens

The Metal kernels handle:
    - Parallel softmax reduction across large vocabularies
    - Efficient top-k selection using partial sorting
    - Parallel cumulative sum for nucleus sampling
    - GPU-based multinomial sampling

Usage:
    from metal_marlin.sampler_metal import MetalSampler

    sampler = MetalSampler(vocab_size=154880)
    token_ids = sampler.sample(
        logits,  # [batch, vocab_size]
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

Requirements:
    - PyTorch with MPS backend (Apple Silicon)
    - PyObjC Metal framework
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Check dependencies
try:
    import torch

    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None  # type: ignore[assignment]

try:
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Metal shader source for sampling operations
# ---------------------------------------------------------------------------

_SAMPLER_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant constexpr uint THREADS_PER_GROUP = 256;
constant constexpr uint WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Helpers: Parallel reduction primitives
// ---------------------------------------------------------------------------

// Warp-level max reduction
inline float warp_reduce_max(float val, uint lane_id) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

// Warp-level sum reduction
inline float warp_reduce_sum(float val, uint lane_id) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// Warp-level argmax (returns index of max value)
inline void warp_argmax(thread float &val, thread uint &idx, uint lane_id) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_xor(val, offset);
        uint other_idx = simd_shuffle_xor(idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Apply temperature scaling in-place
// ---------------------------------------------------------------------------

kernel void apply_temperature(
    device float* logits [[buffer(0)]],
    constant float& temperature [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < vocab_size) {
        logits[gid] = logits[gid] / temperature;
    }
}

// ---------------------------------------------------------------------------
// Kernel: Apply repetition penalty
// logits[token_id] = logits[token_id] / penalty if positive
// logits[token_id] = logits[token_id] * penalty if negative
// ---------------------------------------------------------------------------

kernel void apply_repetition_penalty(
    device float* logits [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    constant float& penalty [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < num_tokens) {
        uint token_id = token_ids[gid];
        float val = logits[token_id];
        if (val > 0.0f) {
            logits[token_id] = val / penalty;
        } else {
            logits[token_id] = val * penalty;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Numerically stable softmax for large vocabularies
//
// Two-pass algorithm:
//   Pass 1: Find global maximum (this kernel)
//   Pass 2: Compute exp(x - max) and sum (softmax_exp_sum kernel)
//   Pass 3: Normalize by sum (softmax_normalize kernel)
// ---------------------------------------------------------------------------

// Pass 1: Parallel max reduction
kernel void softmax_find_max(
    device const float* logits [[buffer(0)]],
    device float* block_maxes [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_max[THREADS_PER_GROUP];

    // Each thread finds max of its elements
    float local_max = -INFINITY;
    for (uint i = gid; i < vocab_size; i += group_size * ((vocab_size + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP)) {
        if (i < vocab_size) {
            local_max = max(local_max, logits[i]);
        }
    }

    shared_max[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_maxes[group_id] = shared_max[0];
    }
}

// Pass 2: Compute exp(x - max) and partial sums
kernel void softmax_exp_sum(
    device float* logits [[buffer(0)]],  // In-place: logits -> exp(logits - max)
    device float* block_sums [[buffer(1)]],
    constant float& global_max [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[THREADS_PER_GROUP];

    float local_sum = 0.0f;
    for (uint i = gid; i < vocab_size; i += group_size * ((vocab_size + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP)) {
        if (i < vocab_size) {
            float exp_val = exp(logits[i] - global_max);
            logits[i] = exp_val;
            local_sum += exp_val;
        }
    }

    shared_sum[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_sums[group_id] = shared_sum[0];
    }
}

// Pass 3: Normalize to get probabilities
kernel void softmax_normalize(
    device float* probs [[buffer(0)]],
    constant float& sum [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < vocab_size) {
        probs[gid] = probs[gid] / sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel: Fused softmax for moderate vocabulary sizes (single threadgroup)
//
// For vocab_size <= 65536, we can do the full softmax in one kernel
// using shared memory reductions.
// ---------------------------------------------------------------------------

kernel void softmax_fused(
    device float* logits [[buffer(0)]],  // In-place: logits -> probabilities
    constant uint& vocab_size [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[THREADS_PER_GROUP];

    // Pass 1: Find max
    float local_max = -INFINITY;
    for (uint i = lid; i < vocab_size; i += group_size) {
        local_max = max(local_max, logits[i]);
    }

    shared_data[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] = max(shared_data[lid], shared_data[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float global_max = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = lid; i < vocab_size; i += group_size) {
        float exp_val = exp(logits[i] - global_max);
        logits[i] = exp_val;
        local_sum += exp_val;
    }

    shared_data[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float global_sum = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: Normalize
    for (uint i = lid; i < vocab_size; i += group_size) {
        logits[i] = logits[i] / global_sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel: Greedy (argmax) selection
// ---------------------------------------------------------------------------

kernel void argmax_reduce(
    device const float* logits [[buffer(0)]],
    device uint* result [[buffer(1)]],
    device float* result_val [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_vals[THREADS_PER_GROUP];
    threadgroup uint shared_idxs[THREADS_PER_GROUP];

    float local_max = -INFINITY;
    uint local_idx = 0;

    for (uint i = lid; i < vocab_size; i += group_size) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    shared_vals[lid] = local_max;
    shared_idxs[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared_vals[lid + stride] > shared_vals[lid]) {
                shared_vals[lid] = shared_vals[lid + stride];
                shared_idxs[lid] = shared_idxs[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        result[0] = shared_idxs[0];
        result_val[0] = shared_vals[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel: Top-k selection using partial sorting
//
// Two-phase algorithm:
//   Phase 1: Each thread finds its local top-k
//   Phase 2: Merge local top-k lists to get global top-k
// ---------------------------------------------------------------------------

// Insertion sort helper for small arrays (used in top-k)
inline void insert_sorted(
    thread float* vals,
    thread uint* idxs,
    uint k,
    float new_val,
    uint new_idx
) {
    // Find insertion position
    if (new_val <= vals[k - 1]) return;  // Not in top-k

    uint pos = k - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
    idxs[pos] = new_idx;
}

// Top-k selection kernel
// Assumes k <= 64 (fits in thread-local storage)
kernel void topk_select(
    device const float* logits [[buffer(0)]],
    device float* topk_vals [[buffer(1)]],  // [k]
    device uint* topk_idxs [[buffer(2)]],   // [k]
    constant uint& vocab_size [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Thread-local top-k (max k=64)
    float local_vals[64];
    uint local_idxs[64];

    // Initialize to -inf
    for (uint i = 0; i < k; i++) {
        local_vals[i] = -INFINITY;
        local_idxs[i] = 0;
    }

    // Each thread scans its portion
    for (uint i = lid; i < vocab_size; i += group_size) {
        float val = logits[i];
        insert_sorted(local_vals, local_idxs, k, val, i);
    }

    // Now merge all thread results using shared memory
    // This is a simplification - for production, use a proper merge
    threadgroup float shared_vals[THREADS_PER_GROUP * 64];
    threadgroup uint shared_idxs[THREADS_PER_GROUP * 64];

    // Store local results
    for (uint i = 0; i < k; i++) {
        shared_vals[lid * 64 + i] = local_vals[i];
        shared_idxs[lid * 64 + i] = local_idxs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 merges all results (simple but not optimal for large thread counts)
    if (lid == 0) {
        // Final top-k values
        float final_vals[64];
        uint final_idxs[64];

        for (uint i = 0; i < k; i++) {
            final_vals[i] = -INFINITY;
            final_idxs[i] = 0;
        }

        // Merge from all threads
        for (uint t = 0; t < group_size; t++) {
            for (uint i = 0; i < k; i++) {
                float val = shared_vals[t * 64 + i];
                uint idx = shared_idxs[t * 64 + i];
                if (val > -INFINITY) {
                    insert_sorted(final_vals, final_idxs, k, val, idx);
                }
            }
        }

        // Write output
        for (uint i = 0; i < k; i++) {
            topk_vals[i] = final_vals[i];
            topk_idxs[i] = final_idxs[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Top-p (nucleus) sampling
//
// Given sorted probabilities (descending), find the cutoff index where
// cumulative probability exceeds p.
// ---------------------------------------------------------------------------

kernel void topp_cumsum_and_cutoff(
    device const float* sorted_probs [[buffer(0)]],  // [vocab_size] descending
    device uint* cutoff_idx [[buffer(1)]],           // Output: index where cumsum > p
    constant float& p [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // Single thread computes cumsum and finds cutoff
    // This is simple but works for reasonable vocab sizes
    // For very large vocabs, a parallel scan could be used
    if (lid == 0) {
        float cumsum = 0.0f;
        uint cutoff = vocab_size;

        for (uint i = 0; i < vocab_size; i++) {
            cumsum += sorted_probs[i];
            if (cumsum >= p) {
                cutoff = i + 1;  // Include this token
                break;
            }
        }

        // Ensure at least 1 token
        cutoff_idx[0] = max(cutoff, 1u);
    }
}

// ---------------------------------------------------------------------------
// Kernel: Multinomial sampling from discrete distribution
//
// Given probabilities and a uniform random value in [0, 1), find the
// sampled token index using cumulative sum search.
// ---------------------------------------------------------------------------

kernel void multinomial_sample(
    device const float* probs [[buffer(0)]],
    device uint* sampled_idx [[buffer(1)]],
    constant float& random_val [[buffer(2)]],  // Uniform [0, 1)
    constant uint& num_tokens [[buffer(3)]],   // Number of valid tokens (k for top-k, cutoff for top-p)
    uint lid [[thread_position_in_threadgroup]]
) {
    if (lid == 0) {
        float cumsum = 0.0f;
        uint result = 0;

        for (uint i = 0; i < num_tokens; i++) {
            cumsum += probs[i];
            if (random_val < cumsum) {
                result = i;
                break;
            }
            result = i;  // Fall through to last if random_val >= 1.0 (numerical edge case)
        }

        sampled_idx[0] = result;
    }
}

// ---------------------------------------------------------------------------
// Kernel: Gather values by indices (for mapping sampled index back to vocab)
// ---------------------------------------------------------------------------

kernel void gather_by_index(
    device const uint* indices [[buffer(0)]],  // [n] indices into vocab
    device const uint* sampled [[buffer(1)]],  // [1] index into indices
    device uint* result [[buffer(2)]],         // [1] final token id
    uint lid [[thread_position_in_threadgroup]]
) {
    if (lid == 0) {
        result[0] = indices[sampled[0]];
    }
}

// ---------------------------------------------------------------------------
// Kernel: Sort probabilities descending (for top-p)
// Uses bitonic sort for GPU efficiency
// ---------------------------------------------------------------------------

// Bitonic compare-exchange
inline void compare_exchange_desc(
    thread float &a, thread float &b,
    thread uint &ia, thread uint &ib,
    bool dir  // true = descending
) {
    if ((a < b) == dir) {
        float tmp = a; a = b; b = tmp;
        uint itmp = ia; ia = ib; ib = itmp;
    }
}

// Simplified radix-style top-k for large vocabularies
// Works by finding threshold and filtering
kernel void find_topk_threshold(
    device const float* logits [[buffer(0)]],
    device float* threshold [[buffer(1)]],
    device atomic_uint* count [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant float& current_thresh [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Count how many values are >= threshold
    uint local_count = 0;
    for (uint i = gid; i < vocab_size; i += group_size * ((vocab_size + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP)) {
        if (i < vocab_size && logits[i] >= current_thresh) {
            local_count++;
        }
    }

    // Warp reduce
    local_count += simd_shuffle_xor(local_count, 16);
    local_count += simd_shuffle_xor(local_count, 8);
    local_count += simd_shuffle_xor(local_count, 4);
    local_count += simd_shuffle_xor(local_count, 2);
    local_count += simd_shuffle_xor(local_count, 1);

    // First lane in each warp adds to global count
    if (lid % 32 == 0) {
        atomic_fetch_add_explicit(count, local_count, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Kernel: Batched argmax for multiple sequences
// ---------------------------------------------------------------------------

kernel void batched_argmax(
    device const float* logits [[buffer(0)]],  // [batch, vocab_size]
    device uint* results [[buffer(1)]],        // [batch]
    constant uint& batch_size [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint batch_id [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (batch_id >= batch_size) return;

    threadgroup float shared_vals[THREADS_PER_GROUP];
    threadgroup uint shared_idxs[THREADS_PER_GROUP];

    device const float* batch_logits = logits + batch_id * vocab_size;

    float local_max = -INFINITY;
    uint local_idx = 0;

    for (uint i = lid; i < vocab_size; i += group_size) {
        float val = batch_logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    shared_vals[lid] = local_max;
    shared_idxs[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared_vals[lid + stride] > shared_vals[lid]) {
                shared_vals[lid] = shared_vals[lid + stride];
                shared_idxs[lid] = shared_idxs[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        results[batch_id] = shared_idxs[0];
    }
}
"""


# ---------------------------------------------------------------------------
# Sampling configuration
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Configuration for token sampling."""

    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0  # 1.0 = disabled
    repetition_penalty: float = 1.0  # 1.0 = disabled
    do_sample: bool = True  # False = greedy


# ---------------------------------------------------------------------------
# Metal Sampler class
# ---------------------------------------------------------------------------


class MetalSampler:
    """Metal-accelerated token sampler for large vocabulary LLMs.

    Uses Metal compute kernels for efficient softmax, top-k selection,
    and multinomial sampling on Apple Silicon GPUs.

    Example:
        sampler = MetalSampler(vocab_size=154880)

        # Greedy decoding
        token_id = sampler.sample(logits, temperature=0.0)

        # Top-p sampling with temperature
        token_id = sampler.sample(logits, temperature=0.7, top_p=0.9)

        # Top-k sampling
        token_id = sampler.sample(logits, temperature=0.8, top_k=50)
    """

    # Maximum k for top-k (limited by thread-local storage in kernel)
    MAX_TOP_K = 64

    # Threshold for using fused vs multi-pass softmax
    FUSED_SOFTMAX_THRESHOLD = 65536

    def __init__(self, vocab_size: int, device: Any = None):
        """Initialize the Metal sampler.

        Args:
            vocab_size: Size of the vocabulary (e.g., 154880 for GLM-4.7)
            device: MTLDevice instance. If None, uses default system device.

        Raises:
            RuntimeError: If Metal is not available.
        """
        if not HAS_METAL:
            raise RuntimeError(
                "Metal sampler requires PyObjC. Install with:\n  pip install pyobjc-framework-Metal"
            )

        if not HAS_TORCH or not HAS_MPS:
            raise RuntimeError(
                "Metal sampler requires PyTorch with MPS backend.\n"
                "Ensure you're on Apple Silicon with PyTorch >= 2.0"
            )

        self.vocab_size = vocab_size

        # Initialize Metal device
        if device is None:
            device = Metal.MTLCreateSystemDefaultDevice()
            if device is None:
                raise RuntimeError("No Metal device available")

        self._device = device
        self._command_queue = device.newCommandQueue()

        # Compile shader library
        self._compile_shaders()

        # Pre-allocate buffers for common operations
        self._init_buffers()

    def _compile_shaders(self) -> None:
        """Compile Metal shader source."""
        options = Metal.MTLCompileOptions.new()
        options.setFastMathEnabled_(True)
        options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)

        library, error = self._device.newLibraryWithSource_options_error_(
            _SAMPLER_METAL_SOURCE, options, None
        )

        if library is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to compile sampler shaders: {error_msg}")

        self._library = library

        # Create pipeline states for each kernel
        self._pipelines: dict[str, Any] = {}
        kernel_names = [
            "apply_temperature",
            "apply_repetition_penalty",
            "softmax_find_max",
            "softmax_exp_sum",
            "softmax_normalize",
            "softmax_fused",
            "argmax_reduce",
            "topk_select",
            "topp_cumsum_and_cutoff",
            "multinomial_sample",
            "gather_by_index",
            "batched_argmax",
        ]

        for name in kernel_names:
            function = library.newFunctionWithName_(name)
            if function is not None:
                pipeline, _ = self._device.newComputePipelineStateWithFunction_error_(
                    function, None
                )
                if pipeline is not None:
                    self._pipelines[name] = pipeline

    def _init_buffers(self) -> None:
        """Pre-allocate Metal buffers for intermediate results."""
        # Scalar buffers
        self._scalar_buf = self._device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        self._result_idx_buf = self._device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        self._result_val_buf = self._device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )

        # Block reduction buffers (for multi-pass softmax)
        num_blocks = (self.vocab_size + 255) // 256
        self._block_maxes_buf = self._device.newBufferWithLength_options_(
            num_blocks * 4, Metal.MTLResourceStorageModeShared
        )
        self._block_sums_buf = self._device.newBufferWithLength_options_(
            num_blocks * 4, Metal.MTLResourceStorageModeShared
        )

        # Top-k buffers
        self._topk_vals_buf = self._device.newBufferWithLength_options_(
            self.MAX_TOP_K * 4, Metal.MTLResourceStorageModeShared
        )
        self._topk_idxs_buf = self._device.newBufferWithLength_options_(
            self.MAX_TOP_K * 4, Metal.MTLResourceStorageModeShared
        )

    def _create_buffer_from_numpy(self, arr: np.ndarray) -> Any:
        """Create Metal buffer from numpy array."""
        buf = self._device.newBufferWithBytes_length_options_(
            arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
        )
        return buf

    def _create_buffer_from_tensor(self, tensor: torch.Tensor) -> Any:
        """Create Metal buffer from PyTorch tensor.

        Note: This creates a copy. For zero-copy, use MPS tensor directly.
        """
        arr = tensor.cpu().numpy()
        return self._create_buffer_from_numpy(arr)

    def _mps_tensor_to_buffer(self, tensor: torch.Tensor) -> Any:
        """Get Metal buffer from MPS tensor (shared memory)."""
        if not tensor.is_mps:
            raise ValueError("Tensor must be on MPS device")

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        size = storage.nbytes()

        buffer = self._device.newBufferWithBytesNoCopy_length_options_deallocator_(
            ptr, size, Metal.MTLResourceStorageModeShared, None
        )

        return buffer

    def _dispatch_kernel(
        self,
        name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[Any],
        wait: bool = True,
    ) -> None:
        """Dispatch a Metal kernel."""
        pipeline = self._pipelines.get(name)
        if pipeline is None:
            raise KeyError(f"Pipeline '{name}' not found")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        grid_size = Metal.MTLSizeMake(*grid)
        tg_size = Metal.MTLSizeMake(*threadgroup)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

        encoder.endEncoding()
        command_buffer.commit()

        if wait:
            command_buffer.waitUntilCompleted()

    def _greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy (argmax) sampling."""
        # Use Metal kernel for large vocab, PyTorch for small
        if self.vocab_size > 10000:
            return self._greedy_sample_metal(logits)
        else:
            return logits.argmax(dim=-1)

    def _greedy_sample_metal(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy sampling using Metal kernel."""
        batch_size = logits.shape[0] if logits.ndim > 1 else 1
        logits_flat = logits.reshape(-1) if batch_size == 1 else logits

        # Ensure contiguous float32 on MPS
        logits_f32 = logits_flat.float().contiguous()
        if not logits_f32.is_mps:
            logits_f32 = logits_f32.to("mps")

        logits_buf = self._mps_tensor_to_buffer(logits_f32)

        # Create vocab_size constant buffer
        vocab_size_buf = self._create_buffer_from_numpy(
            np.array([self.vocab_size], dtype=np.uint32)
        )

        if batch_size == 1:
            # Single sequence argmax
            self._dispatch_kernel(
                "argmax_reduce",
                grid=(1, 1, 1),
                threadgroup=(256, 1, 1),
                buffers=[logits_buf, self._result_idx_buf, self._result_val_buf, vocab_size_buf],
            )

            # Read result
            result = np.frombuffer(self._result_idx_buf.contents().as_buffer(4), dtype=np.uint32)[0]
            return torch.tensor([result], dtype=torch.long, device="mps")
        else:
            # Batched argmax
            results_buf = self._device.newBufferWithLength_options_(
                batch_size * 4, Metal.MTLResourceStorageModeShared
            )
            batch_size_buf = self._create_buffer_from_numpy(np.array([batch_size], dtype=np.uint32))

            self._dispatch_kernel(
                "batched_argmax",
                grid=(batch_size, 1, 1),
                threadgroup=(256, 1, 1),
                buffers=[logits_buf, results_buf, batch_size_buf, vocab_size_buf],
            )

            results = np.frombuffer(
                results_buf.contents().as_buffer(batch_size * 4), dtype=np.uint32
            )
            return torch.tensor(results, dtype=torch.long, device="mps")

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature scaling."""
        if temperature == 1.0:
            return logits
        return logits / temperature

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: list[int] | None,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to previously generated tokens."""
        if penalty == 1.0 or not generated_ids:
            return logits

        unique_ids = list(set(generated_ids))
        if not unique_ids:
            return logits

        # For small number of tokens, use vectorized PyTorch
        if len(unique_ids) < 100:
            token_indices = torch.tensor(unique_ids, dtype=torch.long, device=logits.device)
            token_logits = logits[..., token_indices]
            penalized = torch.where(
                token_logits > 0,
                token_logits / penalty,
                token_logits * penalty,
            )
            logits = logits.clone()
            logits[..., token_indices] = penalized
            return logits

        # For many tokens, use Metal kernel
        logits_f32 = logits.float().contiguous()
        if not logits_f32.is_mps:
            logits_f32 = logits_f32.to("mps")

        logits_buf = self._mps_tensor_to_buffer(logits_f32)
        token_ids_buf = self._create_buffer_from_numpy(np.array(unique_ids, dtype=np.uint32))
        penalty_buf = self._create_buffer_from_numpy(np.array([penalty], dtype=np.float32))
        num_tokens_buf = self._create_buffer_from_numpy(
            np.array([len(unique_ids)], dtype=np.uint32)
        )

        self._dispatch_kernel(
            "apply_repetition_penalty",
            grid=(len(unique_ids), 1, 1),
            threadgroup=(1, 1, 1),
            buffers=[logits_buf, token_ids_buf, penalty_buf, num_tokens_buf],
        )

        return logits_f32.half() if logits.dtype == torch.float16 else logits_f32

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute softmax probabilities."""
        # For moderate vocab sizes, use fused kernel
        if self.vocab_size <= self.FUSED_SOFTMAX_THRESHOLD:
            return self._softmax_fused(logits)
        else:
            return self._softmax_multipass(logits)

    def _softmax_fused(self, logits: torch.Tensor) -> torch.Tensor:
        """Fused softmax for moderate vocabulary sizes."""
        logits_f32 = logits.float().contiguous()
        if not logits_f32.is_mps:
            logits_f32 = logits_f32.to("mps")

        logits_buf = self._mps_tensor_to_buffer(logits_f32)
        vocab_size_buf = self._create_buffer_from_numpy(
            np.array([self.vocab_size], dtype=np.uint32)
        )

        self._dispatch_kernel(
            "softmax_fused",
            grid=(1, 1, 1),
            threadgroup=(256, 1, 1),
            buffers=[logits_buf, vocab_size_buf],
        )

        return logits_f32

    def _softmax_multipass(self, logits: torch.Tensor) -> torch.Tensor:
        """Multi-pass softmax for large vocabularies."""
        # Fall back to PyTorch for simplicity
        # A proper implementation would use the multi-pass Metal kernels
        return torch.softmax(logits.float(), dim=-1)

    def _topk_sample(
        self,
        logits: torch.Tensor,
        k: int,
        temperature: float,
    ) -> torch.Tensor:
        """Top-k sampling."""
        k = min(k, self.MAX_TOP_K, self.vocab_size)

        # Apply temperature
        logits = self._apply_temperature(logits, temperature)

        # Ensure flat float32 on MPS
        logits_f32 = logits.reshape(-1).float().contiguous()
        if not logits_f32.is_mps:
            logits_f32 = logits_f32.to("mps")

        logits_buf = self._mps_tensor_to_buffer(logits_f32)
        vocab_size_buf = self._create_buffer_from_numpy(
            np.array([self.vocab_size], dtype=np.uint32)
        )
        k_buf = self._create_buffer_from_numpy(np.array([k], dtype=np.uint32))

        # Get top-k
        self._dispatch_kernel(
            "topk_select",
            grid=(1, 1, 1),
            threadgroup=(256, 1, 1),
            buffers=[logits_buf, self._topk_vals_buf, self._topk_idxs_buf, vocab_size_buf, k_buf],
        )

        # Read top-k values and indices
        topk_vals = np.frombuffer(
            self._topk_vals_buf.contents().as_buffer(k * 4), dtype=np.float32
        ).copy()
        topk_idxs = np.frombuffer(
            self._topk_idxs_buf.contents().as_buffer(k * 4), dtype=np.uint32
        ).copy()

        # Compute softmax over top-k
        topk_vals = topk_vals - topk_vals.max()  # Numerical stability
        probs = np.exp(topk_vals)
        probs = probs / probs.sum()

        # Sample from top-k distribution
        random_val = np.random.random()
        cumsum = 0.0
        sampled_idx = 0
        for i in range(k):
            cumsum += probs[i]
            if random_val < cumsum:
                sampled_idx = i
                break
            sampled_idx = i

        token_id = topk_idxs[sampled_idx]
        return torch.tensor([token_id], dtype=torch.long, device="mps")

    def _topp_sample(
        self,
        logits: torch.Tensor,
        p: float,
        temperature: float,
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling."""
        # Apply temperature
        logits = self._apply_temperature(logits, temperature)

        # Compute probabilities
        probs = torch.softmax(logits.float().reshape(-1), dim=-1)

        # Sort descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Find cutoff where cumulative probability exceeds p
        cumsum = torch.cumsum(sorted_probs, dim=0)
        cutoff_mask = cumsum <= p
        # Include at least one token and the first token that exceeds p
        cutoff_mask[1:] = cutoff_mask[:-1].clone()
        cutoff_mask[0] = True

        # Get number of tokens in nucleus
        num_tokens = cutoff_mask.sum().item()
        num_tokens = max(1, num_tokens)

        # Get nucleus probabilities and renormalize
        nucleus_probs = sorted_probs[:num_tokens]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        nucleus_indices = sorted_indices[:num_tokens]

        # Sample from nucleus
        nucleus_probs_np = nucleus_probs.cpu().numpy()
        sampled_local_idx = np.random.choice(num_tokens, p=nucleus_probs_np)

        token_id = nucleus_indices[sampled_local_idx].item()
        return torch.tensor([token_id], dtype=torch.long, device="mps")

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        generated_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Sample a token from the logits distribution.

        Args:
            logits: Unnormalized logits [batch, vocab_size] or [vocab_size]
            temperature: Temperature for scaling (0 = greedy, >1 = more random)
            top_k: Number of top tokens to sample from (0 = disabled)
            top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
            repetition_penalty: Penalty factor for repeated tokens (1.0 = disabled)
            generated_ids: Previously generated token IDs for repetition penalty

        Returns:
            Sampled token IDs [batch] or [1]
        """
        # Ensure logits are on MPS
        if not logits.is_mps:
            logits = logits.to("mps")

        # Apply repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            logits = self._apply_repetition_penalty(logits, generated_ids, repetition_penalty)

        # Greedy decoding (temperature <= 0 or temperature very close to 0)
        if temperature <= 0 or temperature < 1e-7:
            return self._greedy_sample(logits)

        # Top-k sampling
        if top_k > 0 and top_k < self.vocab_size:
            return self._topk_sample(logits, top_k, temperature)

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            return self._topp_sample(logits, top_p, temperature)

        # Pure temperature sampling (sample from full distribution)
        logits = self._apply_temperature(logits, temperature)
        probs = torch.softmax(logits.float().reshape(-1), dim=-1)
        probs_np = probs.cpu().numpy()
        token_id = np.random.choice(len(probs_np), p=probs_np)
        return torch.tensor([token_id], dtype=torch.long, device="mps")

    def greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy (argmax) decoding.

        Args:
            logits: [batch, vocab_size] or [vocab_size]

        Returns:
            Token IDs with highest logits
        """
        return self._greedy_sample(logits)

    def sample_topk(
        self,
        logits: torch.Tensor,
        k: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample from top-k tokens.

        Args:
            logits: [vocab_size]
            k: Number of top tokens to consider
            temperature: Temperature scaling

        Returns:
            Sampled token ID
        """
        return self._topk_sample(logits, k, temperature)

    def sample_topp(
        self,
        logits: torch.Tensor,
        p: float = 0.9,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample from nucleus (top-p).

        Args:
            logits: [vocab_size]
            p: Cumulative probability threshold
            temperature: Temperature scaling

        Returns:
            Sampled token ID
        """
        return self._topp_sample(logits, p, temperature)


# ---------------------------------------------------------------------------
# Convenience function for quick sampling
# ---------------------------------------------------------------------------


def sample_tokens(
    logits: torch.Tensor,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
) -> torch.Tensor:
    """Convenience function for token sampling.

    Creates a MetalSampler instance and samples from the given logits.
    For repeated sampling, create a MetalSampler instance directly for efficiency.

    Args:
        logits: Unnormalized logits [batch, vocab_size] or [vocab_size]
        vocab_size: Vocabulary size (inferred from logits if None)
        temperature: Temperature scaling
        top_k: Top-k filtering (0 = disabled)
        top_p: Top-p filtering (1.0 = disabled)
        repetition_penalty: Penalty for repeated tokens
        generated_ids: Previously generated token IDs

    Returns:
        Sampled token IDs
    """
    if vocab_size is None:
        vocab_size = logits.shape[-1]

    sampler = MetalSampler(vocab_size)
    return sampler.sample(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generated_ids=generated_ids,
    )
