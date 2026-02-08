# FastRouterDispatcher - Optimized CPU Router for MoE

## Overview

`FastRouterDispatcher` is a high-performance CPU-side router dispatcher for Mixture of Experts (MoE) models. It handles the critical-path routing computation that determines which experts process each token, with optimizations for both decode and prefill workloads.

## Key Features

### 1. Batched Router Forward Pass
- Processes multiple tokens simultaneously for better cache utilization
- BF16→FP32 conversion with SIMD acceleration
- Efficient matrix-vector multiplication for router logits

### 2. SIMD-Optimized Top-K Selection
- ARM NEON and SSE2 implementations for max-reduction
- Optimized dot product with SIMD vectorization
- Fast top-k selection using insertion sort with early termination

### 3. Pre-computed Expert Indices
- Expert pair lookup table for fast top-2 path
- Pre-computed dispatch info to reduce runtime overhead
- Cache-friendly memory layout

### 4. Async Execution with Double-Buffering
- Overlap CPU router compute with GPU expert execution
- Two-buffer design for pipelining consecutive tokens
- Sequence tracking for buffer management

### 5. Hot Expert Pair Caching
- Tracks frequently-used expert combinations
- Fast path for common pairs (e.g., expert 5+12)
- Configurable cache capacity and threshold

## Performance Characteristics

### Decode Path (1 token)
- **Router forward**: 358K tokens/sec
- **Async dispatch**: 2.0M tokens/sec
- **Latency**: 2.8 μs per token

### Prefill Path (32 tokens)
- **Router forward**: 1.7M tokens/sec
- **With hot pair cache**: ~26% speedup after warmup
- **Throughput**: 2.1M tokens/sec (async)

### Large Models (64 experts)
- **Router forward**: 195K tokens/sec
- **Scalability**: O(num_experts * hidden_dim) complexity
- **Cache effectiveness**: 10-32 hot pairs detected

## Usage

### C++ API

```cpp
#include "moe_router_dispatch.hpp"

using namespace metal_marlin::moe;

// Create dispatcher
FastRouterDispatcher router(
    num_experts=8,
    hidden_dim=128,
    top_k=2,
    max_batch_tokens=128,
    hot_pair_cache_capacity=256,
    hot_pair_threshold=8
);

// Synchronous batch processing
auto output = router.route_batch(
    token_activations_bf16,  // [num_tokens, hidden_dim] BF16
    num_tokens,
    router_weights,          // [num_experts, hidden_dim] FP32
    router_bias              // [num_experts] FP32 (optional)
);

// Access results
for (int t = 0; t < output.num_tokens; ++t) {
    for (int k = 0; k < output.top_k; ++k) {
        int32_t expert_id = output.topk_expert_ids[t * output.top_k + k];
        float prob = output.topk_probs[t * output.top_k + k];
        // Use expert_id and prob...
    }
}

// Async execution with double-buffering
auto expert_launch_fn = [](const RouterBuffer& buffer) {
    // Launch GPU expert computation
    launch_experts(buffer);
};

// Submit async - CPU computes router while GPU processes previous token
auto future = router.submit_async(
    token_activations_bf16,
    num_tokens,
    router_weights,
    router_bias,
    expert_launch_fn
);
```

### Python API

```python
from metal_marlin import FastRouterDispatcher

# Create dispatcher
router = FastRouterDispatcher(
    num_experts=8,
    hidden_dim=128,
    top_k=2
)

# Process batch
output = router.route_batch(
    activations,  # BF16 numpy array [batch, hidden_dim]
    weights,      # FP32 numpy array [num_experts, hidden_dim]
    bias=None     # Optional
)

# Access results
expert_ids = output.topk_expert_ids  # [batch, top_k]
probs = output.topk_probs           # [batch, top_k]
dispatch_info = output.dispatch_info

# Hot pair caching
print(f"Hot pairs: {router.hot_pair_count()}")
router.reset_hot_pair_cache()  # Clear cache
```

## Architecture

### Components

1. **RouterForward**: Computes router logits from activations and weights
   - BF16→FP32 conversion loop
   - Matrix-vector multiplication per token
   - Bias addition (optional)

2. **TopKSelector**: Selects top-k experts per token
   - SIMD max-reduction for top-1
   - Insertion sort for top-k (k=2 in practice)
   - Probability normalization

3. **DispatchBuilder**: Packs expert assignments for efficient execution
   - Token sorting by expert
   - Offset computation per expert
   - Inverse indices for result scattering

4. **HotPairCache**: Caches frequently-used expert combinations
   - Frequency tracking
   - LRU-style eviction
   - Fast path for top-2 dispatch

### Double-Buffering

```
Token N-1: [CPU idle]    [GPU executing experts N-1]   [GPU executing experts N-1]
Token N:   [CPU router N] [GPU executing experts N-1]   [CPU idle]
Token N+1: [CPU idle]     [CPU router N+1]            [GPU executing experts N]
```

This design hides CPU router latency behind GPU expert computation, reducing overall token latency.

## Integration with MoE Pipeline

```cpp
// MoE forward pass using FastRouterDispatcher
FastRouterDispatcher router(num_experts, hidden_dim, top_k);

// Compute routing on CPU
auto routing = router.route_batch(activations, batch_size, router_weights, router_bias);

// Use routing info to gather and dispatch to experts
auto& dispatch = routing.dispatch_info;

// Gather activations per expert
for (int e = 0; e < num_experts; ++e) {
    int32_t expert_batch_size = dispatch.expert_batch_size(e);
    if (expert_batch_size == 0) continue;
    
    // Get tokens for this expert
    auto token_indices = get_expert_tokens(dispatch, e);
    auto gathered_activations = gather_activations(activations, token_indices);
    
    // Run expert computation on GPU
    auto expert_output = execute_expert_gpu(e, gathered_activations);
    
    // Scatter back
    scatter_outputs(outputs, expert_output, token_indices, routing.topk_probs);
}
```

## Testing

### Run Unit Tests

```bash
cd contrib/metal_marlin/cpp
mkdir build_test_router && cd build_test_router
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./test_router_dispatch
```

Expected output:
```
=== FastRouterDispatcher Unit Tests ===
Test: Basic router forward pass... PASSED
Test: Async execution with double-buffering... PASSED
Test: Hot pair caching... (16 hot pairs detected) PASSED
Test: Dispatch info packing... PASSED
Test: Batch processing consistency... PASSED
=== All tests PASSED ===
```

### Run Benchmarks

```bash
./bench_router_dispatch
```

Measures:
- BF16→FP32 conversion throughput
- Router forward pass latency/throughput
- Async dispatch speedup
- Hot pair cache hit rate

## Performance Optimization Tips

1. **Use async dispatch** for decode path to hide CPU latency
2. **Increase hot_pair_cache_capacity** for diverse workloads
3. **Batch multiple tokens** for prefill to improve cache utilization
4. **Set hot_pair_threshold** based on expert diversity (8-16 typical)
5. **Use SIMD-optimized build** with `-march=native` for ARM/Intel

## Limitations and Future Work

### Current Limitations
- Router weights must be FP32 (no quantization support)
- Top-k limited to k=8 (practical use case: k=2)
- No multi-threading (router compute is CPU-bound)

### Future Enhancements
- **Quantized router weights**: INT8/FP8 support for memory efficiency
- **Multi-threaded routing**: Parallelize token processing
- **Hybrid GPU routing**: Offload router to GPU for very large batches
- **Dynamic top-k**: Adaptive k based on expert confidence
- **GPU-side caching**: Cache expert outputs for repeated patterns

## References

- Implementation: `cpp/src/moe_router_dispatch.cpp`
- Header: `cpp/include/moe_router_dispatch.hpp`
- Tests: `cpp/tests/test_router_dispatch.cpp`
- Benchmarks: `cpp/tests/bench_router_dispatch.cpp`
- Python bindings: `cpp/src/python_bindings.mm`
