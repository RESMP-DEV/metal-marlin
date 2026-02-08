# MPS Advanced Indexing Performance Issue

This document describes a critical performance pathology discovered when using PyTorch advanced tensor indexing on MPS (Metal Performance Shaders) backend, particularly affecting Mixture-of-Experts (MoE) token dispatch.

## Problem Statement

Expert selection via advanced tensor indexing is catastrophically slow on MPS:

```python
# This operation takes 20+ seconds on MPS
selected = expert_weights[topk_indices[:, 0]]  # [batch, in_features, out_features]
```

### Benchmark Results (minimal_repro.py)

| Operation | Time | Relative |
|-----------|------|----------|
| Basic matmul `[128, 4096] @ [4096, 14336]` | 34.6 ms | 1.0x |
| Expert loop (3 iterations) | ~100 ms | ~3x |
| Expert selection with indexing | 20,234 ms | **584x** |

The indexing operation alone accounts for over 99% of the total MoE layer time when using Python-level indexing on MPS tensors.

## Root Cause Analysis

### 1. CPU-GPU Synchronization

MPS advanced tensor indexing triggers implicit synchronization:

```
PyTorch indexing: expert_weights[topk_indices[:, 0]]
        │
        ▼
MPS discovers non-contiguous access pattern
        │
        ▼
Forces GPU queue flush (wait for all pending work)
        │
        ▼
Copies indices to CPU for analysis
        │
        ▼
Copies selected rows back to GPU
        │
        ▼
Resume GPU execution
```

Each indexing operation may incur multiple synchronization points, serializing what should be a parallel operation.

### 2. Non-Contiguous Memory Access

When `topk_indices` contains non-sequential expert IDs (e.g., `[3, 7, 1, 5, ...]`), the resulting access pattern is non-contiguous:

```
expert_weights layout (contiguous):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│Expert 0 │Expert 1 │Expert 2 │Expert 3 │Expert 4 │Expert 5 │...
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Requested access pattern (non-contiguous):
        │         │                   │                   │
        ▼         ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│       [3]      [7]                 [1]                 [5]  │
└─────────────────────────────────────────────────────────────┘

Result: Scattered reads that cannot coalesce
```

### 3. Tensor Copy Overhead

MPS may copy the entire source tensor to satisfy advanced indexing:

```python
# expert_weights: [64, 4096, 14336] = 29 GB in FP16
# topk_indices[:, 0]: [128] indices

# MPS behavior (observed):
# 1. Copy full expert_weights to staging buffer
# 2. Extract requested slices
# 3. Copy result back to GPU
```

With 64 experts at 4096x14336 FP16, this means potentially moving 29 GB of data to extract 128 rows.

### 4. Missing Scatter/Gather Optimization

CUDA backends optimize `tensor[indices]` patterns via:
- `cudaMemcpy2DAsync` for strided copies
- Dedicated gather/scatter kernels
- Index buffer caching

MPS lacks equivalent optimizations for advanced indexing, falling back to element-wise operations.

## Solutions Explored

### 1. Batched Expert Dispatch (Implemented)

Group tokens by expert ID before processing, eliminating runtime indexing:

```python
# Instead of: output = expert_weights[topk_indices] @ x
# Do:
for expert_idx in range(num_experts):
    mask = (topk_indices == expert_idx)
    if mask.any():
        tokens = x[mask]
        output[mask] = expert_weights[expert_idx] @ tokens
```

See [Expert Weight Batching](../internals/expert_batching.md) for implementation details.

**Pros:**
- Avoids advanced indexing entirely
- Uses contiguous weight access
- Works with existing kernels

**Cons:**
- Python loop overhead
- Multiple kernel launches
- No batching across experts

### 2. CPU-Side Index Preparation

Pre-compute dispatch tables on CPU before GPU execution:

```python
# Prepare on CPU
dispatch = prepare_expert_dispatch(topk_indices.cpu())

# Use prepared indices (no advanced indexing at runtime)
for expert_id, token_batch in dispatch.items():
    ...
```

**Pros:**
- Eliminates GPU-side indexing
- CPU index operations are fast

**Cons:**
- Requires CPU-GPU sync for indices
- Adds dispatch preparation latency

### 3. Custom Metal Kernels for Gather/Scatter (Planned)

Implement Metal shaders that handle batched expert selection:

```metal
kernel void moe_gather_experts(
    device const half* expert_weights,  // [num_experts, in, out]
    device const uint* expert_ids,      // [batch]
    device half* gathered_weights,      // [batch, in, out]
    constant GatherParams& params,
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint expert_id = expert_ids[batch_idx];
    uint weight_offset = expert_id * params.in_features * params.out_features;

    // Direct indexed copy (no Python indexing overhead)
    for (uint i = tid.y; i < params.in_features; i += params.threads_per_batch) {
        // ... copy row i from expert_id to gathered_weights
    }
}
```

**Pros:**
- Native GPU gather operation
- No CPU-GPU synchronization
- Can fuse with subsequent GEMM

**Cons:**
- Additional kernel development
- Memory for gathered weights

### 4. Expert Grouping with Padding (Planned)

Group tokens by expert and pad to uniform batch size for batched GEMM:

```
Before grouping:
  Token 0 → Expert 3
  Token 1 → Expert 1
  Token 2 → Expert 3
  Token 3 → Expert 5
  Token 4 → Expert 1

After grouping + padding:
  Expert 1: [Token 1, Token 4, PAD]
  Expert 3: [Token 0, Token 2, PAD]
  Expert 5: [Token 3, PAD, PAD]

Batched GEMM: [3 experts, 3 tokens, hidden] @ [3 experts, hidden, out]
```

**Pros:**
- Single batched GEMM for all experts
- Good GPU utilization

**Cons:**
- Padding waste for unbalanced experts
- Requires custom scatter-back operation

## Recommended Fix

### Primary: Batched Expert Dispatch in Metal

Implement a fused Metal kernel that:

1. Takes pre-sorted token assignments (grouped by expert)
2. Processes all tokens for each expert in parallel threadgroups
3. Accumulates weighted outputs directly to final buffer

This avoids Python-level indexing entirely and keeps all operations on GPU.

### Implementation Path

1. **CPU side**: Sort tokens by expert ID, compute offsets
2. **GPU side**: Single kernel dispatch with expert batches
3. **Accumulation**: Use `index_add_` equivalent in Metal for scatter-back

See `moe_dispatch_optimized.metal` for work-in-progress implementation.

### Immediate Workaround

Until fused kernels are wired up, use the grouped processing pattern from `trellis/moe.py`:

```python
# Group tokens by expert (CPU)
sort_order = torch.argsort(flat_expert_ids)
expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)

# Process per-expert batches (avoids indexing large weight tensors)
for expert_idx in range(num_experts):
    if expert_counts[expert_idx] == 0:
        continue
    # Process batch for this expert
    ...
```

This trades Python loop overhead for avoiding the catastrophic indexing penalty.

## Performance Comparison

| Approach | Indexing Time | Total MoE Time | Status |
|----------|---------------|----------------|--------|
| Naive (per-token indexing) | 20,234 ms | ~20,300 ms | Baseline |
| Batched expert loop | 0 ms (avoided) | ~300 ms | Implemented |
| Metal fused dispatch | 0 ms (avoided) | ~50 ms (est.) | Planned |

## Test Reproduction

Run `minimal_repro.py` to reproduce the issue:

```bash
cd contrib/metal_marlin
uv run python minimal_repro.py
```

Expected output:
```
Minimal MoE Slow-down Reproduction:
============================================================

1. Basic Metal matmul:
   34.6ms

2. Expert loop (3 iterations):
   103.2ms (3.0x)

3. Expert selection with indexing:
   20234.5ms
```

The 580x slowdown in step 3 demonstrates the MPS indexing pathology.

## Related Documentation

- [MoE Architecture](../concepts/moe_architecture.md): High-level MoE design for Metal
- [Expert Weight Batching](../internals/expert_batching.md): Token grouping optimization
- [Memory Access Patterns](../internals/memory_access_patterns.md): Coalesced access strategies
- [Batched GEMM](../internals/batched_gemm.md): Batched matrix multiplication kernels

## References

- PyTorch MPS Backend: https://pytorch.org/docs/stable/notes/mps.html
- Apple Metal Best Practices: https://developer.apple.com/documentation/metal/gpu_programming_guide
- vLLM FusedMoE (CUDA reference): https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe
