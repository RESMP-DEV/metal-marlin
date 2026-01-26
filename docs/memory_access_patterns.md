# Metal Marlin Memory Access Patterns

This document details memory access pattern issues identified across Metal Marlin kernels and provides optimization recommendations.

## Executive Summary

Memory access patterns are a critical performance factor for GPU kernels. Non-coalesced access can reduce memory throughput by 10-32x. Our analysis identified 60 patterns requiring attention across 6 priority shader files:

| Category | Count | Severity |
|----------|-------|----------|
| Scale buffer strided access | 15 | Warning |
| General strided patterns | 37 | Warning |
| Scattered/indirect access | 8 | Warning |

## Priority Findings

### 1. FP4 Dequantization (CRITICAL)

**File:** `src/dequant.metal`

**Issue:** Sequential scale access in dequantization kernels

```metal
// Current (inefficient)
uint group_idx = base_idx / group_size;
half scale = scales[group_idx];  // Sequential global load per thread group
```

**Impact:** Each thread independently loads scales from global memory. When group_size=128, all 8 threads processing one packed uint32 share a scale, but adjacent thread groups access different scale addresses without coalescing.

**Fix:** Preload scales into threadgroup memory before the main loop:

```metal
// Optimized
threadgroup half scale_cache[MAX_GROUPS_PER_TILE];

// Cooperative scale prefetch
if (tid < groups_in_tile) {
    scale_cache[tid] = scales[tile_group_start + tid];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Main loop uses cached scale
half scale = scale_cache[local_group_idx];
```

**Alternative:** Use `[[texture(N)]]` attribute on scales buffer to leverage texture cache spatial locality.

### 2. Attention Kernels (WARNING)

**File:** `src/attention.metal`

**Issue 1:** K vector strided sequential load

```metal
// Current (strided)
device const half* k_vec = k_base_ptr + k_idx * head_dim;
for (uint d = 0; d < head_dim; ++d) {
    dot += float(Q_cache[d]) * float(k_vec[d]);  // Sequential per-thread
}
```

**Impact:** Each thread loads head_dim values sequentially, causing strided access between threads (stride = head_dim).

**Fix Options:**

1. **Use tiled variant** (`attention_qk_softmax_tiled`): Pre-loads K vectors into threadgroup memory.

2. **Transpose K layout**: Store K as `[head_dim, seq_k]` for coalesced seq_k access.

3. **Simdgroup shuffle transpose**: Use simd_shuffle to transpose K vectors from registers.

**Issue 2:** Two-pass score materialization

```metal
// Pass 1: Write scores to global memory
p_ptr[k_idx] = half(score);
// ...
// Pass 2: Read scores back
float score = float(p_ptr[k_idx]);
```

**Impact:** Round-trip to global memory for softmax normalization.

**Fix:** Keep scores in registers for small seq_k, or use single-pass online softmax (already implemented in `attention_fused_qkv`).

### 3. MoE Router (WARNING)

**File:** `src/moe_router.metal`

**Issue:** Router weight column-major access

```metal
// Current (column-major in row-major storage)
device const half* w_col = router_weights + expert_idx;
for (uint d = 0; d < hidden_dim; ++d) {
    acc += float(h[d]) * float(w_col[d * num_experts]);  // Stride = num_experts
}
```

**Impact:** Strided memory access with stride equal to num_experts (typically 64-256).

**Fix:** Transpose router weights to `[num_experts, hidden_dim]` row-major:

```metal
// Optimized (row-major weights)
device const half* w_row = router_weights + expert_idx * hidden_dim;
for (uint d = 0; d < hidden_dim; ++d) {
    acc += float(h[d]) * float(w_row[d]);  // Coalesced
}
```

### 4. MoE Expert GEMM (WARNING)

**File:** `src/moe_expert_gemm.metal`

**Issue:** Scattered token activation gather

```metal
// Current (scattered via indirect index)
uint token_id = token_batch[row];
val = activations[token_id * hidden_dim + col];  // Scattered access
```

**Impact:** Indirect indexing causes non-coalesced global loads when tokens have diverse expert assignments.

**Fix:** Pre-sort tokens by expert ID on the host before dispatch:

```python
# Host-side token sorting
sorted_indices = np.argsort(expert_ids)
sorted_tokens = tokens[sorted_indices]
sorted_probs = probs[sorted_indices]
```

This ensures tokens with the same expert assignment are adjacent in memory.

## General Patterns

### Coalesced (Good) Patterns

```metal
// Direct thread ID indexing
data[tid]
buffer[base + tid]

// Vectorized writes (8-byte aligned)
output[tid * 2] = lo;      // half4 store
output[tid * 2 + 1] = hi;
```

### Non-Coalesced (Bad) Patterns

```metal
// Strided by matrix dimension
buffer[row * N + col]      // Thread varies row, stride = N

// Column-major access in row-major layout
data[col * M + row]

// Indirect indexing
buffer[indices[tid]]       // Scattered if indices are non-sequential
```

## Optimization Checklist

### For GEMM Kernels

- [ ] Tile A and B matrices into threadgroup memory
- [ ] Use simdgroup_matrix ops for 8x8 tiles
- [ ] Double/triple buffer to hide memory latency
- [ ] Vectorize loads/stores with half4/float4
- [ ] Cache scales in threadgroup memory

### For Attention Kernels

- [ ] Use tiled variant for seq_k > 64
- [ ] Consider K transpose for very long sequences
- [ ] Fuse softmax normalization to avoid score re-read
- [ ] Use flash attention for memory efficiency

### For MoE Kernels

- [ ] Transpose router weights to row-major
- [ ] Sort tokens by expert before dispatch
- [ ] Use grouped kernel for better locality
- [ ] Consider expert-local output buffers

## Tools

### Memory Audit Tool

Run the memory audit to analyze your kernels:

```python
from metal_marlin.profiling import MemoryAuditor, generate_full_report

# Audit all shaders
auditor = MemoryAuditor()
report = auditor.audit_all()
print(report.summary())

# Or audit specific priority kernels
report = auditor.audit_priority_kernels()

# Generate detailed report
generate_full_report(output_path="audit_report.txt")
```

### Runtime Benchmark

Validate coalescing with runtime benchmarks:

```python
from metal_marlin.profiling import run_coalescing_benchmark

results = run_coalescing_benchmark(pattern="all", sizes=[4096, 16384])
for pattern, data in results.items():
    print(f"{pattern}: {data}")
```

## References

- [NVIDIA GPU Memory Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [Apple Metal Best Practices Guide](https://developer.apple.com/documentation/metal/metal_best_practices_guide)
- [Marlin CUDA Implementation](https://github.com/IST-DASLab/marlin)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
