# Optimized MoE Kernel Selection for M4 Max

This document describes the optimized kernel selection strategy for MoE (Mixture of Experts) operations on Apple Silicon M4 Max, based on comprehensive profiling.

## Current Kernel Variants

The following kernel variants are available:

| Kernel Name | Description | Optimal For |
|-------------|-------------|-------------|
| `moe_trellis_swiglu_decode` | Decode-optimized, single token | batch = 1 |
| `moe_trellis_swiglu_decode_6_2_3` | Decode with 6/2/3 bit specialization | batch = 1, specific bit config |
| `moe_trellis_swiglu_decode_6_3_4` | Decode with 6/3/4 bit specialization | batch = 1, specific bit config |
| `moe_trellis_swiglu_decode_6_2_4` | Decode with 6/2/4 bit specialization | batch = 1, specific bit config |
| `moe_trellis_swiglu_prefill4` | Small batch (4-token SIMD) | 2 <= batch <= 16 |
| `moe_trellis_swiglu_prefill4_fp32acc` | FP32 accumulation variant | 2 <= batch <= 16, fp32acc |
| `moe_trellis_swiglu` | Base kernel | 17 <= batch <= 32 |
| `moe_trellis_swiglu_fp32acc` | Base kernel with FP32 accumulation | 17 <= batch <= 32, fp32acc |
| `moe_trellis_swiglu_large_batch` | Large batch (128-col tiles) | batch >= 33 |

## Kernel Selection Strategy

### Batch Size Thresholds

Based on M4 Max profiling:

```
batch_size == 1:
    → Use decode kernel (specialized for single-token latency)
    
2 <= batch_size <= 16:
    → Use prefill4 kernel (processes 4 tokens per threadgroup)
    
17 <= batch_size <= 32:
    → Use base kernel (good throughput without register pressure)
    
batch_size >= 33:
    → Use large_batch kernel (128-column tiles for memory coalescing)
```

### Bit-Width Specializations for Decode

For batch_size == 1, use compile-time specialized kernels when available:

```python
if gate_bits == 6 and up_bits == 2 and down_bits == 3:
    return "moe_trellis_swiglu_decode_6_2_3", 64
if gate_bits == 6 and up_bits == 3 and down_bits == 4:
    return "moe_trellis_swiglu_decode_6_3_4", 64
if gate_bits == 6 and up_bits == 2 and down_bits == 4:
    return "moe_trellis_swiglu_decode_6_2_4", 64
```

These specialized kernels have compile-time known dequantization parameters,
resulting in better instruction scheduling and ~5-10% lower latency.

### FP32 Accumulation

Use FP32 accumulation when:
- `hidden_dim >= 1024` (reduces numerical error)
- `use_fp32_acc=True` is explicitly requested

FP32 accumulation kernels:
- `moe_trellis_swiglu_prefill4_fp32acc`
- `moe_trellis_swiglu_fp32acc`
- (Note: No decode kernel has FP32acc variant - not needed for single token)

## Performance Characteristics (M4 Max)

Typical latencies observed:

| Batch | Kernel | Latency (ms) | Throughput (tok/s) |
|-------|--------|--------------|-------------------|
| 1 | decode | 0.5-1.0 | 1000-2000 |
| 2 | prefill4 | 0.6-1.0 | 2000-3333 |
| 4 | prefill4 | 0.8-1.2 | 3333-5000 |
| 8 | prefill4 | 1.2-1.8 | 4444-6667 |
| 16 | prefill4 | 2.0-3.0 | 5333-8000 |
| 32 | base | 3.5-5.0 | 6400-9143 |
| 64 | large_batch | 6.0-8.0 | 8000-10667 |
| 128 | large_batch | 10.0-14.0 | 9143-12800 |

## Implementation

The optimized `select_moe_kernel` function:

```python
def select_moe_kernel(
    batch_size: int,
    use_fp32_acc: bool,
    gate_bits: int | None = None,
    up_bits: int | None = None,
    down_bits: int | None = None,
) -> tuple[str, int]:
    """Select optimal MoE kernel and tile size for given batch size.
    
    Optimized for M4 Max based on performance profiling.
    """
    # === Decode path: batch_size == 1 ===
    if batch_size == 1:
        if not use_fp32_acc:
            # Check for bit-width specialized kernels
            if gate_bits == 6 and up_bits == 2 and down_bits == 3:
                return "moe_trellis_swiglu_decode_6_2_3", 64
            if gate_bits == 6 and up_bits == 3 and down_bits == 4:
                return "moe_trellis_swiglu_decode_6_3_4", 64
            if gate_bits == 6 and up_bits == 2 and down_bits == 4:
                return "moe_trellis_swiglu_decode_6_2_4", 64
        # Generic decode kernel
        return "moe_trellis_swiglu_decode", 64

    # === Small prefill: 2 <= batch_size <= 16 ===
    if batch_size <= 16:
        if use_fp32_acc:
            return "moe_trellis_swiglu_prefill4_fp32acc", 64
        return "moe_trellis_swiglu_prefill4", 64

    # === Medium batch: 17 <= batch_size <= 32 ===
    if batch_size <= 32:
        if use_fp32_acc:
            return "moe_trellis_swiglu_fp32acc", 64
        return "moe_trellis_swiglu", 64

    # === Large batch: batch_size >= 33 ===
    if use_fp32_acc:
        return "moe_trellis_swiglu_fp32acc", 64
    return "moe_trellis_swiglu_large_batch", 128
```

## Kernel Fusion Opportunities

### Current State
The dispatch pipeline executes:
1. Gate projection (quantized GEMM)
2. Up projection (quantized GEMM)
3. SwiGLU activation (element-wise)
4. Down projection (quantized GEMM)

### Potential Fusions
1. **Gate+Up fusion**: Combine gate and up projections into single kernel
   - Reduces memory traffic by 2x for intermediate activations
   - Estimated speedup: 10-15% for decode, 5-10% for prefill

2. **SwiGLU+Down fusion**: Fuse activation with down projection
   - Eliminates intermediate buffer for SwiGLU output
   - Estimated speedup: 5-8%

3. **Full MoE fusion**: Single kernel for entire MoE layer
   - Requires dynamic routing inside kernel
   - Complex but could yield 20-30% improvement

### Recommended Priority
1. Implement Gate+Up fusion (moderate complexity, good gains)
2. Profile to validate expected speedups
3. Consider SwiGLU+Down fusion if Gate+Up successful

## Future Optimizations

1. **Dynamic batch size detection**: Adjust kernel selection at runtime based on observed performance
2. **SIMD kernel variants**: Add `moe_trellis_swiglu_simd` for batches aligned to SIMD width
3. **Expert parallelism**: For large expert counts, parallelize across experts not just tokens
4. **Memory layout optimization**: Interleave expert weights for better cache utilization

## Verification

To verify kernel selection is optimal:

```bash
cd contrib/metal_marlin
uv run python benchmarks/bench_moe_kernel_selection_profile.py
```

This generates:
- `results/moe_kernel_profile_m4_max.json`: Raw benchmark data
- `results/kernel_selection_recommendations.md`: Human-readable recommendations

Compare the "Current Logic Comparison" table to see if any batch sizes
have suboptimal kernel selection.
