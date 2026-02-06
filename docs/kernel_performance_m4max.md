# MoE Kernel Performance Guide (M4 Max)

This document describes the performance characteristics of different MoE kernel variants and how the kernel selection logic optimizes for different batch sizes.

## Kernel Variants

The Metal MoE implementation provides several kernel variants optimized for different use cases:

### 1. Decode Kernels (batch_size = 1)

Optimized for single-token generation (autoregressive decode phase).

| Kernel | Description | Tile N | Best For |
|--------|-------------|--------|----------|
| `moe_trellis_swiglu_decode` | Generic decode kernel | 64 | Single token, any bit-width |
| `moe_trellis_swiglu_decode_6_2_3` | Specialized: gate=6, up=2, down=3 | 64 | GLM-4.7-Flash dominant config |
| `moe_trellis_swiglu_decode_6_3_4` | Specialized: gate=6, up=3, down=4 | 64 | Alternative config |
| `moe_trellis_swiglu_decode_6_2_4` | Specialized: gate=6, up=2, down=4 | 64 | Alternative config |

**Performance**: ~0.5-1.0 ms/token (memory-bound)

**Specialization Benefits**:
- Compile-time known dequant parameters (bit shifts/masks)
- Better instruction scheduling due to constant propagation
- Reduced register pressure
- ~5-10% speedup over generic decode kernel

### 2. Prefill4 Kernels (2 <= batch_size <= 16)

Optimized for small-batch processing with 4-token SIMD efficiency.

| Kernel | Description | Tile N | Best For |
|--------|-------------|--------|----------|
| `moe_trellis_swiglu_prefill4` | Standard prefill4 | 64 | Batch 2-16, fp16 accumulation |
| `moe_trellis_swiglu_prefill4_fp32acc` | FP32 accumulation | 64 | Batch 2-16, precision-critical |

**Performance**: ~0.3-0.5 ms/token

**Characteristics**:
- Processes 4 tokens per threadgroup
- Good SIMD utilization for small batches
- Better memory coalescing than base kernel at small batch sizes

### 3. Base Kernels (17 <= batch_size <= 32)

General-purpose kernels for medium batch sizes.

| Kernel | Description | Tile N | Best For |
|--------|-------------|--------|----------|
| `moe_trellis_swiglu` | Standard base kernel | 64 | General use, batch 17-32 |
| `moe_trellis_swiglu_fp32acc` | FP32 accumulation | 64 | Numerical stability |

**Performance**: ~0.2-0.3 ms/token (transition to compute-bound)

**Characteristics**:
- Balanced for medium batch sizes
- Good register utilization
- Supports both fp16 and fp32 accumulation

### 4. Large Batch Kernel (batch_size >= 33)

Optimized for maximum throughput with large batches.

| Kernel | Description | Tile N | Best For |
|--------|-------------|--------|----------|
| `moe_trellis_swiglu_large_batch` | Large batch optimized | 128 | Maximum throughput |

**Performance**: ~0.15-0.2 ms/token (compute-bound)

**Characteristics**:
- Uses tile_n=128 for better memory coalescing
- ~20-30% better throughput than base kernel for large batches
- No fp32acc variant (falls back to base fp32acc if needed)

### 5. SIMD Kernel

Uses simdgroup_matrix 8x8 operations for specific workloads.

| Kernel | Description | Tile N | Best For |
|--------|-------------|--------|----------|
| `moe_trellis_swiglu_simd` | SIMD-optimized | 64 | Aligned workloads |

**Performance**: Variable, depends on alignment

**Characteristics**:
- Uses Metal simdgroup_matrix operations
- Best for specific matrix dimensions
- Not selected by default in `select_moe_kernel`

## Kernel Selection Logic

The `select_moe_kernel()` function in `metal_marlin/trellis/moe_dispatch.py` implements the following selection strategy:

```python
def select_moe_kernel(batch_size, use_fp32_acc, gate_bits, up_bits, down_bits):
    # batch_size == 1: Use decode kernel
    if batch_size == 1:
        # Check for bit-width specialized kernels
        if gate_bits == 6 and up_bits == 2 and down_bits == 3:
            return "moe_trellis_swiglu_decode_6_2_3", 64
        # ... other specialized kernels
        return "moe_trellis_swiglu_decode", 64

    # 2 <= batch_size <= 16: Use prefill4 kernel
    elif batch_size <= 16:
        if use_fp32_acc:
            return "moe_trellis_swiglu_prefill4_fp32acc", 64
        return "moe_trellis_swiglu_prefill4", 64

    # 17 <= batch_size <= 32: Use base kernel
    elif batch_size <= 32:
        if use_fp32_acc:
            return "moe_trellis_swiglu_fp32acc", 64
        return "moe_trellis_swiglu", 64

    # batch_size >= 33: Use large_batch kernel
    else:
        if use_fp32_acc:
            return "moe_trellis_swiglu_fp32acc", 64
        return "moe_trellis_swiglu_large_batch", 128
```

## Selection Summary Table

| Batch Size | Selected Kernel | Tile N | Notes |
|------------|-----------------|--------|-------|
| 1 | `moe_trellis_swiglu_decode*` | 64 | Bit-specialized variants available |
| 2-16 | `moe_trellis_swiglu_prefill4*` | 64 | 4-token SIMD efficiency |
| 17-32 | `moe_trellis_swiglu*` | 64 | General throughput |
| 33+ | `moe_trellis_swiglu_large_batch` | 128 | Maximum throughput |

\* fp32acc variant when `use_fp32_acc=True` (except decode kernels)

## Benchmarking

To verify kernel selection and measure performance:

```bash
cd contrib/metal_marlin

# Test kernel selection logic
uv run benchmarks/bench_moe_kernel_selection.py

# Full kernel performance benchmark (requires model)
uv run benchmarks/bench_kernel_selection.py
```

## FP32 Accumulation

FP32 accumulation is recommended when:
- Hidden dimension >= 1024
- Numerical stability is critical
- Slight performance degradation (~5-10%) is acceptable

FP32 accumulation kernels:
- `moe_trellis_swiglu_fp32acc`
- `moe_trellis_swiglu_prefill4_fp32acc`

Note: No FP32 accumulation variant exists for decode kernels.

## Kernel Fusion Opportunities

Current kernel fusion status:

| Operation | Status | Notes |
|-----------|--------|-------|
| Gate + Up projection | ✅ Fused | Single kernel does both projections |
| SwiGLU activation | ✅ Fused | Integrated into GEMM kernel |
| Down projection | ✅ Fused | Part of same kernel |
| Expert routing | ✅ Fused | Per-token routing in kernel |
| Top-k selection | ❌ Separate | Done on CPU/GPU before kernel |

The `moe_trellis_swiglu` family of kernels already provides comprehensive fusion:
- Gate projection (GEMM)
- Up projection (GEMM)
- SwiGLU activation (element-wise)
- Down projection (GEMM)
- Expert routing (gather)

All executed in a single dispatch, minimizing memory round-trips and kernel launch overhead.

## Performance Tuning Tips

1. **Use specialized decode kernels**: When running GLM-4.7-Flash or similar models with known bit-width patterns, ensure the specialized decode kernels are used for ~5-10% speedup.

2. **Batch size optimization**: For maximum throughput:
   - Batch 1: Decode kernel (latency-optimized)
   - Batch 2-16: Prefill4 kernel (SIMD efficiency)
   - Batch 33+: Large batch kernel (memory coalescing)

3. **FP32 accumulation**: Enable when numerical issues are observed, particularly for larger hidden dimensions.

4. **Tile size**: The selection logic automatically chooses:
   - Tile N=64 for most kernels
   - Tile N=128 for large batch kernel (better memory throughput)

## Future Optimizations

Potential areas for further optimization:

1. **Dynamic kernel selection**: Runtime profiling to select optimal kernel based on actual hardware performance.

2. **Batch size 2-4 specialization**: Possible dedicated kernel for very small batches.

3. **Mixed-precision kernels**: Support for different bit-widths per expert in single dispatch.

4. **Async dispatch**: Overlap MoE computation with other layers.
