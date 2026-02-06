# MoE Kernel Selection Optimization

This document describes the optimized kernel selection strategy for Mixture of Experts (MoE) operations on Apple M4 Max hardware.

## Overview

The `select_moe_kernel()` function in `metal_marlin/trellis/moe_dispatch.py` chooses the optimal Metal kernel variant based on batch size and other parameters. This selection significantly impacts inference performance.

## Kernel Variants

| Kernel Name | Description | Optimal For |
|-------------|-------------|-------------|
| `moe_trellis_swiglu_decode` | Single-token optimized | batch_size == 1 |
| `moe_trellis_swiglu_decode_6_2_3` | Specialized for 6-2-3 bit config | batch_size == 1, specific bits |
| `moe_trellis_swiglu_decode_6_3_4` | Specialized for 6-3-4 bit config | batch_size == 1, specific bits |
| `moe_trellis_swiglu_prefill4` | 4-token SIMD processing | 2 <= batch_size <= 16 |
| `moe_trellis_swiglu` | Base kernel | 17 <= batch_size <= 32 |
| `moe_trellis_swiglu_large_batch` | tile_n=128 for throughput | batch_size >= 33 |

## Selection Strategy (M4 Max Optimized)

```python
def select_moe_kernel(batch_size, use_fp32_acc=False, gate_bits=None, up_bits=None, down_bits=None):
    """
    Returns: (kernel_name, tile_n)
    
    M4 Max Optimized Thresholds:
    - batch_size == 1: decode kernel (64)
    - 2 <= batch_size <= 16: prefill4 kernel (64)
    - 17 <= batch_size <= 32: base kernel (64)
    - batch_size >= 33: large_batch kernel (128)
    """
```

### Batch Size Ranges

| Batch Size | Kernel | Tile N | Rationale |
|------------|--------|--------|-----------|
| 1 | decode | 64 | Minimal overhead for single token |
| 2-16 | prefill4 | 64 | 4-token SIMD efficiency |
| 17-32 | base | 64 | Balanced throughput |
| 33+ | large_batch | 128 | Memory coalescing optimization |

## Specialized Decode Kernels

For batch_size == 1, specialized kernels exist for specific bit-width combinations:

```python
# Example: GLM-4.7-Flash-Trellis-MM uses 6-2-3 bit configuration
if gate_bits == 6 and up_bits == 2 and down_bits == 3:
    return "moe_trellis_swiglu_decode_6_2_3", 64
```

These kernels have compile-time known dequantization parameters for better instruction scheduling.

## FP32 Accumulation

For `hidden_dim >= 1024` or when numerical stability is critical:

```python
# Use FP32 accumulation variants
select_moe_kernel(batch_size, use_fp32_acc=True)
```

Performance cost: ~5-15% on M4 Max.

## Benchmarking

Run the kernel selection benchmark:

```bash
cd contrib/metal_marlin
uv run benchmarks/bench_m4_kernel_selection.py
```

This generates:
- `results/kernel_selection_m4_max.json` - Raw benchmark data
- `results/kernel_recommendations.md` - Analysis and recommendations

## Future Optimizations

### Kernel Fusion Opportunities

The current implementation has potential for further optimization:

1. **Gate + Up Projection Fusion**: The decode kernel already fuses `gate_proj + up_proj + SwiGLU` into a single memory pass. Similar fusion could be applied to prefill kernels to reduce memory traffic by ~2x.

2. **Dynamic Tile Size**: Instead of fixed thresholds, dynamically select tile_n based on runtime characteristics (cache hit rate, memory bandwidth).

3. **Expert-Aware Batching**: Group tokens by expert assignment before kernel dispatch to improve locality.

## Implementation Notes

The kernel selection is implemented in `metal_marlin/trellis/moe_dispatch.py`:

- Lines 1091-1163: `select_moe_kernel()` function
- Lines 1166-1210: `get_moe_kernel()` convenience wrapper

Key design decisions:
1. Thresholds are hardcoded based on M4 Max profiling
2. Specialized kernels for common bit-width patterns
3. FP32 accumulation as optional (not default) for performance
