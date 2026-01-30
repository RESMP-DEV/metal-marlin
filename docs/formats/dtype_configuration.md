# Dtype Configuration

This document describes the centralized dtype configuration system in Metal Marlin, which controls precision throughout the inference pipeline.

## Overview

Metal Marlin uses a unified `DTypeConfig` class to manage data types across all components:

- Weight dequantization
- Activation precision
- GEMM accumulation
- Quantization scale storage
- KV cache storage

This centralized approach replaces scattered hardcoded `mx.float16` references with configurable, consistent dtype handling.

## DTypeConfig Class

```python
from metal_marlin.dtypes import DTypeConfig, get_default_config

# Create custom configuration
config = DTypeConfig(
    weights="bf16",        # Dequantized weight precision
    activations="bf16",    # Input/output activation precision
    accumulation="fp32",   # GEMM accumulator (always FP32)
    scales="fp16",         # Quantization scale storage
    kv_cache="bf16",       # KV cache storage format
)

# Access MLX dtypes
print(config.mlx_weights)      # mx.bfloat16
print(config.mlx_activations)  # mx.bfloat16
print(config.mlx_scales)       # mx.float16

# Access numpy dtypes (for packing/unpacking)
print(config.numpy_scales)     # np.float16

# Access Metal shader type names
print(config.metal_weights)    # "bfloat"
```

### Field Reference

| Field | Type | Default | Options | Description |
|-------|------|---------|---------|-------------|
| `weights` | `WeightDType` | `"bf16"` | `"fp16"`, `"bf16"` | Dtype after weight dequantization |
| `activations` | `ActivationDType` | `"bf16"` | `"fp16"`, `"bf16"` | Input and output activation dtype |
| `accumulation` | `AccumulationDType` | `"fp32"` | `"fp32"` | GEMM accumulator (fixed) |
| `scales` | `ScaleDType` | `"fp16"` | `"fp16"`, `"fp32"` | Quantization scale storage |
| `kv_cache` | `KVCacheDType` | `"bf16"` | `"fp16"`, `"bf16"`, `"fp8"` | KV cache storage format |

## Preset Configurations

Metal Marlin provides four preset configurations for common use cases:

### fp16_config() - Maximum Compatibility

```python
from metal_marlin.dtypes import fp16_config

config = fp16_config()
# weights="fp16", activations="fp16", accumulation="fp32",
# scales="fp16", kv_cache="fp16"
```

Use when:
- Running on older hardware without BF16 support
- Maximum compatibility is required
- Memory is not a constraint

### bf16_config() - Better Dynamic Range (Default)

```python
from metal_marlin.dtypes import bf16_config

config = bf16_config()
# weights="bf16", activations="bf16", accumulation="fp32",
# scales="fp16", kv_cache="bf16"
```

Use when:
- Running on Apple Silicon M1 Pro/Max or newer
- Training or fine-tuning (better gradient flow)
- Working with models that have large weight magnitudes

### memory_efficient_config() - Long Context

```python
from metal_marlin.dtypes import memory_efficient_config

config = memory_efficient_config()
# weights="fp16", activations="fp16", accumulation="fp32",
# scales="fp16", kv_cache="fp8"
```

Use when:
- Running long context inference (>8K tokens)
- Memory is constrained
- Slight accuracy loss is acceptable

### high_precision_config() - Accuracy Critical

```python
from metal_marlin.dtypes import high_precision_config

config = high_precision_config()
# weights="bf16", activations="bf16", accumulation="fp32",
# scales="fp32", kv_cache="bf16"
```

Use when:
- Maximum accuracy is required
- Memory is not a concern
- Running evaluation or benchmarks

## FP16 vs BF16

### Representation Comparison

| Property | FP16 | BF16 |
|----------|------|------|
| Sign bits | 1 | 1 |
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Dynamic range | ±65,504 | ±3.4e38 |
| Precision | 3-4 decimal digits | 2-3 decimal digits |

### When to Use FP16

- **Precision-sensitive operations**: FP16 has 3 more mantissa bits
- **Well-normalized values**: When values stay within ±65,504
- **Legacy compatibility**: Some older code paths assume FP16
- **Softmax outputs**: Probability distributions benefit from precision

### When to Use BF16

- **Large weight magnitudes**: BF16 handles ±3.4e38 without overflow
- **Gradients**: Training benefits from BF16's dynamic range
- **After RMS normalization**: Dynamic range matters more than precision
- **Transformer residuals**: Accumulated residuals can grow large

### Apple Silicon Considerations

Apple Silicon M1 Pro/Max and newer support BF16 natively. However, a Metal compiler bug affects FP16 arithmetic in inline functions (see `docs/metal_half_precision_bug.md`). The workaround uses float intermediates, making the precision choice mostly about memory layout and numerical stability rather than raw performance.

## FP8 KV Cache Tradeoffs

The KV cache can be stored in FP8 format to reduce memory usage by 2x compared to FP16/BF16. This is particularly valuable for long context inference where the KV cache dominates memory.

### Memory Savings

| Format | Bytes/element | 4K context (32L, 8H, D=128) | 32K context |
|--------|---------------|------------------------------|-------------|
| FP16/BF16 | 2.0 | 4.0 GB | 32.0 GB |
| FP8 | 1.0 + scale overhead | ~2.1 GB | ~16.5 GB |

### Accuracy Impact

FP8 E4M3 has limited precision (3 mantissa bits, 4 exponent bits), which can introduce small errors in attention computations:

- **Attention score precision**: QK^T products may have slight rounding errors
- **Value projection precision**: PV accumulation is similarly affected
- **Compounding effects**: Errors can compound across layers

Typical impact: 0.01-0.05 perplexity increase on language modeling tasks, negligible for most inference applications.

### When to Use FP8 KV Cache

**Recommended:**
- Context length >8K tokens
- Memory pressure is limiting batch size or context
- Task tolerance for slight precision loss (chat, summarization)

**Not recommended:**
- Accuracy-critical evaluation
- Short context where FP16 fits comfortably
- Tasks requiring exact numerical reproducibility

### Example Configuration

```python
from metal_marlin.dtypes import DTypeConfig

# Long context with FP8 KV cache
config = DTypeConfig(
    weights="bf16",
    activations="bf16",
    accumulation="fp32",
    scales="fp16",
    kv_cache="fp8",  # 2x memory savings
)
```

## Accumulation Precision

**Accumulation is always FP32.** This is not configurable because it directly impacts numerical stability.

### Why FP32 Accumulation Matters

GEMM operations compute dot products that sum many terms:

```
C[i,j] = sum(A[i,k] * B[k,j] for k in range(K))
```

For large K (>2048), FP16 accumulation causes:
- **Overflow**: Values exceed FP16 max (65,504)
- **Catastrophic cancellation**: Small differences lost to rounding
- **Saturation**: Sums clamp to max representable value

### Metal Implementation

Metal's `simdgroup_matrix` uses FP32 accumulation internally:

```metal
// FP32 accumulator with FP16 inputs
simdgroup_matrix<float, 8, 8> C_acc;

// Inputs can be half
simdgroup_matrix<half, 8, 8> A_tile, B_tile;

// MMA accumulates in FP32
simdgroup_multiply_accumulate(C_acc, A_tile, B_tile, C_acc);

// Convert to FP16 only at final store
for output in C_acc:
    C[...] = half(output);
```

### Testing Accumulation Precision

```python
import mlx.core as mx

# Pathological case: all-ones with large K
K = 32768
a = mx.ones((1, K), dtype=mx.float16)
b = mx.ones((K, 1), dtype=mx.float16)

# Should produce K=32768, not overflow/saturate
result = a @ b
assert result[0, 0] == K  # Fails with FP16 accumulation
```

## Scale Storage Precision

Quantization scales determine dequantization accuracy. They can be stored in FP16 (default) or FP32.

### FP16 Scales (Default)

- 2 bytes per scale
- Sufficient precision for per-group scales
- Standard choice for inference

### FP32 Scales

- 4 bytes per scale (2x memory)
- Slightly better dequantization accuracy
- Useful for evaluation and benchmarks

### Memory Impact

Scale overhead is small compared to packed weights:

| Group Size | Weights per Scale | Scale Overhead |
|------------|-------------------|----------------|
| 32 | 32 | 6.25% (FP16), 12.5% (FP32) |
| 64 | 64 | 3.125% (FP16), 6.25% (FP32) |
| 128 | 128 | 1.56% (FP16), 3.125% (FP32) |

For most use cases, FP16 scales are sufficient.

## Global Configuration

Metal Marlin supports a global default configuration:

```python
from metal_marlin.dtypes import (
    get_default_config,
    set_default_config,
    reset_default_config,
    bf16_config,
)

# Get current default (bf16 by default)
config = get_default_config()

# Set a new global default
set_default_config(bf16_config())

# Reset to factory defaults
reset_default_config()
```

Functions that accept `dtype_config` parameters use the global default when `None` is passed:

```python
from metal_marlin.quantize import pack_fp4_weights

# Uses global default config
packed, scales, meta = pack_fp4_weights(weights, group_size=128)

# Or pass explicit config
packed, scales, meta = pack_fp4_weights(
    weights,
    group_size=128,
    dtype_config=high_precision_config()
)
```

## Migration Guide

### From Hardcoded FP16

**Before:**
```python
import mlx.core as mx

scales = mx.ones((num_groups, N), dtype=mx.float16)
output = result.astype(mx.float16)
```

**After:**
```python
from metal_marlin.dtypes import get_default_config

config = get_default_config()
scales = mx.ones((num_groups, N), dtype=config.mlx_scales)
output = result.astype(config.mlx_activations)
```

### Adding Config to Existing Functions

**Before:**
```python
def quantize_weights(weights: mx.array) -> tuple[mx.array, mx.array]:
    scales = compute_scales(weights).astype(mx.float16)
    packed = pack_weights(weights)
    return packed, scales
```

**After:**
```python
from metal_marlin.dtypes import DTypeConfig, get_default_config

def quantize_weights(
    weights: mx.array,
    dtype_config: DTypeConfig | None = None,
) -> tuple[mx.array, mx.array]:
    config = dtype_config or get_default_config()
    scales = compute_scales(weights).astype(config.mlx_scales)
    packed = pack_weights(weights)
    return packed, scales
```

### Numpy Compatibility

When packing/unpacking weights via numpy:

```python
import numpy as np
from metal_marlin.dtypes import get_default_config

config = get_default_config()

# numpy doesn't have native bf16, uses fp16 as storage
scales_np = scales.astype(config.numpy_scales)  # np.float16

# MLX handles the conversion correctly
scales_mx = mx.array(scales_np).astype(config.mlx_scales)  # mx.bfloat16
```

## Common Patterns

### Layer Initialization

```python
from metal_marlin.dtypes import DTypeConfig, get_default_config

class QuantizedLinear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype_config: DTypeConfig | None = None,
    ):
        self.config = dtype_config or get_default_config()

        self.scales = mx.ones(
            (in_features // self.group_size, out_features),
            dtype=self.config.mlx_scales,
        )

        if self.bias:
            self.bias = mx.zeros(
                (out_features,),
                dtype=self.config.mlx_activations,
            )
```

### Forward Pass

```python
def forward(self, x: mx.array) -> mx.array:
    # Ensure input dtype matches config
    x = x.astype(self.config.mlx_activations)

    # Dequantize and compute
    output = self.quantized_matmul(x)

    # Output in configured dtype
    return output.astype(self.config.mlx_activations)
```

### KV Cache Allocation

```python
def create_kv_cache(
    batch_size: int,
    num_heads: int,
    max_seq_len: int,
    head_dim: int,
    dtype_config: DTypeConfig | None = None,
) -> tuple[mx.array, mx.array]:
    config = dtype_config or get_default_config()

    cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
    k_cache = mx.zeros(cache_shape, dtype=config.mlx_kv_cache)
    v_cache = mx.zeros(cache_shape, dtype=config.mlx_kv_cache)

    return k_cache, v_cache
```

## See Also

- `docs/mixed_precision.md` - Detailed discussion of accumulation precision
- `docs/kv_cache.md` - KV cache design and FP4/FP8 quantization
- `docs/metal_half_precision_bug.md` - Metal compiler bug affecting FP16
- `metal_marlin/dtypes.py` - Implementation source
