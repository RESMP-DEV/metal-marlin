# FLOPs Profiling with profile_ops

The `metal_marlin.utils.profile_ops` module provides tools for calculating theoretical FLOPs (floating-point operations) for neural network layers. This is useful for:

- **Roofline analysis**: Compare actual performance against theoretical peak
- **Bottleneck identification**: Find compute-heavy layers
- **Quantization impact**: Measure overhead from dequantization
- **Model profiling**: Estimate total FLOPs for inference

## Quick Start

```python
from metal_marlin.utils import (
    calculate_matmul_flops,
    calculate_attention_flops,
    LayerFLOPsCounter,
    profile_model_flops
)

# Calculate FLOPs for a GEMM
flops = calculate_matmul_flops(M=4096, N=4096, K=4096)
print(f"GEMM FLOPs: {flops / 1e12:.2f} TFLOPs")

# Profile attention
attn_flops = calculate_attention_flops(
    batch=8, seq_len=2048, num_heads=32, head_dim=128, causal=True
)
print(f"Attention FLOPs: {attn_flops / 1e12:.2f} TFLOPs")
```

## Core Functions

### Matrix Multiplication

```python
calculate_matmul_flops(M, N, K, *, quantized=False) -> int
```

Calculate FLOPs for `C = A @ B` where:
- `A` is `M x K`
- `B` is `K x N`
- `C` is `M x N`

Standard GEMM: `2 * M * N * K` FLOPs (1 multiply + 1 add per element)

Quantized GEMM adds dequantization overhead (approximately 2x more ops).

### Attention

```python
calculate_attention_flops(batch, seq_len, num_heads, head_dim, *, causal=False) -> int
```

Computes FLOPs for scaled dot-product attention:
```
scores = Q @ K^T / sqrt(d)
output = softmax(scores) @ V
```

Causal masking reduces FLOPs by ~50% (triangular attention matrix).

### Feedforward Network

```python
calculate_ffn_flops(batch, seq_len, hidden_dim, ffn_dim, *, gated=False, quantized=False) -> int
```

Standard FFN: `x -> Linear -> Activation -> Linear -> x`

Gated FFN (SwiGLU): Uses two up-projections and element-wise gating.

### Other Utilities

- `calculate_layernorm_flops(batch, seq_len, hidden_dim)`: LayerNorm/RMSNorm
- `calculate_embedding_flops(batch, seq_len, vocab_size, hidden_dim)`: Embedding lookup

## LayerFLOPsCounter

Accumulate FLOPs across multiple layers:

```python
counter = LayerFLOPsCounter()

# Add individual layers
counter.add_matmul("layer.0.qkv", M=2048, N=12288, K=4096, quantized=True)
counter.add_attention("layer.0.attn", batch=1, seq_len=2048, num_heads=32, head_dim=128)
counter.add_ffn("layer.0.ffn", batch=1, seq_len=2048, hidden_dim=4096, ffn_dim=11008, gated=True)

# Print summary
counter.print_summary()
print(f"Total: {counter.total_tflops:.2f} TFLOPs")
```

Output:
```
Layer                                                  TFLOPs  % Total
----------------------------------------------------------------------
layer.0.ffn                                             1.108    65.5%
layer.0.qkv                                             0.412    24.4%
layer.0.attn                                            0.035     2.0%
----------------------------------------------------------------------
TOTAL                                                   1.693   100.0%
```

## Model Profiling

Profile an entire Transformer model:

```python
counter = profile_model_flops(
    batch=1,
    seq_len=2048,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    ffn_dim=11008,
    vocab_size=32000,
    causal=True,
    gated_ffn=True,
    quantized=True
)

counter.print_summary(top_n=5)
```

## Use Cases

### 1. Roofline Analysis

Compare measured throughput against theoretical peak:

```python
from metal_marlin.utils import calculate_matmul_flops
import time

# Theoretical FLOPs
M, N, K = 4096, 4096, 4096
theoretical_flops = calculate_matmul_flops(M, N, K)

# Measure actual time
start = time.perf_counter()
result = metal_marlin_gemm(A, B)  # Your kernel
elapsed = time.perf_counter() - start

# Compute achieved TFLOPs/s
achieved_tflops = (theoretical_flops / elapsed) / 1e12
print(f"Achieved: {achieved_tflops:.2f} TFLOPs/s")

# Compare to peak (e.g., M4 Max GPU: ~14 TFLOPs/s FP16)
peak_tflops = 14.0
efficiency = (achieved_tflops / peak_tflops) * 100
print(f"Efficiency: {efficiency:.1f}%")
```

### 2. Quantization Overhead

```python
fp16_flops = calculate_matmul_flops(M, N, K, quantized=False)
fp4_flops = calculate_matmul_flops(M, N, K, quantized=True)

overhead = (fp4_flops - fp16_flops) / fp16_flops * 100
print(f"Quantization adds {overhead:.1f}% FLOPs overhead")
```

### 3. Identify Bottlenecks

```python
counter = LayerFLOPsCounter()

# Profile all layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        counter.add_matmul(name, M=tokens, N=out_dim, K=in_dim, quantized=True)

# Show top compute consumers
counter.print_summary(top_n=10)
```

## API Reference

### LayerFLOPs Dataclass

```python
@dataclass
class LayerFLOPs:
    name: str
    total_flops: int
    matmul_flops: int
    activation_flops: int
    attention_flops: int
    other_flops: int
    metadata: dict[str, Any]
    
    @property
    def tflops(self) -> float:
        """Total FLOPs in trillions."""
    
    @property
    def gflops(self) -> float:
        """Total FLOPs in billions."""
```

### TransformerLayerFLOPs Dataclass

```python
@dataclass
class TransformerLayerFLOPs:
    attention: int
    ffn: int
    layernorm: int
    total: int
    
    @classmethod
    def from_config(cls, batch, seq_len, hidden_dim, num_heads, ffn_dim, ...) -> TransformerLayerFLOPs:
        """Calculate FLOPs for a Transformer layer."""
```

### LayerFLOPsCounter Methods

- `add_matmul(name, M, N, K, *, quantized=False, metadata=None)`
- `add_attention(name, batch, seq_len, num_heads, head_dim, *, causal=False, metadata=None)`
- `add_ffn(name, batch, seq_len, hidden_dim, ffn_dim, *, gated=False, quantized=False, metadata=None)`
- `add_transformer_layer(name, batch, seq_len, hidden_dim, num_heads, ffn_dim, *, ...)`
- `get_layer(name) -> LayerFLOPs | None`
- `get_layers() -> list[LayerFLOPs]`
- `print_summary(*, top_n=20)`
- `clear()`

## Notes

- **FLOPs vs FLOPs/s**: This module calculates theoretical FLOPs (operation count). To get throughput (FLOPs/s), divide by measured execution time.
- **Memory-bound vs compute-bound**: FLOPs analysis assumes compute-bound operations. Small matrices may be memory-bound.
- **Quantization modeling**: Dequantization overhead is approximate. Actual overhead depends on group size and kernel implementation.
- **Activation functions**: GELU/SiLU FLOPs are approximate (polynomial approximations vary).

## See Also

- `metal_marlin.utils.profiling`: Kernel timing and statistics
- `metal_marlin.profiling.roofline`: Roofline model analysis
- `metal_marlin.profiling.occupancy`: GPU occupancy metrics
