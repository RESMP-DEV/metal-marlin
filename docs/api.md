# API Reference

## Core Functions

### `marlin_gemm_fp4`

```python
marlin_gemm_fp4(
    A: mx.array,           # Activations [M, K], dtype=float16
    B_packed: mx.array,    # Packed weights [K//8, N], dtype=uint32
    scales: mx.array,      # Scales [num_groups, N], dtype=float16
    group_size: int = 128  # Quantization group size
) -> mx.array              # Output [M, N], dtype=float16
```

Compute quantized matrix multiplication: C = A @ dequant(B_packed, scales)

**Parameters:**
- `A`: Input activations in FP16. Shape must be [M, K] where K matches the
  original (unpacked) dimension of B.
- `B_packed`: FP4-quantized weights packed into uint32. Each uint32 contains
  8 x 4-bit values. Shape is [K//8, N].
- `scales`: Per-group scales for dequantization. Shape is [K//group_size, N].
- `group_size`: Number of weights sharing the same scale. Default 128.

**Returns:**
- Output tensor of shape [M, N] in FP16.

**Example:**
```python
M, N, K = 32, 4096, 4096
A = mx.random.normal((M, K), dtype=mx.float16)
B_packed, scales = pack_fp4_weights(random_weights(K, N), group_size=128)
C = marlin_gemm_fp4(A, B_packed, scales)
```

---

### `marlin_gemm_int4`

```python
marlin_gemm_int4(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    zeros: mx.array,       # Zero points [num_groups, N]
    group_size: int = 128
) -> mx.array
```

INT4 (asymmetric) quantized GEMM with zero-point subtraction.

---

### `pack_fp4_weights`

```python
pack_fp4_weights(
    weights: mx.array,     # FP16 weights [K, N]
    group_size: int = 128
) -> Tuple[mx.array, mx.array]  # (packed, scales)
```

Quantize FP16 weights to FP4 and pack into uint32.

---

## Layer Classes

### `MarlinLinear`

```python
class MarlinLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str = "fp4",  # "fp4", "int4", "int4_sym"
        group_size: int = 128
    )

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> MarlinLinear
```

Drop-in replacement for `nn.Linear` using Marlin kernels.

---

### `quantize_model`

```python
quantize_model(
    model: nn.Module,
    quant_type: str = "fp4",
    group_size: int = 128,
    skip_layers: Set[str] = {"lm_head", "embed_tokens"}
) -> nn.Module
```

Quantize all `nn.Linear` layers in a model to Marlin format.
