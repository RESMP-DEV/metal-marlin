# AWQ (Activation-aware Weight Quantization)

## Overview

AWQ (Activation-aware Weight Quantization) is a post-training quantization method for LLMs that provides better accuracy than GPTQ while maintaining fast inference speed.

### Key Features

- **Better Accuracy**: Outperforms GPTQ on perplexity and downstream benchmarks
- **Fast Inference**: Same inference speed as standard 4-bit quantization
- **Activation-aware**: Uses first-order activation statistics instead of second-order Hessian
- **Salient Weight Protection**: Protects only ~1% of most important weights via scaling
- **Simple Implementation**: Easier to implement and tune than GPTQ

### How AWQ Works

1. **Collect Activation Statistics**: Run model on calibration data to collect activation statistics
2. **Identify Salient Weights**: Find weights connected to high-magnitude activations
3. **Apply Salient Scaling**: Scale salient weights to ensure accurate representation after quantization
4. **Quantize Remaining Weights**: Apply standard 4-bit symmetric quantization
5. **Store Metadata**: Save quantized weights, scales, zero-points, and salient scaling factors

### Comparison with GPTQ

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| Information Used | First-order (activation stats) | Second-order (Hessian) |
| Error Compensation | Prevents error via scaling | Compensates error via adjustment |
| Quantization Speed | Faster | Slower (requires Cholesky) |
| Inference Speed | Same as standard 4-bit | Same as standard 4-bit |
| Accuracy | Better on perplexity/benchmarks | Slightly lower on some tasks |
| Implementation Complexity | Simpler | More complex |

## Usage

### Basic Usage

```python
from metal_marlin import awq_quantize, awq_dequantize

# Prepare data
weights = ...  # [in_features, out_features]
activations = ...  # [batch_size, seq_len, in_features]

# Quantize
result = awq_quantize(
    weights,
    activations,
    group_size=128,
    salient_ratio=0.01,
    activation_method="rms",
)

# Dequantize (for validation)
meta = {
    "in_features": weights.shape[0],
    "out_features": weights.shape[1],
    "group_size": 128,
    "quant_type": "awq_int4",
}
dequantized = awq_dequantize(
    result.Q,
    result.scales,
    result.zeros,
    result.q_scale,
    meta,
)
```

### Command-line Quantization

```bash
# Generate activation statistics (one-time)
uv run python scripts/quantize_awq.py \
    --model-path /path/to/model \
    --output-path /path/to/output \
    --activations-path /path/to/activations.npz \
    --group-size 128 \
    --salient-ratio 0.01 \
    --activation-method rms
```

### With Calibration Data

For best results, use real calibration data:

```python
# Collect activations from representative data
activations_dict = {}
for batch in calibration_data:
    # Forward pass to collect activations
    activations_dict[layer_name].append(layer_activations)

# Compute activation statistics per layer
for layer_name, activations in activations_dict.items():
    activations_dict[layer_name] = np.concatenate(activations, axis=0)

# Save activations for quantization
np.savez_compressed("activations.npz", **activations_dict)

# Quantize model
stats = awq_quantize_model(
    model_path="/path/to/model",
    output_path="/path/to/output",
    activations_path="activations.npz",
    group_size=128,
    salient_ratio=0.01,
)
```

## API Reference

### `awq_quantize`

Quantize a weight matrix using AWQ algorithm.

```python
awq_quantize(
    weights: np.ndarray,
    activations: np.ndarray,
    group_size: int = 128,
    salient_ratio: float = 0.01,
    activation_method: Literal["mean", "max", "rms"] = "rms",
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> AWQResult
```

**Parameters:**
- `weights`: Weight matrix [in_features, out_features]
- `activations`: Calibration activations [batch_size, seq_len, in_features]
- `group_size`: Quantization group size (default: 128)
- `salient_ratio`: Fraction of salient weights to protect (default: 0.01)
- `activation_method`: Method for computing channel importance
  - `"mean"`: Mean absolute activation
  - `"max"`: Maximum absolute activation
  - `"rms"`: Root mean square activation (default)
- `output_backend`: Backend for output arrays

**Returns:** `AWQResult` with quantized weights and metadata

### `awq_dequantize`

Dequantize AWQ weights back to float.

```python
awq_dequantize(
    packed: Any,
    scales: Any,
    zeros: Any,
    q_scale: Any,
    meta: dict[str, Any],
    weights_dtype: np.dtype | None = None,
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> Any
```

### `awq_quantize_model`

Quantize all linear layers in a model.

```python
awq_quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    activations_path: str | Path,
    group_size: int = 128,
    salient_ratio: float = 0.01,
    activation_method: Literal["mean", "max", "rms"] = "rms",
    verbose: bool = True,
) -> dict[str, Any]
```

### `compute_activation_stats`

Compute channel-wise activation statistics.

```python
compute_activation_stats(
    activations: np.ndarray,
    method: Literal["mean", "max", "rms"] = "rms",
) -> np.ndarray
```

### `find_salient_weights`

Identify salient weights using activation statistics.

```python
find_salient_weights(
    weights: np.ndarray,
    activation_stats: np.ndarray,
    salient_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]
```

## Configuration

### Salient Ratio

Controls fraction of protected salient weights:
- **0.005** (0.5%): Better compression, slightly lower accuracy
- **0.01** (1%): Default, good balance
- **0.02** (2%): Higher accuracy, slightly larger model

### Activation Method

Method for computing channel importance:
- **"mean"**: Fastest, may miss outliers
- **"max"**: Captures outliers, may be noisy
- **"rms"**: Recommended, balances speed and accuracy

### Group Size

Quantization group size (must divide in_features):
- **64**: Better accuracy, larger model
- **128**: Default, good balance
- **256**: Smaller model, slightly lower accuracy

## Best Practices

1. **Use Real Calibration Data**: Collect activations from representative data (100-500 samples)
2. **Tune Salient Ratio**: Start with 0.01, adjust based on accuracy vs size tradeoff
3. **Use RMS Method**: Provides best balance of speed and accuracy
4. **Validate Quality**: Check perplexity/benchmarks before deploying
5. **Profile Inference**: Ensure latency/throughput meet requirements

## Limitations

- Requires calibration data for optimal results
- Not compatible with existing GPTQ models (different format)
- May not improve accuracy for all model architectures
- Salient weight protection adds small storage overhead (~1%)

## References

- AWQ Paper: https://arxiv.org/abs/2306.00978
- AWQ GitHub: https://github.com/mit-han-lab/llm-awq
- Metal Marlin: https://github.com/YourOrg/metal_marlin

## Troubleshooting

### High Quantization Error

- Use more calibration samples (500+)
- Increase `salient_ratio` to 0.02
- Try `"max"` activation method
- Reduce `group_size` to 64

### Model Size Larger Than Expected

- Reduce `salient_ratio` to 0.005
- Increase `group_size` to 256
- Ensure activation statistics are correctly generated

### Poor Perplexity After Quantization

- Verify calibration data is representative
- Check activation statistics are not corrupted
- Try different `activation_method` values
- Compare with GPTQ for baseline

## License

Same as Metal Marlin project.
