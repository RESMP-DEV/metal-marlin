# AWQ Implementation Summary

## Overview

This implementation adds full AWQ (Activation-aware Weight Quantization) support to Metal Marlin. AWQ provides better accuracy than GPTQ for many LLMs while maintaining the same fast inference speed.

## What Was Added

### Core Implementation

1. **`metal_marlin/awq.py`** - Complete AWQ quantization module
   - `compute_activation_stats()`: Compute channel-wise activation statistics
   - `find_salient_weights()`: Identify salient weights using activation stats
   - `compute_salient_scaling()`: Compute scaling factors for salient weights
   - `awq_quantize()`: Main AWQ quantization function
   - `awq_dequantize()`: Dequantize AWQ weights
   - `awq_quantize_model()`: Quantize entire model
   - `AWQResult`: Dataclass for quantization results

2. **`metal_marlin/__init__.py`** - Updated exports
   - Added AWQ functions to public API
   - Includes all AWQ exports for easy importing

### Testing

3. **`tests/test_awq.py`** - Comprehensive test suite
   - Tests for all core AWQ functions
   - Reconstruction quality validation
   - Edge case handling
   - All tests passing ✓

4. **`examples/example_awq.py`** - Usage examples
   - Basic quantization/dequantization
   - Comparing different salient ratios
   - Comparing activation methods
   - Works with synthetic data

### Tools & Documentation

5. **`scripts/quantize_awq.py`** - Command-line quantization tool
   - Full model quantization pipeline
   - Activation statistics generation
   - Configuration options for salient ratio, group size, etc.
   - Dry-run mode for calibration only

6. **`docs/awq.md`** - Complete documentation
   - Overview and comparison with GPTQ
   - Usage examples (Python and CLI)
   - API reference
   - Best practices and troubleshooting

## Key Features

### AWQ Algorithm

1. **Activation Statistics Collection**
   - First-order activation statistics from calibration data
   - Support for mean, max, and RMS methods
   - Per-channel importance computation

2. **Salient Weight Identification**
   - Identifies top ~1% of weights connected to high-magnitude activations
   - Uses combined metric: activation stats × weight magnitude
   - Configurable salient ratio (default: 0.01)

3. **Salient Weight Scaling**
   - Scales salient weights to ensure accurate representation
   - Per-channel scaling factors (q_scale)
   - Compensates by inverse scaling during inference

4. **INT4 Quantization**
   - Symmetric INT4 quantization (range: -8 to 7)
   - Per-group scales and zero-points
   - Marlin-compatible packing for fast inference

### Advantages over GPTQ

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| Information Used | First-order (activation stats) | Second-order (Hessian) |
| Quantization Speed | Faster (no Cholesky) | Slower |
| Accuracy | Better on perplexity/benchmarks | Slightly lower |
| Implementation Complexity | Simpler | More complex |

## Usage

### Quick Start

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

# Dequantize
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

### Command-Line

```bash
# Generate activation statistics
uv run python scripts/quantize_awq.py \
    --model-path /path/to/model \
    --output-path /path/to/output \
    --activations-path /path/to/activations.npz \
    --group-size 128 \
    --salient-ratio 0.01
```

## Testing

All tests pass:

```bash
cd contrib/metal_marlin
uv run python tests/test_awq.py
```

Output:
```
======================================================================
Running AWQ Tests
======================================================================

  ✓ compute_activation_stats tests passed
  ✓ find_salient_weights tests passed
  ✓ compute_salient_scaling tests passed
  ✓ pack_awq_weights tests passed
  ✓ awq_quantize/dequantize tests passed
  ✓ AWQ reconstruction quality: 0.1170 relative error
  ✓ AWQ edge cases tests passed

======================================================================
All AWQ tests passed! ✓
======================================================================
```

## Examples

Run examples:

```bash
cd contrib/metal_marlin
uv run python examples/example_awq.py
```

Output includes:
- Basic quantization/dequantization
- Compression ratio (~7.5x)
- Comparison of different salient ratios
- Comparison of activation methods

## Configuration

### Recommended Settings

- **Salient ratio**: 0.01 (1%) - Default, good balance
- **Activation method**: "rms" - Recommended, balances speed/accuracy
- **Group size**: 128 - Default, works for most models

### Accuracy vs Size Tradeoff

| Salient Ratio | Accuracy | Model Size |
|--------------|----------|-----------|
| 0.005 (0.5%) | Lower | Smaller |
| 0.01 (1%) | Good | Balanced |
| 0.02 (2%) | Higher | Slightly larger |

## Integration with Metal Marlin

AWQ is now fully integrated:

1. **Import from metal_marlin**
   ```python
   from metal_marlin import (
       awq_quantize,
       awq_dequantize,
       awq_quantize_model,
   )
   ```

2. **Marlin-compatible format**
   - Same packing as INT4 for hardware compatibility
   - Works with existing Marlin kernels
   - Fast inference

3. **Consistent API**
   - Follows Metal Marlin conventions
   - Similar to GPTQ API
   - Easy to adopt

## Future Work

Potential enhancements:

1. **Metal GPU acceleration**
   - Add Metal kernels for faster quantization
   - Follow pattern of gptq_metal.py

2. **Improved calibration**
   - Better activation statistics collection
   - Automatic sample selection

3. **Mixed precision**
   - Use AWQ for some layers, FP8/FP16 for others
   - Layer-wise sensitivity analysis

4. **Format compatibility**
   - Convert AWQ ↔ GPTQ formats
   - Support existing AWQ model files

## References

- AWQ Paper: https://arxiv.org/abs/2306.00978
- AWQ GitHub: https://github.com/mit-han-lab/llm-awq
- Metal Marlin: https://github.com/YourOrg/metal_marlin

## Files Modified/Created

### Created
- `metal_marlin/awq.py` - Core AWQ implementation
- `tests/test_awq.py` - Test suite
- `examples/example_awq.py` - Usage examples
- `scripts/quantize_awq.py` - CLI quantization tool
- `docs/awq.md` - Documentation

### Modified
- `metal_marlin/__init__.py` - Added AWQ exports

## License

Same as Metal Marlin project.
