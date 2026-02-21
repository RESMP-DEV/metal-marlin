# Decode Optimization Investigation

**Status:** In Progress  
**Target:** 10+ tok/s decode (from ~2 tok/s measured)

## Problem

Single-token decode performance is ~2 tok/s on M4 Max, significantly below
the theoretical target of 50+ tok/s for a 3-bit quantized model.

## Hypotheses

1. **Fast path not activating**: The M=1 fast path may not be triggering
2. **Weight dequant overhead**: Per-token dequantization dominates
3. **GPU sync overhead**: .item()/.cpu() calls force synchronization
4. **Memory bandwidth**: Weight loading saturating memory bus

## Fast Path Design

For M=1 decode, we pre-dequantize FP4→FP16 weights and use native PyTorch:

```python
# Instead of: Metal kernel dispatch per layer
# Use: Pre-dequantized weights + F.linear()

if M <= 16 and self._dequant_weight is not None:
    return F.linear(x_2d, self._dequant_weight, self.bias)
```

This trades memory (storing FP16 weights) for speed (no kernel dispatch).

## Diagnostic Tools

| Script | Purpose |
|--------|---------|
| `benchmarks/profile_decode_path.py` | Profile where time is spent |
| `developer_tests/verify_weight_loading.py` | Check weight correctness |
| `developer_tests/profile_memory.py` | Find memory bloat |
| `developer_tests/find_sync_points.py` | Locate GPU syncs |

## Metrics to Track

- Tokens per second (decode)
- Perplexity (quality indicator)
- Memory usage (should be ~15GB for 3-bit)
- Fast path hit rate (should be 100% for M=1)

## Results

*(To be filled after running diagnostics)*
