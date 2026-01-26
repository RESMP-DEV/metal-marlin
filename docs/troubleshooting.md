# Troubleshooting Guide

## Common Issues

### "Metal kernel compilation failed"

**Cause**: Metal shader syntax error or incompatible feature.

**Solution**:
1. Check macOS version (requires 14.0+)
2. Verify Metal device supports simdgroup_matrix:
   ```python
   import mlx.core as mx
   print(mx.metal.device_info())
   ```
3. Look for shader compilation errors in console

### "Dimension mismatch" error

**Cause**: M, N, or K not aligned to expected tile sizes.

**Solution**:
- Marlin requires K divisible by 8 (packing factor)
- K should be divisible by group_size (default 128)
- For best performance, M and N should be multiples of 64

### Numerical differences from FP16

**Expected**: FP4 quantization loses precision. Typical error:
- Per-element: ±5-10% relative error
- Output distribution: ~1% RMSE

**If larger errors**:
1. Check group_size (smaller = more accurate)
2. Verify scales are FP16, not FP32
3. Ensure weights were quantized correctly

### Slow performance

**Checklist**:
1. Using synchronous vs async execution?
   ```python
   # Bad: sync after every op
   for layer in model.layers:
       x = layer(x)
       mx.synchronize()  # Remove this!

   # Good: sync only at end
   output = model(input)
   mx.synchronize()
   ```

2. Kernel compilation cache cold?
   ```python
   from metal_marlin import warm_cache
   warm_cache()  # Pre-compile kernels
   ```

3. Memory pressure?
   - Check Activity Monitor for memory swap
   - Reduce batch size if needed

### "Metal device not found"

**Cause**: Running on non-Apple hardware or missing Metal support.

**Solution**: Metal Marlin requires Apple Silicon (M1/M2/M3/M4).

## Debugging Tools

### Enable kernel profiling

```python
import os
os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"

# Run your code, then check:
# /tmp/metal_shader_compilation_log.txt
```

### Validate quantization

```python
from metal_marlin.debug import validate_quantization

original = mx.random.normal((4096, 4096))
packed, scales = pack_fp4_weights(original)

report = validate_quantization(original, packed, scales)
print(report)
# Output: RMSE: 0.023, Max Error: 0.15, ...
```

### Compare with reference

```python
from metal_marlin.test import compare_with_reference

result = marlin_gemm_fp4(A, B, scales)
ref = reference_gemm(A, dequant(B, scales))

compare_with_reference(result, ref)
# Output: All close: True, Max diff: 0.002
```
