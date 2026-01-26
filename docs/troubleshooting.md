# Troubleshooting Guide

## Common Issues

### "Metal kernel compilation failed"

**Cause**: Metal shader syntax error or incompatible feature.

**Solution**:
1. Check macOS version (requires 14.0+)
2. Verify Metal device supports required features:
   ```python
   from metal_marlin import get_device_info
   print(get_device_info())
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
       device.commit()  # Remove this from inner loop!

   # Good: sync only at end
   output = model(input)
   device.commit()
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

## PyObjC Metal Issues

### "No module named 'Metal'" or "No module named 'MetalPerformanceShaders'"

**Cause**: PyObjC Metal bindings not installed.

**Solution**:
```bash
uv pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
```

### "MTLDevice not available" or device returns None

**Cause**: PyObjC cannot find the Metal device.

**Solution**:
```python
import Metal

# Get the default device
device = Metal.MTLCreateSystemDefaultDevice()
if device is None:
    raise RuntimeError("No Metal device available")

print(f"Using device: {device.name()}")
```

### "Argument type mismatch" or "Invalid buffer binding"

**Cause**: PyObjC bridge converts Python types incorrectly for Metal buffers.

**Solution**:
1. Ensure numpy arrays are contiguous:
   ```python
   import numpy as np
   arr = np.ascontiguousarray(arr, dtype=np.float32)
   ```

2. Use explicit buffer creation:
   ```python
   buffer = device.newBufferWithBytes_length_options_(
       arr.tobytes(),
       arr.nbytes,
       Metal.MTLResourceStorageModeShared
   )
   ```

### "Command buffer execution failed" or GPU timeout

**Cause**: Kernel took too long or hit an error during execution.

**Solution**:
1. Check for infinite loops in shader code
2. Reduce workload size to isolate the issue
3. Enable GPU error logging:
   ```python
   import os
   os.environ["MTL_DEBUG_LAYER"] = "1"
   os.environ["MTL_SHADER_VALIDATION"] = "1"
   ```

### Memory not released after computation

**Cause**: PyObjC autorelease pool not draining properly.

**Solution**:
```python
from Foundation import NSAutoreleasePool

pool = NSAutoreleasePool.alloc().init()
try:
    # Your Metal computations here
    result = run_metal_kernel(...)
finally:
    del pool  # Explicitly drain the pool
```

### "Selector not found" or attribute errors on Metal objects

**Cause**: PyObjC version mismatch or incomplete bindings.

**Solution**:
1. Update PyObjC to latest version:
   ```bash
   uv pip install --upgrade pyobjc-framework-Metal
   ```

2. Verify macOS SDK compatibility:
   ```python
   import Metal
   print(Metal.__file__)  # Check binding location
   ```

## Debugging Tools

### Enable kernel profiling

```python
import os
os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"

# Run your code, then check:
# /tmp/metal_shader_compilation_log.txt
```

### Enable Metal debug layer

```python
import os
os.environ["MTL_DEBUG_LAYER"] = "1"
os.environ["MTL_SHADER_VALIDATION"] = "1"
os.environ["MTL_DEBUG_LAYER_WARNING_MODE"] = "nslog"
```

### Validate quantization

```python
from metal_marlin.debug import validate_quantization
import torch

original = torch.randn(4096, 4096)
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

### Check Metal buffer contents

```python
import numpy as np

def read_metal_buffer(buffer, dtype=np.float32):
    """Read contents of a Metal buffer back to numpy."""
    ptr = buffer.contents()
    length = buffer.length()
    return np.frombuffer(
        (ctypes.c_char * length).from_address(ptr),
        dtype=dtype
    ).copy()
```
