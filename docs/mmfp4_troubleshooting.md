# MMFP4 Troubleshooting Guide

This document covers known issues and workarounds for running inference with MMFP4 quantized models using the Metal Marlin backend.

## Known Issues

### NaN in Multi-Token Forward Pass

**Symptom:** Forward pass with `seq_len >= 4` produces NaN logits

**Root Cause:** Metal kernel tile boundary handling. The GEMM kernel splits computation into tiles, and incorrect boundary condition checks can cause out-of-bounds memory access or incorrect accumulation, resulting in NaN values.

**Workaround:**
```python
# Option 1: Use warmup call before multi-token inference
output = model(input_ids[:1], ...)  # Warmup call
output = model(input_ids, ...)       # Actual multi-token call

# Option 2: Single-token decode (slower but stable)
for token in tokens:
    output = model(token.unsqueeze(0), ...)
```

### KV Cache Generation Fails

**Symptom:** `model.generate(use_cache=True)` produces NaN

**Root Cause:** MLAKVCache prefill implementation has issues with multi-token sequences. The cache format transformation and attention computation for multi-token prefill doesn't handle certain edge cases correctly.

**Workaround:**
```python
# Disable KV cache (slower but works correctly)
output = model.generate(
    input_ids,
    use_cache=False,  # Disable KV cache
    max_new_tokens=100
)
```

### First-Token Latency Spike

**Symptom:** First forward pass takes 3-5x longer than subsequent passes

**Root Cause:** Metal shader compilation and buffer allocation happen on first call. Additionally, the MMFP4 dequantization kernels may have lazy initialization.

**Workaround:**
```python
# Warmup before inference
_ = model(input_ids[:, :1], ...)  # Single-token warmup
_ = model(input_ids[:, :1], ...)  # Second warmup for stability
```

### Out-of-Memory on Large Batch Sizes

**Symptom:** `OutOfMemoryError` when running with batch_size > 1

**Root Cause:** MMFP4 buffers are duplicated per batch item, and the Metal memory allocator doesn't efficiently reuse intermediate buffers.

**Workaround:**
```python
# Reduce batch size
for i in range(num_samples):
    output = model(inputs[i:i+1], ...)

# Or enable memory pooling if available
model.config.enable_memory_pooling = True
```

## Performance Tips

### Enable Warmup for Consistent Latency

Always run warmup calls before benchmarking:
```python
for _ in range(3):
    _ = model(warmup_input, ...)
```

### Use KV Cache for Decode-Heavy Workloads

After fixing the KV cache issues, enable it for better performance:
```python
# After KV cache fixes are applied
output = model.generate(
    input_ids,
    use_cache=True,  # Enable KV cache
    max_new_tokens=200
)
```

### Optimal Sequence Lengths

- **Prefill:** Use `seq_len <= 128` for stable prefill
- **Decode:** Single-token decode is most stable
- **Multi-token decode:** Requires warmup; results may vary

### Metal Device Selection

For MacBooks with multiple GPUs, ensure you're using the correct device:
```python
import torch
device = torch.device("mps")
model = model.to(device)
```

## Debugging Commands

### Enable Metal Debug Mode
```python
import os
os.environ["METAL_DEBUG_MODE"] = "1"
```

### Check for NaN/Inf Values
```python
import torch
output = model(input_ids)
print(f"Has NaN: {torch.isnan(output).any()}")
print(f"Has Inf: {torch.isinf(output).any()}")
```

### Verbose Kernel Logging
```python
import metal_marlin as mm
mm.set_verbose(True)
```

## Related Files

- `src/gemm_fp4_optimized.metal` - Main MMFP4 GEMM kernel
- `src/dequant_fp4_fast.metal` - FP4 dequantization
- `metal_marlin/kv_cache.py` - KV cache implementation
- `developer_tests/diagnose_mla_4tok.py` - Multi-token MLA diagnosis
- `developer_tests/debug_kv_decode.py` - KV decode debugging
