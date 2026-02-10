# MMFP4 Troubleshooting Guide

This guide documents known issues, workarounds, and debugging techniques for MMFP4 (Metal Marlin FP4) inference on Apple Silicon.

---

## Benchmark Results (Feb 2026)

> **Important:** These are actual measured results, not theoretical expectations.

| Test | Result | Expected | Status |
|------|--------|----------|--------|
| Perplexity | 17.15 | <100 | ✅ PASS |
| Prefill avg | 24.6 tok/s | 500-2000 tok/s | ❌ 20-80× slower |
| Decode (KV cache) | 0.27 tok/s | 5-20 tok/s | ❌ 20-80× slower |
| Decode (no cache) | 0.07 tok/s | N/A | Use cache! |

### Prefill by Context Length

| Context | Throughput | Notes |
|---------|------------|-------|
| 128 | 9.0 tok/s | Dispatch overhead dominates |
| 512 | 25.5 tok/s | Better amortization |
| 1024 | 39.3 tok/s | Best prefill efficiency |

### Critical: KV Cache Required

**Without KV cache:** 0.07 tok/s (context recomputed every token)
**With KV cache:** 0.27 tok/s (4× improvement)

Always use `model.generate(use_cache=True)` for autoregressive decoding.

---

## Table of Contents

- [Known Issues](#known-issues)
  - [NaN in Multi-Token Forward Pass](#nan-in-multi-token-forward-pass)
  - [KV Cache Generation Fails](#kv-cache-generation-fails)
  - [First-Call Latency Spikes](#first-call-latency-spikes)
  - [Memory Growth During Long Sequences](#memory-growth-during-long-sequences)
- [Performance Tips](#performance-tips)
- [Debugging Techniques](#debugging-techniques)
- [Quick Reference](#quick-reference)

---

## Known Issues

### NaN in Multi-Token Forward Pass

**Symptom:** Forward pass with `seq_len >= 4` produces NaN logits

**Symptoms Detail:**
- Model outputs NaN values when processing sequences of 4 or more tokens
- Single-token inference (`seq_len = 1`) works correctly
- Issue manifests in both training and inference modes
- Affects all MMFP4 quantized layers

**Root Cause:** 
Metal kernel tile boundary handling in the MMFP4 GEMM shader. The shader uses 8x8 tile dimensions, and when sequence length exceeds the tile boundary (>= 4 tokens due to 2:4 sparsity pattern), the accumulation register is not properly reset between tiles, causing numerical overflow that propagates as NaN.

**Workarounds:**

1. **Use warmup call (Recommended)**
   ```python
   # Perform a dummy forward pass with seq_len=1 before real inference
   dummy_input = torch.zeros(1, 1, dtype=torch.long, device="mps")
   with torch.no_grad():
       _ = model(dummy_input)  # Warmup
   ```

2. **Use single-token decode**
   ```python
   # Process tokens one at a time
   for token_id in token_ids:
       logits = model(token_id.unsqueeze(0))
   ```

3. **Disable MMFP4 for problematic layers**
   ```python
   # Skip quantization on first/last layers
   config = MMFP4Config(skip_layers=[0, -1])
   ```

**Fix Status:** 🟡 In Progress - Kernel fix scheduled for v0.3.0

---

### KV Cache Generation Fails

**Symptom:** `model.generate(use_cache=True)` produces NaN or incorrect outputs

**Symptoms Detail:**
- Text generation produces gibberish or NaN tokens
- Issue occurs during the prefill phase for multi-token sequences
- Cache appears to be populated but values are corrupted
- Problem does not occur with `use_cache=False`

**Root Cause:** 
MLAKVCache prefill kernel has incorrect stride calculations for multi-token sequences. The key/value tensors are written with incorrect memory layout when `seq_len > 1`, causing misaligned reads during subsequent decode steps.

**Workarounds:**

1. **Use `use_cache=False` (Slower but reliable)**
   ```python
   # Disable KV cache - works but ~2-3x slower for long sequences
   output = model.generate(
       input_ids,
       use_cache=False,
       max_new_tokens=100
   )
   ```

2. **Manual cache management**
   ```python
   # Implement custom cache with correct strides
   from metal_marlin.cache import FixedMLAKVCache
   
   cache = FixedMLAKVCache(max_batch_size=1, max_seq_len=2048)
   output = model.generate(input_ids, past_key_values=cache)
   ```

3. **Hybrid approach**
   ```python
   # Prefill without cache, then enable for decode
   # (requires custom generation loop)
   ```

**Fix Status:** 🔴 Not Started - Targeting v0.3.1

---

### First-Call Latency Spikes

**Symptom:** First inference call takes 5-10x longer than subsequent calls

**Symptoms Detail:**
- Initial forward pass: 500-2000ms
- Subsequent calls: 50-100ms
- Spike occurs even for small models (< 1B params)
- Issue is more pronounced on macOS Sonoma+

**Root Cause:** 
Metal Performance Shaders (MPS) lazy initialization and shader compilation. The Metal runtime compiles shaders on first use, causing significant overhead. Additionally, memory allocation for the unified memory architecture requires initial page mapping.

**Workarounds:**

1. **Explicit warmup (Recommended)**
   ```python
   def warmup_model(model, device="mps", num_warmup=3):
       """Warmup all model components."""
       dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
       with torch.no_grad():
           for _ in range(num_warmup):
               _ = model(dummy)
       torch.mps.synchronize()
   ```

2. **Pre-compile shaders**
   ```bash
   # Set environment variable to enable shader caching
   export MTL_SHADER_CACHE_SIZE=256MB
   ```

3. **Load model ahead of time**
   ```python
   # Load during app startup, not on first request
   model = load_model()  # Do this early
   warmup_model(model)   # And warmup immediately
   ```

**Fix Status:** 🟢 Mitigated - Warmup utilities added in v0.2.5

---

### Memory Growth During Long Sequences

**Symptom:** Memory usage grows linearly with sequence length, even with KV cache

**Symptoms Detail:**
- Memory does not plateau as expected
- Growth rate suggests temporary tensors not being freed
- Issue is more severe with larger batch sizes
- Can lead to system memory pressure warnings

**Root Cause:** 
Intermediate activation tensors in the Metal command buffer are not being released promptly. The Python garbage collector holds references longer than necessary due to circular references in the autograd graph.

**Critical Discovery (Feb 2026):** Using underscore `_` as a variable name holds references!

```python
# LEAKS MEMORY - _ holds tensor until overwritten
_ = model(input_ids)

# CORRECT - explicit variable and delete
outputs = model(input_ids)
del outputs
gc.collect()
torch.mps.empty_cache()
```

**Workarounds:**

1. **Force garbage collection (REQUIRED for benchmarks)**
   ```python
   import gc
   
   def cleanup():
       """Full MPS memory cleanup."""
       gc.collect()
       if hasattr(torch, 'mps') and torch.backends.mps.is_available():
           torch.mps.empty_cache()
   
   # After every forward pass in benchmarks
   cleanup()
   ```

2. **Subprocess isolation (for benchmarks)**
   ```python
   # MPS doesn't fully release memory on del
   # Use separate processes for true isolation
   for config in configs:
       subprocess.run(['python', 'bench_single.py', config])
       # Child process terminates → memory fully released
   ```

3. **Use torch.no_grad() for inference**
   ```python
   # Always disable gradients during generation
   with torch.no_grad():
       output = model.generate(input_ids)
   ```

4. **Reduce batch size**
   ```python
   # Process large batches in smaller chunks
   for mini_batch in torch.split(batch, 4):
       output = model(mini_batch)
   ```

**Fix Status:** 🟡 Mitigated - Subprocess isolation pattern documented, cleanup utilities enhanced in v0.2.6

---

## Performance Tips

> **Current Reality (Feb 2026):** MMFP4 achieves 0.27 tok/s decode, 20-80× slower than expected.
> These tips help, but fundamental performance work is needed.

### General Recommendations

| Tip | Impact | Implementation |
|-----|--------|----------------|
| **Use KV cache** | **+300% decode** | `model.generate(use_cache=True)` - CRITICAL |
| Enable warmup | -80% first-call latency | Call `warmup_model()` before serving |
| Subprocess isolation | Prevents memory leaks | Run benchmarks in separate processes |
| Use explicit del | Prevents reference leaks | `outputs = model(...); del outputs` |
| Enable shader cache | -50% startup time | Set `MTL_SHADER_CACHE_SIZE` |

### KV Cache: The Most Important Optimization

**Without cache:** 0.07 tok/s (full context recomputed every token)
**With cache:** 0.27 tok/s (4× improvement)

```python
# ALWAYS use model.generate() with use_cache=True
output = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True  # Creates MLAKVCache automatically
)
```

### Optimal Settings by Use Case

**Real-time Chat (low latency priority):**
```python
config = {
    "use_cache": True,
    "batch_size": 1,
    "warmup": True,
    "compile": False,  # Avoid compilation overhead
}
```

**Batch Processing (throughput priority):**
```python
config = {
    "use_cache": False,  # More memory for batching
    "batch_size": 8,
    "warmup": True,
    "compile": True,     # Worth it for large batches
}
```

---

## Debugging Techniques

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Metal-specific debugging
import os
os.environ["METAL_DEVICE_DEBUG"] = "1"
os.environ["METAL_CAPTURE_ENABLED"] = "1"
```

### Check for NaN Propagation

```python
def check_nan(module, input, output):
    """Hook to detect NaN in forward pass."""
    if torch.isnan(output).any():
        print(f"NaN detected in {module.__class__.__name__}")
        # Capture stack trace
        import traceback
        traceback.print_stack()

# Register on all modules
for name, module in model.named_modules():
    module.register_forward_hook(check_nan)
```

### Memory Profiling

```python
def log_memory_usage(tag=""):
    """Log current MPS memory stats."""
    allocated = torch.mps.current_allocated_memory() / 1e9
    reserved = torch.mps.driver_allocated_memory() / 1e9
    print(f"[{tag}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Use throughout your code
log_memory_usage("before_forward")
output = model(input_ids)
log_memory_usage("after_forward")
```

### Validate Kernel Output

```python
from metal_marlin.utils import validate_kernel_output

# Compare MMFP4 against reference FP16 implementation
validate_kernel_output(
    kernel_fn=mmfp4_gemm,
    reference_fn=torch.matmul,
    input_shape=(1024, 4096),
    tolerance=0.01  # 1% tolerance for FP4
)
```

---

## Quick Reference

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `NaN in output` | Tile boundary bug | Use warmup, single-token decode |
| `Invalid cache state` | KV cache stride issue | Disable cache or use FixedMLAKVCache |
| `MPS out of memory` | Memory leak / large batch | Reduce batch, call `empty_cache()` |
| `Shader compilation failed` | Metal version mismatch | Update macOS, check GPU support |
| `First call too slow` | Lazy initialization | Implement warmup routine |

### Environment Variables

```bash
# Shader caching
export MTL_SHADER_CACHE_SIZE=256MB

# Debug mode (verbose)
export METAL_DEVICE_DEBUG=1

# Disable shader cache for testing
export MTL_SHADER_CACHE_SIZE=0

# Force GPU preference
export MTL_PREFER_DISCRETE_GPU=1
```

### Version Compatibility

| metal-marlin | macOS | PyTorch | Status |
|--------------|-------|---------|--------|
| 0.2.x | 14.0+ | 2.1+ | ✅ Supported |
| 0.2.x | 13.x | 2.0+ | ⚠️ Partial (no KV cache) |
| 0.1.x | 14.0+ | 2.1+ | ❌ Deprecated |

---

## Reporting Issues

When reporting new issues, please include:

1. **System info:** macOS version, Mac model, RAM
2. **Package versions:** `pip show metal-marlin torch`
3. **Minimal reproduction:** Code snippet that triggers the issue
4. **Error logs:** Full traceback and any Metal debug output
5. **Workarounds tried:** Which solutions from this guide were attempted

Open issues at: https://github.com/alphaheng/metal-marlin/issues

---

*Last updated: 2026-02-13*
