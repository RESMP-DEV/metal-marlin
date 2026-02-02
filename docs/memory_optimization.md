# Memory Optimization Guide

This document explains the memory optimization features in metal_marlin
for running large MoE models efficiently on Apple Silicon.

## The Problem: Triple-Copy

Without optimization, loading a Trellis model causes three copies of weights:

1. **Safetensors → CPU**: Weights loaded from disk to CPU memory
2. **CPU → MPS**: `.to("mps")` copies to Metal unified memory
3. **MPS → Metal Buffer**: Zero-copy fails, causing another copy

For a 30B MoE model with 46 layers × 64 experts, this results in:
- Expected: ~6 GB (3-bit quantized)
- Actual: ~23 GB (triple copy)

## The Solution: Direct CPU→Metal

The optimized path:

1. Load weights to CPU only
2. Create Metal buffers directly from CPU numpy arrays
3. Delete CPU tensors after buffer creation

Result: **~6 GB** actual memory usage.

## Usage

### Automatic Optimization (Default)

```python
from metal_marlin.trellis.model import TrellisForCausalLM

# Memory optimization is ON by default
model = TrellisForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-Trellis-3bpw",
    device="mps",
)
```

### Manual Optimization

```python
# Load without automatic optimization
model = TrellisForCausalLM.from_pretrained(
    model_path,
    optimize_memory=False,
)

# ... do other setup ...

# Then optimize manually
stats = model.optimize_memory(verbose=True)
print(f"Freed {stats['freed_gb']:.2f} GB")
```

### Disable Optimization

```python
# For debugging, disable optimization
model = TrellisForCausalLM.from_pretrained(
    model_path,
    optimize_memory=False,
)
```

## Technical Details

### Key Functions

- `cpu_tensor_to_metal_buffer()`: Creates Metal buffer from CPU tensor
- `create_cached_weight_buffers_from_cpu()`: Creates all MoE buffers
- `TrellisMoEMLP._create_buffers_eagerly()`: Initializes buffers during load
- `TrellisForCausalLM.optimize_memory()`: Model-level optimization

### Memory Profiling

```bash
# Profile memory during inference
cd contrib/metal_marlin
uv run python scripts/profile_memory.py --model models/GLM-4.7-Flash-Trellis-3bpw
```

## Caveats

1. **dequantize() unavailable**: After optimization, the PyTorch weight
   tensors are deleted. `TrellisLinear.dequantize()` will not work.

2. **One-time cost**: Buffer creation happens during model load. First
   inference may be slightly slower.

3. **Metal-only**: This optimization only works on MPS (Apple Silicon).
   CUDA would need a different approach.
