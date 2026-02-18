# MMFP4 Inference Guide

## Overview

MMFP4 (Mixed-Matrix Floating Point 4-bit) is a quantization format designed for efficient LLM inference on Apple Silicon GPUs. The format combines group-wise scaling factors with 4-bit weight matrices, enabling significant memory reduction while maintaining model quality.

**GLM-4.7-Flash** is the first model officially supported with MMFP4 quantization via Metal Marlin. This model leverages the Marlin kernel optimized for Apple Silicon's Unified Memory architecture.

## Quick Start

### Generation Example

```bash
metal-marlin mmfp4 generate -m models/GLM-4.7-Flash-Marlin-MMFP4 -p "Hello"
```

### Server Mode

```bash
metal-marlin mmfp4 serve -m models/GLM-4.7-Flash-Marlin-MMFP4 --port 8000
```

### Benchmarking

```bash
metal-marlin mmfp4 bench -m models/GLM-4.7-Flash-Marlin-MMFP4
```

**Important:** Only run benchmarks manually outside of automated tasks. Benchmarks load multi-GB models and can become orphaned on crash.

## Performance Expectations

| Metric | Expected Value |
|--------|-----------------|
| Load Time | ~12 seconds |
| Throughput | 5-10 tokens/second (with fused MoE) |
| Memory Usage | ~17 GB |

Performance varies based on:
- Model size and sequence length
- Whether fused MoE kernels are enabled (`use_fused_moe=True`)
- Available Unified Memory bandwidth

## Troubleshooting

### NaN Outputs

If you're seeing NaN (Not a Number) values in model outputs:
- **Ensure you're running the latest Metal Marlin code**
- Verify the `wait=True` fix is applied in the attention kernel
- Check that model weights are correctly quantized

### Slow Performance

To improve inference speed:
- Enable fused MoE: Set `use_fused_moe=True` in your config
- Use the latest Metal Marlin kernel optimizations
- Ensure Metal Pro/Max GPU is being utilized

### Memory Issues

- MMFP4 requires ~17GB available RAM
- Close other applications to free memory
- Check swap usage: high swap can indicate memory pressure

## Model Format

The MMFP4 format stores:
- 4-bit quantized weights in group sizes
- Per-group scaling factors (FP16)
- Optional bias terms

Models are converted using the Metal Marlin quantization tools and should be placed in the `models/` directory.

## Additional Resources

- See `contrib/metal_marlin/README.md` for full installation instructions
- Check `contrib/metal_marlin/metal_marlin/shaders/` for kernel source
