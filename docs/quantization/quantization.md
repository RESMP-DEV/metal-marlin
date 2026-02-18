# Quantization Guide

Metal Marlin supports quantization from 2-8 bits with multiple formats optimized for Apple Silicon.

## Quick Start

```bash
# FP4 quantization
metal-marlin quantize Qwen/Qwen3-4B --format fp4 -o qwen3_4b_fp4

# Trellis v2 (recommended for MoE models)
metal-marlin quantize Qwen/Qwen3-30B-A3B --format trellis-v2 -o qwen3_30b_trellis
```

## Supported Formats

| Format | Bits | Best For | Notes |
|--------|------|----------|-------|
| FP4 | 4 | Dense models | Fast, good quality |
| INT4 | 4 | Dense models | Integer quantization |
| Trellis v2 | 2-8 | MoE models | Dynamic bit allocation |

## Trellis v2 Quantization

Trellis v2 uses sensitivity-based bit allocation for MoE models:

- **8-bit**: Critical experts (top 10% activation)
- **5-6 bit**: Average experts
- **2-3 bit**: Cold/rarely-used experts

**Result**: 40-50% smaller models with <1% quality loss.

```bash
# Dynamic 2-8 bit allocation for MoE
uv run python -m metal_marlin.quantization.trellis_v2_pipeline \
    --model Qwen/Qwen3-30B-A3B \
    --output ./qwen3_30b_trellis \
    --bits 4 \
    --dynamic-experts \
    --expert-min-bits 2 \
    --expert-max-bits 8
```

### Fast Quantization Mode

For large MoE models (64+ experts), use fast mode for 5-10x speedup:

```python
from metal_marlin.quantization.pipelined_quant import quantize_moe_experts_fast

results, metadata = quantize_moe_experts_fast(
    expert_weights,
    expert_activations,
    quantizer,
    hessian_fn,
    calibration_samples=512,    # 4x fewer samples
    parallel_experts=4,          # Parallel quantization
)
```

**Fast mode optimizations:**
- Gershgorin PSD fast-check (skip eigendecomp)
- 4x fewer calibration samples (512 vs 2048)
- Cold expert skipping (bottom 10%)
- Parallel expert quantization

## Model Format

Quantized models use the trellis_v2 format (uint8 packed weights):

```
model-name-trellis-v2-3bpw/
├── model-00001-of-00010.safetensors
├── model-00002-of-00010.safetensors
├── ...
├── model.safetensors.index.json
├── quantization_config.json
└── config.json
```

### Loading Quantized Models

```python
from metal_marlin import load_quantized_model

model, tokenizer = load_quantized_model("models/Model-Trellis-3bpw")
```

## More Details

- [Calibration Guide](../guides/calibration.md) - Calibration data selection
- [Mixed BPW Inference](../guides/mixed_bpw_inference.md) - Mixed-precision inference
- [Distributed Quantization](../guides/distributed_quantization.md) - Multi-machine quantization
