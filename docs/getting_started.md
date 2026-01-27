# Getting Started

Quantize and run a model in under 5 minutes.

## Installation

```bash
uv pip install metal-marlin
```

## Quick Start (Recommended)

Metal Marlin integrates with HuggingFace Transformers. Load any supported
model, quantize the Linear layers, and run inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from metal_marlin import replace_linear_layers

# 1. Load model via Transformers (handles MoE, MLA, everything)
model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-4.7-Flash",
    torch_dtype=torch.bfloat16,
    device_map="mps",
)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")

# 2. Quantize Linear layers (our Metal kernels take over)
stats = replace_linear_layers(model, bits=4, group_size=128)
print(f"Quantized {stats['replaced_count']} layers")

# 3. Generate (uses Transformers' battle-tested code)
output = model.generate(
    tokenizer("Hello!", return_tensors="pt").input_ids.to("mps"),
    max_new_tokens=50,
)
print(tokenizer.decode(output[0]))
```

## Supported Models

Any model in HuggingFace Transformers with Linear layers can be quantized:

| Model | Architecture | Special Features |
|-------|--------------|------------------|
| GLM-4.7-Flash | glm4_moe_lite | MoE (64 experts), MLA attention |
| Qwen3-30B-A3B | qwen3_moe | MoE (128 experts) |
| Llama-3.1-* | llama | Standard transformer |
| Mistral-* | mistral | Sliding window attention |
| Mixtral-* | mixtral | MoE (8 experts) |

## Pre-Quantized Checkpoints

For faster loading, save quantized checkpoints:

```python
from metal_marlin import load_and_quantize, save_quantized

# Quantize and save
model, stats = load_and_quantize("zai-org/GLM-4.7-Flash", bits=4)
save_quantized(model, "./glm47_fp4")

# Load later (instant)
model = load_quantized("./glm47_fp4", device="mps")
```

## Next Steps

- [API Reference](api.md) — Full Python API
- [CLI Reference](cli.md) — Command-line tools
- [Mixed Precision](mixed_precision.md) — Per-layer precision for MoE models
