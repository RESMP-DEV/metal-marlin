# Getting Started

Quantize and run a model in under 5 minutes.

## Installation

```bash
uv pip install metal-marlin
```

## Quantize a Model

```bash
python -m metal_marlin quantize \
    --input Qwen/Qwen3-4B \
    --output Qwen3-4B-FP4 \
    --method mr-gptq
```

Takes ~10 minutes. Downloads calibration data automatically.

## Run Inference

```python
from metal_marlin.inference import MetalInferenceEngine
from metal_marlin.safetensors_loader import load_model

model = load_model("./Qwen3-4B-FP4")
engine = MetalInferenceEngine(model)

output = engine.generate("The theory of relativity", max_tokens=100)
print(output)
```

## Chat

```python
messages = [{"role": "user", "content": "What is the capital of France?"}]
response = engine.chat(messages)
print(response)
```

## Model Sizes

| Model | FP16 | Quantized | Min RAM |
|-------|------|-----------|---------|
| Qwen3-4B | 8 GB | 2.5 GB | 16 GB |
| Qwen3-8B | 16 GB | 5 GB | 24 GB |

## Next Steps

- [Mixed Precision](mixed_precision.md) — Per-layer precision for MoE models
- [API Reference](api.md) — Full Python API
