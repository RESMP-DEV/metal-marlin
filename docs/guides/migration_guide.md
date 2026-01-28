# Migration Guide: Legacy to Transformers Integration

Metal Marlin v2.0 introduces a new architecture that leverages HuggingFace
Transformers for model structure instead of reimplementing models.

## Why the Change?

The legacy approach required reimplementing each model architecture:
- `MetalGLM47Model` for GLM-4
- `QuantizedLlamaLayer` for Llama
- etc.

This was:
- Error-prone (we missed MoE support in GLM-4.7-Flash)
- Slow to support new models
- Duplicating battle-tested Transformers code

The new approach: Transformers handles architecture, we handle optimization.

## Migration Examples

### Before (Legacy)

```python
from metal_marlin.inference_metal import MetalGLM47Model

model = MetalGLM47Model(config, checkpoint_path, device="mps")
output = model(input_ids)  # Custom forward pass
```

### After (Transformers Integration)

```python
from transformers import AutoModelForCausalLM
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash")
replace_linear_layers(model, bits=4)
output = model.generate(input_ids)  # Transformers' generate()
```

## Loading Existing Checkpoints

Existing quantized checkpoints (created with `hf_loader.py`) still work:

```python
from metal_marlin import load_quantized
model = load_quantized("./my_quantized_model", device="mps")
```

## Deprecated Classes

The following are deprecated and will be removed in v3.0:

| Deprecated | Replacement |
|------------|-------------|
| `MetalGLM47Model` | `AutoModelForCausalLM` + `replace_linear_layers()` |
| `QuantizedLlamaLayer` | `AutoModelForCausalLM` + `replace_linear_layers()` |
| `MarlinPipeline` | `TransformersMarlinPipeline` |

## FAQ

**Q: Will my existing quantized models work?**
A: Yes, `load_quantized()` handles both old and new checkpoint formats.

**Q: Do I need to re-quantize my models?**
A: No, but re-quantizing with `--use-transformers` enables MoE support.

**Q: What about custom attention implementations?**
A: Custom attention (MLA, Flash, etc.) is still used when beneficial.
   Transformers handles the model structure; we optimize the compute.
