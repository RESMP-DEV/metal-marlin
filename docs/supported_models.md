# Supported Models

Metal Marlin is **Transformers-first**. We do not reimplement model
architectures. Instead, we swap `nn.Linear` layers with Metal-accelerated
`MetalQuantizedLinear` and let HuggingFace Transformers handle attention,
MoE routing, caching, and generation.

## How It Works

1. Load a model via `transformers.AutoModelForCausalLM`
2. Replace `nn.Linear` layers using `replace_linear_layers`
3. Run inference with Transformers' `generate` or pipelines

## Verified Models

These models have been exercised with the Transformers integration tests
(some large checkpoints are optional/manual due to memory).

| Model | Type | Parameters | Special Features | Status |
|-------|------|------------|------------------|--------|
| Llama-3.2-1B | Dense | 1B | RoPE | ✅ Verified |
| Llama-3.1-8B | Dense | 8B | RoPE | ✅ Verified |
| Llama-3.1-70B | Dense | 70B | RoPE | ✅ Verified (manual/large) |
| Qwen3-4B | Dense | 4B | - | ✅ Verified |
| Qwen3-32B | Dense | 32B | - | ✅ Verified (manual/large) |
| Qwen3-30B-A3B | MoE | 30B (3B active) | 128 experts | ✅ Verified |
| GLM-4.7-Flash | MoE | 4.7B | 64 experts, MLA | ✅ Verified |
| Mistral-7B | Dense | 7B | Sliding window | ✅ Verified |
| Mixtral-8x7B | MoE | 47B (13B active) | 8 experts | ✅ Verified (manual/large) |
| Phi-2 | Dense | 2.7B | - | ✅ Verified |
| Phi-3-mini-4k-instruct | Dense | 3.8B | Long context | ✅ Verified |

## Adding New Models

No code changes are required for most Transformer models. Use:

```python
from transformers import AutoModelForCausalLM
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained("any/new-model")
replace_linear_layers(model, bits=4, group_size=128)
```

If a model uses custom modules that are not `nn.Linear` (or uses fused
linear kernels), add them to the skip list or provide a custom layer filter
to avoid replacing incompatible layers.
