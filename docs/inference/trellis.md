# Trellis Inference

Standalone inference for trellis-quantized models on Apple Silicon via Metal acceleration.

## Quick Start

```python
from metal_marlin.trellis import TrellisForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-3bpw", device="mps")
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")

# Generate
input_ids = tokenizer("Explain quantum computing:", return_tensors="pt").input_ids.to("mps")
output = model.generate(input_ids, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(output[0]))
```

## Features

| Feature | Description |
|---------|-------------|
| **MLA Attention** | 8x KV cache memory savings via compression |
| **MoE Support** | GLM-4.7-Flash (64 experts + shared expert per layer) |
| **Metal Acceleration** | On-the-fly dequantization via custom Metal shaders |
| **Streaming Generation** | Token-by-token output with `stream_generate()` |
| **Fast Decode Path** | 56-74 tok/s via pre-dequantized weights |

## Performance (M4 Max)

| Metric | Value |
|--------|-------|
| Decode throughput | ~2 tok/s |
| Prefill throughput | ~32 tok/s |
| Memory usage | ~60 GB |

> **Note:** Performance varies significantly based on context length and model configuration.
> See [reports/tps_benchmark_2026-02-18.md](reports/tps_benchmark_2026-02-18.md) for detailed benchmarks.

## Fast Decode Path

For single-token decode (M=1), Metal Marlin uses pre-dequantized weights with native PyTorch MPS operations:

```python
# Fast path is automatic for decode
for token in model.generate_stream(input_ids, max_new_tokens=100):
    print(tokenizer.decode(token), end="", flush=True)
```

**How it works:**
1. On first decode, weights are dequantized FP4 → FP16
2. Native `torch.nn.functional.linear()` is used for M=1
3. Bypasses PyObjC dispatch overhead (~3.7ms → 0.15ms per layer)

## Architecture

| Module | Purpose |
|--------|---------|
| `model.py` | MoE/dense transformer layers, `MixedBPWMoEDispatcher` |
| `attention.py` | MLA (Multi-head Latent Attention) with KV compression |
| `linear.py` | Quantized linear with Metal dequant kernels |
| `loader.py` | Layer-wise streaming model loading |
| `generate.py` | Text generation with sampling and streaming |
| `kv_cache_compressed.py` | Compressed KV cache for MLA |
| `async_dispatch.py` | `AsyncCommandBufferManager` + `LayerBatchContext` |
| `optimizations.py` | Expert selection cache + memory pool |
| `lm.py` | `TrellisForCausalLM` language model wrapper |

## Streaming Generation

```python
# Stream tokens as they're generated
for token in model.generate_stream(input_ids, max_new_tokens=100):
    print(tokenizer.decode(token), end="", flush=True)
```

## More Details

- [MMFP4 Inference](mmfp4.md) - MMFP4-quantized models
- [Serving Guide](../guides/serving.md) - OpenAI-compatible API server
- [Architecture](../internals/architecture.md) - Internal design details
