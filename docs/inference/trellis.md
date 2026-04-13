# Trellis Inference

Standalone inference for trellis-quantized models on Apple Silicon via Metal acceleration.

**Note:** Performance varies based on model configuration, quantization settings, and hardware. For the current benchmark surface, see `benchmarks/glm_flash_benchmark.py`, `benchmarks/bench_comprehensive_e2e.py`, and `reports/tps_benchmark_2026-02-18.md`.

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
| **MLA Attention** | KV cache compression support |
| **MoE Support** | GLM-4.7-Flash (64 experts + shared expert per layer) |
| **Metal Backend** | Models run on Apple MPS device |
| **Streaming Generation** | Token-by-token output with `stream_generate()` |

## Architecture

| Module | Purpose |
|--------|---------|
| `model.py` | MoE/dense transformer layers, `MixedBPWMoEDispatcher` |
| `attention.py` | MLA (Multi-head Latent Attention) with KV compression |
| `linear.py` | Quantized linear operations |
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

## Measured Performance

| Metric | Result | Notes |
|--------|--------|-------|
| **Decode (batch=1, measured E2E)** | ~1.5-2 tok/s | `benchmarks/bench_comprehensive_e2e.py`, summarized in `reports/tps_benchmark_2026-02-18.md` |
| **Prefill (2K context, measured)** | 42 tok/s | `benchmarks/glm_flash_benchmark.py`, M4 Max |

> **Framing:** The current measured end-to-end decode result is ~1.5-2 tok/s. The oft-cited ~43 tok/s figure is a theoretical memory-bandwidth target, not a measured decode result. The gap is still attributed to Metal dispatch overhead, MoE routing, and kernel launch latency.

## More Details

- [MMFP4 Inference](mmfp4.md) - MMFP4-quantized models
- [Serving Guide](../guides/serving.md) - OpenAI-compatible API server
- [Architecture](../internals/architecture.md) - Internal design details
