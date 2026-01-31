# Metal Marlin

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac.

[![Tests](https://img.shields.io/badge/tests-1565%20collected-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

## Overview

Metal Marlin ports [Marlin](https://arxiv.org/abs/2408.11743) quantized GEMM kernels to Apple Silicon via native Metal shaders. It integrates with HuggingFace Transformers—swap `nn.Linear` layers with `MetalQuantizedLinear` and run inference with optimized kernels.

**Key features:**

- 2-8 bit quantization (FP4/FP8/INT4/INT3/INT2)
- MoE support with batched expert dispatch
- Transformers integration (automatic layer replacement)
- OpenAI-compatible serving endpoint

## Requirements

- macOS 13.0+ (Ventura)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12

## Installation

```bash
git clone https://github.com/RESMP-DEV/metal-marlin.git
cd metal-marlin
uv sync --extra all
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Quantize and run
replace_linear_layers(model, bits=4, group_size=128)
output = model.generate(tokenizer("Hello!", return_tensors="pt").input_ids.to("mps"))
print(tokenizer.decode(output[0]))
```

## Serving

OpenAI-compatible API server with streaming, concurrent requests, paged attention, and Prometheus metrics.

```bash
# Quantize and serve
metal-marlin quantize Qwen/Qwen3-4B --format fp4 -o qwen3_4b_fp4
metal-marlin serve qwen3_4b_fp4 --port 8000

# Enable paged attention for higher throughput
metal-marlin serve qwen3_4b_fp4 --port 8000 --enable-batching

# Tune KV cache (with paged attention)
metal-marlin serve qwen3_4b_fp4 --enable-batching --num-kv-blocks 1024 --block-size 32
```

**Endpoints:**
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `POST /v1/completions` - Text completions
- `GET /metrics` - Prometheus metrics
- `GET /health` - Health check

**CLI Options:**
- `--enable-batching` - Enable paged attention with continuous batching
- `--num-kv-blocks N` - Number of KV cache blocks (default: 512)
- `--block-size N` - Tokens per block (default: 16)

Compatible with OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Non-streaming
response = client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Streaming
for chunk in client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Trellis Inference

Standalone inference for trellis-quantized models on Apple Silicon via Metal acceleration.

### Quick Start

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

### Features

- **MLA (Multi-head Latent Attention)**: 8x KV cache memory savings via compression
- **MoE Support**: GLM-4.7-Flash (64 experts + shared expert per layer)
- **Metal Acceleration**: On-the-fly dequantization via custom Metal shaders
- **Streaming Generation**: Token-by-token output with `stream_generate()`
- **HuggingFace Tokenizers**: Native compatibility

### Architecture

| Module | Purpose |
|--------|---------|
| `trellis_model.py` | Complete model with MoE/dense layers |
| `trellis_attention.py` | MLA with KV compression |
| `trellis_linear.py` | Quantized linear with Metal dequant |
| `trellis_loader.py` | Layer-wise model loading |
| `trellis_generate.py` | Text generation with sampling |
| `trellis_kv_cache.py` | Compressed KV cache for MLA |

### Loading Quantized Models

Trellis models use a layer-directory format:
```
model_dir/
  config.json           # Model config
  base_weights.safetensors  # Embeddings, norms, lm_head
  layer_0000/           # Per-layer quantized weights
    index.json
    tensor_*.safetensors
  layer_0001/
  ...
```

The loader auto-detects weight format and handles MoE routing weights separately.

## ASR: Parakeet-TDT-0.6B

Speech recognition with hybrid Metal + ANE acceleration.

### Quick Start

```python
from metal_marlin.asr import build_hybrid_parakeet

model = build_hybrid_parakeet("models/parakeet-tdt-0.6b-fp4", use_ane_conv=True)
transcript = model.transcribe("audio.wav")
print(transcript)
```

### Performance (M4 Max)

| Config | RTF | Memory | WER (test-clean) |
|--------|-----|--------|------------------|
| FP16 MPS | 15x | 1.2 GB | 4.2% |
| 4-bit Metal | 25x | 400 MB | 4.3% |
| 4-bit Hybrid | 35x | 400 MB | 4.3% |

RTF = Real-Time Factor (higher is better)

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/guides/getting_started.md) | Installation and first inference |
| [CLI Reference](docs/guides/cli.md) | Command-line options |
| [Architecture](docs/concepts/architecture.md) | Internal design |
| [STATUS.md](STATUS.md) | Implementation progress and test results |

## Development

```bash
uv run pytest tests/ -v              # Full suite (~4 min, 1565 tests)
uv run pytest tests/ -v -m smoke     # Quick smoke tests
uv run ruff check .                  # Linting
uv run pyright metal_marlin/         # Type checking
```

## License

Apache 2.0

---

## Trellis Inference Usage

Complete guide for running inference with Trellis-quantized models using the simplified `TrellisForCausalLM` API.

### Quick Start

Load and run inference with 5 lines of code:

```python
from metal_marlin.trellis import TrellisForCausalLM
from transformers import AutoTokenizer

model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-EXL3-3bpw", device="mps")
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")

input_ids = tokenizer("Hello, ", return_tensors="pt").input_ids.to("mps")
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### Model Loading

Use `TrellisForCausalLM.from_pretrained()` to load quantized models:

```python
from metal_marlin import TrellisForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-EXL3-3bpw")
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")

# Model is automatically loaded to Metal (MPS) device
print(f"Model loaded: {model.config.model_type}")
print(f"Device: {model.device}")
```

**Supported model formats:**
- Trellis quantized models (`.trellis` or `EXL3` format)
- Layer-wise directory structure with `config.json` and per-layer weights
- MoE and dense architectures

### Generation

Control generation with sampling parameters:

```python
# Basic generation
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,      # Sampling temperature (0.0 = greedy)
    top_k=50,             # Top-k sampling (0 = disabled)
    top_p=0.9,            # Nucleus sampling (1.0 = disabled)
)

# Streaming generation
for token in model.generate_stream(input_ids, max_new_tokens=50):
    print(tokenizer.decode(token), end="", flush=True)
```

**Parameter reference:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 100 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature (0.0 = deterministic) |
| `top_k` | 50 | Top-k sampling cutoff |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `repetition_penalty` | 1.0 | Penalty for repeated tokens |

### Benchmarking

Run evaluation scripts to measure performance:

```bash
# Perplexity evaluation
cd contrib/metal_marlin
python scripts/eval_perplexity.py \
    --model models/GLM-4.7-Flash-EXL3-3bpw \
    --dataset wikitext \
    --split test

# Token throughput benchmark
python scripts/benchmark_throughput.py \
    --model models/GLM-4.7-Flash-EXL3-3bpw \
    --prompt "Explain quantum computing:" \
    --max_tokens 256 \
    --iterations 10

# Memory profiling
python scripts/profile_memory.py \
    --model models/GLM-4.7-Flash-EXL3-3bpw \
    --batch_sizes 1,4,8
```

**Benchmark outputs:**
- `eval_perplexity.py`: Perplexity score, eval time, tokens/sec
- `benchmark_throughput.py`: Average tok/s, latency percentiles (P50, P95, P99)
- `profile_memory.py`: Peak VRAM/RAM usage per batch size

### Memory Usage

Expected VRAM/RAM for different models (Apple Silicon):

| Model | Quantization | Memory | Context (4K) | Context (8K) |
|-------|-------------|--------|--------------|--------------|
| GLM-4.7-Flash | 2-bit  | ~6 GB | ~7 GB | ~9 GB |
| GLM-4.7-Flash | 3-bit  | ~8 GB | ~9 GB | ~11 GB |
| GLM-4.7-Flash | 4-bit  | ~10 GB | ~11 GB | ~13 GB |
| Qwen3-8B | FP4 | ~4 GB | ~5 GB | ~7 GB |
| Llama-3.1-8B | FP4 | ~4 GB | ~5 GB | ~7 GB |
| Llama-3.1-70B | FP4 | ~35 GB | ~40 GB | ~50 GB |

**Notes:**
- Memory includes model weights + KV cache + activations
- MLA (Multi-head Latent Attention) reduces KV cache by ~8x
- MoE models use sparse expert activation (2-4 experts per token)
- Add ~1-2 GB overhead for Metal driver and PyTorch
