# Metal Marlin

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac.

[![Tests](https://img.shields.io/badge/tests-1565%20collected-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

> **✅ C++ Extension Working:** The `_cpp_ext` nanobind module builds and provides 5-10x faster dispatch.
> See [STATUS.md](STATUS.md) for build instructions and available exports.

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
- Python 3.12 (via uv)

## Installation

```bash
git clone https://github.com/RESMP-DEV/metal-marlin.git
cd metal-marlin
uv sync --extra all
```

### C++ Extension (5-10x faster dispatch)

The C++ extension is now working and provides significantly faster kernel dispatch:

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Copy to package
cp _cpp_ext.cpython-312-darwin.so ./metal_marlin/
```

**Available exports:**
```python
from metal_marlin._cpp_ext import (
    BatchDispatch, BufferPool, EncoderCache, LibraryManager,
    ManagedBuffer, MetalContext, MetalDevice, QueueManager,
    dispatch_kernel, create_buffer
)
```

**Dispatch speedup:**
| Method | Latency |
|--------|--------|
| C++ extension | ~5-15μs |
| PyObjC fallback | ~80-150μs |

**Note:** The pure PyObjC path remains fully functional if you don't build the extension.

## Dispatch Performance

Metal Marlin automatically uses the fastest available dispatch path:

| Backend | Overhead per call | Used when |
|---------|-------------------|-----------|
| C++ extension | ~5-15μs | Built and available (default) |
| PyObjC | ~80-150μs | Fallback when C++ unavailable |

Check availability:
```python
from metal_marlin import fast_dispatch_available

if fast_dispatch_available():
    print("Using C++ fast path (~5-15μs overhead)")
else:
    print("Using PyObjC fallback (~80-150μs overhead)")
```

For MoE layers with 2-4 expert dispatches, the C++ path saves 200-600μs per layer
during inference.

## GLM-4.7-Flash Inference (Main Use Case)

Run GLM-4.7-Flash with Trellis quantization for fast MoE inference:

```python
from metal_marlin.trellis import TrellisForCausalLM
from transformers import AutoTokenizer

# Load quantized model (~8GB for 3-bit weights)
model = TrellisForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-trellis-v2-3bpw",
    device="mps"
)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")

# Generate
prompt = "<|user|>\nExplain quantum computing in simple terms.\n<|assistant|>\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("mps")
output = model.generate(input_ids, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Performance (M4 Max)

| Metric | Value |
|--------|-------|
| Decode | 5.4 tok/s (185ms/tok) |
| Prefill | 42 tok/s |
| Memory | 16.9 GB |

The fused Trellis GEMM kernels provide **108x speedup** over naive dequant+matmul.

### Streaming Generation

```python
# Stream tokens as they're generated
for token in model.generate_stream(input_ids, max_new_tokens=100):
    print(tokenizer.decode(token), end="", flush=True)
```

## Quick Start (Transformers Integration)

For other models, use HuggingFace Transformers with layer replacement:

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

## Precompiled Metal Shaders

metal_marlin uses precompiled shaders for fast kernel dispatch:

```bash
# Build precompiled metallib (first time only)
./scripts/build_metallib.sh

# Python automatically uses metallib when available
from metal_marlin import get_precompiled_library
lib = get_precompiled_library()  # ~0.01ms
```

### Performance Comparison

| Dispatch Method | Latency |
|-----------------|---------|
| Precompiled metallib | 0.01 ms |
| JIT (first call) | 50-100 ms |
| JIT (cached) | 0.1 ms |

See [docs/metallib_architecture.md](docs/metallib_architecture.md) for details.

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

# Expose metrics on separate port (e.g., for Prometheus scraping)
metal-marlin serve qwen3_4b_fp4 --port 8000 --metrics-port 9090
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
- `--metrics-port N` - Dedicated port for Prometheus metrics (default: served on main port)

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

## Quantization

Metal Marlin supports 2-8 bit quantization with sensitivity-aware bit allocation for MoE models.

### Quick Quantization

```bash
# Basic FP4 quantization
uv run python -m metal_marlin.hf_loader Qwen/Qwen3-4B ./output --bits 4

# Trellis v2 quantization
uv run python -m metal_marlin.quantization.trellis_v2_pipeline \
    --model Qwen/Qwen3-4B --output ./qwen3_trellis_v2 --bits 4
```

### Dynamic Bit Allocation (MoE Models)

For MoE models like GLM-4.7-Flash or Qwen3-30B-A3B, use dynamic bit allocation:

```bash
# Dynamic 2-8 bit allocation based on expert sensitivity
uv run python scripts/quantize_qwen3_30b.py \
    --model Qwen/Qwen3-30B-A3B \
    --dynamic-experts \
    --expert-min-bits 2 \
    --expert-max-bits 8
```

This maps each expert's sensitivity score to bit precision:
- **8-bit**: Critical experts (top 10% activation)
- **5-6 bit**: Average experts
- **2-3 bit**: Cold/rarely-used experts

**Result**: 40-50% smaller models with <1% quality loss.

### Fast Quantization Mode

For large MoE models (64+ experts), use fast mode for 5-10x speedup:

```python
from metal_marlin.quantization.pipelined_quant import quantize_moe_experts_fast

results, metadata = quantize_moe_experts_fast(
    expert_weights,
    expert_activations,
    quantizer,
    hessian_fn,
    calibration_samples=512,    # 4x fewer samples (was 2048)
    parallel_experts=4,          # Parallel quantization
)
```

**Fast mode optimizations:**
- Gershgorin PSD fast-check (skip eigendecomp if already PSD)
- 4x fewer calibration samples (512 vs 2048)
- Cold expert skipping (bottom 10%)
- Parallel expert quantization

## Quantized Model Format

Metal Marlin supports **only trellis_v2 (uint8 packed)** weights.

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

## Performance

Benchmark results on Apple M4 Max with GLM-4.7-Flash (Trellis 3-bit quantization).

### Throughput (M4 Max)

| Metric | Value |
|--------|-------|
| Prefill (2K context) | 42 tok/s |
| Decode | 5.4 tok/s (185 ms/tok) |
| Memory | 16.9 GB |

### Baseline vs Optimized Comparison

| Configuration | Decode (tok/s) | Speedup | Memory |
|---------------|----------------|---------|--------|
| Naive dequant+matmul | 0.05 tok/s | 1x | 61 GB (unpacked) |
| Fused trellis kernel | 5.4 tok/s | **108x** | 16.9 GB |

### Optimizations Applied

- **Fused GEMM kernels**: `gemm_trellis_packed` combines dequantization and matrix multiplication in a single kernel pass, eliminating intermediate memory traffic
- **Decode-optimized tiles**: `gemm_trellis_packed_decode` uses 32x128 tiles tuned for single-token generation
- **MoE batched dispatch**: `moe_trellis_swiglu` processes all active experts in a single batched kernel call (200x faster than sequential dispatch)
- **Packed weight format**: Maintains uint8 packed weights throughout inference
- **SIMD-aligned memory access**: 128-byte aligned reads/writes for optimal memory bandwidth utilization on Apple Silicon

### GEMM Kernel Performance

Performance on GLM-4.7-Flash expert shapes (2048x1536):

| Batch Size | Reference (ms) | Fused (ms) | Speedup |
|------------|----------------|------------|---------|
| 1 | 145.2 | 2.8 | **51.9x** |
| 32 | 162.4 | 4.2 | **38.7x** |
| 128 | 189.6 | 12.5 | **15.2x** |

### Comparison to Other Backends

| Backend | Device | GLM-4.7-Flash | Notes |
|---------|--------|---------------|-------|
| Metal Marlin | M4 Max | 5.4 tok/s | Fused trellis kernels |
| llama.cpp | M4 Max | ~3 tok/s | Q4_K_M quantization |
| MLX | M4 Max | ~4 tok/s | Native Apple framework |

### Known Limitations

- **Prefill-bound at long contexts**: KV cache writes dominate at >8K context length
- **No speculative decoding**: Single-token generation only (planned)
- **No continuous batching**: Batch size 1 inference (server mode supports concurrent requests via paged attention)
- **MLA overhead**: Multi-head Latent Attention compression adds ~10% decode latency vs standard MHA

### Running Benchmarks

```bash
cd contrib/metal_marlin
uv run python benchmarks/glm_flash_benchmark.py
```

Options:
- `--model PATH`: Path to model directory (default: models/GLM-4.7-Flash-trellis-v2-3bpw)
- `--context-length N`: Context length for prefill benchmark (default: 2048)
- `--num-tokens N`: Number of tokens to decode (default: 100)
- `--num-runs N`: Number of benchmark iterations (default: 3)

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

model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-trellis-v2-3bpw", device="mps")
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
model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-trellis-v2-3bpw")
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")

# Model is automatically loaded to Metal (MPS) device
print(f"Model loaded: {model.config.model_type}")
print(f"Device: {model.device}")
```

**Supported model formats:**
- trellis_v2 (uint8 packed) quantized models
- MoE and dense architectures within trellis_v2

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
    --model models/GLM-4.7-Flash-trellis-v2-3bpw \
    --dataset wikitext \
    --split test

# Token throughput benchmark
python scripts/benchmark_throughput.py \
    --model models/GLM-4.7-Flash-trellis-v2-3bpw \
    --prompt "Explain quantum computing:" \
    --max_tokens 256 \
    --iterations 10

# Memory profiling
python scripts/profile_memory.py \
    --model models/GLM-4.7-Flash-trellis-v2-3bpw \
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

## Development

See [docs/development_setup.md](docs/development_setup.md) for complete environment management guidelines.

**Quick sanity check:**
```bash
# Verify you're using the correct environment
uv run python -c "import sys; print(sys.prefix)"
# Should show: /path/to/contrib/metal_marlin/.venv

# Clean up leftover environments (if any)
./scripts/cleanup_venvs.sh
```

**Running tests:**
```bash
cd contrib/metal_marlin
uv run pytest tests/ -v --tb=short

# Specific test file
uv run pytest tests/test_gemm.py -v

# With pattern matching
uv run pytest -k "moe" tests/ -v
```
