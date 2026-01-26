# Metal Marlin

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac.

**Features:**
- **2-8 bit weights** — FP4/FP8/INT4/INT3/INT2, with mixed precision per-layer
- **MoE optimized** — Higher bits for cold experts, lower bits for hot experts
- **Quantized KV cache** — FP4/INT4 fused into attention kernels
- **2:4 structured sparsity** — 1.6× additional compression
- **Multiple formats** — HuggingFace, Safetensors, GGUF, ONNX

## Requirements

- macOS 13.0+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12

## Installation

```bash
uv pip install numpy safetensors huggingface_hub torch \
    pyobjc-core pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
```

## Quick Start

```bash
# Quantize a model (Qwen3-4B fits on any Mac, ~2GB quantized)
python -m metal_marlin.hf_loader Qwen/Qwen3-4B ./Qwen3-4B-FP4 --bits 4

# Benchmark against GGUF
python -m metal_marlin.benchmark_models --models "Qwen/Qwen3-4B"

# Run inference
python -m metal_marlin.inference ./Qwen3-4B-FP4 --prompt "The capital of France is"
```

```python
from metal_marlin.inference import MetalInferenceEngine
from metal_marlin.safetensors_loader import load_model

model = load_model("./Qwen3-4B-FP4")
engine = MetalInferenceEngine(model)
output = engine.generate("The capital of France is", max_tokens=50)
```

See [CLI Reference](docs/cli.md) for full options.

## OpenAI-Compatible Server

Metal Marlin includes a vLLM-style server for hosting quantized models:

### Quick Start

```bash
# Quantize a model (if not already done)
metal-marlin quantize Qwen/Qwen3-4B --format fp4 -o qwen3_4b_fp4

# Start the server
metal-marlin serve qwen3_4b_fp4 --port 8000
```

### API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)
- `POST /v1/completions` - Text completions (legacy)
- `GET /health` - Health check

### Usage with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",  # No auth required
)

response = client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Performance

On Apple Silicon M4 Max (PyTorch MPS backend):
- Qwen3-4B FP4: ~27 tok/s decode
- GLM-4.7-Flash MLA: In development

**Note:** Currently using PyTorch MPS fallback. Fused Metal kernels are in development for higher throughput.

## Apple Silicon Performance

On M3/M4, BF16 and FP16 have nearly identical throughput (~14.8 TFLOPS on M4 Max), with FP32 only ~10% slower. **Prefer BF16** — same speed as FP16 but 8× larger dynamic range.

Run `python benchmarks/bench_dtype_perf.py` to measure your hardware. See [dtype configuration](docs/dtype_configuration.md) for details.

## Architecture

Metal Marlin uses **PyTorch MPS + native Metal shaders** (via PyObjC), not MLX. This enables GPTQ/AWQ-class quantization with Hessian-informed rounding and per-layer mixed precision.

```
PyTorch MPS tensors → zero-copy MTLBuffer → custom Metal shaders → results back to PyTorch
```

**Quantization formats:**

| Format | Bits | Best For |
|--------|------|----------|
| FP4/INT4 | 4 | Primary weights, hot MoE experts |
| FP8 | 8 | Cold MoE experts, quality-sensitive layers |
| INT3/INT2 | 2-3 | Extreme compression, cold experts |
| 2:4 Sparse | varies | Additional 1.6× compression |

See [Architecture](docs/architecture.md) and [Why Not MLX?](docs/why_not_mlx.md) for details.

## How It Works

Metal Marlin ports [Marlin](https://arxiv.org/abs/2408.11743) (fast quantized GEMM kernels for CUDA) to Apple Silicon. Weights are unpacked using ALU operations instead of lookup tables, eliminating memory bandwidth bottlenecks.

## Documentation

- **[Getting Started](docs/getting_started.md)** — Install, quantize, run, verify
- [CLI Reference](docs/cli.md) — Full command-line options
- [Calibration Guide](docs/calibration.md) — Custom calibration for higher quality
- [Mixed Precision](docs/mixed_precision.md) — Per-layer precision for MoE
- [Architecture](docs/architecture.md) — Internal design
- [Troubleshooting](docs/troubleshooting.md) — Common issues

## References

Based on [Marlin](https://arxiv.org/abs/2408.11743), [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [FlashAttention](https://arxiv.org/abs/2205.14135), and [vLLM PagedAttention](https://arxiv.org/abs/2309.06180). See [docs/references.md](docs/references.md) for full citations.

## Status

See [STATUS.md](STATUS.md) for implementation progress and model compatibility.

## License

Apache 2.0
