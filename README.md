# Metal Marlin

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac.

## Design Philosophy

Metal Marlin **optimizes, not reimplements**. Model architectures (MoE routing, attention patterns, layer structure) come from HuggingFace Transformers. Our job is to swap `nn.Linear` layers with optimized `MetalQuantizedLinear` layers that use Metal shaders for fast dequantization and matrix multiplication.

This means:
- **New model support is automatic** when Transformers adds it
- **Battle-tested code** handles generation, caching, and edge cases
- **Our focus** stays on kernel optimization, not architecture reimplementation

**Features:**
- **2-8 bit weights** — FP4/FP8/INT4/INT3/INT2, with mixed precision per-layer
- **MoE optimized** — Batched expert dispatch via Metal kernels (GLM-4.7-Flash, Mixtral)
- **Quantized KV cache** — FP4/INT4 fused into attention kernels
- **2:4 structured sparsity** — 1.6× additional compression
- **Transformers integration** — Automatic `nn.Linear` → `MetalQuantizedLinear` swap
- **Multiple formats** — HuggingFace, Safetensors, GGUF, ONNX

**Status:** See [STATUS.md](STATUS.md) for implementation progress and latest test results.

## Requirements

- macOS 13.0+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12

## Installation

```bash
# Clone and install
git clone https://github.com/RESMP-DEV/AlphaHENG.git
cd AlphaHENG/contrib/metal_marlin
uv venv && source .venv/bin/activate
uv sync --extra all
```

Or install dependencies manually:

```bash
uv pip install numpy safetensors huggingface_hub torch transformers \
    pyobjc-core pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
```

## Supported Models

See [STATUS.md](STATUS.md) for the full compatibility matrix and verification notes.

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

### Pre-Quantized Checkpoints

```python
from metal_marlin import load_and_quantize, save_quantized

# Quantize and save
model, stats = load_and_quantize("zai-org/GLM-4.7-Flash", bits=4)
save_quantized(model, "./glm47_fp4")

# Load later (instant)
model = load_quantized("./glm47_fp4", device="mps")
```

See [CLI Reference](docs/guides/cli.md) for full options.

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
- Llama-3.1-8B FP4: ~20 tok/s decode (estimated)
- GLM-4.7-Flash (30B-A3B MoE): batched Metal kernel dispatch for routed experts

**Note:** Uses Metal kernel library for quantized GEMM. Fused attention kernels in development for higher throughput.

## Code Quality

```bash
# Linting
uv run ruff check .

# Type checking
uv run pyright metal_marlin/

# Tests (full suite: ~1444 tests, 4 min)
uv run pytest tests/ -v

# Quick smoke tests only
uv run pytest tests/ -v -m smoke

# Skip slow/expensive tests
uv run pytest tests/ -v -m "not slow and not expensive"
```

**Test suite status:** 1444 passing, 0 failing, 53 skipped (256s)

A test cleanup initiative is in progress to reduce redundancy across the 46 test files
(~24K lines). See [STATUS.md](STATUS.md#test-suite-cleanup-in-progress) for details and
`tasks/test_cleanup.yaml` for the task breakdown.

See [STATUS.md](STATUS.md) for the latest results and tracked metrics.

## Apple Silicon Performance

On M3/M4, BF16 and FP16 have nearly identical throughput (~14.8 TFLOPS on M4 Max), with FP32 only ~10% slower. **Prefer BF16** — same speed as FP16 but 8× larger dynamic range.

Run `python benchmarks/bench_dtype_perf.py` to measure your hardware. See [dtype configuration](docs/formats/dtype_configuration.md) for details.

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

See [Architecture](docs/concepts/architecture.md) and [Why Not MLX?](docs/comparisons/why_not_mlx.md) for details.

## How It Works

Metal Marlin ports [Marlin](https://arxiv.org/abs/2408.11743) (fast quantized GEMM kernels for CUDA) to Apple Silicon. Weights are unpacked using ALU operations instead of lookup tables, eliminating memory bandwidth bottlenecks.

## Documentation

- **[Getting Started](docs/guides/getting_started.md)** — Install, quantize, run, verify
- [CLI Reference](docs/guides/cli.md) — Full command-line options
- [Calibration Guide](docs/guides/calibration.md) — Custom calibration for higher quality
- [Mixed Precision](docs/concepts/mixed_precision.md) — Per-layer precision for MoE
- [Architecture](docs/concepts/architecture.md) — Internal design
- [Troubleshooting](docs/guides/troubleshooting.md) — Common issues

## References

Based on [Marlin](https://arxiv.org/abs/2408.11743), [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [FlashAttention](https://arxiv.org/abs/2205.14135), and [vLLM PagedAttention](https://arxiv.org/abs/2309.06180). See [docs/comparisons/references.md](docs/comparisons/references.md) for full citations.

## Status

See [STATUS.md](STATUS.md) for implementation progress and model compatibility.

## License

Apache 2.0
