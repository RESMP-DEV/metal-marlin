# Metal Marlin

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac.

[![Tests](https://img.shields.io/badge/tests-1565%20passing-brightgreen)]()
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
git clone https://github.com/RESMP-DEV/AlphaHENG.git
cd AlphaHENG/contrib/metal_marlin
uv venv && source .venv/bin/activate
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

```bash
# Quantize and serve
metal-marlin quantize Qwen/Qwen3-4B --format fp4 -o qwen3_4b_fp4
metal-marlin serve qwen3_4b_fp4 --port 8000
```

Compatible with OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Performance

| Model | Memory | Speed |
|-------|--------|-------|
| Qwen3-4B FP4 | ~2GB | ~27 tok/s |
| Llama-3.1-8B FP4 | ~4GB | ~20 tok/s |
| GLM-4.7-Flash (30B MoE) | ~15GB | Batched dispatch |

Benchmarked on M4 Max with PyTorch MPS backend.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/guides/getting_started.md) | Installation and first inference |
| [CLI Reference](docs/guides/cli.md) | Command-line options |
| [Architecture](docs/concepts/architecture.md) | Internal design |
| [STATUS.md](STATUS.md) | Implementation progress and test results |

## Development

```bash
uv run pytest tests/ -v              # Full suite (~4 min, 1426 tests)
uv run pytest tests/ -v -m smoke     # Quick smoke tests
uv run ruff check .                  # Linting
uv run pyright metal_marlin/         # Type checking
```

## License

Apache 2.0
