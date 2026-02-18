# Metal Marlin - Fast LLM Inference on Apple Silicon

<p align="center">
  <img src="assets/metalmarlinfish_cropped.png" alt="Metal Marlin" width="500">
</p>

<p align="center">
  <strong>Run GLM-4.7-Flash and other LLMs at 70+ tok/s on your Mac</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#serving">Serving</a> •
  <a href="#documentation">Docs</a> •
  <a href="STATUS.md">Status</a>
</p>

---

## Requirements

- **macOS 13.0+** (Ventura or later)
- **Apple Silicon** (M1/M2/M3/M4)
- **Python 3.12** (via [uv](https://docs.astral.sh/uv/))

## Install

```bash
git clone https://github.com/RESMP-DEV/metal-marlin.git
cd metal-marlin
uv sync --extra all
```

## Quick Start

### GLM-4.7-Flash (Recommended)

The fastest way to run GLM-4.7-Flash with Trellis quantization:

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
prompt = "Explain quantum computing in simple terms."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("mps")
output = model.generate(input_ids, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Performance (M4 Max):** 56-74 tok/s decode, ~15 GB memory

### Streaming Generation

```python
# Stream tokens as they're generated
for token in model.generate_stream(input_ids, max_new_tokens=100):
    print(tokenizer.decode(token), end="", flush=True)
```

### Other Models (Transformers Integration)

For Qwen, LLaMA, etc.:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from metal_marlin import replace_linear_layers
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", 
    torch_dtype=torch.bfloat16, 
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Quantize to 4-bit
replace_linear_layers(model, bits=4, group_size=128)

# Generate
output = model.generate(tokenizer("Hello!", return_tensors="pt").input_ids.to("mps"))
print(tokenizer.decode(output[0]))
```

## Serving

OpenAI-compatible API server:

```bash
# Start server
metal-marlin serve ./qwen3_4b_fp4 --port 8000

# With paged attention for higher throughput
metal-marlin serve ./qwen3_4b_fp4 --port 8000 --enable-batching

# With separate metrics port
metal-marlin serve ./qwen3_4b_fp4 --port 8000 --metrics-port 9090
```

**Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat (supports streaming) |
| `POST /v1/completions` | Text completions |
| `POST /v1/perplexity` | Evaluate text perplexity |
| `GET /v1/perplexity/stats` | Perplexity statistics |
| `GET /metrics` | Prometheus metrics |
| `GET /health` | Health check |

**OpenAI SDK Compatible:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen3_4b_fp4",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Quantization

Convert models to efficient quantized formats:

```bash
# Basic FP4 quantization
metal-marlin quantize Qwen/Qwen3-4B --format fp4 -o qwen3_4b_fp4

# Trellis v2 (recommended for MoE models)
metal-marlin quantize Qwen/Qwen3-30B-A3B --format trellis-v2 -o qwen3_30b_trellis
```

For MoE models, dynamic bit allocation reduces size 40-50% with <1% quality loss.
See [Quantization Guide](docs/quantization/quantization.md) for details.

## C++ Extension (Optional, 5-10x Faster Dispatch)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cp _cpp_ext.cpython-312-darwin.so ../metal_marlin/
```

Check availability:
```python
from metal_marlin import fast_dispatch_available
print(f"Fast dispatch: {fast_dispatch_available()}")  # True if C++ ext built
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/guides/getting_started.md) | Installation and first inference |
| [Quantization Guide](docs/quantization/quantization.md) | FP4/Trellis quantization |
| [Trellis Inference](docs/inference/trellis.md) | GLM-4.7-Flash inference details |
| [MMFP4 Inference](docs/inference/mmfp4.md) | MMFP4-quantized models |
| [Architecture](docs/concepts/architecture.md) | Internal design |
| [Fast Dispatch](docs/internals/fast_dispatch.md) | C++ extension internals |
| [Metal Shaders](docs/internals/metallib_architecture.md) | Precompiled metallib |
| [STATUS.md](STATUS.md) | Implementation status |

## Development

```bash
uv run pytest tests/ -v               # Full test suite
uv run pytest tests/ -v -m smoke      # Quick smoke tests
uv run ruff check .                   # Linting
uv run pyright metal_marlin/          # Type checking
```

## License

Apache 2.0
