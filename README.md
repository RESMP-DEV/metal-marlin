# Metal Marlin - Fast LLM Inference on Apple Silicon

<p align="center">
  <img src="assets/metalmarlinfish_cropped.png" alt="Metal Marlin" width="500">
</p>

<p align="center">
  <strong>GLM-4.7-Flash at 35 tok/s on M4 Max • OpenAI-Compatible API • Production Ready</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#serving">Serving</a> •
  <a href="#performance">Performance</a> •
  <a href="docs/serving_guide.md">Docs</a>
</p>

---

## Performance

**GLM-4.7-Flash (35 tok/s on M4 Max)**
- Throughput: **35.2 tok/s** decode
- Latency: **28.6 ms/step**
- Memory: **12.4 GB**
- Optimization: **4.9× speedup** from baseline

[See optimization details →](docs/optimization_status_final.md)

## Requirements

- **macOS 13.0+** (Ventura or later)
- **Apple Silicon** (M1/M2/M3/M4, M4 Max recommended)
- **Unified Memory**: 32GB+ recommended
- **Python 3.12** (via [uv](https://docs.astral.sh/uv/))

## Install

```bash
git clone https://github.com/RESMP-DEV/metal-marlin.git
cd metal-marlin
uv sync --extra all
```

## Quick Start

### GLM-4.7-Flash Server (Production)

Start an OpenAI-compatible server at 35 tok/s:

```bash
# Start server with optimized config
uv run python scripts/serve_glm47.py \
  --model-path ./models/glm47-flash-mmfp4 \
  --port 8000 \
  --max-batch-size 32

# Test with any OpenAI SDK
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

**Features:**
- ✅ OpenAI API compatible (drop-in replacement)
- ✅ Request batching (32 concurrent)
- ✅ PagedAttention KV cache
- ✅ Streaming support
- ✅ Metrics endpoint

**Full Guide:** [docs/QUICKSTART_SERVING.md](docs/QUICKSTART_SERVING.md)

### Python API

```python
from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline
from metal_marlin.trellis.config import GLM4_TOKENIZER_ID

# Load optimized pipeline (35 tok/s)
pipeline = MMFP4Pipeline.from_pretrained(
    "./models/glm47-flash-mmfp4",
    device="mps"
)

# Generate
output = pipeline(
    "Explain quantum computing",
    max_new_tokens=100,
    temperature=0.7
)
print(output)
```

### Validation

Run end-to-end validation:

```bash
# Validate TPS (target: 35 tok/s) and perplexity
./tests/validation/run_all_validation.sh
```

See [Validation Checklist](docs/VALIDATION_CHECKLIST.md) for details.

---

## Serving

### OpenAI-Compatible Server

**Start server:**
```bash
uv run python scripts/serve_glm47.py \
  --model-path ./models/glm47-flash-mmfp4 \
  --host 0.0.0.0 \
  --port 8000
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
