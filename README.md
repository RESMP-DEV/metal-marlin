# Metal Marlin

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac.

## What This Does

Metal Marlin lets you run quantized LLMs on Apple Silicon using optimized Metal shaders. You don't need to understand GPU programming—just install, quantize (or convert), and run.

**Weight formats:** FP4, FP8, INT4, INT3, INT2 with per-group scales  
**KV cache:** FP4/INT4 quantized (3.8× memory savings for long context)  
**Sparsity:** 2:4 structured sparse with fused metadata decode  
**Input formats:** HuggingFace, Safetensors, GGUF, ONNX  
**Framework:** NumPy-only core, MLX optional for inference

## Installation

```bash
# Using pip
pip install numpy safetensors huggingface_hub

# Using uv (recommended)
uv pip install numpy safetensors huggingface_hub

# For Metal inference (recommended)
pip install mlx  # or: uv pip install mlx

# For PyTorch interop (optional)
pip install torch  # or: uv pip install torch
```

## Quick Start

### Convert an Existing Model

```bash
# From HuggingFace (with calibration)
python -m metal_marlin.hf_loader convert meta-llama/Llama-3.2-1B ./llama-fp4 \
    --calibration bartowski-v3

# From GGUF
python -m metal_marlin.gguf_to_marlin model.gguf ./model-marlin/

# From safetensors (using CLI)
python -m metal_marlin convert --input ./model/ --output ./model-fp4/
```

### Quantize Your Own Model

```bash
# Recommended: MR-GPTQ with built-in calibration
python -m metal_marlin quantize \
    --input Qwen/Qwen3-32B \
    --output Qwen3-32B-FP4 \
    --method mr-gptq \
    --calibration bartowski-v3

# Fast: Round-to-nearest (no calibration needed)
python -m metal_marlin quantize \
    --input Qwen/Qwen3-32B \
    --output Qwen3-32B-FP4 \
    --method rtn
```

For MoE models, add `--mixed-precision moe` to keep routers in full precision.

### Python API

```python
from metal_marlin.quantize import pack_fp4_weights
from metal_marlin.kernels import marlin_gemm_fp4
import numpy as np

# Quantize weights
weight = np.random.randn(4096, 4096).astype(np.float16)
packed, scales = pack_fp4_weights(weight, group_size=128)

# Run GEMM
output = marlin_gemm_fp4(activations, packed, scales, group_size=128)
```

### Mixed-Precision for MoE

```python
from metal_marlin.mixed_precision import MixedPrecisionConfig

config = MixedPrecisionConfig.default_moe()
# Automatically keeps routers in BF16, experts in FP4
```

## How It Works

Metal Marlin ports [Marlin](https://arxiv.org/abs/2312.07723) (NVIDIA's fast quantized GEMM kernels) to Apple Silicon. The key technique is **bitwise dequantization**—weights are unpacked using ALU operations instead of lookup tables, which eliminates memory bandwidth bottlenecks.

The Metal shaders handle all the GPU complexity. From Python, you just call functions.

## Key Features

**Quantized KV Cache:** For long-context inference, the KV cache dominates memory. FP4/INT4 KV cache reduces memory from 4GB to ~1GB at 4K context (32 layers), with dequantization fused into flash attention kernels.

**2:4 Structured Sparsity:** For pruned models, 2:4 sparse format stores only 2 values per 4-element block with metadata. Achieves 1.6× compression with metadata decode interleaved with dequantization to hide latency.

**Chip-Specific Kernels:** Separate shader variants tuned for M1/M2/M3/M4 memory hierarchies and occupancy targets.

## Documentation

**Getting Started**
- [Why Metal Marlin?](docs/why_not_mlx.md) — Comparison with MLX's native quantization
- [Calibration Guide](docs/calibration.md) — Dataset selection and custom calibration
- [Troubleshooting](docs/troubleshooting.md) — Common issues and fixes

**Deep Dives**
- [MR-GPTQ Algorithm](docs/mr_gptq.md) — How Hessian-aware quantization works
- [Architecture](docs/architecture.md) — System design overview
- [CUDA to Metal Porting](docs/cuda_metal_mapping.md) — How we translated the kernels

**Specialized Topics**
- [KV Cache](docs/kv_cache.md) — Quantized KV cache for long-context inference
- [Sparse Format](docs/sparse_format.md) — 2:4 structured sparsity implementation
- [MoE Architecture](docs/moe_architecture.md) — Mixture-of-Experts support
- [Mixed Precision](docs/mixed_precision.md) — Per-layer precision configuration
- [vLLM Comparison](docs/vllm_comparison.md) — Feature parity notes

## Project Structure

```
metal_marlin/
├── src/                    # Metal shaders (30+ kernels)
│   ├── marlin_gemm.metal   # Core quantized GEMM
│   ├── flash_attention.metal
│   ├── moe_*.metal         # MoE dispatch kernels
│   └── sparse_gemm.metal   # 2:4 structured sparsity
├── metal_marlin/           # Python package
│   ├── cli.py              # Command-line interface
│   ├── quantize.py         # Weight packing and quantization
│   ├── kernels.py          # Python bindings to Metal
│   ├── mixed_precision.py  # Per-layer precision config
│   ├── mr_gptq.py          # MR-GPTQ quantization
│   ├── hf_loader.py        # HuggingFace model loading
│   └── gguf_to_marlin.py   # GGUF format conversion
├── converters/             # Format conversion tools
├── docs/                   # Documentation (20+ guides)
├── tests/                  # Test suite (~1400 tests)
├── benchmarks/             # Performance benchmarks
└── examples/               # Usage examples
```

## References

- [Marlin](https://arxiv.org/abs/2312.07723) — Original CUDA kernels this is based on
- [GPTQ](https://arxiv.org/abs/2210.17323) — Hessian-aware quantization
- [QuaRot](https://arxiv.org/abs/2404.00456) — Hadamard rotation for better quality

## Status

See [STATUS.md](STATUS.md) for implementation progress and model compatibility.

## License

Apache 2.0
