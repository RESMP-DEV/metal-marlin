# Metal Marlin

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac.

Supports FP4/FP8/INT4/INT3/INT2 weights, quantized KV cache, and 2:4 structured sparsity. Loads from HuggingFace, Safetensors, GGUF, or ONNX.

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

### Quantize a Model

```bash
python -m metal_marlin quantize \
    --input Qwen/Qwen3-4B \
    --output Qwen3-4B-FP4 \
    --method mr-gptq \
    --calibration bartowski-v3
```

### Run Inference

```python
from metal_marlin.inference import MetalInferenceEngine
from metal_marlin.safetensors_loader import load_model

model = load_model("./Qwen3-4B-FP4")
engine = MetalInferenceEngine(model)
output = engine.generate("The capital of France is", max_tokens=50)
print(output)
```

## Architecture

Metal Marlin uses **PyTorch MPS + native Metal shaders** (via PyObjC), not MLX.

**Why not MLX?** MLX's built-in quantization uses round-to-nearest with uniform affine levels, no calibration. At 3-bit, this produces 38% worse perplexity than GGUF's calibration-aware methods. Metal Marlin uses GPTQ/AWQ-class quantization with Hessian-informed rounding and per-layer mixed precision. See [Why Not MLX?](docs/why_not_mlx.md) for benchmarks.

**How Metal dispatch works:**

```
PyTorch MPS tensors → zero-copy MTLBuffer sharing → custom Metal shaders → results back to PyTorch
```

The `metal_dispatch.py` module compiles `.metal` shader sources at runtime via PyObjC, creates compute pipelines, and dispatches kernels directly. MPS tensors share their underlying Metal buffers with no copy, enabling tight integration between PyTorch's tensor operations and custom quantized GEMM kernels.

**Quantization formats supported:**

| Format | Bits | Use Case |
|--------|------|----------|
| FP4 (E2M1) | 4 | Primary format for weights |
| FP8 (E4M3, E5M2) | 8 | Higher quality, larger models |
| INT4 (U4, S4) | 4 | GPTQ-compatible weights |
| INT3, INT2 | 3, 2 | Extreme compression for cold MoE experts |
| NF4, NF3, NF2 | 4, 3, 2 | QLoRA normal-float formats |
| 2:4 Sparse | variable | Structured sparsity (1.6× compression) |

KV cache quantization (FP4/INT4) is also supported, fused into attention kernels.

## How It Works

Metal Marlin ports [Marlin](https://arxiv.org/abs/2408.11743) (fast quantized GEMM kernels for CUDA) to Apple Silicon. Weights are unpacked using ALU operations instead of lookup tables, eliminating memory bandwidth bottlenecks.

## Documentation

- **[Getting Started](docs/getting_started.md)** — Full walkthrough: install, quantize, run, verify
- [Calibration Guide](docs/calibration.md) — Custom calibration datasets for higher quality
- [Mixed Precision](docs/mixed_precision.md) — Per-layer precision for MoE models
- [Architecture](docs/architecture.md) — How Metal Marlin works internally
- [Why Not MLX?](docs/why_not_mlx.md) — Design decision: PyTorch MPS + native Metal
- [Troubleshooting](docs/troubleshooting.md) — Common issues and fixes

## References

- [Marlin](https://arxiv.org/abs/2408.11743) — Mixed-precision auto-regressive kernels (Frantar et al., 2024)
- [GPTQ](https://arxiv.org/abs/2210.17323) — Hessian-aware post-training quantization (Frantar et al., 2022)
- [AWQ](https://arxiv.org/abs/2306.00978) — Activation-aware weight quantization (Lin et al., 2023)
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — Activation smoothing for W8A8 (Xiao et al., 2022)
- [QuaRot](https://arxiv.org/abs/2404.00456) — Hadamard rotation for outlier-free quantization
- [FlashAttention](https://arxiv.org/abs/2205.14135) — IO-aware exact attention (Dao et al., 2022)
- [vLLM](https://arxiv.org/abs/2309.06180) — PagedAttention for KV cache (Kwon et al., 2023)
- [2:4 Sparsity](https://arxiv.org/abs/2104.08378) — Sparse Tensor Core design (Mishra et al., 2021)

## Citations

```bibtex
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L. and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}

@article{frantar2022gptq,
  title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}
}

@inproceedings{lin2024awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}

@inproceedings{xiao2023smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  booktitle={ICML},
  year={2023}
}

@article{ashkboos2024quarot,
  title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L. and Li, Bo and Jaggi, Martin and Alistarh, Dan and Hoefler, Torsten and Hensman, James},
  journal={arXiv preprint arXiv:2404.00456},
  year={2024}
}

@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{kwon2023vllm,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle={SOSP},
  year={2023}
}

@article{mishra2021sparse,
  title={Accelerating Sparse Deep Neural Networks},
  author={Mishra, Asit and Latorre, Jorge Albericio and Pool, Jeff and Stosic, Darko and Stosic, Dusan and Venkatesh, Ganesh and Yu, Chong and Micikevicius, Paulius},
  journal={arXiv preprint arXiv:2104.08378},
  year={2021}
}
```

## Status

See [STATUS.md](STATUS.md) for implementation progress and model compatibility.

## License

Apache 2.0
