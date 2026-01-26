# Why Metal Marlin Instead of MLX?

MLX is Apple's official ML framework for Apple Silicon. So why build something else?

## The Case for 4-bit

4-bit is not an arbitrary choice. [Dettmers & Zettlemoyer (ICML 2023)](https://arxiv.org/abs/2212.09720) established **k-bit inference scaling laws** showing that 4-bit precision maximizes zero-shot performance per bit across model sizes. The key finding: a 4-bit 60B model outperforms an 8-bit 30B model despite having the same memory footprint.

With proper quantization methods (GPTQ, AWQ, MR-GPTQ), 4-bit inference recovers 96-98% of BF16 quality while:
- Reducing memory footprint 4×
- Reducing memory bandwidth requirements 4× (directly improving throughput)
- Enabling larger models to fit in memory

Running BF16 when 4-bit is available means using 4× the memory for <4% quality difference. For memory-constrained hardware, this tradeoff is straightforward.

## Direct Metal Control

Metal Marlin is a production inference engine that writes Metal shaders directly rather than going through a framework abstraction. This provides:

**Model-specific optimization:** Kernels tuned for specific architectures (MoE expert dispatch, MLA attention, grouped-query attention) ship when the model ships, not when a framework adds support. New model architectures get optimized kernels within days.

**Hardware-specific tuning:** Different M-series chips have different optimal tile sizes, occupancy targets, and memory access patterns. Metal Marlin ships chip-specific shader variants (M1 vs M2 vs M3 vs M4) that extract maximum throughput from each generation.

**Format flexibility:** Supporting a new quantization format requires only shader changes. No framework approval process, no API design committee, no backwards compatibility constraints.

MLX necessarily serves a broad audience and must maintain API stability. Metal Marlin serves one purpose: fastest possible inference on Apple Silicon with the best available quantization methods.

## Features MLX Doesn't Have

**Quantized KV Cache:** For long-context inference (4K+ tokens), the KV cache dominates memory. Metal Marlin supports FP4 and INT4 KV cache with per-row scales, reducing cache memory by 3.8× (4GB → 1GB at 4K context, 32 layers). Dequantization is fused into flash attention kernels with no intermediate materialization.

**2:4 Structured Sparsity:** For models pruned during training, Metal Marlin stores only 2 values per 4-element block with compact metadata. Achieves 1.6× weight compression. The metadata decode is interleaved with value dequantization to hide latency behind memory loads.

**Architecture-Specific Attention:** Separate implementations for standard multi-head attention, grouped-query attention (GQA), multi-latent attention (MLA), and differential attention. Each kernel is tuned for the specific access pattern, not a generic fallback.

## The Measured Quality Gap

We benchmarked MLX's native quantization against GGUF methods on Qwen3-30B-A3B (a 30B-parameter MoE model):

| Method | Bits | File Size | Perplexity | vs Baseline |
|--------|------|-----------|------------|-------------|
| IQ3_M (GGUF) | 3.60 bpw | 14.07 GB | 7.69 ± 0.22 | -1.9% (better) |
| Q3_K_M (GGUF) | 3.60 bpw | 14.07 GB | 7.84 ± 0.23 | baseline |
| **MLX 3-bit affine** | 3.51 bpw | 12.5 GB | **10.80 ± 0.31** | **+37.8% (worse)** |

*Lower perplexity is better. Measured on wikitext-2, 50 chunks, context 512.*

The 0.09 bpw difference (3.51 vs 3.60) cannot explain a 38% quality gap. The difference is in the quantization algorithm.

## Technical Differences in Quantization Methods

MLX uses **round-to-nearest (RTN) with uniform affine levels**:
- Every weight matrix gets the same bit allocation
- Quantization levels are uniformly spaced
- No calibration data informs the quantization

State-of-the-art methods (GPTQ, K-quant, IQ-quant) use:

| Technique | MLX | GGUF K/IQ | Metal Marlin |
|-----------|-----|-----------|--------------|
| Mixed precision per layer | No | Yes | Yes |
| Calibration-aware | No | Yes | Yes (MR-GPTQ) |
| Non-uniform quantization | No | Yes | Yes |
| Hessian-informed rounding | No | Partial | Yes |
| Sensitive layer detection | No | Yes | Yes |

The difference is especially pronounced at low bit-widths (2-4 bit) where naive rounding discards more information.

## NVFP4 Format Specification Difference

MLX labels its 4-bit format as "nvfp4" (NVIDIA FP4). The implementation differs from NVIDIA's specification, documented in [ml-explore/mlx#2962](https://github.com/ml-explore/mlx/issues/2962):

**NVIDIA's NVFP4 specification:**
- Two-level scaling: per-block E4M3 scale + per-tensor FP32 scale
- The FP32 tensor scale normalizes block maxima to fit E4M3 range
- Effective dynamic range: ~61,440× from the combined scaling

**MLX's "nvfp4" implementation:**
- Single-level scaling: per-block E4M3 scale only
- No tensor-level normalization
- Effective dynamic range: ~448× (the E4M3 maximum alone)

This is a 137× difference in representable dynamic range. The MLX maintainers are aware of this difference and have indicated the FP32 scale-of-scales may be added in the future.

The interoperability implication: models quantized with MLX's "nvfp4" use a different format than models produced by NVIDIA's TensorRT Model Optimizer or other spec-compliant tools. Loading one format with tooling expecting the other will produce incorrect dequantization.

## References

- [The case for 4-bit precision](https://arxiv.org/abs/2212.09720) — Dettmers & Zettlemoyer, ICML 2023. k-bit inference scaling laws
- [GPTQ](https://arxiv.org/abs/2210.17323) — Frantar et al. Hessian-aware post-training quantization
- [AWQ](https://arxiv.org/abs/2306.00978) — Lin et al. Activation-aware weight quantization
- [QuaRot](https://arxiv.org/abs/2404.00456) — Hadamard rotation for outlier-free 4-bit inference
- [ml-explore/mlx#2962](https://github.com/ml-explore/mlx/issues/2962) — NVFP4 format specification discussion
- [NVIDIA NVFP4 specification](https://developer.nvidia.com/blog/nvidia-blackwell-tensor-core-fp4-performance-and-quantization-techniques/) — Two-level scaling design
- Benchmark data from testing on M4 Max, January 2026
