# Metal Marlin Documentation

Quantized GEMM kernels for Apple Silicon. Run large language models on your Mac at 70+ tok/s.

---

## 🚀 Start Here

| Guide | Description |
|-------|-------------|
| [**Getting Started**](guides/getting_started.md) | Install, quantize, run a model in 5 minutes |
| [**Trellis Inference**](inference/trellis.md) | Run GLM-4.7-Flash at 70+ tok/s |
| [**Serving Models**](guides/serving.md) | OpenAI-compatible API server |
| [**CLI Reference**](guides/cli.md) | Command-line tools and options |

---

## 📚 Documentation Sections

### [Inference](inference/)
How to run models with Metal Marlin.

- [Trellis Inference](inference/trellis.md) — GLM-4.7-Flash at 70+ tok/s
- [MMFP4 Inference](inference/mmfp4.md) — MMFP4-quantized models

### [Quantization](quantization/)
How to quantize models for efficient inference.

- [Quantization Guide](quantization/quantization.md) — FP4/Trellis quantization

### [User Guides](guides/index.md)
Step-by-step tutorials and workflows.

- [Getting Started](guides/getting_started.md) — Quick installation and first model
- [Building from Source](guides/building.md) — Compilation and C++ extension
- [Serving Models](guides/serving.md) — OpenAI-compatible API server
- [CLI Reference](guides/cli.md) — Command-line tools
- [Troubleshooting](guides/troubleshooting.md) — Fix common problems

### [Metal Kernel Internals](internals/index.md)
Low-level documentation for kernel developers.

- [Fast Dispatch (C++ Ext)](internals/fast_dispatch.md) — 5-10x faster kernel dispatch
- [CUDA to Metal Mapping](internals/cuda_metal_mapping.md) — Translating concepts
- [Tile Sizing](internals/tile_sizing.md) — Choosing dimensions
- [Memory Access Patterns](internals/memory_access_patterns.md) — Coalesced access
- [Metallib Architecture](internals/metallib_architecture.md) — Precompiled shaders

### [API Reference](reference/index.md)
Technical reference for APIs, models, and integrations.

- [Python API](reference/api.md) — Full API documentation
- [Supported Models](reference/supported_models.md) — Model compatibility matrix
- [Hardware Compatibility](reference/compatibility.md) — GPU and macOS version support
- [Integration Guide](reference/integration.md) — Embedding in your application

### [Core Concepts](concepts/index.md)
Understand the fundamental ideas behind Metal Marlin.

- [Architecture Overview](concepts/architecture.md) — System design
- [Inference Architecture](concepts/inference_architecture.md) — End-to-end inference flow
- [MoE Architecture](concepts/moe_architecture.md) — Mixture of Experts support
- [Prompt Sharing (COW)](concepts/cow_prompt_sharing.md) — Copy-on-Write prompt sharing
- [Vision & ViT Support](concepts/vision_1024_implementation.md) — High-res image preprocessing
- [Quantization & Dequantization](concepts/dequantization.md) — How weights work
- [Mixed Precision](concepts/mixed_precision.md) — Per-layer precision strategies
- [KV Cache](concepts/kv_cache.md) — Quantized key-value cache

### [Quantization Formats](formats/index.md)
Supported formats and data type configurations.

- [GGUF Support](formats/gguf_quantization.md) — GGUF format
- [MR-GPTQ](formats/mr_gptq.md) — Metal Marlin GPTQ
- [Data Type Configuration](formats/dtype_configuration.md) — Choosing optimal types

### [Advanced Features](features/index.md)
Optional features and extensions.

- [Balance Loss](features/balance_loss.md) — Auxiliary loss for MoE expert balancing

### [Metal Kernel Internals](internals/index.md)
Low-level documentation for kernel developers.

- [Fast Dispatch (C++ Ext)](internals/fast_dispatch.md) — 5-10x faster kernel dispatch
- [CUDA to Metal Mapping](internals/cuda_metal_mapping.md) — Translating concepts
- [Tile Sizing](internals/tile_sizing.md) — Choosing dimensions
- [Memory Access Patterns](internals/memory_access_patterns.md) — Coalesced access
- [Metallib Architecture](internals/metallib_architecture.md) — Precompiled shaders

### [Technical Audits](audits/index.md)
Investigation reports and bug analyses.

- [Implementation Summary](audits/speculative_decoding_implementation_summary.md) — Speculative decoding implementation details
- [Batch Scheduler Implementation](audits/batch_scheduler_implementation.md) — Dynamic request scheduling
- [Metal Kernel Audit](audits/metal_kernel_audit.md) — Kernel review
- [MMFP4 Debug Summary](audits/mmfp4_debug_summary.md) — MMFP4 inference stack debugging analysis
- [Resolved Bugs](audits/resolved_bugs.md) — Fixed issues
- [Metadata Refactor](audits/mla_proj_refactor.md) — MLA projection changes

### [Comparisons](comparisons/index.md)
How Metal Marlin compares to alternatives.

- [Why Not MLX?](comparisons/why_not_mlx.md) — PyTorch MPS vs MLX
- [vLLM Comparison](comparisons/vllm_comparison.md) — Feature comparison

### [Performance Reports](reports/index.md)
Empirical performance measurements and optimization outcomes.

- [Performance Analysis](reports/performance_analysis.md) — Dispatch, latency, and memory analysis
- [GLM-4.7 Throughput](reports/glm4_throughput.md) — GLM-4.7 throughput profile
- [GLM-4.7 Mixed BPW Optimization](reports/glm47_mixed_bpw_optimization.md) — Optimization deltas for mixed bit-width kernels

---

## 🧩 [Model Architectures](architectures/index.md)

Special architecture support:

- [MLA (Multi-head Latent Attention)](architectures/mla.md) — GLM-4.7-Flash attention
- [Byte-level Models](architectures/byte_models.md) — Byte tokenization
- [FlashAttention-3 Tiling](architectures/fa3.md) — FA3 tiling strategy and implementation notes

---

## 🔧 Contributing

If you are adding kernel support:

1. Identify the target operator and data types
2. Add or extend the Metal shader in `src/`
3. Wire it into the kernel registry
4. Verify numerical parity against a reference implementation
5. Add targeted benchmarks for the new kernel
6. Document any constraints (alignment, tile sizes, supported layouts)

**Testing requirements:**
- Run unit tests covering your kernel path
- Add a focused regression test for edge cases
- Run relevant performance or integration tests before submitting

---

## 📖 Quick Links

| Resource | Link |
|----------|------|
| GitHub Repository | [metal-marlin/metal-marlin](https://github.com/metal-marlin/metal-marlin) |
| Implementation Status | [STATUS.md](../STATUS.md) |
| Academic References | [References](comparisons/references.md) |
