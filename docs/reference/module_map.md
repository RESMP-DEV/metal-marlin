# Metal Marlin Module Map

## Core Modules (metal_marlin/)

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `metal_marlin/` | Main package with core inference components | `__init__.py`, `kernels.py`, `attention.py`, `quantized_linear.py` |
| `metal_marlin/quantization/` | Quantization algorithms and pipelines | `exl3_pipeline.py`, `viterbi_quant.py`, `hessian_streaming.py` |
| `metal_marlin/serving/` | OpenAI-compatible server and batching | `server.py`, `engine.py`, `continuous_batch.py` |
| `metal_marlin/calibration/` | Hessian analysis and calibration tools | `hooks.py`, `hessian_collector.py`, `sensitivity.py` |
| `metal_marlin/speculative/` | Speculative decoding engines | `engine.py`, `draft.py`, `eagle.py`, `verify.py` |
| `metal_marlin/asr/` | Automatic speech recognition (Conformer/Parakeet) | `parakeet_model.py`, `conformer_block.py`, `hybrid_parakeet.py` |
| `metal_marlin/moe/` | Mixture of Experts routing and execution | `moe_dispatch.py`, `token_dispatcher.py`, `adaptive_precision.py` |
| `metal_marlin/kernels/` | Metal kernel implementations | `mla_proj.metal` (Metal Shaders) |
| `metal_marlin/models/` | Model-specific implementations | `deepseek.py` |
| `metal_marlin/ane/` | Apple Neural Engine optimizations | `conv_ane.py`, `depthwise_conv_ane.py`, `buffer_pool.py` |
| `metal_marlin/vision/` | Vision model components | Various vision processing modules |
| `metal_marlin/distributed/` | Multi-GPU distribution | Distributed inference coordination |
| `metal_marlin/paged/` | Paged attention and memory management | KV cache paging and memory optimization |
| `metal_marlin/utils/` | Utility functions and helpers | Common utilities used across modules |
| `metal_marlin/cache/` | Caching mechanisms | Model weight and computation caching |
| `metal_marlin/analysis/` | Performance analysis tools | Profiling and analysis utilities |
| `metal_marlin/hybrid/` | Hybrid execution modes | CPU/GPU/ANE hybrid inference |
| `metal_marlin/inference/` | Core inference engines | Main inference execution logic |
| `metal_marlin/architectures/` | Model architecture definitions | Layer architectures and configurations |
| `metal_marlin/autotuning/` | Automatic performance tuning | Auto-tuning frameworks |
| `metal_marlin/ops/` | Low-level operations | Basic computational operations |
| `metal_marlin/packing/` | Data packing utilities | Efficient data layout management |
| `metal_marlin/profiling/` | Performance profiling | Runtime performance analysis |

## Trellis Inference (trellis_*.py)

| Module | Purpose |
|--------|---------|
| `trellis_model.py` | Main Trellis model class and orchestration |
| `trellis_attention.py` | MLA (Multi-head Latent Attention) implementation |
| `trellis_generate.py` | Generation loop and text decoding |
| `trellis_config.py` | Model configuration and hyperparameters |
| `trellis_dispatch.py` | Operation dispatching and execution routing |
| `trellis_kv_cache.py` | Key-value cache management for efficient inference |
| `trellis_layer.py` | Individual transformer layer implementation |
| `trellis_linear.py` | Quantized linear layers for Trellis |
| `trellis_lm.py` | Language model head and vocabulary projection |
| `trellis_loader.py` | Model loading and state management |
| `trellis_moe.py` | Trellis-specific MoE (Mixture of Experts) implementation |
| `trellis_packing.py` | Data packing and memory layout optimization |

## Benchmarks

| Category | Files | Purpose |
|----------|-------|---------|
| **Core Kernels** | `bench_gemm.py`, `bench_attention.py`, `benchmark_attention.py` | Metal kernel timing and performance analysis |
| **Model Performance** | `bench_glm47.py`, `bench_qwen3.py`, `eval_glm4_full.py` | End-to-end model inference benchmarks |
| **Memory Analysis** | `profile_memory_breakdown.py`, `bench_memory.py`, `profile_kv_cache.py` | Memory usage profiling and optimization |
| **MoE Benchmarks** | `bench_moe_kernel.py`, `bench_moe_gptq_hessian.py` | Mixture of Experts performance testing |
| **Trellis Specific** | `bench_trellis_generation.py`, `bench_trellis_performance.py` | Trellis model specialized benchmarks |
| **Backend Comparison** | `benchmark_backends.py`, `bench_bf16_conversion.py` | Cross-backend performance comparisons |
| **Throughput Tests** | `bench_throughput.py`, `bench_glm4_throughput.py` | Request throughput and latency analysis |
| **Quality Evaluation** | `bench_glm47_quality.py`, `quality_comparison.py` | Model accuracy and quality metrics |
| **Metal Specific** | `bench_metal_e2e.py`, `bench_fp4_metal.py` | Metal shader and kernel performance |
| **MLA Attention** | `bench_mla_attention.py`, `profile_attention.py` | Multi-head Latent Attention benchmarks |
| **ASR Benchmarks** | `benchmark_metal_asr.py`, `eval_glm4_trellis.py` | Speech recognition performance |
| **Profiling Tools** | `profile_dequant.py`, `profile_moe_dispatch.py` | Detailed operation profiling |
| **Framework Testing** | `framework.py`, `baseline_benchmark.py` | Benchmark framework and baseline tests |

## Data Formats and Loaders

| Module | Purpose |
|--------|---------|
| `hf_exl3_loader.py` | HuggingFace EXL3 format model loading |
| `gguf_loader.py` | GGUF format model loader |
| `onnx_graph.py` | ONNX model graph processing |
| `distributed_gptq.py` | Distributed GPTQ quantization support |

## Quantization Formats

| Format | Implementation |
|--------|----------------|
| **GPTQ** | `gptq_metal.py`, `quantize_fp4.py` |
| **EXL3** | `exl3_pipeline.py`, `exl3_quantizer.py` |
| **LDLQ** | `ldlq.py`, `ldl_decomp.py` |
| **Trellis** | `trellis_quant.py`, `trellis_tile.py`, `trellis_codebook.py` |

## Specialized Components

| Component | Purpose |
|-----------|---------|
| `eval_perplexity.py` | Model quality evaluation via perplexity |
| `capacity.py` | Resource capacity planning and estimation |
| `capacity.py` | Resource capacity planning and estimation |
| `_buffer_pool.py` | Internal buffer management |
| `layernorm_metal.py` | Metal-optimized layer normalization |

## Statistics

- **Total Python Files**: 229
- **Main Directories**: 24
- **Core Areas**: Quantization, Serving, Benchmarks, Trellis, ASR, MoE
- **Hardware Support**: Metal (GPU), ANE, Multi-GPU Distribution