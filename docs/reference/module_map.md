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
| `metal_marlin/moe/` | Active MoE grouping helpers and public re-exports | `__init__.py`, `gpu_grouping.py` (core dispatch lives in `metal_marlin/moe_dispatch.py`) |
| `metal_marlin/kernels/` | Split kernel export helpers | `attention.py`, `moe.py`, `../kernels_core.py` |
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

| Category | Representative Files | Purpose |
|----------|----------------------|---------|
| **Core Kernels** | `bench_attention.py`, `bench_fp4_metal.py`, `bench_kernel_variants.py` | Metal kernel timing and performance analysis |
| **Model Performance** | `bench_comprehensive_e2e.py`, `bench_glm47_canonical.py`, `glm_flash_benchmark.py` | End-to-end model inference benchmarks |
| **Memory Analysis** | `bench_memory.py`, `bench_kv_memory_bandwidth.py`, `bench_unified_memory.py` | Memory usage profiling and optimization |
| **MoE Benchmarks** | `bench_moe_fusion.py`, `bench_moe_kernel.py`, `bench_moe_decode_glm_qwen.py` | Mixture of Experts performance testing |
| **Trellis Specific** | `bench_mla_attention.py`, `bench_mla_fused_speedup.py`, `bench_e2e_decode.py` | Trellis / MLA specific benchmarking |
| **Throughput & Repro** | `bench_throughput.py`, `bench_standardized_decode.py`, `reproduce_benchmark.py` | Throughput baselines and reproducibility checks |
| **Archived Experiments** | `benchmarks/_archive/` | Historical scripts kept out of the active benchmark surface |

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

## Notes

- This map highlights major module groupings and representative files rather than serving as an exhaustive inventory.
- Benchmark scripts under `benchmarks/_archive/` are historical and not part of the current active surface.
- The current core areas remain quantization, serving, benchmarks, trellis, ASR, and MoE on Apple Silicon / Metal.
