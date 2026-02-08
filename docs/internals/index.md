# Metal Kernel Internals

Low-level documentation for Metal shader development and optimization.

## CUDA to Metal

- [**CUDA to Metal Mapping**](cuda_metal_mapping.md) — Translating CUDA concepts to Metal
- [**Porting Guide**](porting_guide.md) — How to port new kernels

## Kernel Design

- [**Dequant Algorithm**](dequant_algorithm.md) — Dequantization kernel design
- [**Fused Dequant GEMM**](fused_dequant_gemm.md) — Fused dequantization + GEMM
- [**Batched GEMM**](batched_gemm.md) — Batched matrix multiplication
- [**GEMM Trellis Refactor**](gemm_trellis_moe_refactor.md) — `gemm_trellis_moe` structure and helper mapping
- [**Expert Batching**](expert_batching.md) — MoE token grouping optimization
- [**Trellis Kernels**](trellis_kernels.md) — Trellis 3-bit quantization kernels

## Implementation Details

- [**AWQ Implementation**](awq_implementation.md) — Technical AWQ details
- [**Async Dispatch**](async_dispatch.md) — GPU command execution pipeline
- [**Metal ASR**](metal_asr.md) — Conformer/Parakeet ASR backend
- [**Fast Router Dispatcher**](fast_router_dispatcher.md) — CPU-side MoE routing acceleration design
- [**Metallib Architecture**](metallib_architecture.md) — Precompiled shader build/runtime architecture

## Optimization

- [**Kernel Optimization**](kernel_optimization.md) — Applied kernel optimization techniques
- [**MoE Optimization**](moe_optimization.md) — MoE architecture and dispatch strategies
- [**Compressed KV Cache (MLA)**](compressed_kv_cache_mla.md) — Memory and latency optimizations for GLM-4.7
- [**Enhanced Expert Memory Pool**](enhanced_expert_memory_pool.md) — Pooling/defrag strategy for MoE experts
- [**Kernel Selection**](kernel_selection.md) — Core MoE kernel selection strategy
- [**Mixed Kernel Selection**](mixed_kernel_selection.md) — Mixed precision selector heuristics
- [**Mixed Kernel Selection (Implementation)**](kernel_selection_mixed_implementation.md) — Implementation details for mixed-BPW routing
- [**KV Cache Consolidation**](kv_cache_consolidation.md) — Consolidated KV cache module design
- [**Non-Power-of-2 Hadamard**](hadamard_npow2_optimization.md) — Hadamard optimization for unsupported sizes
- [**Large-Vocab Sampling Optimization**](sampling_large_vocab_optimization.md) — Sampling kernels for vocab >100K
- [**Tile Sizing**](tile_sizing.md) — Choosing optimal tile dimensions
- [**Occupancy Tuning**](occupancy_tuning.md) — Maximizing GPU utilization
- [**Memory Access Patterns**](memory_access_patterns.md) — Coalesced memory access
- [**Memory Optimization**](memory_optimization.md) — Unified memory management
- [**Stripe Partitioning**](stripe_partitioning.md) — Work distribution strategies
- [**Pipeline Stages**](pipeline_stages.md) — Pipelining and latency hiding
- [**Kernel Profiling**](kernel_profiling.md) — Performance profiling strategy

## Metal-Specific

- [**SIMD Group Async Copy**](simdgroup_async_copy.md) — Async memory operations
- [**Metal 4 Tensor Ops**](metal4_tensor_ops.md) — Metal 4 tensor operations (M4+)
