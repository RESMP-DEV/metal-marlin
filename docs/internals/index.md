# Metal Kernel Internals

Low-level documentation for Metal shader development and optimization.

## CUDA to Metal

- [**CUDA to Metal Mapping**](cuda_metal_mapping.md) — Translating CUDA concepts to Metal
- [**Porting Guide**](porting_guide.md) — How to port new kernels

## Kernel Design

- [**Dequant Algorithm**](dequant_algorithm.md) — Dequantization kernel design
- [**Fused Dequant GEMM**](fused_dequant_gemm.md) — Fused dequantization + GEMM
- [**Batched GEMM**](batched_gemm.md) — Batched matrix multiplication
- [**Expert Batching**](expert_batching.md) — MoE token grouping optimization
- [**Trellis Kernels**](trellis_kernels.md) — Trellis 3-bit quantization kernels

## Implementation Details

- [**AWQ Implementation**](awq_implementation.md) — Technical AWQ details
- [**Async Dispatch**](async_dispatch.md) — GPU command execution pipeline
- [**Metal ASR**](metal_asr.md) — Conformer/Parakeet ASR backend

## Optimization

- [**Kernel Optimization**](kernel_optimization.md) — Applied kernel optimization techniques
- [**MoE Optimization**](moe_optimization.md) — MoE architecture and dispatch strategies
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
