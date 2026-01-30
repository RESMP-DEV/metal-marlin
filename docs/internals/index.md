# Metal Kernel Internals

Low-level documentation for Metal shader development and optimization.

## CUDA to Metal

- [**CUDA to Metal Mapping**](cuda_metal_mapping.md) — Translating CUDA concepts to Metal
- [**Porting Guide**](porting_guide.md) — How to port new kernels

## Kernel Design

- [**Dequant Algorithm**](dequant_algorithm.md) — Dequantization kernel design
- [**Fused Dequant GEMM**](fused_dequant_gemm.md) — Fused dequantization + GEMM
- [**Batched GEMM**](batched_gemm.md) — Batched matrix multiplication

## Optimization

- [**Tile Sizing**](tile_sizing.md) — Choosing optimal tile dimensions
- [**Occupancy Tuning**](occupancy_tuning.md) — Maximizing GPU utilization
- [**Memory Access Patterns**](memory_access_patterns.md) — Coalesced memory access
- [**Stripe Partitioning**](stripe_partitioning.md) — Work distribution strategies
- [**Pipeline Stages**](pipeline_stages.md) — Pipelining and latency hiding

## Metal-Specific

- [**SIMD Group Async Copy**](simdgroup_async_copy.md) — Async memory operations
- [**Metal 4 Tensor Ops**](metal4_tensor_ops.md) — Metal 4 tensor operations (M4+)
