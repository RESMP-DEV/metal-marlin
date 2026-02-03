# BF16 Optimization Guide

This document describes the BF16 optimization path used by Metal Marlin, why it exists, and how to select the right kernel variants and buffer handling strategy.

## Architecture overview: old vs new path

**Old path (conversion-heavy):**

```
BF16 storage (uint16)
  -> FP32 (expand)
  -> FP16 (pack for simdgroup_matrix)
  -> simdgroup_multiply_accumulate
  -> FP16 (store)
  -> FP32 (expand)
  -> BF16 (truncate)
```

This path incurred multiple conversions per element and forced BF16 data to round-trip through FP16 for simdgroup operations.

**New path (zero-copy BF16 with FP32 accumulation):**

```
BF16 storage (uint16)
  -> FP32 (expand once)
  -> FP32 accumulate in simdgroup_matrix<float>
  -> BF16 (truncate once on store)
```

Key changes:
- BF16 values remain in their native 16-bit storage format until the final conversion boundary.
- FP32 accumulation removes the FP16 round-trip and stabilizes long-K reductions.
- The conversion helpers in `contrib/metal_marlin/src/bf16_compat.metal` are the boundary between storage and compute.

## When to use BF16

BF16 is preferred when you need FP32-like dynamic range without the FP32 memory cost.

Recommended:
- **Apple Silicon M3+** (native bfloat support and better BF16 codegen).
- **Router/attention-sensitive layers** where overflow risk is higher.
- **Large-K GEMMs** where accumulator precision is a bottleneck.

Use FP16 instead when:
- Running on **M1/M2** where BF16 support is limited or slower.
- You need maximum compatibility with older Metal drivers.

See `contrib/metal_marlin/docs/dtype_configuration.md` for the configuration interface.

## Kernel variant selection logic

Selection is based on shape and precision requirements. The variants exist in `contrib/metal_marlin/src/` and are selected by higher-level dispatchers.

Recommended selection:
- **Default GEMM:** `marlin_gemm_fp4` for most FP4 weight GEMMs.
- **Precision-critical GEMM:** `marlin_gemm_fp4_fp32acc` or `marlin_gemm_fused_fp4_fp32acc` when K is large or BF16 stability is required.
- **Decode GEMV:** `dense_decode_gemv_fp4` for M=1 decode, where N-parallelism dominates.
- **MoE routing:** `moe_dispatch_optimized` when batch and expert counts are large enough to amortize routing overhead.

Notes:
- FP32 accumulator variants reduce numerical error and can be faster for BF16 inputs because they avoid BF16 -> FP16 -> FP32 conversions.
- Fused kernels reduce memory traffic by avoiding intermediate B tiles.

## Zero-copy buffer handling

Metal Marlin avoids host/device copies by binding MPS tensors directly to Metal buffers.

Key details:
- `MetalKernelLibrary._get_metal_buffer()` uses `newBufferWithBytesNoCopy` to wrap the MPS tensor memory.
- Buffers are allocated in shared storage mode to allow GPU access without an explicit copy.
- Inputs must be contiguous; non-contiguous tensors are made contiguous before binding.
- Small constant parameters are still passed via `newBufferWithBytes`, which copies a tiny scalar payload into a shared buffer.

This zero-copy path is critical for BF16 performance because it prevents extra BF16 -> FP16 staging buffers from dominating memory bandwidth.
