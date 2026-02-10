# Metal Kernel Duplication Report

## Overview
This report identifies duplicate and near-duplicate functions, kernels, and utility logic across the Metal shader codebase. Significant duplication exists in dequantization logic, MoE routing/dispatch, and evolutionary versions of attention kernels.

---

## 1. FP4 E2M1 Dequantization Logic
The most prevalent duplication is the implementation of FP4 E2M1 dequantization.

| Shader File | Functions / Constants | Recommendation |
|-------------|-----------------------|----------------|
| `contrib/metal_marlin/metal_marlin/shaders/mmfp4_dequant.metal` | `mmfp4_e2m1_value` | Standardize to a single helper in a header. |
| `contrib/metal_marlin/metal_marlin/shaders/mmfp4_fused_moe.metal` | `FP4_LUT`, `dequant_fp4x8_vec` | Standardize. |
| `contrib/metal_marlin/metal_marlin/shaders/mmfp4_gemm.metal` | `dequant_fp4_scalar`, `dequant_fp4x8` | Standardize. |
| `contrib/metal_marlin/src/dequant.metal` | `FP4_LUT_CONST`, `dequant_fp4_scalar`, `dequant_fp4_lut_x8` | Standardize. |
| `contrib/metal_marlin/src/gemm_fp4_optimized.metal` | `FP4_LUT`, `dequant_fp4_lut`, `dequant_fp4x8` | Standardize. |
| `contrib/metal_marlin/src/moe_dispatch_optimized.metal` | `FP4_LUT`, `opt_dequant_fp4x8_lut` | Standardize. |
| `contrib/metal_marlin/src/flash_attention.metal` | `dequant_fp4_scalar` | Standardize. |
| `contrib/metal_marlin/src/attention_mla_fused.metal` | `dequant_fp4_scalar` | Standardize. |
| `contrib/metal_marlin/src/decode_gemv.metal` | `dequant_fp4_scalar` | Standardize. |
| ... and ~15 other files | `dequant_fp4_scalar` | Standardize. |

**Observation**: There are three main implementation styles: LUT-based (fastest on some hardware), `switch`-based, and branchless ALU-based. These should be unified into a single header with an `#ifdef` or template for the preferred method.

---

## 2. MoE Dispatch and Routing
Multiple versions of MoE kernels exist, often with overlapping functionality.

| Shader File | Function / Kernel | Relationship | Recommendation |
|-------------|-------------------|--------------|----------------|
| `moe_dispatch.metal` | `moe_dispatch_fused` | Base version | Keep as legacy/reference. |
| `moe_dispatch_optimized.metal` | `moe_dispatch_optimized` | GLM-4.7 specialized | Keep, but document specialization. |
| `moe_dispatch_metal.metal` | `moe_compute_expert_counts`, etc. | Multi-kernel version | Keep for large batch fallback. |
| `moe_router.metal` | `moe_router_fused` | Base router | Consolidate with `moe_router_fused.metal`. |
| `moe_router_fused.metal` | `moe_router_argsort_fused` | Enhanced router | Merge into a single `moe_router.metal`. |
| `moe_router_sparse.metal` | `moe_router_sparse` | Learned candidate version | Keep specialized. |

---

## 3. Flash Attention Evolutionary Versions
Versioned files contain near-identical code with incremental improvements.

| Shader File | Relationship | Recommendation |
|-------------|--------------|----------------|
| `flash_attention.metal` | Version 1 | Keep as baseline. |
| `flash_attention_v2.metal` | Version 2 | Consider deprecating in favor of V3. |
| `flash_attention_v3.metal` | Version 3 (Adds causal optimization) | Primary version. |
| `paged_attention.metal` | Contains `v1` and `v2` kernels | Consolidate common logic into helpers. |

---

## 4. Common Utility Redundancy
Basic mathematical and SIMD utilities are redefined in almost every file.

| Utility Function | Impacted Files | Recommendation |
|------------------|----------------|----------------|
| `simd_sum`, `simd_max` | `moe_router.metal`, `flash_attention.metal`, `layernorm.metal`, etc. | Use `reduction_helpers.metal` exclusively. |
| `div_ceil` | `moe_router.metal`, `moe_router_int8.metal`, `moe_router_sparse.metal`, etc. | Move to a common `utils.metal` header. |
| `safe_exp` | `moe_router.metal`, `moe_router_int8.metal`, `moe_router_sparse.metal` | Move to a common `utils.metal` header. |
| `silu` / `fast_silu` | `mmfp4_fused_moe.metal`, `gemm_trellis_swiglu.metal`, etc. | Move to a common `activations.metal` header. |

---

## 5. Backup and Orphaned Files
Several files appear to be temporary backups or disabled tests left in the source tree.

| File Path | Status | Recommendation |
|-----------|--------|----------------|
| `contrib/metal_marlin/src/gemm_trellis_moe.metal.bak` | Backup | Delete. |
| `contrib/metal_marlin/src/moe_scatter_gather_optimized.metal.bak` | Backup | Delete. |
| `contrib/metal_marlin/src/test_async_copy.metal.disabled` | Disabled | Move to a `tests/` or `benchmarks/` directory. |

---

## 6. GEMM Tiling Constants
Constants like `TILE_M`, `TILE_N`, `TILE_K` are hardcoded in many GEMM variants.

| Shader File | Constants | Recommendation |
|-------------|-----------|----------------|
| `gemm_trellis.metal` | 128x128x32 | Use preprocessor defines or specialization constants. |
| `gemm_trellis_int8.metal` | 128x128x32 | Standardize via a shared configuration header. |
| `gemm_fp4_optimized.metal` | Varied | Standardize. |

---

## Conclusion
The `metal_marlin` codebase would benefit significantly from a "Common Header" refactor. By moving dequantization primitives, SIMD reductions, and activation functions into shared `.metal` headers, the total lines of code could be reduced by approximately 15-20%, improving maintainability and reducing the surface area for bugs.
