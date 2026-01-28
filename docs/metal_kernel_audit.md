# Metal Kernel Audit

**Date**: January 28, 2026  
**Metal sources (src/)**: 37 `.metal` files  
**Kernel entry points (src/)**: 242 `kernel void` functions (counted)  
**Verification inputs**: `scripts/verify_kernels.py`, `contrib/metal_marlin/STATUS.md` ("Metal Shader Status"), and the current `src/` inventory

## Verified Kernel Status

### ✅ Compile + Run Correctly (verified)

| Kernel | Source | Evidence |
|--------|--------|----------|
| `marlin_gemm_fp4` | `src/marlin_gemm.metal` | Compiles, loads, and **works** (per status + verify script) |

### ⚠️ Compile but Not Fully Tested (compile+load verified)

`verify_kernels.py` only checks pipeline creation; runtime correctness is still pending for these kernels:

| Kernel | Source | Status |
|--------|--------|--------|
| `flash_attention_v2` | `src/flash_attention_v2.metal` | Compiles and loads |
| `dense_gemm` | `src/dense_gemm.metal` | Compiles and loads |
| `moe_dispatch_optimized` | `src/moe_dispatch_optimized.metal` | Compiles and loads |
| `simdgroup_attention` | `src/simdgroup_attention.metal` | Compiles and loads |

### 🟡 Stubs or Incomplete Wiring (Python-side)

| Function | Location | Status |
|----------|----------|--------|
| `flash_attention_fp4_kv` | `metal_marlin/kernels.py` | Stub — needs a fused Metal kernel |
| `moe_router_topk` | `metal_marlin/kernels.py` | PyTorch fallback (Metal kernel not wired) |

## Current Metal Source Inventory (src/)

Files listed below exist in `contrib/metal_marlin/src/`. Items marked **Verified** are covered by `verify_kernels.py`; everything else is present but not compile-verified by that script.

| File | Status |
|------|--------|
| `all_reduce.metal` | Present (not verified) |
| `attention.metal` | Present (not verified) |
| `batched_gemm.metal` | Present (not verified) |
| `bf16_compat.metal` | Present (not verified) |
| `decode_gemv.metal` | Present (not verified) |
| `dense_gemm.metal` | ✅ Verified compile+load (`dense_gemm`) |
| `dequant_fp8.metal` | Present (not verified) |
| `dequant_int8.metal` | Present (not verified) |
| `dequant_sub4bit.metal` | Present (not verified) |
| `dequant.metal` | Present (not verified) |
| `diff_attention.metal` | Present (not verified) |
| `flash_attention_v2.metal` | ✅ Verified compile+load (`flash_attention_v2`) |
| `flash_attention.metal` | Present (not verified) |
| `fusion/attention_residual.metal` | Present (not verified) |
| `fusion/gated_mlp.metal` | Present (not verified) |
| `fusion/norm_linear.metal` | Present (not verified) |
| `gemm_epilogue.metal` | Present (not verified) |
| `gemm_fp4_optimized.metal` | Present (not verified) |
| `hadamard.metal` | Present (not verified) |
| `kernels_autotune.metal` | Present (not verified) |
| `marlin_gemm.metal` | ✅ Verified compile+run (`marlin_gemm_fp4`) |
| `mla_proj.metal` | Present (not verified) |
| `moe_dispatch_optimized.metal` | ✅ Verified compile+load (`moe_dispatch_optimized`) |
| `moe_dispatch.metal` | Present (not verified) |
| `moe_expert_gemm.metal` | Present (not verified) |
| `moe_router.metal` | Present (not verified) |
| `moe_shared_expert.metal` | Present (not verified) |
| `paged_attention.metal` | Present (not verified) |
| `rope.metal` | Present (not verified) |
| `rwkv_wkv.metal` | Present (not verified) |
| `sampling.metal` | Present (not verified) |
| `simdgroup_attention.metal` | ✅ Verified compile+load (`simdgroup_attention`) |
| `sliding_window_attention.metal` | Present (not verified) |
| `sparse_gemm.metal` | Present (not verified) |
| `sparse.metal` | Present (not verified) |
| `tree_attention.metal` | Present (not verified) |
| `vision_preprocess.metal` | Present (not verified) |

## Notes

- `scripts/verify_kernels.py` currently validates only the five kernels listed above.
- If additional kernels become compile-verified, add them to `verify_kernels.py` and move them into the verified sections here.
