# Resolved Bugs and Retired Audits

This file consolidates issues that were previously tracked in *_audit.md documents and
are now resolved or superseded. It serves as a historical ledger so the live audits
can focus on remaining gaps.

## 2026-01-28
- BF16 conversion helpers in `src/bf16_compat.metal` are now vectorized (e.g.
  `bf16x4_to_float4_direct`, `float4_to_bf16x4_rne_direct`) and are actively used by
  BF16-capable kernels. This supersedes the earlier standalone BF16 conversion audit.
- BF16 attention paths exist in `src/flash_attention_v2.metal` and
  `src/simdgroup_attention.metal` when compiled with `USE_BF16_INPUTS`.
- BF16 GEMM support was added for:
  - `src/marlin_gemm.metal` (`marlin_gemm_fp4_fp32acc` with `USE_BF16_INPUTS`)
  - `src/dense_gemm.metal` (`dense_gemm_small_batch_fp4_fp32acc`)
- BF16 MoE dispatch kernels were added in `src/moe_dispatch_optimized.metal`
  (`*_bf16` variants).
- `metal_dispatch.dispatch_gemm_fp4` now pads M/N/K (and packed weights/scales)
  when padding is enabled, addressing the earlier “no N/K padding” limitation for
  that dispatch path.
- GEMM parameter buffers in `metal_marlin/kernels.py` now use private buffers via
  managed staging (instead of shared), reducing shared-memory pressure in those
  paths.
