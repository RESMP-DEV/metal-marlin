# GEMM BF16 Audit

Status (2026-01-28)
- BF16 input support exists in `marlin_gemm.metal` via `USE_BF16_INPUTS` (optional
  `USE_BF16_OUTPUTS`). The Python dispatch compiles `marlin_gemm_fp4_fp32acc` with
  BF16 inputs; output remains FP16 unless `USE_BF16_OUTPUTS` is also defined.
- `dense_gemm.metal` includes a BF16 kernel (`dense_gemm_small_batch_fp4_fp32acc`) with
  BF16 input/output and FP32 accumulation, but no in-repo Python dispatch uses it yet.
- `batched_gemm.metal`, `gemm_fp4_optimized.metal`, and `moe_expert_gemm.metal` remain
  FP16-only.

Scope: `contrib/metal_marlin/src/batched_gemm.metal`,
`contrib/metal_marlin/src/dense_gemm.metal`,
`contrib/metal_marlin/src/marlin_gemm.metal`,
`contrib/metal_marlin/src/gemm_fp4_optimized.metal`,
`contrib/metal_marlin/src/moe_expert_gemm.metal`.

## batched_gemm.metal
- No BF16 path; all kernels use `half` inputs/outputs.

## dense_gemm.metal
- BF16 kernel present:
  - `dense_gemm_small_batch_fp4_fp32acc` uses BF16 inputs/outputs (`ushort`) with
    FP32 accumulation.
- Other kernels remain FP16-only.

## marlin_gemm.metal
- BF16 input path gated by `USE_BF16_INPUTS` in:
  - `marlin_gemm_fp4_fp32acc`
  - `marlin_gemm_fused_fp4_fp32acc`
- Outputs stay FP16 unless `USE_BF16_OUTPUTS` is defined (not set by default in
  `metal_marlin/kernels.py`).

## gemm_fp4_optimized.metal
- No BF16 path.

## moe_expert_gemm.metal
- No BF16 path.

## Remaining gaps
- No BF16 support in `batched_gemm.metal`, `gemm_fp4_optimized.metal`, or
  `moe_expert_gemm.metal`.
- BF16 output variants for `marlin_gemm` are not enabled by default.
- No Python dispatch for `dense_gemm_small_batch_fp4_fp32acc` yet.

## Resolved issues
- The earlier “no GEMM kernels support BF16” finding is outdated.
  See `docs/resolved_bugs.md`.
