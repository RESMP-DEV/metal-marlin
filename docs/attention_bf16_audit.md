# BF16 Attention Kernel Audit

Status (2026-01-28)
- BF16 input/output paths exist in `flash_attention_v2.metal` and
  `simdgroup_attention.metal` when compiled with `USE_BF16_INPUTS`.
- `paged_attention.metal`, `sliding_window_attention.metal`, `tree_attention.metal`,
  and `mla_proj.metal` remain FP16-only.
- Softmax math is still FP32 across attention kernels (unchanged).

Scope: `contrib/metal_marlin/src/flash_attention_v2.metal`,
`contrib/metal_marlin/src/simdgroup_attention.metal`,
`contrib/metal_marlin/src/paged_attention.metal`,
`contrib/metal_marlin/src/sliding_window_attention.metal`,
`contrib/metal_marlin/src/tree_attention.metal`,
`contrib/metal_marlin/src/mla_proj.metal`.

## flash_attention_v2.metal
- Inputs/outputs:
  - With `USE_BF16_INPUTS`: `input_t = bf16_t`, `output_t = ushort` (BF16 storage).
  - Otherwise: `half` in/out.
- Softmax accumulation: `float` (FP32).
- BF16 conversions use `bf16_compat.metal` helpers and vectorized stores
  (`store_output_bf16_vectorized`).

## simdgroup_attention.metal
- Same BF16 gating via `USE_BF16_INPUTS` as `flash_attention_v2.metal`.
- Outputs are BF16 (`ushort`) only when BF16 is enabled at compile time.
- `simdgroup_attention_pv` still uses `simdgroup_matrix<half>` in the FP16 path.

## paged_attention.metal
- Inputs/outputs: `half` only; quantized variants dequantize to `half`.
- No BF16 path in this file.

## sliding_window_attention.metal
- Inputs/outputs: `half` only.
- No BF16 path in this file.

## tree_attention.metal
- Inputs/outputs: `half` only.
- No BF16 path in this file.

## mla_proj.metal
- Projection/GEMM kernels; inputs/outputs are `half`.
- No BF16 path in this file.

## BF16 status summary
- Avoiding FP16 round-trip: possible for `flash_attention_v2` and
  `simdgroup_attention` when compiled with `USE_BF16_INPUTS`.
- Still required for `paged_attention`, `sliding_window_attention`,
  `tree_attention`, and `mla_proj` (no BF16 path yet).

## Resolved issues
- The earlier “no BF16 attention kernels exist” finding is no longer true.
  See `docs/resolved_bugs.md`.
