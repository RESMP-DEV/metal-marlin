# MoE BF16 Audit (Metal kernels)

Status (2026-01-28)
- BF16 kernels exist in `src/moe_dispatch_optimized.metal` (`*_bf16` variants).
- `src/moe_dispatch.metal`, `src/moe_expert_gemm.metal`, `src/moe_router.metal`, and
  `src/moe_shared_expert.metal` remain FP16-only.
- Router softmax is still FP32 (unchanged).

Scope:
- src/moe_dispatch.metal
- src/moe_dispatch_optimized.metal
- src/moe_expert_gemm.metal
- src/moe_router.metal
- src/moe_shared_expert.metal

Focus:
1) Router softmax precision
2) Token dispatch/gather conversions
3) Expert GEMM input/output dtypes
4) Shared expert handling vs routed experts

## Executive summary
- Router softmax stays FP32 across router variants; outputs are stored as half.
- BF16 storage and conversion helpers are now used in the optimized dispatch path
  (`moe_dispatch_optimized.metal`), but other MoE kernels still operate on `half`.
- Bandwidth inefficiencies (activation reloads per expert, slow-path staging) still
  exist in FP16 paths and were not fully re-audited for the BF16 variants.

## Findings by file

### src/moe_router.metal
- Softmax precision: logits and softmax computed in float (FP32).
- Output precision: top-k probabilities stored as half.
- No BF16 path in this file.

### src/moe_dispatch.metal
- Activations and outputs are `half`; FP4 weights dequantize to `half`.
- Accumulation uses `simdgroup_matrix<half>` (half precision).
- No BF16 path in this file.

### src/moe_dispatch_optimized.metal
- BF16 kernels present:
  - `moe_dispatch_ultra_optimized_bf16`
  - `moe_dispatch_optimized_prerouted_bf16`
  - `moe_dispatch_decode_bf16`
- BF16 activations are stored as `ushort` and converted via `bf16_compat.metal`.
- Compute still unpacks BF16 tiles to `half` for the main FP16 matmul path; slow
  paths may still stage through half buffers before final BF16 writes.

### src/moe_expert_gemm.metal
- Inputs/outputs are `half`; no BF16 path.
- Accumulation is `simdgroup_matrix<half>` (half precision).

### src/moe_shared_expert.metal
- Shared and routed experts operate on `half`.
- Activation reloads per expert remain (bandwidth amplification).
- No BF16 path in this file.

## BF16-focused inefficiencies (current)
1) BF16 coverage is limited to optimized dispatch kernels
   - Other MoE kernels still treat BF16 inputs as FP16.
2) Activation reloads in shared/routed fusion
   - Still present in `moe_shared_expert.metal` (FP16 path).
3) Slow-path staging still uses half buffers
   - BF16 kernels convert to half for compute and may stage through half buffers.

## Recommendations (still applicable)
1) Keep router softmax FP32, but avoid early half conversion when possible.
2) Reduce half-staging in slow paths (compute `val * prob` in float and cast once).
3) Cache per-token activation tiles across routed experts in shared expert kernels.
4) Extend BF16 storage strategy beyond the optimized dispatch path if BF16 is the
   intended dtype for MoE in the full pipeline.

## Resolved issues
- “No BF16 kernels exist in MoE dispatch” is no longer true.
  See `docs/resolved_bugs.md`.
