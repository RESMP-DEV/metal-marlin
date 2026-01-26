# GEMM BF16 audit

Scope: `contrib/metal_marlin/src/batched_gemm.metal`, `contrib/metal_marlin/src/dense_gemm.metal`, `contrib/metal_marlin/src/marlin_gemm.metal`, `contrib/metal_marlin/src/gemm_fp4_optimized.metal`, `contrib/metal_marlin/src/moe_expert_gemm.metal`.

Summary:
- No kernels in these files use BF16 types or explicit BF16 conversion helpers. All inputs/outputs are `half` or `float`, with FP4 weights dequantized to `half`.
- `simdgroup_multiply_accumulate` is used in all simdgroup-based GEMMs; non-simdgroup kernels use scalar `float` accumulation.

## contrib/metal_marlin/src/batched_gemm.metal

| Kernel | Input activation dtype | Output activation dtype | Accumulator dtype | BF16 conversion calls | simdgroup MMA used |
| --- | --- | --- | --- | --- | --- |
| `marlin_gemm_batched_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_grouped_attention` | half (Q/K) | half (scores) | half (`simdgroup_matrix<half>`) | none | yes |

Notes: Weights are packed FP4 with `half` scales; dequantization writes `half` into threadgroup tiles.

## contrib/metal_marlin/src/dense_gemm.metal

| Kernel | Input activation dtype | Output activation dtype | Accumulator dtype | BF16 conversion calls | simdgroup MMA used |
| --- | --- | --- | --- | --- | --- |
| `dense_decode_gemv_fp4` | half | half | float (scalar `float acc[4]`) | none | no |
| `dense_decode_gemv_fp4_tiled` | half | half | float (scalar `float acc[4]`) | none | no |
| `dense_gemm_small_batch_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `dense_gemm_prefill_splitk_fp4` | half | float (`partial_out`) | float (`simdgroup_matrix<float>`) | none | yes |
| `dense_splitk_reduce` | n/a (reads FP32 partials) | half | float (scalar reduction) | none | no |
| `dense_fused_gate_up_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `dense_batched_decode_gemv_fp4` | half | half | float (scalar `float acc[4]`) | none | no |

Notes: Split-K uses FP32 partials but still loads activations and dequantized weights as `half`.

## contrib/metal_marlin/src/marlin_gemm.metal

| Kernel | Input activation dtype | Output activation dtype | Accumulator dtype | BF16 conversion calls | simdgroup MMA used |
| --- | --- | --- | --- | --- | --- |
| `marlin_gemm_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fp4_fp32acc` | half | half | float (`simdgroup_matrix<float>`) | none | yes |
| `marlin_gemm_fp4_3stage` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_divergent_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fp4_single_stage` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fp16_single_stage` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fp4_striped` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_zero_reduction` | n/a (utility) | half (reduction buffer) | half (scalar) | none | no |
| `marlin_gemm_fused_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fused_fp4_fp32acc` | half | half | float (`simdgroup_matrix<float>`) | none | yes |
| `marlin_gemm_fused_u4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fp16_pipelined` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fp4_async` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `marlin_gemm_fp4_async_deep` | half | half | half (`simdgroup_matrix<half>`) | none | yes |

Notes: FP32 accumulator variants still load A/B as `half` and dequantize FP4 into `half`.

## contrib/metal_marlin/src/gemm_fp4_optimized.metal

| Kernel | Input activation dtype | Output activation dtype | Accumulator dtype | BF16 conversion calls | simdgroup MMA used |
| --- | --- | --- | --- | --- | --- |
| `gemm_fp4_optimized` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `gemm_fp4_optimized_fp32acc` | half | half | float (`simdgroup_matrix<float>`) | none | yes |
| `gemm_fp4_optimized_large_m` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `gemm_fp4_optimized_decode` | half | half | half (`simdgroup_matrix<half>`) | none | yes |

Notes: FP32 accumulator variant still uses `half` A tiles and dequantized `half` B fragments.

## contrib/metal_marlin/src/moe_expert_gemm.metal

| Kernel | Input activation dtype | Output activation dtype | Accumulator dtype | BF16 conversion calls | simdgroup MMA used |
| --- | --- | --- | --- | --- | --- |
| `moe_expert_gemm_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `moe_expert_gemm_fp4_grouped` | half | half | half (`simdgroup_matrix<half>`) | none | yes |
| `moe_expert_gemm_shared_fp4` | half | half | half (`simdgroup_matrix<half>`) | none | yes |

Notes: Expert probabilities are `half` and applied after accumulation (still in `half`).

## FP32-only accumulation candidates (BF16 -> FP32 -> GEMM -> FP32 -> BF16)

Kernels that already use scalar FP32 accumulation (no simdgroup MMA) are the lowest-effort candidates to switch to BF16 inputs and FP32 compute without touching `half` math:
- `dense_decode_gemv_fp4`
- `dense_decode_gemv_fp4_tiled`
- `dense_batched_decode_gemv_fp4`
- `dense_splitk_reduce` (already FP32 accumulation; output conversion remains)

Kernels that already use FP32 accumulators with simdgroup MMA but still load `half` tiles could be upgraded to FP32-only if A/B tiles and dequantized values are promoted to `float` and `simdgroup_matrix<float>` fragments are used:
- `dense_gemm_prefill_splitk_fp4`
- `marlin_gemm_fp4_fp32acc`
- `marlin_gemm_fused_fp4_fp32acc`
- `gemm_fp4_optimized_fp32acc`

All other kernels are `half`-accumulating simdgroup MMA paths and would require a larger rewrite (float tiles + float fragments) to avoid FP16 entirely.
