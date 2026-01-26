# MoE BF16 Audit (Metal kernels)

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
- Router softmax is computed in FP32 across all router variants (good for stability). Outputs are stored as half.
- Most kernels operate on half activations and half outputs. There is no explicit BF16 storage or bfloat type usage; any “BF16” path is effectively FP16 in these kernels.
- The main BF16/bandwidth inefficiencies are repeated reloads of activation tiles per expert (shared/routed fusion), and avoidable float->half->half roundtrips in slow-path accumulation.

## Findings by file

### src/moe_router.metal
- Softmax precision: logits and softmax are computed in float (FP32) with max/sum reductions in float and exp() using float. This meets the “FP32 for stability” requirement.
- Output precision: top-k probabilities are converted to half before storing (expert_probs). If downstream uses half, this is expected; if BF16 is desired, consider a bfloat-compatible storage format or keep float until the final aggregation step.
- Conversion notes: dot product casts half inputs to float on every multiply. This is unavoidable for FP32 accumulation but is the main conversion cost in the router.

### src/moe_dispatch.metal
- Dispatch/gather: activations are loaded as half into A_tiles; expert weights are dequantized into half tiles. There is no extra conversion back-and-forth besides the required FP4 -> half dequantization.
- Accumulation: simdgroup_matrix<half> accumulates in half. If BF16 accumulation is intended, this is already a low-precision path. If higher precision is desired, a float accumulator would reduce error but increase bandwidth/compute.
- Potential inefficiency: representative-expert shortcut means the compute path can be repeated for tokens that do not actually use that expert; this is not a BF16 conversion issue but can amplify memory traffic relative to useful compute.

### src/moe_dispatch_optimized.metal
- Router softmax: top-k selection and softmax use float for logits/probs; outputs are stored as half in threadgroup memory. This is stable, but conversion to half happens immediately after softmax.
- Slow-path accumulation: in the per-token slow path, accumulation is done in float, then cast to half, then multiplied by prob (half). This introduces a float->half->half roundtrip per output element.
  - Impact: extra conversion and reduced effective precision; also increases bandwidth when writing to half staging for an intermediate value.
- Shared expert vs routed experts: the shared expert uses shared_weights/scales and dequantizes to half; routed experts repeat dequantization per expert. This is expected, but in bandwidth-bound regimes the repeated dequantization dominates.

### src/moe_expert_gemm.metal
- Input/output dtypes: activations are half; FP4 weights are dequantized to half; outputs are half. There is no explicit BF16 path.
- Grouped kernel: activations are gathered into A_tiles once per token batch, which is good. However, output accumulation uses half and writes directly to output with +=, so there is no float staging. This favors bandwidth but reduces accumulation precision.
- Non-grouped kernel: representative-expert approximation can result in unnecessary work for tokens that do not use the representative expert. This is a compute/memory inefficiency rather than a conversion issue.

### src/moe_shared_expert.metal
- Shared expert path (FP16 weights): shared expert output is computed and stored in threadgroup half, then routed experts are computed per token and added in half using safe_fma. This is a single-precision path (half only).
- Shared expert path (FP4 weights): dequantization occurs for each expert and each token, and activations are reloaded for each expert. This causes repeated reads of half activations per expert and per token.
  - Impact: memory bandwidth amplification and repeated half load -> float compute -> half store cycles across experts.
- Scatter kernel: per-token kernel computes dense dot products in float but stores half output. This is accurate for compute, but still ends in half.

## BF16-focused inefficiencies

1) Lack of explicit BF16 storage
- All kernels use half for activations and outputs. If the intended dtype is BF16, it is effectively handled as FP16 here. This can cause extra conversions outside these kernels or mismatches with higher-level BF16 pipelines.

2) Repeated activation reloads in shared/routed fusion
- In src/moe_shared_expert.metal, each routed expert recomputes the A tile by reloading hidden_states for the same token. For top-k=4 and multiple experts, this multiplies activation traffic and repeated half conversions.

3) Float->half->half roundtrip in slow paths
- In src/moe_dispatch_optimized.metal slow path, float accumulators are cast to half before multiplying by prob (half), then added to half staging. This adds a conversion and precision loss that can be avoided.

4) Half-only accumulation for expert GEMMs
- In src/moe_dispatch.metal and src/moe_expert_gemm.metal, simdgroup_matrix<half> accumulates in half. If the expected precision is BF16 (or higher), there is no higher-precision accumulation to amortize quantization noise from FP4 weights.

## Recommendations

1) Keep router softmax FP32, but avoid early half conversion when possible
- Maintain float probabilities in threadgroup memory until the aggregation step, then cast once to half for final output. This avoids repeated float->half->float cycles when probabilities are reused.

2) Eliminate float->half->half roundtrip in slow-path accumulation
- In src/moe_dispatch_optimized.metal, compute `val * prob` in float and cast once when adding to half staging. Alternatively, use float staging for the slow path only.

3) Cache per-token activation tiles for routed experts
- In src/moe_shared_expert.metal, load A_tiles once per token and reuse across experts in the inner loop. This reduces activation bandwidth and avoids repeated half loads for top-k experts.

4) Consider a BF16 storage strategy if BF16 is the intended dtype
- If the external pipeline uses BF16, add a bfloat-compatible storage or explicit conversion boundaries to avoid multiple implicit conversions between host BF16 and device half.

## Quick checklist (per requirement)
- Router softmax precision: FP32 in all router kernels (meets requirement).
- Token dispatch/gather conversions: mostly half->half; biggest issue is repeated activation reloads and slow-path float->half->half roundtrip.
- Expert GEMM input/output dtypes: half in/out everywhere; no explicit BF16 usage.
- Shared expert handling: shared expert and routed experts both accumulate in half; routed experts reload activations per expert, increasing bandwidth.
