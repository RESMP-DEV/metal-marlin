# BF16 Attention Kernel Audit

Scope: `contrib/metal_marlin/src/flash_attention_v2.metal`, `contrib/metal_marlin/src/simdgroup_attention.metal`, `contrib/metal_marlin/src/paged_attention.metal`, `contrib/metal_marlin/src/sliding_window_attention.metal`, `contrib/metal_marlin/src/tree_attention.metal`, `contrib/metal_marlin/src/mla_proj.metal`.

## flash_attention_v2.metal

Kernels:
- `flash_attention_v2`
- `flash_attention_v2_causal`
- `flash_attention_v2_decode`
- `flash_attention_v2_gqa`
- `flash_attention_v2_mqa`
- `flash_attention_v2_fp8_kv`
- `flash_attention_v2_fp8_kv_causal`

Dtypes and paths:
- Q/K/V input dtypes + load path:
  - Non-FP8 variants: `device const half* Q/K/V`, loaded into `threadgroup half` tiles and then cast to `float` for math (e.g., `q_reg[r][i] = float(Q_tile[q_row][d]);`).
  - FP8 KV variants: `device const uint8_t* K_fp8/V_fp8` + `device const half* K_scales/V_scales`, dequantized to `half` in threadgroup memory via `dequant_fp8_e4m3`.
- Softmax accumulator dtype:
  - `float` everywhere (scores, `m_prev`, `l_prev`, `o_acc`). This is FP32 as required for stability.
- Output dtype + conversion path:
  - `device half* O`, stored as `half(o_acc * inv_l)`.
- Explicit BF16 calls:
  - None. No `bfloat`/`bf16` types or conversion intrinsics in this file.

## simdgroup_attention.metal

Kernels:
- `simdgroup_attention_qk`
- `simdgroup_attention`
- `simdgroup_attention_pv`

Dtypes and paths:
- Q/K/V input dtypes + load path:
  - Q/K/V are `device const half*`. Threadgroup tiles are `half`, then cast to `float` for dot products.
  - `simdgroup_attention_pv` uses `P` as `device const half*` (softmax weights) and `V` as `half`.
- Softmax accumulator dtype:
  - `simdgroup_attention` uses `float` for scores and softmax stats (`m0/m1`, `l0/l1`, `o*_acc`).
  - `simdgroup_attention_qk` and `simdgroup_attention_pv` are not softmax kernels; they only compute scores or apply precomputed weights.
- Output dtype + conversion path:
  - `simdgroup_attention_qk` writes `S` as `half` (score computed in float).
  - `simdgroup_attention` writes `O` as `half` from `float` accumulators.
  - `simdgroup_attention_pv` writes `O` as `half` (accumulators are `simdgroup_matrix<half>` with half precision math).
- Explicit BF16 calls:
  - None.

## paged_attention.metal

Kernels:
- `paged_attention_v1`
- `paged_attention_v2`
- `paged_attention_v2_reduce`
- `paged_attention_v1_fp4`
- `paged_attention_v1_int4`
- `paged_attention_v1_fp8`
- `paged_attention_v2_fp8`

Dtypes and paths:
- Q/K/V input dtypes + load path:
  - Base kernels: Q/K/V are `device const half*`, loaded into `threadgroup half` and cast to `float` for math.
  - FP4/INT4 kernels: packed `uint` KV + `half` scales, dequantized to `half` in threadgroup memory.
  - FP8 kernels: `uint8_t` KV + `half` scales, dequantized to `half` in threadgroup memory (`dequant_fp8_e4m3`).
- Softmax accumulator dtype:
  - `float` for `scores`, `m_prev`, `l_prev`, `o_acc` in all attention kernels (v1/v2 and quantized variants).
  - `paged_attention_v2_reduce` uses `float` throughout (partial buffers are `float`).
- Output dtype + conversion path:
  - v1/v2/quantized kernels write `output` as `half` from float accumulators.
  - `paged_attention_v2_reduce` writes `device half* output` from float `o_final`.
- Explicit BF16 calls:
  - None.

## sliding_window_attention.metal

Kernels:
- `sliding_window_attention_prefill`
- `sliding_window_attention_causal`
- `sliding_window_attention_decode`
- `sliding_window_attention_gqa`

Dtypes and paths:
- Q/K/V input dtypes + load path:
  - Q/K/V are `device const half*`, loaded into `threadgroup half` and cast to `float` for dot products.
- Softmax accumulator dtype:
  - `float` for scores, `m_prev`, `l_prev`, and `o_acc` in all variants.
- Output dtype + conversion path:
  - `device half* O`, stored as `half(o_acc * inv_l)`.
- Explicit BF16 calls:
  - None.

## tree_attention.metal

Kernels:
- `tree_attention_forward`
- `tree_attention_forward_with_prefix_causal`
- `tree_attention_forward_packed_mask`
- `build_tree_mask` (mask construction helper)

Dtypes and paths:
- Q/K/V input dtypes + load path:
  - Q/K/V are `device const half*`, loaded into `threadgroup half` tiles. Dot products are computed in `float`.
- Softmax accumulator dtype:
  - `float` for scores, `running_max`, `running_sum`, and `o_accum` in all attention kernels.
  - `build_tree_mask` is a mask kernel (no softmax).
- Output dtype + conversion path:
  - `device half* O`, stored as `half(o_accum * inv_sum)`.
- Explicit BF16 calls:
  - None.

## mla_proj.metal

Kernels (selection):
- `mla_proj_fp4_k16`, `mla_proj_fp4_k32` (plus other MLA projection variants)

Dtypes and paths:
- Q/K/V input dtypes + load path:
  - This file implements projection/GEMM kernels, not attention. Inputs are `half` activations and FP4-packed weights; there is no Q/K/V attention path here.
- Softmax accumulator dtype:
  - Not applicable (no softmax in these kernels).
- Output dtype + conversion path:
  - Outputs are written as `half` (either direct `simdgroup_store` or via threadgroup staging).
- Explicit BF16 calls:
  - None.

## BF16 Optimization Assessment

- Current kernels uniformly use `half` for Q/K/V inputs and `half` for outputs, with `float` accumulators for softmax.
- There are no BF16 types, buffers, or conversion intrinsics in these kernels.
- If BF16 inputs are provided upstream, they must be converted to FP16 before launching these kernels (an implicit FP16 round-trip).

Answer to: "Can we avoid the FP16 round-trip entirely for BF16 inputs?"
- As implemented: no. These kernels are hardwired to `half` input/output buffers and `threadgroup half` tiles.
- To avoid the FP16 round-trip, you would need a BF16-specific path, e.g.:
  - Use `device const bfloat*` (or equivalent MSL `bfloat` type if available on target OS/GPU) for Q/K/V buffers.
  - Convert `bfloat` to `float` directly in registers for dot products and softmax (keeping FP32 accumulators).
  - Store output to BF16 (`bfloat`) buffers or keep `half` output if downstream expects FP16.
  - Replace `threadgroup half` tiles with `threadgroup bfloat` or load directly into `float` registers where possible.
- Any such change needs Metal language support and ABI alignment for BF16 buffers; none of that exists in these files today.
