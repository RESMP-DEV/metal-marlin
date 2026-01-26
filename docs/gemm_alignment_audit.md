# GEMM alignment audit (M/N/K divisibility by 8)

## Why this matters
Metal MPS matmul performance drops when M, N, or K are not divisible by 8. Most kernels here use simdgroup 8x8 tiles and FP4 packing (8 values per uint32), so alignment affects both throughput and row-stride efficiency.

## Common model sizes (M, N, K)
M is runtime-dependent (batch * sequence). K and N follow layer shapes.

### GLM-4.7 (hidden=3584, intermediate=9216, vocab=128000)
- QKV projection: M = batch*seq, K = 3584, N = 3584 (or 3*3584 if fused QKV)
- Attention output: M = batch*seq, K = 3584, N = 3584
- MLP up/gate: M = batch*seq, K = 3584, N = 9216
- MLP down: M = batch*seq, K = 9216, N = 3584
- LM head: M = batch*seq, K = 3584, N = 128000

### Qwen2 (hidden=4096, intermediate=11008, vocab=152064)
- QKV projection: M = batch*seq, K = 4096, N = 4096 (or 3*4096 if fused QKV)
- Attention output: M = batch*seq, K = 4096, N = 4096
- MLP up/gate: M = batch*seq, K = 4096, N = 11008
- MLP down: M = batch*seq, K = 11008, N = 4096
- LM head: M = batch*seq, K = 4096, N = 152064

### Llama (hidden=4096, intermediate=11008, vocab=128256)
- QKV projection: M = batch*seq, K = 4096, N = 4096 (or 3*4096 if fused QKV)
- Attention output: M = batch*seq, K = 4096, N = 4096
- MLP up/gate: M = batch*seq, K = 4096, N = 11008
- MLP down: M = batch*seq, K = 11008, N = 4096
- LM head: M = batch*seq, K = 4096, N = 128256

All of the above K and N values are divisible by 8. M often is not (especially decode M=1).

## Python dispatch + packing checks

### `contrib/metal_marlin/metal_marlin/kernels.py`
- `marlin_gemm_fp4`: M = product of leading dims, K = last dim, N = B_packed.shape[1]. Uses `dispatch_gemm_fp4`. No N/K padding. Now pads M to a multiple of 8 via `pad_torch_2d` before dispatch, then slices output back to original M.
- `marlin_gemm_int4`: Same M/N/K inference; now pads M to multiple of 8 before dispatch.

### `contrib/metal_marlin/metal_marlin/hf_loader.py`
- `should_quantize_tensor` requires `in_feat % 8 == 0` and `in_feat % group_size == 0`. It rejects non-aligned weights instead of padding.
- For common model sizes above, in_feat is hidden or intermediate and already divisible by 8, so the packing path remains aligned.

### `contrib/metal_marlin/metal_marlin/mr_gptq.py`
- `_pack_fp4_weights` requires `in_feat % 8 == 0` (no padding).
- `MRGPTQQuantizer.quantize_layer` enforces `in_feat % 8 == 0` and `in_feat % group_size == 0` before optional Hadamard padding. Hadamard can pad to `block_size`, but the pre-check still forbids non-8-aligned `in_feat`.

## Metal GEMM shader alignment notes (all in `contrib/metal_marlin/src/`)

All GEMM kernels use 8x8 simdgroup tiles and FP4 packing with 8 values per uint32. Where K/N are not divisible by 8, the kernels rely on bounds checks and zero-fill, but still assume the packed layout is padded.

### `marlin_gemm.metal`
- Layout: A [M,K], B [K/8,N], scales [K/group_size, N], C [M,N].
- M/N/K passed as constants; row stride is K for A and N for C.
- Bounds checks for edges, but no explicit padding for M/N/K.

### `gemm_fp4_optimized.metal`
- Same layout as `marlin_gemm.metal`, plus optional epilogue.
- Uses `k_packs = ceil(K/8)`; requires B/scales to be padded for non-8 K.
- No explicit padding for M/N/K.

### `gemm_epilogue.metal`
- Epilogue uses the same tile geometry and A/B/C layout as `marlin_gemm.metal`.
- No explicit padding; expects aligned row strides for best performance.

### `kernels_autotune.metal`
- Macro-generated variants with tile sizes {32/64/128} on M/N and K in {16/32/64}.
- Same A/B/scales/C layout and FP4 packing assumptions.

### `dense_gemm.metal`
- Dense decode (M=1) and small-batch kernels: A [M,K], B [K/8,N], scales [K/group_size,N].
- Prefill split-K kernels use the same layout with reduction buffers.
- Uses bounds checks; no padding for M/N/K. Decode M=1 is unaligned without padding.

### `decode_gemv.metal`
- Specialized GEMV path for M=1. Same B packing (K/8, N) and scale layout.
- No explicit padding; row stride still N.

### `batched_gemm.metal`
- Strided batched GEMM with `A_batch_stride`, `B_batch_stride`, `C_batch_stride` (in elements).
- Per-batch A [M,K], B [K/8,N], scales [K/group_size,N].
- No explicit padding for M/N/K; stride values should favor 8-aligned row pitches.

### `sparse_gemm.metal` / `sparse.metal`
- Sparse 2:4/4:8 variants use same dense tile shapes and packed B layout.
- Bounds checks for edges; no explicit padding.

### `moe_expert_gemm.metal`
- Params: batch_size (M), hidden_dim (K), out_dim (N).
- Expert weights layout uses `hidden_dim/8` packs; assumes K divisible by 8.
- No padding for M or N.

### `moe_router.metal`
- Router GEMM: [batch, hidden] @ [hidden, num_experts].
- Uses dense half/float accumulators; no explicit padding.

### `moe_dispatch.metal` / `moe_dispatch_optimized.metal` / `moe_shared_expert.metal`
- Fused routing + GEMM paths use `hidden_dim/8` packed weights.
- Same alignment assumptions as `moe_expert_gemm.metal`.

### `fusion/norm_linear.metal`
- Fused RMSNorm + quantized GEMM. Uses packed weights with `K/8` and `K/group_size`.
- No explicit padding; uses bounds checks in K loop.

### `fusion/attention_residual.metal`
- Fused residual add with GEMM output. GEMM layout matches `marlin_gemm.metal`.
- No explicit padding; relies on bounds checks.

### `mla_proj.metal`
- GEMM-style projection with tile sizes matching core GEMM.
- Uses packed B layout with K/8, scales K/group_size.
- No explicit padding for M/N/K.

## Alignment gaps and padding status
- K/N alignment: Weight packing in `hf_loader.py` and `mr_gptq.py` rejects non-8-aligned `in_feat`; no padding is applied there.
- M alignment: Previously unpadded in Python dispatch. Added M padding to multiple of 8 in `kernels.py` and `metal_dispatch.py` to avoid slow edge tiles when M is not divisible by 8.
- Row stride: C uses stride N; for optimal performance N should be padded to multiple of 8. No runtime padding is implemented for N/K yet.

## Recommended follow-ups
- If you want to support non-8-aligned N/K, pad packed weights/scales at load time (and store original N for slicing).
- Expose a shared padding helper for weight packing paths (hf_loader/mr_gptq) instead of rejecting non-8-aligned tensors.
- Consider padding M to 64 for large M to reduce partial threadgroups further, with a small-switch threshold.
