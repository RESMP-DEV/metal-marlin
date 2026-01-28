# Metal Kernel Audit

**Date**: June 2025  
**Total .metal files**: 37  
**Total kernel functions**: 228  
**Python-wired kernels**: ~45 (estimated)

## Summary

The metal_marlin project has substantial Metal kernel infrastructure that is partially wired to Python. This audit identifies:
1. What's built (all 228 kernel functions)
2. What's wired (callable from Python)
3. What's missing (gaps to fill)

## Wiring Status by Category

### ✅ Fully Wired (Python bindings exist and work)

| Category | File | Kernels | Python Module |
|----------|------|---------|---------------|
| GEMM | `marlin_gemm.metal` | `marlin_gemm_fp4`, `marlin_gemm_int4`, `marlin_gemm_fused_*` | `kernels.py` |
| Attention | `flash_attention_v2.metal` | `flash_attention_v2_*` (6 variants) | `flash_attention_v2.py` |
| Attention | `sliding_window_attention.metal` | `sliding_window_attention_*` | `attention.py` |
| Dequant | `dequant.metal` | `dequant_fp4_*`, `dequant_int4_*` | `kernels.py` |
| Hadamard | Embedded in `kernels.py` | `hadamard_transform_*` | `kernels.py` |
| RoPE | `rope.metal` | `rope_*` variants | `rope.py` |

### ⚠️ Partially Wired (bindings exist but use slow paths)

| Category | File | Issue | Impact |
|----------|------|-------|--------|
| **MoE GEMM** | `moe_expert_gemm.metal` | Python loop per expert instead of batched kernel | 64x slower on GLM-4.7-Flash |
| **MoE Dispatch** | `moe_dispatch_optimized.metal` | Not used—Python-side token grouping instead | Suboptimal memory access |
| **MoE Router** | `moe_router.metal` | Basic `moe_router_topk` only, advanced variants unused | Missing aux loss, batched variants |

### ❌ Not Wired (Metal kernels exist, no Python bindings)

| File | Kernels | Priority | Notes |
|------|---------|----------|-------|
| `paged_attention.metal` | `paged_attention_v1_*`, `paged_attention_v2_*` | P1 | Critical for vLLM-style serving |
| `tree_attention.metal` | `tree_attention_forward_*` | P2 | Speculative decoding |
| `diff_attention.metal` | `diff_attention_*` | P2 | Research model support |
| `rwkv_wkv.metal` | `rwkv_wkv_*`, `rwkv_*` | P3 | RWKV model support |
| `mla_proj.metal` | `rope_mla_*` | P2 | DeepSeek-style MLA |
| `sparse_gemm.metal` | `marlin_gemm_sparse_*` | P1 | N:M sparsity |
| `all_reduce.metal` | (if exists) | P3 | Multi-GPU |
| `vision_preprocess.metal` | `image_*`, `vit_*` | P2 | Vision-language models |

---

## Detailed Kernel Inventory (228 functions)

### GEMM Kernels (30+ variants)

```
marlin_gemm.metal (2963 lines):
  - marlin_gemm_fp4
  - marlin_gemm_fp4_fp32acc
  - marlin_gemm_fp4_single_stage
  - marlin_gemm_fp4_3stage
  - marlin_gemm_fp4_async
  - marlin_gemm_fp4_async_deep
  - marlin_gemm_fp4_striped
  - marlin_gemm_fused_fp4
  - marlin_gemm_fused_fp4_epilogue
  - marlin_gemm_fused_fp4_epilogue_fc
  - marlin_gemm_fused_fp4_fp32acc
  - marlin_gemm_fused_u4
  - marlin_gemm_divergent_fp4
  - marlin_gemm_fp16_pipelined
  - marlin_gemm_fp16_single_stage
  - marlin_zero_reduction
  
batched_gemm.metal:
  - marlin_gemm_batched_fp4
  - marlin_gemm_grouped_attention
  
dense_gemm.metal:
  - dense_gemm
  - dense_gemm_small_batch_fp4
  - dense_gemm_small_batch_fp4_fp32acc
  - dense_gemm_prefill_splitk_fp4
  - dense_splitk_reduce
  - dense_fused_gate_up_fp4
  - dense_batched_decode_gemv_fp4
  
decode_gemv.metal:
  - decode_gemv_fp4
  - decode_gemv_fp4_wide
  - decode_gemv_fp4_tiled
  - decode_gemv_fp4_simd
  - decode_gemv_fp4_batched
  - dense_decode_gemv_fp4
  - dense_decode_gemv_fp4_tiled
  
sparse_gemm.metal:
  - marlin_gemm_sparse_fp4
  - marlin_gemm_sparse_fp4_fused
  - marlin_gemm_sparse_nm
  
gemm_fp4_optimized.metal:
  - gemm_fp4_optimized
  - gemm_fp4_optimized_decode
  - gemm_fp4_optimized_fp32acc
  - gemm_fp4_optimized_large_m
  
gemm_epilogue.metal:
  - epilogue_standalone
```

### Attention Kernels (25+ variants)

```
flash_attention_v2.metal (1856 lines):
  ✅ flash_attention_v2
  ✅ flash_attention_v2_causal
  ✅ flash_attention_v2_decode
  ✅ flash_attention_v2_gqa
  ✅ flash_attention_v2_mqa
  ✅ flash_attention_v2_fp8_kv
  ✅ flash_attention_v2_fp8_kv_causal
  
flash_attention.metal:
  - flash_attention
  - flash_attention_causal
  - flash_attention_gqa
  - flash_attention_kv_fp4
  - flash_attention_kv_int4
  
sliding_window_attention.metal:
  ✅ sliding_window_attention_causal
  ✅ sliding_window_attention_decode
  ✅ sliding_window_attention_gqa
  ✅ sliding_window_attention_prefill
  
attention.metal:
  - attention_qk_softmax
  - attention_qk_softmax_tiled
  - attention_pv
  - attention_fused_qkv
  
simdgroup_attention.metal:
  - simdgroup_attention
  - simdgroup_attention_pv
  - simdgroup_attention_qk
  
paged_attention.metal:
  ❌ paged_attention_v1
  ❌ paged_attention_v1_fp4
  ❌ paged_attention_v1_fp8
  ❌ paged_attention_v1_int4
  ❌ paged_attention_v2
  ❌ paged_attention_v2_fp8
  ❌ paged_attention_v2_reduce
  
tree_attention.metal:
  ❌ tree_attention_forward
  ❌ tree_attention_forward_packed_mask
  ❌ tree_attention_forward_with_prefix_causal
  ❌ build_tree_mask
  
diff_attention.metal:
  ❌ diff_attention
  ❌ diff_attention_causal
  ❌ diff_attention_gqa
```

### MoE Kernels (20+ variants) ⚠️ PRIORITY

```
moe_router.metal (1049 lines):
  ⚠️ moe_router_topk (basic version wired)
  ❌ moe_router_fused
  ❌ moe_router_fused_small
  ❌ moe_router_batched
  ❌ moe_router_with_aux_loss
  ❌ test_moe_router_single
  ❌ test_moe_softmax
  ❌ test_moe_topk
  
moe_expert_gemm.metal (735 lines):
  ⚠️ moe_expert_gemm_fp4 (Python loop, not batched)
  ❌ moe_expert_gemm_fp4_grouped
  ❌ moe_expert_gemm_shared_fp4
  
moe_dispatch.metal (764 lines):
  ❌ moe_dispatch_decode
  ❌ moe_dispatch_decode_bf16
  ❌ moe_dispatch_grouped
  ❌ moe_dispatch_fused
  
moe_dispatch_optimized.metal (1951 lines):
  ❌ moe_dispatch_optimized
  ❌ moe_dispatch_optimized_prerouted
  ❌ moe_dispatch_optimized_prerouted_bf16
  ❌ moe_dispatch_ultra_optimized_bf16
  ❌ moe_compute_offsets
  ❌ moe_compute_grouping
  ❌ moe_scatter_indices
  ❌ moe_aggregate_weighted
  
moe_shared_expert.metal (901 lines):
  ❌ moe_shared_expert_fused
  ❌ moe_shared_expert_fused_fp4
  ❌ moe_shared_expert_scatter
```

### Dequantization Kernels (35+ variants)

```
dequant.metal:
  ✅ dequant_fp4_kernel, dequant_fp4_aligned_kernel
  ✅ dequant_int4_kernel, dequant_int4_aligned_kernel  
  ✅ dequant_fp4_bulk, dequant_int4_bulk
  ✅ dequant_fp4_optimal, dequant_fp4_optimal_rowmajor
  - Various test kernels
  
dequant_fp8.metal:
  ✅ dequant_fp8_kernel (via fp8_utils.py)
  ✅ dequant_fp8_aligned_kernel
  ✅ dequant_fp8_e5m2_kernel
  - Test kernels
  
dequant_int8.metal:
  - dequant_int8_kernel
  - dequant_int8_aligned_kernel
  - Test kernels
  
dequant_sub4bit.metal:
  ❌ dequant_int2_bulk
  ❌ dequant_int3_bulk
  ❌ dequant_nf3_bulk
  - Test kernels
```

### Position Encoding (15+ variants)

```
rope.metal:
  ✅ rope_forward
  ✅ rope_inplace
  ✅ rope_generate_cache
  ✅ rope_qk_fused
  ✅ rope_small_dim
  ✅ rope_yarn_* (all YaRN variants)
  ❌ rope_mla_latent (DeepSeek MLA)
  ❌ rope_mla_latent_small
  ❌ rope_mla_split_fused
```

### Sampling Kernels

```
sampling.metal:
  ✅ sample_categorical
  ✅ sample_top_k
  ✅ sample_top_p
  ✅ argmax
  ✅ apply_temperature
  ✅ apply_repetition_penalty
  ✅ softmax
```

### Vision Preprocessing (12+ variants)

```
vision_preprocess.metal:
  ❌ image_normalize, image_normalize_f16
  ❌ image_resize_bilinear, image_resize_bicubic
  ❌ image_resize_normalize_fused
  ❌ center_crop, channel_swap_rgb_bgr
  ❌ uint8_to_float
  ❌ vit_patch_extract
  ❌ dynamic_resize_qwen2vl
```

### Hadamard Transform

```
hadamard.metal:
  ✅ hadamard_transform_32
  ✅ hadamard_transform_64
  ✅ hadamard_transform_128
  ✅ hadamard_inverse_64
  - Test kernels
```

### BF16 Compatibility

```
bf16_compat.metal:
  ✅ bf16_to_half_kernel
  ✅ half_to_bf16_kernel
  ✅ bf16_to_float_kernel
  ✅ float_to_bf16_kernel
```

### RWKV Architecture

```
rwkv_wkv.metal:
  ❌ rwkv_wkv_single_token
  ❌ rwkv_wkv_batched
  ❌ rwkv_token_shift
  ❌ rwkv_apply_time_decay
  ❌ rwkv_squared_relu
```

---

## Priority Action Items

### P0: GLM-4.7-Flash MoE Pipeline (Phase 42)

1. **Wire batched MoE GEMM**
   - File: `moe_expert_gemm.metal`
   - Target: `moe_expert_gemm_fp4_grouped` kernel
   - Impact: 64x speedup over Python loop

2. **Wire optimized dispatch**
   - File: `moe_dispatch_optimized.metal`
   - Target: `moe_dispatch_optimized_prerouted` kernel
   - Impact: Fused grouping + scatter

3. **Wire shared expert**
   - File: `moe_shared_expert.metal`
   - Target: `moe_shared_expert_fused_fp4`
   - Impact: GLM-4.7-Flash has 1 shared expert

### P1: Serving Infrastructure

1. **Paged Attention**
   - File: `paged_attention.metal`
   - 7 kernel variants (v1, v2, quantized)
   - Critical for vLLM-style KV cache management

2. **Sparse GEMM**
   - File: `sparse_gemm.metal`
   - N:M sparsity patterns
   - ~2x memory reduction

### P2: Model Coverage

1. **DeepSeek MLA** - `rope_mla_*` kernels
2. **Tree Attention** - speculative decoding
3. **Diff Attention** - Differential Transformer
4. **Vision Preprocessing** - VLM support

### P3: Research

1. **RWKV** - Linear attention alternative
2. **Sub-4bit** - INT2/INT3/NF3 quantization

---

## Testing Status

| Category | Unit Tests | Integration Tests |
|----------|------------|-------------------|
| GEMM FP4/INT4 | ✅ | ✅ |
| Flash Attention | ✅ | ✅ |
| Sliding Window | ✅ | Partial |
| RoPE/YaRN | ✅ | ✅ |
| Hadamard | ✅ | ✅ |
| MoE | ❌ | ❌ |
| Paged Attention | ❌ | ❌ |
| Vision | ❌ | ❌ |

---

## Architecture Notes

### Loading Patterns

1. **Embedded strings** (kernels.py): GEMM, dequant basics
2. **Direct file load**: flash_attention_v2.py, sliding_window, autotune
3. **get_shader_source()**: Generic loader from src/ directory

### Key Classes

- `MetalKernelLibrary`: Compiles and caches .metal sources
- `dispatch_kernel()`: PyObjC Metal dispatch wrapper
- `mps_tensor_to_metal_buffer()`: PyTorch MPS ↔ Metal interop

### Performance Characteristics

| Kernel Type | Typical Speedup vs PyTorch |
|-------------|---------------------------|
| Fused GEMM (FP4) | 3-5x |
| Flash Attention | 2-4x |
| Fused decode GEMV | 5-10x |
| MoE (batched, theoretical) | 10-64x vs Python loop |
