# Metal Marlin Status

**Last Updated:** 2026-01-26T02:15

## Summary

| Component | Status |
|-----------|--------|
| Metal Shaders | **34/34** ✅ |
| MLX Removal | **Complete** ✅ |
| Weight Quantization | Tested (GLM-4.7-Flash) |
| Quantized Inference | **Not implemented** |

---

## Recent: GLM-4.7-Flash Weight Quantization

Quantized `zai-org/GLM-4.7-Flash` weights (not end-to-end inference):

| Metric | Value |
|--------|-------|
| Compression | 9.38x |
| Weight RMSE | 0.003 |
| Format | 99.5% FP4, 0.5% BF16 (gates) |

**Note:** These are weight-level metrics only. End-to-end inference not yet implemented.

---

## Blockers

### 1. ~~Failing Shader: marlin_gemm.metal~~ FIXED ✅

All 34 Metal shaders now compile successfully.

---

### 2. Quantized Inference Not Implemented

Current limitation: Benchmark computes quality metrics at **weight level** (RMSE 0.003 = excellent), but cannot run end-to-end inference with FP4 weights because:

1. PyTorch cannot natively load our FP4 tensors
2. Dequantization + GEMM must be fused in Metal kernels
3. Model forward pass needs integration with quantized weight loader

**Next steps:** See `quantized_inference_tasks.yaml` for implementation plan.

---

## Working Shaders (31/32)

All compiling:
- **Attention:** attention, flash_attention, flash_attention_v2, simdgroup_attention, diff_attention, paged_attention
- **GEMM:** dense_gemm, batched_gemm, gemm_fp4_optimized, sparse_gemm, moe_expert_gemm
- **Dequant:** dequant (FP4/INT4), dequant_fp8, dequant_int8, dequant_sub4bit
- **MoE:** moe_dispatch, moe_dispatch_optimized, moe_router, moe_shared_expert
- **Other:** rope, sampling, hadamard, all_reduce, sparse, rwkv_wkv, mla_proj, bf16_compat, decode_gemv, vision_preprocess, gemm_epilogue, kernels_autotune

---

## Quantization Formats

| Format | Bits | Status | Use Case |
|--------|------|--------|----------|
| FP4 E2M1 | 4.0 | ✅ Primary | Default weight format |
| INT4 U4/S4 | 4.0 | ✅ Working | GPTQ compatibility |
| INT3/INT2 | 3.0/2.0 | ✅ Working | Cold MoE experts |
| NF3/NF2 | 3.0/2.0 | ✅ Working | QLoRA formats |
| FP8 E4M3/E5M2 | 8.0 | ✅ Working | Higher precision |
| 2:4 Sparse | variable | ✅ Working | Structured sparsity |

---

## Tasks

See `tasks/quantized_inference.yaml` for implementation plan (24 tasks).

---

## Codebase Metrics

| Metric | Count |
|--------|-------|
| Python files (metal_marlin/) | ~125 |
| Metal shaders | 35 |

---

## Commands

```bash
# Test shader compilation
uv run python -c "
from metal_marlin.metal_dispatch import MetalKernelLibrary
lib = MetalKernelLibrary.from_source_dir()
print(f'Compiled: {len(lib._libraries)} libraries')
"

# Run tests
uv run pytest tests/ -v
```
