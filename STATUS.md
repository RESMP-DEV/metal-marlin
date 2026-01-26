# Metal Marlin Status

**Last Updated:** 2026-01-25T22:52

## Summary

Metal shader compilation: **31/32** (97%)
MLX removal: **0 files remaining** ✅ COMPLETE
Task queue: 305 completed

---

## Blockers

### 1. Failing Shader: marlin_gemm.metal

```
error: no matching function for call to 'store_results_fp32'
```

**Root cause:** `store_results_fp32` signature changed to require staging buffer, but call sites not updated.

**Fix required:** Update all kernel call sites to pass staging buffer parameter.

---

### 2. MLX Removal: COMPLETE ✅

| Status | Count |
|--------|-------|
| Project code | 0 files |
| .venv (third-party) | 3 files (transformers, safetensors - not our code) |

**Progress:** 52 → 0 files after 305 agent tasks

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

| Format | Bits | Status |
|--------|------|--------|
| FP4 E2M1 | 4.0 | Primary |
| INT4 U4/S4 | 4.0 | Working |
| INT3/INT2 | 3.0/2.0 | Working |
| NF3/NF2 | 3.0/2.0 | Working |
| FP8 E4M3/E5M2 | 8.0 | Working |

---

## Quantized Models

**To quantize:**

| Model | Type | Size (FP16) | Target Config |
|-------|------|-------------|---------------|
| zai-org/GLM-4.7-Flash | MoE (64 experts) | ~18 GB | FP8 attention, INT2 cold experts |
| Qwen/Qwen3-4B | Dense | ~8 GB | FP4 weights, FP8 attention |

---

## Codebase Metrics

| Metric | Count |
|--------|-------|
| Python files (metal_marlin/) | 125 |
| Python files (tests/) | 30 |
| Python files (benchmarks/) | 4 |
| Metal shaders | 35 |

---

## Dependencies

```toml
pyobjc-core
pyobjc-framework-Metal
pyobjc-framework-MetalPerformanceShaders
torch  # MPS backend
safetensors
transformers
```

---

## Commands

```bash
# Test shader compilation
uv run python -c "
from metal_marlin.metal_dispatch import MetalKernelLibrary
lib = MetalKernelLibrary.from_source_dir()
print(f'Compiled: {len(lib._libraries)} libraries')
"

# Check MLX references
grep -rl 'mx\.fast\|import mlx' --include='*.py' . | grep -v .venv

# Run tests
uv run pytest tests/ -v

# Regenerate this file
uv run python scripts/collect_status_facts.py
```
