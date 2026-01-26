# Metal Marlin Status

**Last Updated:** 2026-01-26T21:00

## Summary

| Component | Status |
|-----------|--------|
| Qwen3-4B FP4 Inference | **Working** ✅ ~27 tok/s |
| OpenAI Server | **Scaffolded** 🔄 |
| Metal Shaders | **30/35** compiling |
| MLX Removal | **Complete** ✅ |
| GLM-4.7-Flash MLA | **In Progress** 🔄 |
| Ruff Linting | **0 errors** ✅ |
| Pyright Errors | **0 errors** ✅ |

---

## Working: Qwen3-4B FP4 Inference

End-to-end inference working via PyTorch MPS fallback:

```bash
cd contrib/metal_marlin
python3 -c "
from metal_marlin.inference.pipeline import MarlinPipeline
pipe = MarlinPipeline.from_pretrained('benchmarks/results/qwen3_4b_fp4', device='mps')
print(pipe('The capital of France is', max_tokens=20))
"
```

| Metric | Value |
|--------|-------|
| Model | Qwen3-4B FP4 |
| Throughput | ~27 tok/s decode |
| Compression | 7.76x |
| Backend | PyTorch MPS (fallback) |

**Note:** Using PyTorch dequant+matmul, not fused Metal kernels yet.

---

## Implementation Progress

### Model Layers (Phase 33 - In Progress)

| Model | Attention | MLP | Layer | Status |
|-------|-----------|-----|-------|--------|
| Llama | ✅ QuantizedLlamaAttention | ✅ QuantizedLlamaMLP | ✅ QuantizedLlamaLayer | Complete |
| Qwen3 | 🔄 QuantizedQwen3Attention | 🔄 QuantizedQwen3MLP | 🔄 QuantizedQwen3Layer | In Progress |
| GLM-4 | 🔄 QuantizedGLM4Attention (MLA) | 🔄 QuantizedGLM4MLP | 🔄 QuantizedGLM4Layer | In Progress |
| Mixtral | ✅ MixtralAttention | ✅ MixtralExpertMLP | ✅ MixtralLayer | Complete |
| DeepSeek | 🔄 DeepSeekMLA | 🔄 DeepSeekMoE | 🔄 DeepSeekLayer | Partial |

### Attention Implementations

| Implementation | Location | Purpose | Status |
|---------------|----------|---------|--------|
| MetalAttention | inference_metal.py | Standard MHA with Metal | ✅ Working |
| MetalMLAAttention | inference_metal.py | MLA for GLM-4/DeepSeek | 🔄 Partial |
| MLAAttention | mla_attention.py | Latent attention module | ✅ Working |
| FlashAttention | flash_attention_v2.py | Flash attention v2 | ✅ Working |
| DifferentialAttention | architectures/diff_transformer.py | Diff-transformer | ✅ Working |
| TreeAttention | tree_attention.py | Speculative tree attn | ✅ Working |

### MLP Implementations

| Implementation | Location | Purpose | Status |
|---------------|----------|---------|--------|
| MetalMLP | inference_metal.py | SwiGLU with Metal | ✅ Working |
| MarlinMLP | mlp.py | Quantized MLP | ✅ Working |
| TensorParallelMLP | distributed/tensor_parallel.py | TP-sharded MLP | ✅ Working |
| MixtralExpertMLP | models/mixtral.py | MoE expert | ✅ Working |

---

## OpenAI-Compatible Server

vLLM-style server scaffolded in `metal_marlin/serving/`:

```bash
# Start server
metal-marlin serve benchmarks/results/qwen3_4b_fp4 --port 8000

# Or with Python
python -m metal_marlin serve benchmarks/results/qwen3_4b_fp4
```

**Endpoints:**
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `POST /v1/completions` - Text completions
- `GET /health` - Health check

---

## Metal Shader Status

### ✅ Compiling (30/35)

| Category | Shaders |
|----------|---------|
| **GEMM** | marlin_gemm ✅, gemm_fp4_optimized, sparse_gemm, batched_gemm |
| **Attention** | attention, flash_attention, simdgroup_attention, paged_attention, diff_attention |
| **Dequant** | dequant, dequant_fp8, dequant_int8, dequant_sub4bit |
| **MoE** | moe_dispatch, moe_router, moe_expert_gemm, moe_shared_expert |
| **Other** | rope, sampling, hadamard, bf16_compat, decode_gemv, mla_proj |

### ❌ Issues (5/35)

| Shader | Issue |
|--------|-------|
| dense_gemm | Missing `SMALL_SG_M_TILES` defines |
| flash_attention_v2 | Compile warning only |
| moe_dispatch_optimized | Function not found |
| simdgroup_attention | Function not found |
| test_async_copy | Removed (unsupported AIR intrinsics) |

---

## Stubs & Incomplete Implementations

The following are intentional stubs awaiting full implementation:

| Location | Function/Class | Status |
|----------|---------------|--------|
| kernels.py:1702 | flash_attention_fp4_kv | Stub - needs fused kernel |
| kernels.py:1731 | moe_expert_gemm_fp4 | Stub - needs dispatch |
| kernels.py:1751 | moe_router_topk | Stub - needs kernel |
| speculative/engine.py:46 | TargetModel.__call__ | Protocol - by design |
| speculative/engine.py:50 | TargetModel.create_kv_cache | Protocol - by design |

---

## Blockers

### 1. Fused Kernel Integration (P0)

`marlin_gemm_fp4` shader compiles ✅ but runtime buffer conversion fails:
```
TypeError: converting to a C array (PyObjC Metal buffer issue)
```

Current workaround: PyTorch MPS fallback (~27 tok/s vs target ~100 tok/s).

### 2. GLM-4.7-Flash MLA (P1)

Multi-Latent Attention requires:
- Latent projection (down_proj, up_proj)
- Rotary position embedding on latents
- Quantized KV cache support

MLA implementation started in `mla_attention.py` and `mla_kv_cache.py`.

---

## Quantization Formats

| Format | Bits | Status | Use Case |
|--------|------|--------|----------|
| FP4 E2M1 | 4.0 | ✅ Working | Default weights |
| INT4 U4/S4 | 4.0 | ✅ Working | GPTQ compat |
| FP8 E4M3 | 8.0 | ✅ Working | Higher precision |
| INT3/INT2 | 3/2 | ✅ Working | Cold experts |
| 2:4 Sparse | var | ✅ Working | Sparsity |

---

## Task Queue

Current swarm status (Phase 32-33):

| Phase | Tasks | Status |
|-------|-------|--------|
| 32 | Buffer cache fix, INT4 export, linting | ✅ Complete |
| 33 | Qwen3/GLM4 layer implementations | 🔄 In Progress |

---

## Commands

```bash
# Test inference
cd contrib/metal_marlin
python3 -c "
from metal_marlin.inference.pipeline import MarlinPipeline
pipe = MarlinPipeline.from_pretrained('benchmarks/results/qwen3_4b_fp4', device='mps')
print(pipe('Hello world', max_tokens=20))
"

# Verify kernel compilation
python3 scripts/verify_kernels.py

# Quantize a new model
python -m metal_marlin.hf_loader Qwen/Qwen3-4B ./output --bits 4

# Run linting
uv run ruff check .
uv run pyright metal_marlin/
```
