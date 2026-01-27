# Metal Marlin Status

**Last Updated:** 2026-01-27T00:00

## Summary

| Component | Status |
|-----------|--------|
| Test Suite | **90% passing** (1337/1482) |
| Qwen3-4B FP4 Inference | **PyTorch MPS fallback** ~27 tok/s |
| OpenAI Server | **Scaffolded** 🔄 |
| Metal Shaders | **5/5 compiling** ✅ |
| Inference Tests | **31/31 passing** ✅ |
| MLX Removal | **Complete** ✅ |
| GLM-4.7-Flash MLA | **Working** ✅ |
| Ruff Linting | **0 errors** ✅ |
| Pyright Errors | **0 errors, 184 warnings** ✅ |

---

## Test Results

**Last run:** 334.40s (5 min 34 sec)

| Category | Count |
|----------|-------|
| Passed | 1337 |
| Failed | 145 |
| Skipped | 36 |
| xfailed | 11 |
| xpassed | 14 |
| Errors | 0 |

**Phase 35 improvements:**
- All 5 Metal kernels now compile and load ✅
- All 31 inference tests passing ✅
- MetalTransformerBlock tests passing ✅
- GLM-4.7 model tests passing ✅
- ZeroModule signature fixed ✅

**Remaining failures (145):**
- GEMM kernel dispatch (kernels compile but output zeros) - 80+ tests
- Hadamard transform (outputs zeros) - 4 tests
- Stripe partition (depends on GEMM) - 14 tests
- Qwen3 LayerNorm device mismatch - 1 test
- Edge cases - various

---

## Inference Pipeline

Inference uses PyTorch MPS fallback (not fused Metal kernels):

```bash
cd contrib/metal_marlin
uv run python3 -c "
from metal_marlin.inference.pipeline import MarlinPipeline
pipe = MarlinPipeline.from_pretrained('benchmarks/results/qwen3_4b_fp4', device='mps')
print(pipe('The capital of France is', max_tokens=20))
"
```

**Requires:** `transformers` package installed.

| Metric | Value |
|--------|-------|
| Backend | PyTorch MPS (fallback) |
| Target | Fused Metal kernels (~100 tok/s) |

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

Verified via `scripts/verify_kernels.py`:

### ✅ All Compiling (5/5)

| Shader | Status |
|--------|--------|
| marlin_gemm_fp4 | ✅ Compiles and loads |
| flash_attention_v2 | ✅ Compiles and loads |
| dense_gemm | ✅ Compiles and loads |
| moe_dispatch_optimized | ✅ Compiles and loads |
| simdgroup_attention | ✅ Compiles and loads |

### ⚠️ Runtime Issue

Kernels compile but GEMM tests output zeros. The issue is in kernel dispatch/buffer binding, not compilation.

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

### 1. GEMM Kernel Dispatch (P0)

All 5 Metal kernels compile ✅ but GEMM operations return zeros.
- Kernels load successfully via PyObjC
- Output buffers remain zeros after dispatch
- Likely issue: buffer binding, thread group configuration, or compute encoder setup

**Affects:** ~100 tests (GEMM boundaries, accuracy, stripe partition, Hadamard)

### 2. Qwen3 LayerNorm Device (P1)

`RuntimeError: Expected all tensors to be on the same device (mps:0 vs cpu)`
- LayerNorm weights not moving with `.to(device)`
- Affects 1 test: `test_qwen3_layer_forward`

### 3. GLM-4.7-Flash MLA (Resolved ✅)

MLA implementation now working - all GLM-4.7 model tests pass.

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

Current swarm status:

| Phase | Tasks | Status |
|-------|-------|--------|
| 32 | Buffer cache fix, INT4 export, linting | ✅ Complete |
| 33 | Qwen3/GLM4 layer implementations | ✅ Complete |
| 34 | Test failures, kernel integration | ✅ Complete |
| 35 | Kernel compilation, device mismatch, ZeroModule | ✅ Complete |
| 36 | GEMM dispatch debugging, LayerNorm device | 🔄 Next |

### Phase 35 Results

**Fixed:**
- ✅ All 5 Metal kernels now compile
- ✅ ZeroModule forward signature
- ✅ Inference tests (31/31)
- ✅ GLM-4.7 model tests

**Remaining:**
- ❌ GEMM kernel dispatch (outputs zeros despite compiling)
- ❌ Qwen3 LayerNorm device mismatch (1 test)

---

## Commands

```bash
# Run tests
cd contrib/metal_marlin
uv run pytest tests/ -v --tb=short

# Verify kernel compilation
uv run python3 scripts/verify_kernels.py

# Quantize a new model
uv run python3 -m metal_marlin.hf_loader Qwen/Qwen3-4B ./output --bits 4

# Run linting
uv run ruff check .
uv run pyright metal_marlin/
```
