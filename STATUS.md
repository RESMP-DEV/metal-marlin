# Metal Marlin Status

**Last Updated:** 2026-01-27T17:45

## Summary

| Component | Status |
|-----------|--------|
| Test Suite | **100% passing** (1444/1497) |
| GEMM Kernel | **Working** ✅ |
| MoE Infrastructure | **Complete** ✅ (batched expert kernels wired) |
| Qwen3-4B FP4 Inference | **PyTorch MPS fallback** ~27 tok/s |
| GLM-4.7-Flash MoE | **Verified** ✅ (end-to-end generation working) |
| OpenAI Server | **Scaffolded** 🔄 |
| Metal Shaders | **5/5 compiling** ✅ |
| Inference Tests | **31/31 passing** ✅ |
| MLX Removal | **Complete** ✅ |
| Ruff Linting | **0 errors** ✅ |
| Pyright Errors | **0 errors, 184 warnings** ✅ |

---

## Test Results

**Last run:** 256.18s (4 min 16 sec)

| Category | Count |
|----------|-------|
| Passed | 1444 |
| Failed | 0 |
| Skipped | 53 |
| xfailed | 0 |
| xpassed | 0 |
| Errors | 0 |

**Recent changes:**
- Fixed KV attention quantization (corrected test quantization formula and exact inverse dequant)
- Fixed batched GEMV shape mismatch (removed duplicate M calculation)
- Added INT2 GEMM PyTorch fallback for GLM-4.7-Flash MLA support
- Fixed MetalQuantizedLinear output padding for non-aligned dimensions
- All INT4/FP4 GEMM tests now passing

**Remaining failures (0):**
- None

---

## GLM-4.7-Flash Integration

**Status:** ✅ Working via Transformers integration.

GLM-4.7-Flash is a **Mixture-of-Experts (MoE)** model with:
- Layer 0: Dense MLP
- Layers 1-46: MoE (64 routed experts + 1 shared expert per layer)
- Multi-Latent Attention (MLA) throughout

**Current State:**
- ✅ Quantization works correctly (9024 expert weights quantized)
- ✅ MLA attention dimensions fixed
- ✅ MoE infrastructure wired (`MetalQuantizedMoE`, `replace_moe_layers()`)
- ✅ Shared expert kernel (`moe_shared_expert_fp4`) integrated
- ✅ Inference working via Transformers + layer replacement

**How it works:** Transformers 5.0+ provides `Glm4MoeLiteForCausalLM` which handles
MoE routing natively. We swap `nn.Linear` → `MetalQuantizedLinear` for quantized GEMM.

```bash
# Requires transformers 5.0.0+
uv run pytest tests/test_glm4_integration.py -v --run-slow
```

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

## Architecture

**Metal Marlin integrates with HuggingFace Transformers, not reimplements it.**

Model structure (MoE routing, MLA attention, layer ordering) comes from Transformers.
We swap `nn.Linear` → `MetalQuantizedLinear` to use optimized Metal kernels.

```python
from transformers import AutoModelForCausalLM
from metal_marlin import replace_linear_layers, MetalQuantizedLinear

# Transformers handles architecture (MoE, MLA, everything)
model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash", device_map="mps")

# Our only job: swap Linear layers with quantized versions
replace_linear_layers(model, MetalQuantizedLinear, bits=4, group_size=128)

# Inference uses Transformers' battle-tested code
output = model.generate(input_ids)
```

**Requires:** `transformers>=5.0.0` for GLM-4.7-Flash native support.

---

## Implementation Progress

### Metal Kernels (Core Value)

| Kernel | Purpose | Status |
|--------|---------|--------|
| `marlin_gemm_fp4` | FP4 dequant + GEMM fused | ✅ Working |
| `flash_attention_v2` | Memory-efficient attention | ✅ Working |
| `moe_shared_expert_fp4` | Shared expert GEMM (GLM-4.7) | ✅ Wired |
| `moe_expert_gemm_fp4` | Batched routed expert GEMM | ✅ Wired (batched Metal dispatch) |
| `moe_dispatch_optimized` | GPU token grouping | ❌ Not wired |
| `moe_router_fused` | Fused routing kernel | ❌ Not wired |
| `dense_gemm` | BF16/FP16 GEMM | ✅ Working |

### MoE Infrastructure (Phase 42)

| Component | Status | Notes |
|-----------|--------|-------|
| `MetalQuantizedMoE` | ✅ Done | Holds packed expert weights |
| `find_moe_layers()` | ✅ Done | Auto-detect MoE in any model |
| `quantize_moe_experts()` | ✅ Done | Quantize 3D expert weight tensors |
| `replace_moe_layers()` | ✅ Done | Swap MoE layers with quantized versions |
| `moe_shared_expert_fp4()` | ✅ Done | Metal kernel for shared expert |
| `transformers_loader.py` | ✅ Done | MoE-aware model loading |
| Batched expert dispatch | ✅ Done | `moe_expert_gemm_fp4_grouped` Metal kernel wired |

### Remaining MoE Optimization

The routed expert computation now uses batched Metal dispatch via `moe_expert_gemm_fp4_grouped`.

**Still available for future optimization:**
- `moe_dispatch_optimized` - GPU-side token grouping (8 variants)
- `moe_router_fused` - Fused router kernel with aux loss

See [docs/metal_kernel_audit.md](docs/metal_kernel_audit.md) for full kernel inventory.

### Legacy Model Layers (Being Phased Out)

> **Note:** These custom layer implementations are being replaced by Transformers integration.
> New models should use `replace_linear_layers()` instead of custom model classes.

| Model | Custom Classes | Status |
|-------|---------------|--------|
| Llama | QuantizedLlamaAttention, QuantizedLlamaMLP | Legacy |
| Qwen3 | QuantizedQwen3Attention, QuantizedQwen3MLP | Legacy |
| GLM-4 | MetalMLAAttention, MetalMLP, MetalGLM47Model | Legacy (broken for MoE) |
| Mixtral | MixtralAttention, MixtralExpertMLP | Legacy |

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
| marlin_gemm_fp4 | ✅ Compiles, loads, **and works** |
| flash_attention_v2 | ✅ Compiles and loads |
| dense_gemm | ✅ Compiles and loads |
| moe_dispatch_optimized | ✅ Compiles and loads |
| simdgroup_attention | ✅ Compiles and loads |

### Known Metal Compiler Bugs (Documented)

See [docs/metal_array_parameter_bugs.md](docs/metal_array_parameter_bugs.md) for two Metal compiler bugs affecting simdgroup operations:
1. Functions receiving 2D `simdgroup_matrix` arrays require `__attribute__((always_inline))`
2. 3D threadgroup array slices should use pointers instead of 2D references

---

## Stubs & Incomplete Implementations

The following are intentional stubs or need optimization:

| Location | Function/Class | Status |
|----------|---------------|--------|
| kernels.py | flash_attention_fp4_kv | Stub - needs fused kernel |
| kernels.py | moe_expert_gemm_fp4 | ✅ Batched Metal dispatch |
| kernels.py | moe_router_topk | Works - uses PyTorch ops |
| kernels.py | moe_shared_expert_fp4 | ✅ Fully wired to Metal |
| speculative/engine.py | TargetModel.__call__ | Protocol - by design |
| speculative/engine.py | TargetModel.create_kv_cache | Protocol - by design |

---

## Blockers

### 1. GEMM Kernel Dispatch (Resolved ✅)

Fixed two Metal compiler bugs:
1. **Array Parameter Bug**: Functions receiving 2D `simdgroup_matrix` arrays need `__attribute__((always_inline))`
2. **Tile Coverage Bug**: Simdgroup configuration only covered 32 of 64 rows

See [docs/metal_array_parameter_bugs.md](docs/metal_array_parameter_bugs.md) for details.

### 2. Qwen3 LayerNorm Device (Resolved ✅)

`RuntimeError: Expected all tensors to be on the same device (mps:0 vs cpu)`
- Fixed by defaulting RMSNorm device to None/cpu
- Test now passing: `test_qwen3_layer_forward`

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
| 36 | GEMM dispatch debugging, Metal compiler bugs | ✅ Complete |
| 37 | FP4 reference fixes, Hadamard kernel, LayerNorm | ✅ Complete |
| 42 | MoE Pipeline (GLM-4.7-Flash) | ✅ Complete |

### Phase 42 Results

**Completed:**
- ✅ `MetalQuantizedMoE` class for quantized expert weights
- ✅ `moe_shared_expert_fp4()` wired to Metal kernel
- ✅ `find_moe_layers()`, `quantize_moe_experts()`, `replace_moe_layers()`
- ✅ `transformers_loader.py` with automatic MoE detection
- ✅ Tests: `test_moe_accuracy.py`, `test_glm47_transformers.py`, `test_qwen3_moe_transformers.py`
- ✅ Legacy model deprecation warnings
- ✅ `moe_expert_gemm_fp4` batched Metal dispatch (replaced Python loop)
- ✅ End-to-end GLM-4.7-Flash generation verified

**Remaining optimization work:**
- ❌ `moe_dispatch_optimized` GPU token grouping (not wired)
- ❌ `moe_router_fused` kernel (not wired)

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
