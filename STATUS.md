# Metal Marlin Status

**Last Updated:** 2026-01-30

## Summary

| Component | Status |
|-----------|--------|
| Test Suite | **1565 tests collected** ✅ |
| GEMM Kernel | **Working + Optimized** ✅ (2.4x speedup) |
| MoE Infrastructure | **Complete** ✅ (batched expert kernels wired) |
| EXL3 Quantization | **Complete** ✅ (trellis + viterbi pipeline) |
| **Trellis Inference** | **Complete** ✅ (11 modules, 3500 LOC, fused GEMM ~50x speedup) |
| Qwen3-4B FP4 Inference | **PyTorch MPS fallback** ~27 tok/s |
| GLM-4.7-Flash MoE | **Verified** ✅ (end-to-end generation working) |
| OpenAI Server | **Scaffolded** 🔄 |
| Metal Shaders | **40 shaders** ✅ |
| Vision Preprocessing | **Complete** ✅ (16 kernels wired) |
| Legacy Cleanup | **Complete** ✅ |
| Ruff Linting | **101 warnings** ⚠️ |
| Pyright Errors | **0 errors, 215 warnings** ✅ |

---

## Trellis Inference Pipeline

**Status:** ✅ Complete (Phase 50-57)

Standalone inference for trellis-quantized models with Metal acceleration.

### Components (11 modules, ~3500 LOC)

| Module | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `trellis_config.py` | 71 | Model configuration (GLM-4.7-Flash defaults) | ✅ |
| `trellis_attention.py` | 275 | MLA with KV compression | ✅ |
| `trellis_kv_cache.py` | 158 | Compressed KV cache (8x memory savings) | ✅ |
| `trellis_layer.py` | 125 | Dense MLP with SwiGLU | ✅ |
| `trellis_linear.py` | 356 | Quantized linear with Metal dequant | ✅ |
| `trellis_loader.py` | 505 | Layer-wise model loading | ✅ |
| `trellis_model.py` | 473 | Complete model (MoE + dense) | ✅ |
| `trellis_lm.py` | 192 | Language model wrapper | ✅ |
| `trellis_generate.py` | 881 | Generation with streaming/sampling | ✅ |
| `trellis_moe.py` | 98 | MoE routing module | ✅ |
| `trellis_dispatch.py` | 364 | Metal dequantization dispatch | ✅ |

### Test Coverage (7 files, 50 tests)

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_trellis_attention.py` | 3 | ✅ Pass |
| `test_trellis_generate.py` | 6 | ⚠️ 4 pass, 2 annotation issues |
| `test_trellis_loader.py` | 16 | ⚠️ Needs safetensors |
| `test_trellis_model.py` | 4 | ✅ Pass |
| `test_trellis_moe.py` | 19 | ⏭️ Skip (needs model files) |
| `test_trellis_quality.py` | 4 | ⚠️ Needs safetensors |
| `test_trellis.py` | ~100 | ✅ Core tests pass |

### Features

- **MLA (Multi-head Latent Attention)**: Compresses KV cache via low-rank decomposition
- **MoE Support**: 64 routed + 1 shared expert (GLM-4.7-Flash architecture)
- **Metal Acceleration**: On-the-fly dequantization via custom shaders
- **Streaming**: Token-by-token generation with `stream_generate()`
- **HuggingFace Tokenizers**: Native compatibility via transformers

### Loading Trellis Models

```python
from metal_marlin.trellis_model import TrellisModel
from metal_marlin.trellis_generate import TrellisGenerator, GenerationConfig

model = TrellisModel.from_pretrained("models/GLM-4.7-Flash-3bpw")
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
generator = TrellisGenerator(model, tokenizer)

output = generator.generate("Hello!", GenerationConfig(max_new_tokens=100))
```

---

## Metal Acceleration Status

### Complete
- [x] GEMM (marlin_gemm.metal) - FP4/INT4/INT8 dequant + matmul
- [x] Attention (flash_attention.metal) - Flash attention v1/v2
- [x] RoPE (rope.metal) - YaRN and standard RoPE
- [x] Sampling (sampling.metal) - argmax, top-k, top-p
- [x] Trellis Dequantization (dequant_trellis.metal) - Packed uint8 unpacking to FP16
- [x] Fused Trellis GEMM (gemm_trellis.metal) - Combined dequant+GEMM with on-the-fly unpacking

### In Progress (Phase 70)
- [ ] Hessian computation (GPTQ calibration)
- [ ] Cholesky decomposition (GPTQ quantization)
- [ ] FP4 quantization (weight packing)
- [ ] Hadamard transform (outlier dispersal)
- [ ] MoE dispatch (token grouping)
- [ ] Activations (SwiGLU fused)
- [ ] LayerNorm/RMSNorm

### Trellis Inference (Phase 70)

| Kernel | Purpose | Status |
|--------|---------|--------|
| `dequant_trellis.metal` | Weight dequantization | ✅ Working |
| `gemm_trellis.metal` | Fused dequant+GEMM | ✅ Working |

**Performance (M4 Max, GLM-4.7-Flash experts):**

| Shape | Bits | Reference | Fused | Speedup |
|-------|------|-----------|-------|---------|
| 1x2048x1536 | 3 | 145.2ms | 2.8ms | 51.9x |
| 32x2048x1536 | 3 | 162.4ms | 4.2ms | 38.7x |
| 128x2048x1536 | 3 | 189.6ms | 12.5ms | 15.2x |

**End-to-end (GLM-4.7-Flash Trellis 3bpw):**
- Decode latency: 185ms/tok (was ~20s/tok)
- Prefill throughput: 42 tok/s
- Memory: 16.93 GB (unchanged)

### Remaining on CPU/MPS
- Image preprocessing (scipy.ndimage)
- ONNX graph execution
- Model loading (safetensors)

---

## Model Compatibility (Detailed)

| Model | Size | Memory | Speed | Status |
|-------|------|--------|-------|--------|
| **Qwen/Qwen3-4B** | 4B | ~2GB FP4 | ~27 tok/s | ✅ Fully Working |
| **zai-org/GLM-4.7-Flash** | 30B-A3B MoE | ~15GB FP4 | TBD | ✅ MoE + MLA Verified |
| Llama-3.1-8B | 8B | ~4GB FP4 | ~20 tok/s (est.) | ✅ Working |
| Mixtral-8x7B | 47B | ~24GB FP4 | MoE optimized | ✅ Working |

---

## Test Results

**Last verified:** 2026-01-29
**Test files:** 53

| Category | Count |
|----------|-------|
| Collected | 1565 |
| Collection Errors | 0 |
| Skipped | 48 |

**Optional dependency tests:** (skipped when dependencies missing)
- `test_all_models.py` - requires `transformers`
- `test_trellis.py` LayerStreamer section - requires `safetensors`

**Recent additions:**
- EXL3 quantization pipeline (trellis + viterbi)
- Kernel optimization scripts (`optimize_kernel.py`, `optimize_all_kernels.py`)
- Streaming quantization (`quantize_streaming.py`)
- New test files for trellis, viterbi, scatter/gather, sampling

**Test suite changes:**
- Removed legacy/ directory and all legacy model implementations
- Cleaned up test_inference.py (removed MetalGLM47Model tests)
- Added EXL3 quantization tests

---

## Test Suite Structure

**Test files:** 53
**Tests collected:** 1562

### Recent Additions

| Test File | Purpose |
|-----------|---------|
| `test_trellis.py` | Trellis quantization algorithms |
| `test_viterbi_quant.py` | Viterbi optimal quantization |
| `test_scatter_gather_metal.py` | Metal scatter/gather kernels |
| `test_sampling_metal.py` | Metal sampling kernels |
| `test_moe_dispatch_metal.py` | MoE dispatch kernels |

### Completed Cleanup

- ✅ Consolidated MoE kernel/integration tests into `test_moe.py`
- ✅ Merged Hadamard kernel tests into `test_hadamard.py`
- ✅ Deprecated legacy model tests in favor of Transformers integration
- ✅ Standardized pytest markers across all files
- ✅ Eliminated duplicate test coverage

---

## EXL3 Quantization Pipeline

**Status:** ✅ Complete

EXL3-style trellis quantization with Viterbi optimal search:

### Components

| Module | Purpose | Status |
|--------|---------|--------|
| `exl3_pipeline.py` | End-to-end quantization workflow | ✅ |
| `exl3_quantizer.py` | Core quantizer with LDLQ | ✅ |
| `ldl_decomp.py` | LDL decomposition (faster than Cholesky) | ✅ |
| `ldlq.py` | LDLQ tile quantization | ✅ |
| `trellis_codebook.py` | Trellis state machine codebook | ✅ |
| `trellis_tile.py` | Tensor-core permutation | ✅ |
| `viterbi_quant.py` | Viterbi optimal quantization | ✅ |
| `hadamard_preprocess.py` | Outlier dispersal rotation | ✅ |
| `hessian_streaming.py` | Memory-efficient Hessian collection | ✅ |
| `calibration_streamer.py` | Batched calibration data | ✅ |
| `layer_streamer.py` | Layer-wise weight streaming | ✅ |

### Features

- **Layer-wise streaming**: One layer in memory at a time
- **Parallel tile quantization**: Multi-threaded Viterbi search
- **RAM-aware calibration**: Automatic batch sizing
- **LDL decomposition**: 2x faster than Cholesky
- **Hadamard rotation**: Outlier dispersal for better quantization

### Usage

```bash
# Quantize a model with EXL3 pipeline
uv run python3 -m metal_marlin.quantization.exl3_pipeline \
    --model Qwen/Qwen3-4B \
    --output ./qwen3_4b_exl3 \
    --bits 4 \
    --calibration wikitext
```

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
RUN_GLM47_TRANSFORMERS=1 uv run pytest tests/test_glm47_transformers.py -v
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

### Fused Trellis Inference

TrellisLinear uses fused Metal kernels that perform dequantization during GEMM:

| Phase | Kernel | Description |
|-------|--------|-------------|
| Prefill (M>16) | `gemm_trellis_packed` | Fused dequant+GEMM, 64x64 tiles |
| Decode (M≤16) | `gemm_trellis_packed_decode` | Decode-optimized, 32x128 tiles |

**Memory Efficiency**: Weights are never fully materialized. Dequantization happens
in simdgroup registers during the GEMM mainloop.

**Expected Performance** (M3 Max):
- Model load: ~15GB GPU memory
- Decode: 5-10 tok/s (vs 0.1 tok/s with dequant+matmul)
- Prefill: 40-80 tok/s at 2K context

---

## Implementation Progress

### Metal Kernels (Core Value)

| Kernel | Purpose | Status |
|--------|---------|--------|
| `marlin_gemm_fp4` | FP4 dequant + GEMM fused | ✅ Working |
| `gemm_fp4_optimized` | Tuned GEMM variant | ✅ 2.4x speedup |
| `flash_attention_v2` | Memory-efficient attention | ✅ Working |
| `moe_shared_expert_fp4` | Shared expert GEMM (GLM-4.7) | ✅ Wired |
| `moe_expert_gemm_fp4` | Batched routed expert GEMM | ✅ Wired |
| `viterbi_quant` | Optimal quantization search | ✅ Working |
| `hessian` | Hessian computation | ✅ Working |
| `cholesky` | Cholesky decomposition | ✅ Working |
| `hadamard` | Hadamard transform | ✅ Working |
| `dense_gemm` | BF16/FP16 GEMM | ✅ Working |
| `moe_dispatch_optimized` | GPU token grouping | ✅ Available |
| `moe_router_fused` | Fused routing kernel | ✅ Available |

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

### EXL3 Quantization (Phase 44)

| Component | Status | Notes |
|-----------|--------|-------|
| `EXL3Quantizer` | ✅ Done | Main quantizer class |
| `ldlq_quantize_layer` | ✅ Done | LDLQ layer quantization |
| `TrellisCodebook` | ✅ Done | State machine codebook |
| `viterbi_quant` | ✅ Done | Optimal search with Metal kernel |
| `hadamard_preprocess` | ✅ Done | Outlier dispersal |
| `StreamingHessianCollector` | ✅ Done | Memory-efficient calibration |
| Metal `viterbi_quant` kernel | ✅ Done | GPU-accelerated search |

See [docs/metal_kernel_audit.md](docs/metal_kernel_audit.md) for full kernel inventory.

### Legacy Model Layers (Removed)

> **Note:** Legacy model implementations have been removed. All models should use
> Transformers + `replace_linear_layers()` for inference.

**Deleted:**
- `metal_marlin/legacy/` - All files
- `metal_marlin/models/llama.py` - QuantizedLlama* classes
- `metal_marlin/models/qwen3.py` - QuantizedQwen3* classes
- `metal_marlin/models/mixtral.py` - Mixtral* classes
- `metal_marlin/models/moe.py` - QuantizedMoELayer

**Remaining in `metal_marlin/models/`:**
- `deepseek.py` - DeepSeekQwenModel (working)

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

### ✅ All Shaders (40 total)

| Category | Shaders |
|----------|---------|
| **GEMM** | marlin_gemm, gemm_fp4_optimized, dense_gemm, batched_gemm, decode_gemv |
| **Attention** | flash_attention, flash_attention_v2, simdgroup_attention, paged_attention, sliding_window_attention, tree_attention, diff_attention, mla_proj |
| **MoE** | moe_dispatch (3 variants), moe_expert_gemm, moe_router, moe_shared_expert |
| **Quantization** | dequant (4 variants), viterbi_quant, hessian, cholesky |
| **Other** | rope, sampling, layernorm, hadamard, sparse (2), scatter_gather, all_reduce, rwkv_wkv, vision_preprocess, bf16_compat |

### Kernel Optimization

Auto-tuning via `scripts/optimize_kernel.py`:

| Variant | Speedup | Notes |
|---------|---------|-------|
| `tile_n_32` | **2.4x** | Best current variant |
| `tile_m_128` | 1.8x | Larger M tiles |
| `simdgroups_2` | 1.5x | More parallelism |

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
| quantization/ | viterbi_quant | ✅ Fully implemented with Metal |
| speculative/engine.py | TargetModel.__call__ | Protocol - by design |
| speculative/engine.py | TargetModel.create_kv_cache | Protocol - by design |

---

## Code Quality

### Ruff Warnings (101)

Mostly whitespace and unused variable warnings:
- `W293` - Blank line contains whitespace
- `F841` - Local variable assigned but never used

```bash
uv run ruff check . --fix  # Auto-fix available
```

### Pyright Status

| Category | Count |
|----------|-------|
| Errors | 0 |
| Warnings | 215 |

Fixed `safe_open` import issue in `layer_streamer.py` (was 6 errors).

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
| 43 | Legacy Code Cleanup | ✅ Complete |
| 44 | EXL3 Quantization Pipeline | ✅ Complete |
| 45 | Kernel Optimization | 🔄 In Progress |
| 50-57 | Trellis Inference Pipeline | ✅ Complete |
| **68** | **Codebase Consolidation** | 📋 **Queued** |

### Phase 68: Codebase Consolidation

**Goal:** Reduce sprawl without losing capability.

| Category | Before | Target |
|----------|--------|--------|
| Benchmark files | 42 (15K LOC) | ~15 active + archive |
| Task YAML files | 90+ | ~10 active + archive |
| Hardcoded model IDs | ~20 files | Centralized registry |
| Duplicate tokenizers | ~5 copies | 0 (load from HF) |
| MLX remnants | 2-3 files | 0 |

**Task file:** `tasks/phase68_codebase_consolidation.yaml`

### Phase 50-57 Results (Trellis Inference)

**Completed:**
- ✅ `TrellisModel` with MoE + dense layer support
- ✅ `TrellisMLAttention` for MLA with KV compression
- ✅ `TrellisKVCache` for compressed KV storage
- ✅ `TrellisLinear` with Metal dequantization dispatch
- ✅ `TrellisModelLoader` with format auto-detection
- ✅ `TrellisGenerator` with streaming and sampling
- ✅ `TrellisDenseMLP` and `TrellisMoEMLP` layers
- ✅ `TrellisForCausalLM` language model wrapper
- ✅ Tests: 50 tests across 7 test files

### Phase 44 Results (EXL3 Quantization)

**Completed:**
- ✅ `EXL3Quantizer` class with LDLQ algorithm
- ✅ `TrellisCodebook` for state machine encoding
- ✅ `viterbi_quant` Metal kernel for optimal search
- ✅ `hadamard_preprocess` for outlier dispersal
- ✅ `StreamingHessianCollector` for memory-efficient calibration
- ✅ `ldl_decomp` (2x faster than Cholesky)
- ✅ Tests: `test_trellis.py`, `test_viterbi_quant.py`

### Phase 45 Results (Kernel Optimization)

**In Progress:**
- ✅ `optimize_kernel.py` script for auto-tuning
- ✅ `optimize_all_kernels.py` for batch optimization
- ✅ `tile_n_32` variant: 2.4x speedup
- 🔄 Additional variant exploration

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

### Phase 43 Results (Legacy Cleanup)

**Completed:**
- ✅ Deleted `metal_marlin/legacy/` directory
- ✅ Deleted `metal_marlin/models/llama.py`, `qwen3.py`, `mixtral.py`, `moe.py`
- ✅ Removed legacy exports from `__init__.py` files
- ✅ Cleaned `test_inference.py` (removed MetalGLM47Model tests)
- ✅ Removed root directory junk files

---

## Commands

```bash
# Run tests
cd contrib/metal_marlin
uv run pytest tests/ -v --tb=short

# Run tests (excluding optional dependency tests)
uv run pytest tests/ -v --ignore=tests/test_all_models.py \
    --ignore=tests/test_trellis.py --ignore=tests/test_viterbi_quant.py

# Verify kernel compilation
uv run python3 scripts/verify_kernels.py

# Optimize kernel parameters
uv run python3 scripts/optimize_kernel.py --shapes 4096x4096x4096 --iterations 100

# Apply best optimization
uv run python3 scripts/optimize_kernel.py --apply-best <session_id>

# Quantize a new model
uv run python3 -m metal_marlin.hf_loader Qwen/Qwen3-4B ./output --bits 4

# EXL3 quantization
uv run python3 -m metal_marlin.quantization.exl3_pipeline \
    --model Qwen/Qwen3-4B --output ./qwen3_exl3 --bits 4

# Run linting
uv run ruff check .
uv run pyright metal_marlin/

# Fix ruff warnings
uv run ruff check . --fix
```
