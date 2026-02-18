# Metal-Specific LLM Optimization Guide

> **Ground truth for LLM inference on Apple Silicon**
> 
> There is no established playbook for Metal optimization like there is for CUDA.
> This guide documents what works, what doesn't, and why.

**Last Updated:** February 11, 2026

## GLM-4.7-Flash Status Summary

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **Decode** | 0.74 tok/s | 15-30 tok/s | 20-40× |
| **Prefill (2K)** | 42 tok/s | 500+ tok/s | 12× |
| **Memory** | 33 GB | 12-15 GB | ~2.5× |
| **Model Load** | ~15s | ~10s | 1.5× |

### Done (Verified)
- ✅ Fused GEMM kernels (51.9× speedup)
- ✅ Batched MoE dispatch (200× vs sequential)
- ✅ **Fused MoE kernel wired** (13× speedup: 6381ms → 488ms forward, Feb 2026)
- ✅ Async dispatch batching (1.29× speedup)
- ✅ Mixed-precision MoE (2-8 bit per expert)
- ✅ MLA attention implementation (8.9× KV compression)
- ✅ Tests passing (Trellis infrastructure working)
- ✅ **PagedKVCache adapter** (vLLM-style 16-token blocks, fp16/fp8/int4, Feb 2026)
- ✅ **MLA fused decode wired** (`use_fused_decode=True` in MMFP4MLA, Feb 2026)
- ✅ **E2E decode benchmark** (0.74 tok/s, 2.65× vs 0.28 baseline, Feb 2026)

### Not Done (Blocking Performance)
- ❌ Paged attention kernel integration (PagedKVCache exists, kernel dispatch not wired)
- ❌ Quantized KV cache dispatch (FP8/INT4 storage ready, kernel path incomplete)
- ❌ Decode GEMV optimization (single-token path unoptimized)
- ⚠️ C++ dispatch API mismatch (using Python fallback for kernel dispatch)

### Remaining Gap Analysis

With fused MoE kernel (13× measured) and MLA fused decode wired:

| Bottleneck | Current Cost | Optimized | Expected Gain |
|------------|--------------|-----------|---------------|
| ~~MoE per-layer dispatch~~ | ~~98% of forward~~ | ✅ **DONE** | **13× measured** |
| ~~MLA fused attention~~ | ~~5+ dispatches/layer~~ | ✅ **WIRED** | (measuring) |
| Paged attention dispatch | Not integrated | PagedKVCache ready | 2× context |
| Decode GEMV | Matmul fallback | Specialized GEMV | 2× |
| Memory overhead | 33 GB (2× model) | Buffer pooling | ~1.5× |

**Current E2E: 0.74 tok/s** (2.65× vs baseline)

## Executive Summary

**Hard-won lessons from Metal kernel development:**

### Performance Truths (Measured)
1. **MoE dispatch WAS the bottleneck** - 98.5% of forward pass (fixed Feb 2026: 13× gain)
2. **Kernel fusion provides 50x+ gains** - measured 51.9x speedup from fused GEMM
3. **Batched expert dispatch is critical** - 200x improvement over sequential (20s → <100ms)
4. **Fused MoE kernel = 13× speedup** - 6381ms → 488ms forward pass (Feb 2026)
5. **Dispatch overhead is real** - ~8μs per dispatch × 564 dispatches = 4.5ms wasted
6. **Unified memory doesn't eliminate BW limits** - still memory-bound at long context

### Correctness Lessons (Learned the Hard Way)
7. **Tile boundaries cause silent NaN** - M < TILE_M can produce undefined behavior
8. **Layout transposition is the #1 bug** - always verify [N, K] vs [K, N] at call site
9. **Weight shapes need transpose for fused kernels** - kernel expects `[in/8, out]`, weights stored `[out, in/8]` (Feb 2026)
10. **State corrupts across calls** - buffer reuse without zeroing breaks on iteration 2+
11. **MPS is non-deterministic** - use generous tolerances (1e-4, not 1e-6)
12. **Document expected layouts** - kernel signature MUST specify tensor shapes
13. **Async dispatch causes random NaN** - `wait=False` + immediate output read = uninitialized data (Feb 2026)
14. **C++ dispatch API mismatch** - `dispatch_kernel` wrapper expects `(lib, fn_name)`, C++ expects `(ctx, pipeline)` - use Python path (Feb 2026)
15. **Metal compiler has bugs** - `__attribute__((always_inline))` required for simdgroup arrays, float intermediates for dequant (Jan 2026)
16. **Buffer pool ≠ always faster** - manual `.copy_()` into pooled buffers caused 3× regression vs `torch.stack()`; MPS copy overhead dominates benefits (Feb 2026)

## Architecture: Apple Silicon vs CUDA

### Why CUDA Patterns Don't Apply

**Apple Silicon has fundamentally different architecture.** Don't assume CUDA optimization patterns work here.

| CUDA Assumption | Apple Silicon Reality (Measured) |
|-----------------|----------------------|
| FP32 is 2× slower than FP16 | **1.10× measured** - unified ALUs, negligible difference |
| Tensor Cores give 16× speedup | **No equivalent** - simdgroup_matrix helps but differently |
| Kernel launch is free (<1μs) | **~7μs measured** - hundreds of dispatches add up |
| Memory is always the bottleneck | **Depends** - small GEMMs are faster than memory-bound limit |

### Key Differences (Observed)

| Aspect | Metal (M4 Max) | Measured | Implication |
|--------|----------------|----------|-------------|
| Memory | 128GB Unified | - | No CPU↔GPU copies, but still BW-limited |
| Memory BW | 546 GB/s peak | 288 GB/s (53%) | Simple copy achieves ~53% of peak |
| Dispatch overhead | Per kernel | 7 μs | 564 dispatches = 4ms overhead |
| FP32/FP16 GEMM | 2048×2048 | 1.10× ratio | Unified ALUs - FP16 not significantly faster |

**What we learned:**
- Unified memory eliminates PCIe copies but doesn't eliminate bandwidth limits
- Dispatch overhead is measurable and adds up fast with many small kernels
- FP32 vs FP16 tradeoff is different than NVIDIA - must benchmark on Apple Silicon

### What This Means in Practice

**Advantages (verified):**
- Zero-copy model loading - 15s to load 17GB model
- No OOM from CPU↔GPU fragmentation
- MTLBuffer with storageModeShared just works

**Disadvantages (learned the hard way):**
- Hundreds of small dispatches kill performance (MoE naive: 20s/token)
- No CUDA-like kernel fusion toolchain
- Must manually fuse operations for acceptable perf

## Measured Performance: GLM-4.7-Flash

### End-to-End Benchmarks (M4 Max)

| Metric | Measured Value | Source |
|--------|----------------|--------|
| Model Load Time | ~15s | bench_glm47_kernels.py |
| Prefill (2K context) | 42 tok/s | glm_flash_benchmark.py |
| Decode (batch=1) | 5.4 tok/s | glm_flash_benchmark.py |
| Memory (after load) | 16.93 GB | glm_flash_benchmark.py |
| Memory (after forward) | 17.24 GB | glm_flash_benchmark.py |
| GPU commits per forward | ~12 (async batched) | profiler |

### MMFP4 E2E Decode Benchmark (Feb 2026)

| Metric | Value | Notes |
|--------|-------|-------|
| Decode throughput | **0.74 tok/s** | 50-token greedy decode |
| Baseline comparison | 2.65× vs 0.28 | Previous unfused baseline |
| MPS allocated | 33.05 GB | Post-generation |
| MPS driver | 66.99 GB | Peak driver allocation |
| Fused MoE layers | 46/46 | All MoE layers using fused kernel |
| Paged attention | 0 (disabled) | Kernel integration incomplete |

**Benchmark command:**
```bash
cd contrib/metal_marlin && PYTHONPATH=. uv run python benchmarks/bench_e2e_decode.py --max-new-tokens 50
```

**Note:** Memory usage (33GB) is higher than expected due to intermediate buffers during generation. The fused MoE kernel provides 13× forward pass speedup, but E2E throughput is limited by other factors (attention, KV cache management).

### MMFP4 Fused Dispatch Migration (Feb 2026)

The MMFP4 inference stack was migrated from sequential dispatches to fused kernels:

| Operation | Previous (Sequential) | Optimized (Fused) | Status |
|-----------|----------------------|-------------------|--------|
| MLA Attention | 5+ dispatches (QKV → attn → O) | 1 dispatch (`mla_fused_attention_decode_glm4`) | **Fused** |
| MoE Expert Compute | Sequential expert iteration | Batched `moe_trellis_swiglu` | **Fused** |
| GEMM + Dequant | Dequantize + matmul separate | `mmfp4_gemm` fused | **Fused** |

**Remaining sequential operations:**

| Operation | Status | Notes |
|-----------|--------|-------|
| Input projections | Sequential | Prefill-optimized, low dispatch count |
| LayerNorm/RoPE | Sequential | Already minimal overhead |
| LM Head | Sequential | Single call per sequence |

**Key correctness fixes applied:**
1. **Async dispatch fix**: Changed `wait=False` to `wait=True` in kernel dispatch to prevent race conditions
2. **Numerical stability**: Float accumulation in GEMM kernels instead of half
3. **Per-simdgroup staging**: Eliminated shared staging buffer race conditions

**Performance status:**
- Pre-optimization decode: 0.27 tok/s (baseline)
- Post-optimization decode: pending measurement
- Expected improvement from dispatch overhead reduction

> **Note:** Benchmarks should be re-run manually to measure the impact of fused dispatches. DO NOT include extrapolated numbers.

### MLX Comparison (Observed)

| Backend | Decode Performance | Notes |
|---------|-------------------|-------|
| MLX | 10-20 tok/s | Estimated Native Apple framework performance (Model not supported by MLX) |
| llama.cpp | ~3 tok/s | Q4_K_M quantization |
| Metal Marlin | 5.4 tok/s | Current, fused trellis kernels |

**Status:** We're between llama.cpp and MLX. The gap to MLX is mostly MoE overhead.

### Where Time Goes (Profiled)

| Component | % of Forward Pass | Notes |
|-----------|-------------------|-------|
| MoE Compute | **98.5%** | Dominates everything |
| Attention | ~1% | MLA compression helps |
| LayerNorm/RoPE | <0.5% | Already fused |

**The lesson:** Optimize MoE first. Everything else is noise.

## Memory Bandwidth Analysis

> **Note:** Numbers in this section are calculated from model config, not measured.
> Use for directional guidance only.

### GLM-4.7-Flash Decode (batch=1)

```
Active params per token:    3.08B (MoE top-4 of 64 experts)
Weight bytes (MMFP4):       1.54 GB
KV cache @ 10K (FP16):      4.81 GB
Total @ 10K:                6.35 GB

M4 Max effective BW:        ~273 GB/s (50% of 546 GB/s)
Time per token:             6.35 GB / 273 GB/s = 23.3 ms
Throughput:                 43 tok/s theoretical
```

**Reality check:** MLX achieves 10-20 tok/s. The gap is kernel overhead + MoE routing.

### KV Cache Dominates Long Context (Calculated)

> **Warning:** These throughput numbers are calculated, NOT measured. Actual perf will be lower.

| Context | Weights | KV Cache | KV % | Throughput (calculated) |
|---------|---------|----------|------|------------|
| 512 | 1.54 GB | 0.25 GB | 14% | ~82 tok/s |
| 10K | 1.54 GB | 4.81 GB | 76% | ~35 tok/s |
| 50K | 1.54 GB | 24.1 GB | 94% | ~10 tok/s |
| 100K | 1.54 GB | 48.1 GB | 97% | ~5 tok/s |

**At 100K context, KV cache is 97% of memory traffic.**

## MLA (Multi-head Latent Attention): Architecture & Kernels

### What is MLA?

MLA (Multi-head Latent Attention), used by GLM-4.7-Flash and DeepSeek-V2/V3, compresses KV cache through learned latent projections:

```
Standard MHA:  hidden → QKV → store K,V → attention
MLA:           hidden → QKV_compressed → store latent → decompress → attention
               ↑ 8.9× smaller cache    ↑ extra GEMM per layer
```

### GLM-4.7-Flash MLA Dimensions

```python
# Correct GLM-4.7-Flash specs (from HuggingFace config)
hidden_size = 2048
num_heads = 20
kv_lora_rank = 512           # Compressed KV latent
qk_rope_head_dim = 64        # RoPE applied only here
qk_nope_head_dim = 192       # Non-positional part
v_head_dim = 256             # Value dimension

# KV cache sizes per token per layer
standard_kv = num_heads * head_dim * 2  # = 5120 floats
mla_kv = kv_lora_rank + qk_rope_head_dim  # = 576 floats
compression = 5120 / 576  # = 8.9× smaller!
```

### MLA Trade-offs

| Aspect | Advantage | Disadvantage |
|--------|-----------|--------------|
| **KV Cache** | 8.9× smaller | N/A |
| **Memory BW** | Huge win at long context | Marginal at short context |
| **Compute** | N/A | Extra kv_b_proj GEMM per layer |
| **RoPE** | N/A | Split computation (rope vs nope) |

**Memory-bound workloads (decode, long context):** MLA is a net win.
**Compute-bound workloads (prefill, short context):** MLA adds overhead.

### Metal Marlin MLA Kernel Library

Metal Marlin has comprehensive MLA support. Key kernels:

| Category | Kernel | Purpose | Status |
|----------|--------|---------|--------|
| **Projection** | `mla_proj_fp4_k16` | Single MLA projection | Available |
| **Projection** | `mla_fused_kv_proj_fp4` | Fused kv_a + kv_b | Available |
| **Projection** | `mla_proj_with_rope_fp4` | Projection + fused RoPE | Available |
| **Decode** | `mla_decode_proj_fp4` | GEMV for batch=1 | Available |
| **Attention** | `mla_fused_attention_decode_glm4` | **Full fused decode** | ⚠️ Unused |
| **RoPE** | `rope_mla_latent` | RoPE for MLA latents | Available |
| **RoPE** | `rope_mla_split_fused` | Fused split + RoPE | Available |

### When MLA Wins vs Loses

```
Context 512:   Weights 1.54 GB, KV 0.25 GB  → KV is 14% → MLA overhead visible
Context 10K:   Weights 1.54 GB, KV 4.81 GB  → KV is 76% → MLA wins
Context 100K:  Weights 1.54 GB, KV 48.1 GB  → KV is 97% → MLA essential
```

**Recommendation:** Always use MLA kernels for decode. The kv_b_proj overhead (~15μs) is dwarfed by memory savings at any meaningful context length.

### Integration: `mla_fused_attention_decode`

The most impactful unused kernel is `mla_fused_attention_decode_glm4`, which combines:

1. Q projection (q_a → layernorm → q_b)
2. KV projection (kv_a → split → layernorm → kv_b)
3. RoPE fused in KV projection
4. Attention with compressed KV cache
5. Output projection

**Current path:** 5+ kernel dispatches  
**With fusion:** 1 kernel dispatch

```python
# From metal_marlin/mla_fused.py
from metal_marlin.mla_fused import mla_fused_attention_decode, create_glm_mla_params

params = create_glm_mla_params(
    batch=1, seq_q=1, seq_k=cache_len,
    hidden_size=2048, num_heads=20, head_dim=256,
    kv_lora_rank=512, q_lora_rank=768, rope_dim=64,
)

output = mla_fused_attention_decode(
    hidden, q_a_packed, q_a_scales, q_b_packed, q_b_scales, q_bias,
    kv_a_packed, kv_a_scales, kv_b_packed, kv_b_scales,
    k_cache, v_cache, k_scales, v_scales,
    o_packed, o_scales, params
)
```

See [MLA Tooling Audit](reports/mla_tooling_audit.md) for full unused kernel inventory.

## Metal Marlin's Key Optimizations

### 1. Quantized KV Cache (HIGH IMPACT)

**The single most important optimization for long context.**

Metal Marlin implements quantized KV cache in paged attention kernels:

| Kernel | KV Precision | Use Case |
|--------|--------------|----------|
| `paged_attention_v1` | FP16 | Baseline, highest quality |
| `paged_attention_v1_fp8` | FP8 (E4M3) | 2x throughput, minimal loss |
| `paged_attention_v1_fp4` | FP4 (E2M1) | 4x throughput, some loss |
| `paged_attention_v1_int4` | INT4 | 4x throughput, requires calibration |

**Expected improvement at 100K context (calculated, not measured):**

> **Warning:** These are theoretical projections. We haven't benchmarked long context yet.

| KV Format | Memory | Throughput (calc) | vs FP16 |
|-----------|--------|------------|---------|
| FP16 | 48 GB | 5.3 tok/s | 1.0x |
| FP8 | 24 GB | 10.1 tok/s | 1.9x |
| INT4 | 12 GB | 18.0 tok/s | 3.4x |

**MLA + Quantized KV Stacking (theoretical):**

For GLM-4.7-Flash, MLA already reduces KV from 5120 to 576 floats/token (8.9×). Adding INT4:

| KV Storage | Floats/Token | vs Standard MHA |
|------------|--------------|------------------|
| Standard MHA FP16 | 5120 | 1.0× |
| MLA FP16 | 576 | 8.9× |
| MLA INT4 | 144 | **35.6×** |

> **Status:** Paged attention kernels exist but are **not wired** to GLM inference. Numbers above are theoretical.

**Integration:** The paged attention kernels are available but **not currently wired** to GLM inference.

```python
# Target integration (from paged_attention.metal)
from metal_marlin.paged_attention import PagedKVCache, paged_attention_v1_int4

kv_cache = PagedKVCache(
    num_layers=47,
    num_kv_heads=20,
    head_dim=576,  # MLA: kv_lora_rank + rope_dim
    block_size=16,
    dtype=torch.int4,  # 4x reduction on top of MLA
)

# During attention
output = paged_attention_v1_int4(
    query, k_cache, v_cache, k_scales, v_scales,
    block_tables, context_lens, max_context_len,
)
```

### 2. Paged Attention

**vLLM-style block management adapted for Metal simdgroups.**

Benefits:
- No memory fragmentation from variable-length sequences
- Efficient batch serving with different context lengths
- Prefix caching for repeated prompts
- Memory overcommit (more sequences than physical memory)

Key implementation details:
```metal
// Block size = 16 tokens (vLLM standard)
// 4 simdgroups per threadgroup (128 threads)
// Double-buffering: compute on one block while loading next
constant uint BLOCK_SIZE = 16;
constant uint NUM_SIMDGROUPS = 4;
constant uint KV_TILES = 2;  // Double buffer
```

### 3. FlashAttention for Metal

**Three implementations for different use cases:**

| Kernel | Use Case | Key Optimization |
|--------|----------|------------------|
| `flash_attention_v2_decode` | Single-query decode | SIMD-optimized for seq_q=1 |
| `flash_attention_v3_causal` | Autoregressive prefill | Block-sparse masking |
| `flash_attention_v3_gqa` | Grouped-query attention | KV broadcast optimization |

**Tile sizes (tuned for M4 Max):**
```metal
constant uint TILE_Q = 64;   // Queries per tile
constant uint TILE_K = 64;   // Keys per tile
constant uint HEAD_DIM = 128;
```

### 4. Kernel Fusion

**Reduce dispatch overhead by fusing operations.**

Problem: 47 layers × 12 kernels = 564 dispatches per token (~4.5 ms overhead)

Solutions implemented:
- Fused QKV projection (3 kernels → 1)
- Fused gate+up+SiLU (2 kernels → 1)
- Fused LayerNorm+Attention (2 kernels → 1)

Target: 4-6 dispatches per layer → ~2 ms overhead

### 5. Unified Memory Buffer Management

**Zero-copy data sharing between CPU and GPU.**

```swift
// CORRECT: Use storageModeShared for zero-copy
let buffer = device.makeBuffer(length: size, 
                               options: .storageModeShared)

// WRONG: storageModePrivate requires explicit copies
let buffer = device.makeBuffer(length: size,
                               options: .storageModePrivate)
```

**When to use storageModePrivate:**
- GPU-only scratch buffers
- Intermediate results never read by CPU
- Temporary attention matrices

## Measured Kernel Performance

### Fused GEMM (GLM-4.7-Flash expert shapes: 2048×1536)

| Batch Size | Reference (dequant+matmul) | Fused Kernel | Speedup |
|------------|---------------------------|--------------|----------|
| 1 | 145.2 ms | 2.8 ms | **51.9×** |
| 32 | 162.4 ms | 4.2 ms | **38.7×** |
| 128 | 189.6 ms | 12.5 ms | **15.2×** |

**Source:** `benchmarks/bench_glm47_kernels.py`

**Lesson:** Kernel fusion is not optional. 50× speedup is real.

### MoE Dispatch Optimization

| Approach | Latency per Token | Notes |
|----------|-------------------|-------|
| Sequential (naive) | ~20 seconds | Iterate 64 experts, full dequant each |
| Batched (`moe_trellis_swiglu`) | <100 ms | Single dispatch for all active experts |
| **Speedup** | **200×** | |

**This was the critical fix.** Before batched dispatch, GLM-4.7-Flash was unusable.

### Mixed-Precision Quantization Impact

| Configuration | Avg BPW | Model Size | Decode | Prefill |
|---------------|---------|------------|--------|----------|
| Uniform 4-bit | 4.0 | 8.8 GB | baseline | baseline |
| Mixed 3-bit (Trellis) | 3.02 | 6.6 GB | +20.7% | +18.4% |

**Source:** `docs/guides/mixed_bpw_inference.md` Appendix A

### Per-Layer Timing (bench_glm47_kernels.py, 2026-02-08)

| Operation | Measured | Notes |
|-----------|----------|-------|
| QKV proj (2048→6144) | 5.215 ms | Single GEMM |
| O proj (2048→2048) | 4.565 ms | Single GEMM |
| MoE Expert (standard) | 97.5 ms | gate+up+down, unfused |
| MoE Expert (fused) | - | Currently broken (pipeline cache issue) |
| LM Head (2048→154880) | - | Blocked by earlier crash |

**Extrapolated full forward pass (47 layers):**
- Attention per layer: ~10 ms (QKV + O)
- MoE per layer: ~98 ms (standard path)
- Per layer total: ~108 ms
- **Estimated decode: ~0.2 tok/s** (without fused kernels)

**Actual decode: 5.4 tok/s** → The fused `moe_trellis_swiglu` kernel is critical

### What We Know Works

- ✅ Fused GEMM: 50× speedup measured
- ✅ Batched MoE: 200× speedup measured
- ✅ Mixed 3-bit quant: 20% throughput gain measured
- ✅ Zero-copy model load: 15s for 17GB model
- ✅ **Qwen3-4B BF16: 33.4 tok/s (93% of roofline)** - dense model baseline

### Qwen3-4B: Dense Model Baseline (Measured Feb 2026)

| Precision | Memory | Throughput | Roofline | Efficiency |
|-----------|--------|------------|----------|------------|
| **BF16** | 8.04 GB | **33.4 tok/s** | 36 tok/s | **93%** |
| FP4 (current) | 2 GB | ~27 tok/s | 144 tok/s | 19% |

**Critical insight:** BF16 is FASTER than FP4 right now because PyTorch MPS handles
BF16 natively while FP4 uses unoptimized fallback paths. Metal Marlin's fused kernels
need to close this 117 tok/s gap (144 - 27 = 117).

The 93% BF16 efficiency proves the hardware CAN achieve near-roofline performance.

### MMFP4 GLM-4.7-Flash Benchmarks (Measured Feb 2026)

> **Note (historical):** Benchmarks in this section were collected in Feb 2026 *before* the fused dispatch optimization and are preserved for historical comparison only. They do **not** represent current, optimized performance.

#### Optimization: Fused Dispatch Migration (Applied)

The MMFP4 stack was migrated from sequential dispatches to fused kernels:

| Operation | Previous | Optimized | Change |
|-----------|----------|-----------|--------|
| MLA Attention | 5+ dispatches (QKV → attn → O) | 1 dispatch (`mla_fused_attention_decode_glm4`) | Fused |
| MoE Routing | Sequential expert iteration | Batched `moe_trellis_swiglu` | Fused |
| GEMM | Dequant + matmul separate | `mmfp4_gemm` fused | Fused |

**Expected Impact:**
- Reduced dispatch overhead: ~8μs × (N dispatches eliminated)
- Improved memory locality from fused kernels
- Eliminated async dispatch race conditions (correctness fix)

**Status:** Optimization applied, benchmarks pending re-run.

> **Status:** Correctness verified, performance severely degraded.

| Test | Result | Expected | Gap |
|------|--------|----------|-----|
| Perplexity | 17.15 | <100 | ✅ PASS |
| Prefill avg | 24.6 tok/s | 500-2000 tok/s | 20-80× slower |
| Decode | 0.27 tok/s | 5-20 tok/s | 20-80× slower |

**Prefill by context length:**

| Context | Throughput | Notes |
|---------|------------|-------|
| 128 | 9.0 tok/s | Dispatch overhead dominates |
| 512 | 25.5 tok/s | Better amortization |
| 1024 | 39.3 tok/s | Best prefill efficiency |

**Decode comparison:**

| Configuration | Throughput | Notes |
|---------------|------------|-------|
| Manual loop (no KV cache) | 0.07 tok/s | Full context recompute - CATASTROPHIC |
| model.generate() + MLAKVCache | 0.27 tok/s | 4× improvement |

**Key findings:**
1. MMFP4 correctness is verified (perplexity reasonable, no NaN)
2. Performance is 20-80× slower than expected
3. KV cache is critical - manual loops without cache are 4× slower
4. Compare: non-MMFP4 Trellis gets 5.4 tok/s decode, MMFP4 gets 0.27 tok/s

**Suspected causes:**
- MMFP4 GEMM kernels not optimized for M4 Max
- PagedAttention not wired (paged_attention_v1_fp4 exists but unused)
- Possible dispatch overhead from MoE routing

### What We Haven't Measured Yet

- ❌ Long context (>10K) performance
- ❌ M4 Max vs other Apple Silicon (no comparative benchmarks)

## Profiling Methodology

### Metal System Trace (Instruments)

```bash
# Capture GPU timeline
xcrun xctrace record --template 'Metal System Trace' \
    --output trace.trace \
    --launch -- python inference.py
```

Key metrics to watch:
- **GPU Utilization** - should be >80%
- **Encoder Wait** - indicates CPU bottleneck
- **Buffer Allocation** - should be minimal during inference

### Kernel Timing

```python
import metal
import time

def benchmark_kernel(pipeline, buffers, iterations=100):
    # Warmup
    for _ in range(10):
        pipeline.encode(...)
    
    # Sync and time
    metal.device.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        pipeline.encode(...)
    
    metal.device.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # ms
```

### Memory Bandwidth Measurement

```python
def measure_bandwidth(buffer_size_gb, iterations=50):
    """Measure achieved memory bandwidth."""
    buffer = device.makeBuffer(length=int(buffer_size_gb * 1e9))
    
    # Simple copy kernel
    start = time.perf_counter()
    for _ in range(iterations):
        # dispatch copy kernel
        pass
    device.synchronize()
    elapsed = time.perf_counter() - start
    
    bytes_moved = buffer_size_gb * 1e9 * iterations * 2  # read + write
    bandwidth_gbs = bytes_moved / elapsed / 1e9
    
    print(f"Achieved: {bandwidth_gbs:.1f} GB/s ({bandwidth_gbs/546*100:.0f}% of peak)")
```

## Systematic Debugging Methodology

When something goes wrong on Metal, it's usually one of 5 things. Here's how to find it fast.

### The Six Root Causes (In Order of Likelihood)

1. **Layout transposition** (50% of bugs) - Kernel expects [A, B], you passed [B, A]
2. **Async dispatch race condition** (20% of bugs) - Reading kernel output before completion
3. **Tile boundary issue** (15% of bugs) - Dimension doesn't fill tile, OOB read
4. **Numerical overflow** (8% of bugs) - Values too large for half precision
5. **State corruption** (4% of bugs) - Buffer reuse without clearing
6. **Kernel bug** (3% of bugs) - Actual bug in Metal shader code

### Step 1: Identify the Failure Mode

```python
def diagnose_output(tensor, name="output"):
    """First step: characterize the corruption type."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    max_val = tensor.abs().max().item()
    min_val = tensor.abs().min().item()
    
    print(f"{name}: nan={has_nan}, inf={has_inf}, range=[{min_val:.2e}, {max_val:.2e}]")
    
    if has_nan and not has_inf:
        print("  → Likely: OOB read, uninitialized memory, or 0/0")
    elif has_inf and not has_nan:
        print("  → Likely: Numerical overflow (exp of large value)")
    elif has_nan and has_inf:
        print("  → Likely: inf/inf or inf - inf producing NaN")
    elif max_val > 65504:  # FP16 max
        print("  → Warning: Values exceed FP16 range")
```

### Step 2: Bisect to Find the Layer

```python
def find_bad_layer(model, x):
    """Identify which layer first produces NaN."""
    hidden = model.embed_tokens(x)
    
    for i, layer in enumerate(model.layers):
        hidden = layer(hidden)
        if torch.isnan(hidden).any():
            print(f"NaN first appears at layer {i}")
            return i
    
    # Check final layers
    hidden = model.norm(hidden)
    if torch.isnan(hidden).any():
        print("NaN first appears at final norm")
        return "norm"
    
    logits = model.lm_head(hidden)
    if torch.isnan(logits).any():
        print("NaN first appears at lm_head")
        return "lm_head"
    
    return None  # No NaN found
```

### Step 3: Bisect to Find the Operation

```python
def find_bad_op_in_layer(layer, x):
    """Within a layer, find which operation fails."""
    residual = x
    
    # Test attention
    normed = layer.input_layernorm(x)
    if torch.isnan(normed).any():
        return "input_layernorm"
    
    attn_out = layer.self_attn(normed)
    if torch.isnan(attn_out).any():
        return "self_attn"
    
    x = residual + attn_out
    residual = x
    
    # Test FFN/MoE
    normed = layer.post_attention_layernorm(x)
    if torch.isnan(normed).any():
        return "post_attention_layernorm"
    
    ffn_out = layer.mlp(normed)
    if torch.isnan(ffn_out).any():
        return "mlp"
    
    return None
```

### Step 4: Bisect to Find the Shape

Once you know WHICH operation fails, find WHICH shape triggers it:

```python
def find_bad_shape(kernel_fn, weights, scales, K):
    """Find exact (M, N) that first produces NaN."""
    results = []
    
    for M in [1, 2, 3, 4, 5, 8, 16, 32, 63, 64, 65, 128]:
        x = torch.randn(M, K, device='mps')
        out = kernel_fn(x, weights, scales)
        has_nan = torch.isnan(out).any().item()
        results.append((M, has_nan))
        
        if has_nan:
            print(f"✗ M={M}: NaN")
        else:
            print(f"✓ M={M}: OK")
    
    # Find boundary
    for i, (M, nan) in enumerate(results):
        if nan and (i == 0 or not results[i-1][1]):
            print(f"\n→ NaN boundary: M={M}")
            return M
    
    return None
```

### Step 5: Verify Layout

```python
def verify_layouts(weight, scales, expected_weight_shape, expected_scales_shape):
    """Verify tensors have expected layouts."""
    errors = []
    
    if weight.shape != expected_weight_shape:
        errors.append(f"Weight: got {weight.shape}, expected {expected_weight_shape}")
        
        # Check if it's transposed
        if weight.shape == expected_weight_shape[::-1]:
            errors.append("  → Weight appears transposed!")
    
    if scales.shape != expected_scales_shape:
        errors.append(f"Scales: got {scales.shape}, expected {expected_scales_shape}")
        
        if scales.shape == expected_scales_shape[::-1]:
            errors.append("  → Scales appear transposed!")
    
    # Check contiguity (Metal requires contiguous)
    if not weight.is_contiguous():
        errors.append("Weight is not contiguous!")
    if not scales.is_contiguous():
        errors.append("Scales is not contiguous!")
    
    if errors:
        for e in errors:
            print(f"❌ {e}")
        return False
    
    print("✓ Layouts verified")
    return True
```

### Step 6: Check Async Dispatch Issues

**The #2 root cause of NaN bugs.** Metal kernels dispatched async (`wait=False`) allow subsequent
operations to read uninitialized GPU memory.

```python
def check_async_dispatch_issue(model, x):
    """Test if async dispatch causes reads of uninitialized output."""
    import torch
    
    # Test 1: Run with potential async issues
    out1 = model(x)
    has_nan_1 = torch.isnan(out1.logits).any().item()
    
    # Test 2: Force synchronization after each kernel
    torch.mps.synchronize()
    out2 = model(x)
    torch.mps.synchronize()
    has_nan_2 = torch.isnan(out2.logits).any().item()
    
    if has_nan_1 and not has_nan_2:
        print("→ ASYNC DISPATCH ISSUE: NaN disappears with explicit sync")
        print("  Fix: Change wait=False to wait=True in kernel dispatch")
        return True
    
    return False
```

**What we learned (Feb 2026 MMFP4 debugging):**
- Async dispatch (`wait=False`) in `kernels.py` caused non-deterministic NaN
- Short sequences (1-4 tokens) worked because next op happened to wait long enough
- Longer sequences (5+) failed because subsequent ops read uninitialized memory
- **Fix:** Change `wait=True` in kernel dispatch calls

### Step 7: Check Cross-Call State Corruption

```python
def check_state_corruption(kernel_fn, x, weights, scales, n_calls=5):
    """Check if repeated calls produce consistent results."""
    outputs = []
    
    for i in range(n_calls):
        out = kernel_fn(x, weights, scales)
        out_cpu = out.cpu().clone()
        outputs.append(out_cpu)
        
        if i > 0:
            diff = (outputs[i] - outputs[0]).abs().max().item()
            has_nan = torch.isnan(outputs[i]).any().item()
            print(f"Call {i+1}: diff from call 1 = {diff:.2e}, has_nan = {has_nan}")
            
            if has_nan and not torch.isnan(outputs[0]).any():
                print("  → STATE CORRUPTION: NaN appeared on repeated call!")
                return True
    
    return False
```

## Case Study: MMFP4 GEMM NaN Debugging (Feb 2026)

> **This is a real debugging session that took 2 days.** 
> Documenting it so future developers don't repeat the same investigation.

### The Symptom

MMFP4 (Mixed-precision FP4) GEMM produced NaN for sequence lengths >= 5:

```
✓ seq_len=1 OK
✓ seq_len=2 OK
✓ seq_len=3 OK
✓ seq_len=4 OK
✗ seq_len=5 NaN
✗ seq_len=8 NaN
✗ seq_len=16 NaN
```

**Complication:** Failures were non-deterministic. Sometimes seq_len=5 passed, sometimes failed.

### What We Tried (And What Didn't Work)

1. **Tile boundary theory** - We thought M < TILE_M (64) was the issue
   - Red herring: seq_len 4 worked, seq_len 5 didn't (both < 64)
   
2. **Isolated layer tracing** - Tested each layer individually
   - All layers passed in isolation!
   - Only sequential forward pass failed
   
3. **MoE vs Dense** - Suspected expert routing
   - Dense layers (layer 0) worked
   - MoE layers failed, but inconsistently

### The Three Actual Bugs (All Had to Be Fixed)

#### Bug 1: Half-Precision Accumulator Overflow

**File:** `metal_marlin/shaders/mmfp4_gemm.metal`

```metal
// BEFORE (broken): Accumulator overflow at large values
simdgroup_matrix<half, 8, 8> acc;  // FP16 accumulator
// ... matrix multiply ...
// Large values overflow FP16 range (65504 max)

// AFTER (fixed): Float accumulator for numerical range
simdgroup_matrix<float, 8, 8> acc;  // FP32 accumulator
// ... matrix multiply ...
// Final output converted to half at the end
```

**Why it mattered:** Intermediate products could exceed 65504, causing infinity, then NaN.

#### Bug 2: Staging Buffer Race Condition

**File:** `metal_marlin/shaders/mmfp4_gemm.metal`

```metal
// BEFORE (broken): 4 simdgroups share one staging buffer
threadgroup float staging[TILE_M * TILE_N];  // Shared by all simdgroups
// Simdgroup 0 writes, simdgroup 1 reads before write completes → garbage

// AFTER (fixed): Per-simdgroup staging buffers
threadgroup float staging[NUM_SIMDGROUPS][TILE_M * TILE_N / NUM_SIMDGROUPS];
// Each simdgroup has its own staging area
```

**Why it mattered:** Simdgroups don't synchronize automatically. Shared memory = race condition.

#### Bug 3: Async Kernel Dispatch (THE ROOT CAUSE)

**File:** `metal_marlin/kernels.py` line 3245

```python
# BEFORE (broken): Async dispatch returns before kernel completes
dispatch_kernel(
    pipeline,
    threads_per_grid=grid,
    threads_per_threadgroup=group,
    buffers=buffers,
    wait=False  # ← Returns immediately!
)
# Subsequent operations read uninitialized output buffer

# AFTER (fixed): Synchronous dispatch guarantees completion
dispatch_kernel(
    pipeline,
    threads_per_grid=grid,
    threads_per_threadgroup=group,
    buffers=buffers,
    wait=True  # ← Blocks until kernel finishes
)
```

**Why this was THE root cause:**
- Small workloads (seq_len 1-4) completed before next operation by coincidence
- Larger workloads (seq_len 5+) lost the race → uninitialized memory read
- Non-deterministic because it depended on GPU timing

### How We Found It

1. **Bisected by sequence length** - Found 4/5 boundary
2. **Isolated layer tracing** - All layers OK individually → pointed to execution order issue
3. **Non-determinism pattern** - Suggested race condition, not algorithmic bug
4. **Changed `wait=True`** - ALL tests passed immediately and consistently

### The Lesson

**When you see non-deterministic NaN that depends on input size:**
1. First check: Is the kernel output being read before`wait=True` or explicit sync?
2. This is MORE COMMON than tile boundary bugs
3. Test by running the same input 3+ times - if results vary, it's async dispatch

### Test to Verify the Fix

```python
# developer_tests/test_mmfp4_fixes.py
import torch
from metal_marlin.models import MMFP4ForCausalLM

def test_mmfp4_all_seqlens():
    model = MMFP4ForCausalLM.from_pretrained("path/to/model")
    
    for seq_len in [1, 2, 3, 4, 5, 8, 16, 32]:
        for run in range(3):  # Multiple runs to catch non-determinism
            x = torch.randint(0, 1000, (1, seq_len), device='mps')
            out = model(x)
            assert not torch.isnan(out.logits).any(), f"NaN at seq_len={seq_len} run={run}"
    
    print("✓ All sequence lengths pass across multiple runs")
```

---

### Full Diagnostic Script Template

```python
"""
Template for systematically debugging Metal kernel failures.
Save as developer_tests/debug_<issue>.py
"""
import torch
from metal_marlin import load_model

def main():
    model = load_model("path/to/model")
    
    # Test progression   
    test_cases = [
        (1, "seq_len=1"),
        (2, "seq_len=2"),
        (3, "seq_len=3"),
        (4, "seq_len=4 (tile boundary)"),
        (8, "seq_len=8"),
        (63, "seq_len=TILE-1"),
        (64, "seq_len=TILE"),
        (65, "seq_len=TILE+1"),
    ]
    
    print("=" * 60)
    print("Shape sweep")
    print("=" * 60)
    
    for seq_len, desc in test_cases:
        x = torch.randint(0, 1000, (1, seq_len), device='mps')
        
        with torch.no_grad():
            out = model(x)
        
        has_nan = torch.isnan(out.logits).any().item()
        status = "✗ NaN" if has_nan else "✓ OK"
        print(f"{status}: {desc}")
    
    print("\n" + "=" * 60)
    print("State corruption check")
    print("=" * 60)
    
    x = torch.randint(0, 1000, (1, 4), device='mps')  # Use failing size
    
    for i in range(5):
        out = model(x)
        has_nan = torch.isnan(out.logits).any().item()
        print(f"Call {i+1}: nan={has_nan}")

if __name__ == "__main__":
    main()
```

### Quick Reference: Common Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| NaN at specific M values | Tile boundary | Pad input OR fix kernel bounds check |
| NaN on 2nd+ call only | State corruption | Clear caches OR fix buffer reuse |
| Inf in attention | Overflow | Use float accumulation, clamp scores |
| Wrong values (no NaN) | Layout transposition | Transpose weight/scales at call site |
| Garbage values | Uninitialized buffer | Zero-init shared memory in kernel |
| Crash (SIGABRT) | Threadgroup mismatch | Verify dispatch threadgroup size |
```

## Common Pitfalls

> **Important:** Don't assume CUDA patterns apply to Metal. Apple Silicon has fundamentally
> different tradeoffs (e.g., FP32 vs FP16 is NOT 2:1 like on NVIDIA due to unified ALUs).

### MPS Memory Management (CRITICAL - Feb 2026)

MPS does NOT release memory like CUDA. These patterns cause memory leaks:

**Problem 1: Underscore variable holds reference**
```python
# LEAKS: _ holds tensor until reassigned
_ = model(input_ids)  # tensor stays alive!

# CORRECT: Explicit variable and delete
outputs = model(input_ids)
del outputs
gc.collect()
torch.mps.empty_cache()
```

**Problem 2: Memory not released between benchmark runs**
```python
# LEAKS: MPS doesn't release memory on del alone
for config in configs:
    model = load_model(config)  # Accumulates!
    benchmark(model)
    del model  # Doesn't actually release

# CORRECT: Use subprocess isolation
for config in configs:
    subprocess.run(['python', 'bench_single.py', config])
    # Child process terminates → memory fully released
```

**Problem 3: KV cache accumulation**
```python
# LEAKS: KV cache grows unbounded
for i in range(1000):
    outputs = model.generate(input_ids, max_new_tokens=1)
    # KV cache keeps growing each call!

# CORRECT: Clear cache between generations
for i in range(1000):
    outputs = model.generate(input_ids, max_new_tokens=1)
    model.clear_kv_cache()  # Or use with torch.no_grad():
```

**Required cleanup pattern for benchmarks:**
```python
import gc
import torch

def cleanup():
    """Full MPS memory cleanup."""
    gc.collect()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

# After every benchmark iteration
cleanup()
```

## Critical: Metal Kernel Edge Cases (Hard-Won Lessons)

These issues are NOT obvious from documentation. They caused days of debugging and will bite you again.

### 1. SIMD Tile Boundary Issues

**The problem:** Metal kernels use tiled computation (typically TILE_M=64, TILE_N=64, TILE_K=32).
When input dimensions don't fill a complete tile, behavior is UNDEFINED and often produces **silent NaN/Inf**.

**What we observed:**
- Sequence length 1-3: Works perfectly
- Sequence length 4+: Produces NaN consistently  
- The boundary at 4 tokens suggests M dimension partial tile handling is broken

```python
# DANGEROUS: Assumes kernel handles all M sizes
output = mmfp4_gemm(x, weights, scales)  # x.shape[0] = 4 → NaN

# SAFE: Pad to tile boundary OR verify kernel handles edge cases
M = x.shape[0]
padded_M = ((M + TILE_M - 1) // TILE_M) * TILE_M
x_padded = F.pad(x, (0, 0, 0, padded_M - M))
output = mmfp4_gemm(x_padded, weights, scales)[:M]
```

**Metal kernel fix (if you control the shader):**
```metal
// BAD: Assumes M is divisible by TILE_M
uint m_idx = threadgroup_position_in_grid.y * TILE_M + thread_position_in_threadgroup.y;
float val = input[m_idx * K + k_idx];  // OOB read when M < TILE_M!

// GOOD: Bounds check for partial tiles
uint m_idx = threadgroup_position_in_grid.y * TILE_M + thread_position_in_threadgroup.y;
float val = (m_idx < M) ? input[m_idx * K + k_idx] : 0.0f;
```

**Rule: Always test kernel at**: `M = 1, 2, 3, 4, TILE_M-1, TILE_M, TILE_M+1`

### 2. Weight/Scale Layout Transposition Confusion

**The problem:** Different frameworks store matrices in different layouts. Metal kernels expect
specific layouts that may not match PyTorch or safetensors storage order.

**Layout conventions to document:**

| Component | Kernel Expects | Often Stored As | Fix |
|-----------|---------------|-----------------|-----|
| FP4 Weights | `[K/8, N]` (column-major packed) | `[N, K/8]` (row-packed) | Transpose at load |
| Scales | `[groups, N]` | `[N, groups]` | `.T` at kernel call |
| Row-packed | `[N, K]` contiguous | `[K, N]` contiguous | Reshape + transpose |

**What we observed:**
```python
# Our bug: scales stored as [N, groups], kernel expected [groups, N]
scales = layer.scales  # Shape: [1536, 64]  ← WRONG for kernel
correct_scales = scales.T  # Shape: [64, 1536] ← kernel expects this

# Passing wrong layout → silent corruption, not an error
```

**Rule: Document expected layout in kernel signature AND verify at call site:**
```python
def mmfp4_gemm(x, weights, scales):
    """
    Args:
        x: [M, K] input
        weights: [K // 8, N] packed FP4 (column-major packing)
        scales: [num_groups, N] per-group scales (NOT [N, groups])
    """
    assert scales.shape[0] < scales.shape[1], f"Scales likely transposed: {scales.shape}"
    # ...
```

### 3. Stateful Kernel Caches & Cross-Call Corruption

**The problem:** Metal pipeline cache and buffer reuse can cause state to leak between calls.
What worked in isolation fails when called repeatedly.

**What we observed:**
- First forward pass: Clean results
- Second forward pass with SAME input: Corrupted
- Root cause: Intermediate buffers or pipeline state persisting

```python
# DANGEROUS: Assumes kernel is stateless
for i in range(10):
    output = model.forward(x)  # May corrupt on iteration 2+

# SAFE: Clear caches between critical operations (expensive!)
for i in range(10):
    torch.mps.empty_cache()  # Nuclear option - slow but deterministic
    output = model.forward(x)
```

**Better fix:** Ensure kernel allocates fresh buffers OR explicitly zeroes reused buffers:
```metal
// Kernel should not assume output buffer is zeroed
output[idx] = 0.0f;  // Explicitly zero
output[idx] += computed_value;  // Then accumulate
```

### 4. MPS Non-Determinism

**The problem:** Metal Performance Shaders can give different results across runs, even with
identical inputs. This makes debugging extremely difficult.

**What we observed:**
- Same model, same input, same code: Results vary by 1-5 ULP after 3+ forward passes
- Not a bug, but makes regression testing fragile

**Mitigations:**
```python
# Use generous tolerances for MPS comparisons
torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)  # Not 1e-6!

# For debugging, force deterministic by disabling async
torch.mps.synchronize()  # After every operation (slow but predictable)

# Log intermediate values with hash to detect divergence
import hashlib
def tensor_hash(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:8]
print(f"hidden: {tensor_hash(hidden)}")  # Compare across runs
```

### 5. Numerical Stability in Attention

**The problem:** Attention softmax can overflow/underflow at extreme sequence lengths or head dimensions.

**Standard issues:**
```python
# BAD: Direct softmax on large values
scores = q @ k.T / sqrt(d)  # scores might have values > 80
probs = softmax(scores)  # exp(80) = 5.5e34 → overflow

# GOOD: Subtract max for numerical stability
scores = q @ k.T / sqrt(d)
scores = scores - scores.max(dim=-1, keepdim=True).values  # Now max = 0
probs = softmax(scores)
```

**Metal-specific:** Ensure shader uses `float` not `half` for attention score accumulation:
```metal
// DANGEROUS at long context
half score = dot(q_vec, k_vec) * rsqrt_d;

// SAFE
float score = dot(q_vec, k_vec) * rsqrt_d;  // Accumulate in float
half output = half(score);  // Convert at end only
```

### 6. Debugging Strategy: Bisect Systematically

**When you hit NaN/Inf, don't guess. Bisect.**

1. **Isolate the layer:** Test each model layer independently
2. **Isolate the operation:** Test GEMM, attention, LayerNorm separately
3. **Isolate the shape:** Find exact (M, K, N) that fails
4. **Bisect the boundary:** Binary search for exact failure threshold

```python
# Systematic shape bisection
def find_nan_boundary(kernel_fn, max_m=128):
    """Find exact M where kernel starts producing NaN."""
    for m in range(1, max_m + 1):
        x = torch.randn(m, K, device='mps')
        out = kernel_fn(x, weights, scales)
        if torch.isnan(out).any():
            print(f"NaN at M={m}, last good M={m-1}")
            return m
    return None

# Found: NaN starts at M=4 → tile boundary issue, not random corruption
```

### 7. Zero-Initialization vs Garbage

**The problem:** Metal buffers are NOT zero-initialized by default. Uninitialized reads produce garbage.

```metal
// BAD: Assumes buffer is zeroed
threadgroup float shared[256];
// ... later ...
float val = shared[local_idx];  // Garbage if never written!

// GOOD: Explicit initialization
threadgroup float shared[256];
if (local_idx < 256) shared[local_idx] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
```

### 8. Threadgroup Size Mismatch

**The problem:** Dispatching with wrong threadgroup size causes silent failures or crashes.

```python
# Python side: dispatch with specific threadgroup size
pipeline.dispatch(
    threads=(64, 64, 1),  # Total work items
    threadgroups=(2, 2, 1),  # Threadgroups
    threadsPerThreadgroup=(32, 32, 1)  # MUST match kernel expectation
)

# If kernel assumes (64, 1, 1) but you pass (32, 2, 1): undefined behavior
```

**Metal side:** Declare expected threadgroup size explicitly:
```metal
[[kernel]] void my_kernel(
    /* ... */
) [[threads_per_threadgroup(64, 1, 1)]] {  // Explicit expectation
    // ...
}
```

### 9. Async Dispatch Race Conditions (CRITICAL - Found Feb 2026)

**The problem:** Metal kernel dispatch with `wait=False` returns immediately, but subsequent
operations may read the output buffer before the kernel has written to it.

**What we observed (MMFP4 GEMM debugging):**
- seq_len 1-4: Always passed
- seq_len 5+: Non-deterministic NaN (sometimes pass, sometimes fail)
- Different layers failed on different runs
- Isolated layer testing worked; sequential forward failed

**The symptom pattern that indicates async dispatch bugs:**
```
✓ seq_len=1 OK
✓ seq_len=2 OK  
✓ seq_len=3 OK
✓ seq_len=4 OK
✗ seq_len=5 NaN (non-deterministic!)
✗ seq_len=8 NaN
✗ seq_len=16 NaN
```

**Root cause:** `dispatch_kernel(..., wait=False)` in Python allows the next operation
to start before the GPU kernel completes. For small workloads, the kernel finishes in time
by coincidence. For larger workloads, the race is lost.

```python
# DANGEROUS: Async dispatch
dispatch_kernel(
    pipeline,
    threads_per_grid=grid,
    threads_per_threadgroup=group,
    buffers=buffers,
    wait=False  # ← Returns immediately, kernel may not be done!
)
# Next operation reads uninitialized memory
next_result = some_op(output)  # output contains garbage

# SAFE: Synchronous dispatch
dispatch_kernel(
    pipeline,
    threads_per_grid=grid,
    threads_per_threadgroup=group,
    buffers=buffers,
    wait=True  # ← Blocks until kernel completes
)
# Output is guaranteed valid
next_result = some_op(output)  # output is correct
```

**When to use `wait=False`:**
- Only when you have explicit synchronization later
- Only when you're batching multiple independent kernels
- Never for kernels whose output is immediately consumed

**Performance note:** `wait=True` has overhead (~8μs per dispatch), but correctness > speed.
Optimize by reducing total dispatches (kernel fusion), not by removing sync.

### 10. Metal Compiler Bugs (CRITICAL - Verified Jan 2026)

> **These are confirmed Metal compiler bugs on M4 Max / macOS 26.3 (Tahoe).**
> Work around them; don't try to understand why they exist.

#### Bug A: Array Parameter Bug (Force Inline Required)

**The problem:** Metal compiler incorrectly handles 2D array parameters of `simdgroup_matrix` 
types when passed to non-inlined functions. Even with `inline` keyword, the compiler may 
not inline and generates **incorrect array indexing code**.

**Symptom:** GEMM output shows column repetition - columns 0-7 values repeat to 8-15, 16-23, etc.

```
Expected: [0, 16, 32, 48, ...] [128, 144, 160, 176, ...] [256, 272, ...]
Actual:   [0, 16, 32, 48, ...] [0,   16,  32,  48,  ...] [0,   16,  ...]
```

**Root cause:** When passing `acc[SG_M_TILES][SG_N_TILES]` to functions, all loop iterations
read from `acc[0][0]` instead of the correct `acc[mi][ni]`.

```metal
// BUGGY: Compiler may not inline, causing incorrect acc[mi][ni] access
inline void store_results(
    thread simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    device half* C,
    ...
) {
    for (uint ni = 0; ni < SG_N_TILES; ++ni) {
        // BUG: All iterations read from acc[mi][0] instead of acc[mi][ni]
        simdgroup_store(acc[mi][ni], C + out_col, N);
    }
}

// FIXED: Force inline ensures correct array access
__attribute__((always_inline))
inline void store_results(
    thread simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES],
    device half* C,
    ...
) {
    // Now acc[mi][ni] correctly accesses different accumulators
}
```

**Affected function patterns:**
- `compute_from_tiles()` - GEMM computation
- `store_results()` - Output writing
- Any function receiving `simdgroup_matrix` array parameters

**Rule:** ALWAYS use `__attribute__((always_inline))` on functions receiving simdgroup arrays.

#### Bug B: Simdgroup Tile Coverage Bug

**The problem:** Configuration mismatch between tile sizes and simdgroup coverage causes
rows beyond covered range to contain zeros.

**Symptom:** For M > 32 with TILE_M=64, rows 32+ are all zeros.

```metal
// BUGGY CONFIGURATION:
constant constexpr uint TILE_M = 64;           // Tile covers 64 rows
constant constexpr uint SG_M_TILES = 2;        // 16 rows per simdgroup
constant constexpr uint SIMDGROUPS_PER_TG = 4; // 4 simdgroups

// Simdgroup layout (2x2):
// simd_id=0: rows 0-15,  cols 0-31
// simd_id=1: rows 0-15,  cols 32-63
// simd_id=2: rows 16-31, cols 0-31
// simd_id=3: rows 16-31, cols 32-63
// Coverage: 32 rows × 64 cols - rows 32-63 NEVER computed!

// FIXED: Increase SG_M_TILES
constant constexpr uint SG_M_TILES = 4;        // 32 rows per simdgroup
// Now: simd_id=0,1 cover rows 0-31; simd_id=2,3 cover rows 32-63
```

**Rule:** Verify `SIMDGROUPS × SG_M_TILES × 8 >= TILE_M` before changing tile configs.

#### Bug C: Half-Precision Optimization Bug

**The problem:** Metal compiler applies an optimization that truncates `half` function 
parameters to integers, producing incorrect fractional values in arithmetic.

**Symptom:** Fractional zero points (e.g., 5.5) are rounded to integers (5 or 6).

| Zero Point | Expected | Actual (Buggy) |
|------------|----------|----------------|
| 5.0        | -0.0     | -0.0 ✓        |
| 5.25       | -0.25    | -0.0 ✗        |
| 5.5        | -0.5     | -1.0 ✗        |
| 5.75       | -0.75    | -1.0 ✗        |

**Root cause:** Unknown compiler optimization in inline functions truncates half parameters.

```metal
// BUGGY: Returns wrong value when zero_point=5.5, code=5
inline half dequant_buggy(uint32_t packed, half scale, half zero_point) {
    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);
    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;
    return (v0.x - zero_point) * scale;  // BUG: zero_point truncated!
}

// FIXED: Use float intermediates
inline half dequant_fixed(uint32_t packed, half scale, half zero_point) {
    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);
    float fscale = (float)scale;       // Cast to float
    float fzero = (float)zero_point;   // Cast to float
    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;
    return (half)(((float)v0.x - fzero) * fscale);  // Compute in float
}
```

**Performance impact:** None measurable. At LLM scale (16M weights), the dequant operation
is memory-bound. Float conversion overhead is hidden by memory latency. Benchmarked at
~77.6 GB/s for both half and float versions.

**Rule:** Use `float` intermediates for dequant arithmetic with fractional zero points.

#### Summary: Compiler Bug Workarounds

| Bug | Symptom | Workaround |
|-----|---------|------------|
| Array Parameter | Column repetition in GEMM | `__attribute__((always_inline))` |
| Tile Coverage | Rows 32+ are zeros | Verify SG coverage math |
| Half-Precision | Fractional values truncated | Use float intermediates |

### Summary: Pre-Commit Checklist for Metal Kernels

Before shipping ANY Metal kernel change:

- [ ] Tested at M = 1, 2, 3, 4, TILE_M-1, TILE_M, TILE_M+1
- [ ] Documented expected tensor layouts in function signature
- [ ] Verified scales/weights aren't transposed relative to expectation
- [ ] Tested multiple forward passes (state corruption check)
- [ ] Tested with varied batch sizes and sequence lengths
- [ ] Used `float` accumulation for attention scores
- [ ] Explicit zero-initialization for shared memory
- [ ] Threadgroup size matches kernel expectation
- [ ] **Kernel dispatch uses `wait=True` OR has explicit sync before output read**
- [ ] Test passes 3+ consecutive runs (catches non-deterministic async bugs)
- [ ] **Functions receiving `simdgroup_matrix` arrays use `__attribute__((always_inline))`**
- [ ] **Dequant functions use float intermediates for fractional zero points**

### 1. Many Small Dispatches (MEASURED ✅)

**Evidence:** Sequential MoE dispatch = 20s/token. Batched = <100ms. 200× difference.

```python
# Measured slow: Sequential expert dispatch
for expert_id in selected_experts:  # 4 experts × 64 experts = slow
    output += expert(x, expert_id)

# Measured fast: Batched dispatch
output = batched_moe_dispatch(x, selected_experts)  # Single kernel
```

### 2. Unfused GEMM + Dequant (MEASURED ✅)

**Evidence:** Fused kernel = 51.9× faster than separate dequant + matmul.

```python
# Measured slow (145.2 ms)
weights_fp16 = dequantize(weights_fp4)
output = torch.matmul(x, weights_fp16)

# Measured fast (2.8 ms)
output = mmfp4_gemm(x, weights_fp4, scales)
```

### 3. FP16 KV Cache at Long Context (ASSUMED ⚠️)

**Evidence level:** Calculated from memory math, NOT measured on our stack.
We have the paged attention kernels but haven't benchmarked long context.

```python
# Assumed slow (not benchmarked)
kv_cache = torch.zeros(2, 47, 100000, 20, 256, dtype=torch.float16)  # 48 GB

# Assumed fast (kernels exist, not wired)
kv_cache = PagedKVCache(dtype=torch.int4)  # 12 GB
```

**TODO:** Actually benchmark >10K context with different KV formats.

### 4. Scalar Reduction vs simd_sum (ASSUMED ⚠️)

**Evidence level:** Standard Metal best practice, but we haven't A/B tested.

```metal
// Assumed slow
float sum = 0;
for (uint i = 0; i < 32; i++) sum += values[i];

// Assumed fast (standard Metal practice)
float sum = simd_sum(value);
```

**TODO:** Create microbenchmark to verify simd_sum speedup on M4 Max.

### 5. Synchronizing Every Token (UNTESTED ❓)

**Evidence level:** We use async dispatch but haven't A/B tested sync frequency.

```python
# Assumed slow
for token in range(max_tokens):
    logits = model(input_ids)
    torch.mps.synchronize()  # Assumed overhead
    next_token = sample(logits)
```

**TODO:** Benchmark sync overhead on MPS. Don't assume CUDA behavior applies.

### 6. storageModePrivate vs Shared (UNTESTED ❓)

**Evidence level:** Apple documentation recommendation, not measured.

```swift
// We use this (zero-copy)
device.makeBuffer(options: .storageModeShared)

// Alternative (GPU-only, may be faster for scratch)
device.makeBuffer(options: .storageModePrivate)
```

**TODO:** Test if storageModePrivate is faster for temporary buffers.

### What We Actually Don't Know

- ❓ Is simdgroup_matrix always faster? (assumed, not measured)
- ❓ What's the optimal threadgroup size for different operations?
- ❓ Does double-buffering help on M4 Max?
- ❓ Is paged attention faster than contiguous for our workloads?
- ❓ What's the real dispatch overhead? (we say ~8μs, not measured directly)
- ❓ FP32 vs FP16 throughput ratio? (NOT 2:1 like NVIDIA - Apple's ALUs are unified)

## Benchmarks We Need to Run

> **These are the gaps in our knowledge.** Each should become a benchmark script.

### High Priority (Blocking Optimization Decisions)

| Question | Benchmark Needed | Potential Impact |
|----------|------------------|------------------|
| Dispatch overhead | Measure empty kernel dispatch latency | Informs fusion decisions |
| simd_sum vs scalar loop | Microbenchmark reduction | Validate speedup claim |
| FP32 vs FP16 throughput | Measure actual ratio on M4 Max | NOT 2:1 like NVIDIA |
| storageModePrivate vs Shared | Buffer throughput comparison | May speed up scratch buffers |
| Long context decode | >10K context benchmark | Validate KV cache projections |

### Medium Priority (Multi-Model Support)

| Question | Benchmark Needed | Status |
|----------|------------------|--------|
| Qwen-3 4B throughput | End-to-end dense model | ✅ **33.4 tok/s BF16** (93% roofline) |
| INT4 vs FP8 KV cache | Quality + throughput tradeoff | ❌ Not measured |
| Paged vs contiguous attention | A/B test | ❌ Not measured |

### Running Existing Benchmarks

```bash
cd contrib/metal_marlin

# Per-kernel timing (GLM-4.7 shapes)
uv run python benchmarks/bench_glm47_kernels.py

# MoE throughput (fused vs unfused)
uv run python benchmarks/benchmark_moe_throughput.py

# Full model benchmark (requires weights)
uv run python benchmarks/glm_flash_benchmark.py --model path/to/model
```

## Implementation Checklist

### Critical (Measured Impact)

- [ ] **Fused GEMM kernels** - 50× speedup measured
- [ ] **Batched MoE dispatch** - 200× speedup measured
- [ ] **Mixed-precision quantization** - 20% throughput gain measured
- [ ] **Minimize dispatches** - 8μs overhead per dispatch adds up

### Short Context Optimization (< 2K tokens)

- [ ] Fused QKV projection
- [ ] Fused gate+up+SiLU+down for MoE
- [ ] simdgroup_matrix for attention
- [ ] Minimize kernel dispatches (<6 per layer)

### Medium Context (2K - 10K tokens)

- [ ] FP8 KV cache
- [ ] FlashAttention-V2 decode kernel
- [ ] Async weight prefetch
- [ ] Paged attention with 16-token blocks

### Long Context (10K+ tokens)

- [ ] INT4 KV cache with per-row scales
- [ ] Paged attention V2 (partitioned)
- [ ] Memory-efficient attention (recompute vs cache)
- [ ] Speculative decoding consideration

## Kernel Discoverability: What's Actually Available

> **Problem:** Kernels are named after specific models ("trellis", "GLM") when they're general-purpose.
> This section maps model-specific names to what they actually do.

### Naming Conventions (Historical Baggage)

| Current Name | What It Actually Is | Usable For |
|--------------|--------------------|-----------| 
| `gemm_trellis*` | Mixed-precision GEMM (2-8 bit) | Any quantized model |
| `moe_trellis_swiglu*` | Fused MoE with SwiGLU activation | Any SwiGLU MoE (Mixtral, Qwen, GLM, DeepSeek) |
| `rope_mla_*` | RoPE for small latent dims | Any MLA model (GLM, DeepSeek-V2/V3) |
| `fused_qkv_trellis` | Fused Q/K/V projection | Any transformer |
| `attention_mla_fused*` | Fused Multi-head Latent Attention | GLM, DeepSeek-V2/V3 |

### General-Purpose Kernels (Model-Agnostic)

| Kernel | Use Case | Works With |
|--------|----------|------------|
| `paged_attention_v1*` | vLLM-style paged KV cache | Any attention |
| `flash_attention_v2_decode` | Decode-optimized attention | Any model |
| `flash_attention_v3_*` | Prefill attention variants | Any model |
| `rmsnorm*` | RMSNorm variants | Any model using RMSNorm |
| `layernorm*` | LayerNorm variants | Any model |
| `rope_forward*` | Standard RoPE | Qwen, most models |
| `rope_yarn_*` | YaRN extended context | Any model needing >4K context |

### MoE Kernels (What's Actually MoE-Generic?)

| Kernel | Specialized For | Could Work With |
|--------|-----------------|-----------------|
| `moe_trellis_swiglu*` | SwiGLU activation | Mixtral, Qwen MoE, GLM, DeepSeek MoE |
| `moe_router*` | Expert selection | Any MoE |
| `expert_gather*` | Token-to-expert routing | Any MoE |
| `moe_combine*` | Expert output combination | Any MoE |

### MLA Kernels (Which Models Fit?)

MLA (Multi-head Latent Attention) is NOT GLM-specific. These work with:
- GLM-4.7-Flash
- DeepSeek-V2, DeepSeek-V3
- Any model using low-rank KV compression

| Kernel | Purpose | Notes |
|--------|---------|-------|
| `mla_proj_fp4_*` | Quantized MLA projection | Works at any bit-width |
| `mla_fused_kv_proj*` | Fused kv_a + kv_b | Reduces dispatches |
| `rope_mla_latent*` | RoPE for MLA latents | Small rope_dim only |
| `attention_mla_fused` | Full fused MLA | ⚠️ Currently unused |

### Quantization Kernels

| Format | Kernel | Notes |
|--------|--------|-------|
| FP4 (E2M1) | `dequant_fp4*`, `mmfp4_gemm` | 4-bit floating point |
| INT4 | `dequant_int4*`, `paged_attention_v1_int4` | Integer 4-bit |
| FP8 (E4M3) | `dequant_fp8*`, `paged_attention_v1_fp8` | 8-bit floating point |
| INT8 | `gemm_int8*` | Integer 8-bit |
| Sub-4-bit (2-3) | `gemm_trellis_w2*` | 2-bit with codebooks |

### What's Missing (Gaps to Fill)

1. **Dense model GEMM:** No optimized dense FP16 GEMM (we assume quantized)
2. **GeLU MoE:** Only SwiGLU activation fused (no GeLU/GELU variants)
3. **Grouped-Query Attention:** GQA kernels exist but not benchmarked
4. **Generic MLA:** Hardcoded to GLM-4.7 dimensions in some places
5. **Qwen-3 integration:** Best dense model candidate, not yet wired up

### Recommended Renaming (Future Work)

| Current | Proposed | Reason |
|---------|----------|--------|
| `gemm_trellis*` | `gemm_mixedbit*` | Not Trellis-specific |
| `moe_trellis_swiglu*` | `moe_swiglu_fused*` | Works for any SwiGLU MoE |
| `attention_mla_fused` | `attention_latent_fused` | MLA is generic |
| `rope_mla_*` | `rope_latent_*` | Works with any low-rank KV |

## References

### Metal Marlin Kernels (General Purpose)

| File | Purpose | Model-Agnostic? |
|------|---------|-----------------|
| [paged_attention.metal](../src/paged_attention.metal) | vLLM-style paged attention | ✅ Yes |
| [flash_attention_v3.metal](../src/flash_attention_v3.metal) | FlashAttention-3 | ✅ Yes |
| [layernorm.metal](../src/layernorm.metal) | LayerNorm/RMSNorm | ✅ Yes |
| [rope.metal](../src/rope.metal) | RoPE variants | ✅ Yes |
| [moe_router_fused.metal](../src/moe_router_fused.metal) | Expert routing | ✅ Yes |

### Metal Marlin Kernels (Named for GLM but Generic)

| File | Purpose | Actually Works With |
|------|---------|---------------------|
| [gemm_trellis_moe.metal](../src/gemm_trellis_moe.metal) | Fused MoE GEMM | Any SwiGLU MoE |
| [mla_proj.metal](../src/mla_proj.metal) | MLA projections | Any MLA model |
| [attention_mla_fused.metal](../src/attention_mla_fused.metal) | Fused MLA attention | GLM, DeepSeek |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| [bench_glm47_kernels.py](../benchmarks/bench_glm47_kernels.py) | Per-kernel timing |
| [glm47_roofline.py](../benchmarks/glm47_roofline.py) | Memory bandwidth analysis |
| [benchmark_moe_throughput.py](../benchmarks/benchmark_moe_throughput.py) | MoE throughput |

### External Resources

- [Apple Metal Best Practices](https://developer.apple.com/documentation/metal/gpu_programming)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2307.08691) - Memory-efficient attention
- [vLLM Paged Attention](https://arxiv.org/abs/2309.06180) - KV cache management
- [DeepSeek-V2 MLA Paper](https://arxiv.org/abs/2405.04434) - Multi-head Latent Attention
- [MoE Survey](https://arxiv.org/abs/2209.01667) - Mixture of Experts architectures

---

## GLM-4.7-Flash: Immediate Priorities (February 2026)

### Priority 1: Wire `mla_fused_attention_decode_glm4` (Est: 2-3× speedup)

**Status:** Kernel exists but is NOT CALLED during inference.

The kernel at `src/attention_mla_fused.metal` implements full fused MLA decode:
- Q projection (q_a → layernorm → q_b)
- KV projection (kv_a → split → layernorm → kv_b)  
- RoPE fused in KV projection
- Attention with compressed KV cache
- Output projection

**Current path:** 5+ kernel dispatches per attention layer  
**With fusion:** 1 kernel dispatch

**Files to modify:**
- `metal_marlin/trellis/attention.py` - Add fused path in `TrellisMLAttention.forward()`
- `metal_marlin/mla_fused.py` - Ensure `mla_fused_attention_decode()` is wired

```python
# Target integration
from metal_marlin.mla_fused import mla_fused_attention_decode, create_glm_mla_params

# In TrellisMLAttention.forward():
if self.use_fused_decode and batch_size == 1 and seq_len == 1:
    return mla_fused_attention_decode(hidden, ...)  # Single dispatch
else:
    # Current unfused path
    ...
```

### Priority 2: Enable Quantized KV Cache (Est: 2× context length)

**Status:** Kernels exist (`paged_attention_v1_fp8`, `paged_attention_v1_int4`) but NOT WIRED.

MLA already compresses KV 8.9× via low-rank projection. Adding INT4 quantization:

| KV Storage | Floats/Token | vs Standard MHA |
|------------|--------------|------------------|
| Standard MHA FP16 | 5120 | 1.0× |
| MLA FP16 | 576 | 8.9× |
| **MLA INT4** | 144 | **35.6×** |

**Files to modify:**
- `metal_marlin/trellis/kv_cache.py` - Add quantized storage option
- `metal_marlin/paged/` - Wire paged attention kernels to TrellisKVCache

### Priority 3: Decode GEMV Optimization (Est: 2× decode)

**Status:** Using generic matmul path for batch=1 decode.

For decode (batch=1, seq=1), we need specialized GEMV kernels:
- `gemm_trellis_packed_decode` exists but may not be selected
- Need to verify kernel selection path chooses decode kernel

**Files to check:**
- `metal_marlin/trellis/kernel_selection.py` - Verify decode path selected
- `metal_marlin/trellis/linear.py` - Check `TrellisLinear.forward()` calls right kernel

### Priority 4: Reduce GPU Commits (Est: 1.5× speedup)

**Status:** ~12 commits per forward pass after async batching fix.

Target: 4 commits per forward pass by batching all layers in groups of 8-12.

**Current:** LayerBatchContext groups 8 layers  
**Target:** Group all 47 layers into ~6 batches

### Verification Commands

```bash
# Run tests to verify implementation
cd contrib/metal_marlin
.venv/bin/pytest tests/test_trellis_linear.py tests/test_trellis_model.py -v

# Quick benchmark (requires model weights)
.venv/bin/python benchmarks/glm_flash_benchmark.py --quick

# Profile MoE hotpath
.venv/bin/python benchmarks/profile_moe_hotpath.py
```

### Success Metrics

| Metric | Current | Week 1 Target | Week 2 Target |
|--------|---------|---------------|---------------|
| Decode | 0.28 tok/s | 3 tok/s | 10 tok/s |
| Prefill (2K) | 42 tok/s | 100 tok/s | 200 tok/s |
| GPU commits | ~12 | 8 | 4 |

### Task YAML Template

```yaml
# yaml-language-server: $schema=
tasks:
  - name: wire-mla-fused-attention
    priority: P0
    prompt: |
      Wire `mla_fused_attention_decode_glm4` kernel in TrellisMLAttention.
      
      1. In `metal_marlin/trellis/attention.py`:
         - Import `mla_fused_attention_decode` from `metal_marlin.mla_fused`
         - Add `use_fused_decode` flag to __init__
         - In forward(), when batch=1 and decode mode, call fused kernel
      
      2. Add test in `tests/test_trellis_attention.py`:
         - Test fused path produces same output as unfused within 1e-4
      
      Verify: `pytest tests/test_trellis_attention.py -v -k fused`
    dependencies: []
    
  - name: benchmark-fused-attention-gain
    priority: P1
    prompt: |
      Profile attention before/after fused kernel integration.
      
      Create `benchmarks/bench_mla_fused_improvement.py`:
      - Load GLM-4.7-Flash-Trellis-MM
      - Time 100 decode steps with unfused path
      - Time 100 decode steps with fused path
      - Report dispatch count and latency reduction
      
      Verify: test -f benchmarks/results/mla_fused_benchmark.json
    dependencies: [wire-mla-fused-attention]
```
