# MMFP4 Inference Stack Debugging Summary

**Date:** Session continuation
**Status:** Root cause identified, task list created

## Problem Statement

GLM-4.7-Flash MMFP4 inference stack produces NaN for sequences with 4+ tokens.

## Diagnostic Results

### Token Length vs NaN

| Tokens | Result |
|--------|--------|
| 1 | ✓ Works |
| 2 | ✓ Works |
| 3 | ✓ Works |
| 4 | ✗ NaN (consistent) |
| 5 | ✗ NaN (sometimes) |
| 6+ | ✗ NaN (consistent) |

### Layer-by-Layer Analysis

For 4-token input:
- Embeddings: OK
- Layer 0 (MMFP4MLP): OK, but mean values start diverging
- Layer 1 (MMFP4MoE): Explosive growth in hidden states
- Final layers: Values range from -62205 to +63294

For 3-token input:
- All layers: OK, values stay bounded
- Final layer mean: ~9.2

### Key Observations

1. **Tile boundary issue**: 4 is a significant boundary (SIMD group size, tile dimension)
2. **Non-determinism at boundary**: seq_len=5 shows variable results (1/3 NaN)
3. **No explicit NaN in individual layers**: Values explode but don't become NaN until logit computation
4. **KV cache not the root cause**: Issue exists with `kv_cache=None`

## Suspected Root Causes

### Primary: Metal Kernel Tile Handling

The `mmfp4_gemm.metal` kernel uses:
- TILE_M = 64, TILE_N = 64, TILE_K = 32
- When M < TILE_M (e.g., M=4 for 4 tokens), boundary threads may read/write garbage

### Secondary: Attention Score Overflow

- Attention scores may not be clamped before softmax
- Missing numerical stability (subtract max before exp)
- Causal mask may use `-inf` instead of `-1e4`

## Code Changes Made (Then Reverted)

My changes to `mmfp4_linear.py` and `kernels.py` introduced additional bugs:
1. Changed `mmfp4_gemm()` to call `dispatch_gemm_fp4()` directly
2. Added layout detection logic that was incorrect
3. Added `B_orig` parameter handling

These were reverted; the current HEAD has the pre-existing 4+ token bug.

## Files to Fix

1. `contrib/metal_marlin/metal_marlin/layers/mmfp4_mla.py` - Attention masking
2. `contrib/metal_marlin/metal_marlin/shaders/mmfp4_gemm.metal` - Tile boundaries
3. `contrib/metal_marlin/metal_marlin/models/mmfp4_causal_lm.py` - Warmup in generate()

## Task List

Created: `agent_workspace/mmfp4_inference_fixes.yaml`

9 tasks across 5 phases:
- P0: Diagnosis (2 tasks)
- P0: Attention fixes (2 tasks)
- P1: Kernel fixes (1 task)
- P1: KV cache fixes (2 tasks)
- P1: E2E validation (1 task)
- P2: Documentation (1 task)

## Workarounds (Current)

1. **Use single-token decode only**: Works for autoregressive generation
2. **Add warmup pass**: Single-token forward before actual inference
3. **Disable KV cache**: `use_cache=False` works for multi-token prefill

## Optimization Applied

### Summary

Following the debugging and root cause analysis, the following optimizations were applied to the MMFP4 inference stack:

#### 1. Fused Dispatch Migration

**Sequential dispatches replaced with fused kernels:**

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| MLA Attention | Multiple dispatches (q_proj, k_proj, v_proj, attention, o_proj) | `mla_fused_attention_decode_glm4` | Fused |
| MoE Expert Compute | Sequential expert loops | `moe_trellis_swiglu` batched dispatch | Fused |
| GEMM + Dequant | Separate dequantize + matmul | `mmfp4_gemm` fused kernel | Fused |

#### 2. Async Dispatch Fix

**Critical fix:** Changed kernel dispatch from `wait=False` to `wait=True` in `kernels.py` to prevent race conditions that caused non-deterministic NaN for sequences >= 5 tokens.

#### 3. Numerical Stability Improvements

- Changed accumulator from `half` to `float` in `mmfp4_gemm.metal` to prevent overflow
- Added per-simdgroup staging buffers to eliminate race conditions

### Pending Benchmarks

**Note:** Performance benchmarks should be re-run manually to measure the impact of these optimizations.

- Compare pre-optimization vs post-optimization throughput
- Verify correctness is maintained after fusion
- Measure dispatch overhead reduction

**Status:** Performance numbers pending measurement - DO NOT include extrapolated or estimated values.

---

## Benchmark Results (Feb 2026)

### Summary

| Test | Result | Expected | Gap |
|------|--------|----------|-----|
| Perplexity | 17.15 | <100 | ✅ PASS |
| Prefill avg | 24.6 tok/s | 500-2000 tok/s | 20-80× slower |
| Decode | 0.27 tok/s | 5-20 tok/s | 20-80× slower |

### Prefill Benchmark (Single Forward Pass)

| Context Length | Throughput | Notes |
|----------------|------------|-------|
| 128 tokens | 9.0 tok/s | Short context, dispatch overhead dominates |
| 512 tokens | 25.5 tok/s | Better amortization |
| 1024 tokens | 39.3 tok/s | Best prefill efficiency |

### Decode Benchmark (Autoregressive)

| Configuration | Throughput | Notes |
|---------------|------------|-------|
| Without KV cache (manual loop) | 0.07 tok/s | Full context recomputation each token - CATASTROPHIC |
| With KV cache (model.generate()) | 0.27 tok/s | 4× improvement with MLAKVCache |

**Critical lesson:** Never use manual `for token in range(n): model(input_ids)` loops. 
Always use `model.generate(use_cache=True)` which automatically creates and uses `MLAKVCache`.

### Performance Gap Analysis

The 20-80× performance gap vs expected has multiple causes:

1. **MoE dispatch overhead** - Despite having `moe_trellis_swiglu` fused kernel, something is still slow
2. **MMFP4 GEMM efficiency** - kernels may not be fully optimized for M4 Max
3. **Missing PagedAttention** - `paged_attention_v1_fp4` exists but isn't wired to MMFP4

**Comparison with other backends:**

| Backend | Decode | Prefill | Notes |
|---------|--------|---------|-------|
| MLX (native) | 10-20 tok/s | N/A | Model not supported |
| llama.cpp Q4_K_M | ~3 tok/s | N/A | Different quant format |
| Metal Marlin (Trellis, non-MMFP4) | 5.4 tok/s | 42 tok/s | Working baseline |
| **MMFP4** | **0.27 tok/s** | **25 tok/s** | Current, needs optimization |

### Memory Management Lessons

MPS memory management is tricky:

1. **MPS doesn't release memory** on `del` alone - requires explicit cleanup:
   ```python
   outputs = model(input_ids)
   del outputs
   gc.collect()
   torch.mps.empty_cache()
   ```

2. **Subprocess isolation required** for true memory release between benchmark runs.
   Created `developer_tests/bench_mmfp4_all.py` which runs each test in a subprocess.

3. **Reference leaks**: `_ = model(...)` holds tensor reference until variable is reassigned.
   Always use explicit variable names and `del`.

### Benchmark Scripts Created

| Script | Purpose |
|--------|---------|
| `developer_tests/bench_mmfp4_perplexity.py` | Numerical validation via perplexity |
| `developer_tests/bench_mmfp4_prefill.py` | Context 128/512/1024 prefill throughput |
| `developer_tests/bench_mmfp4_decode.py` | Autoregressive decode with KV cache |
| `developer_tests/bench_mmfp4_all.py` | Subprocess runner for all tests |
| `developer_tests/bench_mmfp4_correctness.py` | Numerical correctness (sequential vs fused) |

## Next Steps

### Immediate (Correctness verified, performance unacceptable)

1. **Profile kernel dispatch** - Where is time actually going in MMFP4 forward pass?
2. **Wire PagedAttention to MMFP4** - `paged_attention_v1_fp4` exists but unused
3. **Compare with non-MMFP4 Trellis** - Why is Trellis 5.4 tok/s but MMFP4 is 0.27 tok/s?

### Task List

1. Run task list: `uv run alphaheng tasks add agent_workspace/mmfp4_inference_fixes.yaml`
2. Start coordinator: `uv run alphaheng coordinator --local-workers 50`
3. Monitor: `uv run alphaheng status`

---

## MLA GQA Shape Fix (Feb 2026)

### Bug

`attn_output.view()` expected 4096 elements, got 8192

### Root Cause

The MLA attention implementation had a shape mismatch bug in the GQA (Grouped Query Attention) handling. When the number of key/value heads (kv_heads) differs from the number of query heads (num_heads), the output tensor shape calculation was incorrect.

Specifically, the `attn_output` tensor from the attention computation had shape `[batch, num_heads, seq_len, head_dim]` but the final projection expected `[batch, seq_len, num_heads * head_dim]`. The `view()` operation was failing because:

1. For GQA configurations where `num_heads != kv_heads`, the attention output was being computed with expanded key/value dimensions
2. The reshape logic didn't account for the GQA ratio properly, causing the total element count to be double what was expected
3. The fix involved correctly reshaping the attention output before the final linear projection

### Fix

Updated the attention output reshaping logic in `mmfp4_mla.py` to properly handle GQA configurations by:
1. Computing the correct output shape based on `num_heads` (not `kv_heads`)
2. Ensuring the attention scores are computed with proper head grouping
3. Fixing the view/reshape operations to match the expected dimensions for all GQA ratios

### Verified

GQA ratios all pass:
- 32:32 (standard MHA)
- 32:16 (2:1 grouping)
- 32:8 (4:1 grouping)
- 32:2 (16:1 grouping)
- 32:1 (full GQA/MQA)

All configurations now produce correct output shapes and pass numerical correctness tests.
