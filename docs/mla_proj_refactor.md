# MLA Projection Kernel Refactor Analysis

**File:** `contrib/metal_marlin/src/mla_proj.metal`
**Total Issues:** 30+ (19 barriers + 11 threadgroup memory declarations)
**Priority:** High - barriers are performance bottlenecks

## Summary

The `mla_proj.metal` file implements Multi-head Latent Attention (MLA) projection kernels for GLM-4 and DeepSeek-V2/V3 style compressed KV cache. The file has significant code duplication across 6 kernel variants, each containing similar barrier and threadgroup memory patterns that should be refactored.

---

## Kernel Functions

| Line | Kernel Name | Purpose | Barriers | TG Memory |
|------|-------------|---------|----------|-----------|
| 135 | `mla_proj_fp4_k16` | Single projection, TILE_K=16 | 4 | 3 arrays |
| 247 | `mla_proj_fp4_k32` | Single projection, TILE_K=32 | 4 | 3 arrays |
| 375 | `mla_fused_kv_proj_fp4` | Fused kv_a + kv_b projection | 9 | 4 arrays |
| 634 | `mla_proj_with_rope_fp4` | Projection with RoPE fusion | 4 | 3 arrays |
| 820 | `mla_decode_proj_fp4` | GEMV for single token decode | 0 | 0 |
| 869 | `mla_decode_batched_fp4` | Batched GEMV (batch <= 8) | 0 | 0 |

---

## Helper Functions

| Line | Function | Purpose | Uses TG Memory | Needs Refactor |
|------|----------|---------|----------------|----------------|
| 61 | `dequant_fp4_scalar` | Dequantize single FP4 nibble | No | No |
| 72 | `dequant_fp4x8` | Dequantize 8 FP4 values from packed uint32 | No | No |
| 92 | `apply_rope_pair` | Apply RoPE rotation to (x, y) pair | **Yes** (lines 93-94) | **Yes** |
| 106 | `apply_rope_tg` | Apply RoPE to vector in TG memory | **Yes** (line 107) | **Yes** |

---

## MLA Computation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLA Projection Pipeline                          │
└─────────────────────────────────────────────────────────────────────┘

1. PREFILL PHASE (mla_proj_fp4_k16/k32):
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ hidden   │────▶│ kv_a_proj│────▶│ latent   │  (large K → small K)
   │ [M, 4096]│     │ W_a[4096,│     │ [M, 512] │
   └──────────┘     │      512]│     └──────────┘
                    └──────────┘

   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ latent   │────▶│ kv_b_proj│────▶│ output   │  (small K → large N)
   │ [M, 512] │     │ W_b[512, │     │ [M, 8192]│
   └──────────┘     │     8192]│     └──────────┘
                    └──────────┘

2. FUSED PROJECTION (mla_fused_kv_proj_fp4):
   ┌──────────┐                      ┌──────────┐
   │ hidden   │───────┬─────────────▶│ output   │
   │ [M, 4096]│       │              │ [M, 8192]│
   └──────────┘       │              └──────────┘
                      │
              ┌───────▼───────┐
              │ Fused in TG   │
              │ W_a @ W_b     │
              │ (no latent    │
              │  materialized)│
              └───────────────┘

3. ROPE FUSION (mla_proj_with_rope_fp4):
   Same as #1, but applies RoPE to output[:, :rope_dim]
   for decoupled rope head dimension (GLM-4 style)

4. DECODE PHASE (mla_decode_*):
   GEMV: single token (or small batch ≤ 8) through projection
   No TG memory needed - pure register accumulation
```

---

## Barrier Analysis by Kernel

### mla_proj_fp4_k16 (lines 135-244)

| Line | Barrier Purpose | Can Eliminate? |
|------|-----------------|----------------|
| 210 | After loading A_tile and B_tile | No - required |
| 222 | After K-reduction loop iteration | **Yes** - merge with 210 |
| 232 | After simdgroup_store to staging | No - required for partial tiles |
| 240 | After writing partial tile output | **Yes** - last iteration only |

### mla_proj_fp4_k32 (lines 247-356)

Same pattern as k16:
| Line | Barrier Purpose | Can Eliminate? |
|------|-----------------|----------------|
| 322 | After loading tiles | No |
| 334 | After K-reduction | **Yes** - merge |
| 344 | After staging store | No |
| 352 | After partial write | **Yes** - last only |

### mla_fused_kv_proj_fp4 (lines 375-622)

Most complex - 9 barriers:
| Line | Barrier Purpose | Can Eliminate? |
|------|-----------------|----------------|
| 432 | After zeroing latent_tile | **Yes** - use make_filled |
| 492 | After loading hidden_tile + W_a_tile | No |
| 533 | After accumulating to latent_tile | No |
| 575 | After loading W_b_tile | No |
| 597 | After latent @ W_b accumulation | **Yes** - merge with next |
| 610 | After staging store | No |
| 618 | After partial write | **Yes** - last only |

### mla_proj_with_rope_fp4 (lines 634-811)

| Line | Barrier Purpose | Can Eliminate? |
|------|-----------------|----------------|
| 725 | After loading tiles | No |
| 741 | After K-reduction | **Yes** - merge |
| 752 | Before RoPE application | No - required |
| 798 | After RoPE, before write | No - required |

---

## Threadgroup Memory Analysis

### Shared Patterns Across Kernels

```metal
// Pattern 1: Input tile (all GEMM kernels)
threadgroup half A_tile[TILE_M_MLA][TILE_K];

// Pattern 2: Weight tile (all GEMM kernels)
threadgroup half B_tile[TILE_K][TILE_N_MLA];

// Pattern 3: Output staging (partial tiles)
threadgroup half out_staging[8][8];

// Pattern 4: Fused kernel intermediate
threadgroup half latent_tile[TILE_M_MLA][TILE_K_MLA_LARGE];

// Pattern 5: RoPE kernel output buffer
threadgroup half out_tile[TILE_M_MLA][TILE_N_MLA];
```

---

## Refactoring Recommendations

### 1. Extract Common GEMM Core (High Priority)

The k16 and k32 variants differ only in TILE_K. Extract a template:

```metal
template <uint TILE_K_VAL, uint K_TILES_VAL>
void mla_proj_core(
    threadgroup half (&A_tile)[TILE_M_MLA][TILE_K_VAL],
    threadgroup half (&B_tile)[TILE_K_VAL][TILE_N_MLA],
    threadgroup half (&out_staging)[8][8],
    /* ... params ... */
);
```

**Impact:** Eliminates ~200 lines of duplication, centralizes barrier logic.

### 2. Eliminate Redundant End-of-Loop Barriers

The pattern:
```metal
threadgroup_barrier(mem_flags::mem_threadgroup);  // line X (start of loop body)
// ... load tiles ...
// ... compute ...
threadgroup_barrier(mem_flags::mem_threadgroup);  // line Y (end of loop body)
```

Can become:
```metal
for (...) {
    // First iteration: no barrier needed before load
    // ... load tiles ...
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ... compute ...
}
// Final barrier only if needed for output
```

**Impact:** Remove 4-5 barriers across kernels.

### 3. Use Initialization Instead of Zero + Barrier

Replace:
```metal
for (uint i = thread_idx; i < TILE_M_MLA * TILE_K_MLA_LARGE; i += THREADS_PER_TG_MLA) {
    latent_tile[row][col] = half(0.0h);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

With simdgroup_fill or structure initialization where possible.

### 4. RoPE Helper Refactoring

`apply_rope_pair` and `apply_rope_tg` take threadgroup pointers but are called from thread context. Consider:

```metal
// Option A: Thread-local RoPE (registers)
inline void apply_rope_pair_thread(thread half& x, thread half& y, half cos_val, half sin_val);

// Option B: Keep TG version but add assertion
inline void apply_rope_tg(
    threadgroup half* vec,
    // ...
) {
    // Document: caller must ensure barrier before and after
}
```

### 5. Partial Tile Output Helper

Extract the partial tile handling pattern into a helper:

```metal
template <uint ROWS, uint COLS>
void store_partial_tile(
    simdgroup_matrix<half, 8, 8>& acc,
    threadgroup half (&staging)[8][8],
    device half* out,
    uint out_row, uint out_col,
    uint M, uint N, uint N_stride,
    uint simd_lane
);
```

**Impact:** Centralizes the barrier-heavy partial tile write logic.

---

## Priority Ranking

| Priority | Refactor | Lines Saved | Barriers Eliminated |
|----------|----------|-------------|---------------------|
| P0 | Extract GEMM core template | ~200 | 4 (centralized) |
| P0 | Remove redundant loop-end barriers | 0 | 4-5 |
| P1 | Partial tile output helper | ~40 | 2 (centralized) |
| P1 | Zero-init optimization | ~10 | 2 |
| P2 | RoPE helper cleanup | ~10 | 0 |

---

## Verification Commands

After refactoring, verify correctness:

```bash
cd contrib/metal_marlin && uv run pytest tests/test_mla_proj.py -v
cd contrib/metal_marlin && uv run python benchmark_all_layers.py --filter mla
```

---

## Related Files

- `contrib/metal_marlin/metal_marlin/trellis/linear.py` - Python interface
- `contrib/metal_marlin/docs/architectures/mla.md` - MLA architecture docs
- `contrib/metal_marlin/docs/audits/barrier_optimization.md` - General barrier guidance
