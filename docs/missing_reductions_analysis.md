# FP4 Dequantization Function Analysis

## Summary

There are **8 distinct `dequant_fp4x8` function definitions** spread across **7 Metal shader files**. These functions all perform the same operation (dequantize 8 FP4 values from a packed uint32) but use different naming conventions and implementations.

## Definitions Found

| File | Function Name | Implementation |
|------|---------------|----------------|
| `mla_proj.metal:72` | `dequant_fp4x8` | Uses `dequant_fp4_scalar()` helper |
| `gemm_epilogue.metal:372` | `fused_dequant_fp4x8` | Uses `safe_dequant()` + `fused_dequant_fp4_scalar()` |
| `marlin_gemm.metal:2031` | `fused_dequant_fp4x8` | Identical to gemm_epilogue.metal |
| `moe_dispatch_optimized.metal:97` | `opt_dequant_fp4x8_lut` | LUT-based with `#pragma unroll` |
| `moe_dispatch_optimized.metal:107` | `opt_dequant_fp4x8_bitwise` | Bitwise computation fallback |
| `gemm_fp4_optimized.metal:84` | `unpack_dequant_fp4x8` | Direct LUT indexing, no intermediate float |
| `moe_fused_router.metal:88` | `fused_dequant_fp4x8` | Uses `fused_dequant_fp4_scalar()` |
| `moe_dispatch.metal:84` | `dispatch_dequant_fp4x8` | Uses `dispatch_dequant_fp4_scalar()` |

## Implementation Variants

### 1. Scalar Helper Pattern (most common)
```metal
inline void dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    out[0] = (half)((float)dequant_fp4_scalar((packed >>  0) & 0xF) * fscale);
    // ... repeat for indices 1-7
}
```
Used in: `mla_proj.metal`, `moe_dispatch.metal`, `moe_fused_router.metal`

### 2. Safe Dequant Pattern
```metal
inline void fused_dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    out[0] = safe_dequant(fused_dequant_fp4_scalar((packed >>  0) & 0xF), scale);
    // ... repeat for indices 1-7
}
```
Used in: `gemm_epilogue.metal`, `marlin_gemm.metal`

### 3. Direct LUT Pattern (fastest)
```metal
inline void unpack_dequant_fp4x8(uint32_t packed, half scale, thread half* out) {
    out[0] = FP4_LUT[(packed >>  0) & 0xF] * scale;
    // ... repeat for indices 1-7
}
```
Used in: `gemm_fp4_optimized.metal`

### 4. Unrolled LUT Pattern
```metal
inline void opt_dequant_fp4x8_lut(uint packed, half scale, thread half* out) {
    float fscale = (float)scale;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = half(fscale * (float)FP4_LUT[nibble]);
    }
}
```
Used in: `moe_dispatch_optimized.metal`

## Usage Analysis

| File | Function | Call Count |
|------|----------|------------|
| `moe_dispatch_optimized.metal` | `opt_dequant_fp4x8_lut` | 9 |
| `mla_proj.metal` | `dequant_fp4x8` | 8 |
| `moe_dispatch.metal` | `dispatch_dequant_fp4x8` | 6 |
| `gemm_fp4_optimized.metal` | `unpack_dequant_fp4x8` | 4 |
| `moe_fused_router.metal` | `fused_dequant_fp4x8` | 4 |
| `marlin_gemm.metal` | `fused_dequant_fp4x8` | 2 |
| `gemm_epilogue.metal` | `fused_dequant_fp4x8` | 1 |

**Total calls: 34**

## Duplication Analysis

### Exact Duplicates
- `gemm_epilogue.metal:372` and `marlin_gemm.metal:2031` are **identical**.

### Semantic Duplicates (same result, different code paths)
All 8 functions produce the same output given the same input. The differences are:
1. **Naming**: prefix varies (`fused_`, `opt_`, `dispatch_`, `unpack_`)
2. **Intermediate precision**: some convert to float, some stay in half
3. **Safety guards**: `safe_dequant()` clamps extreme values
4. **Loop structure**: unrolled vs explicit 8 statements

## Recommendation

These should be consolidated into a **common header** (`dequant_common.h`):

```metal
// dequant_common.h

// FP4 lookup table (E2M1 format)
constant half FP4_LUT[16] = {
    0.0h, 0.25h, 0.5h, 0.75h, 1.0h, 1.5h, 2.0h, 3.0h,
    -0.0h, -0.25h, -0.5h, -0.75h, -1.0h, -1.5h, -2.0h, -3.0h
};

// Fast path: direct LUT indexing (use when numerical stability not critical)
inline void dequant_fp4x8_fast(uint32_t packed, half scale, thread half* out) {
    out[0] = FP4_LUT[(packed >>  0) & 0xF] * scale;
    out[1] = FP4_LUT[(packed >>  4) & 0xF] * scale;
    out[2] = FP4_LUT[(packed >>  8) & 0xF] * scale;
    out[3] = FP4_LUT[(packed >> 12) & 0xF] * scale;
    out[4] = FP4_LUT[(packed >> 16) & 0xF] * scale;
    out[5] = FP4_LUT[(packed >> 20) & 0xF] * scale;
    out[6] = FP4_LUT[(packed >> 24) & 0xF] * scale;
    out[7] = FP4_LUT[(packed >> 28) & 0xF] * scale;
}

// Safe path: intermediate float, clamping (use for numerical stability)
inline void dequant_fp4x8_safe(uint32_t packed, half scale, thread half* out) {
    float fscale = clamp((float)scale, -65504.0f, 65504.0f);
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = half(fscale * (float)FP4_LUT[nibble]);
    }
}
```

### Migration Path

1. Create `src/dequant_common.h` with unified implementations
2. Replace all variants with `#include "dequant_common.h"`
3. Map existing functions to the appropriate variant:
   - `unpack_dequant_fp4x8` -> `dequant_fp4x8_fast`
   - `opt_dequant_fp4x8_lut` -> `dequant_fp4x8_fast`
   - `fused_dequant_fp4x8` -> `dequant_fp4x8_safe`
   - `dispatch_dequant_fp4x8` -> `dequant_fp4x8_fast`
   - `dequant_fp4x8` -> `dequant_fp4x8_fast`

### Benefits
- **Reduced maintenance**: Single source of truth
- **Consistent behavior**: No subtle numerical differences
- **Easier optimization**: Tune once, benefit everywhere
- **Better testing**: Test one implementation, not 8

## No Conflicts Found

There are no conflicting definitions (same function name, different behavior). The duplication is parallel development resulting in multiple implementations of the same algorithm. The `fused_dequant_fp4x8` name appears in 3 files but the implementations are functionally equivalent.

---

# Threadgroup Reduction Functions Analysis

## Summary

The threadgroup reduction functions (`threadgroup_reduce_max`, `threadgroup_reduce_sum`, `THREADGROUP_REDUCE_MAX`, `THREADGROUP_REDUCE_SUM`) are defined in `src/reduction_helpers.metal`. This file exists but is **untracked in git** and needs to be committed.

## Current State

### File Location
```
contrib/metal_marlin/src/reduction_helpers.metal  (UNTRACKED)
```

### Defined Functions

| Function/Macro | Type | Description |
|----------------|------|-------------|
| `threadgroup_reduce_max` | inline function | Tree reduction for max across 32 threads |
| `threadgroup_reduce_sum` | inline function | Tree reduction for sum across 32 threads |
| `threadgroup_reduce_max_tree` | inline function | Alias wrapper for tree attention |
| `THREADGROUP_REDUCE_MAX` | macro | Macro version for inline expansion |
| `THREADGROUP_REDUCE_SUM` | macro | Macro version for inline expansion |

### Files Including This Header

1. `src/tree_attention.metal` - includes at line 1
2. `src/attention.metal` - includes at line 35

## Historical Context

The file was created during recent development but never committed. Git history shows no record of the file since it's untracked:

```bash
$ git status contrib/metal_marlin/src/reduction_helpers.metal
Untracked files:
  contrib/metal_marlin/src/reduction_helpers.metal
```

## Alternative Implementations

Many Metal files define their own simd-level reductions locally rather than using shared helpers:

| File | Local Functions |
|------|-----------------|
| `diff_attention.metal` | `simd_max_f32`, `simd_sum_f32` |
| `flash_attention.metal` | `simd_max_f32`, `simd_sum_f32` |
| `layernorm.metal` | `simd_sum_f32` |
| `sliding_window_attention.metal` | `simd_max_sw`, `simd_sum_sw` |
| `moe_router.metal` | `simd_max`, `simd_sum` |
| `sampling.metal` | `simd_max`, `simd_sum` |
| `tree_attention.metal` | `THREADGROUP_REDUCE_MAX_TREE`, `THREADGROUP_REDUCE_SUM_TREE` (own macros) |

The `tree_attention.metal` file is interesting: it includes `reduction_helpers.metal` at line 1 but then defines its own `THREADGROUP_REDUCE_MAX_TREE` and `THREADGROUP_REDUCE_SUM_TREE` macros (lines 79-120). These macros use simd intrinsics for 4-simdgroup reduction rather than the 32-thread tree reduction in the helper.

## Resolution

### Option 1: Commit the Helper File (Recommended)
```bash
git add contrib/metal_marlin/src/reduction_helpers.metal
git commit -m "Add shared threadgroup reduction helpers for Metal kernels"
```

### Option 2: Remove Dependency

If the helper file should not exist, update the files that include it:

1. `tree_attention.metal` - Already has its own macros, remove the include
2. `attention.metal` - Uses `parallel_reduce_*` functions defined locally, may not need the include

## Implementation Details

### reduction_helpers.metal Functions

```metal
// 32-thread tree reduction for max
inline float threadgroup_reduce_max(float thread_value, uint tid, threadgroup float* scratch) {
    scratch[tid] = thread_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < 32; offset *= 2) {
        if (tid % (2 * offset) == 0 && tid + offset < 32) {
            scratch[tid] = max(scratch[tid], scratch[tid + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}
```

### tree_attention.metal Macros (Different Algorithm)

```metal
// 4-simdgroup reduction using simd_max first, then cross-simdgroup
#define THREADGROUP_REDUCE_MAX_TREE(val, tid, scratch, result_var) do { \
    float _tgrm_sg_max = simd_max(val); \
    uint _tgrm_sg_id = (tid) / 32; \
    uint _tgrm_lane = (tid) % 32; \
    if (_tgrm_lane == 0) { \
        (scratch)[_tgrm_sg_id] = _tgrm_sg_max; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    // ... second phase across simdgroups ...
} while(0)
```

The tree_attention version is optimized for 128-thread (4 simdgroup) threadgroups, while reduction_helpers is for 32-thread threadgroups.

## Recommendation

1. **Commit `reduction_helpers.metal`** to prevent build failures
2. **Audit includes** in `tree_attention.metal` and `attention.metal` to verify the include is actually needed
3. **Consider consolidating** reduction implementations across the codebase for maintainability
