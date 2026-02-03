# gemm_trellis_moe.metal Structure Analysis

This document analyzes the structure of `src/gemm_trellis_moe.metal` to support refactoring efforts.

## Overview

The file implements fused MoE (Mixture of Experts) GEMM kernels with Trellis 3bpw quantization. It contains 11 kernel functions and 18 helper functions totaling ~3,800 lines.

**Compilation Status**: The file compiles successfully with only unused variable warnings (no threadgroup-related errors).

## Kernel Functions (11 total)

| Line | Kernel | Description |
|------|--------|-------------|
| 922 | `moe_trellis_swiglu` | Main MoE kernel with SwiGLU, double-buffered weights |
| 1343 | `moe_trellis_swiglu_fp32acc` | FP32 accumulator variant for numerical stability |
| 1624 | `moe_trellis_swiglu_large_batch` | Large batch variant (MOE_TILE_N_LARGE=128) |
| 1842 | `moe_trellis_swiglu_simd` | SIMD-optimized using simdgroup_matrix 8x8 ops |
| 2735 | `moe_trellis_swiglu_decode` | Single-token decode with double-buffered prefetch |
| 3062 | `moe_trellis_swiglu_prefill4` | 4-token batched prefill |
| 3278 | `moe_trellis_swiglu_prefill4_fp32acc` | 4-token prefill with FP32 accumulators |
| 3512 | `moe_trellis_swiglu_grouped` | Expert-grouped processing (GGUF-style) |
| 3780 | `moe_count_tokens_per_expert` | Routing: count tokens per expert |
| 3797 | `moe_compute_expert_offsets` | Routing: prefix sum for expert offsets |
| 3815 | `moe_scatter_tokens_to_experts` | Routing: scatter tokens to sorted positions |

## Helper Functions (18 total)

### Dequantization Helpers

| Line | Function | Threadgroup Params |
|------|----------|-------------------|
| 164 | `unpack_3bit_x4` | None (pure math) |
| 176 | `unpack_3bit_x8` | None (pure math) |
| 206 | `dequant_grid_vec4` | None (pure math) |
| 229 | `apply_signs_vec4` | None (pure math) |
| 351 | `trellis_dequant_3bit` | None (device memory only) |
| 463 | `trellis_dequant_3bit_cached` | `threadgroup const half*` params (3x) |

### Atomic Add Helpers

| Line | Function | Threadgroup Params |
|------|----------|-------------------|
| 248 | `atomic_add_fp32_direct` | None (device atomics) |
| 268 | `atomic_add_fp32_simd_coordinated` | None (calls direct) |
| 283 | `atomic_add_fp32_vec4` | None (calls direct 4x) |

### Scale/Sign Prefetch

| Line | Function | Threadgroup Params |
|------|----------|-------------------|
| 528 | `prefetch_scale_sign_caches` | `threadgroup half*` params (3x) |

### Weight Tile Loaders

| Line | Function | Threadgroup Params |
|------|----------|-------------------|
| 415 | `load_trellis_tile` | `threadgroup half (&B_buf)[K][N]` |
| 620 | `load_trellis_tile_cached` | `threadgroup half (&B_buf)[K][N]`, `threadgroup const half*` caches (3x) |
| 673 | `load_trellis_tile_vec4` | `threadgroup half (&B_buf)[K][N]`, `threadgroup const half*` caches (3x) |
| 1585 | `load_trellis_tile_large` | `threadgroup half (&B_buf)[K][N_LARGE]` |

### Decode-Specific Helpers

| Line | Function | Threadgroup Params |
|------|----------|-------------------|
| 2232 | `prefetch_packed_tiles_decode` | `threadgroup uint8_t (&packed_cache)[SIZE]` |
| 2294 | `load_trellis_tile_decode_from_cache` | `threadgroup const uint8_t (&)[SIZE]`, `threadgroup half (&B_buf)[K][N]` |
| 2373 | `load_trellis_tile_decode` | `threadgroup half (&B_buf)[K][N]` |
| 2422 | `prefetch_packed_tiles_decode_doublebuf` | `threadgroup uint8_t (&)[SIZE]` (2x) |
| 2488 | `load_trellis_tile_decode_doublebuf_from_cache` | `threadgroup const uint8_t (&)[SIZE]` (2x), `threadgroup half (&)[K][N]` (2x) |
| 2596 | `load_trellis_tile_decode_doublebuf_async` | `threadgroup half (&)[K][N]` (2x) |
| 2680 | `load_trellis_tile_decode_doublebuf` | `threadgroup half (&)[K][N]` (2x) |

### Activation Loaders

| Line | Function | Threadgroup Params |
|------|----------|-------------------|
| 808 | `load_activation_tile` | `threadgroup half (&A_buf)[K]` |
| 868 | `load_activation_vec8` | `threadgroup half (&A_buf)[K]` |
| 3039 | `load_activation_tiles_prefill` | `threadgroup half (&A_tiles)[BATCH][K]` |

## Call Graph: Kernel → Helpers

### moe_trellis_swiglu (line 922)

```
moe_trellis_swiglu
├── load_activation_tile
├── prefetch_scale_sign_caches (gate, up)
├── load_trellis_tile_cached (gate, up)
├── prefetch_scale_sign_caches (down)
├── load_trellis_tile_cached (down)
└── atomic_add_fp32_vec4, atomic_add_fp32_direct
```

### moe_trellis_swiglu_fp32acc (line 1343)

```
moe_trellis_swiglu_fp32acc
├── load_activation_tile
├── load_trellis_tile (gate, up, down) [no caching]
└── atomic_add_fp32_vec4, atomic_add_fp32_direct
```

### moe_trellis_swiglu_large_batch (line 1624)

```
moe_trellis_swiglu_large_batch
├── load_activation_tile
├── load_trellis_tile_large (gate, up, down)
└── atomic_add_fp32_vec4, atomic_add_fp32_direct
```

### moe_trellis_swiglu_simd (line 1842)

```
moe_trellis_swiglu_simd
├── [inline dequant via trellis_dequant_3bit - no tile loader]
└── simdgroup_matrix operations
```

### moe_trellis_swiglu_decode (line 2735)

```
moe_trellis_swiglu_decode
├── load_trellis_tile_decode_doublebuf_async (gate+up)
├── load_trellis_tile_decode (down)
└── [inline atomic accumulation]
```

### moe_trellis_swiglu_prefill4 (line 3062)

```
moe_trellis_swiglu_prefill4
├── load_activation_tiles_prefill
├── load_trellis_tile (gate, up, down)
└── [inline atomic accumulation]
```

### moe_trellis_swiglu_prefill4_fp32acc (line 3278)

```
moe_trellis_swiglu_prefill4_fp32acc
├── load_activation_tiles_prefill
├── load_trellis_tile (gate, up, down)
└── [inline atomic accumulation]
```

### moe_trellis_swiglu_grouped (line 3512)

```
moe_trellis_swiglu_grouped
├── load_trellis_tile (gate, up, down)
└── [inline atomic accumulation]
```

### Routing Kernels (lines 3780, 3797, 3815)

No helper function calls - simple standalone kernels.

## Threadgroup Memory Patterns

All threadgroup memory is correctly declared inside kernel functions, not helpers. The helpers receive threadgroup memory via **reference parameters** (correct Metal practice):

```metal
// Example from load_trellis_tile_cached:
inline void load_trellis_tile_cached(
    device const uint8_t* packed_weights,
    threadgroup const half* scale_cache,    // Passed by pointer
    threadgroup const half* su_cache,       // Passed by pointer
    threadgroup const half* sv_cache,       // Passed by pointer
    device const half* grid,
    threadgroup half (&B_buf)[K][N_STRIDE], // Passed by reference
    ...
)
```

## Threadgroup Memory Sizes (from file header)

| Kernel | Memory Usage | Status |
|--------|-------------|--------|
| moe_trellis_swiglu | 6,688 bytes | 21% of 32KB |
| moe_trellis_swiglu_fp32acc | 7,200 bytes | 22% of 32KB |
| moe_trellis_swiglu_large_batch | 14,368 bytes | 44% of 32KB |
| moe_trellis_swiglu_simd | 4,224 bytes | 13% of 32KB |
| moe_trellis_swiglu_decode | 11,328 bytes | 35% of 32KB |
| moe_trellis_swiglu_prefill4 | 8,320 bytes | 26% of 32KB |
| moe_trellis_swiglu_prefill4_fp32acc | 10,368 bytes | 32% of 32KB |

## Observations

1. **No Invalid Threadgroup Declarations**: All helpers correctly receive threadgroup memory as parameters. There are no inline functions that declare `threadgroup` variables in their bodies.

2. **Code Duplication**: Several patterns are repeated:
   - `load_trellis_tile` vs `load_trellis_tile_cached` vs `load_trellis_tile_vec4` vs `load_trellis_tile_large`
   - `load_trellis_tile_decode` variants (5 functions)
   - Gate/up/down loading code is copy-pasted with different buffer targets

3. **Potential Refactoring**:
   - Template-based tile loader to reduce 9 loader variants
   - Shared SwiGLU computation helper
   - Unified prefetch infrastructure

4. **Unused Variables** (from compiler warnings):
   - Line 1236, 2952: `k_down_global`
   - Line 2624-2625: `gate_scale_base`, `up_scale_base`
   - Line 2935-2936: `down_ping`, `down_pong`
   - Line 2774: `token_idx` (in decode kernel)

## Recommendations

1. **Low Priority**: The file structure is correct. No blocking issues.

2. **Medium Priority**: Clean up unused variables to eliminate compiler warnings.

3. **Future Enhancement**: Consider templating the loader functions to reduce duplication while maintaining specialized code paths for different tile sizes.
