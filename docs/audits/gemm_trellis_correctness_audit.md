# Fused GEMM Trellis Shader Correctness Audit

**Date**: January 31, 2026
**File**: `src/gemm_trellis.metal`
**Kernels audited**: `gemm_trellis_packed`, `gemm_trellis_packed_fp32acc`, `gemm_trellis_packed_decode`

## Reference Computation

The explicit Python reference:

```python
for k in range(in_features):
    for n in range(out_features):
        dequant = grid[idx[k,n]] * scale[group[k], n] * su[k] * sv[n]
        output[n] += input[k] * dequant
```

For batched inputs with M rows:
```python
for m in range(M):
    for k in range(K):
        for n in range(N):
            dequant = grid[idx[k,n]] * scale[k // group_size, n] * su[k] * sv[n]
            output[m, n] += input[m, k] * dequant
```

This is equivalent to `Y[M,N] = X[M,K] @ W[K,N]` where `W[k,n] = dequant(k,n)`.

---

## Audit Findings

### 1. Dequantization Formula

**Shader implementation** (lines 116-124):
```metal
inline half dequant_trellis_element(
    uint idx, float scale, float su, float sv, device const float* grid
) {
    return half(grid[idx] * scale * su * sv);
}
```

**Verdict**: ✅ CORRECT

The formula `grid[idx] * scale * su * sv` exactly matches the reference `grid[idx[k,n]] * scale[group[k], n] * su[k] * sv[n]` when called with the appropriate per-element values.

---

### 2. Matrix Multiply Semantics (y = x @ W)

**Expected**: `C[m, n] = sum_k(A[m, k] * W[k, n])`

**Shader implementation**: The kernel uses Apple's simdgroup_multiply_accumulate intrinsic (line 406):
```metal
simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
```

The A fragment represents an 8x8 block from rows `[sg_row_offset + mi*8, sg_row_offset + mi*8 + 8)` and K indices `[kk*8, kk*8 + 8)`.

The B fragment represents an 8x8 block from K indices `[kk*8, kk*8 + 8)` and columns `[b_col_base, b_col_base + 8)`.

The MMA computes `acc += A_frag @ B_frag`, accumulating the contribution from 8 K elements.

**Verdict**: ✅ CORRECT

The shader correctly computes `C = A @ B` through tiled matrix multiplication.

---

### 3. Index-in-Tile Computation

**Documentation** (trellis_kernels.md:234):
```
idx_in_tile = local_k * TILE_DIM + local_n  # Row-major within tile
```

**Shader implementation** (line 366):
```metal
uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;  // Transposed weight
```

**Analysis**: The shader uses column-major indexing (`n * 16 + k`), while the documentation shows row-major (`k * 16 + n`). The comment "Transposed weight" is key.

**Interpretation**: The packed_indices tensor stores indices in a layout where column n varies slowest within each tile. This means either:
1. The indices are packed in column-major order, OR
2. The original weights were transposed before packing

Given that GEMM computes `Y = X @ W`, and the kernel loads B as `W[k,n]`, if the physical storage is `W^T[n,k]` (column-major), then indexing with `n * TILE_DIM + k` is correct.

**Verdict**: ✅ CORRECT (WITH TRANSPOSE)

The Metal shader uses column-major indexing (`local_n * 16 + local_k`) intentionally. This is documented in `scripts/debug_kernel_params.py`:

```
Trellis tile indexing (TRANSPOSED):
    idx_in_tile = local_n * 16 + local_k
    This means tiles store weights in [out_features/16, in_features/16] order
    but elements within tiles are stored column-major for efficient column loading.
```

The naming convention difference also matters:
- TrellisWeight loader: K=out_features, N=in_features (weight matrix perspective)
- GEMM kernel: K=inner_dim, N=output_cols (standard GEMM notation)

For `y = x @ W^T`, the loader's K maps to GEMM's N and vice versa. The column-major tile indexing compensates for this transposition.

The Python reference in `dispatch.py:738` uses `local_k * TILE_DIM + local_n`, which is correct for its own context (weight matrix perspective). The Metal kernel's indexing handles the transposition needed for GEMM.

**Note**: The documentation in `trellis_kernels.md:234` should be updated to clarify this transposed indexing convention, or add a note about the naming conventions.

---

### 4. Accumulation Order

The shader uses a triply-nested loop structure:

```
for kt in range(num_k_tiles):           # K tiles (32 elements each)
    for kk in range(K_TILES):           # K sub-tiles (8 elements each)
        load A fragments
        for ni in range(SG_N_TILES):    # N sub-tiles (8 columns each)
            dequant B tile
            acc[mi][ni] += A[mi] @ B
```

**Verdict**: ✅ CORRECT

The K dimension is fully reduced: all contributions from k=0 to k=K-1 are accumulated before storing the result. The order of accumulation (which K value is processed first) does not affect the mathematical result (floating-point non-associativity aside).

---

### 5. Scale Indexing

**Shader implementation** (lines 303, 341-342):
```metal
uint group_idx = k_sub_base / group_size;
uint scale_idx = group_idx * N + b_col;
float scale = scales[scale_idx];
```

**Reference**: `scale[group[k], n]` where `group[k] = k // group_size`

**Verdict**: ✅ CORRECT

The scale tensor layout is `[n_groups, N]` row-major. The shader correctly computes `group_idx = k // group_size` and indexes with `group_idx * N + n`.

---

### 6. Sign Vector (su/sv) Indexing

**Shader implementation** (lines 345, 348-353):
```metal
float sign_n = sv[b_col];

float su_vec[8];
for (uint row = 0; row < 8; ++row) {
    uint k_idx = k_sub_base + row;
    su_vec[row] = (k_idx < K) ? su[k_idx] : 0.0f;
}
```

**Reference**: `su[k]` and `sv[n]`

**Verdict**: ✅ CORRECT

Sign vectors are indexed directly by k and n indices. Boundary checking for k < K is present.

---

### 7. Trellis Tile Boundary Handling

When k_sub_base spans a trellis tile boundary (multiples of 16), the shader handles this correctly (lines 364-366):

```metal
uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
uint actual_tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;
```

For each of the 8 rows in the sub-tile, it recalculates:
1. `actual_tile_k` - which trellis tile contains this k index
2. `actual_tile_offset` - byte offset to that tile in packed_indices
3. `local_k` - position within the trellis tile (using modulo)

**Verdict**: ✅ CORRECT

Boundary crossings are handled by per-element tile offset recalculation.

---

### 8. Boundary Checks

| Location | Check | Status |
|----------|-------|--------|
| Line 181 | A tile load: `global_row < M && global_col < K` | ✅ |
| Line 245 | Threadgroup early exit: `tg_row >= M` | ✅ |
| Line 331 | B column bounds: `b_col < N` | ✅ |
| Line 353 | K bounds for su: `k_idx < K` | ✅ |
| Line 362 | K bounds for dequant: `k_idx < K` | ✅ |
| Line 370 | Index overflow: `trellis_idx >= n_levels` | ✅ |
| Line 432-451 | Epilogue partial tile handling | ✅ |

**Verdict**: ✅ CORRECT

All boundary conditions are properly handled.

---

### 9. Staging Buffer Layout

**B_staging layout** (line 387-391):
```metal
// Layout: B_staging[simd_id][k][n] where k,n in [0,7]
for (uint row = 0; row < 8; ++row) {
    B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
}
```

**simdgroup_load** (line 399):
```metal
simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);
```

The B fragment is loaded with stride 8, meaning it reads `B_staging[sg][k][0:8]` for k in 0..7. Each lane wrote its dequant column into `B_staging[sg][row][lane]`.

**Verdict**: ✅ CORRECT

The layout `B_staging[sg][k][n]` matches the load pattern. The B fragment represents B[k, n] for k,n in [0,8).

---

## Summary

| Item | Status | Notes |
|------|--------|-------|
| Dequant formula | ✅ | Exact match |
| Matrix multiply semantics | ✅ | Correct A @ B |
| Index-in-tile computation | ✅ | Column-major for transpose (documented in debug_kernel_params.py) |
| Accumulation order | ✅ | K fully reduced |
| Scale indexing | ✅ | Correct [group, n] layout |
| Sign vector indexing | ✅ | Direct k and n indexing |
| Tile boundary handling | ✅ | Per-element recalculation |
| Boundary checks | ✅ | Comprehensive |
| Staging buffer layout | ✅ | Correct B[k,n] layout |

**Overall**: The shader is mathematically correct for computing `Y = X @ dequant(W)`.

**Recommendation**: Update `trellis_kernels.md` section 4.3 to clarify that the tile indexing formula depends on the caller's naming convention, and reference `scripts/debug_kernel_params.py` which documents the transposition behavior.

---

## Appendix: Numerical Precision

The shader offers two variants:
- `gemm_trellis_packed`: FP16 accumulation
- `gemm_trellis_packed_fp32acc`: FP32 accumulation with FP16 output

For large K (>1024), the FP32 variant should be used to avoid accumulation overflow. The dequantization itself is performed in FP32 before conversion to FP16 (line 123):

```metal
return half(grid[idx] * scale * su * sv);
```

This maintains precision during the scale/sign multiplication before truncation to FP16.
