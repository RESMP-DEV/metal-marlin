# Trellis Dequantization Shader Audit

This document audits the Metal dequantization shaders for correctness against the Python-side quantization layout.

## Files Audited

- `src/gemm_trellis.metal` - Fused dequant+GEMM kernel
- `src/dequant_trellis.metal` - Standalone dequantization kernel
- `src/gemm_trellis_moe.metal` - MoE fused kernel with Trellis 3bpw

## 1. Tile Indexing

### Python Layout (exl3_quantizer.py)

The Python quantizer stores tiles in `[tiles_n, tiles_k, 256]` order:

```python
# exl3_quantizer.py:256-257
encoded = all_indices.reshape(tiles_n, tiles_k, 256)
```

Where tiles are extracted as:

```python
# exl3_quantizer.py:213-215
W_tiles = W_padded.reshape(tiles_n, tile_size, tiles_k, tile_size)
W_tiles = W_tiles.transpose(0, 2, 1, 3)  # [tiles_n, tiles_k, 16, 16]
```

**Python tile ordering:** `tile_idx = tile_n * tiles_k + tile_k`

### Metal Implementation

In `dequant_trellis.metal:135`:
```metal
uint tile_offset = tile_k * tiles_n + tile_n;
```

In `gemm_trellis.metal:337`:
```metal
uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
```

**Metal tile ordering:** `tile_idx = tile_k * tiles_n + tile_n`

### Discrepancy: TILE INDEXING MISMATCH

**Python uses:** `tile_n * tiles_k + tile_k` (row-major over N, then K)
**Metal uses:** `tile_k * tiles_n + tile_n` (row-major over K, then N)

This is a **CRITICAL BUG**. The tile ordering is transposed between Python and Metal.

**Fix Required:** Either:
1. Change Python to store as `[tiles_k, tiles_n, 256]` (match Metal), OR
2. Change Metal to use `tile_offset = tile_n * tiles_k + tile_k`

Note: Looking at `EXL3QuantResult` docstring suggests `[tiles_k, tiles_n, 256]` was intended:
```python
trellis_indices: NDArray[np.int16]  # [tiles_k, tiles_n, 256]
```

But the actual reshape at line 257 produces `[tiles_n, tiles_k, 256]`. The docstring is wrong.

## 2. Scale Indexing

### Python Layout (exl3_quantizer.py)

```python
# exl3_quantizer.py:201
scales = np.zeros((n_groups, out_features), dtype=np.float32)
```

Scales are indexed as `scales[group_idx, out_idx]` which linearizes to:
```python
scale_linear_idx = group_idx * out_features + out_idx
```

Where:
- `group_idx = k_idx / group_size` (groups along K dimension)
- `out_idx` = column index (N dimension)

### Metal Implementation

In `gemm_trellis.metal:341`:
```metal
uint scale_idx = group_idx * N + b_col;
float scale = scales[scale_idx];
```

In `dequant_trellis.metal:154`:
```metal
float scale = scales[group_idx * N + n_idx];
```

### Status: CORRECT

Metal correctly indexes scales as `group_idx * N + n_idx` which matches the Python layout `[n_groups, out_features]`.

Note: There's a comment in `gemm_trellis_moe.metal:129` that shows a different pattern:
```metal
half scale = scales[expert_id * N_dim * n_groups + group_idx * N_dim + global_n];
```

This is for the MoE case where scales have an additional expert dimension: `[num_experts, n_groups, N]`.

## 3. Sign Flip Indexing (su * sv)

### Python Layout

From `EXL3QuantResult`:
```python
su: NDArray[np.float64]  # Input sign flips [K]
sv: NDArray[np.float64]  # Output sign flips [N]
```

The dequantization formula from the Hadamard preprocessing:
```
W_dequant[k, n] = grid[idx] * scale * su[k] * sv[n]
```

Where:
- `su` has shape `[K]` (one sign per row/input dimension)
- `sv` has shape `[N]` (one sign per column/output dimension)
- `k` indexes the K dimension (row in weight matrix, input feature)
- `n` indexes the N dimension (column in weight matrix, output feature)

### Metal Implementation

In `gemm_trellis.metal:372-373`:
```metal
dequant_vals[row] = dequant_trellis_element(
    trellis_idx, scale, su_vec[row], sign_n, grid);
```

Where:
- `su_vec[row]` comes from `su[k_idx]` (line 352-353)
- `sign_n` comes from `sv[b_col]` (line 345)
- `row` iterates over K dimension
- `b_col` is the N dimension index

In `dequant_trellis.metal:514-516`:
```metal
float sign_k = su[k_idx];
float sign_n = sv[n_idx];
dequant_val *= sign_k * sign_n;
```

### Status: CORRECT

The Metal implementation correctly applies:
- `su[k]` for the K (row/input) dimension
- `sv[n]` for the N (column/output) dimension

The formula `grid[idx] * scale * su[k] * sv[n]` is correctly implemented.

## 4. Within-Tile Indexing (Transposed Layout)

### Issue: Potential Transpose Confusion

In `gemm_trellis.metal:366`:
```metal
uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;  // Transposed weight
```

In `dequant_trellis.metal:601`:
```metal
uint idx_in_tile = local_n * TILE_DIM + local_k;  // Transposed weight
```

In `gemm_trellis_moe.metal:107`:
```metal
uint idx_in_tile = n_in_tile * TRELLIS_TILE + k_in_tile;
```

The comment says "Transposed weight" suggesting the weight matrix W is stored transposed.

### Python Layout (trellis_tile.py)

The Python code flattens tiles row-major:
```python
# viterbi_quant.py:162
tiles_np = np.stack([t.data.reshape(256) for t in tiles], axis=0)
```

Where `t.data` is a 16x16 tile in row-major order. Flattening gives:
```
idx_in_tile = row * 16 + col
```

For a weight tile W[row, col] = W[k_local, n_local]:
```
idx_in_tile = k_local * 16 + n_local  (row-major)
```

### Discrepancy: WITHIN-TILE INDEXING

**Python (row-major):** `idx_in_tile = k_local * 16 + n_local`
**Metal (transposed):** `idx_in_tile = n_local * 16 + k_local`

This suggests the Metal kernel expects weights stored in column-major (transposed) order within tiles, but Python stores them row-major.

**Possible explanations:**
1. The GEMM kernel deliberately transposes B during dequant (common for GEMM efficiency)
2. There's a missing transpose step in the packing

Looking at the GEMM structure in `gemm_trellis.metal`, the B matrix (weights) is loaded column-by-column for the MMA operations. The "transposed" comment suggests this is intentional to match the GEMM's expected layout where B is accessed column-major.

**Verification needed:** Check if the TrellisLinear loader performs a transpose when loading weights. If so, this is correct. If not, this is a bug.

## 5. Packed Bytes Calculation

### Metal Implementation

In `gemm_trellis.metal:106-108`:
```metal
inline uint packed_bytes_per_trellis_tile(uint bits) {
    return (TRELLIS_TILE_SIZE * bits + 7) / 8;  // ceil(256 * bits / 8)
}
```

Results:
- 2-bit: 64 bytes
- 3-bit: 96 bytes
- 4-bit: 128 bytes

### Python/Test Files

From `test_trellis_linear.py:48`:
```python
packed_bytes = 128  # Hardcoded for 4-bit
```

### Status: CORRECT

The packed bytes calculation is correct. For 4-bit (the most common case), 256 elements * 4 bits / 8 = 128 bytes.

## Summary of Issues

| Issue | Severity | Status | Description |
|-------|----------|--------|-------------|
| Tile indexing | CRITICAL | MISMATCH | Python: `tile_n * tiles_k + tile_k`, Metal: `tile_k * tiles_n + tile_n` |
| Scale indexing | N/A | CORRECT | Both use `group_idx * N + n_idx` |
| Sign flip indexing | N/A | CORRECT | su[k] * sv[n] correctly applied |
| Within-tile indexing | MEDIUM | INVESTIGATE | Python row-major vs Metal column-major (transposed) |

## Recommended Actions

1. **CRITICAL:** Reconcile tile indexing between Python and Metal
   - Either change Python `exl3_quantizer.py:257` to `encoded = all_indices.reshape(tiles_k, tiles_n, 256)`
   - Or change Metal to use `tile_n * tiles_k + tile_k`

2. **INVESTIGATE:** Verify if TrellisLinear loader transposes weight tiles
   - If yes: document this requirement
   - If no: fix either Python packing or Metal unpacking

3. **UPDATE:** Fix the docstring in `EXL3QuantResult` to match actual output shape

## Test Verification

After fixing tile indexing, run:
```bash
cd contrib/metal_marlin && uv run pytest tests/test_trellis_linear.py -v
```

The tests currently show:
- Decode path (M=1): works
- Prefill path (M>16): produces Inf values (marked skip)

The tile indexing bug could be the root cause of the prefill failures.
