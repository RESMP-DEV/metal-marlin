# Trellis Kernel Architecture

This document describes the Metal kernel architecture for Trellis-quantized GEMM operations.
Trellis is the quantization format used by EXL3, which stores weights using a Viterbi-encoded
codebook approach with 16x16 tiles.

## 1. Trellis Quantization Format

### 1.1 Overview

Trellis quantization encodes weights using:
- **Packed indices**: Codebook lookups packed at 2, 3, or 4 bits per element
- **Scales**: Per-group scale factors (typically 32 or 128 elements per group)
- **Grid**: Precomputed codebook of quantization centers
- **Sign vectors**: su (row signs) and sv (column signs) for Hadamard inverse

The dequantization formula is:

```
w[k, n] = grid[idx] * scale * su[k] * sv[n]
```

Where `idx` is the unpacked trellis index for position (k, n).

### 1.2 Tile Structure

Weights are organized in 16x16 tiles:

```
packed_indices: [tiles_k, tiles_n, packed_bytes] uint8
                 │         │        │
                 │         │        └─ Bytes per tile: 64 (2-bit), 96 (3-bit), 128 (4-bit)
                 │         └─ ceil(N / 16)
                 └─ ceil(K / 16)
```

Each tile contains 256 elements (16 × 16). The packed byte size depends on bit width:

| Bits | Indices per Tile | Packed Bytes | Formula |
|------|------------------|--------------|---------|
| 2    | 256              | 64           | (256 × 2) / 8 |
| 3    | 256              | 96           | (256 × 3) / 8 |
| 4    | 256              | 128          | (256 × 4) / 8 |

### 1.3 Index Packing

Indices are packed LSB-first within each byte:

**2-bit**: 4 indices per byte
```
Byte layout: [idx0:2] [idx1:2] [idx2:2] [idx3:2]
             bits 0-1  bits 2-3  bits 4-5  bits 6-7
```

**3-bit**: 8 indices per 3 bytes (24 bits)
```
Indices span byte boundaries. Bit offset = idx_in_tile * 3.
Read 2 bytes if the index crosses a byte boundary.
```

**4-bit**: 2 indices per byte
```
Byte layout: [idx0:4] [idx1:4]
             bits 0-3  bits 4-7
```


## 2. Kernel Dispatch Flow

### 2.1 TrellisLinear.forward()

The `TrellisLinear` module (trellis/linear.py:238) dispatches to different kernels based on
batch size M:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    M = x.view(-1, self.in_features).shape[0]

    if M <= 16:
        # Decode kernel: optimized for autoregressive generation
        output = dispatch_gemm_trellis_decode(...)
    else:
        # Prefill kernel: optimized for batch processing
        output = dispatch_gemm_trellis_packed(...)
```

### 2.2 Dispatch Functions

The dispatch layer (trellis/dispatch.py) handles:

1. **Buffer creation**: Converts PyTorch MPS tensors to Metal buffers
2. **Grid sizing**: Computes threadgroup grid based on tile dimensions
3. **Parameter buffers**: Creates uint32 constant buffers for kernel parameters

```python
# Prefill: 64x64 output tiles
TILE_M, TILE_N = 64, 64
grid_x = (N + TILE_N - 1) // TILE_N
grid_y = (M + TILE_M - 1) // TILE_M

# Decode: 32x128 output tiles (wider for small M)
TILE_M, TILE_N = 32, 128
grid_x = (N + TILE_N - 1) // TILE_N
grid_y = (M + TILE_M - 1) // TILE_M
```


## 3. Metal Kernel Architecture

### 3.1 gemm_trellis_packed (Prefill Kernel)

Located in `src/gemm_trellis.metal`. Optimized for large M (batch processing, prefill phase).

**Tile dimensions:**
- Output tile: 64 × 64
- K reduction: 32 elements per iteration
- Threadgroup: 4 simdgroups × 32 threads = 128 threads

**Threadgroup memory layout:**
```
A_tiles[2][64][32]        ← Double-buffered A tiles (8 KB)
B_staging[4][8][8]        ← Per-simdgroup B staging (512 B)
epilogue_staging[4][8][8] ← Output staging (512 B)
──────────────────────────
Total: ~9 KB (well under 32 KB limit)
```

**Simdgroup assignment:**
```
64x64 output tile split into 2x2 grid of 32x32 sub-tiles:

  SG 0: rows [0,31],  cols [0,31]
  SG 1: rows [0,31],  cols [32,63]
  SG 2: rows [32,63], cols [0,31]
  SG 3: rows [32,63], cols [32,63]
```

**Pipeline:**
1. Cooperative A tile load (all 128 threads)
2. For each K sub-tile (8 elements):
   - Lanes 0-7 dequantize B column from packed indices
   - Write to per-simdgroup staging buffer
   - simdgroup_barrier (32 threads only)
   - simdgroup_load B fragment
   - simdgroup_multiply_accumulate with A fragment

### 3.2 gemm_trellis_packed_decode (Decode Kernel)

Optimized for small M (autoregressive decode, M = 1-16).

**Tile dimensions:**
- Output tile: 32 × 128 (narrower M, wider N)
- Each simdgroup handles 32 × 32 output region
- 4 simdgroups cover the 128-wide N dimension

**Key differences from prefill:**
- Smaller M tile reduces wasted compute for M=1 case
- Wider N tile maximizes memory bandwidth utilization
- Same fused dequant + GEMM approach

### 3.3 Dequantization Flow

Each lane dequantizes one column of an 8×8 B sub-tile:

```metal
if (simd_lane_id < 8) {
    uint b_col = b_col_base + simd_lane_id;
    half dequant_vals[8];

    // Calculate trellis tile coordinates
    uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
    uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
    uint local_n = b_col % TRELLIS_TILE_DIM;

    // Compute packed offset
    uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;

    // Dequant 8 elements
    for (uint row = 0; row < 8; ++row) {
        uint idx_in_tile = local_k * TRELLIS_TILE_DIM + local_n;
        uint trellis_idx = unpack_trellis_index(packed + tile_offset, idx_in_tile, bits);
        dequant_vals[row] = grid[trellis_idx] * scale * su[k_idx] * sv[b_col];
    }

    // Write to staging
    for (uint row = 0; row < 8; ++row) {
        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
    }
}
```


## 4. Memory Layout Expectations

### 4.1 Input Tensors

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| A | [M, K] | float16 | Row-major activations |
| packed_indices | [tiles_k, tiles_n, packed_bytes] | uint8 | Packed trellis indices |
| scales | [n_groups, N] | float32 | Per-group scale factors |
| grid | [n_levels] | float32 | Codebook (e.g., 8 values for 3-bit) |
| su | [K] | float32 | Row sign vector |
| sv | [N] | float32 | Column sign vector |

### 4.2 packed_bytes Calculation

```python
TILE_SIZE = 256  # 16 × 16 elements per tile

def packed_bytes_per_tile(bits: int) -> int:
    return (TILE_SIZE * bits + 7) // 8

# Examples:
# 2-bit: (256 * 2 + 7) // 8 = 64 bytes
# 3-bit: (256 * 3 + 7) // 8 = 96 bytes
# 4-bit: (256 * 4 + 7) // 8 = 128 bytes
```

### 4.3 Tile Indexing

To access element at position (k, n):

```python
TILE_DIM = 16
tile_k = k // TILE_DIM
tile_n = n // TILE_DIM
local_k = k % TILE_DIM
local_n = n % TILE_DIM

tiles_n = (N + TILE_DIM - 1) // TILE_DIM
tile_offset = (tile_k * tiles_n + tile_n) * packed_bytes

idx_in_tile = local_k * TILE_DIM + local_n  # Row-major within tile
```


## 5. Common Failure Modes and Debugging

### 5.1 NaN/Inf Output

**Symptoms**: Output contains NaN or infinite values.

**Causes**:
1. **Codebook index overflow**: Index exceeds n_levels
2. **Scale explosion**: Scales contain extreme values
3. **Sign vector corruption**: su/sv contain non-±1 values

**Debugging**:
```bash
# Enable debug logging
METAL_DEBUG=1 python your_script.py
```

The dispatch functions log pre/post values when `METAL_DEBUG=1`:
```
=== gemm_trellis_packed_decode PRE-DISPATCH ===
A: min=-0.5, max=0.5, has_nan=False
packed_indices: shape=[512, 256, 96], first_10_bytes=[...]
scales: min=-1.2, max=1.2, has_nan=False
grid: values=[-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
su: min=-1.0, max=1.0, unique=[-1.0, 1.0]
sv: min=-1.0, max=1.0, unique=[-1.0, 1.0]
```

**Fixes**:
- Clamp trellis indices: `if (trellis_idx >= n_levels) trellis_idx = 0;`
- Verify grid values are normalized
- Check su/sv contain only ±1

### 5.2 Incorrect Output Dimensions

**Symptoms**: Shape mismatch or garbage in output edges.

**Causes**:
1. **tiles_k/tiles_n mismatch**: Packed indices shape doesn't match K/N
2. **group_size mismatch**: Scale groups don't align with K dimension

**Debugging**:
```python
# Verify shapes match
tiles_k = (K + 15) // 16
tiles_n = (N + 15) // 16
assert packed_indices.shape[0] == tiles_k
assert packed_indices.shape[1] == tiles_n
assert packed_indices.shape[2] == packed_bytes_per_tile(bits)

# Verify scales
n_groups = scales.shape[0]
expected_groups = (K + group_size - 1) // group_size
assert n_groups == expected_groups
```

### 5.3 Threadgroup Memory Overflow

**Symptoms**: Kernel crashes or undefined behavior.

**Causes**: Tile dimensions exceed 32 KB threadgroup memory limit.

**Current budget (gemm_trellis_packed)**:
```
A_tiles[2][64][32] * 2B     = 8,192 bytes
B_staging[4][8][8] * 2B     =   512 bytes
epilogue_staging[4][8][8]   =   512 bytes
────────────────────────────────────────────
Total                        ≈ 9,216 bytes (OK)
```

### 5.4 Performance Degradation

**Symptoms**: Slow inference, not hitting expected throughput.

**Causes**:
1. **Wrong kernel selection**: Using prefill kernel for M=1
2. **Uncoalesced memory access**: Packed indices not contiguous
3. **Excessive synchronization**: Too many barriers

**Debugging**:
```bash
# Profile with Metal System Trace
xcrun metal-profiler capture -o trace.gputrace your_app
```

**Fixes**:
- Ensure M <= 16 uses decode kernel
- Verify packed_indices tensor is contiguous
- Check simdgroup_barrier vs threadgroup_barrier usage


## 6. Performance Characteristics

### 6.1 Prefill vs Decode Kernels

| Kernel | Tile Size | Best For | Bottleneck |
|--------|-----------|----------|------------|
| gemm_trellis_packed | 64×64 | M > 16 (prefill) | Compute |
| gemm_trellis_packed_decode | 32×128 | M ≤ 16 (decode) | Memory bandwidth |

### 6.2 Throughput Expectations (M4 Max)

| Bit Width | Prefill (M=512) | Decode (M=1) |
|-----------|-----------------|--------------|
| 3-bit     | ~15 TFLOPS      | ~800 tok/s   |
| 4-bit     | ~14 TFLOPS      | ~750 tok/s   |

### 6.3 Memory Efficiency

Trellis format provides significant memory savings:

| Bit Width | FP16 Equivalent | Compression Ratio |
|-----------|-----------------|-------------------|
| 2-bit     | 16-bit          | 8:1               |
| 3-bit     | 16-bit          | 5.33:1            |
| 4-bit     | 16-bit          | 4:1               |

### 6.4 L2 Cache Considerations

The fused dequant approach relies on L2 cache for repeated B tile access:
- Each packed uint32 may be loaded multiple times (once per M sub-tile)
- Working set per K-block: 32 × N_tile / 8 × packed_bytes ≈ 1-2 KB
- Apple Silicon L2 (16-64 MB) easily absorbs this working set

For memory-bound decode workloads, the L2 hit rate is critical. Profile with
Metal performance counters to verify cache behavior.


## 7. MoE Extension

The MoE kernel (`src/gemm_trellis_moe.metal`) extends the base Trellis GEMM for
Mixture-of-Experts architectures:

- Per-expert weight storage with contiguous layout
- Token routing via expert_ids and expert_probs tensors
- Fused SwiGLU activation (gate_proj, up_proj, down_proj)
- Probability-weighted output accumulation

See `TrellisParams` struct for MoE-specific parameters:

```metal
struct TrellisParams {
    uint M;           // Batch size (tokens)
    uint K;           // Input hidden dimension
    uint N;           // Output dimension
    uint num_experts; // Total experts
    uint top_k;       // Experts per token
    uint bits;        // 2, 3, or 4
    uint group_size;  // Quantization group size
    uint n_levels;    // Codebook levels
};
```


## References

- `src/gemm_trellis.metal`: Main GEMM kernels
- `src/gemm_trellis_moe.metal`: MoE variant
- `trellis/dispatch.py`: Python dispatch layer
- `trellis/linear.py`: TrellisLinear module
- `trellis/packing.py`: Index packing utilities
