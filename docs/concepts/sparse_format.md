# 2:4 Structured Sparse Weight Format for Metal Marlin

This document specifies the memory layout for 2:4 structured sparse weights in
Metal Marlin's FP4/INT4 quantized GEMM kernels, and explains how metadata
decoding interleaves with value dequantization during the GEMM inner loop.

Reference implementation: `src/sparse.metal`


## 1. Structured Sparsity Overview

NVIDIA's 2:4 structured sparsity keeps exactly 2 non-zero values per 4-element
block. This achieves 2x compression of the weight matrix with hardware-friendly
access patterns: every 4-element block produces exactly 2 values, so the sparse
representation has a fixed, predictable layout regardless of which elements are
non-zero.

The 6 valid patterns within a 4-element block (positions of the 2 kept values):

```
(0,1)  (0,2)  (0,3)  (1,2)  (1,3)  (2,3)
```

All other combinations are invalid for 2:4 sparsity.


## 2. Dense Format Recap

For a K x N weight matrix in dense FP4:

- Each FP4 value occupies 4 bits
- 8 values pack into one uint32 (8 * 4 = 32 bits)
- Storage shape: `[K/8, N]` uint32 values
- Total storage: `K * N / 8` uint32s = `K * N / 2` bytes

The packing within a uint32 stores 8 consecutive values along the K dimension:

```
uint32 word at [k_group, n]:
  bits [3:0]   = weight[k_group*8 + 0, n]
  bits [7:4]   = weight[k_group*8 + 1, n]
  bits [11:8]  = weight[k_group*8 + 2, n]
  bits [15:12] = weight[k_group*8 + 3, n]
  bits [19:16] = weight[k_group*8 + 4, n]
  bits [23:20] = weight[k_group*8 + 5, n]
  bits [27:24] = weight[k_group*8 + 6, n]
  bits [31:28] = weight[k_group*8 + 7, n]
```


## 3. Sparse Format

The sparse format consists of two arrays: compressed values and metadata.

### 3.1 Values Array

The values array stores only the 2 non-zero values per 4-element block, packed
identically to the dense format but at half the size. Since 2:4 sparsity keeps
50% of values, each group of 8 dense K-elements (2 blocks of 4) produces 4
sparse values, which pack into one uint32.

- Storage shape: `[K/16, N]` uint32 values
- Total storage: `K * N / 16` uint32s = `K * N / 4` bytes (half of dense)

Packing within a uint32 for sparse values at `[k_sparse_group, n]`:

```
Dense K range covered: [k_sparse_group*16, k_sparse_group*16 + 15]
This spans 4 blocks of 4 elements, producing 4*2 = 8 sparse values.

Wait -- correction: 16 dense elements / 4 per block = 4 blocks.
4 blocks * 2 values each = 8 sparse values = one uint32.
```

More precisely, each uint32 in the values array covers 16 dense K-positions
(4 blocks), storing the 8 non-zero values:

```
uint32 word at [k_sparse_group, n]:
  bits [3:0]   = block0_val0  (first non-zero in dense[k*16+0..3, n])
  bits [7:4]   = block0_val1  (second non-zero in dense[k*16+0..3, n])
  bits [11:8]  = block1_val0  (first non-zero in dense[k*16+4..7, n])
  bits [15:12] = block1_val1  (second non-zero in dense[k*16+8..11, n])
  bits [19:16] = block2_val0  (first non-zero in dense[k*16+8..11, n])
  bits [23:20] = block2_val1  (second non-zero in dense[k*16+8..11, n])
  bits [27:24] = block3_val0  (first non-zero in dense[k*16+12..15, n])
  bits [31:28] = block3_val1  (second non-zero in dense[k*16+12..15, n])
```

### 3.2 Metadata Array

The metadata encodes which 2 of 4 positions are non-zero within each block.
Each block requires 4 bits of metadata: two 2-bit position indices.

```
Per-block metadata (4 bits):
  bits [1:0] = pos0 (position of first kept value, 0-3)
  bits [3:2] = pos1 (position of second kept value, 0-3)
```

A single uint32 packs metadata for 8 blocks (8 * 4 = 32 bits), covering
8 * 4 = 32 dense K-elements.

- Storage shape: `[K/32, N]` uint32 metadata words
- Equivalently: `[K/16, N/2]` if you prefer matching the values array K-stride
- Total storage: `K * N / 32` uint32s = `K * N / 8` bytes

Alternatively, express as: for every 2 value uint32s along N, there is 1
metadata uint32.

### 3.3 Combined Layout

For a K x N weight matrix:

| Component | Shape (uint32) | Bytes | Ratio vs Dense |
|-----------|----------------|-------|----------------|
| Dense FP4 | [K/8, N] | K*N/2 | 1.0x |
| Sparse values | [K/16, N] | K*N/4 | 0.5x |
| Sparse metadata | [K/32, N] | K*N/8 | 0.125x |
| **Sparse total** | | **K*N*3/8** | **0.625x** |

The sparse format achieves 1.6x compression (62.5% of dense size), not quite
2x, due to the metadata overhead. The metadata is 25% of the compressed values
size.

For INT4 the arithmetic is identical (same 4-bit packing).


## 4. Metadata Encoding Details

### 4.1 Worked Example

Original 4-element block: `[a, 0, b, 0]` where a, b are non-zero.

- pos0 = 0 (position of `a`)
- pos1 = 2 (position of `b`)
- Metadata nibble = `(pos1 << 2) | pos0` = `(2 << 2) | 0` = `0b1000` = 8

Values stored: `[a, b]` in consecutive nibble positions.

To reconstruct the dense block:
1. Read metadata nibble: 8 = `0b1000`
2. Decode: pos0 = `8 & 0x3` = 0, pos1 = `(8 >> 2) & 0x3` = 2
3. Place val0 at position 0, val1 at position 2, zeros elsewhere
4. Result: `[a, 0, b, 0]`

### 4.2 All Valid Encodings

| Pattern | pos0 | pos1 | Nibble (binary) | Nibble (decimal) |
|---------|------|------|-----------------|------------------|
| (0,1) | 0 | 1 | 0100 | 4 |
| (0,2) | 0 | 2 | 1000 | 8 |
| (0,3) | 0 | 3 | 1100 | 12 |
| (1,2) | 1 | 2 | 1001 | 9 |
| (1,3) | 1 | 3 | 1101 | 13 |
| (2,3) | 2 | 3 | 1110 | 14 |

Note: pos0 < pos1 is canonical. Values 0-3, 5-7, 10-11, 15 are unused/invalid
metadata nibbles.

### 4.3 Metadata Packing into uint32

Eight consecutive blocks' metadata nibbles pack into one uint32, LSB-first:

```
uint32 metadata word:
  bits [3:0]   = block 0 metadata nibble
  bits [7:4]   = block 1 metadata nibble
  bits [11:8]  = block 2 metadata nibble
  bits [15:12] = block 3 metadata nibble
  bits [19:16] = block 4 metadata nibble
  bits [23:20] = block 5 metadata nibble
  bits [27:24] = block 6 metadata nibble
  bits [31:28] = block 7 metadata nibble
```

Extraction: `nibble_i = (metadata_word >> (i * 4)) & 0xF`


## 5. Interleaving Metadata Decode with Value Dequant

The critical design goal is to hide metadata decode latency within the
dequantization pipeline. On Apple Silicon, integer ALU and FP16 ALU can
execute concurrently on different execution units.

### 5.1 The Dense Dequant Pipeline

In the dense kernel, the inner loop processes one uint32 of packed FP4 values:

```
1. Load uint32 from weight buffer (8 FP4 values)
2. Dequant pairs via bitwise extraction (4 half2 results)
3. Apply per-group scale
4. Feed to simdgroup_multiply_accumulate
```

### 5.2 The Sparse Dequant Pipeline

The sparse kernel must additionally decode metadata to know where each
dequantized value belongs in the logical K-dimension:

```
1. Load uint32 from values buffer (8 sparse FP4 values, covering 16 K-positions)
2. Load uint32 from metadata buffer (covers 8 blocks = 32 K-positions)
3. Decode metadata: extract 4 nibbles (for the 4 blocks this values word covers)
4. Dequant the 8 sparse values into 4 half2 pairs
5. Scatter: place each value at its correct K-position within the 4-wide block
6. Apply per-group scale
7. Feed expanded 16-element column to simdgroup_multiply_accumulate
```

### 5.3 Interleaving Schedule

The metadata decode (step 3) uses integer ALU only. The dequant (step 4) uses
bitwise integer ops followed by FP16 multiply. These can be interleaved:

```metal
// Load both arrays simultaneously (independent memory ops)
uint32_t vals = values_buf[val_idx];
uint32_t meta = metadata_buf[meta_idx];

// --- Begin interleaved decode+dequant ---

// Decode first 2 block positions (integer ALU)
uint2 pos_block0 = uint2(meta & 0x3u, (meta >> 2) & 0x3u);
uint2 pos_block1 = uint2((meta >> 4) & 0x3u, (meta >> 6) & 0x3u);

// Dequant first 2 value pairs (bitwise + FP16 mul, overlaps with above on GPU)
half2 pair0 = dequant_fp4_pair_biased(prepare_pair(vals, 0));
half2 pair1 = dequant_fp4_pair_biased(prepare_pair(vals, 1));

// Decode next 2 block positions
uint2 pos_block2 = uint2((meta >> 8) & 0x3u, (meta >> 10) & 0x3u);
uint2 pos_block3 = uint2((meta >> 12) & 0x3u, (meta >> 14) & 0x3u);

// Dequant next 2 value pairs
half2 pair2 = dequant_fp4_pair_biased(prepare_pair(vals, 2));
half2 pair3 = dequant_fp4_pair_biased(prepare_pair(vals, 3));

// --- Scatter to dense representation ---
// Each block expands 2 sparse values → 4 dense positions
half dense_k[16];  // 16 K-positions covered by this uint32 pair
for (uint i = 0; i < 16; ++i) dense_k[i] = 0.0h;

dense_k[0 * 4 + pos_block0.x] = pair0[0];
dense_k[0 * 4 + pos_block0.y] = pair0[1];
dense_k[1 * 4 + pos_block1.x] = pair1[0];
dense_k[1 * 4 + pos_block1.y] = pair1[1];
dense_k[2 * 4 + pos_block2.x] = pair2[0];
dense_k[2 * 4 + pos_block2.y] = pair2[1];
dense_k[3 * 4 + pos_block3.x] = pair3[0];
dense_k[3 * 4 + pos_block3.y] = pair3[1];
```

### 5.4 Why Not Scatter-Free Accumulation?

An alternative to scattering is to accumulate sparse values directly into the
output without reconstructing the dense column. This avoids the scatter step
but requires separate MMA calls per sparse position, which underutilizes the
simdgroup matrix units.

The scatter approach is preferred because:
1. simdgroup_multiply_accumulate expects contiguous 8x8 tiles
2. The scatter is register-only (no memory traffic)
3. 16 half values fit in 8 registers; the zero-fill is 4 vector moves
4. Apple Silicon's register file is large enough to hold the expanded block

### 5.5 Metadata Access Pattern

The metadata array is accessed at half the K-rate of the values array. For every
uint32 of values (covering 16 dense K-positions = 4 blocks), we need 4 nibbles
from the metadata. Since one metadata uint32 covers 8 blocks (32 K-positions),
two consecutive value loads share one metadata load:

```
Iteration 0: values[0], metadata[0] nibbles [0:3]  → K-positions [0:15]
Iteration 1: values[1], metadata[0] nibbles [4:7]  → K-positions [16:31]
Iteration 2: values[2], metadata[1] nibbles [0:3]  → K-positions [32:47]
Iteration 3: values[3], metadata[1] nibbles [4:7]  → K-positions [48:63]
```

This means metadata bandwidth is only 1/2 of values bandwidth. In practice,
metadata loads can be double-buffered alongside values loads with minimal
overhead.


## 6. Address Computation

Given a thread processing output column `n` at K-iteration `k_iter` (where each
iteration covers 16 dense K-positions):

```metal
// Values address: straightforward row-major
uint val_idx = k_iter * N + n;
// → values_buf[val_idx] gives 8 FP4 sparse values

// Metadata address: 2x fewer along K, 1x along N
uint meta_k = k_iter / 2;       // two value iterations share one metadata word
uint meta_nibble_offset = (k_iter % 2) * 4;  // which 4 nibbles to use (0 or 4)
uint meta_idx = meta_k * N + n;
// → metadata_buf[meta_idx] gives 8 block metadata nibbles
// → use nibbles [meta_nibble_offset : meta_nibble_offset+3]
```

For the tiled GEMM with TILE_K steps per iteration, the K-loop advances by
TILE_K/16 value words per step and TILE_K/32 metadata words per step.


## 7. Scale/Group Quantization Interaction

Per-group quantization scales apply to groups of G consecutive K-elements. With
2:4 sparsity, the group boundaries are in dense K-space. Since the sparse format
stores values in dense-K order (just with zeros removed), the group index for a
sparse value at dense position `k_dense` is `k_dense / G`.

The scatter step reconstructs the dense K-position, so scale lookup uses the
same group index as the dense kernel. No special handling is needed: after
scatter, the value is at its correct K-position and the scale lookup proceeds
normally.


## 8. Pruning and Weight Preparation

### 8.1 Selecting the 2:4 Pattern

Given a dense FP4 weight matrix, the pruning step selects which 2 of 4 values to
keep per block. For each 4-element block along K:

1. Compute magnitude of each value: `|w[k, n]|` for k in block
2. Keep the 2 largest magnitudes (ties broken by lower index)
3. Record positions as metadata

This is done offline during model preparation, not at inference time.

### 8.2 Packing Procedure

```python
def pack_sparse_fp4(dense_weights: ndarray, group_size: int) -> tuple[ndarray, ndarray]:
    """Convert dense FP4 weights to sparse format.

    Args:
        dense_weights: [K, N] array of FP4 values (stored as uint8, 2 per byte)
        group_size: quantization group size along K

    Returns:
        values: [K//16, N] uint32 array of packed sparse FP4 values
        metadata: [K//32, N] uint32 array of packed metadata
    """
    K, N = dense_weights.shape
    assert K % 16 == 0, "K must be multiple of 16 for 2:4 sparse packing"

    values = np.zeros((K // 16, N), dtype=np.uint32)
    metadata = np.zeros((K // 32, N), dtype=np.uint32)

    for n in range(N):
        for block_start in range(0, K, 4):
            block = dense_weights[block_start:block_start+4, n]

            # Find top-2 by magnitude
            magnitudes = np.abs(block)
            top2_indices = np.argsort(magnitudes)[-2:]  # ascending, take last 2
            top2_indices = np.sort(top2_indices)  # canonical order: pos0 < pos1

            pos0, pos1 = top2_indices
            val0, val1 = block[pos0], block[pos1]

            # Pack values
            block_idx = block_start // 4
            sparse_pair_idx = block_idx % 4  # which pair within the uint32
            val_word_idx = block_start // 16
            bit_offset = sparse_pair_idx * 8  # 2 nibbles = 8 bits per block
            values[val_word_idx, n] |= (int(val0) & 0xF) << bit_offset
            values[val_word_idx, n] |= (int(val1) & 0xF) << (bit_offset + 4)

            # Pack metadata
            nibble = (pos1 << 2) | pos0
            meta_block_idx = block_idx % 8  # which nibble within the uint32
            meta_word_idx = block_start // 32
            metadata[meta_word_idx, n] |= nibble << (meta_block_idx * 4)

    return values, metadata
```

### 8.3 Buffer Allocation

For inference, allocate three Metal buffers:

```python
# For K x N weight matrix
values_buf = device.make_buffer(K * N // 4)      # sparse FP4 values
metadata_buf = device.make_buffer(K * N // 8)    # 2:4 metadata
scales_buf = device.make_buffer(K * N // group_size * 2)  # FP16 scales
```


## 9. Performance Considerations

### 9.1 Bandwidth Savings

For a memory-bound kernel (batch size 1), the sparse format reduces weight
loads from `K*N/2` bytes to `K*N*3/8` bytes (values + metadata), a 25%
bandwidth reduction. This directly translates to 1.33x speedup for
memory-bandwidth-limited inference.

### 9.2 Compute Overhead

The scatter step adds 16 register writes (8 zeros + 8 value placements) per
uint32 of values processed. On Apple Silicon, register moves are essentially
free (1 cycle each, fully pipelined). The metadata decode adds 4-8 shift+mask
operations per 8 values, also ~1 cycle each.

Total overhead: ~24 cycles per 16 dense K-positions reconstructed. At ~1 GHz
shader clock, this is 24 ns for 16 FP16 values. The memory load of 8 bytes
(uint32 values + uint32 metadata) at ~400 GB/s takes 20 ns. The compute and
memory overlap, so the scatter is hidden behind the load latency.

### 9.3 When Sparse Format Wins

The sparse format is beneficial when:
- The model has been pruned to 2:4 sparsity during training or post-training
- Batch size is small (memory-bandwidth bound)
- K is large relative to M and N (more weight loads per output element)

It is NOT beneficial when:
- The model is not actually sparse (random pruning violates 2:4 structure)
- Batch size is large (compute-bound; the reduced FLOPs from sparsity matter
  more than bandwidth, and the scatter overhead becomes visible)
- The hardware has dedicated sparse MMA units (NVIDIA Ampere+ with
  `mma.sp`; Apple Silicon does not have equivalent hardware)


## 10. Comparison with NVIDIA's Native Sparse Tensor Core Format

NVIDIA's sparse tensor cores (Ampere, Hopper) use a compressed format where
metadata is consumed directly by the hardware MMA unit. The metadata format
differs from what we use here:

- NVIDIA packs metadata for an entire 16x16 MMA tile in a specific interleaved
  pattern matching the tensor core's internal data flow
- The metadata is loaded alongside values in a single instruction (`ldmatrix`)
- Hardware performs the scatter internally during the MMA operation

On Metal, we lack dedicated sparse MMA hardware, so we perform the scatter in
software (registers) before feeding to `simdgroup_multiply_accumulate`. This
adds compute overhead but achieves the same bandwidth reduction. The format
described in this document is designed for software scatter efficiency, not
hardware MMA compatibility, and intentionally uses a simpler linear packing
that maps directly to the K-iteration structure of Marlin's tiled GEMM.
