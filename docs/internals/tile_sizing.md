# Tile Sizing Analysis for Apple M4 Max GPU

Optimal tile dimensions for the Metal Marlin GEMM kernels, derived from M4 Max hardware constraints and the simdgroup matrix programming model.

---

## 1. Hardware Parameters

| Parameter | M4 Max Value | Constraint Source |
|-----------|:------------:|-------------------|
| GPU cores | 40 | `metal_marlin.profiling.AppleSiliconGPU` |
| Threads per simdgroup | 32 | Fixed (Apple GPU ISA) |
| Max threads per threadgroup | 1024 | Metal API limit |
| Threadgroup memory | 32 KB | Per-threadgroup limit (Metal) |
| simdgroup_matrix tile | 8x8 | Metal Shading Language fixed |
| L1 cache per core | Not publicly specified | Device-dependent |
| L2 cache (shared) | Not publicly specified | Device-dependent |
| Register file | Not publicly specified | Device-dependent |
| Peak FP16 throughput | ~32 TFLOPS | `metal_marlin.profiling.AppleSiliconGPU` |
| Memory bandwidth | 546 GB/s | `metal_marlin.profiling.AppleSiliconGPU` |

---

## 2. Threadgroup Memory Budget

Each threadgroup has 32,768 bytes (32 KB) of fast on-chip memory. The tile configuration must fit within this budget while leaving headroom for the compiler to allocate spill slots.

### 2.1 Double-Buffered Separate Dequant (marlin_gemm_fp4)

```
A_tiles[2][TILE_M][TILE_K]:  2 * M * K * sizeof(half)  bytes
B_tiles[2][TILE_K][TILE_N]:  2 * K * N * sizeof(half)  bytes
                              ─────────────────────────────
Total:                        4 * (M*K + K*N) bytes  (since sizeof(half) = 2, factor of 2 cancels with double-buffer 2x)
                            = 2 * (M*K + K*N) * 2
```

Equivalently: `4 * K * (M + N)` bytes for the double-buffered pair.

For the default TILE_M=64, TILE_N=64, TILE_K=32:

```
A_tiles: 2 * 64 * 32 * 2 =  8,192 bytes
B_tiles: 2 * 32 * 64 * 2 =  8,192 bytes
                             ──────────────
Total:                       16,384 bytes  (50% of 32 KB budget)
```

### 2.2 Fused Dequant (marlin_gemm_fused_fp4)

The fused kernel eliminates the full B_tile, replacing it with per-simdgroup staging:

```
A_tile[TILE_M][TILE_K]:           M * K * 2        bytes
B_staging[SIMDGROUPS][8][8]:      SG * 64 * 2      bytes
                                  ───────────────────────
Total (64x64x32, 4 SG):          64*32*2 + 4*64*2
                                = 4,096 + 512
                                = 4,608 bytes  (14% of budget)
```

This 3.5x reduction in threadgroup memory enables significantly higher occupancy.

### 2.3 Memory Budget Table

| Configuration | Kernel | A bytes | B bytes | Total | Budget % |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 64x64x32 | separate (2-buf) | 8,192 | 8,192 | 16,384 | 50% |
| 64x64x32 | fused | 4,096 | 512 | 4,608 | 14% |
| 128x64x16 | separate (2-buf) | 8,192 | 4,096 | 12,288 | 38% |
| 128x64x16 | fused | 4,096 | 512 | 4,608 | 14% |
| 32x128x32 | separate (2-buf) | 4,096 | 16,384 | 20,480 | 63% |
| 32x128x32 | fused | 2,048 | 512 | 2,560 | 8% |
| 128x128x16 | separate (2-buf) | 8,192 | 8,192 | 16,384 | 50% |
| 128x128x16 | fused | 4,096 | 512 | 4,608 | 14% |
| 64x64x64 | separate (2-buf) | 16,384 | 16,384 | 32,768 | 100% |

The 64x64x64 configuration is the theoretical maximum for double-buffered separate dequant; it leaves zero headroom and will likely fail allocation with compiler spills. TILE_K=64 also means each simdgroup iterates over 8 K sub-tiles per mainloop step, which exceeds the compute needed to hide the load latency.

---

## 3. Simdgroup Constraints

### 3.1 Simdgroup Matrix: Fixed 8x8 Tiles

The Metal `simdgroup_matrix<half, 8, 8>` type is the atomic unit of matrix computation. All tile dimensions must be multiples of 8. Each `simdgroup_multiply_accumulate` call computes:

```
C[8x8] += A[8x8] * B[8x8]
```

consuming 128 FMA operations (8*8*2 per output element = 2 MADs per element for the K=8 reduction). At FP16, this is 128 multiply-add pairs = 256 FLOPs per call.

### 3.2 Work Distribution Within a Threadgroup

With 4 simdgroups (128 threads) and TILE_M=TILE_N=64, each simdgroup computes a
4x4 block of 8x8 tiles (32x32 output). The 4 simdgroups form a 2x2 grid over the
64x64 output tile. This matches the constants in `src/marlin_gemm.metal`:

```
SG_M_TILES = 4    (4 rows of 8x8 = 32 rows per simdgroup)
SG_N_TILES = 4    (4 cols of 8x8 = 32 cols per simdgroup)

sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8)  // 0 or 32
sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8)  // 0 or 32
```

Layout:

```
  ┌───────────────────────────────────────────────┐
  │  SG 0 (rows 0-31, cols 0-31)  │  SG 1 (rows 0-31, cols 32-63) │
  ├───────────────────────────────────────────────┤
  │  SG 2 (rows 32-63, cols 0-31) │  SG 3 (rows 32-63, cols 32-63)│
  └───────────────────────────────────────────────┘
```

### 3.3 Alternative Simdgroup Decompositions

For tiles larger than 64x64, you have two primary options:

1. Increase SG_M_TILES / SG_N_TILES so each simdgroup covers a larger block
   (more accumulators and register pressure), or
2. Keep SG_M_TILES / SG_N_TILES fixed and loop over additional M or N blocks
   (more passes, less register pressure).

Both approaches are valid; choose based on register pressure, occupancy, and
dispatch overhead for your target workload.

### 3.4 Accumulator Register Pressure

Each `simdgroup_matrix<half, 8, 8>` accumulator occupies 128 bytes across the 32 threads in the simdgroup (4 bytes per thread: 2 half values, though Metal may use float accumulators internally, doubling this to 8 bytes per thread).

| Config | SG_M | SG_N | Accumulators | Bytes/thread (FP16) | Bytes/thread (FP32) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 64x64 (2x2 SG grid) | 4 | 4 | 16 | 64 | 128 |
| 64x64 (current: 2x4) | 2 | 4 | 8 | 32 | 64 |
| 128x64 (4x1 SG) | 4 | 8 | 32 | 128 | 256 |
| 32x128 (1x4 SG) | 4 | 4 | 16 | 64 | 128 |

Apple Silicon has large register files, but the exact size is not publicly
specified. Even 32 accumulators (256 bytes/thread with FP32 internal) are
typically manageable, yet register pressure can still limit occupancy in some
kernels. Use the occupancy tooling to validate per-kernel limits.

---

## 4. Occupancy Analysis

Occupancy on Apple Silicon GPUs is the number of concurrent threadgroups executing per GPU core. Higher occupancy helps hide memory latency by switching between threadgroups.

### 4.1 Limiting Factors

Each GPU core has:
- A per-threadgroup memory limit of 32 KB (Metal)
- A fixed number of thread slots (device-dependent)
- A register file shared across active simdgroups

**Threadgroup memory is often the primary limiter.** The formula
`floor(32768 / X)` is a useful upper bound, but actual occupancy can be lower
due to thread and register limits. Use the occupancy tooling for device-specific
results.

| Kernel Variant | TG Memory | Max Concurrent TGs | Threads/core |
|:---:|:---:|:---:|:---:|
| Separate (64x64x32, 2-buf) | 16,384 | 2 | 256 |
| Fused (64x64x32) | 4,608 | 7 | 896 |
| Separate (128x64x16, 2-buf) | 12,288 | 2 | 256 |
| Fused (128x64x16) | 4,608 | 7 | 896 |
| Fused (32x128x32) | 2,560 | 12 | 1,536 |

### 4.2 Occupancy vs Compute Density Trade-off

Higher occupancy (more concurrent TGs) helps hide memory latency but reduces per-TG compute. For compute-bound kernels (which quantized GEMM typically is after the dequant), 2-4 concurrent threadgroups is sufficient:

- **2 TGs/core**: Adequate when each TG has high arithmetic intensity (AI > 2). The load-compute pipeline within a single TG can hide most latency with double-buffering.
- **4+ TGs/core**: Beneficial for memory-bound phases (small M, long K) where the TG spends significant time waiting for device memory loads.
- **8+ TGs/core**: Diminishing returns for GEMM; useful for bandwidth-bound kernels.

### 4.3 GPU-Wide Parallelism

With 40 GPU cores on M4 Max:

| Config | TGs/core | Total active TGs | Output tiles in flight |
|:---:|:---:|:---:|:---:|
| Separate 64x64 | 2 | 80 | 80 |
| Fused 64x64 | 7 | 280 | 280 |
| Fused 32x128 | 12 | 480 | 480 |

For a 4096x4096 GEMM with TILE_M=64, TILE_N=64: there are 64x64 = 4096 output tiles. With 80 active TGs, the kernel completes in ~51 waves. With 280 active TGs, ~15 waves. Fewer waves means less tail-effect waste.

---

## 5. Compute-to-Memory Ratio Analysis

The key metric for choosing tile sizes is **arithmetic intensity (AI)**: FLOPs per byte loaded from device memory. The higher the AI, the more compute-bound (and thus efficient) the kernel.

### 5.1 FP4 Quantized GEMM

For C[M,N] = A[M,K] @ dequant(B_packed[K/8,N]):

Per output tile (TILE_M x TILE_N x TILE_K reduction):
- **Compute**: `2 * TILE_M * TILE_N * TILE_K` FLOPs (multiply-accumulate)
- **A load**: `TILE_M * TILE_K * 2` bytes (FP16)
- **B load**: `TILE_K * TILE_N * 0.5` bytes (FP4 packed, 4 bits per element)
- **Scale load**: `(TILE_K / group_size) * TILE_N * 2` bytes (FP16, amortized)

Ignoring scales (small contribution for group_size >= 32):

```
AI = (2 * M * N * K) / (M*K*2 + K*N*0.5)   [per K-block, repeated K/TILE_K times]

Per tile iteration:
  Compute = 2 * TILE_M * TILE_N * TILE_K
  Memory  = TILE_M * TILE_K * 2 + TILE_K * TILE_N * 0.5
```

| Config (MxNxK) | FLOPs | A bytes | B bytes (FP4) | AI (FLOPs/byte) |
|:---:|:---:|:---:|:---:|:---:|
| 64x64x32 | 262,144 | 4,096 | 1,024 | **51.2** |
| 128x64x16 | 262,144 | 4,096 | 512 | **56.9** |
| 32x128x32 | 262,144 | 2,048 | 2,048 | **64.0** |
| 128x128x16 | 524,288 | 4,096 | 1,024 | **102.4** |
| 64x64x64 | 524,288 | 8,192 | 2,048 | **51.2** |

All configurations have high arithmetic intensity. Using the M4 Max model values
in `metal_marlin.profiling` (about 32 TFLOPS FP16 and 546 GB/s), the machine
balance is roughly 59 FLOPs/byte. With that balance, 64x64x32 is near the
compute/memory boundary, while larger tiles are more comfortably compute-bound.
Recompute this threshold for your exact device using the profiling tools.

### 5.2 Comparison with FP16 (No Quantization)

With FP16 B weights (2 bytes per element instead of 0.5):

| Config | AI (FP16 weights) | AI (FP4 weights) | Speedup from quantization |
|:---:|:---:|:---:|:---:|
| 64x64x32 | 32.0 | 51.2 | 1.6x less memory traffic |
| 128x64x16 | 42.7 | 56.9 | 1.3x |
| 32x128x32 | 32.0 | 64.0 | 2.0x |

FP4 quantization reduces B-matrix memory traffic by 4x, increasing arithmetic
intensity. Whether a configuration is compute-bound or memory-bound still
depends on the device's peak FLOPs/byte balance.

### 5.3 Dequant Compute Overhead

The dequantization adds ALU operations that do not contribute to useful FLOPs but consume cycles:

- **FP4 E2M1 dequant**: ~8 ALU ops per element (shift, mask, select, multiply)
- **INT4 U4B8 dequant**: ~4 ALU ops per element (OR, subtract, multiply)
- **Useful GEMM FLOPs**: 2 per (M,N,K) triple

Per tile iteration (64x64x32):
- Useful compute: 262,144 FLOPs
- Dequant overhead (FP4): 32*64 * 8 = 16,384 ops
- Dequant overhead (INT4): 32*64 * 4 = 8,192 ops
- Overhead ratio: 6.25% (FP4) or 3.1% (INT4)

The dequant overhead is small relative to the MMA compute, validating the fused approach where dequant runs concurrently with A-tile loads.

---

## 6. Recommended Configurations

### 6.1 Primary: TILE_M=64, TILE_N=64, TILE_K=32

```
Use case:        General-purpose, balanced M and N dimensions
TG memory:       16,384 bytes (separate) / 4,608 bytes (fused)
Simdgroups:      4 (128 threads)
SG partition:    Each SG handles 32x32 output (SG_M=4, SG_N=4)
K sub-tiles:     4 (32/8)
MMA ops/iter:    64 (4*4*4)
Occupancy:       2 TGs/core (separate) / 7 TGs/core (fused)
AI:              51.2 FLOPs/byte (FP4)
Best for:        Square matrices, batch size 32-128, inference workloads
```

This is the default configuration in `marlin_gemm.metal`. It balances threadgroup memory usage with compute density. The double-buffered variant already hides load latency effectively due to the high K_TILES count (4 sub-tiles of 8x8 MMA per mainloop step).

### 6.2 Large-M: TILE_M=128, TILE_N=64, TILE_K=16

```
Use case:        Large batch sizes (M >= 128), wide activation matrices
TG memory:       12,288 bytes (separate) / 4,608 bytes (fused)
Simdgroups:      4 (128 threads)
SG partition:    Each SG handles 32x64 (SG_M=4, SG_N=8)
K sub-tiles:     2 (16/8)
MMA ops/iter:    64 (4*8*2)
Occupancy:       2 TGs/core (separate) / 7 TGs/core (fused)
AI:              56.9 FLOPs/byte (FP4)
Best for:        Prefill phase, training forward pass, M > 128
```

The larger M-tile amortizes the B-weight load across more output rows. Fewer K sub-tiles per iteration means lower compute per load cycle, but the higher AI compensates. This configuration excels when M is large enough that tile-edge waste is negligible.

### 6.3 Large-N: TILE_M=32, TILE_N=128, TILE_K=32

```
Use case:        Wide weight matrices (large N), single-token decode
TG memory:       20,480 bytes (separate) / 2,560 bytes (fused)
Simdgroups:      4 (128 threads)
SG partition:    Each SG handles 32x32 (SG_M=4, SG_N=4) — only 1 SG row
K sub-tiles:     4 (32/8)
MMA ops/iter:    64 (4*4*4)
Occupancy:       1 TG/core (separate) / 12 TGs/core (fused)
AI:              64.0 FLOPs/byte (FP4)
Best for:        M=1-16 decode, wide FFN layers (N=11008 for Llama-7B)
```

For autoregressive decode with small M, the N dimension dominates. A wide N-tile reduces the number of output tiles (fewer kernel launches or dispatch groups) and maximizes the A-tile reuse across columns. The fused variant achieves exceptional occupancy (12 TGs/core) with only 2.5 KB of threadgroup memory.

### 6.4 Maximum Compute: TILE_M=128, TILE_N=128, TILE_K=16

```
Use case:        Very large matrices where tile-edge waste matters
TG memory:       16,384 bytes (separate) / 4,608 bytes (fused)
Simdgroups:      4 (128 threads) — each SG handles 64x32
K sub-tiles:     2 (16/8)
MMA ops/iter:    128 (8*8*2)
Occupancy:       2 TGs/core (separate) / 7 TGs/core (fused)
AI:              102.4 FLOPs/byte (FP4)
Best for:        M >= 256, N >= 2048, training workloads
Note:            Requires SG_M=8, SG_N=4 (128 accumulators per SG)
                 Register pressure: 512 bytes/thread (FP32 accumulators)
                 Still within ~25% of register budget
```

This configuration achieves the highest arithmetic intensity but requires each simdgroup to maintain 8*4 = 32 accumulators. The register pressure is significant but manageable on Apple Silicon's large register file.

---

## 7. Memory Layout Diagrams

### 7.1 Threadgroup Memory Layout (64x64x32, Double-Buffered)

```
Offset (bytes)    Contents
────────────────────────────────────────────────
0x0000            ┌─────────────────────────────┐
                  │  A_tiles[0][64][32]          │  4,096 bytes
                  │  (buffer 0, 64 rows x 32 K) │
0x1000            ├─────────────────────────────┤
                  │  A_tiles[1][64][32]          │  4,096 bytes
                  │  (buffer 1, 64 rows x 32 K) │
0x2000            ├─────────────────────────────┤
                  │  B_tiles[0][32][64]          │  4,096 bytes
                  │  (buffer 0, 32 K x 64 cols) │
0x3000            ├─────────────────────────────┤
                  │  B_tiles[1][32][64]          │  4,096 bytes
                  │  (buffer 1, 32 K x 64 cols) │
0x4000            └─────────────────────────────┘
                  Total: 16,384 bytes (16 KB)
                  Remaining: 16 KB for compiler spills
```

### 7.2 A-Tile Memory Access Pattern

```
A_tile[64][32] — row-major, stride = TILE_K = 32

Thread cooperative load: 128 threads load 2048 elements (64*32)
  → 16 elements per thread (sequential in row-major order)

  Thread 0:  elements [0..15]    → row 0, cols [0..15]
  Thread 1:  elements [16..31]   → row 0, cols [16..31]
  Thread 2:  elements [32..47]   → row 1, cols [0..15]
  ...
  Thread 127: elements [2032..2047] → row 63, cols [16..31]

simdgroup_load reads 8 consecutive rows at stride 32:
  A_frag = A_tile[sg_row + mi*8 .. sg_row + mi*8 + 7][kt*8 .. kt*8 + 7]
  → 8x8 sub-tile, stride = 32 (efficient: no bank conflicts with stride > 8)
```

### 7.3 B-Tile Dequantization + Access Pattern

```
Device memory (packed FP4):
  B[K/8][N] — each uint32 holds 8 FP4 values (one column of 8 K-values)

  For tile at (k_block, tg_col):
    packed = B[(k_block/8) * N + tg_col + n_idx]
    → One uint32 load gives 8 consecutive K-values for column n_idx

Threadgroup memory (after dequant):
  B_tile[32][64] — row = K dimension, col = N dimension

  simdgroup_load reads 8 consecutive K-rows at stride 64:
    B_frag = B_tile[kt*8 .. kt*8+7][sg_col + ni*8 .. sg_col + ni*8 + 7]
    → 8x8 sub-tile, stride = 64 (potential bank conflicts at stride 64)
```

### 7.4 Fused Kernel B-Staging Layout

```
B_staging[4][8][8] — per-simdgroup, 128 bytes each

  Each simdgroup has its own 8x8 staging buffer:
    SG 0: offset 0x000, 128 bytes
    SG 1: offset 0x080, 128 bytes
    SG 2: offset 0x100, 128 bytes
    SG 3: offset 0x180, 128 bytes
    Total: 512 bytes

  Access pattern (fused dequant):
    Lanes 0-7 each load one packed uint32 from global B
    → Dequant 8 K-values in registers
    → Write column [row][lane] to B_staging
    → simdgroup_barrier (not threadgroup_barrier!)
    → simdgroup_load reads the 8x8 staging buffer

  No cross-simdgroup sharing: each SG's staging is private.
  The simdgroup_barrier is ~4 cycles vs ~20 for threadgroup_barrier.
```

---

## 8. Pipeline Timing Estimate

### 8.1 Double-Buffered Separate (64x64x32)

```
Timeline per K-block iteration (32 K-elements):

  ┌────────────────────────────────────────────────────────────────┐
  │ Phase 1: Load next tile    │ Phase 2: Compute current tile     │
  │ (128 threads cooperate)    │ (4 simdgroups in parallel)        │
  ├────────────────────────────┼───────────────────────────────────┤
  │ A: 4096B from device       │ K_TILES=4 iterations:             │
  │   128 * 16 * 2B = 4096B    │   Each: 4*4 = 16 MMA ops         │
  │   = 32B/thread             │   Total: 64 MMA ops               │
  │                            │   = 16,384 FLOPs                  │
  │ B: load packed FP4         │                                   │
  │   + dequant to TG mem      │                                   │
  │   512B packed + ALU         │                                   │
  ├────────────────────────────┼───────────────────────────────────┤
  │ Latency: ~100-200 cycles   │ Latency: ~64 * 4 = 256 cycles    │
  │ (device mem + dequant ALU) │ (4 cycles/MMA estimated)          │
  └────────────────────────────┴───────────────────────────────────┘

  Compute > Load → pipeline is compute-bound after warmup.
  Double-buffering ensures load of tile[k+1] overlaps compute of tile[k].
```

### 8.2 Fused Kernel (64x64x32)

```
Timeline per K-block iteration:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ A tile load (128 threads, cooperative, one barrier)                  │
  │ → 4096 bytes, ~50-100 cycles                                        │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Inner loop: 4 K sub-tiles x (SG_M * SG_N) output tiles              │
  │                                                                      │
  │   Per (kt, mi, ni) iteration:                                        │
  │     ┌──────────────┬─────────────────┬──────────────────────┐       │
  │     │ Lanes 0-7:   │ simdgroup_      │ simdgroup_multiply_  │       │
  │     │ Load packed B│ barrier          │ accumulate            │       │
  │     │ + dequant    │ (~4 cycles)     │ (~4 cycles)           │       │
  │     │ (~12 cycles) │                 │                       │       │
  │     └──────────────┴─────────────────┴──────────────────────┘       │
  │                                                                      │
  │   Total inner: 4 * SG_M * SG_N * ~20 cycles                         │
  │              = 4 * 4 * 4 * 20 = 1280 cycles (with 4x4 SG partition) │
  │                                                                      │
  │   Note: lanes 8-31 idle during dequant phase (25% efficiency loss)   │
  │   Trade-off: no full threadgroup barrier for B → net win             │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Configuration Selection Guide

| Workload | M range | N range | Recommended Config | Reason |
|----------|:-------:|:-------:|:------------------:|--------|
| Decode (single token) | 1-8 | 4096+ | 32x128x32 fused | Max N-coverage, high occupancy |
| Small batch decode | 8-32 | 4096+ | 64x64x32 fused | Balanced, good occupancy |
| Prefill (short) | 32-128 | 4096+ | 64x64x32 separate | Good compute density |
| Prefill (long) | 128-512 | 4096+ | 128x64x16 separate | Amortize B across M |
| Large batch / training | 512+ | 4096+ | 128x128x16 fused | Maximum AI, minimal waves |
| Attention (QK^T) | seq_len | head_dim | 64x64x32 | head_dim typically 64-128 |
| Attention (softmax*V) | seq_len | head_dim | 32x128x32 | V is wide, benefit from N-tile |

### 9.1 Dynamic Tile Selection Strategy

For an inference engine serving variable batch sizes, implement a dispatch table:

```metal
// Host-side tile selection (pseudocode)
TileConfig select_tile(uint M, uint N, uint K) {
    if (M <= 16)
        return {32, 128, 32};   // decode-optimized
    else if (M <= 64)
        return {64, 64, 32};    // balanced
    else if (M <= 256)
        return {128, 64, 16};   // large-M
    else
        return {128, 128, 16};  // maximum compute
}
```

This requires compiling multiple kernel specializations (Metal function constants or separate kernel functions).

---

## 10. Comparison with Marlin CUDA (A100)

| Aspect | Marlin CUDA (A100) | Metal Marlin (M4 Max) |
|--------|:------------------:|:--------------------:|
| MMA tile | 16x8x16 (Tensor Core) | 8x8x8 (simdgroup_matrix) |
| Warp/simdgroup | 32 threads | 32 threads |
| Shared/TG memory | 164 KB | 32 KB |
| Pipeline stages | 4 (cp.async) | 2 (double-buffer) |
| Typical tile | 16x256x64 per warp | 64x64x32 per TG |
| Dequant location | Registers (CUDA cores) | Registers (fused) or TG mem (separate) |
| Occupancy model | Blocks/SM | TGs/core |

The A100's 164 KB shared memory allows much larger tiles and deeper pipelines. Apple Silicon compensates with the fused dequant approach (eliminating B-tile from TG memory) and the unified memory architecture (lower global memory latency).

---

## References

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Sections 2.16 (threadgroup memory), 6.9 (simdgroup matrix)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/gpu_programming_guide/) - Occupancy and memory optimization
- [Marlin: Near-Ideal 4-bit GPU Inference](https://arxiv.org/abs/2312.07723) - Original tile sizing analysis for CUDA
- `src/marlin_gemm.metal` - Current implementation reference
