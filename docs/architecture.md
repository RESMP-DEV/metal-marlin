# Architecture

## System Overview

```
+------------------------------------------------------------------+
|                         Python API                                |
|  +----------------+  +------------------+  +-------------------+  |
|  | MarlinLinear   |  | quantized_linear |  | pack_fp4_weights  |  |
|  +----------------+  +------------------+  +-------------------+  |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|                    MLX Custom Kernel Bridge                       |
|            mx.fast.metal_kernel("marlin_gemm_fp4")               |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|                      Metal Kernels                                |
|  +-------------------+  +-------------------+  +---------------+  |
|  | marlin_gemm_fp4   |  | marlin_gemm_fused |  | dequant_*     |  |
|  | (double-buffered) |  | _fp4 / _u4        |  | (standalone)  |  |
|  +-------------------+  +-------------------+  +---------------+  |
|  +-------------------+  +-------------------+                     |
|  | marlin_gemm_fp4   |  | marlin_zero       |                     |
|  | _striped (K-par)  |  | _reduction        |                     |
|  +-------------------+  +-------------------+                     |
+------------------------------------------------------------------+
```

## Kernel Variants

| Kernel | Grid | Pipeline | Use Case |
|--------|------|----------|----------|
| `marlin_gemm_fp4` | 2D (N/64, M/64) | Double-buffered | Standard GEMM, all shapes |
| `marlin_gemm_fp4_striped` | 1D (num_tgs) | Single-buffer per work unit | K-parallel reduction, load balancing |
| `marlin_gemm_fused_fp4` | 2D (N/64, M/64) | Fused dequant-in-registers | Lowest TG memory, highest occupancy |
| `marlin_gemm_fused_u4` | 2D (N/64, M/64) | Fused dequant-in-registers | INT4 with zero points |
| `marlin_zero_reduction` | 1D | N/A | Helper: zero reduction buffers |

## Tiling Strategy

```
Output C [M, N]
+---------------------------------------+
|    TILE_N=64                          |
|  +---------+                          |
|  |Threadgrp| TILE_M=64               |
|  | Output  |                          |
|  +---------+                          |
|                                        |
+---------------------------------------+

Each threadgroup computes a 64x64 output tile.
Grid: (N/64, M/64, 1) threadgroups
128 threads per threadgroup (4 simdgroups of 32)
```

### Simdgroup Layout

```
64x64 threadgroup output tile
+---+---+---+---+---+---+---+---+
|       SG0     |       SG1     |  rows 0-15
| (2x4 of 8x8) | (2x4 of 8x8) |
+---+---+---+---+---+---+---+---+
|       SG2     |       SG3     |  rows 16-31
| (2x4 of 8x8) | (2x4 of 8x8) |
+---+---+---+---+---+---+---+---+

Layout mapping:
  sg_row_offset = (simd_id / 2) * 16   -> 0 or 16
  sg_col_offset = (simd_id % 2) * 32   -> 0 or 32

Each simdgroup owns SG_M_TILES=2 x SG_N_TILES=4 grid of 8x8 tiles.
Computed via simdgroup_multiply_accumulate (8x8 matrix ops).
```

### K-Dimension Reduction

```
K reduced in chunks of TILE_K=32:
  K_TILES = TILE_K / 8 = 4 sub-iterations per chunk

Per sub-iteration (kt):
  Load 8x8 A fragment from threadgroup memory
  Load 8x8 B fragment (from TG memory or fused dequant)
  simdgroup_multiply_accumulate into 8x8 accumulator
```

## Double-Buffered Pipeline (marlin_gemm_fp4)

```
Timeline:

K iteration:  0      1      2      3      ...
              -----------------------------------
Prologue:    [Load buf 0]
              |
Loop:              [Load buf 1]
              [Compute buf 0]
                    |
                         [Load buf 0]
                    [Compute buf 1]
                         |
                              ...

While computing on buffer N, loading into buffer (1-N).
Threadgroup memory: 2 * (64*32*2B + 32*64*2B) = 16384 bytes.
Within 32KB M4 Max threadgroup memory budget.
```

## Fused Dequant Architecture (marlin_gemm_fused_fp4)

The fused kernel eliminates the full B tile from threadgroup memory. Instead of dequantizing B into a shared 4096-byte buffer, each simdgroup dequantizes its own 8x8 sub-tile in registers.

```
Separate approach (marlin_gemm_fp4):
  A_tile[64][32] (TG memory) + B_tile[32][64] (TG memory) = 8192B per buffer
  Requires full threadgroup_barrier between dequant and compute

Fused approach (marlin_gemm_fused_fp4):
  A_tile[64][32] (TG memory) + B_staging[4][8][8] (per-SG TG memory) = 4608B total
  Uses simdgroup_barrier instead of threadgroup_barrier for B

Inner loop per K sub-tile:
  1. Lanes 0-7 each load one column from global B (coalesced)
  2. Lanes 0-7 dequantize in registers (pure ALU)
  3. Write to B_staging[simd_id][8][8]
  4. simdgroup_barrier (32 threads, not 128)
  5. simdgroup_load B fragment from staging
  6. simdgroup_multiply_accumulate
```

Trade-off: each packed uint32 from B is loaded K_TILES times (4x for TILE_K=32) per k_block, but these are L2 hits due to high temporal locality and M4 Max's 4MB L2 cache.

## Stripe-Partitioned GEMM (marlin_gemm_fp4_striped)

Stripe partitioning linearizes the 2D output tile grid into a 1D work schedule, enabling K-parallel reduction.

```
Tile grid (m_tiles x n_tiles) linearized column-major:
  total_work = m_tiles * n_tiles * parallel_factor

Each threadgroup processes:
  work_per_tg = ceil(total_work / num_tgs)

Work unit -> (tile_row, tile_col, k_slice):
  tile_linear = work_idx / parallel
  k_slice = work_idx % parallel
  tile_row = tile_linear / n_tiles
  tile_col = tile_linear % n_tiles

K-parallel reduction (parallel > 1):
  Phase 1: Each slice writes to reduction_buf[k_slice * M * N]
  Phase 2: Last slice (via atomic counter) reduces all partial sums into C

Avoids FP16 atomicAdd (not natively supported on Metal) by serializing
the final reduction to one threadgroup per output tile.
```

## Memory Layout

### Weight Packing (FP4)

```
Original weights [K, N] in FP16:
+------------------------+
| w[0,0] w[0,1] ... N   |
| w[1,0] w[1,1]         |
| ...                    |
| K                      |
+------------------------+

Packed weights [K/8, N] in uint32:
+------------------------+
| [w[0:8, 0]] [w[0:8,1]]|  Each uint32 holds 8 FP4 values along K
| [w[8:16,0]] ...        |
+------------------------+

Bit layout in uint32:
  bits  0-3:  w[k+0]
  bits  4-7:  w[k+1]
  bits  8-11: w[k+2]
  bits 12-15: w[k+3]
  bits 16-19: w[k+4]
  bits 20-23: w[k+5]
  bits 24-27: w[k+6]
  bits 28-31: w[k+7]
```

### Scales Layout

```
scales[K/group_size, N] in FP16
  One scale per group of `group_size` consecutive K elements, per column.
  Accessed as: scales[k_block / group_size * N + col]
```

## Threadgroup Memory Budget (M4 Max)

```
Available: 32KB per threadgroup

marlin_gemm_fp4 (double-buffered):
  A_tiles: 2 * 64 * 32 * 2B = 8192B
  B_tiles: 2 * 32 * 64 * 2B = 8192B
  Total: 16384B (50% of budget)

marlin_gemm_fused_fp4:
  A_tile: 64 * 32 * 2B = 4096B
  B_staging: 4 * 8 * 8 * 2B = 512B
  Total: 4608B (14% of budget)
  -> Higher occupancy possible
```

## FP4 E2M1 Encoding

| Code | Sign | Exp | Mant | Value |
|------|------|-----|------|-------|
| 0000 | 0    | 00  | 0    | 0.0   |
| 0001 | 0    | 00  | 1    | 0.5   |
| 0010 | 0    | 01  | 0    | 1.0   |
| 0011 | 0    | 01  | 1    | 1.5   |
| 0100 | 0    | 10  | 0    | 2.0   |
| 0101 | 0    | 10  | 1    | 3.0   |
| 0110 | 0    | 11  | 0    | 4.0   |
| 0111 | 0    | 11  | 1    | 6.0   |
| 1xxx | 1    | xx  | x    | -val  |

E2M1 with bias=1:
- Subnormal (E=0): val = M * 0.5 (so 0.0 or 0.5)
- Normal (E>0): val = 2^(E-1) * (1 + M*0.5)
- Dynamic range: [0, 6.0], signed [-6.0, 6.0]

### FP4 -> FP16 Bit Conversion

```
FP4:  [S(1) | E(2) | M(1)]   bias=1
FP16: [S(1) | E(5) | M(10)]  bias=15

Normal case (E4 > 0):
  fp16_sign     = S << 15
  fp16_exponent = (E4 + 14) << 10     (bias delta: 15 - 1 = 14)
  fp16_mantissa = M << 9              (1-bit -> 10-bit, left-aligned)

Subnormal (E4=0, M=1): value = 0.5
  fp16_bits = S<<15 | 14<<10 | 0      (0x3800 unsigned, or 0xB800 signed)

Zero (E4=0, M=0):
  fp16_bits = S<<15
```

## INT4 (U4/S4) Dequantization

The INT4 path uses the "magic number" trick from Marlin/GPTQ:

```
Magic bias: 0x6400 = FP16 representation of 1024.0
  Exponent = 0x19 (25 decimal), mantissa = 0
  Value = 2^(25-15) * 1.0 = 1024.0

Trick: OR a 4-bit value N into mantissa bits [3:0]:
  result_fp16 = 2^10 * (1 + N/1024) = 1024 + N

Subtract 1024.0 -> recovers N as FP16 float.

For pairs (processing two uint16 lanes simultaneously):
  n_biased = (packed & 0x000F000F) | 0x64006400
  values = as_type<half2>(n_biased) - as_type<half2>(0x64006400)

Signed S4 (offset binary): stored = actual + 8
  Combined offset = 8 + zero_point
  result = (stored_u4 - combined_offset) * scale
```

## CUDA -> Metal Mapping

| CUDA Concept | Metal Equivalent | Notes |
|------|------|-------|
| `mma.sync.aligned.m16n8k16` | `simdgroup_multiply_accumulate` | 8x8 tiles on Metal |
| `cp.async.cg.shared.global` | Cooperative threadgroup load + barrier | No async copy HW |
| Warp (32 threads) | Simdgroup (32 threads) | Identical size |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | |
| Shared memory (48KB) | Threadgroup memory (32KB on M4 Max) | Smaller budget |
| `lop3.b32` (ternary logic) | `&`, `|`, `^` composed | Compiler fuses |
| `atomicAdd(float)` | Not natively available | Use lock-based reduction |
| Grid-stride loop | Stripe partitioning | 1D dispatch |
| Stream/kernel overlap | CommandBuffer pipelining | Different paradigm |
