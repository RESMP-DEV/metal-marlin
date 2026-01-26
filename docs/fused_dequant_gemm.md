# Fused Dequant-GEMM Architecture

This document describes the fused dequant-GEMM kernel architecture that eliminates
the B_tile threadgroup memory buffer. The key insight, borrowed from Marlin's CUDA
kernels, is to dequantize packed weights in registers rather than in threadgroup
memory, removing both the memory allocation and the full-threadgroup barrier
between dequant and compute.


## 1. Traditional vs. Fused Data Flow

### 1.1 Traditional Approach (marlin_gemm_fp4)

The traditional kernel dequantizes the entire B tile cooperatively into threadgroup
memory before any simdgroup can begin computing.

```
                    ┌──────────────────────────────────────────────────┐
                    │              Threadgroup Memory                  │
                    │                                                  │
                    │   A_tile[64][32]         B_tile[32][64]          │
                    │   (4096 bytes)           (4096 bytes)            │
                    │                                                  │
                    └──────────────────────────────────────────────────┘
                              ▲                        ▲
                              │                        │
                    ┌─────────┴─────────┐   ┌─────────┴──────────────┐
                    │  Cooperative Load  │   │  Cooperative Dequant  │
                    │  (128 threads)     │   │  (128 threads)        │
                    └─────────┬─────────┘   │                        │
                              │             │  B_packed[K/8, N]      │
                              │             │  → unpack FP4 nibble   │
   Global Memory:             │             │  → dequant with scale  │
   A[M, K]  ──────────────────┘             │  → store to B_tile     │
   B[K/8, N] ──────────────────────────────>│                        │
   scales[K/gs, N] ────────────────────────>└────────────────────────┘

                    ╔══════════════════════════════════════════════════╗
                    ║  threadgroup_barrier(mem_flags::mem_threadgroup) ║
                    ╚══════════════════════════════════════════════════╝

                    ┌──────────────────────────────────────────────────┐
                    │              Simdgroup Compute                   │
                    │                                                  │
                    │   simdgroup_load(a_frag, &A_tile[...])           │
                    │   simdgroup_load(b_frag, &B_tile[...])           │
                    │   simdgroup_multiply_accumulate(...)             │
                    │                                                  │
                    └──────────────────────────────────────────────────┘
```

Pipeline summary:
```
Global B_packed ──→ Threadgroup B_tile (dequantized) ──→ Registers ──→ MMA
```

### 1.2 Fused Approach (marlin_gemm_fused_fp4)

The fused kernel has each simdgroup dequantize only the 8x8 B sub-tile it needs,
directly from global memory into a per-simdgroup 128-byte staging buffer.

```
                    ┌────────────────────────────────────────────────────┐
                    │              Threadgroup Memory                    │
                    │                                                    │
                    │   A_tile[64][32]           B_staging[4][8][8]      │
                    │   (4096 bytes)             (512 bytes total)       │
                    │                            128B per simdgroup      │
                    └────────────────────────────────────────────────────┘
                              ▲                        ▲
                              │                        │
                    ┌─────────┴─────────┐   ┌─────────┴──────────────────┐
                    │  Cooperative Load  │   │  Per-Simdgroup Dequant    │
                    │  (128 threads)     │   │  (lanes 0-7 only)         │
                    └─────────┬─────────┘   │                            │
                              │             │  lane i loads B[..., col+i]│
   Global Memory:             │             │  → dequant 8 FP4 values    │
   A[M, K]  ──────────────────┘             │  → write column to staging │
   B[K/8, N] ──────────────────────────────>│                            │
   scales[K/gs, N] ────────────────────────>└────────────────────────────┘

                    ╔═════════════════════════════════════════════════════╗
                    ║  simdgroup_barrier(mem_flags::mem_threadgroup)      ║
                    ║  (32 threads only, NOT 128)                         ║
                    ╚═════════════════════════════════════════════════════╝

                    ┌────────────────────────────────────────────────────┐
                    │              Simdgroup Compute                     │
                    │                                                    │
                    │   simdgroup_load(a_frag, &A_tile[...])             │
                    │   simdgroup_load(b_frag, &B_staging[sg_id][...])   │
                    │   simdgroup_multiply_accumulate(...)               │
                    │                                                    │
                    └────────────────────────────────────────────────────┘
```

Pipeline summary:
```
Global B_packed ──→ Registers (dequant inline) ──→ Staging[8][8] ──→ MMA
```


## 2. Memory Budget Comparison

### 2.1 Traditional Kernel

```
Threadgroup memory per threadgroup:

  Double-buffered:
    A_tiles[2][64][32] * 2B  =  8,192 bytes
    B_tiles[2][32][64] * 2B  =  8,192 bytes
    ──────────────────────────────────────
    Total                     = 16,384 bytes

  Single-buffered:
    A_tile[64][32] * 2B      =  4,096 bytes
    B_tile[32][64] * 2B      =  4,096 bytes
    ──────────────────────────────────────
    Total                     =  8,192 bytes
```

### 2.2 Fused Kernel

```
Threadgroup memory per threadgroup:

    A_tile[64][32] * 2B                =  4,096 bytes
    B_staging[4][8][8] * 2B            =    512 bytes
    ──────────────────────────────────────────────────
    Total                              =  4,608 bytes

    Savings vs single-buffered:  3,584 bytes (43.7% reduction)
    Savings vs double-buffered: 11,776 bytes (71.9% reduction)
```

### 2.3 Occupancy Impact

M4 Max has 32 KB threadgroup memory per compute unit. The maximum concurrent
threadgroups per CU is limited by:

```
Traditional (double-buffered):  floor(32768 / 16384) = 2 threadgroups/CU
Traditional (single-buffered):  floor(32768 / 8192)  = 4 threadgroups/CU
Fused:                           floor(32768 / 4608)  = 7 threadgroups/CU
```

Higher occupancy means the GPU can hide latency by switching between more
threadgroups when one stalls on memory.


## 3. Barrier Elimination

### 3.1 Traditional: Full Threadgroup Barrier

In the traditional kernel, all 128 threads must participate in filling B_tile
before any thread can read from it. This creates a hard synchronization point:

```
Thread timeline (traditional):

  Thread 0:   ████ dequant ████ ┃ barrier ┃ ████ compute ████
  Thread 31:  ██ dequant ██████ ┃ barrier ┃ ████ compute ████
  Thread 32:  ████ dequant ████ ┃ barrier ┃ ████ compute ████
  Thread 63:  ██ dequant ██████ ┃ barrier ┃ ████ compute ████
  Thread 64:  ████ dequant ████ ┃ barrier ┃ ████ compute ████
  Thread 95:  ██ dequant ██████ ┃ barrier ┃ ████ compute ████
  Thread 96:  ████ dequant ████ ┃ barrier ┃ ████ compute ████
  Thread 127: ██ dequant ██████ ┃ barrier ┃ ████ compute ████
                                  ^
                                  128-thread synchronization
```

The slowest thread in any simdgroup stalls the entire threadgroup. On M4 Max,
threadgroup_barrier costs approximately 16 cycles of idle time across all
threads.

### 3.2 Fused: Simdgroup Barrier Only

Each simdgroup independently dequantizes its own B sub-tile. Only the 32 threads
within that simdgroup need to synchronize:

```
Thread timeline (fused):

  Simdgroup 0 (threads 0-31):
    Lanes 0-7:  ██ dequant ██ ┃ sg_bar ┃ ████ compute ████
    Lanes 8-31:  (idle)       ┃ sg_bar ┃ ████ compute ████

  Simdgroup 1 (threads 32-63):            [runs independently]
    Lanes 0-7:  ██ dequant ██ ┃ sg_bar ┃ ████ compute ████
    Lanes 8-31:  (idle)       ┃ sg_bar ┃ ████ compute ████

  Simdgroup 2 (threads 64-95):            [runs independently]
    ...
  Simdgroup 3 (threads 96-127):           [runs independently]
    ...
```

Each simdgroup can proceed to compute as soon as its own 8 active lanes finish
dequantizing. No cross-simdgroup dependency exists for B data.


## 4. Inner Loop Structure

### 4.1 Iteration Order

The fused kernel's inner loop iterates over K sub-tiles (kt), M sub-tiles (mi),
and N sub-tiles (ni). For each (kt, mi, ni) combination, it performs dequant
and compute in a single fused step:

```
for kt in 0..K_TILES (4 iterations, covering 32 K values):
    k_sub_base = k_block + kt * 8
    k_pack_idx = k_sub_base / 8       ← which packed uint32 row
    group_idx  = k_sub_base / group_size  ← which scale group

    for mi in 0..SG_M_TILES (2 iterations):
        simdgroup_load(a_frag, A_tile[sg_row + mi*8][kt*8])

        for ni in 0..SG_N_TILES (4 iterations):
            b_col_base = tg_col + sg_col_offset + ni * 8

            ┌─────────────────────────────────────────────┐
            │  FUSED DEQUANT (lanes 0-7 only):            │
            │                                             │
            │  lane i:                                    │
            │    packed = B[k_pack_idx * N + b_col + i]   │
            │    scale  = scales[group_idx * N + b_col+i] │
            │    dequant_fp4x8(packed, scale, vals[8])    │
            │    B_staging[sg_id][0..7][i] = vals[0..7]   │
            └─────────────────────────────────────────────┘

            simdgroup_barrier(mem_threadgroup)

            simdgroup_load(b_frag, B_staging[sg_id])
            simdgroup_multiply_accumulate(acc, a_frag, b_frag)
```

### 4.2 Why Lanes 0-7?

An 8x8 sub-tile of B has 8 columns. Each of the 8 active lanes loads one column
(8 FP4 values packed into one uint32). This gives coalesced memory access: 8
adjacent lanes read 8 adjacent uint32 words in global memory.

Lanes 8-31 are idle during dequant but participate in the subsequent
simdgroup_load and simdgroup_multiply_accumulate operations. On Apple Silicon,
simdgroup_load distributes the 64-element 8x8 matrix across all 32 lanes
(2 elements per lane), so all lanes are needed for the MMA.

### 4.3 Global Load Pattern

```
B layout: [K/8, N] packed uint32, row-major
          Each row contains N packed uint32 values
          Each uint32 holds 8 FP4 values (one column of 8 K values)

For sub-tile at (kt, ni):
  Row index:    k_pack_idx = (k_block + kt*8) / 8
  Column range: [b_col_base, b_col_base + 8)

  Lane 0: B[k_pack_idx * N + b_col_base + 0]  ← column 0
  Lane 1: B[k_pack_idx * N + b_col_base + 1]  ← column 1
  ...
  Lane 7: B[k_pack_idx * N + b_col_base + 7]  ← column 7
```

All 8 loads are contiguous in memory, so the GPU can coalesce them into a
single 32-byte memory transaction.


## 5. The Trade-off: Redundant Global Loads

### 5.1 Repeated Access Pattern

In the traditional kernel, each packed uint32 from B is loaded exactly once into
B_tile, then read K_TILES times from threadgroup memory by different simdgroups
computing different M sub-tiles.

In the fused kernel, each simdgroup loads from global memory independently. For
a given K sub-tile (kt) and N sub-tile (ni), only one simdgroup needs that 8x8
B sub-tile. But across different M sub-tiles (mi), the same simdgroup reloads
the same B data:

```
Traditional:                         Fused:
  Global → B_tile (1 load)            Global → Registers (SG_M_TILES loads)
  B_tile → 2 simdgroups (2 reads)     = 2 loads of same data
```

Each packed uint32 is loaded from global memory SG_M_TILES = 2 times per
simdgroup (once for each mi iteration). Since simdgroups are assigned different
M regions but share N columns, a column of B data is accessed by 2 simdgroups
(those sharing the same sg_col_offset), for a total of SG_M_TILES * 2 = 4
global loads per packed word per k_block.

### 5.2 L2 Cache Mitigation

The repeated loads are acceptable because:

1. **Temporal locality**: All accesses to a given B word occur within the same
   k_block iteration, spanning only the mi and ni inner loops. On M4 Max with
   its 4 MB L2 cache, the B sub-tile data (8 * 4B = 32 bytes per lane, 256
   bytes per sub-tile) remains in L2 across these iterations.

2. **Working set fits in L2**: For one k_block, the total B data accessed by
   one threadgroup is TILE_K * TILE_N / 8 * 4B = 32 * 64 / 8 * 4 = 1024 bytes.
   This is negligible relative to the 4 MB L2.

3. **L2 hit latency vs. threadgroup memory**: L2 hits on M4 Max cost
   approximately 15-20 cycles. Threadgroup memory loads cost 10-15 cycles but
   require the threadgroup_barrier overhead. The net difference is minimal, and
   the fused kernel wins by eliminating the barrier.

### 5.3 When This Trade-off Fails

The fused approach becomes less favorable when:

- **Large TILE_N**: If TILE_N is much larger than the L2 line size, the working
  set might exceed L2 capacity, causing global memory round-trips.
- **Many concurrent threadgroups**: High occupancy means more L2 contention.
  The B working sets from different threadgroups may evict each other.
- **Very small K**: If K is small, the K-loop runs few iterations and the
  barrier overhead per iteration is amortized over less compute, reducing the
  fused kernel's relative advantage.

For the typical LLM inference workload (K = 4096-16384, N = 4096-16384, M = 1-64),
the L2 cache is more than sufficient and the fused approach dominates.


## 6. Comparison: B_staging vs. B_tile

```
                    B_tile (traditional)         B_staging (fused)
                    ────────────────────         ─────────────────
Size:               [TILE_K][TILE_N]             [4][8][8]
                    = [32][64] = 4096B           = 512B total
                                                 128B per simdgroup

Scope:              Entire threadgroup           Per-simdgroup
Writers:            All 128 threads              Lanes 0-7 of one simdgroup
Readers:            All 128 threads              Same simdgroup (all 32 lanes)
Lifetime:           One k_block iteration        One (kt, ni) sub-tile
Sync:               threadgroup_barrier          simdgroup_barrier
Content:            Full dequantized tile         One 8x8 sub-tile
Reuse:              K_TILES * all simdgroups     Once (consumed immediately)
```


## 7. Putting It Together: Full Pipeline Comparison

### 7.1 Traditional Double-Buffered Pipeline

```
Time ─────────────────────────────────────────────────────────────────────→

K-tile 0:
  ┌───────────────────┐ ┌───────────────────┐
  │ Load A[0] + B[0]  │ │ Load A[1] + B[1]  │   ← cooperative, 128 threads
  │ into buf[0]       │ │ into buf[1]       │
  └───────────────────┘ └───────────────────┘
           │                      │
           ▼                      │
  ╔═══════════════════╗           │
  ║ TG Barrier        ║           │
  ╚═══════════════════╝           │
           │                      │
           ▼                      │
  ┌───────────────────┐           │
  │ Compute on buf[0] │           │
  │ (4 simdgroups)    │           │
  └───────────────────┘           │
           │                      │
           ▼                      ▼
  ╔═══════════════════╗ ╔═══════════════════╗
  ║ TG Barrier        ║ ║ TG Barrier        ║
  ╚═══════════════════╝ ╚═══════════════════╝
           │                      │
           │                      ▼
           │            ┌───────────────────┐
           │            │ Compute on buf[1] │
           │            └───────────────────┘
           ...
```

Overhead per K-tile: 2 threadgroup barriers (load + swap), 8192B TG memory.

### 7.2 Fused Pipeline

```
Time ─────────────────────────────────────────────────────────────────────→

K-tile 0:
  ┌───────────────────────────┐
  │ Load A_tile cooperatively │  ← 128 threads, 4096B
  └───────────────────────────┘
           │
           ▼
  ╔═══════════════════════════╗
  ║ TG Barrier (A_tile ready) ║
  ╚═══════════════════════════╝
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │ Inner loop: kt=0..3, mi=0..1, ni=0..3              │
  │                                                     │
  │   For each (kt, mi, ni):                           │
  │     ┌──────────────────────────────────────────┐   │
  │     │ Lanes 0-7: load B[..] + dequant → staging│   │
  │     └──────────────────────────────────────────┘   │
  │              │                                      │
  │     ╔════════╧════════════╗                         │
  │     ║ simdgroup_barrier   ║  ← 32 threads only     │
  │     ╚════════╤════════════╝                         │
  │              │                                      │
  │     ┌────────┴─────────────────────────────────┐   │
  │     │ simdgroup_load(b_frag) + MMA             │   │
  │     └──────────────────────────────────────────┘   │
  │                                                     │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ╔═══════════════════════════╗
  ║ TG Barrier (next k_block) ║
  ╚═══════════════════════════╝
```

Overhead per K-tile: 1 threadgroup barrier (A_tile ready) + K_TILES * SG_M_TILES
* SG_N_TILES simdgroup barriers. The simdgroup barriers are approximately free
(same warp, no cross-warp synchronization needed on Apple Silicon).


## 8. INT4 Variant (marlin_gemm_fused_u4)

The INT4 fused kernel follows the same architecture but uses the magic-bias
dequantization trick instead of FP4 bit manipulation. The differences:

```
FP4 fused:
  fused_dequant_fp4x8(packed, scale, vals)
  - 8 nibble extractions
  - 8 select() for subnormal handling
  - 8 multiplies (scale)

INT4 fused:
  fused_dequant_u4x8(packed, scale, zero_point, vals)
  - 4 paired half2 constructions (magic bias OR)
  - 4 half2 subtractions (bias removal)
  - 8 subtractions (zero_point)
  - 8 multiplies (scale)
```

The INT4 variant additionally takes a zero_point buffer for asymmetric
quantization. The per-group zero_point is loaded alongside the scale, with the
same coalesced access pattern (adjacent lanes load adjacent columns).


## 9. When to Use Which Kernel

| Kernel | Use Case |
|--------|----------|
| `marlin_gemm_fp4` | Double-buffered, highest throughput for large K |
| `marlin_gemm_fp4_striped` | Load-balanced distribution, K-parallel reduction |
| `marlin_gemm_fused_fp4` | Register-dequant, best for occupancy-limited workloads |
| `marlin_gemm_fused_u4` | Same as above, for INT4 weights with zero points |
| `marlin_gemm_fp4_single_stage` | Baseline for benchmarking (not shown here) |

The fused kernels are expected to outperform the traditional kernels when:
- Occupancy is the limiting factor (many small matrices, batch=1 inference)
- K is large enough that L2 cache absorbs repeated B loads
- The GPU is not already fully memory-bandwidth-saturated

For large batch sizes where the GPU is compute-bound and fully occupied, the
traditional double-buffered kernel may perform equivalently or better due to
its reduced total global memory traffic.
