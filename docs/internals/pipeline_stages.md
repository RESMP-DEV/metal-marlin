# Pipeline Stages: Triple-Buffered Latency Hiding

## Background: CUDA Marlin's 4-Stage Pipeline

CUDA Marlin uses `cp.async` to create a 4-stage software pipeline that fully overlaps
global-to-shared memory copies with tensor core compute:

```
Stage 0: cp.async load tile[k+3] -> buffer 0
Stage 1: cp.async load tile[k+2] -> buffer 1 ; compute tile[k-1] from buffer 3
Stage 2: cp.async load tile[k+1] -> buffer 2 ; compute tile[k-2] from buffer 0
Stage 3: cp.async_wait_group<1>   ; compute tile[k-3] from buffer 1
```

The key hardware feature is `cp.async.cg.shared.global`, which issues non-blocking
128-bit copies that bypass the register file entirely. The `cp.async.wait_group<N>`
primitive allows partial pipeline draining (wait until at most N groups outstanding),
enabling fine-grained overlap control.

Metal has no equivalent of `cp.async`. All threadgroup memory loads are register-mediated
and require explicit barriers for visibility. This fundamentally changes the pipeline
design space.

## Current Design: Double-Buffered (2-Stage)

The existing `marlin_gemm_fp4` uses double-buffering:

```
Prologue: Load tile[0] -> buffer 0 ; barrier
Loop:     Load tile[k+1] -> buffer[1-cur] ; Compute on buffer[cur] ; barrier ; swap
```

Memory: 2 x (A_tile + B_tile) = 2 x (64x32x2B + 32x64x2B) = 16,384 bytes (50% of 32KB)

This works because on M4 Max, the FP4 dequant + 4x 8x8 MMA per K-tile provides
sufficient compute density (~128 FMA ops per element loaded) to mask the ~16 cycle
threadgroup memory latency within a single iteration. The load and compute phases
are serialized within each thread but the hardware memory subsystem can overlap
the device memory fetch with prior-iteration ALU work.

The limitation: when K is small (K <= 128, i.e. 4 tiles or fewer), the pipeline has
insufficient depth to hide the initial load latency. The prologue load stalls without
useful compute to overlap against.

## Proposed Design: Triple-Buffered (3-Stage) with Simdgroup Specialization

### Concept

Split the 4 simdgroups into two functional roles:

- **Loader simdgroups (SG 0-1):** Responsible for cooperative tile loads
- **Compute simdgroups (SG 2-3):** Responsible for simdgroup MMA accumulation

With 3 buffers, the loader pair can be filling buffer N+1 while the compute pair
operates on buffer N, and buffer N-1 drains. This provides one full iteration of
look-ahead compared to double-buffering.

```
Timeline (steady state):

Iteration:     k         k+1       k+2       k+3
             +---------+---------+---------+---------+
SG 0-1:     | Load[2] | Load[0] | Load[1] | Load[2] |  (buffer index = k % 3)
             +---------+---------+---------+---------+
SG 2-3:     | Comp[1] | Comp[2] | Comp[0] | Comp[1] |  (buffer index = (k-1) % 3)
             +---------+---------+---------+---------+
Barrier:           ^         ^         ^         ^
```

The barrier between iterations ensures:
1. Loaders have finished writing to buffer[(k) % 3]
2. Compute has finished reading from buffer[(k-1) % 3]

After the barrier, loaders advance to buffer[(k+1) % 3] (which compute just finished
reading), and compute advances to buffer[(k) % 3] (which loaders just finished writing).

### Why 3 Stages, Not 4

On CUDA, 4 stages are justified because `cp.async` has non-trivial in-flight latency
(the copy queue can buffer multiple groups). On Metal, threadgroup loads are synchronous
within the issuing thread. The latency being hidden is device memory fetch latency
(~100-200 cycles for L2 miss on M4 Max), not copy queue depth.

3 stages provide:
- The compute simdgroups always have a fully-loaded buffer ready
- The loader simdgroups have one buffer worth of compute time to complete the next load
- One buffer acts as the "drain" slot (just finished being read by compute)

4 stages would add 8,192 bytes of threadgroup memory (total 32,768 bytes), exactly
exhausting the 32KB budget and preventing any concurrent threadgroup on the same core.

## Memory Budget Analysis

### Triple-Buffered with Current Tile Sizes (64x64x32)

```
Per buffer:
  A_tile: TILE_M x TILE_K x sizeof(half) = 64 x 32 x 2 = 4,096 bytes
  B_tile: TILE_K x TILE_N x sizeof(half) = 32 x 64 x 2 = 4,096 bytes
  Subtotal: 8,192 bytes

Triple-buffered total: 3 x 8,192 = 24,576 bytes (75% of 32KB)
```

This leaves 7,680 bytes free, insufficient for a second concurrent threadgroup
on the same core (which would need its own 24,576 bytes). Occupancy drops to
1 threadgroup per core, meaning we rely entirely on pipeline depth for latency
hiding rather than threadgroup-level multithreading.

Acceptable trade-off: the explicit pipeline overlap within the threadgroup replaces
the latency hiding that inter-threadgroup scheduling would otherwise provide.

### Reduced Tile Size Option (48x48x32)

```
Per buffer:
  A_tile: 48 x 32 x 2 = 3,072 bytes
  B_tile: 32 x 48 x 2 = 3,072 bytes
  Subtotal: 6,144 bytes

Triple-buffered total: 3 x 6,144 = 18,432 bytes (56% of 32KB)
```

Leaves 13,824 bytes free, but 48 is not a multiple of 8 (simdgroup matrix size).
Use 48 = 6 x 8, requiring 6 sub-tiles per dimension instead of 8, which changes
the simdgroup layout.

### Reduced Tile Size Option (32x64x32)

```
Per buffer:
  A_tile: 32 x 32 x 2 = 2,048 bytes
  B_tile: 32 x 64 x 2 = 4,096 bytes
  Subtotal: 6,144 bytes

Triple-buffered total: 3 x 6,144 = 18,432 bytes (56% of 32KB)
```

Halving TILE_M reduces the output tile to 32x64 (2,048 elements vs 4,096).
Each threadgroup produces half the output, requiring 2x more threadgroups for the
same problem. This increases dispatch overhead but may improve occupancy.

### Recommended: 64x64x32, 3 Buffers, Accept Single-TG Occupancy

The 64x64x32 triple-buffered configuration (24,576 bytes) is the recommended starting
point. The large output tile maximizes the compute-to-load ratio (4,096 output elements
computed per 8,192 bytes loaded), and the triple-buffering provides the pipeline depth
to compensate for single-threadgroup-per-core occupancy.

## Barrier Strategy

### Basic Ping-Pong (Double-Buffer, Current Design)

```metal
// Single barrier separates load from compute across ALL simdgroups
threadgroup_barrier(mem_flags::mem_threadgroup);
```

All 4 simdgroups participate in both loading and computing. The barrier
ensures the load phase completes before compute begins, and compute
completes before the next load overwrites the buffer.

### Triple-Buffer with Simdgroup Specialization

The simdgroup-divergent approach requires careful barrier placement to avoid
deadlocks while maintaining correctness.

```metal
threadgroup half A_buf[3][TILE_M][TILE_K];
threadgroup half B_buf[3][TILE_K][TILE_N];

uint buf_load = 0;       // buffer index for current load
uint buf_compute = 2;    // buffer index for current compute (starts at last primed)

// Prologue: SG 0-1 prime buffers 0 and 1
// All simdgroups execute the prologue loads cooperatively for simplicity
load_A_tile(A, A_buf[0], ...);
load_B_tile_dequant(B, scales, B_buf[0], ...);
threadgroup_barrier(mem_flags::mem_threadgroup);

load_A_tile(A, A_buf[1], ...);
load_B_tile_dequant(B, scales, B_buf[1], ...);
threadgroup_barrier(mem_flags::mem_threadgroup);

buf_compute = 0;  // First compute uses buffer 0
buf_load = 2;     // Next load goes to buffer 2

// Main loop
for (uint kt = 0; kt < num_k_tiles; ++kt) {
    uint next_k = (kt + 2) * TILE_K;  // 2 ahead because 2 are pre-loaded

    if (simd_id < 2) {
        // LOADER: Fill buf_load with the tile 2 iterations ahead
        if (next_k < K) {
            load_partial_A(A, A_buf[buf_load], next_k, simd_id);
            load_partial_B(B, scales, B_buf[buf_load], next_k, simd_id);
        }
    } else {
        // COMPUTE: Process buf_compute
        compute_partial(A_buf[buf_compute], B_buf[buf_compute],
                        acc, simd_id - 2);
    }

    // Full barrier: ensures loaders are done writing buf_load
    // AND compute is done reading buf_compute
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rotate: buf_compute advances to what was just loaded,
    // buf_load advances to what compute just finished reading
    buf_compute = (buf_compute + 1) % 3;
    buf_load = (buf_load + 1) % 3;
}
```

### Critical Barrier Invariant

At each barrier:
- `buf_load` was just written by SG 0-1 (safe to read next iteration)
- `buf_compute` was just read by SG 2-3 (safe to overwrite next iteration)
- The third buffer is neither being written nor read (transition slot)

The modulo-3 rotation ensures loaders never write to a buffer that compute is
currently reading, because at any point:
- `buf_load`, `buf_compute`, and the idle buffer are three distinct indices
- Advancing both by 1 (mod 3) maintains this invariant

### Deadlock Avoidance

The single `threadgroup_barrier` per iteration is sufficient and cannot deadlock
because:
1. All 4 simdgroups reach the barrier (no conditional barriers)
2. Loaders and compute take comparable time (balanced work partitioning; see below)
3. No nested barriers or conditional control flow around the barrier

The if/else on `simd_id` is purely a work assignment divergence, not a control flow
divergence that could prevent barrier reach.

## Work Partitioning for Load/Compute Balance

The pipeline is only effective if the load and compute phases take approximately
the same time. If compute finishes before loading, the pipeline stall moves to
the barrier (waiting for loaders). If loading finishes before compute, the extra
buffer depth is wasted.

### Load Phase (SG 0-1, 64 threads)

Per K-tile, the loaders must fill:
- A_tile: 64 x 32 = 2,048 half elements = 4,096 bytes
- B_tile: 32 x 64 = 2,048 half elements = 4,096 bytes (after FP4 dequant)

Total device memory reads (FP4 packed): A reads 4,096 bytes + B reads 512 bytes
(packed FP4) + scales ~128 bytes. Approximately 4,736 bytes from device memory,
producing 8,192 bytes in threadgroup memory (with dequant expansion).

At M4 Max's ~400 GB/s bandwidth with 64 threads loading:
- Ideal: 4,736 bytes / (400 GB/s / 40 cores) = ~0.47 us per tile
- With overhead (address calc, bounds check, dequant ALU): ~0.8-1.2 us per tile

### Compute Phase (SG 2-3, 64 threads)

Per K-tile, the compute simdgroups execute:
- 4 K sub-tiles (K_TILES = 32/8 = 4)
- Per sub-tile: SG_M_TILES x SG_N_TILES = 2x4 = 8 simdgroup_multiply_accumulate calls
- Per compute simdgroup: 4 x 8 = 32 MMA operations

But with only 2 compute simdgroups (SG 2-3), each must cover half the 64x64 output.
The simdgroup layout changes from 2x2 to 1x2:

```
Compute-only layout (SG 2-3):
+---+---+---+---+---+---+---+---+
|               SG2               |  rows 0-31 (4 x 8x8 M-tiles)
| (4x4 of 8x8 tiles = 32x32)    |
+---+---+---+---+---+---+---+---+
|               SG3               |  rows 32-63
| (4x4 of 8x8 tiles = 32x32)    |
+---+---+---+---+---+---+---+---+
```

Each compute simdgroup handles 4 x 4 = 16 tiles of 8x8, for 4 K sub-tiles:
- 4 x 16 = 64 simdgroup_multiply_accumulate per compute SG per K-tile

At ~4 cycles per 8x8 MMA on M4 Max (estimated from AMX throughput):
- 64 MMA x 4 cycles = 256 cycles
- At 1.4 GHz: 256 / 1.4e9 = ~0.18 us per tile

### Balance Assessment

Load: ~0.8-1.2 us vs Compute: ~0.18 us. The pipeline is memory-bound, meaning
compute simdgroups will idle at the barrier waiting for loaders. This is expected
for a quantized GEMM at inference batch sizes: the FP4 packing reduces memory
traffic by 4x compared to FP16, but the compute intensity per byte is still
limited by the low arithmetic density of the dequant+multiply pattern.

The triple-buffering still helps because:
1. The loaders start the NEXT tile immediately after the barrier, while compute
   works on the CURRENT tile. This creates a 1-tile overlap window.
2. If the hardware memory subsystem can serve two outstanding requests per core
   (one from the current load, one prefetched), the effective load latency drops.
3. For larger batch sizes (M >= 16), the compute phase grows proportionally while
   load phase grows sublinearly (A tile loads scale with M, B tile loads are fixed),
   bringing the pipeline closer to balance.

## Tile Size Trade-offs

| Config | A_tile | B_tile | x3 Total | Compute/Load | Occupancy |
|--------|--------|--------|----------|--------------|-----------|
| 64x64x32 | 4096B | 4096B | 24,576B | High | 1 TG/core |
| 64x64x16 | 2048B | 2048B | 12,288B | Medium | 2 TG/core possible |
| 32x64x32 | 2048B | 4096B | 18,432B | Medium | 1-2 TG/core |
| 32x32x32 | 2048B | 2048B | 12,288B | Low | 2 TG/core possible |

### TILE_K = 16 Variant

Halving TILE_K from 32 to 16 halves the per-buffer memory (6,144 bytes per buffer
pair, 18,432 bytes total for triple-buffering) but doubles the number of K-loop
iterations. Each iteration has half the compute (2 K sub-tiles instead of 4),
making the pipeline more memory-bound per iteration but with more iterations
to amortize the prologue cost.

This is preferable when K is small (K < 256) because more pipeline iterations
mean more opportunities for load/compute overlap. For large K (K >= 4096), the
larger TILE_K = 32 is better because it reduces loop overhead and barrier count.

### TILE_M = 32 for Decode (M=1)

For single-token decode (M=1), TILE_M=64 wastes 63/64 of the A tile loads on
zero-padded rows. Reducing TILE_M to 32 (or 16) reduces wasted memory traffic
and makes the pipeline load phase faster, improving balance. A dedicated decode
kernel variant with TILE_M=16 and TILE_K=32 would use:
- Triple-buffered: 3 x (16x32x2 + 32x64x2) = 3 x 5120 = 15,360 bytes
- Allows 2 concurrent threadgroups per core for better latency hiding via
  occupancy rather than pipeline depth.

## Alternative: Fused Dequant + Triple-Buffer Hybrid

The fused kernel (`marlin_gemm_fused_fp4`) already eliminates B from threadgroup
memory, using only per-simdgroup 8x8 staging buffers. Combining fused dequant
with triple-buffering of only the A tile:

```
Per buffer: A_tile only = 64 x 32 x 2 = 4,096 bytes
Triple-buffered A: 3 x 4,096 = 12,288 bytes
B_staging: 4 x 8 x 8 x 2 = 512 bytes
Total: 12,800 bytes (39% of 32KB)
```

This leaves 19,712 bytes free, potentially enabling 2 concurrent threadgroups
(2 x 12,800 = 25,600 bytes < 32KB). The trade-off is that B loads remain
register-mediated per-simdgroup (hitting L2 4x per K-tile) instead of shared
via threadgroup memory, but the occupancy gain from 2 concurrent threadgroups
may compensate.

## Implementation Recommendations

1. **Start with the straightforward triple-buffer** (24,576 bytes, simdgroup-divergent)
   as a correctness baseline. Benchmark against the existing double-buffered kernel.

2. **Profile with Metal System Trace** to determine whether the bottleneck is
   memory latency, ALU throughput, or barrier stalls. If barrier stalls dominate,
   the pipeline is working as intended. If memory latency dominates, the pipeline
   is too shallow (consider the fused hybrid).

3. **For decode (M=1-4)**, implement a TILE_M=16 variant with triple-buffered A
   only (15,360 bytes). The reduced compute per tile makes occupancy-based latency
   hiding (2 TG/core) more effective than pipeline-based hiding.

4. **For prefill (M=64-4096)**, the full 64x64x32 triple-buffer is optimal. The
   high M makes the compute phase longer, naturally balancing the load/compute
   overlap without requiring simdgroup specialization. All 4 simdgroups can
   participate in both phases with a standard prologue+loop pattern.

5. **Measure actual L2 hit rates** for the fused kernel's repeated B loads.
   If L2 hit rate is > 95%, the fused+triple-A hybrid is strictly superior
   to the full triple-buffer approach (less memory, higher occupancy, comparable
   bandwidth). If L2 eviction occurs (large N causing B column spread), the
   full triple-buffer with cooperative B loads into threadgroup memory wins.
