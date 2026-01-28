# Marlin Stripe Partitioning for Metal

This document explains Marlin's stripe partitioning algorithm for distributing
output tiles across GPU compute units, and how to implement it for Metal
dispatch on Apple Silicon.

Reference: https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin


## 1. Why Stripes Instead of 2D Grid Dispatch

### 1.1 The Problem with Naive 2D Dispatch

A standard approach to tiled GEMM is to launch a 2D grid of threadgroups,
one per output tile:

```
dispatch_threadgroups({n_tiles, m_tiles, 1}, threads_per_tg);
```

This has three problems for quantized GEMM workloads:

1. **Load imbalance when tile count is not a multiple of SM/CU count.** If
   you have 7 tiles and 5 compute units, 2 units get 2 tiles while 3 get 1.
   The 3 idle units wait for the 2 busy ones. With stripe partitioning, work
   is distributed evenly regardless of tile count.

2. **No support for K-parallel reduction.** A 2D grid maps one threadgroup
   per output tile. When K is large relative to M*N, each threadgroup does
   excessive sequential work along the K dimension. Stripe partitioning
   naturally extends to split K across multiple threadgroups computing partial
   sums for the same output tile.

3. **Partial tile overhead at matrix edges.** When M or N is not a multiple
   of the tile size, edge tiles do less useful work. Stripe partitioning
   allows interleaving edge tiles with full tiles so no compute unit is stuck
   processing only partial tiles.

### 1.2 The Stripe Pattern

Instead of a 2D grid, Marlin launches a fixed number of threadgroups (matching
the GPU's compute unit count or a chosen occupancy target) and each threadgroup
iterates over a contiguous stripe of the linearized tile space.

For a 3x3 tile grid on 5 threadgroups:

```
Tile grid (row, col):        Threadgroup assignment:
  (0,0) (0,1) (0,2)              0     1     3
  (1,0) (1,1) (1,2)              0     2     3
  (2,0) (2,1) (2,2)              1     2     4
```

Each number is the threadgroup index that processes that tile. The assignment
follows column-major linearization: tiles are numbered 0-8 going down each
column, then each threadgroup gets a contiguous range of ~2 tile indices.

The column-major order ensures that threadgroups processing adjacent tiles
in the M dimension can share loaded A-matrix fragments.


## 2. Stripe Calculation

### 2.1 Core Algorithm

Given the output matrix tiled into `m_tiles x n_tiles` tiles:

```
total_work = m_tiles * n_tiles * k_parallel_factor
work_per_tg = ceil(total_work / num_threadgroups)

// Each threadgroup computes its own work range:
work_start = threadgroup_id * work_per_tg
work_end   = min(work_start + work_per_tg, total_work)

for (uint work_unit = work_start; work_unit < work_end; work_unit++) {
    uint tile_linear = work_unit / k_parallel_factor;
    uint k_slice     = work_unit % k_parallel_factor;

    // Column-major tile indexing (Marlin convention):
    uint tile_row = tile_linear % m_tiles;
    uint tile_col = tile_linear / m_tiles;

    // Compute output tile (tile_row, tile_col) for K-slice k_slice
    process_tile(tile_row, tile_col, k_slice, k_parallel_factor);
}
```

### 2.2 Why Column-Major Linearization

Marlin uses column-major order (tiles enumerated down columns first) rather
than row-major for two reasons:

1. **A-matrix reuse.** Adjacent tiles in the same column share the same rows
   of A. When a threadgroup processes multiple tiles in a column, the A-matrix
   fragments loaded for one tile can be reused for the next, reducing global
   memory traffic.

2. **B-matrix streaming.** The quantized weight matrix B is stored column-major
   in Marlin's packed format. Column-major tile order produces sequential
   access patterns for B, improving cache line utilization.

### 2.3 Worked Example

Matrix: M=48, N=64, tile_size=16, K_parallel=1

```
m_tiles = 48 / 16 = 3
n_tiles = 64 / 16 = 4
total_work = 3 * 4 * 1 = 12 tiles
num_threadgroups = 5
work_per_tg = ceil(12 / 5) = 3
```

Threadgroup assignments (last TG gets fewer):

| TG | work_start | work_end | Tiles (linear) | Grid positions (row,col) |
|----|-----------|----------|----------------|--------------------------|
| 0  | 0         | 3        | 0, 1, 2        | (0,0), (1,0), (2,0)      |
| 1  | 3         | 6        | 3, 4, 5        | (0,1), (1,1), (2,1)      |
| 2  | 6         | 9        | 6, 7, 8        | (0,2), (1,2), (2,2)      |
| 3  | 9         | 12       | 9, 10, 11      | (0,3), (1,3), (2,3)      |
| 4  | 12        | 12       | (none)         | (idle)                   |

With 5 TGs and 12 tiles, TG 4 is idle. But with `work_per_tg = ceil(12/5) = 3`,
the distribution is actually:

| TG | Tiles | Comment |
|----|-------|---------|
| 0  | 0,1,2 | Full column 0 |
| 1  | 3,4,5 | Full column 1 |
| 2  | 6,7,8 | Full column 2 |
| 3  | 9,10,11 | Full column 3 |
| 4  | (none) | Idle |

In practice, Marlin sets `num_threadgroups` to match the number of tiles or
to a value that divides evenly, avoiding idle units. When occupancy constraints
force fewer threadgroups than tiles, each TG processes multiple tiles.


## 3. K-Parallel Reduction

### 3.1 When to Split K

K-parallel reduction is profitable when the K dimension is large relative to
the output tile count. The decision criterion:

```
k_parallel_factor = 1  // default

if (K / tile_k > threshold && m_tiles * n_tiles < num_compute_units) {
    // More compute units than output tiles - some would be idle
    // Use K-splitting to engage all units
    k_parallel_factor = min(
        num_compute_units / (m_tiles * n_tiles),
        K / tile_k  // can't split more than the number of K-tiles
    );
}
```

Typical scenario: batch_size=1 inference with a large weight matrix (e.g.,
M=1, N=4096, K=11008). Here m_tiles * n_tiles might be only ~256, while an
M4 Max has ~40 GPU cores. Without K-splitting, the single M-tile row limits
parallelism.

### 3.2 Partial Sum Computation

Each K-slice computes a partial sum over a fraction of the K dimension:

```metal
uint k_start = k_slice * (K / k_parallel_factor);
uint k_end   = (k_slice + 1) * (K / k_parallel_factor);
if (k_slice == k_parallel_factor - 1) k_end = K;  // last slice gets remainder

// Accumulate only over [k_start, k_end)
for (uint k = k_start; k < k_end; k += tile_k) {
    load_a_tile(tile_row, k);
    load_b_tile(k, tile_col);
    simdgroup_multiply_accumulate(C, A, B, C);
}
```

### 3.3 Reduction Strategy

After all K-slices complete, their partial sums must be combined. Two
approaches:

**Approach A: Atomic reduction (Marlin CUDA default)**

```metal
// Each threadgroup atomically adds its partial sum to the output
device atomic<float>* out_ptr = ...;

for (uint i = 0; i < tile_elements; i++) {
    atomic_fetch_add_explicit(out_ptr + i, partial[i], memory_order_relaxed);
}
```

Pros: Simple, no extra memory, works with any k_parallel_factor.
Cons: Atomic contention when many K-slices target the same tile. FP32 atomics
on Metal require the output buffer to be float (not half).

**Approach B: Workspace reduction**

```metal
// Each K-slice writes to a separate workspace region
device float* workspace = ...;  // [k_parallel_factor][m_tiles][n_tiles][tile_m][tile_n]

uint ws_offset = k_slice * m_tiles * n_tiles * tile_m * tile_n
               + tile_linear * tile_m * tile_n;
store_tile(workspace + ws_offset, partial);

// Separate reduction kernel combines partials
threadgroup_barrier(mem_flags::mem_device);  // NOT sufficient cross-TG

// --- Launch second kernel ---
// reduce_kernel: sum over k_parallel_factor dimension, write to output
```

Pros: No atomics, deterministic, supports FP16 output.
Cons: Extra memory (workspace), extra kernel launch.

**Recommendation for Metal:** Use workspace reduction. Metal's atomic_float
support is limited (no native FP16 atomics, FP32 atomics have lower throughput
than NVIDIA), and the extra kernel launch overhead is minimal for the typical
k_parallel_factor of 2-8.

### 3.4 Synchronization

K-parallel reduction requires a global barrier between the compute phase and
the reduction phase. Metal has no cross-threadgroup barrier within a single
dispatch, so K-parallel reduction requires one of:

1. **Two kernel dispatches** (compute + reduce), synchronized by the command
   buffer. This is the recommended approach.

2. **Spin-lock on a device-memory flag** (fragile, not recommended on Metal
   due to lack of forward progress guarantees across threadgroups).

3. **Atomic counter with threadgroup polling** (possible but wastes cycles and
   introduces non-determinism).


## 4. Metal Dispatch Parameters

### 4.1 Threadgroup Configuration

```metal
// Kernel signature
[[kernel]] void marlin_gemm(
    device const uint32_t* A        [[buffer(0)]],
    device const uint32_t* B_packed [[buffer(1)]],
    device float*          C        [[buffer(2)]],
    constant GemmParams&   params   [[buffer(3)]],
    uint tg_id    [[threadgroup_position_in_grid]],
    uint tg_size  [[threads_per_threadgroup]],
    uint tid      [[thread_index_in_threadgroup]],
    uint simd_id  [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // Stripe calculation
    uint total_work = params.m_tiles * params.n_tiles * params.k_parallel;
    uint work_per_tg = (total_work + params.num_threadgroups - 1) / params.num_threadgroups;
    uint work_start = tg_id * work_per_tg;
    uint work_end = min(work_start + work_per_tg, total_work);

    for (uint work = work_start; work < work_end; work++) {
        uint tile_linear = work / params.k_parallel;
        uint k_slice = work % params.k_parallel;
        uint tile_row = tile_linear % params.m_tiles;
        uint tile_col = tile_linear / params.m_tiles;

        compute_tile(tile_row, tile_col, k_slice, ...);
    }
}
```

### 4.2 Dispatch Call

```metal
// Host-side dispatch (Metal-cpp or Objective-C)
uint num_threadgroups = min(
    m_tiles * n_tiles * k_parallel,  // don't launch more TGs than work items
    gpu_core_count * occupancy_target // target occupancy
);

uint threads_per_tg = simdgroup_size * simdgroups_per_tg;
// Typical: 32 * 4 = 128 threads per threadgroup

encoder->dispatchThreadgroups(
    MTL::Size(num_threadgroups, 1, 1),   // 1D grid of threadgroups
    MTL::Size(threads_per_tg, 1, 1)      // threads per threadgroup
);
```

Note the 1D dispatch: the stripe pattern replaces the 2D grid with a 1D
array of threadgroups, each computing its own work range. This is fundamental
to Marlin's approach and differs from most Metal GEMM implementations that
use 2D grids.

### 4.3 Choosing num_threadgroups

The optimal threadgroup count depends on the GPU and problem size:

| Apple GPU | GPU Cores | Max Concurrent TGs/Core | Recommended TGs |
|-----------|-----------|------------------------|-----------------|
| M1        | 8         | 4-8                    | 32-64           |
| M2 Pro    | 19        | 4-8                    | 76-152          |
| M3 Max    | 40        | 4-8                    | 160-320         |
| M4 Max    | 40        | 4-8                    | 160-320         |

For large matrices (M*N > 1M elements), set `num_threadgroups` to
`gpu_cores * 4`. For small matrices, use `min(total_tiles, gpu_cores * 2)` to
avoid launch overhead exceeding compute time.

Query the GPU core count at runtime:

```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
NSUInteger cores = device.maxThreadgroupMemoryLength > 32768 ? 40 : 10;
// Approximate; Metal doesn't directly expose core count.
// Use MTLGPUFamily checks or hardcode per-chip.
```

### 4.4 Threadgroup Memory Budget

Each threadgroup needs space for:

```
A_tile:  tile_m * tile_k * sizeof(half)    = 16 * 16 * 2 = 512 bytes
B_tile:  tile_k * tile_n * sizeof(half)    = 16 * 16 * 2 = 512 bytes (after dequant)
Staging: 2x for double-buffering           = 2048 bytes total

Total per TG: ~2-4 KB (with double buffering)
```

Apple Silicon provides 32 KB threadgroup memory. At 4 KB per TG, up to 8 TGs
can be resident per core, which matches the occupancy target above.

The fused dequant approach keeps B in registers (dequantized from packed INT4
on the fly), so the actual threadgroup memory usage can be as low as the A
tile staging buffer alone (~1 KB per stage).


## 5. Comparison: Stripe vs 2D Grid

### 5.1 Performance Characteristics

| Aspect | 2D Grid | Stripe |
|--------|---------|--------|
| Launch overhead | One TG per tile | Fixed TG count |
| Load balance | Poor if tiles % CUs != 0 | Even distribution |
| K-parallel | Not supported | Natural extension |
| A-matrix reuse | Depends on schedule | Column-major enables reuse |
| Complexity | Simple dispatch | Slightly more kernel logic |
| Tail effect | Last wave partially filled | Last TG may have fewer tiles |

### 5.2 When Stripes Win

Stripes provide the largest benefit when:

- **M is small** (batch_size 1-8): Few M-tiles means the 2D grid has few rows,
  leading to poor CU utilization. Stripes with K-parallel keep all CUs busy.

- **Matrix dimensions are not tile-aligned**: Stripes distribute partial tiles
  across all CUs rather than concentrating them on the last row/column.

- **K is very large** (e.g., intermediate layers with K=11008): K-parallel
  reduction amortizes the per-tile compute across more CUs.

### 5.3 When Stripes Add No Benefit

- **Large square matrices** where m_tiles * n_tiles >> num_CUs: The 2D grid
  already achieves good load balance. Stripes add index computation overhead
  with no occupancy benefit.

- **k_parallel = 1 and tiles divisible by CU count**: Stripe degenerates to
  a simple linear mapping equivalent to the 2D grid.


## 6. Diagram: Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Host: Compute dispatch parameters                                   │
│   num_tg = min(m_tiles * n_tiles * k_par, gpu_cores * occupancy)    │
│   threads_per_tg = 128 (4 simdgroups * 32 threads)                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ dispatchThreadgroups({num_tg, 1, 1}, ...)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GPU Kernel Entry                                                     │
│                                                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Stripe calculation:                                              │ │
│ │   work_per_tg = ceil(total_work / num_tg)                        │ │
│ │   my_start = tg_id * work_per_tg                                 │ │
│ │   my_end = min(my_start + work_per_tg, total_work)               │ │
│ └────────────────────────────┬────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ For each work_unit in [my_start, my_end):                        │ │
│ │                                                                  │ │
│ │   tile_linear = work_unit / k_parallel                           │ │
│ │   k_slice = work_unit % k_parallel                               │ │
│ │   tile_row = tile_linear % m_tiles                               │ │
│ │   tile_col = tile_linear / m_tiles                               │ │
│ │                                                                  │ │
│ │   ┌───────────────────────────────────────────────────────────┐  │ │
│ │   │ K-loop over [k_start, k_end) for this slice:              │  │ │
│ │   │                                                           │  │ │
│ │   │   1. Load A[tile_row*tile_m : +tile_m, k : +tile_k]      │  │ │
│ │   │      → threadgroup memory (async if double-buffered)      │  │ │
│ │   │                                                           │  │ │
│ │   │   2. Load B_packed[k : +tile_k, tile_col*tile_n : +tile_n]│  │ │
│ │   │      → registers (packed INT4/FP4)                        │  │ │
│ │   │                                                           │  │ │
│ │   │   3. Dequant B in registers (bitwise, ~2 cycles)          │  │ │
│ │   │      → half2 fragments                                    │  │ │
│ │   │                                                           │  │ │
│ │   │   4. simdgroup_multiply_accumulate(C, A_frag, B_frag, C)  │  │ │
│ │   └───────────────────────────────────────────────────────────┘  │ │
│ │                                                                  │ │
│ │   Store partial sum to workspace[k_slice][tile_linear]           │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼ (if k_parallel > 1: second dispatch)
┌─────────────────────────────────────────────────────────────────────┐
│ Reduce Kernel:                                                       │
│   output[tile] = sum(workspace[0..k_par-1][tile])                    │
└─────────────────────────────────────────────────────────────────────┘
```


## 7. Implementation Notes for Metal

### 7.1 Avoiding Divergence

All threads in a simdgroup must execute the same tile. The work loop assigns
entire tiles to threadgroups, so divergence only occurs at the work_end
boundary. Ensure the loop condition is uniform across the threadgroup:

```metal
// Good: uniform loop bound (all threads in TG see same work_end)
for (uint work = work_start; work < work_end; work++) { ... }

// Bad: per-thread conditions inside the tile loop
```

### 7.2 Register Pressure with Multiple Tiles

When a threadgroup processes multiple tiles sequentially, the accumulator
registers can be reused between tiles (store result, zero accumulators, process
next tile). This keeps register pressure constant regardless of how many tiles
are assigned to a threadgroup.

### 7.3 Tile Size Selection for Apple Silicon

Apple Silicon's simdgroup_matrix supports 8x8 tiles natively. For Marlin-style
16x16 logical tiles, compose from 2x2 blocks of 8x8:

```
Logical 16x16 tile:
  ┌────────┬────────┐
  │ 8x8    │ 8x8    │
  │ (0,0)  │ (0,1)  │
  ├────────┼────────┤
  │ 8x8    │ 8x8    │
  │ (1,0)  │ (1,1)  │
  └────────┴────────┘

4 simdgroup_multiply_accumulate calls per 16x16 MMA.
```

This matches Marlin's 16x16 tile granularity while using Metal's native 8x8
operations.

### 7.4 Double-Buffering the A Tile

To hide global memory latency, use double-buffered threadgroup memory for A:

```metal
threadgroup half A_buf[2][tile_m][tile_k];
uint buf_idx = 0;

// Prefetch first tile
async_load_a(A_buf[buf_idx], tile_row, k_start);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint k = k_start; k < k_end; k += tile_k) {
    // Start loading next A tile into other buffer
    if (k + tile_k < k_end) {
        async_load_a(A_buf[1 - buf_idx], tile_row, k + tile_k);
    }

    // Compute with current buffer
    load_a_fragments(A_buf[buf_idx]);
    load_and_dequant_b(k, tile_col);
    simdgroup_multiply_accumulate(C, A, B, C);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf_idx = 1 - buf_idx;
}
```
