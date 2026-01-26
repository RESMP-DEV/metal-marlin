# CUDA to Metal Translation Reference: Marlin Kernels

Reference for porting vLLM's Marlin quantized GEMM kernels from CUDA to Apple Metal.
Source: `vllm/csrc/quantization/marlin/` (dequant.h, marlin_mma.h, dense/marlin_cuda_kernel.cu).

---

## 1. Tensor Core MMA (`mma.sync.aligned.m16n8k16`)

### CUDA: PTX Inline Assembly

Marlin uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for the core
matrix-multiply-accumulate. Each warp (32 threads) collectively computes a
16x8 output tile consuming a 16x16 input tile.

```cuda
// From marlin_mma.h - the core MMA instruction
// Each thread holds fragments of the operands:
//   A: 8 half values (4 half2 registers) from a 16x16 tile
//   B: 4 half values (2 half2 registers) from a 16x8 tile
//   C/D: 4 float accumulators from a 16x8 tile

asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3)
);
```

**Tile geometry**: m16n8k16 means the instruction computes C[16x8] += A[16x16] * B[16x8],
where A is row-major and B is column-major. Each thread owns a specific subset
of the matrix elements according to the Tensor Core thread-data mapping.

### Metal: `simdgroup_multiply_accumulate` with 8x8 Tiles

Metal's simdgroup matrix operations work on **8x8 tiles** rather than CUDA's
16x8x16. A single `mma.sync.m16n8k16` maps to multiple 8x8 operations.

```metal
#include <metal_simdgroup_matrix>
using namespace metal;

// Metal's fundamental unit: 8x8 matrix distributed across a simdgroup (32 threads)
simdgroup_matrix<half, 8, 8> A;    // 8x8 tile of A
simdgroup_matrix<half, 8, 8> B;    // 8x8 tile of B
simdgroup_matrix<float, 8, 8> C;   // 8x8 accumulator

// Load from threadgroup memory (equivalent to shared memory fragments)
simdgroup_load(A, tg_A_ptr, A_stride);
simdgroup_load(B, tg_B_ptr, B_stride);

// The MMA operation: C += A * B
simdgroup_multiply_accumulate(C, A, B, C);

// Store result back
simdgroup_store(C, tg_C_ptr, C_stride);
```

### Tile Shape Decomposition: m16n8k16 → 8x8 Ops

To emulate one `mma.sync.m16n8k16`, decompose into four 8x8 operations:

```
CUDA m16n8k16:  C[16x8] += A[16x16] * B[16x8]

Decompose A[16x16] into:
  A_top_left[8x8],  A_top_right[8x8]
  A_bot_left[8x8],  A_bot_right[8x8]

Decompose B[16x8] into:
  B_top[8x8]
  B_bot[8x8]

Decompose C[16x8] into:
  C_top[8x8]
  C_bot[8x8]

Metal equivalent (4 simdgroup_multiply_accumulate calls):
  C_top += A_top_left  * B_top
  C_top += A_top_right * B_bot
  C_bot += A_bot_left  * B_top
  C_bot += A_bot_right * B_bot
```

```metal
// Full m16n8k16 equivalent in Metal
simdgroup_matrix<half, 8, 8> A00, A01, A10, A11;
simdgroup_matrix<half, 8, 8> B0, B1;
simdgroup_matrix<float, 8, 8> C0, C1;

// Load A[16x16] as four 8x8 blocks
simdgroup_load(A00, tg_A, stride, ulong2(0, 0));
simdgroup_load(A01, tg_A, stride, ulong2(8, 0));
simdgroup_load(A10, tg_A, stride, ulong2(0, 8));
simdgroup_load(A11, tg_A, stride, ulong2(8, 8));

// Load B[16x8] as two 8x8 blocks (B is col-major, transpose on load)
simdgroup_load(B0, tg_B, stride, ulong2(0, 0), /*transpose=*/true);
simdgroup_load(B1, tg_B, stride, ulong2(0, 8), /*transpose=*/true);

// Compute C[16x8] = A[16x16] * B[16x8]
simdgroup_multiply_accumulate(C0, A00, B0, C0);  // C_top += A_tl * B_top
simdgroup_multiply_accumulate(C0, A01, B1, C0);  // C_top += A_tr * B_bot
simdgroup_multiply_accumulate(C1, A10, B0, C1);  // C_bot += A_bl * B_top
simdgroup_multiply_accumulate(C1, A11, B1, C1);  // C_bot += A_br * B_bot
```

**Performance note**: Apple Silicon's AMX/simdgroup matrix hardware is optimized
for 8x8. The four-op decomposition is the canonical approach and performs well
because all ops execute within one simdgroup with no cross-group synchronization.

---

## 2. `lop3.b32` (3-Input Logical Operation)

### CUDA: PTX LOP3 Instruction

`lop3.b32` performs a 3-input bitwise logical operation on 32-bit integers.
The operation is specified by an 8-bit lookup table (LUT) that encodes the
truth table for every combination of the 3 input bits.

Marlin uses this for fast INT4 (U4B8) to FP16 dequantization:

```cuda
// From dequant.h - INT4 dequant using lop3
// Template parameter immLut encodes the logical function
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
    int result;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c), "n"(lut)
    );
    return result;
}

// Usage in INT4 dequant:
const int LO  = 0x000f000f;  // mask: low 4 bits of each 16-bit lane
const int EX  = 0x64006400;  // FP16 1024.0 (exponent bias)

// Extract low nibble and OR with exponent bits in one instruction:
// LUT = (0xf0 & 0xcc) | 0xaa = 0xea
//   Meaning: result = (a & b) | c
int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
```

### LUT Interpretation

The 8-bit LUT encodes the output for all 8 combinations of 3 input bits (a, b, c):

| Bit position | a | b | c | LUT bit meaning |
|:---:|:---:|:---:|:---:|:---|
| 0 | 0 | 0 | 0 | f(0,0,0) |
| 1 | 0 | 0 | 1 | f(0,0,1) |
| 2 | 0 | 1 | 0 | f(0,1,0) |
| 3 | 0 | 1 | 1 | f(0,1,1) |
| 4 | 1 | 0 | 0 | f(1,0,0) |
| 5 | 1 | 0 | 1 | f(1,0,1) |
| 6 | 1 | 1 | 0 | f(1,1,0) |
| 7 | 1 | 1 | 1 | f(1,1,1) |

For LUT = 0xEA = 0b11101010:
- bit7=1, bit6=1, bit5=1, bit4=0, bit3=1, bit2=0, bit1=1, bit0=0
- This is equivalent to: `(a & b) | c`

Verification:
```
a=1,b=1,c=1 → (1&1)|1 = 1 → bit7 = 1 ✓
a=1,b=1,c=0 → (1&1)|0 = 1 → bit6 = 1 ✓
a=1,b=0,c=1 → (1&0)|1 = 1 → bit5 = 1 ✓
a=1,b=0,c=0 → (1&0)|0 = 0 → bit4 = 0 ✓
a=0,b=1,c=1 → (0&1)|1 = 1 → bit3 = 1 ✓
a=0,b=1,c=0 → (0&1)|0 = 0 → bit2 = 0 ✓
a=0,b=0,c=1 → (0&0)|1 = 1 → bit1 = 1 ✓
a=0,b=0,c=0 → (0&0)|0 = 0 → bit0 = 0 ✓
```

### Metal: Composed Bitwise Operations

Metal has no single 3-input logical instruction. Decompose based on the LUT:

```metal
// Generic lop3 decomposition for LUT = 0xEA: (a & b) | c
inline int lop3_0xea(int a, int b, int c) {
    return (a & b) | c;
}
```

For the full INT4 dequant context:

```metal
// Metal equivalent of Marlin's INT4 (U4B8) dequant
inline half2 dequant_u4b8(int q, half scale, half zero) {
    const int LO = 0x000f000f;  // low nibble mask per 16-bit lane
    const int HI = 0x00f000f0;  // high nibble mask per 16-bit lane
    const int EX = 0x64006400;  // FP16 1024.0 in both lanes

    // lop3<0xEA>(q, LO, EX) = (q & LO) | EX
    // Extracts low 4 bits and places them as mantissa of 1024+n
    int lo = (q & LO) | EX;

    // lop3<0xEA>(q, HI, EX) = (q & HI) | EX
    // Extracts high 4 bits (still shifted) with exponent
    int hi = (q & HI) | EX;

    // Reinterpret as half2, subtract bias (1024.0), apply scale/zero
    half2 lo_f = as_type<half2>(lo) - half2(1024.0h);
    half2 hi_f = as_type<half2>(hi) - half2(1024.0h);

    // Note: high nibble needs additional shift correction
    hi_f = hi_f * half2(0.0625h);  // divide by 16 (shift right 4 bits in float domain)

    // Apply scale and zero-point
    lo_f = lo_f * half2(scale) + half2(zero);
    hi_f = hi_f * half2(scale) + half2(zero);

    return lo_f;  // (return both lo and hi in practice)
}
```

**Performance**: On Apple Silicon, `(a & b) | c` compiles to 2 ALU instructions
(AND + OR). CUDA's `lop3.b32` does it in 1 cycle. The 2-instruction decomposition
is still vastly faster than a memory-bound LUT lookup.

### Other LOP3 LUTs Used in Marlin

| LUT | Expression | Usage |
|-----|-----------|-------|
| 0xEA | `(a & b) \| c` | Primary INT4 dequant (extract + bias) |
| 0xFE | `a \| b \| c` | Combine extracted fields |
| 0x54 | `a & ~b & ~c` | Isolate specific bit ranges |

General Metal decomposition for any LUT:

```metal
// Brute-force any LUT via bit selection (fallback, rarely needed)
inline int lop3_generic(int a, int b, int c, constant int lut) {
    int result = 0;
    // For each bit position independently:
    // Select the LUT output based on the (a,b,c) bit pattern
    // This expands to ~8 ops worst case; specific LUTs reduce to 1-3 ops
    result = select(0, (lut >> 0) & 1, (~a & ~b & ~c));  // impractical for 32b
    // ... In practice, always hand-derive the minimal expression
    return result;
}

// In practice: identify the Boolean expression for your LUT and write it directly.
// All LUTs used in Marlin reduce to 1-3 bitwise ops.
```

---

## 3. `prmt.b32` (Byte Permutation)

### CUDA: PTX Byte Permute Instruction

`prmt.b32` selects and rearranges bytes from two 32-bit source registers based
on a control mask. Marlin uses this for INT8 to FP16 conversion.

```cuda
// From dequant.h - byte permutation for INT8 dequant
// prmt selects 4 bytes from {b, a} (concatenated as 64 bits)
// Each 4-bit nibble in the selector picks a byte index (0-7)
template <int start_byte, int mask>
__device__ inline int prmt(int a) {
    int result;
    asm volatile(
        "prmt.b32 %0, %1, %2, %3;\n"
        : "=r"(result)
        : "r"(a), "r"(0), "n"((start_byte << 0) | (mask << 4))
    );
    return result;
}

// Example usage in INT8→FP16 dequant:
// Extract bytes 0,1 from a 32-bit packed int8x4 value
int pair_01 = prmt<0, 0x5140>(packed);  // bytes at positions 0,1 → half2
int pair_23 = prmt<0, 0x7362>(packed);  // bytes at positions 2,3 → half2
```

### Byte Selection Semantics

The control word has 4 nibbles, each selecting a byte from the concatenated
64-bit value `{b, a}`:
- Nibble values 0-3: select byte 0-3 from `a`
- Nibble values 4-7: select byte 0-3 from `b`
- Bit 3 of each nibble: if set with certain modes, applies sign extension

### Metal: `extract_bits` and Manual Byte Manipulation

Metal has no byte permutation instruction. Use `extract_bits` (available in
Metal 2.0+) and shift/mask operations:

```metal
// Metal equivalent of prmt for INT8 → FP16 conversion
// Extract individual bytes from a packed int32
inline half2 extract_int8_pair(int packed, int byte_idx_lo, int byte_idx_hi) {
    // extract_bits(value, offset_in_bits, num_bits)
    int8_t lo = int8_t(extract_bits(packed, byte_idx_lo * 8, 8));
    int8_t hi = int8_t(extract_bits(packed, byte_idx_hi * 8, 8));

    return half2(half(lo), half(hi));
}

// Full INT8x4 → 2x half2 dequant (replaces two prmt calls)
inline void dequant_int8x4(int packed, thread half2 &out0, thread half2 &out1) {
    // Equivalent to:
    //   prmt<0, 0x5140>(packed) → bytes 0,1
    //   prmt<0, 0x7362>(packed) → bytes 2,3
    int8_t b0 = int8_t(packed & 0xFF);
    int8_t b1 = int8_t((packed >> 8) & 0xFF);
    int8_t b2 = int8_t((packed >> 16) & 0xFF);
    int8_t b3 = int8_t((packed >> 24) & 0xFF);

    out0 = half2(half(b0), half(b1));
    out1 = half2(half(b2), half(b3));
}
```

**Alternative using `as_type` reinterpretation** (more efficient for packed formats):

```metal
// When the byte layout matches the target format, reinterpret directly
inline half2 prmt_bytes_01(int packed) {
    // Extract low 16 bits, reinterpret as two int8 values
    short2 bytes = short2(packed & 0xFF, (packed >> 8) & 0xFF);
    return half2(half(bytes.x), half(bytes.y));
}

inline half2 prmt_bytes_23(int packed) {
    short2 bytes = short2((packed >> 16) & 0xFF, (packed >> 24) & 0xFF);
    return half2(half(bytes.x), half(bytes.y));
}
```

**Performance**: The shift+mask approach compiles to 2-3 ALU ops per byte pair
on Apple Silicon. CUDA's `prmt` does it in 1 cycle. The overhead is acceptable
because the subsequent FMA operations dominate the pipeline.

---

## 4. Async Copy (`cp.async.cg.shared.global`)

### CUDA: Asynchronous Global → Shared Memory Copy

Marlin uses a multi-stage pipeline to overlap global memory loads with compute.
`cp.async` bypasses the L1 cache and loads directly to shared memory without
consuming register file bandwidth.

```cuda
// From marlin_cuda_kernel.cu - async copy pipeline
// Stage global → shared memory load without blocking
__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;  // 128-bit (4x float) copy
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(smem_to_generic(smem_ptr)),
           "l"(glob_ptr),
           "n"(BYTES)
    );
}

// Commit the async copy group
__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait for N-1 outstanding groups (pipeline depth control)
template <int N>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Marlin's 4-stage pipeline pattern:
// Stage 0: Load tile[i+3] from global → shared (async)
// Stage 1: Load tile[i+2] from global → shared (async)
// Stage 2: Load tile[i+1] from global → shared (async)
// Stage 3: Compute on tile[i] (data guaranteed ready)
for (int k_iter = 0; k_iter < K / TILE_K; k_iter++) {
    cp_async4(&sh_A[stage_wr][...], &gl_A[k_iter + 3][...]);
    cp_async4(&sh_B[stage_wr][...], &gl_B[k_iter + 3][...]);
    cp_async_fence();
    cp_async_wait<3>();  // wait until group (current-3) completes
    __syncthreads();

    // Compute MMA on stage_rd (3 stages behind writes)
    mma_sync(..., sh_A[stage_rd], sh_B[stage_rd]);
    stage_wr = (stage_wr + 1) % 4;
    stage_rd = (stage_rd + 1) % 4;
}
```

### Metal: Threadgroup Memory Staging Patterns

Metal does not have a direct equivalent of `cp.async`. Two approaches:

#### Approach A: Manual Double/Multi-Buffering with `threadgroup_barrier`

```metal
// Manual multi-stage pipeline in Metal
// Each threadgroup has N staging buffers in threadgroup memory
threadgroup half tg_A[NUM_STAGES][TILE_M][TILE_K];
threadgroup half tg_B[NUM_STAGES][TILE_K][TILE_N];

int stage_wr = 0;
int stage_rd = 0;

// Prime the pipeline: fill first (NUM_STAGES - 1) buffers
for (int s = 0; s < NUM_STAGES - 1; s++) {
    // Each thread in the threadgroup loads its portion
    uint tid = thread_position_in_threadgroup;
    for (uint i = tid; i < TILE_M * TILE_K / 2; i += threads_per_threadgroup) {
        tg_A[s][i / TILE_K][i % TILE_K] = gl_A[...];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_wr = (stage_wr + 1) % NUM_STAGES;
}

// Main loop: overlap load of next tile with compute on current
for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    // Load next tile (all threads participate)
    for (uint i = tid; i < tile_elements; i += threads_per_threadgroup) {
        tg_A[stage_wr][...] = gl_A[k_iter + NUM_STAGES - 1][...];
        tg_B[stage_wr][...] = gl_B[k_iter + NUM_STAGES - 1][...];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute on ready tile using simdgroup matrix ops
    simdgroup_load(A_tile, &tg_A[stage_rd][0][0], TILE_K);
    simdgroup_load(B_tile, &tg_B[stage_rd][0][0], TILE_N);
    simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);

    stage_wr = (stage_wr + 1) % NUM_STAGES;
    stage_rd = (stage_rd + 1) % NUM_STAGES;
}
```

#### Approach B: `simdgroup_async_copy` (Metal 3.1+, Apple Silicon M3+)

```metal
// Available on M3+ GPUs with Metal 3.1
// Async copy from device memory to threadgroup memory
simdgroup_async_copy<half, TILE_M * TILE_K>(
    &tg_A[stage_wr][0][0],    // destination (threadgroup)
    gl_A_ptr + offset,         // source (device)
    TILE_M * TILE_K            // element count
);

// Barrier to ensure copy completion (replaces cp.async.wait_group)
simdgroup_async_copy_fence();
threadgroup_barrier(mem_flags::mem_threadgroup);
```

**Key differences from CUDA**:
- No cache bypass hint (Metal's memory subsystem handles this internally)
- `simdgroup_async_copy` is per-simdgroup, not per-thread like `cp.async`
- Apple Silicon's unified memory architecture means global→threadgroup copies
  are L1/L2 mediated regardless; the latency hiding benefit comes from the
  multi-buffering pattern itself, not from bypassing caches

---

## 5. Shared Memory Barriers

### CUDA: `__syncthreads()` and `cp.async.wait_group`

```cuda
// Block-level barrier (all threads in threadblock must reach this point)
__syncthreads();

// Async copy completion: wait until at most N groups are outstanding
cp_async_wait<0>();   // wait for ALL outstanding copies
cp_async_wait<1>();   // wait until at most 1 group outstanding

// Combined pattern in Marlin:
cp_async_fence();            // commit current group
cp_async_wait<NUM_STAGES - 2>();  // wait for oldest group
__syncthreads();             // ensure all threads see the data
// ... now safe to read from shared memory stage that was being written
```

### Metal: `threadgroup_barrier`

```metal
// Threadgroup barrier: all threads in the threadgroup must reach this point
// mem_flags specifies which memory operations must be visible

// Full threadgroup barrier (equivalent to __syncthreads)
threadgroup_barrier(mem_flags::mem_threadgroup);

// Barrier for device memory visibility (less common in GEMM kernels)
threadgroup_barrier(mem_flags::mem_device);

// Barrier for both
threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

// Texture memory barrier (not used in Marlin port)
threadgroup_barrier(mem_flags::mem_texture);
```

**No direct equivalent of `cp.async.wait_group`** in Metal. The multi-stage
pipeline must use explicit barriers between load and compute phases:

```metal
// Metal pattern replacing cp.async fence/wait:
// Load phase
for (uint i = tid; i < tile_size; i += tg_size) {
    tg_buf[stage_wr][i] = device_buf[offset + i];
}
// Fence: ensure all writes are visible to all threads
threadgroup_barrier(mem_flags::mem_threadgroup);
// Now safe to read from stage_wr in compute phase

// There is no "wait for N-1 groups" - each barrier is all-or-nothing.
// Multi-stage pipelining still works: the compiler/hardware can overlap
// device memory fetches with ALU work between barriers, but the programmer
// cannot express partial completion like cp.async.wait_group<N>.
```

**Simdgroup-level barrier** (within a single simdgroup, no cross-simdgroup sync):

```metal
// Barrier within a simdgroup only (lighter weight, 32 threads)
simdgroup_barrier(mem_flags::mem_threadgroup);

// Use this when only simdgroup-local data needs synchronization
// (e.g., after simdgroup_store before another simdgroup_load on same memory)
```

---

## 6. Warp-Level Primitives

### CUDA Warp (32 Threads) → Metal Simdgroup (32 Threads on Apple Silicon)

The mapping is direct: CUDA warps and Metal simdgroups both contain 32 threads
that execute in lockstep (SIMT). Apple Silicon GPUs have 32-wide SIMD execution
units.

### `__shfl_sync` → `simd_shuffle`

```cuda
// CUDA: broadcast value from lane `src` to all lanes
int val = __shfl_sync(0xFFFFFFFF, my_val, src_lane);

// CUDA: shift down by `delta` lanes (each thread reads from lane+delta)
int val = __shfl_down_sync(0xFFFFFFFF, my_val, delta);

// CUDA: shift up by `delta` lanes
int val = __shfl_up_sync(0xFFFFFFFF, my_val, delta);

// CUDA: read from arbitrary lane via XOR
int val = __shfl_xor_sync(0xFFFFFFFF, my_val, lane_mask);
```

```metal
// Metal: broadcast from specific lane (equivalent to __shfl_sync)
int val = simd_shuffle(my_val, src_lane);

// Metal: shift down (equivalent to __shfl_down_sync)
int val = simd_shuffle_down(my_val, delta);

// Metal: shift up (equivalent to __shfl_up_sync)
int val = simd_shuffle_up(my_val, delta);

// Metal: XOR shuffle (equivalent to __shfl_xor_sync)
int val = simd_shuffle_xor(my_val, lane_mask);
```

**Key differences**:
- Metal omits the `mask` parameter (no partial-warp execution on Apple Silicon;
  all 32 lanes always participate)
- Metal's `simd_shuffle` family works on all scalar and vector types natively:
  `int`, `float`, `half`, `half2`, etc.
- No `__activemask()` equivalent needed; Apple Silicon does not have independent
  thread scheduling within a simdgroup

### Warp-Level Reductions

```cuda
// CUDA: warp-level reduction (manual tree)
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
// Lane 0 now has the sum
```

```metal
// Metal: built-in simd reductions (preferred)
float sum = simd_sum(val);           // sum across all lanes
float max = simd_max(val);           // max across all lanes
float min = simd_min(val);           // min across all lanes

// Or manual tree reduction (equivalent to CUDA pattern)
for (int offset = 16; offset > 0; offset >>= 1) {
    val += simd_shuffle_down(val, offset);
}
```

### Warp Vote Functions

```cuda
// CUDA: warp vote
bool all = __all_sync(0xFFFFFFFF, predicate);
bool any = __any_sync(0xFFFFFFFF, predicate);
uint ballot = __ballot_sync(0xFFFFFFFF, predicate);
```

```metal
// Metal: simd vote equivalents
bool all = simd_all(predicate);
bool any = simd_any(predicate);
// No direct ballot equivalent; use simd_ballot() on supported targets:
simd_vote vote = simd_ballot(predicate);  // Metal 2.1+
```

### Lane ID

```cuda
// CUDA: thread index within warp
int lane_id = threadIdx.x % 32;
// or
int lane_id = threadIdx.x & 0x1F;
```

```metal
// Metal: thread index within simdgroup
uint lane_id = thread_index_in_simdgroup;
// Available as a kernel parameter or via built-in variable
```

---

## Summary: Performance Characteristics

| Primitive | CUDA Cycles | Metal Equivalent | Metal Cycles (est.) | Notes |
|-----------|:-----------:|-----------------|:-------------------:|-------|
| mma.sync.m16n8k16 | ~8 | 4x simdgroup_multiply_accumulate | ~12-16 | 8x8 tiles; AMX unit |
| lop3.b32 | 1 | 2-3 bitwise ops | 2-3 | No single 3-input op |
| prmt.b32 | 1 | extract_bits + shifts | 2-4 | Manual byte manipulation |
| cp.async (16B) | ~async | threadgroup load + barrier | ~sync | No true async bypass |
| __syncthreads | ~20-40 | threadgroup_barrier | ~similar | Both full-group sync |
| __shfl_sync | 1 | simd_shuffle | 1 | Direct 1:1 mapping |

### Key Architectural Differences

1. **Unified Memory**: Apple Silicon's unified memory means there is no
   distinct "global memory" latency profile. Threadgroup memory still provides
   benefits (broadcast, reduced bank conflicts) but the DRAM→SRAM gap is
   smaller than on discrete GPUs.

2. **No Independent Thread Scheduling**: All 32 threads in a simdgroup execute
   the same instruction. No need for `__syncwarp()` or active masks.

3. **Register Pressure**: Apple Silicon GPUs have a large register file per
   simdgroup. The register-heavy Marlin approach (dequant in registers, not
   shared memory) translates well.

4. **Occupancy Model**: Metal uses `[[max_total_threads_per_threadgroup(N)]]`
   attribute instead of CUDA's `__launch_bounds__`. Apple Silicon's occupancy
   is primarily limited by threadgroup memory and register usage.

---

## Appendix: Thread Indexing Translation

```cuda
// CUDA thread indexing
int tid = threadIdx.x + threadIdx.y * blockDim.x;
int warp_id = tid / 32;
int lane_id = tid % 32;
int block_id = blockIdx.x;
```

```metal
// Metal thread indexing (passed as kernel parameters)
kernel void my_kernel(
    uint tid [[thread_position_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_grid]]
) {
    // tid ≡ threadIdx.x (for 1D threadgroup)
    // simdgroup_id ≡ warp_id
    // lane_id ≡ threadIdx.x % 32
    // tg_id ≡ blockIdx.x
    // thread_id ≡ blockIdx.x * blockDim.x + threadIdx.x
}
```

---

## References

- [vLLM Marlin dequant.h](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/dequant.h)
- [vLLM Marlin marlin_mma.h](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/marlin_mma.h)
- [vLLM Marlin dense kernel](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/dense/marlin_cuda_kernel.cu)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Sections 6.9 (simdgroup), 6.10 (threadgroup barrier)
- [Metal simdgroup_matrix Reference](https://developer.apple.com/documentation/metal/simdgroup_matrix)
- [PTX ISA - lop3.b32](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-lop3)
- [PTX ISA - prmt.b32](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt)
- [Marlin Paper](https://arxiv.org/abs/2312.07723)
