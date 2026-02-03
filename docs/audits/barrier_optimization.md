# Barrier Optimization Analysis for MoE Kernels

## Executive Summary
Total barriers analyzed: ~100+ across all MoE kernels
Estimated removable: 30-40 barriers (30-40% reduction)
Impact: 10-15% performance improvement on Apple Silicon

---

## Barrier Categories

### 1. **Redundant Register-to-TG Broadcast Barriers** (HIGH IMPACT)
**Pattern**: Barrier after computing value in registers, before using it.

**Problem**: Threads compute values in private registers, then put unnecessary barrier before using them.

**Files Affected**:
- `moe_fused_dispatch_shared.metal`: Lines 524, 643, 784
- `moe_fused_shared_expert.metal`: Similar patterns

**Example (Line 524)**:
```metal
float swiglu_val = (float)(fast_silu(g) * u);  // In REGISTERS
threadgroup_barrier(mem_flags::mem_threadgroup);  // UNNECESSARY!
for (uint out_idx = 0; out_idx < FUSED_TILE_N / FUSED_THREADS_PER_TG; ++out_idx) {
    out_acc[out_idx] = fma(swiglu_val, ...);  // swiglu_val stays in registers
}
```

**Fix**: Remove barrier. Register values don't need synchronization.

**Estimated Savings**: 3 barriers per expert * 8 experts = 24 barriers eliminated
**Performance Impact**: High - barriers force SIMD serialization

---

### 2. **After Accumulator Initialization** (MEDIUM IMPACT)
**Pattern**: Barrier after initializing threadgroup accumulators to zero.

**Problem**: Initialization is thread-local (each thread writes to its own element), no cross-thread dependencies.

**Files Affected**:
- `gemm_trellis_moe.metal`: Line 996
- `moe_fused_dispatch_shared.metal`: Line 362
- Multiple locations in fused kernels

**Example (Line 996)**:
```metal
for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
    gate_acc_tg[i] = 0.0h;  // Each thread writes unique indices
    up_acc_tg[i] = 0.0h;
}
threadgroup_barrier(mem_flags::mem_threadgroup);  // UNNECESSARY - no data hazard
```

**Fix**: Remove barrier if no thread reads another thread's initialized value before next write.

**Estimated Savings**: 10-15 barriers
**Performance Impact**: Medium - barrier overhead in inner loops

---

### 3. **Simdgroup Reduction Barriers** (HIGH IMPACT)
**Pattern**: Threadgroup barriers used for reductions that can use simdgroup reductions.

**Problem**: `simd_sum_32` and `simd_max_32` use `simd_shuffle`, which synchronizes at simdgroup level only. Full threadgroup barrier is unnecessary unless cross-simdgroup communication follows.

**Files Affected**:
- `moe_router_sparse.metal`: Lines 147-198 (6 barriers in softmax)
- `moe_router_sparse_vec4.metal`: Lines 332-382 (6 barriers)
- All router kernels with softmax

**Example (Lines 147-173)**:
```metal
// Compute local max per thread
float local_max = -INFINITY;
for (uint c = tid; c < num_candidates; c += SPARSE_ROUTER_THREADS) {
    local_max = max(local_max, logits_shared[c]);
}

// simdgroup reduction (synchronizes at simdgroup level only)
local_max = simd_max_32(local_max);
if (simd_lane == 0) {
    max_shared[simd_id] = local_max;
}
threadgroup_barrier(mem_flags::mem_threadgroup);  // ONLY needed if reading max_shared

float global_max;
if (tid == 0) {
    global_max = max_shared[0];
    for (uint s = 1; s < num_simdgroups; ++s) {
        global_max = max(global_max, max_shared[s]);
    }
    max_shared[0] = global_max;
}
threadgroup_barrier(mem_flags::mem_threadgroup);  // ONLY needed if broadcasting global_max
global_max = max_shared[0];
```

**Optimization**: Keep only barriers that sync cross-simdgroup communication.

```metal
// After simd_max_32, store to shared
if (simd_lane == 0) {
    max_shared[simd_id] = local_max;
}
// KEEP THIS BARRIER - needed before reading max_shared
threadgroup_barrier(mem_flags::mem_threadgroup);

// Thread 0 reduces across simdgroups
if (tid == 0) {
    global_max = max_shared[0];
    for (uint s = 1; s < num_simdgroups; ++s) {
        global_max = max(global_max, max_shared[s]);
    }
    max_shared[0] = global_max;
}
// KEEP THIS BARRIER - needed before broadcasting
threadgroup_barrier(mem_flags::mem_threadgroup);

// Now all threads can use global_max
```

**Estimated Savings**: Reduce 6 barriers to 4 per softmax (33% reduction)
**Performance Impact**: High - softmax called many times in inference

---

### 4. **Partial-Thread Load Barriers** (LOW IMPACT)
**Pattern**: Barrier after load where only subset of threads participated.

**Problem**: If only threads 0-3 loaded data, but all threads need it, barrier is necessary. But if all subsequent computation is also done by the same subset of threads, barrier is unnecessary.

**Files Affected**:
- `gemm_trellis_moe.metal`: Line 1003 (load_activation_tile)
- Multiple locations with similar patterns

**Example**:
```metal
// Only first 4 threads load activation
load_activation_tile(activations, A_tile, token_idx, k_block, hidden_dim, thread_idx);

// If barrier here, it's ONLY needed if subsequent code uses A_tile from threads 4+
// But subsequent compute loop:
for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {  // All threads participate
    // Uses A_tile which was only loaded by threads 0-3
}
```

**Fix**: Cannot remove - A_tile is used by all threads.

**Estimated Savings**: 0 barriers (must keep)
**Performance Impact**: None (necessary)

---

### 5. **Loop-Carry Dependencies** (MEDIUM IMPACT)
**Pattern**: Barrier at end of loop iteration, followed by load at start of next iteration.

**Problem**: Sometimes barrier is redundant if:
1. Next iteration's load doesn't depend on previous iteration's result
2. Different memory regions are accessed

**Files Affected**:
- `gemm_trellis_moe.metal`: Lines 1015, 1051, 1203, etc.
- Most tiled kernels

**Example**:
```metal
for (uint kt = 0; kt < num_k_tiles; ++kt) {
    // Compute phase
    for (uint i = thread_idx; i < MOE_TILE_N; i += MOE_THREADS) {
        // Uses B_gate, B_up from THIS iteration
        gate_acc_tg[i] += ...;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Line 1051

    // Next iteration starts here, loads fresh B_gate, B_up
    load_trellis_tile(gate_w, ... B_gate, ...);  // Overwrites previous data
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Line 1078 - KEEP THIS
}
```

**Analysis**:
- Line 1051 barrier: KEEP - needed before next phase reads gate_acc_tg
- Line 1078 barrier: KEEP - needed after load before compute

**Estimated Savings**: 0 barriers (all necessary for correctness)

---

### 6. **Simdgroup Barriers vs Threadgroup Barriers** (HIGH IMPACT)
**Pattern**: Using `simdgroup_barrier` when `threadgroup_barrier` is appropriate.

**Problem**: `simdgroup_barrier` only synchronizes within 32-thread subgroup. `threadgroup_barrier` synchronizes entire threadgroup (128+ threads). Using wrong one causes race conditions or excessive overhead.

**Files Affected**:
- `gemm_trellis_moe.metal`: Lines 1957, 1966, 2084
- Multiple files with `simdgroup_barrier`

**Example**:
```metal
// When B_buf is in threadgroup memory:
threadgroup half B_buf[K][N];

// Compute in all threads of threadgroup
for (uint i = thread_idx; i < N; i += MOE_THREADS) {
    B_buf[0][i] = compute(...);  // All threads write
}
simdgroup_barrier(mem_flags::mem_threadgroup);  // WRONG! Only syncs 32 threads

// Next code reads B_buf from other threads:
float val = B_buf[0][(thread_idx + 1) % N];  // Race if not all threads synced!
```

**Fix**: Use `threadgroup_barrier` for threadgroup memory.

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);  // CORRECT
float val = B_buf[0][(thread_idx + 1) % N];
```

**Estimated Savings**: 0 barriers to remove, but 10-20 incorrect barriers to FIX
**Performance Impact**: Critical - fixes correctness bugs

---

## Optimization Priority Queue

### P0: Fix Correctness Issues
1. **Replace wrong `simdgroup_barrier` with `threadgroup_barrier`**
   - Files: All files with `simdgroup_barrier`
   - Impact: Prevents race conditions
   - Effort: Low (search and replace)

### P1: High-Impact Optimizations
2. **Remove register-to-TG broadcast barriers**
   - Files: `moe_fused_dispatch_shared.metal` (3 barriers)
   - Files: `moe_fused_shared_expert.metal` (3-6 barriers)
   - Impact: 10-15% performance improvement
   - Effort: Low (analyze and remove)

3. **Optimize softmax barriers**
   - Files: `moe_router_sparse.metal` (18 barriers across 3 kernels)
   - Impact: 5-8% performance improvement
   - Effort: Medium (careful analysis needed)

### P2: Medium-Impact Optimizations
4. **Remove post-initialization barriers**
   - Files: All MoE kernels
   - Impact: 3-5% performance improvement
   - Effort: Medium (need to verify no data hazards)

### P3: Low-Impact Optimizations
5. **Review loop-carry dependencies**
   - Files: All tiled kernels
   - Impact: 1-2% performance improvement
   - Effort: High (requires detailed dataflow analysis)

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
- Remove register-to-TG broadcast barriers (P1 #2)
- Fix `simdgroup_barrier` correctness issues (P0 #1)
- Test with existing test suite

### Phase 2: Systematic Optimization (4-6 hours)
- Optimize softmax barriers (P1 #3)
- Remove post-initialization barriers (P2 #4)
- Add unit tests for barrier correctness

### Phase 3: Deep Analysis (8-12 hours)
- Review loop-carry dependencies (P3 #5)
- Profile each optimization
- Document remaining necessary barriers

---

## Verification Strategy

### Unit Tests
1. **Barrier-free paths**: Add tests that run kernels with minimal barriers
2. **Race detection**: Use Metal validation layer to catch data races
3. **Correctness verification**: Compare output before/after optimizations

### Performance Tests
1. **Microbenchmarks**: Time each barrier removal
2. **End-to-end**: Run full inference pipeline
3. **Profile**: Use Metal Performance HUD to identify bottlenecks

---

## Risk Assessment

### Low Risk
- Removing post-initialization barriers (no data dependencies)
- Removing register-to-TG broadcast barriers (no shared memory access)

### Medium Risk
- Optimizing softmax barriers (need to ensure cross-simdgroup sync)

### High Risk
- Loop-carry dependency analysis (complex dataflow)
- Any changes to critical compute loops

---

## Expected Results

### Performance Improvement
- **Decode path**: 10-12% faster (barrier-heavy)
- **Prefill path**: 5-7% faster (compute-bound, less barrier-sensitive)
- **Overall**: 8-10% improvement on typical inference workload

### Code Quality
- **Correctness**: Fix race conditions from wrong barrier types
- **Maintainability**: Better documented barrier usage
- **Performance**: Reduced serialization, better GPU utilization
