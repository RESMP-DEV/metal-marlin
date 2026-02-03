# Barrier Optimization Quick Reference

## Summary
- **Total barriers analyzed:** ~100+ across all MoE kernels
- **Estimated removable:** 30-40 barriers (30-40%)
- **Performance impact:** 8-10% faster inference

## Quick Wins (P1 - Execute First)

### 1. Remove Register-to-TG Broadcast Barriers
**Files:**
- `src/moe_fused_dispatch_shared.metal`: Lines 524, 643, 784
- `src/moe_fused_shared_expert.metal`: Similar patterns

**Pattern:**
```metal
float val = compute();  // In registers
threadgroup_barrier(...);  // REMOVE THIS
out_acc[i] += val;       // val never left registers
```

**Why safe:** Values stay in thread-private registers, no cross-thread communication needed.

**Estimated gain:** 10-15% in decode path

---

### 2. Fix Simdgroup Barrier Correctness (P0)
**Files:** All files with `simdgroup_barrier`

**Pattern to fix:**
```metal
// WRONG
simdgroup_barrier(...);  // Only syncs 32 threads
float val = B_buf[other_thread_index];  // Race!

// CORRECT
threadgroup_barrier(...);  // Syncs all 128 threads
float val = B_buf[other_thread_index];  // Safe
```

**Why critical:** Prevents race conditions and correctness bugs.

**Estimated gain:** Correctness (no performance penalty)

---

### 3. Optimize Softmax Barriers
**Files:**
- `src/moe_router_sparse.metal`: Lines 147-198, 332-382, 488-533

**Pattern:**
```metal
// After simdgroup reduction
if (simd_lane == 0) {
    max_shared[simd_id] = local_max;
}
threadgroup_barrier(...);  // KEEP - needed for cross-simdgroup

// Reduce across simdgroups
if (tid == 0) {
    global_max = max(max_shared[0], max_shared[1]);
    max_shared[0] = global_max;
}
threadgroup_barrier(...);  // KEEP - needed for broadcast
```

**Optimization:** Keep only barriers that sync cross-simdgroup communication.

**Estimated gain:** 33% fewer barriers in softmax (5-8% performance)

---

## Execution Commands

### Run All Optimization Tasks
```bash
# From repository root
uv run alphaheng tasks add contrib/metal_marlin/tasks/barrier_optimization_tasks.yaml
uv run alphaheng coordinator --local-workers 50
```

### Run Only Quick Wins (P0-P1)
```bash
uv run alphaheng tasks add contrib/metal_marlin/tasks/barrier_optimization_tasks.yaml
# Then filter by priority in CLI or edit task file
```

### Verify After Changes
```bash
cd contrib/metal_marlin
uv run pytest tests/ -v -k "test_moe or test_gemm"
```

---

## Barrier Categories

| Category | Removable? | Impact | Examples |
|----------|-------------|---------|----------|
| Register-to-TG broadcast | YES | HIGH | Line 524 in dispatch |
| Post-initialization | MAYBE | MEDIUM | Line 996 in gemm_trellis_moe |
| Softmax reduction | PARTIAL | HIGH | Router kernels |
| Load-sync | NO | N/A | After cooperative loads |
| Compute-sync | NO | N/A | Between compute phases |
| Simdgroup correctness | FIX | CRITICAL | Wrong barrier type |

---

## Risk Assessment

### Low Risk (Safe to remove)
- Register-to-TG broadcast barriers
  - Reason: No threadgroup memory access
  - Verification: Check no TG writes between compute and barrier

### Medium Risk (Requires verification)
- Post-initialization barriers
  - Reason: May have hidden cross-thread dependencies
  - Verification: Analyze next phase's memory access pattern

### High Risk (Deep analysis required)
- Loop-carry dependency barriers
  - Reason: Complex dataflow across iterations
  - Verification: Requires detailed dependency graph

---

## Expected Results

### Performance
- **Decode path:** 10-12% faster (barrier-heavy)
- **Prefill path:** 5-7% faster (compute-bound)
- **Overall:** 8-10% improvement

### Code Quality
- **Correctness:** Fix race conditions from wrong barrier types
- **Maintainability:** Better documented barrier usage
- **Performance:** Reduced GPU serialization

---

## Verification Checklist

After each task:
- [ ] Unit tests pass: `uv run pytest tests/ -v -k "test_moe"`
- [ ] No data races in Metal validation layer
- [ ] Output matches reference implementation
- [ ] Performance improved or maintained
- [ ] Documented remaining barriers with purpose

---

## Next Steps

1. **Immediate:** Execute P0 task (fix correctness)
2. **Quick:** Execute P1 tasks (quick wins)
3. **Verify:** Run test suite
4. **Iterate:** Fix any regressions
5. **Document:** Update barrier comments

See `contrib/metal_marlin/docs/barrier_optimization_analysis.md` for detailed analysis.
