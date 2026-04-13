# MoE Dispatch Consolidation Audit

**Date:** 2026-04-11  
**Status:** Active consolidation in progress  
**Priority:** HIGH (per STATUS.md)  
**Task:** "Unify MoE dispatch paths — Route all dispatches through AsyncCommandBufferManager"

---

## Executive Summary

The MoE (Mixture-of-Experts) dispatch system spans **two separate codebases** (root-level `moe_dispatch.py` + `moe/` package) and **two specialized implementations** (Trellis, MMFP4). The older "20 files" claim came from a snapshot that still carried a `metal_marlin/moe/_deprecated/` archive. That archive has since been removed from the live tree; Git history is the reference for those experiments.

**What HAS been consolidated:**
- ✅ Historical experimental `_deprecated/` archive removed from the live tree
- ✅ Single active GPU grouping module (`moe/gpu_grouping.py`) 
- ✅ `group_tokens_by_expert_full` delegates to GPU grouping automatically
- ✅ `AsyncCommandBufferManager` exists in `trellis/async_dispatch.py`
- ✅ Layer batching via `LayerBatchContext`

**What STILL needs consolidation:**
- ⚠️ `BatchedDispatcher` (trellis/moe_dispatch.py) and `AsyncCommandBufferManager` (trellis/async_dispatch.py) are **parallel, non-integrated paths**
- ⚠️ Token grouping logic is **duplicated across 5+ locations**
- ⚠️ `dispatch_moe_trellis_swiglu()` has a `cmd_manager` parameter but many call sites pass `None`
- ⚠️ `MixedBPWMoEDispatcher` has its own independent dispatch path

---

## File Inventory

### A. Root-Level MoE Dispatch Modules (~5300 LOC total)

| File | Lines | Primary Role | Status |
|------|-------|--------------|--------|
| `metal_marlin/moe_dispatch.py` | ~1,400 | **Primary dispatch hub** — token grouping, dispatcher classes, aggregation | Active |
| `metal_marlin/moe_dispatch_metal.py` | ~900 | Metal-specific dispatch, `AsyncExpertDispatcher`, sparse paths | Active (delegates) |
| `metal_marlin/async_dispatch.py` | ~375 | Standalone `AsyncCommandBuffer`, `AsyncFuture`, `AsyncCommandQueue` | Active |

### B. `metal_marlin/moe/` Package (~450 LOC active surface)

| File | Lines | Primary Role | Status |
|------|-------|--------------|--------|
| `moe/__init__.py` | 46 | Re-exports dispatcher + grouping symbols | Active (proxy) |
| `moe/gpu_grouping.py` | 404 | **Active GPU grouping** — `GPUGroupingResult`, `group_tokens_by_expert_fast/gpu_optimized/auto` | Active |

**Historical note:** Earlier audit snapshots referenced an in-repo `_deprecated/` archive. That runtime archive has been deleted; removed modules are preserved only in Git history.

### C. Trellis-Specific MoE (~3,500 LOC)

| File | Lines | Primary Role | Status |
|------|-------|--------------|--------|
| `trellis/moe_dispatch.py` | ~2,600 | **Trellis kernel dispatch** — `dispatch_moe_trellis_swiglu`, `BatchedDispatcher`, `CachedWeightBuffers`, `MoEBufferPool` | Active |
| `trellis/moe.py` | ~1,000 | Trellis MoE layer (`TrellisMoELayer`, `ExpertCache`) | Active |
| `trellis/async_dispatch.py` | ~350 | `AsyncCommandBufferManager`, `LayerBatchContext` | Active |
| `trellis/optimizations.py` | ~400 | `MixedBPWMoEDispatcher` (re-exported from `mixed_bpw_dispatch.py`) | Active |
| `trellis/mixed_bpw_dispatch.py` | ~700 | Mixed bit-width expert dispatch | Active |

### D. Layer Implementations (~600 LOC)

| File | Lines | Primary Role | Status |
|------|-------|--------------|--------|
| `layers/mmfp4_moe.py` | ~500 | MMFP4 MoE layer implementation | Active |
| `layers/mmfp4_fused_moe.py` | ~150 | MMFP4 fused MoE | Active |

### E. Standalone Support (~800 LOC)

| File | Lines | Primary Role | Status |
|------|-------|--------------|--------|
| `moe_scatter_gather.py` | ~300 | Token scatter/gather for MoE | Active |
| `moe_router_cpp.py` | ~100 | C++ router bindings | Active |
| `moe_ops.py` | ~100 | Low-level MoE ops | Active |

**Grand Total: ~35 Python files with MoE dispatch code across 5 directories.**

---

## Dispatch Path Architecture

### Path 1: Generic Token Grouping (Root-Level)

```
group_tokens_by_expert_full(expert_ids, num_experts)
    └── group_tokens_by_expert_fast(expert_ids, num_experts)  [moe/gpu_grouping.py]
            └── PyTorch bincount + cumsum + scatter  (GPU-accelerated)
    OR  (falls back to)
    └── group_tokens_by_expert(expert_ids, num_experts)  [moe_dispatch.py]
            └── PyTorch counting sort (CPU fallback)

Result: MoEDispatchInfo { sorted_token_indices, expert_offsets, inverse_indices }
```

**Used by:** `MoEDispatcher`, `FusedMoEDispatcher` (root-level classes)

### Path 2: Metal GPU Grouping (Root-Level)

```
group_tokens_by_expert_full_gpu(expert_ids, expert_probs, num_experts)  [moe_dispatch.py]
    └── group_tokens_by_expert_gpu(expert_ids, expert_probs, num_experts)
            └── Metal kernel moe_gpu_sort
    OR
    └── group_tokens_by_expert_gpu_optimized(expert_ids, ...)  [moe/gpu_grouping.py]
            └── Delegates to group_tokens_by_expert_gpu

Result: MoEDispatchInfo with GPU-native sorting
```

**Used by:** Trellis trellis_moe_dispatch's `_group_tokens_mixed_bpw_primary_gpu()`

### Path 3: Trellis Fused Kernel Dispatch (Specialized)

```
dispatch_moe_trellis_swiglu(...)
    ├── CachedWeightBuffers (static weights, initialized once)
    ├── MoEBufferPool (dynamic activations, reused per call)
    ├── AsyncCommandBufferManager cmd_manager=None
    │       └── dispatch_immediate() path (no batching)
    └── Metal kernel gemm_trellis_moe.metal

OR with cmd_manager:
    └── cmd_manager.begin_batch() / dispatch() / commit_and_wait()
            └── AsyncCommandBufferManager
```

**Used by:** `TrellisMoEMLP` forward pass

### Path 4: Mixed-BPW Expert Dispatch (Optimization)

```
MixedBPWMoEDispatcher.dispatch(...)
    ├── prepare_fairway_grouped_inputs()
    │       └── _group_tokens_mixed_bpw_primary_gpu()
    ├── dispatch_moe_trellis_swiglu_grouped_fairway()
    └── Per-bit-group kernel dispatch

OR via trellis/optimizations.py:
    └── dispatch_mixed_bpw_moe()
```

**Used by:** GLM-4.7-Flash mixed-precision MoE layers

### Path 5: Layer-Batched Dispatch (Cross-Layer)

```
LayerBatchContext(model, batch_size=8)
    └── AsyncCommandBufferManager (shared across layers)
            └── Groups 8 consecutive MoE layers per commit
            └── 4.8x fewer GPU commits (58 → 12)

Requires: All MoE layers to use the SAME cmd_manager instance
```

**Used by:** TrellisModel end-to-end inference

---

## Consolidation Status Analysis

### ✅ Already Consolidated

1. **Deprecated modules removed from the live tree**: The earlier experimental archive is gone, and `moe/__init__.py` now cleanly re-exports only active symbols.

2. **GPU grouping unified**: `group_tokens_by_expert_full()` in `moe_dispatch.py` automatically delegates to `group_tokens_by_expert_fast()` (GPU path) on MPS devices. Callers don't need to choose.

3. **Single active grouping module**: `moe/gpu_grouping.py` contains all active GPU grouping code. It defines `GPUGroupingResult` as the canonical result type, with a `.to_dispatch_info()` converter for compatibility.

4. **Fused router kernels**: `_fused_router_topk()` in `moe_dispatch.py` handles both PyTorch fallback and Metal kernel dispatch. No call-site branching needed.

5. **Load balancing**: `compute_load_balancing_loss()` in `moe_dispatch.py` is the canonical implementation.

### ⚠️ Remaining Fragmentation Issues

#### Issue 1: Two Parallel Batching Systems

**Problem:** `BatchedDispatcher` (trellis/moe_dispatch.py:389) and `AsyncCommandBufferManager` (trellis/async_dispatch.py:35) are separate, non-integrated batching systems.

- `BatchedDispatcher` queues dispatches into a list, encodes them all in one command buffer, commits.
- `AsyncCommandBufferManager` queues `_PendingDispatch` objects, encodes them, commits.
- They have nearly identical interfaces (`queue_moe_dispatch` vs `dispatch()`, `commit_and_wait()` vs `commit_batch()`) but different APIs.

**Impact:** `dispatch_moe_trellis_swiglu()` accepts an optional `cmd_manager` parameter. When `None`, it creates buffers and dispatches immediately (non-batched). This means many call sites may bypass batching.

**Recommendation:** Deprecate `BatchedDispatcher` in favor of `AsyncCommandBufferManager`. Make `cmd_manager` a required parameter (or default to creating one) so all dispatches are batched by default.

#### Issue 2: Token Grouping Duplicated in 5 Locations

**Problem:** The counting-sort-based grouping algorithm appears in:
1. `moe_dispatch.py:group_tokens_by_expert()` — CPU path, returns 3-tuple
2. `moe_dispatch.py:group_tokens_by_expert_gpu()` — Metal kernel path
3. `moe_dispatch.py:group_tokens_by_expert_full_gpu()` — full info, GPU path
4. `moe/gpu_grouping.py:group_tokens_by_expert_fast()` — GPU path, returns `GPUGroupingResult`
5. `trellis/moe_dispatch.py:prepare_fairway_grouped_inputs()` — delegates to `_group_tokens_mixed_bpw_primary_gpu()`

**Impact:** Maintenance burden, potential for inconsistency, confusing API.

**Recommendation:** Collapse into two functions:
- `group_tokens_by_expert(expert_ids, num_experts)` → `MoEDispatchInfo` (CPU fallback)
- `group_tokens_by_expert_gpu(expert_ids, expert_probs, num_experts)` → `MoEDispatchInfo` (GPU/Metal path)

Remove `group_tokens_by_expert_full()`, `group_tokens_by_expert_full_gpu()`, `group_tokens_by_expert_fast()`, `group_tokens_by_expert_gpu_optimized()`, `group_tokens_by_expert_auto()` as separate entry points. Replace with auto-detection inside the two canonical functions.

#### Issue 3: `dispatch_moe_trellis_swiglu` Bypass Risk

**Problem:** `dispatch_moe_trellis_swiglu()` has `cmd_manager: Any | None = None`. Many call sites may pass `None`, falling back to immediate dispatch:

```python
# This bypasses batching:
output = dispatch_moe_trellis_swiglu(..., cmd_manager=None)

# This uses batching:
cmd_manager.begin_batch()
for layer in layers:
    dispatch_moe_trellis_swiglu(..., cmd_manager=cmd_manager)
cmd_manager.commit_and_wait()
```

**Recommendation:** Change `cmd_manager=None` to `cmd_manager=None` with a runtime warning when `None`, or create a default `AsyncCommandBufferManager` per-model.

#### Issue 4: `MixedBPWMoEDispatcher` Is Independent

**Problem:** `MixedBPWMoEDispatcher` in both `trellis/optimizations.py` and `trellis/mixed_bpw_dispatch.py` has its own dispatch path with `prepare_fairway_grouped_inputs()` and `dispatch_moe_trellis_swiglu_grouped_fairway()`. These are separate from the main `dispatch_moe_trellis_swiglu()` path.

**Recommendation:** Integrate mixed-BPW into the main dispatch path, gated by `bits` being a tuple vs scalar.

#### Issue 5: Historical `_deprecated/` archive (resolved)

**Status:** The `_deprecated/` runtime archive has been removed from the live tree.

**Current policy:** Removed experiments live in Git history, while the runtime package exports only the active MoE surface.

---

## Key Data Structures

### MoEDispatchInfo (Canonical)

```python
@dataclass
class MoEDispatchInfo:
    sorted_token_indices: torch.Tensor   # [total_assignments] gather indices
    sorted_expert_indices: torch.Tensor # [total_assignments] slot indices
    expert_offsets: torch.Tensor         # [num_experts + 1] ranges
    inverse_indices: torch.Tensor       # [total_assignments] scatter indices
    num_tokens: int
    top_k: int
    num_experts: int
```

### GPUGroupingResult (Legacy Wrapper)

```python
@dataclass  
class GPUGroupingResult:
    sorted_token_indices: torch.Tensor
    sorted_expert_indices: torch.Tensor
    expert_offsets: torch.Tensor
    inverse_indices: torch.Tensor
    num_tokens: int
    top_k: int
    num_experts: int
    # Has .to_dispatch_info() converter
```

### CachedWeightBuffers (Trellis)

```python
@dataclass
class CachedWeightBuffers:
    gate_weights, gate_scales
    up_weights, up_scales
    down_weights, down_scales
    gate_su, gate_sv, up_su, up_sv, down_su, down_sv
    grid
    # All Metal buffers, created once, reused per call
```

---

## Call Graph (High-Level)

```
TrellisMoEMLP.forward()
  └── dispatch_moe_trellis_swiglu(..., cmd_manager)
          ├── prepare_fairway_grouped_inputs() [if mixed BPW]
          │       └── _group_tokens_mixed_bpw_primary_gpu()
          │               └── group_tokens_by_expert_full_gpu()
          │                       └── group_tokens_by_expert_gpu_optimized()
          │                               └── group_tokens_by_expert_gpu()  ← Metal kernel
          └── cmd_manager.dispatch("gemm_trellis_moe_swiglu", ...)

LayerBatchContext
  └── AsyncCommandBufferManager.start_batch()
          └── dispatch_moe_trellis_swiglu(..., cmd_manager) [×N layers]
          └── AsyncCommandBufferManager.commit_batch()
          └── AsyncCommandBufferManager.waitUntilCompleted()
```

---

## Recommendations Summary

| Priority | Item | Action |
|----------|------|--------|
| P0 | `cmd_manager=None` bypass | Add runtime warning when `None`; auto-create manager if not provided |
| P0 | `BatchedDispatcher` vs `AsyncCommandBufferManager` | Deprecate `BatchedDispatcher`; converge on `AsyncCommandBufferManager` |
| P1 | Token grouping API surface | Collapse 5+ grouping functions → 2 (`cpu`, `gpu`) with auto-detection |
| P1 | MixedBPW path integration | Merge `dispatch_moe_trellis_swiglu_grouped_fairway` into main path |
| P2 | Historical archive cleanup | Resolved — removed from the live tree; use Git history for reference |
| P2 | `GPUGroupingResult` removal | Remove `.to_dispatch_info()` wrapper; use `MoEDispatchInfo` directly |
| P3 | `moe_scatter_gather.py` | Evaluate if deduplicated against `scatter_expert_outputs` in `moe_dispatch.py` |

---

## Appendix: Removed Experimental Module Summary

| Deprecated File | Original Purpose | Superseded By |
|-----------------|------------------|---------------|
| `batched_dispatch.py` | Early batched expert execution | `AsyncCommandBufferManager` |
| `entropy_regularization.py` | Load balancing via entropy | `compute_load_balancing_loss` |
| `expert_grouping.py` | Token grouping logic | `gpu_grouping.py` |
| `expert_memory_pool.py` | Expert weight caching | `ExpertCache` in `trellis/moe.py` |
| `expert_selection_cache.py` | Routing decision cache | Per-layer routing caches |
| `fused_router.py` | Fused router kernel | `_fused_router_topk_metal` in `moe_dispatch.py` |
| `moe_dispatch_metal.py` | Metal-specific dispatch | `AsyncExpertDispatcher` in `moe_dispatch_metal.py` |
| `prefetch.py` | Expert prefetching | `ExpertCache` speculative loading |
| `sorted_dispatch.py` | Sorted token dispatch | `group_tokens_by_expert` family |
| `sparse_dispatch.py` | Sparse expert routing | `group_tokens_by_expert_sparse` |
| `sparse_routing.py` | Sparse routing variants | Consolidated routing |
| `token_dispatcher.py` | Token dispatcher base | `MoEDispatcher` class |
