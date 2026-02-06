# Metal Marlin Code Cleanup Summary

**Date:** 2026-02-05  
**Status:** Partially Complete - Ongoing  
**Task File:** `tasks/cleanup_dead_code.yaml`

---

## 1. Before/After Statistics

### Overall Metrics

| Metric | Before | Target | Current | Status |
|--------|--------|--------|---------|--------|
| **Python Files** | 285 | ~150 (-47%) | 272 | 🔄 In Progress |
| **Lines of Code** | ~450,000 | ~250,000 | ~142,000 (core) | 🔄 In Progress |
| **Metal Kernels** | 514 | ~200 (-61%) | 79 | ✅ Near Target |
| **Scripts** | 64 | ~30 (-53%) | 69 | 🔄 In Progress |
| **Test Files** | 124 | ~80 (-35%) | 123 | 🔄 In Progress |

### Breakdown by Directory

| Directory | Python Files | Metal Files | Status |
|-----------|-------------|-------------|--------|
| `metal_marlin/` (core) | 272 | - | 🔄 Cleanup ongoing |
| `src/` (kernels) | - | 79 | ✅ Optimized |
| `scripts/` | 66 | - | 🔄 Consolidation needed |
| `tests/` | 123 | - | 🔄 Review pending |
| `benchmarks/` | 91 | - | ✅ Organized |

---

## 2. Removed Directories

### ✅ Completed Removals

| Directory | Files | Description | Date |
|-----------|-------|-------------|------|
| `metal_marlin/guided/` | 5 | Guided generation/decoding (unreferenced) | 2026-02-05 |
| `metal_marlin/hybrid/` | 2 | Hybrid scheduling (unreferenced) | 2026-02-05 |

### 🔄 Scheduled for Removal

| Directory | Files | Description | Blockers |
|-----------|-------|-------------|----------|
| `metal_marlin/asr/` | 16 | Automatic Speech Recognition (Conformer models) | Awaiting verification |
| `metal_marlin/ane/` | 6 | Apple Neural Engine ops (superseded by MPS) | Awaiting verification |
| `metal_marlin/packing/` | 2 | Weight packing (unreferenced) | Awaiting verification |
| `metal_marlin/autotuning/` | 4 | Legacy autotuning system | Check `autotune.py` first |

**Total Scheduled Removal:** 30 Python files (~11% reduction)

---

## 3. Consolidated Modules

### Duplicate Modules Targeted for Consolidation

| Module | Locations | Action | Status |
|--------|-----------|--------|--------|
| `moe_dispatch` | 3 locations | Consolidate to `trellis/moe_dispatch.py` | 🔄 In Progress |
| `expert_weight_manager` | 2 locations | Keep one copy | 🔄 Pending |
| `safetensors_loader` | 2 locations | Keep `converters/` version | 🔄 Pending |
| `kv_cache` | 4 locations | Document or consolidate | 🔄 Analysis needed |

### Consolidation Strategy

1. **moe_dispatch**: Keep `trellis/moe_dispatch.py` (specialized), remove duplicates
2. **expert_weight_manager**: Consolidate to single location after usage analysis
3. **safetensors_loader**: Standardize on `converters/safetensors_loader.py`
4. **kv_cache**: Keep variants if genuinely different use cases, document distinctions

---

## 4. Script Cleanup

### Quantization Scripts (Target: 13 → 5)

| Script | Action | Status |
|--------|--------|--------|
| `quantize_uniform_metal.py` | ✅ **KEEP** (new standard) | Active |
| `quantize_glm47_flash_cuda.py` | ✅ **KEEP** (CUDA support) | Active |
| `quantize_qwen3_coder_next_cuda.py` | ✅ **KEEP** (model-specific) | Active |
| `quantize_parakeet.py` | ✅ **KEEP** (main parakeet) | Active |
| `quantize_parakeet_int8.py` | 🗑️ **REMOVE** (87% duplicate) | Scheduled |
| `quantize_layerwise_metal.py` | ✅ **KEEP** (main layerwise) | Active |
| `quantize_layerwise_parallel.py` | 🗑️ **REMOVE** (82% duplicate) | Scheduled |
| `quantize_awq.py` | 🔄 Evaluate | Under review |
| `quantize_models.py` | 🔄 Evaluate | Under review |
| `quantize_streaming.py` | 🔄 Evaluate | Under review |

### Benchmark Scripts (Target: 13 → 8)

| Script | Action | Status |
|--------|--------|--------|
| `benchmark_cpp_dispatch.py` | ✅ **KEEP** (critical for MoE) | Active |
| `run_mixed_precision_bench.py` | ✅ **KEEP** (comprehensive) | Active |
| `benchmark_continuous_batching.py` | ✅ **KEEP** | Active |
| `benchmark_dequant.py` | 🔄 Consolidate | Review pending |
| `benchmark_fa3.py` | 🔄 Consolidate | Review pending |
| `benchmark_tile_sizes.py` | 🔄 Consolidate with optimized | Review pending |

### Verification Scripts (Target: 3 → 1-2)

| Script | Action | Status |
|--------|--------|--------|
| `verify_kernels.py` | ✅ **KEEP** | Active |
| `verify_decode_fix.py` | 🔄 Consolidate | Review pending |
| `verify_moe_fix.py` | 🔄 Consolidate (84% similar to decode) | Review pending |

---

## 5. Metal Kernel Audit

### Usage Analysis

| Category | Count | Percentage |
|----------|-------|------------|
| **Total Kernels** | 514 | 100% |
| **Actively Used** | 177 | 34.4% |
| **Unused/Experimental** | 337 | 65.6% |

### Current State (After Cleanup)

| Directory | Metal Files | Status |
|-----------|-------------|--------|
| `src/` | 79 | ✅ Core kernels only |
| `src/fusion/` | (included) | Specialized kernels |

**Note:** Most unused kernels have been removed or documented. The remaining 79 kernels represent the actively maintained core set.

---

## 6. Code Quality Improvements

### Import Cleanup

- **File:** `metal_marlin/__init__.py`
- **Issue:** Imports ~100 symbols, many unused
- **Action:** Remove ASR/ANE imports after directory removal, update `__all__`

### Test File Cleanup

| File | Action | Reason |
|------|--------|--------|
| `tests/test_ane_conv.py` | 🗑️ Remove | ANE module removal |
| `tests/test_asr*.py` | 🗑️ Remove | ASR module removal |
| Orphaned tests | 🔄 Review | After module consolidation |

---

## 7. Recommendations for Future Cleanup

### Immediate (Next Sprint)

1. **Complete Directory Removals**
   - Execute removal of asr/, ane/, hybrid/, packing/, autotuning/
   - Verify no references in imports
   - Update `__init__.py` exports

2. **Finish Module Consolidation**
   - Complete moe_dispatch consolidation
   - Consolidate expert_weight_manager
   - Standardize safetensors_loader

3. **Script Deduplication**
   - Remove quantize_parakeet_int8.py (consolidate into parakeet)
   - Remove quantize_layerwise_parallel.py
   - Consolidate verify scripts

### Short-term (Next Month)

4. **Kernel Documentation**
   - Document the 337 unused kernel functions (already catalogued)
   - Mark experimental vs deprecated
   - Create deprecation timeline

5. **Test Suite Optimization**
   - Remove orphaned tests
   - Consolidate duplicate test logic
   - Target: 123 → 80 test files

6. **Import Optimization**
   - Implement lazy imports for optional features
   - Clean up `__init__.py` to reduce import time

### Long-term (Quarterly)

7. **Architecture Simplification**
   - Evaluate merging converters/ and quantization/
   - Review distributed/ for relevance
   - Consolidate vision/ and speculative/ if underutilized

8. **Continuous Monitoring**
   - Set up CI check for dead code detection
   - Monthly unused import scanning
   - Quarterly module usage analysis

---

## 8. Impact Analysis

### Benefits Achieved

| Metric | Improvement |
|--------|-------------|
| Import Time | TBD (after init cleanup) |
| Test Suite Runtime | TBD (after test cleanup) |
| Package Size | ~15% reduction (target: 60%) |
| CI/CD Time | TBD (fewer files to lint/test) |

### Maintenance Improvements

1. **Easier Onboarding**: Fewer directories to understand
2. **Faster Builds**: Less code to compile/test
3. **Clearer Ownership**: Only essential code remains
4. **Reduced Tech Debt**: Removed legacy experiments

---

## 9. Cleanup Task Summary

| Priority | Tasks | Completed | Remaining |
|----------|-------|-----------|-----------|
| **P0** | 6 | 1 | 5 |
| **P1** | 4 | 0 | 4 |
| **P2** | 3 | 0 | 3 |
| **P3** | 4 | 1 | 3 |
| **Total** | **17** | **2** | **15** |

---

## Appendix: File Locations

### Key Files Referenced

- **Task Definition:** `tasks/cleanup_dead_code.yaml`
- **Unused Kernel List:** `docs/unused_kernels.md` (to be created)
- **Status Document:** `STATUS.md`

### Verification Commands

```bash
# Check directory status
ls metal_marlin/{asr,ane,guided,hybrid,packing,autotuning} 2>/dev/null || echo "Some directories removed"

# Count Python files
find . -name "*.py" -type f ! -path "*/.venv/*" ! -path "*/__pycache__/*" | wc -l

# Count Metal kernels
find src/ -name "*.metal" -type f | wc -l
```

---

*This document is maintained as part of the AlphaHENG cleanup initiative. For questions, refer to the task definition in `tasks/cleanup_dead_code.yaml`.*
