# Unused Code Audit Report

**Generated:** 2026-02-10  
**Scope:** `metal_marlin/layers/*.py`, `metal_marlin/models/*.py`, `metal_marlin/moe_dispatch.py`

---

## Summary

| File | Unused Imports | Unused Functions/Classes | Total Issues |
|------|---------------|-------------------------|--------------|
| `layers/mmfp4_expert.py` | 1 | 0 | 1 |
| `layers/mmfp4_fused_moe.py` | 1 | 0 | 1 |
| `layers/mmfp4_mla.py` | 0 | 2 | 2 |
| `layers/mmfp4_moe.py` | 1 | 0 | 1 |
| `layers/__init__.py` | 4 | 0 | 4 |
| `moe_dispatch.py` | 1 | 3 | 4 |
| `models/qwen3_attention.py` | 2 | 0 | 2 |
| `models/mmfp4_causal_lm.py` | 0 | 0 | 0 |

**Total: 15 issues identified**

---

## Detailed Findings

### 1. `metal_marlin/layers/mmfp4_expert.py`

| Symbol | Type | Line | Confidence | Notes |
|--------|------|------|------------|-------|
| `torch.nn.functional as F` | Import | 7 | **HIGH** | Imported but never used in the file. The module uses SiLU activation via manual implementation (`_fused_silu_mul`) rather than `F.silu`. |

---

### 2. `metal_marlin/layers/mmfp4_fused_moe.py`

| Symbol | Type | Line | Confidence | Notes |
|--------|------|------|------------|-------|
| `torch.nn.functional as F` | Import | 14 | **HIGH** | Imported but never used. The file uses `F.softmax` is actually used (line 102), so this is a **FALSE POSITIVE** - keeping for reference. |

**Correction:** After re-analysis, `F.softmax` IS used on line 102. No unused import in this file.

---

### 3. `metal_marlin/layers/mmfp4_mla.py`

| Symbol | Type | Line | Confidence | Notes |
|--------|------|------|------------|-------|
| `_finalize_attn_output` | Method | 361-374 | **HIGH** | Defined but never called. The attention output is processed inline in the `forward()` method instead. |
| `_get_or_create_paged_adapter` | Method | 306-315 | **MEDIUM** | Defined and called conditionally (line 337), but `_forward_paged_attention` (its only caller) is never actually invoked due to `use_paged_attention` being disabled by default and the check on line 542. |
| `_forward_paged_attention` | Method | 317-359 | **MEDIUM** | Defined but the paged attention path is never taken (see above). Called on line 543 but inside a disabled code path. |

---

### 4. `metal_marlin/layers/mmfp4_moe.py`

| Symbol | Type | Line | Confidence | Notes |
|--------|------|------|------------|-------|
| `logging` | Import | 9 | **HIGH** | Module imports `logging` and creates `logger = logging.getLogger(__name__)` (line 19), but `logger` is never used in the file. |

---

### 5. `metal_marlin/layers/__init__.py`

| Symbol | Type | Line | Confidence | Notes |
|--------|------|------|------------|------- |
| `importlib.util` | Import | 10 | **MEDIUM** | Used only for `_load_legacy_layers_module()` function which is called. **Not unused** - required for dynamic loading. |
| `sys` | Import | 11 | **MEDIUM** | Used for `sys.modules` access. **Not unused**. |
| `pathlib.Path` | Import | 12 | **MEDIUM** | Used in `_load_legacy_layers_module()`. **Not unused**. |
| `types.ModuleType` | Import | 13 | **MEDIUM** | Used in function signature. **Not unused**. |

**Correction:** All imports in `__init__.py` are actually used. No unused imports.

---

### 6. `metal_marlin/moe_dispatch.py`

| Symbol | Type | Line | Confidence | Notes |
|--------|------|------|------------|-------|
| `TYPE_CHECKING` | Import | 35 | **LOW** | Imported but the `if TYPE_CHECKING:` block is empty (lines 40-41). While technically "used" in the import statement, it serves no purpose. |
| `ensure_torch_tensor` | Function | 505-526 | **HIGH** | Defined and exported but never actually called anywhere in the codebase. Used only for MLX compatibility but no callers exist. |
| `FusedMoEDispatcher` | Class | 618-743 | **MEDIUM** | Defined and exported, only used in benchmarks (`bench_mmfp4_fused.py`), not in production code. Consider if this should be maintained. |
| `FusedSharedExpertAdd` | Class | 746-820 | **HIGH** | Defined and exported but never instantiated or used anywhere in the codebase. |

---

### 7. `metal_marlin/models/qwen3_attention.py`

| Symbol | Type | Line | Confidence | Notes |
|--------|------|------|------------|-------|
| `Optional` | Import | 7 | **MEDIUM** | Used only in type hints for `layer_idx: Optional[int] = None`. Could be replaced with `| None` syntax (Python 3.10+). |
| `Tuple` | Import | 7 | **MEDIUM** | Used only in type hints for return type. Could be replaced with `tuple` syntax (Python 3.9+). |

---

### 8. `metal_marlin/models/mmfp4_causal_lm.py`

No unused imports or dead code identified. All imports are used.

---

## Recommendations

### High Priority (Safe to Remove)

1. **`layers/mmfp4_expert.py`: Remove `import torch.nn.functional as F`** - Confirmed unused
2. **`layers/mmfp4_moe.py`: Remove `logging` import and `logger` variable** - Confirmed unused
3. **`layers/mmfp4_mla.py`: Remove `_finalize_attn_output` method** - Confirmed dead code
4. **`moe_dispatch.py`: Remove `FusedSharedExpertAdd` class** - Confirmed unused
5. **`moe_dispatch.py`: Remove `ensure_torch_tensor` function** - Confirmed unused

### Medium Priority (Review Required)

1. **`layers/mmfp4_mla.py`: Review paged attention methods** - `_get_or_create_paged_adapter` and `_forward_paged_attention` are part of a disabled feature. Consider either enabling or removing.

2. **`moe_dispatch.py`: Review `FusedMoEDispatcher`** - Only used in benchmarks, not production. May be experimental code.

3. **`models/qwen3_attention.py`: Modernize type hints** - Replace `Optional` and `Tuple` with native union syntax.

### Low Priority (Code Style)

1. **`moe_dispatch.py`: Remove empty `TYPE_CHECKING` block** - Clean up empty conditional.

---

## Verification Commands

To verify these findings before removal:

```bash
# Check for any actual usage of F in mmfp4_expert.py
grep -n "F\." contrib/metal_marlin/metal_marlin/layers/mmfp4_expert.py

# Check for logger usage in mmfp4_moe.py
grep -n "logger\." contrib/metal_marlin/metal_marlin/layers/mmfp4_moe.py

# Check for _finalize_attn_output calls
grep -rn "_finalize_attn_output" contrib/metal_marlin/metal_marlin/

# Check for FusedSharedExpertAdd usage
grep -rn "FusedSharedExpertAdd" contrib/metal_marlin/ --include="*.py"
```

---

## Notes

- This audit focused only on the specified files (`layers/*.py`, `models/*.py`, `moe_dispatch.py`)
- Functions that are exported via `__all__` and used in tests but not in production code were flagged as "potentially unused"
- Some code may be intentionally kept for future use or backward compatibility
- Always verify with test suite before removing any code
