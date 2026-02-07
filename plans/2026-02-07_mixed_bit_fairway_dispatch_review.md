# Mixed Bit Fairway Dispatch Review and PR Plan

Date: 2026-02-07
Repo: `contrib/metal_marlin`
Branch base: `main`

## Scope
This document captures:
- What landed from the GLM-4.7 mixed-bit fairway task batch
- What is still missing or risky
- Current validation and performance signals
- PR scope and follow-up execution plan

## Task Batch Outcome
Queue status reached completed for the batch, but task-level review shows partial completion:

Completed (selected):
- `create-moe-kernel-symbol-inventory-script`
- `gate-specialized-decode-kernels-by-availability`
- `delegate-moe-dispatch-selection-to-kernel-selection-module`
- `add-fairway-grouping-prep-helper`
- `add-grouped-fairway-dispatch-entrypoint`
- `wire-per-bit-tuple-dispatch-to-grouped-fairway-kernel`
- `make-forward-grouped-fallback-granular-per-bit-tuple`
- `add-partial-fallback-regression-test-for-fairway-dispatch`
- `generate-atomic-fairway-autotune-task-yaml`
- benchmark arg rename tasks (`kernel_override` -> `kernel_name_override`)

Failed in queue metadata (but code appears present now):
- `add-kernel-name-override-parameter-to-dispatch`
- `add-mixed-bpw-fairway-profile-to-optimize-kernel-script`
- `plumb-fairway-profile-through-optimize-all-kernels`
- `add-fairway-mode-to-optimize-structural-v2`

Primary queue failure causes were orchestration/back-end retries and malformed verify patterns (see Known Issues).

## Landed Changes (High-Level)

### Dispatch and selection
- Added availability-gated decode specialization in `metal_marlin/trellis/kernel_selection.py`.
- `select_moe_kernel` in `metal_marlin/trellis/moe_dispatch.py` now delegates to `kernel_selection.get_kernel_for_batch_size`.
- Added benchmark override API in `dispatch_moe_trellis_swiglu(..., kernel_name_override=...)`.
- Added grouped fairway path:
  - `prepare_fairway_grouped_inputs(...)`
  - `dispatch_moe_trellis_swiglu_grouped_fairway(...)`
  - primary usage inside `dispatch_moe_per_bit_tuple(...)` with legacy fallback.

### Grouped fallback behavior
- `_forward_grouped(...)` in `metal_marlin/trellis/model.py` now supports per-bit-tuple granular fallback instead of all-or-nothing fallback.

### Scripts and benchmarks
- Added `scripts/inspect_moe_kernel_symbols.py`.
- Added profile plumbing:
  - `scripts/optimize_kernel.py --profile mixed_bpw_fairway_glm47`
  - `scripts/optimize_all_kernels.py --profile ...`
  - `scripts/optimize_structural_v2.py --mode mixed_bpw_fairway`
- Updated benchmark callers to `kernel_name_override=`.
- Added `scripts/generate_mixed_bpw_fairway_tasks.py` which generates `tasks/mixed_bpw_fairway_autotune.yaml`.

### Tests
- Added/updated tests:
  - `tests/test_kernel_selection.py`
  - `tests/test_fairway_grouped_inputs.py`
  - `tests/test_mixed_bpw_grouped_dispatch.py`
  - `tests/test_mixed_bpw_partial_group_fallback.py`

## Known Issues / Gaps

1) Fairway path is not yet the dominant hot path for mixed-BPW inference.
- `TrellisMoEMLP.forward_fast` and `forward` prefer `_dispatch_mixed_precision` when `_bit_group_buffers` exists.
- This can bypass the new fairway route in `_forward_grouped` / `dispatch_moe_per_bit_tuple`.
- Evidence: live benchmark counters show no grouping calls.

2) Specialized decode kernels are still missing in Metal symbols.
- Live inventory confirms missing:
  - `moe_trellis_swiglu_decode_6_2_3`
  - `moe_trellis_swiglu_decode_6_3_4`
  - `moe_trellis_swiglu_decode_6_2_4`
- Inventory artifact: `benchmarks/results/moe_kernel_symbol_inventory_live_review.json`.

3) Some report artifacts overstate specialization activity.
- `benchmarks/results/mixed_bpw_kernel_selection.json` and derived report content describe specialized decode kernels as active despite symbol absence.

4) Verify-command bug in source task YAML.
- `tasks/glm47_mixed_bit_fairway_dispatch.yaml` uses:
  - `rg -q "--profile" ...`
  - `rg -q "--mode" ...`
- `rg` interprets these as flags without `--`; should be `rg -q -- "--profile" ...` etc.

5) Minor test quality warning.
- `tests/test_kernel_selection.py` uses return codes in pytest test function; pytest warns that tests should return `None`.

## Validation Runbook and Current Results

### Tests run
Command:
```bash
cd contrib/metal_marlin
uv run pytest tests/test_kernel_selection.py tests/test_fairway_grouped_inputs.py tests/test_mixed_bpw_grouped_dispatch.py tests/test_mixed_bpw_partial_group_fallback.py -v
```
Result: 6 passed.

### Standalone kernel-selection test run
Command:
```bash
cd contrib/metal_marlin
uv run python tests/test_kernel_selection.py
```
Result: passes.

### Symbol inventory (live)
Command:
```bash
cd contrib/metal_marlin
uv run python scripts/inspect_moe_kernel_symbols.py --output benchmarks/results/moe_kernel_symbol_inventory_live_review.json
```
Result summary:
- available: 7
- missing: 3 specialized decode kernels

### Live decode benchmark (quick smoke)
Command:
```bash
cd contrib/metal_marlin
uv run python benchmarks/benchmark_mixed_bpw_decode.py --model-path models/GLM-4.7-Flash-Trellis-MM --prompt-len 8 --decode-tokens 4 --warmup 1 --runs 1 --output benchmarks/results/mixed_bpw_decode_live_review_2026-02-06.json
```
Result summary:
- ~6575 ms/token
- ~0.152 tok/s
- mixed-BPW grouping counters remained 0 in this run

## PR Scope (Current)
This PR is intended to capture the current landed delta as-is:
- Dispatch/kernel-selection integration
- Fairway grouped dispatch plumbing
- Script/profile improvements
- New tests and review documentation

## Follow-Up Plan After PR
1. Route mixed-BPW hot path through fairway grouped dispatch (or merge fairway into `_dispatch_mixed_precision`) and re-measure counters.
2. Add runtime-available-kernel set into selection call path where decode selection occurs.
3. Regenerate performance/report artifacts from live benchmark runs only; remove synthetic/stale claims.
4. Fix `rg` verify patterns in task YAML and add one meta-test for task-verify command linting.
