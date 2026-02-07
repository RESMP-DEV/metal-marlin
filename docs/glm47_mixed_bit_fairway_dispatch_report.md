# GLM-4.7 Mixed-Bit Fairway Dispatch Report

**Date:** 2026-02-06  
**Model:** `GLM-4.7-Flash-Trellis-MM`  
**Primary artifacts:**  
- `contrib/metal_marlin/benchmarks/results/mixed_bpw_decode_optimized.json`  
- `contrib/metal_marlin/benchmarks/results/mixed_bpw_kernel_selection.json`  
- `contrib/metal_marlin/benchmarks/results/kernel_selection_results.json`  
- `contrib/metal_marlin/benchmarks/results/baseline_glm4_pre.txt`  
- `contrib/metal_marlin/benchmarks/results/post_phase1.txt`  
- `contrib/metal_marlin/metal_marlin/trellis/kernel_selection.py`  
- `contrib/metal_marlin/metal_marlin/trellis/moe_dispatch.py`  
- `contrib/metal_marlin/src/gemm_trellis_moe.metal`  

## 1. Kernel symbol inventory summary.

### Fairway/MoE dispatch symbol set (`src/gemm_trellis_moe.metal`)

| Category | Symbols |
|---|---|
| Core dispatch kernels | `moe_trellis_swiglu`, `moe_trellis_swiglu_fp32acc`, `moe_trellis_swiglu_large_batch`, `moe_trellis_swiglu_simd`, `moe_trellis_swiglu_decode`, `moe_trellis_swiglu_prefill4`, `moe_trellis_swiglu_prefill4_fp32acc`, `moe_trellis_swiglu_grouped` |
| Grouping helpers | `moe_count_tokens_per_expert`, `moe_scatter_tokens_to_experts` |

### Policy-level tuple-specialized decode symbols

- Declared by selector policy:  
  `moe_trellis_swiglu_decode_6_2_3`, `moe_trellis_swiglu_decode_6_3_4`, `moe_trellis_swiglu_decode_6_2_4`
- Current source inventory note: no direct `kernel void` entry points with those exact names were found in `src/*.metal`.

### Current tree census (for context)

- `src/*.metal`: 73 files  
- `kernel void` entries across `src/*.metal`: 511

## 2. Baseline vs optimized decode ms/token and tok/s.

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| Decode ms/token | 9910.00 | 8213.33 | -1696.67 (-17.12%) |
| Decode tok/s | 0.1009 | 0.1218 | +0.0208 (+20.66%) |
| Fallback count | 46 | 46 | 0 |

## 3. Grouped fairway dispatch hit rate and fallback reasons.

### Hit rate

- Estimated grouped fairway hit rate on the compared decode artifacts: **0.0%**.
- Calculation basis: 46 fallback warnings observed and no grouped-dispatch success events in the compared run logs.

### Fallback reason distribution (from warning lines)

All fallback warnings in both baseline and optimized logs carry the same reason class:

- `Mixed-precision MoE detected ... Fused batched dispatch disabled - using per-expert Metal dispatch.`

Bit-set breakdown (`46` total fallback events):

| Bit-set signature | Count | Share |
|---|---:|---:|
| `[2, 3, 4, 5, 6]` | 31 | 67.4% |
| `[2, 3, 4, 6]` | 13 | 28.3% |
| `[2, 3, 6]` | 2 | 4.3% |

Notes:
- Baseline and optimized distributions are identical (`46 -> 46`), indicating stable fallback behavior rather than a routing regression.
- Counter snapshots in `mixed_bpw_decode_with_group_counters.json` report zero grouping calls for that specific run, so warning-log analysis is the reliable fallback-reason source here.

## 4. Kernel-selection policy notes for GLM dominant tuples.

### Batch-range policy (M4 Max)

- `batch_size <= 1`: decode path
- `2 <= batch_size <= 16`: `prefill4`
- `17 <= batch_size <= 32`: base kernel
- `batch_size >= 33`: `large_batch` (`tile_n=128`)

### GLM dominant tuple handling

- Primary tuple: `(6, 2, 3)` -> `moe_trellis_swiglu_decode_6_2_3`
- Secondary tuple: `(6, 3, 4)` -> `moe_trellis_swiglu_decode_6_3_4`
- Tertiary tuple: `(6, 2, 4)` -> `moe_trellis_swiglu_decode_6_2_4`

### Selection caveats

- Tuple specialization is decode-only (`batch_size == 1`, non-FP32 path).
- Policy fallback for tuple mismatch or unavailable specialized kernel is generic `moe_trellis_swiglu_decode`.
- The `select_moe_kernel()` wrapper currently calls `get_kernel_for_batch_size(...)` without an `available_kernels` set, so specialized tuple routing depends on caller behavior/overrides rather than unconditional runtime selection.

## 5. Next tuning knobs for `optimize_kernel.py`/`optimize_structural_v2.py`.

### `scripts/optimize_kernel.py` (parameter-space tuning)

| Knob | Why it matters for fairway decode | Suggested next sweep |
|---|---|---|
| `--profile mixed_bpw_fairway_glm47` | Targets GLM-4.7 mixed-BPW decode/prefill and grouped-dispatch shapes | Keep as default profile for this effort |
| `--num-random` | Increases entropy search over Metal-specific non-obvious optima | 20 -> 40/60 (two passes) |
| `--random-seed` | Reproducible variant sets for A/B reruns | Fix seed per pass; rotate between passes |
| `--no-explore` (ablation) | Separates deterministic vs exploratory gains | Run one deterministic control session |
| `--iters` / `--warmup` | Reduces noisy ranking near small deltas | Warmup 10, iters 50 for finalists |

Priority parameter families to emphasize in generated variants:
- `TILE_N` (`64` vs `128`) for decode/prefill boundary behavior
- `SIMDGROUPS_PER_TG` (`1/2/4`) for occupancy-pressure balancing
- `SG_M_TILES`, `SG_N_TILES` for simdgroup decomposition
- `NUM_BUFFERS` (`2/3/4`) for memory-latency hiding

### `scripts/optimize_structural_v2.py` (structural transforms)

| Knob | Why it matters | Suggested next sweep |
|---|---|---|
| `--kernel gemm_trellis_moe.metal --run` | Hits the fairway dispatch kernel source directly | Run first as single-kernel direct benchmark |
| `simdgroup_barrier` transform | Can cut over-synchronization overhead in MoE paths | Evaluate first for MoE kernels |
| `async_prefetch_hint` transform | Guides memory-latency-hiding edits for bandwidth-bound regions | Evaluate with MoE + GEMM kernels |
| `--list` before task generation | Prevents generating no-op transforms | Use before each session |

Recommended execution order:
1. `optimize_kernel.py` with `mixed_bpw_fairway_glm47` profile to find top parameter candidates.  
2. `optimize_structural_v2.py --run` on `gemm_trellis_moe.metal` for structural deltas.  
3. Re-run mixed-BPW decode benchmark and compare fallback-hit/fallback-reason distributions.
