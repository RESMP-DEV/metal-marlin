# GLM-4.7 Mixed-BPW Optimization Report

**Date:** 2026-02-06  
**Result JSON:** `contrib/metal_marlin/benchmarks/results/mixed_bpw_decode_optimized.json`

## Decode Comparison (Baseline vs Optimized)

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| Decode ms/token | 9910.00 | 8213.33 | -1696.67 (-17.12%) |
| Decode tok/s | 0.1009 | 0.1218 | +0.0208 (+20.66%) |
| Fallback count | 46 | 46 | 0 |

## Notes

- Kernel selection updates are present and verified (`kernel_selection_results.json`):
  - Batch thresholds: decode `<=1`, prefill4 `2-16`, base `17-32`, large-batch `>=33`.
  - Specialized decode kernels are selected for mixed tuples: `6_2_3`, `6_3_4`, `6_2_4`.
- Grouped dispatch stability for mixed-bpw decode is currently stable on the fallback path:
  - Both runs show the same fallback warning count (`46 -> 46`), indicating no regression but no fallback reduction yet.
- Baseline and optimized values above are computed from the existing decode logs:
  - `contrib/metal_marlin/benchmarks/results/baseline_glm4_pre.txt`
  - `contrib/metal_marlin/benchmarks/results/post_phase1.txt`
