# MR-GPTQ + Trellis Performance Summary

Generated: 2026-02-07T08:24:52.047948+00:00

## 1) MR-GPTQ backend timing deltas (GLM vs Qwen workloads)

- Baseline backend: `numpy`
- Optimized backend: `mps` (unavailable in current environment)

| Workload | Hessian (ms) | Cholesky (ms) | Quantize (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| glm47_moe (baseline) | 14.996 | 170.112 | 260.018 | 448.756 |
| qwen3_coder_next (baseline) | 14.910 | 169.331 | 162.462 | 347.831 |

GLM vs Qwen delta on baseline backend (`glm47_moe - qwen3_coder_next`):

| Metric | Delta | Delta % vs Qwen |
|---|---:|---:|
| hessian_ms | 0.085 | 0.57% |
| cholesky_ms | 0.780 | 0.46% |
| quantize_ms | 97.556 | 60.05% |
| total_ms | 100.925 | 29.02% |

## 2) Trellis decode throughput deltas (GLM vs Qwen presets)

- GLM preset (measured):
  - Baseline: 6636.677 ms/token, 0.150678 tok/s
  - Optimized: 4687.295 ms/token, 0.213343 tok/s
  - Delta: -1949.382 ms/token (-29.37%), 0.062665 tok/s (41.59%)

- Qwen preset (measured): unavailable
  - Reason: No Qwen decode regression artifact was found in benchmarks/results for this report window.

- Historical GLM artifact delta (`mixed_bpw_decode_optimized.json`):
  - ms/token: -1696.6667 (-17.12%)
  - tok/s: 0.0208 (20.66%)

## 3) Remaining fallback counters and known bottlenecks

- Historical fallback warnings: baseline=46, optimized=46, delta=0
- Latest structured fallback counters (all captured as reported):
  - `trellis_mixed_bpw_grouping`: `{"counters": {"grouping_calls_total": 0, "grouping_cpu_fallback_total": 0, "grouping_gpu_primary_success_total": 0}}`
  - `trellis_moe_dispatch`: `{"cpu_router_fallback": 0, "metal_router_calls": 0, "metal_router_success": 0, "total_experts_activated": 0, "total_tokens_processed": 0}`
  - `trellis_moe_metrics`: `{"fallback_used": 0, "fast_path_used": 0, "nan_detected": 0, "tokens_processed": 0}`
  - `trellis_nan_guard`: `{"last_nan_timestamp": null, "layers_affected": {}, "most_affected_layer": null, "sample_triggering_inputs": [], "total_failures": 0, "total_recoveries": 0}`
- Known bottlenecks:
  - Mixed-BPW decode still spends most time in decode path (multi-second/token), despite recovery gains.
  - Qwen decode preset regression artifact is missing, so GLM vs Qwen decode comparison remains incomplete.
  - MR-GPTQ optimized backend (MPS) unavailable in this execution environment; only baseline NumPy backend timings were measurable.
  - Historical mixed-BPW runs reported persistent per-expert fallback warnings (count unchanged at 46).
