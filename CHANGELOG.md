# Changelog

This file tracks AlphaHENG-local changes, operator-facing behavior changes, and concise upstream summaries for `metal_marlin`.

## Recording rules

- Keep `Unreleased` current.
- Record kernel behavior, dispatch behavior, benchmark interpretation changes, and validation updates.
- Summarize the effect of syncs or research sweeps instead of copying commit logs.

## [Unreleased] - 2026-05-05

### Added
- Mixed-BPW autotuner compatibility helpers, runtime lookup/export state, and
  feedback-based kernel selection for decode/prefill mixed-bit workloads.
- Predictive MoE expert prefetch helper and a backward-compatible
  `metal_marlin.trellis_config` import surface.
- Qwen 3.6 27B shape contract (`agent_workspace/qwen36_27b/shape_contract.json`):
  canonical text-runtime dimensions fetched from HuggingFace `Qwen/Qwen3.6-27B`
  config.json. Confirmed as a **dense** SwiGLU model (hidden_size=5120, 64 layers,
  intermediate_size=17408, GQA 24Q/4KV), fundamentally different from the MoE-based
  Qwen3.6-35B-A3B. No full model weights were loaded.
- Apple Silicon performance contract for the Qwen3.6 fused path, including
  FP32 accumulation policy, packed-int4 decode rules, simdgroup assumptions,
  launch-consolidation targets, and template-artifact benchmark limits.

### Changed
- Hardened Trellis/MMFP4 compatibility tests to import package modules directly
  and use bounded synthetic values for MPS numerical checks.
- Aligned `gelu_metal` with the tanh approximation used by transformer MLP
  kernels.
- Parallelized the Qwen3.6-27B int4 projection and DeltaNet update kernels and
  aligned the dispatch/test coverage with the new threadgroup launch shapes.
- Added a direct benchmark option to skip LM-head dispatches for block-body
  bottleneck measurement.
- Added static MPS buffer reuse and CPU-prepared bit-group payloads for Trellis
  mixed-BPW dispatch while keeping tuple-scoped fallback deterministic.

### Fixed
- Preserved generated tokens when streaming mocks and real generate paths differ
  on whether the prompt is passed through the streamer.
- Routed unmasked attention through PyTorch SDPA directly while preserving causal
  handling.
- Accepted both packed-weight orientations in the MMFP4 expert MLP CPU fallback
  and sanitized non-finite TrellisLinear fallback outputs without changing the
  public output dtype.
- Corrected buffer-pool accounting, paged-cache COW accounting, FP4 dequant
  dispatch, and block-sparse attention fallback behavior for local Metal runs.

## [Unreleased] - 2026-02-10

### Fixed
- Defined the `_compat` module logger before feature-flag initialization so
  Linux validation can import the package.
- Kept MMFP4 layer iteration on CPU when the default MPS target is unavailable
  so Linux validation does not attempt unsupported tensor transfers.
- Scoped PyObjC runtime dependencies to macOS so Linux validation can install
  Metal Marlin without trying to build `pyobjc-framework-metal`.
- MLA GQA shape bug: attn_output.view() shape mismatch with 8192 vs 4096 elements
- MoE dispatch logging: now correctly reports "fused" vs "sequential"
- MoE fallback: sequential dispatch works on CPU when fused unavailable

### Added
- Fused MoE dispatch: 95% reduction in kernel dispatches (422 vs ~9024)
- PagedAttention adapter for MMFP4 MLA decode path
- GQA shape verification tests (MHA, GQA 2:1, 4:1, 16:1, MQA)
- dispatch counter tool for profiling

### Changed
- developer_tests/ audit and cleanup (see AUDIT_REPORT.md)
