# Changelog

This file tracks AlphaHENG-local changes, operator-facing behavior changes, and concise upstream summaries for `metal_marlin`.

## Recording rules

- Keep `Unreleased` current.
- Record kernel behavior, dispatch behavior, benchmark interpretation changes, and validation updates.
- Summarize the effect of syncs or research sweeps instead of copying commit logs.

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
