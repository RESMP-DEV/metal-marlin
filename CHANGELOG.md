# Changelog

## [Unreleased] - 2026-02-10

### Fixed
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