# Technical Audits

Investigation reports and bug analyses. These documents track issues discovered
during development and their resolutions.

## Kernel Audits

- [**Metal Kernel Audit**](metal_kernel_audit.md) — Comprehensive kernel review
- [**Shader Structural Audit**](shader_audit.md) — Shader code quality review
- [**Attention BF16 Audit**](attention_bf16_audit.md) — BF16 attention precision analysis
- [**GEMM BF16 Audit**](gemm_bf16_audit.md) — BF16 GEMM precision analysis
- [**GEMM Alignment Audit**](gemm_alignment_audit.md) — Memory alignment requirements
- [**MoE BF16 Audit**](moe_bf16_audit.md) — MoE BF16 precision analysis
- [**Barrier Optimization**](barrier_optimization.md) — Barrier and sync analysis

## Performance & Benchmarks

- [**Speculative Decoding Implementation Summary**](speculative_decoding_implementation_summary.md) — Draft model generation loop implementation details
- [**Batch Scheduler Implementation**](batch_scheduler_implementation.md) — Dynamic request scheduling behavior and usage
- [**MLA Projection Refactor Analysis**](mla_proj_refactor.md) — MLA kernel refactor flow and barrier notes
- [**Missing Reductions Analysis**](missing_reductions_analysis.md) — FP4 dequant reduction path analysis
- [**MPS Indexing Issue**](mps_indexing_issue.md) — Advanced indexing bottleneck investigation
- [**Parakeet Benchmarks**](parakeet_benchmarks.md) — ASR performance benchmarks
- [**Parakeet Layer Analysis**](parakeet_layers.md) — Layer-wise ASR breakdown

## Buffer & Memory Audits

- [**Buffer Copy Audit**](buffer_copy_audit.md) — Buffer copy performance
- [**Storage Mode Audit**](storage_mode_audit.md) — Metal storage mode analysis

## Bug Reports

- [**Metal Array Parameter Bugs**](metal_array_parameter_bugs.md) — Array parameter issues
- [**Metal Half Precision Bug**](metal_half_precision_bug.md) — Half precision edge cases
- [**Resolved Bugs**](resolved_bugs.md) — Previously resolved issues

## Debug Investigations

- [**MMFP4 Debug Summary**](mmfp4_debug_summary.md) — MMFP4 inference stack NaN debugging analysis

## Maintenance Summaries

- [**Code Cleanup Summary**](code_cleanup_summary.md) — Module/script cleanup and consolidation summary
- [**Final Cleanup Summary**](final_cleanup_summary.md) — Final project cleanup pass notes
