# User Guides

Step-by-step tutorials for common Metal Marlin workflows.

## Getting Started

- [**Getting Started**](getting_started.md) — Install, quantize, run in 5 minutes
- [**Building from Source**](building.md) — Compile kernels and build local artifacts
- [**Development Setup**](development_setup.md) — Set up and manage the dev environment
- [**CLI Reference**](cli.md) — Command-line tools and options

## Workflows

- [**Serving Models**](serving.md) — OpenAI-compatible API server
- [**Deployment Checklist**](deployment_checklist.md) — Best practices for deployment
- [**Calibration Guide**](calibration.md) — Custom calibration for higher quality
- [**Mixed Precision**](../concepts/mixed_precision.md) — Per-layer precision for MoE models
- [**Mixed BPW Inference**](mixed_bpw_inference.md) — Developer workflow for mixed bit-width inference
- [**Migration Guide**](migration_guide.md) — Migrate from other frameworks

## Optimization

- [**Optimization Guide**](optimization.md) — Comprehensive performance tuning guide
- [**Optimization Workflow**](optimization_workflow.md) — Repeatable optimization process for regressions
- [**Mixed BPW Autotuning**](autotune_mixed_bpw.md) — Auto-tune mixed bit-width trellis kernels
- [**FLOPs Profiling**](profile_ops_guide.md) — Profile per-op FLOPs and hotspot math

## Troubleshooting

- [**Troubleshooting**](troubleshooting.md) — Common issues and solutions
- [**Metallib Troubleshooting**](metallib_troubleshooting.md) — Diagnose precompiled shader issues
- [**Metal Compiler Error Guide**](metal_error_guide.md) — Common compiler failures and fixes
