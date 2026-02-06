# Benchmarks

This directory contains benchmark scripts for evaluating the performance of metal_marlin components.

## Quick Reference

| Benchmark | Purpose | Command |
|-----------|---------|---------|
| `benchmark_cpp_dispatch.py` | MoE C++ vs Python dispatch latency | `uv run python scripts/benchmark_cpp_dispatch.py` |
| `run_mixed_precision_bench.py` | Mixed-precision MoE strategies | `uv run python scripts/run_mixed_precision_bench.py` |

## Scripts (contrib/metal_marlin/scripts/)

### Core Benchmarks (Keepers)

#### `benchmark_cpp_dispatch.py`
**Purpose:** Benchmark C++ FastPath dispatch latency vs Python/PyObjC dispatch.

**What it measures:**
- Per-dispatch overhead for MoE kernel launches
- Python dispatch: ~80-150μs per call
- C++ dispatch: ~5-15μs per call
- Expected speedup: 5-10x

**Usage:**
```bash
# Full benchmark
uv run python scripts/benchmark_cpp_dispatch.py

# Quick mode
uv run python scripts/benchmark_cpp_dispatch.py --quick

# Custom dispatch counts
uv run python scripts/benchmark_cpp_dispatch.py --counts 100 1000 5000
```

**Why keep:** Critical for MoE performance optimization. The 64-expert GLM-4.7 model
launches thousands of dispatches; this benchmark quantifies the C++ extension benefit.

---

#### `run_mixed_precision_bench.py`
**Purpose:** Comprehensive benchmark runner for mixed-precision MoE dispatch strategies.

**What it measures:**
- Slow path (sequential per-expert dispatch)
- Fast uniform (batched with uniform bits)
- Fast mixed (batched with per-projection bits)
- Hybrid (batched for common, sequential for rare)
- Max bits padded (pad to max bits, single dispatch)

**Usage:**
```bash
# Quick synthetic benchmark
uv run python scripts/run_mixed_precision_bench.py

# Full benchmark with all strategies
uv run python scripts/run_mixed_precision_bench.py --full

# Compare specific strategies
uv run python scripts/run_mixed_precision_bench.py --strategies slow_path fast_mixed hybrid

# Real GLM-4.7 model
uv run python scripts/run_mixed_precision_bench.py --model glm4
```

**Why keep:** The primary benchmark for mixed-precision MoE. Tests both synthetic
and real models with configurable strategies.

---

## Core Benchmarks (contrib/metal_marlin/benchmarks/)

### End-to-End Benchmarks

#### `baseline_benchmark.py`
Comprehensive baseline for comparison with optimized implementations.

#### `ab_test_kernels.py`
A/B testing framework for kernel variants with statistical significance testing.

### Component Benchmarks

#### `bench_gemm.py` / `bench_gemm_trellis.py`
GEMM kernel benchmarks for standard and Trellis-quantized matrices.

#### `bench_attention.py` / `bench_attention_variants.py`
Attention kernel benchmarks comparing different implementations.

#### `analyze_attention_bandwidth.py`
Detailed analysis of attention memory bandwidth utilization.

#### `bench_bf16_conversion.py` / `bench_bf16_optimized.py`
Benchmark BF16 conversion kernels (standard vs optimized).

#### `bench_dtype_perf.py`
Cross-data-type performance comparison (FP16, BF16, FP32).

### MoE-Specific Benchmarks

#### `bench_moe_kernel.py`
Benchmark individual MoE kernel implementations.

#### `bench_moe_kernel_dispatch.py`
Benchmark MoE dispatch mechanisms.

#### `bench_moe_multicase.py`
Multi-scenario MoE benchmarking.

#### `mixed_precision_bench.py`
Core mixed-precision benchmarking framework (used by run_mixed_precision_bench.py).

### Memory Benchmarks

#### `bench_kv_cache_layout.py`
Benchmark different KV cache memory layouts (BHSD, BSHD, HBSD).

#### `bench_unified_memory.py`
Test unified memory performance characteristics.

#### `bench_memory_access.py`
Memory access pattern benchmarks.

#### `bench_buffer_ops.py`
Low-level buffer operation benchmarks.

### Utility Benchmarks

#### `bench_clean.py`
Minimal clean-room benchmark for sanity checking.

#### `bench_components.py`
Component-level benchmark suite.

#### `bench_debug.py`
Debug/diagnostic benchmark utilities.

#### `bench_dynamic_dispatch.py`
Benchmark dynamic kernel dispatch overhead.

#### `bench_forward_profile.py`
Profile forward pass performance.

#### `bench_fused_e2e.py`
End-to-end fused kernel benchmark.

#### `bench_fused_vs_pytorch.py`
Compare fused kernels against PyTorch baseline.

### Model-Specific Benchmarks

#### `bench_glm47_trellis.py`
GLM-4.7-Flash Trellis-specific benchmark.

#### `bench_glm47_quality.py`
Quality/speed tradeoff for GLM-4.7.

#### `bench_glm4_throughput.py`
Throughput-focused GLM-4 benchmark.

### Specialized Benchmarks

#### `bench_quantization.py`
Quantization kernel benchmarks.

#### `bench_fp4_metal.py`
FP4-specific Metal kernel benchmarks.

#### `bench_kernel_selection.py`
Benchmark kernel selection strategies.

#### `bench_kernel_variants.py`
Compare different kernel implementations.

#### `bench_profile.py`
Profiling-focused benchmark suite.

#### `bench_trellis_performance.py`
Trellis-specific performance tests.

---

## Removed Benchmarks

The following benchmarks were removed due to redundancy:

| Removed Script | Reason | Replacement |
|---------------|--------|-------------|
| `benchmark_tile_sizes.py` | Highly specific tile tuning | Use kernel optimization workflow |
| `benchmark_optimized.py` | Simple validation only | Use `run_mixed_precision_bench.py --model glm4` |
| `benchmark_fa3.py` | FA3 not implemented (placeholders) | Use `bench_attention.py` for FA2 |
| `benchmark_dequant.py` | Redundant with kernel tests | Use `bench_quantization.py` |
| `bench_throughput.py` | Hessian-specific | Use `bench_glm4_throughput.py` |
| `benchmark_continuous_batching.py` | Complex mock-based | Use real serving benchmarks |
| `benchmark_kv_layout.py` | Merged into `bench_kv_cache_layout.py` |
| `benchmark_moe_fusion.py` | Redundant with `benchmark_cpp_dispatch.py` |
| `perf_comparison.py` | Static print script | Use `run_mixed_precision_bench.py` output |
| `perf_summary.py` | Static print script | Use `run_mixed_precision_bench.py` output |
| `quick_bench.py` | Quick GLM4 test | Use `run_mixed_precision_bench.py --model glm4` |

---

## Running Benchmarks

### Prerequisites
```bash
cd contrib/metal_marlin
uv sync --extra all
```

### Basic Usage
```bash
# Run a specific benchmark
uv run python benchmarks/bench_gemm.py

# Run script benchmark
uv run python scripts/benchmark_cpp_dispatch.py
```

### Full Benchmark Suite
```bash
# Core benchmarks
uv run python scripts/benchmark_cpp_dispatch.py --quick
uv run python scripts/run_mixed_precision_bench.py

# Component benchmarks
uv run python benchmarks/bench_gemm.py
uv run python benchmarks/bench_attention.py
```

---

## Benchmark Design Principles

1. **Minimal Overhead:** Benchmarks use `time.perf_counter()` with MPS synchronization
2. **Warmup:** All benchmarks include warmup iterations to prime caches
3. **Statistical Rigor:** Multiple iterations with mean/std reporting
4. **Memory Cleanup:** Explicit `gc.collect()` and `torch.mps.empty_cache()` between runs
5. **Device Agnostic:** Fall back to CPU if MPS unavailable (with warning)
