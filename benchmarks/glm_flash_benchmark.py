#!/usr/bin/env python3
"""Comprehensive benchmark for GLM-4.7-Flash on Apple Silicon.

Measures:
1. Model load time
2. Prefill throughput (tokens/sec for initial prompt)
3. Decode throughput (tokens/sec for generation)
4. Time per layer breakdown (attention, MoE/dense, other)
5. Memory usage (MPS allocated, peak)

Includes comparison with baseline (before optimizations) when available.

Usage:
    # Quick benchmark (reduced iterations for fast validation)
    .venv/bin/python benchmarks/glm_flash_benchmark.py --quick

    # Full benchmark
    .venv/bin/python benchmarks/glm_flash_benchmark.py

    # Custom settings
    .venv/bin/python benchmarks/glm_flash_benchmark.py --prefill-lengths 128,512,2048 --decode-tokens 200
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metal_marlin.inference.pipeline_v2 import TransformersMarlinPipeline

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

import torch
import torch.nn as nn

# Ensure project is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@dataclass
class LayerTiming:
    """Timing breakdown for a single layer."""

    layer_idx: int
    attention_ms: float
    mlp_ms: float
    norm_ms: float
    total_ms: float

    @property
    def attention_pct(self) -> float:
        return (self.attention_ms / self.total_ms * 100) if self.total_ms > 0 else 0.0

    @property
    def mlp_pct(self) -> float:
        return (self.mlp_ms / self.total_ms * 100) if self.total_ms > 0 else 0.0

    @property
    def other_pct(self) -> float:
        return 100.0 - self.attention_pct - self.mlp_pct


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    allocated_gb: float
    peak_gb: float
    model_size_gb: float


@dataclass
class PrefillMetrics:
    """Prefill performance metrics."""

    seq_len: int
    time_ms: float
    throughput_tok_s: float
    times: list[float] = field(default_factory=list)

    @property
    def time_std_ms(self) -> float:
        return statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0.0


@dataclass
class DecodeMetrics:
    """Decode performance metrics."""

    num_tokens: int
    total_time_s: float
    throughput_tok_s: float
    ms_per_token: float
    p50_ms: float
    p99_ms: float
    latencies_ms: list[float] = field(default_factory=list)


@dataclass
class LayerBreakdown:
    """Layer timing breakdown across all layers."""

    attention_total_ms: float
    mlp_total_ms: float
    other_total_ms: float
    layer_timings: list[LayerTiming] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.attention_total_ms + self.mlp_total_ms + self.other_total_ms

    @property
    def attention_pct(self) -> float:
        return (self.attention_total_ms / self.total_ms * 100) if self.total_ms > 0 else 0.0

    @property
    def mlp_pct(self) -> float:
        return (self.mlp_total_ms / self.total_ms * 100) if self.total_ms > 0 else 0.0

    @property
    def other_pct(self) -> float:
        return (self.other_total_ms / self.total_ms * 100) if self.total_ms > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    model_name: str
    load_time_s: float
    memory: MemoryMetrics
    prefill_metrics: list[PrefillMetrics]
    decode_metrics: DecodeMetrics
    layer_breakdown: LayerBreakdown | None
    hardware_info: dict[str, Any]


def _mps_sync() -> None:
    """Synchronize MPS device."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _percentile(values: list[float], quantile: float) -> float:
    """Compute percentile of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * quantile), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _get_memory_gb() -> float:
    """Get current MPS memory allocation in GB."""
    if hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def _get_hardware_info() -> dict[str, Any]:
    """Get hardware information."""
    info: dict[str, Any] = {
        "platform": sys.platform,
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }

    try:
        from metal_marlin.profiling.occupancy import detect_gpu

        gpu = detect_gpu()
        info["gpu_name"] = gpu.name.replace("_", " ").title()
        info["gpu_cores"] = gpu.gpu_cores
        info["memory_bandwidth_gbs"] = gpu.memory_bandwidth_gbs
    except Exception:
        info["gpu_name"] = "Unknown"
        info["gpu_cores"] = 0
        info["memory_bandwidth_gbs"] = 0.0

    return info


class LayerTimingHooks:
    """Context manager for layer timing hooks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.timings: dict[int, dict[str, float]] = {}
        self._hooks: list[Any] = []
        self._layer_start_times: dict[int, float] = {}
        self._attn_times: dict[int, float] = {}
        self._mlp_times: dict[int, float] = {}

    def __enter__(self) -> LayerTimingHooks:
        """Install timing hooks on model layers."""
        # Find layers in the model
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers

        if layers is None:
            return self

        for layer_idx, layer in enumerate(layers):
            self.timings[layer_idx] = {
                "attention_ms": 0.0, "mlp_ms": 0.0, "total_ms": 0.0}

            # Hook for attention
            if hasattr(layer, "self_attn") and layer.self_attn is not None:

                def make_attn_pre_hook(idx: int):
                    def hook(module: nn.Module, inputs: Any) -> None:
                        _mps_sync()
                        self._attn_times[idx] = time.perf_counter()

                    return hook

                def make_attn_post_hook(idx: int):
                    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
                        _mps_sync()
                        elapsed = (time.perf_counter() -
                                   self._attn_times[idx]) * 1000
                        self.timings[idx]["attention_ms"] += elapsed

                    return hook

                self._hooks.append(
                    layer.self_attn.register_forward_pre_hook(
                        make_attn_pre_hook(layer_idx))
                )
                self._hooks.append(
                    layer.self_attn.register_forward_hook(
                        make_attn_post_hook(layer_idx))
                )

            # Hook for MLP
            if hasattr(layer, "mlp") and layer.mlp is not None:

                def make_mlp_pre_hook(idx: int):
                    def hook(module: nn.Module, inputs: Any) -> None:
                        _mps_sync()
                        self._mlp_times[idx] = time.perf_counter()

                    return hook

                def make_mlp_post_hook(idx: int):
                    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
                        _mps_sync()
                        elapsed = (time.perf_counter() -
                                   self._mlp_times[idx]) * 1000
                        self.timings[idx]["mlp_ms"] += elapsed

                    return hook

                self._hooks.append(
                    layer.mlp.register_forward_pre_hook(
                        make_mlp_pre_hook(layer_idx))
                )
                self._hooks.append(layer.mlp.register_forward_hook(
                    make_mlp_post_hook(layer_idx)))

            # Hook for entire layer
            def make_layer_pre_hook(idx: int):
                def hook(module: nn.Module, inputs: Any) -> None:
                    _mps_sync()
                    self._layer_start_times[idx] = time.perf_counter()

                return hook

            def make_layer_post_hook(idx: int):
                def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
                    _mps_sync()
                    elapsed = (time.perf_counter() -
                               self._layer_start_times[idx]) * 1000
                    self.timings[idx]["total_ms"] += elapsed

                return hook

            self._hooks.append(layer.register_forward_pre_hook(
                make_layer_pre_hook(layer_idx)))
            self._hooks.append(layer.register_forward_hook(
                make_layer_post_hook(layer_idx)))

        return self

    def __exit__(self, *args: Any) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_breakdown(self) -> LayerBreakdown:
        """Get layer breakdown from collected timings."""
        layer_timings = []
        attention_total = 0.0
        mlp_total = 0.0
        other_total = 0.0

        for layer_idx, timing in sorted(self.timings.items()):
            attention_ms = timing["attention_ms"]
            mlp_ms = timing["mlp_ms"]
            total_ms = timing["total_ms"]
            other_ms = max(0.0, total_ms - attention_ms - mlp_ms)

            layer_timings.append(
                LayerTiming(
                    layer_idx=layer_idx,
                    attention_ms=attention_ms,
                    mlp_ms=mlp_ms,
                    norm_ms=other_ms,  # Includes norms and residuals
                    total_ms=total_ms,
                )
            )

            attention_total += attention_ms
            mlp_total += mlp_ms
            other_total += other_ms

        return LayerBreakdown(
            attention_total_ms=attention_total,
            mlp_total_ms=mlp_total,
            other_total_ms=other_total,
            layer_timings=layer_timings,
        )

    def reset(self) -> None:
        """Reset all timing counters."""
        for layer_idx in self.timings:
            self.timings[layer_idx] = {
                "attention_ms": 0.0, "mlp_ms": 0.0, "total_ms": 0.0}


def benchmark_model_load(model_path: str) -> tuple[TransformersMarlinPipeline, float, MemoryMetrics]:
    """Benchmark model loading time and memory."""
    gc.collect()
    torch.mps.empty_cache()

    initial_mem = _get_memory_gb()

    print(f"Loading model from {model_path}...")
    start = time.perf_counter()
    pipeline = TransformersMarlinPipeline.from_pretrained(
        model_path, device="mps")
    _mps_sync()
    load_time = time.perf_counter() - start

    _mps_sync()
    allocated = _get_memory_gb()
    model_size = allocated - initial_mem

    # Run a dummy forward to trigger any lazy initialization
    with torch.no_grad():
        dummy = pipeline.tokenizer.encode(
            "Hello", return_tensors="pt").to("mps")
        _ = pipeline.model(dummy)
    _mps_sync()

    peak = _get_memory_gb()

    memory = MemoryMetrics(allocated_gb=allocated,
                           peak_gb=peak, model_size_gb=model_size)

    return pipeline, load_time, memory


def benchmark_prefill(
    pipeline: TransformersMarlinPipeline,
    seq_lengths: list[int],
    warmup: int = 2,
    iterations: int = 5,
) -> list[PrefillMetrics]:
    """Benchmark prefill throughput for various sequence lengths."""
    results = []
    model = pipeline.model

    for seq_len in seq_lengths:
        print(f"  Benchmarking prefill ({seq_len} tokens)...")
        input_ids = torch.randint(0, 1000, (1, seq_len), device="mps")

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_ids)
                _mps_sync()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                _mps_sync()
                start = time.perf_counter()
                _ = model(input_ids)
                _mps_sync()
                times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times)
        throughput = seq_len / avg_time

        results.append(
            PrefillMetrics(
                seq_len=seq_len,
                time_ms=avg_time * 1000,
                throughput_tok_s=throughput,
                times=times,
            )
        )

        print(f"    {throughput:.1f} tok/s ({avg_time * 1000:.1f} ms)")

    return results


def benchmark_decode(
    pipeline: TransformersMarlinPipeline,
    num_tokens: int = 100,
    warmup: int = 5,
    num_runs: int = 3,
) -> DecodeMetrics:
    """Benchmark decode throughput (single token generation)."""
    print(f"  Benchmarking decode ({num_tokens} tokens x {num_runs} runs)...")
    model = pipeline.model

    # Use single-token input to simulate decode phase
    input_ids = torch.randint(0, 1000, (1, 1), device="mps")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
            _mps_sync()

    # Benchmark
    all_latencies: list[float] = []
    run_times: list[float] = []

    with torch.no_grad():
        for run in range(num_runs):
            run_latencies: list[float] = []
            run_start = time.perf_counter()

            for _ in range(num_tokens):
                _mps_sync()
                step_start = time.perf_counter()
                _ = model(input_ids)
                _mps_sync()
                step_ms = (time.perf_counter() - step_start) * 1000
                run_latencies.append(step_ms)

            run_time = time.perf_counter() - run_start
            run_times.append(run_time)
            all_latencies.extend(run_latencies)

            avg_ms = statistics.mean(run_latencies)
            print(
                f"    Run {run + 1}: {1000 / avg_ms:.1f} tok/s ({avg_ms:.1f} ms/tok)")

    avg_time = statistics.mean(run_times)
    throughput = num_tokens / avg_time
    ms_per_token = statistics.mean(all_latencies)
    p50 = _percentile(all_latencies, 0.50)
    p99 = _percentile(all_latencies, 0.99)

    return DecodeMetrics(
        num_tokens=num_tokens * num_runs,
        total_time_s=sum(run_times),
        throughput_tok_s=throughput,
        ms_per_token=ms_per_token,
        p50_ms=p50,
        p99_ms=p99,
        latencies_ms=all_latencies,
    )


def benchmark_layer_breakdown(
    pipeline: TransformersMarlinPipeline,
    seq_len: int = 128,
    iterations: int = 3,
) -> LayerBreakdown | None:
    """Benchmark per-layer timing breakdown."""
    print(
        f"  Profiling layer breakdown ({seq_len} tokens, {iterations} iterations)...")
    model = pipeline.model

    input_ids = torch.randint(0, 1000, (1, seq_len), device="mps")

    # Warmup without hooks
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)
            _mps_sync()

    try:
        with LayerTimingHooks(model) as hooks:
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_ids)
                    _mps_sync()

            breakdown = hooks.get_breakdown()

            # Normalize by iterations
            breakdown.attention_total_ms /= iterations
            breakdown.mlp_total_ms /= iterations
            breakdown.other_total_ms /= iterations
            for lt in breakdown.layer_timings:
                lt.attention_ms /= iterations
                lt.mlp_ms /= iterations
                lt.norm_ms /= iterations
                lt.total_ms /= iterations

            return breakdown
    except Exception as e:
        print(f"    Warning: Could not profile layer breakdown: {e}")
        return None


def load_baseline(baseline_path: Path | None) -> dict[str, Any] | None:
    """Load baseline results for comparison."""
    if baseline_path is None:
        default_baseline = Path(__file__).parent / \
            "results" / "glm_flash_baseline.json"
        if default_baseline.exists():
            baseline_path = default_baseline
        else:
            return None

    try:
        with baseline_path.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load baseline from {baseline_path}: {e}")
        return None


def print_comparison(current: float, baseline: float, label: str, higher_is_better: bool = True) -> str:
    """Print comparison with baseline."""
    if baseline <= 0:
        return ""

    diff = ((current - baseline) / baseline) * 100
    if higher_is_better:
        improvement = diff
    else:
        improvement = -diff

    if improvement > 0:
        return f" ({label}: +{improvement:.1f}%)"
    elif improvement < 0:
        return f" ({label}: {improvement:.1f}%)"
    return ""


def print_results(result: BenchmarkResult, baseline: dict[str, Any] | None = None) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("GLM-4.7-Flash Benchmark Results")
    print("=" * 70)

    # Hardware
    print(f"\nHardware: {result.hardware_info.get('gpu_name', 'Unknown')}")
    if result.hardware_info.get("gpu_cores"):
        print(f"GPU Cores: {result.hardware_info['gpu_cores']}")
    if result.hardware_info.get("memory_bandwidth_gbs"):
        print(
            f"Memory Bandwidth: {result.hardware_info['memory_bandwidth_gbs']:.0f} GB/s")

    # Load time
    baseline_load = baseline.get("load_time_s", 0) if baseline else 0
    comp = print_comparison(result.load_time_s, baseline_load,
                            "vs baseline", higher_is_better=False)
    print(f"\n1. Model Load: {result.load_time_s:.2f}s{comp}")

    # Memory
    print("\n2. Memory Usage:")
    print(f"   Model size: {result.memory.model_size_gb:.2f} GB")
    print(f"   Allocated:  {result.memory.allocated_gb:.2f} GB")
    print(f"   Peak:       {result.memory.peak_gb:.2f} GB")

    # Prefill
    print("\n3. Prefill Throughput:")
    for pm in result.prefill_metrics:
        baseline_prefill = 0
        if baseline and "prefill_metrics" in baseline:
            for bp in baseline["prefill_metrics"]:
                if bp.get("seq_len") == pm.seq_len:
                    baseline_prefill = bp.get("throughput_tok_s", 0)
                    break
        comp = print_comparison(pm.throughput_tok_s,
                                baseline_prefill, "vs baseline")
        print(
            f"   {pm.seq_len:4d} tokens: {pm.throughput_tok_s:7.1f} tok/s "
            f"({pm.time_ms:6.1f} ms ± {pm.time_std_ms:.1f}){comp}"
        )

    # Decode
    print("\n4. Decode Throughput:")
    dm = result.decode_metrics
    baseline_decode = baseline.get("decode_metrics", {}).get(
        "throughput_tok_s", 0) if baseline else 0
    comp = print_comparison(dm.throughput_tok_s,
                            baseline_decode, "vs baseline")
    print(f"   Throughput: {dm.throughput_tok_s:.1f} tok/s{comp}")
    print(f"   Latency:    {dm.ms_per_token:.2f} ms/token")
    print(f"   P50:        {dm.p50_ms:.2f} ms")
    print(f"   P99:        {dm.p99_ms:.2f} ms")

    # Layer breakdown
    if result.layer_breakdown:
        lb = result.layer_breakdown
        print("\n5. Layer Breakdown (per forward pass):")
        print(
            f"   Attention: {lb.attention_total_ms:6.2f} ms ({lb.attention_pct:.1f}%)")
        print(f"   MLP/MoE:   {lb.mlp_total_ms:6.2f} ms ({lb.mlp_pct:.1f}%)")
        print(
            f"   Other:     {lb.other_total_ms:6.2f} ms ({lb.other_pct:.1f}%)")
        print(f"   Total:     {lb.total_ms:6.2f} ms")

        # Show top 5 slowest layers
        sorted_layers = sorted(
            lb.layer_timings, key=lambda x: x.total_ms, reverse=True)[:5]
        if sorted_layers:
            print("\n   Top 5 Slowest Layers:")
            for lt in sorted_layers:
                print(
                    f"     Layer {lt.layer_idx:2d}: {lt.total_ms:.2f} ms "
                    f"(attn: {lt.attention_ms:.2f}, mlp: {lt.mlp_ms:.2f})"
                )

    print("\n" + "=" * 70)


def save_results(result: BenchmarkResult, output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_name": result.model_name,
        "load_time_s": result.load_time_s,
        "memory": {
            "allocated_gb": result.memory.allocated_gb,
            "peak_gb": result.memory.peak_gb,
            "model_size_gb": result.memory.model_size_gb,
        },
        "prefill_metrics": [
            {
                "seq_len": pm.seq_len,
                "time_ms": pm.time_ms,
                "throughput_tok_s": pm.throughput_tok_s,
            }
            for pm in result.prefill_metrics
        ],
        "decode_metrics": {
            "num_tokens": result.decode_metrics.num_tokens,
            "throughput_tok_s": result.decode_metrics.throughput_tok_s,
            "ms_per_token": result.decode_metrics.ms_per_token,
            "p50_ms": result.decode_metrics.p50_ms,
            "p99_ms": result.decode_metrics.p99_ms,
        },
        "hardware_info": result.hardware_info,
    }

    if result.layer_breakdown:
        data["layer_breakdown"] = {
            "attention_total_ms": result.layer_breakdown.attention_total_ms,
            "mlp_total_ms": result.layer_breakdown.mlp_total_ms,
            "other_total_ms": result.layer_breakdown.other_total_ms,
            "attention_pct": result.layer_breakdown.attention_pct,
            "mlp_pct": result.layer_breakdown.mlp_pct,
        }

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive GLM-4.7-Flash benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=str(_ROOT / "models" / "GLM-4.7-Flash-Marlin-MMFP4"),
        help="Path to model directory",
    )
    parser.add_argument(
        "--prefill-lengths",
        type=str,
        default="128,512,2048",
        help="Comma-separated prefill sequence lengths",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=100,
        help="Number of tokens for decode benchmark",
    )
    parser.add_argument(
        "--decode-runs",
        type=int,
        default=3,
        help="Number of decode benchmark runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark mode (reduced iterations)",
    )
    parser.add_argument(
        "--no-layer-breakdown",
        action="store_true",
        help="Skip layer breakdown profiling",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline results JSON for comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baseline",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.prefill_lengths = "64,256"
        args.decode_tokens = 20
        args.decode_runs = 2
        args.warmup = 2
        args.iterations = 3

    prefill_lengths = [int(x) for x in args.prefill_lengths.split(",")]

    # Load baseline for comparison
    baseline = load_baseline(args.baseline)

    # Run benchmarks
    print("=" * 70)
    print("GLM-4.7-Flash Comprehensive Benchmark")
    print("=" * 70)

    # 1. Model loading
    print("\n[1/5] Model Loading...")
    pipeline, load_time, memory = benchmark_model_load(args.model)
    print(
        f"  Loaded in {load_time:.2f}s, memory: {memory.allocated_gb:.2f} GB")

    # Get model info
    model = pipeline.model
    num_layers = 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    print(f"  Model: {num_layers} layers")

    # 2. Prefill benchmark
    print("\n[2/5] Prefill Benchmark...")
    prefill_metrics = benchmark_prefill(
        pipeline, prefill_lengths, warmup=args.warmup, iterations=args.iterations
    )

    # 3. Decode benchmark
    print("\n[3/5] Decode Benchmark...")
    decode_metrics = benchmark_decode(
        pipeline,
        num_tokens=args.decode_tokens,
        warmup=args.warmup,
        num_runs=args.decode_runs,
    )

    # 4. Layer breakdown (optional)
    layer_breakdown = None
    if not args.no_layer_breakdown:
        print("\n[4/5] Layer Breakdown...")
        layer_breakdown = benchmark_layer_breakdown(
            pipeline,
            seq_len=min(256, max(prefill_lengths)),
            iterations=args.iterations,
        )
    else:
        print("\n[4/5] Layer Breakdown... (skipped)")

    # 5. Compile results
    print("\n[5/5] Compiling Results...")
    result = BenchmarkResult(
        model_name=Path(args.model).name,
        load_time_s=load_time,
        memory=memory,
        prefill_metrics=prefill_metrics,
        decode_metrics=decode_metrics,
        layer_breakdown=layer_breakdown,
        hardware_info=_get_hardware_info(),
    )

    # Print results
    print_results(result, baseline)

    # Save results
    if args.output:
        save_results(result, args.output)

    if args.save_baseline:
        baseline_path = Path(__file__).parent / \
            "results" / "glm_flash_baseline.json"
        save_results(result, baseline_path)


if __name__ == "__main__":
    main()
