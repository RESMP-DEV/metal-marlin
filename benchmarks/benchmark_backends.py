#!/usr/bin/env python3
"""Comprehensive backend comparison benchmark.

Compares all execution backends for Parakeet encoder:
- PyTorch MPS (baseline)
- Metal INT8 (custom kernel)
- Metal FP4 (custom kernel)
- CoreML ANE
- Hybrid Metal+ANE
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from metal_marlin.asr import ConformerConfig, ParakeetTDT
from metal_marlin.asr.tdt_config import TDTConfig


@dataclass
class BenchmarkResult:
    """Result from a single benchmark configuration."""

    backend: str
    audio_sec: float
    avg_latency_ms: float
    p50_ms: float
    p95_ms: float
    realtime_factor: float
    memory_mb: float


def create_model() -> ParakeetTDT:
    """Create Parakeet 0.6B config model."""
    conformer_cfg = ConformerConfig(
        num_layers=17,
        hidden_size=512,
        num_attention_heads=8,
        ffn_intermediate_size=2048,
        conv_kernel_size=31,
        dropout=0.0,
        n_mels=80,
        subsampling_factor=4,
    )
    tdt_cfg = TDTConfig(
        vocab_size=1024,
        predictor_hidden_size=320,
        encoder_hidden_size=512,
    )
    return ParakeetTDT(conformer_cfg, tdt_cfg)


def benchmark_mps_baseline(
    model: ParakeetTDT, audio_sec: float, num_runs: int = 10
) -> BenchmarkResult:
    """Benchmark PyTorch MPS baseline."""
    model = model.to("mps")
    model.eval()

    n_frames = int(audio_sec * 100)
    mel = torch.randn(1, n_frames, 80, device="mps")
    lengths = torch.tensor([n_frames], device="mps")

    # Warmup
    for _ in range(3):
        _ = model.encode(mel, lengths)
        torch.mps.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        torch.mps.synchronize()
        start = time.perf_counter()
        _ = model.encode(mel, lengths)
        torch.mps.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(latencies)
    return BenchmarkResult(
        backend="mps_baseline",
        audio_sec=audio_sec,
        avg_latency_ms=avg,
        p50_ms=statistics.median(latencies),
        p95_ms=sorted(latencies)[int(len(latencies) * 0.95)]
        if len(latencies) >= 20
        else max(latencies),
        realtime_factor=audio_sec / (avg / 1000),
        memory_mb=torch.mps.current_allocated_memory() / 1024 / 1024,
    )


def benchmark_metal_int8(
    model: ParakeetTDT, audio_sec: float, num_runs: int = 10
) -> BenchmarkResult:
    """Benchmark Metal INT8 custom kernel."""
    try:
        from metal_marlin.asr.replace_layers_metal import replace_parakeet_encoder_layers

        model = replace_parakeet_encoder_layers(model, quant_type="int8")
    except Exception as e:
        print(f"Metal INT8 setup failed: {e}")
        return BenchmarkResult("metal_int8", audio_sec, -1, -1, -1, -1, 0)

    model = model.to("mps")
    model.eval()

    n_frames = int(audio_sec * 100)
    mel = torch.randn(1, n_frames, 80, device="mps")
    lengths = torch.tensor([n_frames], device="mps")

    # Warmup
    for _ in range(3):
        try:
            _ = model.encode(mel, lengths)
            torch.mps.synchronize()
        except Exception as e:
            print(f"Metal INT8 warmup failed: {e}")
            return BenchmarkResult("metal_int8", audio_sec, -1, -1, -1, -1, 0)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        torch.mps.synchronize()
        start = time.perf_counter()
        _ = model.encode(mel, lengths)
        torch.mps.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(latencies)
    return BenchmarkResult(
        backend="metal_int8",
        audio_sec=audio_sec,
        avg_latency_ms=avg,
        p50_ms=statistics.median(latencies),
        p95_ms=sorted(latencies)[int(len(latencies) * 0.95)]
        if len(latencies) >= 20
        else max(latencies),
        realtime_factor=audio_sec / (avg / 1000),
        memory_mb=torch.mps.current_allocated_memory() / 1024 / 1024,
    )


def benchmark_coreml_ane(
    model: ParakeetTDT, audio_sec: float, num_runs: int = 10
) -> BenchmarkResult:
    """Benchmark CoreML ANE execution with INT8 quantization."""
    try:
        from metal_marlin.ane import HAS_COREMLTOOLS, ANEEncoder, export_encoder_to_coreml

        if not HAS_COREMLTOOLS:
            raise ImportError("coremltools not available")

        # Export with INT8 quantization for ANE
        mlmodel_path = Path("/tmp/parakeet_encoder_int8.mlpackage")
        if not mlmodel_path.exists():
            print("Exporting encoder to CoreML with INT8 quantization...")
            export_encoder_to_coreml(
                model.encoder,
                mlmodel_path,
                quantize_weights="int8",  # INT8 for ANE
            )

        ane_encoder = ANEEncoder(mlmodel_path)
    except Exception as e:
        print(f"CoreML ANE setup failed: {e}")
        return BenchmarkResult("coreml_ane", audio_sec, -1, -1, -1, -1, 0)

    n_frames = int(audio_sec * 100)
    mel = torch.randn(1, n_frames, 80)

    # Warmup
    for _ in range(3):
        try:
            _ = ane_encoder(mel)
        except Exception as e:
            print(f"CoreML ANE warmup failed: {e}")
            return BenchmarkResult("coreml_ane", audio_sec, -1, -1, -1, -1, 0)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = ane_encoder(mel)
        latencies.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(latencies)
    return BenchmarkResult(
        backend="coreml_ane",
        audio_sec=audio_sec,
        avg_latency_ms=avg,
        p50_ms=statistics.median(latencies),
        p95_ms=sorted(latencies)[int(len(latencies) * 0.95)]
        if len(latencies) >= 20
        else max(latencies),
        realtime_factor=audio_sec / (avg / 1000),
        memory_mb=0,  # Can't easily measure CoreML memory
    )


def main():
    parser = argparse.ArgumentParser(description="Backend comparison benchmark")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mps_baseline", "metal_int8", "coreml_ane"],
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--audio-lengths",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 10.0],
        help="Audio lengths in seconds",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/backend_comparison.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    print("=== Backend Comparison Benchmark ===\n")

    results = []

    for audio_sec in args.audio_lengths:
        print(f"\n--- Audio length: {audio_sec}s ---")

        for backend in args.backends:
            model = create_model()

            if backend == "mps_baseline":
                r = benchmark_mps_baseline(model, audio_sec, args.runs)
            elif backend == "metal_int8":
                r = benchmark_metal_int8(model, audio_sec, args.runs)
            elif backend == "coreml_ane":
                r = benchmark_coreml_ane(model, audio_sec, args.runs)
            else:
                print(f"Unknown backend: {backend}")
                continue

            results.append(asdict(r))

            if r.avg_latency_ms > 0:
                print(f"  {backend}: {r.avg_latency_ms:.1f}ms ({r.realtime_factor:.0f}x realtime)")
            else:
                print(f"  {backend}: FAILED")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
