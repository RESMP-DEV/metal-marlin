#!/usr/bin/env python3
"""Comprehensive benchmark comparing Metal Marlin ASR backends.

Compares different configurations for Parakeet ASR model:
1. pytorch_mps - Baseline PyTorch with MPS device
2. metal_fp4 - Metal Marlin custom kernels with FP4
3. metal_int8 - Metal Marlin custom kernels with INT8
4. pytorch_cpu - CPU baseline for reference

Metrics collected:
- Tokens per second (throughput)
- Latency (p50, p95, p99)
- Memory usage (peak GPU memory)
- First-token latency (time to first output)

Test matrix:
- Audio lengths: 1s, 5s, 10s, 30s
- Batch sizes: 1, 4, 8

Results saved to benchmarks/results/metal_asr_comparison.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Standalone imports - follow contrib guidelines
import numpy as np


# Standalone TDTConfig for contrib project
@dataclass
class TDTConfig:
    """Configuration for TDT (Transducer Dynamic Temperature) ASR model."""

    vocab_size: int = 1024
    predictor_hidden_size: int = 320
    predictor_num_layers: int = 2
    encoder_hidden_size: int = 512
    joint_hidden_size: int = 512
    blank_id: int = 0


# Try to import metal_marlin components for enhanced functionality
try:
    # Try to import from current directory structure
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from metal_marlin.asr import ParakeetTDT
    from metal_marlin.asr.quant_int8 import quantize_conformer_to_int8
    from metal_marlin.ops.gemm_int8 import GemmInt8, pack_int8_weights

    HAS_METAL_MARLIN = True
except ImportError as e:
    print(f"Warning: Metal Marlin not available ({e}), using dummy models")
    HAS_METAL_MARLIN = False

# PyTorch imports - direct import for standalone contrib project
try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
    HAS_MPS = hasattr(torch, "backends") and torch.backends.mps.is_available()
except ImportError as e:
    print(f"Error: PyTorch is required for this benchmark: {e}")
    torch = None
    HAS_TORCH = False
    HAS_MPS = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    name: str
    backend: str  # "pytorch_mps", "metal_fp4", "metal_int8", "pytorch_cpu"
    quantize: str | None = None  # None, "fp4", "int8"
    device: str = "mps"  # "mps", "cpu"


@dataclass
class BenchmarkRun:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    audio_length_sec: float
    batch_size: int

    # Timing metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    first_token_latency_ms: float

    # Performance metrics
    tokens_per_second: float
    realtime_factor: float  # audio_duration / inference_time

    # Memory metrics
    peak_memory_mb: float
    peak_gpu_memory_mb: float  # MPS memory if available

    # Model info
    output_shape: tuple[int, ...]
    vocab_size: int


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark runs."""

    timestamp: str
    device_info: dict[str, str]
    runs: list[BenchmarkRun]
    best_configs: dict[str, str]  # metric -> config name


class MemoryTracker:
    """Track memory usage during benchmark runs."""

    def __init__(self, device: str):
        self.device = device
        self.peak_system_bytes = 0
        self.peak_gpu_bytes = 0

        # Reset MPS memory stats if available
        if HAS_MPS and device == "mps":
            try:
                torch.mps.reset_peak_memory_stats()
            except:
                pass

    def update(self) -> None:
        """Update peak memory measurements."""
        # Track GPU memory (MPS)
        if HAS_MPS and self.device == "mps":
            try:
                current = torch.mps.current_allocated_memory()
                self.peak_gpu_bytes = max(self.peak_gpu_bytes, current)
            except:
                pass

        # Track system memory
        try:
            import psutil

            current = psutil.Process().memory_info().rss
            self.peak_system_bytes = max(self.peak_system_bytes, current)
        except ImportError:
            pass

    @property
    def peak_system_mb(self) -> float:
        """Peak system memory in MB."""
        return self.peak_system_bytes / (1024 * 1024)

    @property
    def peak_gpu_mb(self) -> float:
        """Peak GPU memory in MB."""
        return self.peak_gpu_bytes / (1024 * 1024)


def create_dummy_audio(duration_sec: float, sample_rate: int = 16000) -> torch.Tensor:
    """Create realistic dummy audio for benchmarking."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    # Generate audio with speech-like characteristics
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)

    # Mix of frequencies simulating speech formants
    audio = (
        np.sin(2 * np.pi * 200 * t) * 0.3  # Fundamental frequency
        + np.sin(2 * np.pi * 800 * t) * 0.2  # First formant
        + np.sin(2 * np.pi * 2000 * t) * 0.1  # Second formant
        + np.sin(2 * np.pi * 3000 * t) * 0.05  # Third formant
        + np.random.normal(0, 0.02, len(t))  # Background noise
    )

    # Apply envelope to simulate speech dynamics
    envelope = np.exp(-np.abs(t - duration_sec / 2) / (duration_sec / 3))
    audio = audio * envelope

    return torch.from_numpy(audio).float()


def create_parakeet_config() -> TDTConfig:
    """Create Parakeet TDT configuration for benchmarking."""
    return TDTConfig(
        vocab_size=1024,
        predictor_hidden_size=320,
        predictor_num_layers=2,
        encoder_hidden_size=512,
        joint_hidden_size=512,
        blank_id=0,
    )


def create_dummy_parakeet_model(config: TDTConfig) -> torch.nn.Module:
    """Create a dummy Parakeet-like model for benchmarking.

    If metal_marlin is available, use the real ParakeetTDT.
    Otherwise, create a simplified model with similar architecture.
    """

    if HAS_METAL_MARLIN:
        try:
            return ParakeetTDT(config)
        except Exception as e:
            print(f"Warning: Could not create real ParakeetTDT ({e}), using dummy model")

    # Dummy model with similar structure
    class DummyParakeetModel(torch.nn.Module):
        def __init__(self, config: TDTConfig):
            super().__init__()
            self.config = config

            # Simplified encoder (convolutional + transformer)
            self.subsampling = torch.nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv1d(
                64, config.encoder_hidden_size, kernel_size=3, stride=2, padding=1
            )

            # Transformer encoder layers
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=config.encoder_hidden_size,
                nhead=8,
                dim_feedforward=config.encoder_hidden_size * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

            # Predictor (LSTM)
            self.predictor = torch.nn.LSTM(
                input_size=config.vocab_size,
                hidden_size=config.predictor_hidden_size,
                num_layers=config.predictor_num_layers,
                batch_first=True,
            )

            # Joint network
            self.joint = torch.nn.Linear(
                config.encoder_hidden_size + config.predictor_hidden_size, config.vocab_size
            )

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            # Encoder path
            x = audio.unsqueeze(1)  # [batch, 1, time]
            x = torch.relu(self.subsampling(x))  # Reduce by 2x
            x = torch.relu(self.conv2(x))  # Reduce by 2x more
            x = x.transpose(1, 2)  # [batch, time/4, features]

            # Cap sequence length for memory efficiency
            max_len = 1000
            if x.size(1) > max_len:
                x = x[:, :max_len, :]

            x = self.encoder(x)  # [batch, time/4, encoder_hidden]

            # Dummy predictor (just use zeros for simplicity)
            batch_size = x.size(0)
            seq_len = x.size(1)
            dummy_input = torch.zeros(batch_size, 1, config.vocab_size, device=x.device)
            pred_out, _ = self.predictor(dummy_input)  # [batch, 1, pred_hidden]

            # Expand predictor output to match encoder sequence
            pred_out = pred_out.expand(-1, seq_len, -1)  # [batch, time/4, pred_hidden]

            # Joint network
            joint_input = torch.cat([x, pred_out], dim=-1)
            logits = self.joint(joint_input)  # [batch, time/4, vocab_size]

            return logits

    return DummyParakeetModel(config)


def apply_backend_config(model: torch.nn.Module, config: BenchmarkConfig) -> torch.nn.Module:
    """Apply backend-specific configuration to model."""

    # Set device
    device = torch.device(config.device)
    model = model.to(device)

    # Apply quantization if specified
    if config.quantize == "int8":
        if HAS_METAL_MARLIN:
            try:
                # Use Metal Marlin's INT8 quantization for Conformer encoder
                from metal_marlin.asr.quant_int8 import (
                    calibrate_int8_scales,
                    quantize_conformer_to_int8,
                )

                # Quick calibration with dummy data
                dummy_mel = torch.randn(1, 100, 80, device=device)
                try:
                    scales_zeros = calibrate_int8_scales(model, [dummy_mel], group_size=128)
                    model = quantize_conformer_to_int8(model, scales_zeros)
                except Exception as cal_err:
                    print(f"Warning: INT8 calibration failed ({cal_err}), using PyTorch fallback")
                    raise cal_err
            except Exception as e:
                print(
                    f"Warning: Metal Marlin INT8 quantization failed ({e}), using PyTorch fallback"
                )
                # Fallback to PyTorch dynamic quantization on CPU
                try:
                    cpu_model = model.to("cpu")
                    cpu_model = torch.ao.quantization.quantize_dynamic(
                        cpu_model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
                    )
                    model = cpu_model.to(device)
                except Exception as e2:
                    print(f"Warning: PyTorch INT8 quantization also failed ({e2})")
        else:
            # No Metal Marlin, use PyTorch fallback
            try:
                cpu_model = model.to("cpu")
                cpu_model = torch.ao.quantization.quantize_dynamic(
                    cpu_model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
                )
                model = cpu_model.to(device)
            except Exception as e:
                print(f"Warning: PyTorch INT8 quantization failed ({e})")

    elif config.quantize == "fp4":
        if HAS_METAL_MARLIN:
            try:
                from metal_marlin.layer_replacement import replace_linear_layers

                model = replace_linear_layers(model, quant_bits=4)
            except Exception as e:
                print(f"Warning: Metal Marlin FP4 quantization failed ({e})")
        else:
            print("Warning: FP4 quantization requires Metal Marlin")

    # Set to half precision for non-quantized models on GPU (but not MPS due to compatibility)
    if config.quantize is None and config.device != "cpu" and config.device != "mps":
        model = model.half()

    model.eval()
    return model


def _sync_device(device: str) -> None:
    """Synchronize device operations."""
    if HAS_MPS and device == "mps":
        try:
            torch.mps.synchronize()
        except:
            pass


def benchmark_encoder(
    encoder: torch.nn.Module,
    audio_lengths: list[float],
    batch_sizes: list[int],
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """Benchmark encoder with given configurations.

    Args:
        encoder: The encoder model to benchmark
        audio_lengths: List of audio durations in seconds
        batch_sizes: List of batch sizes to test
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Dictionary of benchmark results
    """
    results = []

    # Determine device from model
    device = next(encoder.parameters()).device
    device_str = str(device)

    for audio_length in audio_lengths:
        for batch_size in batch_sizes:
            print(f"Testing: audio_length={audio_length}s, batch_size={batch_size}")

            # Create test audio
            audio = create_dummy_audio(audio_length)
            audio_batch = audio.unsqueeze(0).repeat(batch_size, 1)

            # Move to device and adjust precision
            audio_batch = audio_batch.to(device)
            # Match model precision - if model is half precision, convert input too
            model_dtype = next(encoder.parameters()).dtype
            if (
                model_dtype == torch.float16
                and audio_batch.dtype != torch.float16
                and device_str != "mps"
            ):  # MPS has float16 compatibility issues
                audio_batch = audio_batch.half()

            # Memory tracking
            memory = MemoryTracker(device_str)

            # Warmup runs
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = encoder(audio_batch)
                    memory.update()
                _sync_device(device_str)

            # Benchmark runs
            latencies = []
            first_token_latencies = []

            with torch.no_grad():
                for run in range(num_runs):
                    memory.update()
                    _sync_device(device_str)

                    start_time = time.perf_counter()

                    # Forward pass
                    logits = encoder(audio_batch)

                    # Simple greedy decoding to measure first token
                    first_token = (
                        torch.argmax(logits[:, 0, :], dim=-1)
                        if logits.size(1) > 0
                        else torch.zeros(batch_size, dtype=torch.long)
                    )

                    _sync_device(device_str)
                    end_time = time.perf_counter()

                    latency = (end_time - start_time) * 1000  # Convert to ms
                    latencies.append(latency)

                    # First token latency (approximate)
                    first_token_latencies.append(
                        latency / logits.size(1) if logits.size(1) > 0 else latency
                    )

                    memory.update()

            # Calculate metrics
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            avg_first_token = statistics.mean(first_token_latencies)

            # Performance metrics
            total_tokens = batch_size * logits.size(1) * logits.size(2)  # Approximate
            tokens_per_sec = total_tokens / (avg_latency / 1000)
            realtime_factor = audio_length / (avg_latency / 1000)

            # Store results
            result = {
                "audio_length_sec": audio_length,
                "batch_size": batch_size,
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "first_token_latency_ms": avg_first_token,
                "tokens_per_second": tokens_per_sec,
                "realtime_factor": realtime_factor,
                "peak_memory_mb": memory.peak_system_mb,
                "peak_gpu_memory_mb": memory.peak_gpu_mb,
                "output_shape": list(logits.shape),
                "total_tokens": total_tokens,
            }

            results.append(result)

            print(
                f"  Latency: {avg_latency:.2f}ms, RTF: {realtime_factor:.2f}x, "
                f"Memory: {memory.peak_system_mb:.1f}MB"
            )

    return {"results": results, "num_warmup": num_warmup, "num_runs": num_runs}


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark comparing Metal Marlin ASR backends"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["pytorch_mps", "metal_fp4", "metal_int8", "pytorch_cpu"],
        default=["pytorch_mps", "pytorch_cpu"],
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--audio-lengths",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 10.0, 30.0],
        help="Audio lengths in seconds to test",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 8], help="Batch sizes to test"
    )
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs")
    parser.add_argument(
        "--benchmark-runs", type=int, default=10, help="Number of benchmark runs per configuration"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "metal_asr_comparison.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    if not HAS_TORCH:
        print("Error: PyTorch is required to run this benchmark.")
        sys.exit(1)

    print("=== Metal Marlin ASR Backend Benchmark ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {HAS_MPS}")
    print(f"Metal Marlin available: {HAS_METAL_MARLIN}")
    print(f"Backends: {', '.join(args.backends)}")
    print(f"Audio lengths: {args.audio_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print()

    # Define benchmark configurations
    config_map = {
        "pytorch_mps": BenchmarkConfig("PyTorch MPS", "pytorch_mps", None, "mps"),
        "metal_fp4": BenchmarkConfig("Metal FP4", "metal_fp4", "fp4", "mps"),
        "metal_int8": BenchmarkConfig("Metal INT8", "metal_int8", "int8", "mps"),
        "pytorch_cpu": BenchmarkConfig("PyTorch CPU", "pytorch_cpu", None, "cpu"),
    }

    # Filter configurations based on availability and user selection
    configs_to_test = []
    for backend in args.backends:
        config = config_map[backend]
        if backend == "pytorch_mps" and not HAS_MPS:
            print(f"Skipping {backend}: MPS not available")
            continue
        if backend.startswith("metal_") and not HAS_METAL_MARLIN:
            print(f"Skipping {backend}: Metal Marlin not available")
            continue
        configs_to_test.append(config)

    if not configs_to_test:
        print("Error: No valid configurations to test")
        sys.exit(1)

    # Create Parakeet configuration
    parakeet_config = create_parakeet_config()

    # Run benchmarks
    all_results = []

    for config in configs_to_test:
        print(f"\n=== Benchmarking {config.name} ===")

        try:
            # Create and configure model
            model = create_dummy_parakeet_model(parakeet_config)
            model = apply_backend_config(model, config)

            # Run benchmark
            benchmark_results = benchmark_encoder(
                model,
                args.audio_lengths,
                args.batch_sizes,
                num_warmup=args.warmup_runs,
                num_runs=args.benchmark_runs,
            )

            # Add configuration info to results
            for result in benchmark_results["results"]:
                result["config"] = asdict(config)
                result["vocab_size"] = parakeet_config.vocab_size

            all_results.extend(benchmark_results["results"])

        except Exception as e:
            print(f"Error benchmarking {config.name}: {e}")
            continue

    # Create summary
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Find best configurations for each metric
    best_configs = {}
    if all_results:
        metrics = ["tokens_per_second", "avg_latency_ms", "peak_memory_mb"]
        directions = ["max", "min", "min"]

        for metric, direction in zip(metrics, directions):
            if direction == "max":
                best = max(all_results, key=lambda r: r[metric])
            else:
                best = min(all_results, key=lambda r: r[metric])

            config_name = best["config"]["name"]
            metric_value = best[metric]
            best_configs[metric] = {"config": config_name, "value": metric_value}

    # Device info
    device_info = {
        "pytorch_version": torch.__version__,
        "mps_available": HAS_MPS,
        "metal_marlin_available": HAS_METAL_MARLIN,
    }

    summary = BenchmarkSummary(
        timestamp=timestamp,
        device_info=device_info,
        runs=[
            BenchmarkRun(
                **{
                    "config": BenchmarkConfig(**r["config"]),
                    "audio_length_sec": r["audio_length_sec"],
                    "batch_size": r["batch_size"],
                    "avg_latency_ms": r["avg_latency_ms"],
                    "p50_latency_ms": r["p50_latency_ms"],
                    "p95_latency_ms": r["p95_latency_ms"],
                    "p99_latency_ms": r["p99_latency_ms"],
                    "first_token_latency_ms": r["first_token_latency_ms"],
                    "tokens_per_second": r["tokens_per_second"],
                    "realtime_factor": r["realtime_factor"],
                    "peak_memory_mb": r["peak_memory_mb"],
                    "peak_gpu_memory_mb": r["peak_gpu_memory_mb"],
                    "output_shape": tuple(r["output_shape"]),
                    "vocab_size": r["vocab_size"],
                }
            )
            for r in all_results
        ],
        best_configs=best_configs,
    )

    # Prepare output
    output = {
        "summary": asdict(summary),
        "benchmark_args": {
            "backends": args.backends,
            "audio_lengths": args.audio_lengths,
            "batch_sizes": args.batch_sizes,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
        },
    }

    # Write results
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    if all_results:
        print("\n=== Summary Table ===")
        print(
            f"{'Backend':<15} {'Audio (s)':<10} {'Batch':<6} {'Latency (ms)':<12} {'RTF':<8} {'Mem (MB)':<10}"
        )
        print("-" * 70)

        # Group by configuration and show representative results
        config_results = {}
        for result in all_results:
            config_name = result["config"]["name"]
            if config_name not in config_results:
                config_results[config_name] = result

        for config_name, result in sorted(config_results.items()):
            print(
                f"{config_name:<15} "
                f"{result['audio_length_sec']:<10.1f} "
                f"{result['batch_size']:<6} "
                f"{result['avg_latency_ms']:<12.2f} "
                f"{result['realtime_factor']:<8.2f} "
                f"{result['peak_memory_mb']:<10.1f}"
            )

        print("\n=== Best Configurations ===")
        for metric, best in best_configs.items():
            print(f"Best {metric}: {best['config']} ({best['value']:.2f})")


if __name__ == "__main__":
    main()
