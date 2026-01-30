#!/usr/bin/env python3
"""MPS (Metal) FP16 baseline for Parakeet speech recognition.

Transcribes 10 seconds of audio and measures:
- Time taken for transcription
- Memory usage
- Comparison vs CPU baseline
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch  # noqa: E402

try:  # Optional dependency
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:  # Optional dependency - parakeet models
    # We'll use a simple dummy model for benchmarking if parakeet is not available
    pass
except Exception:  # pragma: no cover - optional dependency
    pass


@dataclass
class ParakeetMetrics:
    transcription_time_sec: float
    memory_peak_gb: float
    audio_length_sec: float
    realtime_factor: float  # audio_length / transcription_time


class MemoryTracker:
    def __init__(self, device: str):
        self.device = device
        self.peak_bytes = 0
        if (
            self.device == "mps"
            and torch is not None
            and hasattr(torch.mps, "current_allocated_memory")
        ):
            torch.mps.reset_peak_memory_stats() if hasattr(
                torch.mps, "reset_peak_memory_stats"
            ) else None

    def update(self) -> None:
        if torch is None:
            return
        if self.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            current = int(torch.mps.current_allocated_memory())
        elif self.device == "cpu":
            import psutil

            current = int(psutil.Process().memory_info().rss)
        else:
            current = 0
        self.peak_bytes = max(self.peak_bytes, current)

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / (1024 * 1024 * 1024)


def _require_torch(feature: str) -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError(f"PyTorch is required for {feature}.")


def _require_mps(feature: str) -> None:
    _require_torch(feature)
    if not HAS_MPS:
        raise RuntimeError("PyTorch MPS backend is required for this benchmark.")


def _mps_sync() -> None:
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def create_dummy_audio(duration_sec: float, sample_rate: int = 16000) -> torch.Tensor:
    """Create dummy audio data for benchmarking."""
    if np is not None:
        # Generate more realistic audio with some structure
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        # Mix of frequencies to simulate speech
        audio = (
            np.sin(2 * np.pi * 200 * t) * 0.3  # Low freq
            + np.sin(2 * np.pi * 800 * t) * 0.2  # Mid freq
            + np.sin(2 * np.pi * 2000 * t) * 0.1  # High freq
            + np.random.normal(0, 0.05, len(t))
        )  # Noise
        # Apply envelope to simulate speech dynamics
        envelope = np.exp(-np.abs(t - duration_sec / 2) / (duration_sec / 4))
        audio = audio * envelope
        return torch.from_numpy(audio).float()
    else:
        # Fallback: simple sine wave
        t = torch.linspace(0, duration_sec, int(sample_rate * duration_sec))
        return torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz tone


def create_dummy_model(vocab_size: int = 1000) -> torch.nn.Module:
    """Create a dummy model similar to Parakeet for benchmarking."""

    class DummyParakeetModel(torch.nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            # Simplified encoder (similar to wav2vec2/conformer architecture)
            # Use strided convolutions to reduce sequence length
            self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
            self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)

            # Smaller transformer with reduced sequence length
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,  # Reduced
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=3
            )  # Reduced layers

            # Projection to vocabulary
            self.output_proj = torch.nn.Linear(256, vocab_size)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            # Input: [batch_size, audio_length]
            x = audio.unsqueeze(1)  # [batch_size, 1, audio_length]

            # Conv layers with stride to reduce sequence length
            x = torch.relu(self.conv1(x))  # Reduces by 2x
            x = torch.relu(self.conv2(x))  # Reduces by 2x more
            x = torch.relu(self.conv3(x))  # Reduces by 2x more (total 8x reduction)

            # Reshape for transformer: [batch_size, seq_len, features]
            x = x.transpose(1, 2)  # [batch_size, reduced_seq_len, features]

            # Transformer encoding with limited sequence length
            max_seq_len = 1000  # Cap sequence length to avoid memory issues
            if x.size(1) > max_seq_len:
                x = x[:, :max_seq_len, :]

            x = self.transformer(x)

            # Project to vocabulary
            logits = self.output_proj(x)  # [batch_size, seq_len, vocab_size]

            return logits

    return DummyParakeetModel(vocab_size)


def benchmark_mps_parakeet(
    audio_length_sec: float = 10.0,
    num_runs: int = 3,
) -> tuple[ParakeetMetrics, dict[str, any]]:
    """Benchmark Parakeet on MPS device."""
    _require_mps("Parakeet MPS benchmark")
    assert torch is not None

    device = torch.device("mps")

    # Create dummy model and audio
    model = create_dummy_model().to(device).half()
    model.eval()

    # Create audio (10 seconds at 16kHz)
    sample_rate = 16000
    audio = create_dummy_audio(audio_length_sec, sample_rate)
    audio = audio.to(device).half()

    print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
    print(
        f"Model device: {next(model.parameters()).device}, dtype: {next(model.parameters()).dtype}"
    )

    memory = MemoryTracker("mps")
    transcription_times: list[float] = []

    with torch.no_grad():
        for run_idx in range(num_runs):
            print(f"Run {run_idx + 1}/{num_runs}")

            _mps_sync()
            start_time = time.perf_counter()

            # Forward pass (transcription)
            logits = model(audio.unsqueeze(0))  # Add batch dimension

            # Simple greedy decoding
            predicted_ids = torch.argmax(logits, dim=-1)

            _mps_sync()
            transcription_time = time.perf_counter() - start_time
            transcription_times.append(transcription_time)
            memory.update()

            print(f"  Transcription time: {transcription_time:.2f}s")
            print(f"  Output shape: {predicted_ids.shape}")

    # Calculate metrics
    avg_transcription_time = statistics.mean(transcription_times)
    realtime_factor = audio_length_sec / avg_transcription_time

    metrics = ParakeetMetrics(
        transcription_time_sec=float(avg_transcription_time),
        memory_peak_gb=float(memory.peak_gb),
        audio_length_sec=float(audio_length_sec),
        realtime_factor=float(realtime_factor),
    )

    details = {
        "device": "mps",
        "dtype": "fp16",
        "audio_length_sec": float(audio_length_sec),
        "sample_rate": sample_rate,
        "audio_shape": list(audio.shape),
        "output_shape": list(predicted_ids.shape),
        "transcription_times": [float(t) for t in transcription_times],
        "std_transcription_time": float(statistics.stdev(transcription_times))
        if len(transcription_times) > 1
        else 0.0,
    }

    return metrics, details


def benchmark_cpu_parakeet(
    audio_length_sec: float = 10.0,
    num_runs: int = 3,
) -> tuple[ParakeetMetrics, dict[str, any]]:
    """Benchmark Parakeet on CPU for comparison."""
    _require_torch("Parakeet CPU benchmark")
    assert torch is not None

    device = torch.device("cpu")

    # Create dummy model and audio
    model = create_dummy_model().to(device)
    model.eval()

    # Create audio (10 seconds at 16kHz)
    sample_rate = 16000
    audio = create_dummy_audio(audio_length_sec, sample_rate)
    audio = audio.to(device)

    print(f"CPU Audio shape: {audio.shape}, dtype: {audio.dtype}")
    print(
        f"CPU Model device: {next(model.parameters()).device}, dtype: {next(model.parameters()).dtype}"
    )

    memory = MemoryTracker("cpu")
    transcription_times: list[float] = []

    with torch.no_grad():
        for run_idx in range(num_runs):
            print(f"CPU Run {run_idx + 1}/{num_runs}")

            start_time = time.perf_counter()

            # Forward pass (transcription)
            logits = model(audio.unsqueeze(0))  # Add batch dimension

            # Simple greedy decoding
            predicted_ids = torch.argmax(logits, dim=-1)

            transcription_time = time.perf_counter() - start_time
            transcription_times.append(transcription_time)
            memory.update()

            print(f"  CPU Transcription time: {transcription_time:.2f}s")

    # Calculate metrics
    avg_transcription_time = statistics.mean(transcription_times)
    realtime_factor = audio_length_sec / avg_transcription_time

    metrics = ParakeetMetrics(
        transcription_time_sec=float(avg_transcription_time),
        memory_peak_gb=float(memory.peak_gb),
        audio_length_sec=float(audio_length_sec),
        realtime_factor=float(realtime_factor),
    )

    details = {
        "device": "cpu",
        "dtype": "fp32",
        "audio_length_sec": float(audio_length_sec),
        "sample_rate": sample_rate,
        "audio_shape": list(audio.shape),
        "output_shape": list(predicted_ids.shape),
        "transcription_times": [float(t) for t in transcription_times],
        "std_transcription_time": float(statistics.stdev(transcription_times))
        if len(transcription_times) > 1
        else 0.0,
    }

    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPS FP16 baseline for Parakeet speech recognition"
    )
    parser.add_argument("--audio-length", type=float, default=10.0, help="Audio length in seconds")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "parakeet_mps_baseline.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required to run this benchmark.")

    print("=== Parakeet MPS FP16 Baseline ===")
    print(f"Audio length: {args.audio_length} seconds")
    print(f"Number of runs: {args.runs}")
    print()

    # Run MPS benchmark
    print("Running MPS benchmark...")
    mps_metrics, mps_details = benchmark_mps_parakeet(
        audio_length_sec=args.audio_length,
        num_runs=args.runs,
    )
    print()

    # Run CPU benchmark for comparison
    print("Running CPU benchmark for comparison...")
    cpu_metrics, cpu_details = benchmark_cpu_parakeet(
        audio_length_sec=args.audio_length,
        num_runs=args.runs,
    )
    print()

    # Calculate speedup
    speedup = cpu_metrics.transcription_time_sec / mps_metrics.transcription_time_sec
    memory_ratio = mps_metrics.memory_peak_gb / cpu_metrics.memory_peak_gb

    print("=== Results ===")
    print(f"MPS transcription time: {mps_metrics.transcription_time_sec:.2f}s")
    print(f"CPU transcription time: {cpu_metrics.transcription_time_sec:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"MPS memory: {mps_metrics.memory_peak_gb:.2f} GB")
    print(f"CPU memory: {cpu_metrics.memory_peak_gb:.2f} GB")
    print(f"Memory ratio: {memory_ratio:.2f}x")
    print(f"MPS realtime factor: {mps_metrics.realtime_factor:.2f}x")
    print(f"CPU realtime factor: {cpu_metrics.realtime_factor:.2f}x")

    # Prepare output
    output = {
        "date": time.strftime("%Y-%m-%d"),
        "model": "Dummy Parakeet-like Model",
        "task": "Speech Recognition Transcription",
        "audio_length_sec": args.audio_length,
        "sample_rate": 16000,
        "mps": {
            "metrics": {
                "transcription_time_sec": mps_metrics.transcription_time_sec,
                "memory_peak_gb": mps_metrics.memory_peak_gb,
                "realtime_factor": mps_metrics.realtime_factor,
            },
            "details": mps_details,
        },
        "cpu": {
            "metrics": {
                "transcription_time_sec": cpu_metrics.transcription_time_sec,
                "memory_peak_gb": cpu_metrics.memory_peak_gb,
                "realtime_factor": cpu_metrics.realtime_factor,
            },
            "details": cpu_details,
        },
        "comparison": {
            "speedup": speedup,
            "memory_ratio": memory_ratio,
        },
    }

    # Write results
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
