#!/usr/bin/env python3
"""Benchmark Parakeet encoder inference performance.

Compares:
- PyTorch MPS baseline
- Metal Marlin FP4
- Metal Marlin INT8

Focus on encoder-only since it's 95%+ of compute.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from metal_marlin.asr import ConformerConfig, ParakeetTDT
from metal_marlin.asr.tdt_config import TDTConfig


def create_model() -> ParakeetTDT:
    """Create Parakeet 0.6B v3 model (108M params)."""
    conformer_cfg = ConformerConfig(
        num_layers=17,
        hidden_size=512,
        num_attention_heads=8,
        ffn_intermediate_size=2048,
        conv_kernel_size=31,
        dropout=0.0,
        n_mels=80,
        sample_rate=16000,
        subsampling_factor=4,
    )
    tdt_cfg = TDTConfig(
        vocab_size=1024,
        predictor_hidden_size=320,
        predictor_num_layers=2,
        encoder_hidden_size=512,
        joint_hidden_size=512,
        max_duration=100,
    )
    return ParakeetTDT(conformer_cfg, tdt_cfg)


def benchmark_encoder(
    model: torch.nn.Module,
    device: str,
    audio_sec: float = 5.0,
    num_runs: int = 10,
    warmup: int = 3,
) -> dict:
    """Benchmark encoder performance.

    Args:
        model: ParakeetTDT model
        device: Device to benchmark on ('mps', 'cpu')
        audio_sec: Audio duration in seconds
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs

    Returns:
        Dictionary with timing results
    """
    model = model.to(device)
    model.eval()

    # Create dummy mel spectrogram
    # 16kHz audio, 10ms hop = 100 frames/sec
    n_frames = int(audio_sec * 100)
    mel = torch.randn(1, n_frames, 80, device=device)
    lengths = torch.tensor([n_frames], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.encode(mel, lengths)
            if device == "mps":
                torch.mps.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            out, out_lens = model.encode(mel, lengths)
            if device == "mps":
                torch.mps.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

    avg_ms = statistics.mean(latencies)
    rtf = audio_sec / (avg_ms / 1000)

    return {
        "device": device,
        "audio_sec": audio_sec,
        "avg_latency_ms": avg_ms,
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)]
        if len(latencies) >= 20
        else max(latencies),
        "realtime_factor": rtf,
        "output_shape": list(out.shape),
    }


def main():
    print("=== Parakeet Encoder Inference Benchmark ===\n")

    model = create_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}\n")

    results = []

    # Test different audio lengths
    for audio_sec in [1.0, 5.0, 10.0]:
        print(f"Audio length: {audio_sec}s")

        # PyTorch MPS
        if torch.backends.mps.is_available():
            r = benchmark_encoder(model, "mps", audio_sec)
            print(f"  MPS: {r['avg_latency_ms']:.1f}ms ({r['realtime_factor']:.0f}x realtime)")
            results.append({**r, "backend": "mps"})

        # CPU baseline
        r = benchmark_encoder(model, "cpu", audio_sec)
        print(f"  CPU: {r['avg_latency_ms']:.1f}ms ({r['realtime_factor']:.0f}x realtime)")
        results.append({**r, "backend": "cpu"})

        print()

    print("Done!")


if __name__ == "__main__":
    main()
