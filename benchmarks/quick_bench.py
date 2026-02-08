#!/usr/bin/env python3
"""Quick benchmark for GLM-4.7-Flash Trellis model."""

import gc
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.trellis.linear import TrellisLinear
from metal_marlin.trellis.lm import TrellisForCausalLM

import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

# Disable dequant cache for memory efficiency
TrellisLinear.enable_cache = False


def main():
    model_path = _ROOT / "models" / "GLM-4.7-Flash-Trellis-3bpw"

    print("=" * 60)
    print("GLM-4.7-Flash Trellis 3bpw Quick Benchmark")
    print("=" * 60)

    # Disk size
    disk_size = sum(f.stat().st_size for f in model_path.rglob("*.safetensors")) / 1e9
    print(f"Model disk size: {disk_size:.2f} GB")

    # Load model
    print("\nLoading model...")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    model = TrellisForCausalLM.from_pretrained(str(model_path), device="mps")
    model.eval()

    torch.mps.synchronize()
    mem_load = torch.mps.current_allocated_memory() / 1e9
    print(f"Memory after load: {mem_load:.2f} GB")
    print(f"Memory efficiency: {disk_size / mem_load:.1%}")

    # Prefill benchmark
    print("\n--- Prefill Benchmark ---")
    for seq_len in [32, 64, 128]:
        x = torch.randint(0, 1000, (1, seq_len), device="mps")

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Benchmark
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
        t1 = time.perf_counter()

        print(
            f"  {seq_len:4d} tokens: {(t1 - t0) * 1000:7.1f} ms ({seq_len / (t1 - t0):6.1f} tok/s)"
        )

    # Decode benchmark (single token)
    print("\n--- Decode Benchmark ---")
    x_single = torch.tensor([[1]], device="mps")

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x_single)

    # Benchmark
    times = []
    for _ in range(10):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_single)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    print(f"  Single token: {avg_ms:.1f} ms ({1000 / avg_ms:.1f} tok/s)")

    # Memory after forward passes
    mem_final = torch.mps.current_allocated_memory() / 1e9
    print(f"\nFinal memory: {mem_final:.2f} GB")

    print("\n" + "=" * 60)
    print("Summary:")
    print("  Model: GLM-4.7-Flash Trellis 3bpw")
    print(f"  Disk: {disk_size:.2f} GB")
    print(f"  Memory: {mem_load:.2f} GB")
    print(f"  Efficiency: {disk_size / mem_load:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
