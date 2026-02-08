#!/usr/bin/env python3
"""GLM-4.7-Flash Trellis Benchmark - Reports memory and estimates performance."""

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

# Disable dequant cache
TrellisLinear.enable_cache = False


def main():
    model_path = _ROOT / "models" / "GLM-4.7-Flash-Trellis-3bpw"

    print("=" * 70)
    print("GLM-4.7-Flash Trellis 3bpw Benchmark")
    print("=" * 70)

    # Disk size
    disk_size = sum(f.stat().st_size for f in model_path.rglob("*.safetensors")) / 1e9
    print(f"\nModel disk size: {disk_size:.2f} GB")

    # Load model
    print("\nLoading model...")
    gc.collect()
    torch.mps.empty_cache()

    model = TrellisForCausalLM.from_pretrained(str(model_path), device="mps")
    model.eval()

    torch.mps.synchronize()
    mem_load = torch.mps.current_allocated_memory() / 1e9

    # Count modules
    n_trellis = sum(1 for m in model.modules() if isinstance(m, TrellisLinear))

    print("\n--- Model Statistics ---")
    print(f"TrellisLinear modules: {n_trellis}")
    print(f"Memory after load: {mem_load:.2f} GB")
    print(f"Memory efficiency: {disk_size / mem_load:.1%}")

    # Single token decode time (representative of actual use)
    print("\n--- Decode Benchmark (single token) ---")
    x = torch.tensor([[1]], device="mps")

    # Warmup
    with torch.no_grad():
        _ = model(x)

    # Measure decode
    torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    torch.mps.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000

    print(f"Single token decode: {decode_ms:.0f} ms")
    print(f"Decode throughput: {1000 / decode_ms:.2f} tok/s")

    # Memory after decode
    mem_decode = torch.mps.current_allocated_memory() / 1e9
    print(f"Memory after decode: {mem_decode:.2f} GB")

    print("\n--- Summary ---")
    print("Model: GLM-4.7-Flash Trellis 3bpw (47 layers, 64 experts)")
    print("Quantization: 3-bit Trellis (EXL3)")
    print(f"Disk size: {disk_size:.2f} GB")
    print(f"GPU memory: {mem_load:.2f} GB")
    print(f"Compression ratio: {(47 * 64 * 3 * 2048 * 1536 * 2) / 1e9 / disk_size:.1f}x vs FP16")
    print(f"Decode latency: {decode_ms:.0f} ms/tok")
    print("")
    print("NOTE: Current implementation dequantizes weights on each forward pass.")
    print("A fused GEMM kernel would dramatically improve throughput.")
    print("=" * 70)


if __name__ == "__main__":
    main()
