#!/usr/bin/env python3
"""Benchmark memory usage of quantized vs FP16 model loading."""

import gc
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def get_mps_memory() -> float:
    """Get current MPS allocated memory in GB."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def benchmark_quantized_loading(model_path: str) -> dict:
    """Benchmark quantized model memory usage."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    baseline = get_mps_memory()
    print(f"Baseline memory: {baseline:.2f} GB")

    from metal_marlin.trellis.lm import TrellisForCausalLM

    print("Loading quantized model...")
    model = TrellisForCausalLM.from_pretrained(model_path, device="mps")

    after_load = get_mps_memory()
    print(f"After load: {after_load:.2f} GB (+{after_load - baseline:.2f} GB)")

    # Run forward pass
    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4, 5] * 100], device="mps")
        _ = model(input_ids)

    after_forward = get_mps_memory()
    print(f"After forward: {after_forward:.2f} GB (+{after_forward - after_load:.2f} GB)")

    # Expected: disk_size ≈ load_size (no 5x inflation)
    disk_size_gb = sum(f.stat().st_size for f in Path(model_path).rglob("*.safetensors")) / 1e9
    print(f"\nDisk size: {disk_size_gb:.2f} GB")
    print(f"Memory efficiency: {disk_size_gb / (after_load - baseline):.1%}")

    return {
        "disk_size_gb": disk_size_gb,
        "load_memory_gb": after_load - baseline,
        "forward_memory_gb": after_forward - baseline,
        "efficiency": disk_size_gb / max(after_load - baseline, 0.1),
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/GLM-4.7-Flash-Marlin-MMFP4"
    results = benchmark_quantized_loading(model_path)
    print(f"\nResults: {results}")

    # Assert efficiency is reasonable (>80%)
    assert results["efficiency"] > 0.8, f"Memory efficiency too low: {results['efficiency']:.1%}"
