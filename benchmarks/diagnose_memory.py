#!/usr/bin/env python3
"""Diagnose model memory usage."""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.trellis_linear import TrellisLinear
from metal_marlin.trellis_lm import TrellisForCausalLM


def main():
    print("Loading model...")
    model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-Trellis-3bpw/", device="mps")
    print(f"Config: {model.config.num_hidden_layers} layers, {model.config.num_experts} experts")

    # Count TrellisLinear modules and their sizes
    total_compressed = 0
    total_dequant = 0
    n_linear = 0
    for name, m in model.named_modules():
        if isinstance(m, TrellisLinear):
            n_linear += 1
            # Compressed size (indices + scales + su + sv)
            compressed = m.indices.element_size() * m.indices.numel()
            compressed += m.scales.element_size() * m.scales.numel()
            compressed += m.su.element_size() * m.su.numel()
            compressed += m.sv.element_size() * m.sv.numel()
            total_compressed += compressed
            # Dequantized size (FP16)
            total_dequant += m.in_features * m.out_features * 2

    print(f"TrellisLinear modules: {n_linear}")
    print(f"Compressed size: {total_compressed / 1e9:.2f} GB")
    print(f"Dequantized size (if cached): {total_dequant / 1e9:.2f} GB")

    # Check MPS memory
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        print(
            f"MPS allocated (before forward): {torch.mps.current_allocated_memory() / 1e9:.2f} GB"
        )

    # Run a forward pass to trigger dequantization
    print("\nRunning forward pass...")
    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4]], device="mps")
        logits = model(input_ids)
        torch.mps.synchronize()

    print(f"Output shape: {logits.shape}")

    # Check MPS memory after forward (dequant caches filled)
    if torch.backends.mps.is_available():
        print(f"MPS allocated (after forward): {torch.mps.current_allocated_memory() / 1e9:.2f} GB")

    # Count cached dequantized weights
    n_cached = sum(
        1
        for _, m in model.named_modules()
        if isinstance(m, TrellisLinear) and m._dequantized_cache is not None
    )
    print(f"TrellisLinear modules with dequant cache: {n_cached}/{n_linear}")


if __name__ == "__main__":
    main()
