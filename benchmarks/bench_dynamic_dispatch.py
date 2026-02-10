#!/usr/bin/env python3
"""Benchmark dynamic C++ vs Python MoE dispatch."""
import time
from pathlib import Path

import torch
from metal_marlin.trellis.model import TrellisForCausalLM
from transformers import AutoTokenizer

# Model path relative to this script
MODEL_PATH = Path(__file__).parent.parent / "models" / "GLM-4.7-Flash-Marlin-MMFP4"

print("Loading model...")
model = TrellisForCausalLM.from_pretrained(str(MODEL_PATH), device="mps")
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH), trust_remote_code=True)

tokens = tokenizer("Hello", return_tensors="pt").input_ids.to("mps")

# Check if dynamic dispatch is enabled (mixed-precision path)
using_dynamic = any(
    hasattr(layer.mlp, '_dynamic_dispatcher') and
    layer.mlp._dynamic_dispatcher is not None
    for layer in model.model.layers
    if hasattr(layer.mlp, '_dynamic_dispatcher')
)
using_batched = any(
    hasattr(layer.mlp, '_batched_dispatcher') and
    layer.mlp._batched_dispatcher is not None
    for layer in model.model.layers
    if hasattr(layer.mlp, '_batched_dispatcher')
)
is_mixed = any(
    hasattr(layer.mlp, '_is_mixed_precision') and
    layer.mlp._is_mixed_precision
    for layer in model.model.layers
    if hasattr(layer.mlp, '_is_mixed_precision')
)
print(f"Mixed precision: {is_mixed}")
print(f"Using dynamic C++ dispatch: {using_dynamic}")
print(f"Using batched dispatcher: {using_batched}")

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model(tokens)
torch.mps.synchronize()

# Benchmark
times = []
for _ in range(10):
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(tokens)
    torch.mps.synchronize()
    times.append(time.perf_counter() - start)

avg = sum(times) / len(times)
print(f"\nAverage: {avg*1000:.0f}ms ({1/avg:.1f} tok/s)")
if using_dynamic:
    print("Dispatch type: Dynamic C++ (mixed-precision)")
elif using_batched:
    print("Dispatch type: Batched C++")
else:
    print("Dispatch type: Python Fallback")
