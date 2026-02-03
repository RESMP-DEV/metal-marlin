#!/usr/bin/env python3
"""Count kernel dispatches per forward pass."""

import time

import torch

# Counters
call_count = {"dispatch": 0, "mps_tensor": 0, "command_buffer": 0}

# Patch metal_dispatch module before importing model
import metal_marlin.metal_dispatch as md

orig_dispatch = md.dispatch_kernel
orig_mps = md.mps_tensor_to_metal_buffer


def counting_dispatch(*args, **kwargs):
    call_count["dispatch"] += 1
    return orig_dispatch(*args, **kwargs)


def counting_mps(*args, **kwargs):
    call_count["mps_tensor"] += 1
    return orig_mps(*args, **kwargs)


md.dispatch_kernel = counting_dispatch
md.mps_tensor_to_metal_buffer = counting_mps

# Patch moe_dispatch too
import metal_marlin.trellis.moe_dispatch as moe_d

moe_d.dispatch_kernel = counting_dispatch
moe_d.mps_tensor_to_metal_buffer = counting_mps

# Now import model
from metal_marlin.trellis.model import TrellisForCausalLM


def main():
    print("Loading model...")
    model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-Trellis-3bpw", device="mps")

    x = torch.tensor([[1]], device="mps")

    # Reset counters
    call_count["dispatch"] = 0
    call_count["mps_tensor"] = 0

    print("Running forward pass...")
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    print("\nResults:")
    print(f"  dispatch_kernel calls: {call_count['dispatch']}")
    print(f"  mps_tensor_to_metal_buffer calls: {call_count['mps_tensor']}")
    print(f"  Forward time: {elapsed * 1000:.0f} ms")

    if call_count["dispatch"] > 0:
        print(f"\n  Each dispatch has wait=True, causing {call_count['dispatch']} GPU stalls!")
        print(
            f"  Estimated overhead: {call_count['dispatch'] * 0.1:.1f} ms just for synchronization"
        )


if __name__ == "__main__":
    main()
