#!/usr/bin/env python3
"""Time GPU wait separately from Python overhead."""

import time
from contextlib import contextmanager

import torch

from metal_marlin.metal_dispatch import MetalKernelLibrary
from metal_marlin.trellis.model import TrellisForCausalLM


# Patch batch_dispatch to time the GPU wait
@contextmanager
def timed_batch_dispatch(self):
    """Timed version of batch_dispatch."""
    self._batch_mode = True
    self._batch_command_buffer = self.command_queue.commandBuffer()
    self._batch_encoder = self._batch_command_buffer.computeCommandEncoder()
    try:
        yield
    finally:
        self._batch_encoder.endEncoding()
        self._batch_command_buffer.commit()

        t_commit = time.perf_counter()
        self._batch_command_buffer.waitUntilCompleted()
        t_wait = time.perf_counter()

        print(f"GPU wait time: {(t_wait - t_commit) * 1000:.0f}ms")

        self._batch_mode = False
        self._batch_encoder = None
        self._batch_command_buffer = None


def main():
    # Patch before loading model
    MetalKernelLibrary.batch_dispatch = timed_batch_dispatch

    print("Loading model...")
    model = TrellisForCausalLM.from_pretrained("models/GLM-4.7-Flash-Trellis-3bpw", device="mps")

    x = torch.tensor([[1]], device="mps")

    # Warmup
    print("Warmup...")
    with torch.no_grad():
        _ = model(x)
    torch.mps.synchronize()

    print("\n--- Timed forward ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    torch.mps.synchronize()
    total = time.perf_counter() - t0
    print(f"Total forward: {total * 1000:.0f}ms")


if __name__ == "__main__":
    main()
