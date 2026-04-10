"""CUDA stream manager for overlapping weight transfers with compute.

This module provides a dedicated CUDA stream for host-to-device weight transfers,
enabling overlapping of weight loading with inference compute on NVIDIA GPUs.
"""

from __future__ import annotations

import torch


class TransferStream:
    """Manages a dedicated CUDA stream for host-to-device weight transfers.

    Enables overlapping of weight loading with inference compute:
    - Transfer stream: loads next layer's weights via DMA
    - Compute stream: runs current layer's kernels on SMs
    - Event synchronization ensures transfer completes before compute reads

    Usage:
        ts = TransferStream(device="cuda:0")
        # Start prefetching next layer while current layer runs
        event = ts.transfer_async(cpu_tensors, device="cuda:0")
        # ... run current layer on default stream ...
        event.wait()  # Ensure transfer complete before using tensors
    """

    def __init__(self, device: str = "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self._stream = torch.cuda.Stream(device=device)
        self._device = device

    def transfer_async(self, tensors: dict[str, torch.Tensor],
                     device: str | None = None) -> torch.cuda.Event:
        """Transfer tensors on dedicated stream, return sync event."""
        target_device = device if device is not None else self._device

        with torch.cuda.stream(self._stream):
            for name, tensor in tensors.items():
                tensors[name] = tensor.to(target_device, non_blocking=True)

        event = torch.cuda.Event()
        event.record(self._stream)
        return event

    def synchronize(self):
        """Wait for all pending transfers."""
        self._stream.synchronize()
