"""Dedicated CUDA stream manager for overlapping weight transfers with compute.

On CUDA, you can overlap H2D transfers (copy engine) with GPU compute (SMs) using
separate streams. This is critical for Ubuntu performance.
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

    def __init__(self, device: str = "cuda:0") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TransferStream requires CUDA. torch.cuda.is_available() returned False."
            )
        self._stream = torch.cuda.Stream(device=device)
        self._device = device

    def transfer_async(
        self, tensors: dict[str, torch.Tensor], device: str | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.cuda.Event]:
        """Transfer tensors on dedicated stream, return transferred tensors and sync event.

        Args:
            tensors: Dictionary mapping tensor names to CPU tensors to transfer.
            device: Target device (defaults to the device specified at init).

        Returns:
            A tuple of (transferred_tensors, event) where event can be waited on
            to ensure transfers complete.
        """
        target_device = device if device is not None else self._device
        transferred: dict[str, torch.Tensor] = {}

        with torch.cuda.stream(self._stream):
            for name, tensor in tensors.items():
                transferred[name] = tensor.to(target_device, non_blocking=True)

        # Record an event on the transfer stream for synchronization
        event = torch.cuda.Event()
        event.record(self._stream)
        return transferred, event

    def synchronize(self) -> None:
        """Wait for all pending transfers."""
        self._stream.synchronize()
