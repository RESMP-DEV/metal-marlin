"""Pre-allocated buffers for CPU<->ANE<->MPS transfers."""

import numpy as np
import torch


class ANEBufferPool:
    """Pre-allocated buffers for CPU<->ANE<->MPS transfers."""

    def __init__(self, max_seq_len: int, hidden_size: int):
        """Initialize the buffer pool with pre-allocated tensors.

        Args:
            max_seq_len: Maximum sequence length for buffer allocation
            hidden_size: Hidden dimension size for buffer allocation
        """
        self._cpu_input = torch.empty(1, max_seq_len, hidden_size, dtype=torch.float32)
        self._cpu_output = torch.empty(1, max_seq_len, hidden_size, dtype=torch.float32)

    def to_ane(self, x: torch.Tensor) -> np.ndarray:
        """Copy MPS tensor to pre-allocated CPU buffer and return as numpy array.

        Args:
            x: Input tensor on MPS device

        Returns:
            Numpy array view of the CPU buffer containing the tensor data
        """
        # Copy MPS -> pre-allocated CPU buffer
        self._cpu_input[:, : x.size(1), :].copy_(x.cpu())
        return self._cpu_input[:, : x.size(1), :].numpy()

    def from_ane(self, out: np.ndarray, device: torch.device) -> torch.Tensor:
        """Copy ANE output to target device with minimal allocation.

        Args:
            out: Numpy array from ANE computation
            device: Target device (typically MPS)

        Returns:
            Tensor on the specified device containing the ANE output
        """
        # Copy ANE output -> MPS with minimal allocation
        return torch.from_numpy(out).to(device, non_blocking=True)
