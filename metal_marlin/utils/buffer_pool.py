"""Pre-allocated buffer pool for inference."""


import torch


class BufferPool:
    """Pool of pre-allocated tensors to avoid allocation overhead."""

    def __init__(self, device: torch.device):
        self.device = device
        self.buffers: dict[str, torch.Tensor] = {}

    def get(
        self,
        name: str,
        shape: tuple,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Get or create a buffer."""
        key = f"{name}_{shape}_{dtype}"
        if key not in self.buffers:
            self.buffers[key] = torch.zeros(
                shape, dtype=dtype, device=self.device
            )
        return self.buffers[key]

    def clear(self):
        """Release all buffers."""
        self.buffers.clear()
