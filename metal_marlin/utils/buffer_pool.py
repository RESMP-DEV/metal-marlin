"""Pre-allocated buffer pool for inference."""
import logging


import torch



logger = logging.getLogger(__name__)

class BufferPool:
    """Pool of pre-allocated tensors to avoid allocation overhead."""

    def __init__(self, device: torch.device):
        logger.debug("initializing %s with device=%s", type(self).__name__, device)
        self.device = device
        self.buffers: dict[str, torch.Tensor] = {}

    def get(
        self,
        name: str,
        shape: tuple,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Get or create a buffer."""
        logger.debug("get called with name=%s, shape=%s, dtype=%s", name, shape, dtype)
        key = f"{name}_{shape}_{dtype}"
        if key not in self.buffers:
            self.buffers[key] = torch.zeros(
                shape, dtype=dtype, device=self.device
            )
        return self.buffers[key]

    def clear(self):
        """Release all buffers."""
        logger.debug("clear called")
        self.buffers.clear()
