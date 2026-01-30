import concurrent.futures

import torch


class ANEAsyncExecutor:
    """
    Execute ANE models asynchronously to overlap with GPU.

    Uses thread pool since Core ML is thread-safe and ANE
    execution doesn't block the GPU.
    """

    def __init__(self, max_workers: int = 2):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers)

    def submit(self, ane_model, input_tensor: torch.Tensor) -> concurrent.futures.Future:
        """Submit ANE work, returns Future."""

        def _run():
            # Transfer to CPU
            x_np = input_tensor.cpu().numpy()
            # Run ANE
            out = ane_model.predict({"input": x_np})
            # Return numpy, caller converts to MPS
            return out["output"]

        return self._executor.submit(_run)

    def shutdown(self):
        self._executor.shutdown(wait=True)


# Global executor
_ANE_EXECUTOR: ANEAsyncExecutor | None = None


def get_ane_executor() -> ANEAsyncExecutor:
    global _ANE_EXECUTOR
    if _ANE_EXECUTOR is None:
        _ANE_EXECUTOR = ANEAsyncExecutor()
    return _ANE_EXECUTOR
