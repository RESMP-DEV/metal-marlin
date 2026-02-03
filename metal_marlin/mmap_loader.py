"""Memory-mapped safetensors loading for large models.

Uses mmap() to avoid loading weights into RAM upfront:
- OS pages in weights on demand during forward pass
- Reduces model load time from O(model_size) to near-instant
- Memory pressure handled by OS virtual memory subsystem

The safetensors format stores tensors contiguously with a JSON header,
making it ideal for memory-mapped access. Each tensor can be accessed
independently without reading the entire file.

Usage:
    from metal_marlin.mmap_loader import MmapSafetensorsLoader

    # Open files for mmap access
    loader = MmapSafetensorsLoader(Path("model/"))

    # Lazy tensor access - only pages in when data is accessed
    tensor = loader.get_tensor("model.layers.0.mlp.gate_proj.weight")

    # Keep loader open during inference to maintain mmap
    # Close when done to release mmap handles
    loader.close()

    # Or use as context manager:
    with MmapSafetensorsLoader(Path("model/")) as loader:
        tensor = loader.get_tensor("name")
"""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

if TYPE_CHECKING:
    from collections.abc import Iterator


class MmapSafetensorsLoader:
    """Memory-mapped loader for safetensors files.

    Opens safetensor files with mmap, allowing the OS to page in
    weight data on demand. This dramatically reduces load time and
    peak memory usage for large models.

    Attributes:
        files: Dictionary mapping file paths to open safe_open handles.
        tensor_to_file: Maps tensor names to their containing file.
    """

    def __init__(
        self,
        path: Path | str,
        device: str = "cpu",
        framework: str = "pt",
    ):
        """Initialize the mmap loader.

        Args:
            path: Path to a safetensors file or directory containing shards.
            device: Device for loaded tensors ("cpu", "mps", etc.).
                    Memory mapping only works with CPU tensors initially.
            framework: Safetensors framework ("pt" for PyTorch, "np" for numpy).
        """
        self.path = Path(path)
        self.device = device
        self.framework = framework
        self._handles: dict[Path, safe_open] = {}
        self._tensor_to_file: dict[str, Path] = {}
        self._closed = False

        self._discover_and_open()

    def _discover_and_open(self) -> None:
        """Discover safetensor files and open them with mmap."""
        if self.path.is_file():
            files = [self.path]
        else:
            files = sorted(self.path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(
                    f"No .safetensors files found in {self.path}"
                )

        for file_path in files:
            # safe_open uses mmap internally when framework is pt/np
            # The device="cpu" is required for mmap to work
            handle = safe_open(str(file_path), framework=self.framework)
            self._handles[file_path] = handle

            # Build tensor->file index
            for name in handle.keys():
                if name in self._tensor_to_file:
                    raise ValueError(
                        f"Duplicate tensor name '{name}' found in "
                        f"{file_path} and {self._tensor_to_file[name]}"
                    )
                self._tensor_to_file[name] = file_path

    def keys(self) -> Iterator[str]:
        """Iterate over all tensor names."""
        self._check_closed()
        return iter(self._tensor_to_file.keys())

    def __len__(self) -> int:
        """Number of tensors available."""
        return len(self._tensor_to_file)

    def __contains__(self, name: str) -> bool:
        """Check if a tensor name exists."""
        return name in self._tensor_to_file

    def get_tensor(self, name: str) -> torch.Tensor:
        """Get a tensor by name.

        The tensor data is memory-mapped, so actual memory allocation
        only happens when the tensor data is accessed (e.g., during
        a forward pass or when moved to GPU).

        Args:
            name: Tensor name as stored in the safetensors file.

        Returns:
            PyTorch tensor (on CPU, memory-mapped).

        Raises:
            KeyError: If tensor name not found.
            RuntimeError: If loader has been closed.
        """
        self._check_closed()

        if name not in self._tensor_to_file:
            raise KeyError(f"Tensor '{name}' not found. Available: {list(self._tensor_to_file.keys())[:10]}...")

        file_path = self._tensor_to_file[name]
        handle = self._handles[file_path]

        # get_tensor returns a view into the mmap'd buffer
        tensor = handle.get_tensor(name)

        # Move to device if not CPU
        if self.device != "cpu":
            tensor = tensor.to(self.device)

        return tensor

    def get_slice(self, name: str) -> TensorSlice:
        """Get a slice accessor for a tensor.

        Returns a lightweight object that can be used to access
        tensor metadata (shape, dtype) without loading the data.

        Args:
            name: Tensor name.

        Returns:
            TensorSlice object with shape, dtype properties.
        """
        self._check_closed()

        if name not in self._tensor_to_file:
            raise KeyError(f"Tensor '{name}' not found")

        file_path = self._tensor_to_file[name]
        handle = self._handles[file_path]
        return handle.get_slice(name)

    def get_shape(self, name: str) -> tuple[int, ...]:
        """Get tensor shape without loading data.

        Args:
            name: Tensor name.

        Returns:
            Shape tuple.
        """
        slice_obj = self.get_slice(name)
        return tuple(slice_obj.get_shape())

    def get_dtype(self, name: str) -> torch.dtype:
        """Get tensor dtype without loading data.

        Args:
            name: Tensor name.

        Returns:
            PyTorch dtype.
        """
        slice_obj = self.get_slice(name)
        dtype_str = slice_obj.get_dtype()
        return _safetensors_dtype_to_torch(dtype_str)

    def load_subset(
        self,
        names: list[str],
        device: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load a subset of tensors.

        Useful for loading specific layers while keeping memory
        usage bounded.

        Args:
            names: List of tensor names to load.
            device: Target device (defaults to loader's device).

        Returns:
            Dictionary of loaded tensors.
        """
        device = device or self.device
        result = {}
        for name in names:
            tensor = self.get_tensor(name)
            if device != "cpu":
                tensor = tensor.to(device)
            result[name] = tensor
        return result

    def load_layer(
        self,
        layer_idx: int,
        pattern: str = "model.layers.{layer}.",
    ) -> dict[str, torch.Tensor]:
        """Load all tensors for a specific layer.

        Args:
            layer_idx: Layer index.
            pattern: Pattern to match layer tensors. {layer} is replaced
                    with the layer index.

        Returns:
            Dictionary of tensors for the layer.
        """
        prefix = pattern.format(layer=layer_idx)
        names = [n for n in self._tensor_to_file if n.startswith(prefix)]
        return self.load_subset(names)

    def close(self) -> None:
        """Close all file handles and release mmaps."""
        if self._closed:
            return

        for handle in self._handles.values():
            # safe_open handles don't have explicit close, but we clear refs
            pass

        self._handles.clear()
        self._tensor_to_file.clear()
        self._closed = True

    def _check_closed(self) -> None:
        """Raise if loader has been closed."""
        if self._closed:
            raise RuntimeError("MmapSafetensorsLoader has been closed")

    def __enter__(self) -> MmapSafetensorsLoader:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _safetensors_dtype_to_torch(dtype_str: str) -> torch.dtype:
    """Convert safetensors dtype string to torch dtype."""
    dtype_map = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F64": torch.float64,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    return dtype_map.get(dtype_str, torch.float32)


def load_safetensors_mmap(
    path: Path | str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load all tensors from safetensors file(s) using mmap.

    Convenience function that loads all tensors at once, but still
    uses mmap to avoid memory copies during the load phase.

    Args:
        path: Path to safetensors file or directory.
        device: Target device for tensors.

    Returns:
        State dict with all tensors.
    """
    with MmapSafetensorsLoader(path, device=device) as loader:
        return {name: loader.get_tensor(name) for name in loader.keys()}


def estimate_mmap_overhead(
    path: Path | str,
) -> dict[str, int | float]:
    """Estimate memory overhead for mmap loading.

    Reports the size of data that will be mmap'd vs what would
    be allocated in a traditional load.

    Args:
        path: Path to safetensors file(s).

    Returns:
        Dict with mmap_bytes, virtual_bytes, and page_size.
    """

    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.safetensors"))

    total_bytes = sum(f.stat().st_size for f in files)
    page_size = mmap.PAGESIZE

    return {
        "file_bytes": total_bytes,
        "file_gb": total_bytes / (1024**3),
        "page_size": page_size,
        "pages": (total_bytes + page_size - 1) // page_size,
        # Virtual memory overhead is minimal - just page table entries
        "vm_overhead_estimate_mb": ((total_bytes // page_size) * 8) / (1024**2),
    }
