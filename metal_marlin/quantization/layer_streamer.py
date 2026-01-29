"""Layer-wise model streaming for memory-efficient quantization.

Streams model layers one at a time from safetensors files,
never loading more than one layer into memory at once.
Supports sharded models (model-00001-of-00004.safetensors).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import safetensors.torch
import torch


@dataclass
class LayerWeights:
    """Single layer's weights loaded from disk."""

    name: str
    weight: torch.Tensor  # [out, in] on CPU
    bias: torch.Tensor | None
    layer_idx: int
    total_layers: int


class LayerStreamer:
    """Stream model layers one at a time from safetensors.

    Never loads more than one layer into memory at once.
    Supports sharded models (model-00001-of-00004.safetensors).
    """

    def __init__(self, model_path: Path):
        """Initialize the layer streamer.

        Args:
            model_path: Path to model directory or single safetensors file.
        """
        self.model_path = Path(model_path)
        self._index = self._build_tensor_index()

    def _build_tensor_index(self) -> dict[str, tuple[Path, str]]:
        """Map tensor names to (file, key) for lazy loading.

        Returns:
            Dictionary mapping tensor names to (file_path, tensor_key).
        """
        index: dict[str, tuple[Path, str]] = {}

        if self.model_path.is_file() and self.model_path.suffix == ".safetensors":
            # Single file model
            with safetensors.torch.safe_open(self.model_path, framework="pt") as f:
                for key in f.keys():
                    index[key] = (self.model_path, key)

        elif self.model_path.is_dir():
            # Check for sharded model index
            index_file = self.model_path / "model.safetensors.index.json"

            if index_file.exists():
                # Sharded model with index
                with open(index_file, encoding="utf-8") as f:
                    metadata = json.load(f)

                weight_map = metadata.get("weight_map", {})
                for tensor_name, filename in weight_map.items():
                    file_path = self.model_path / filename
                    index[tensor_name] = (file_path, tensor_name)
            else:
                # Look for safetensors files in directory
                safetensor_files = sorted(self.model_path.glob("*.safetensors"))
                for file_path in safetensor_files:
                    with safetensors.torch.safe_open(file_path, framework="pt") as f:
                        for key in f.keys():
                            index[key] = (file_path, key)

        return index

    def _is_linear(self, key: str) -> bool:
        """Check if a tensor key corresponds to a linear layer weight.

        Args:
            key: Tensor key name.

        Returns:
            True if key is a linear layer weight (ends with .weight and
            contains typical linear layer patterns).
        """
        if not key.endswith(".weight"):
            return False

        # Skip embeddings, norms, and other non-linear layers
        skip_patterns = [
            "embed",
            "norm",
            "ln_",
            "layernorm",
            "bias",
            "lm_head",
            "word_embeddings",
        ]

        lower_key = key.lower()
        for pattern in skip_patterns:
            if pattern in lower_key:
                return False

        # Must be a 2D tensor for linear layer [out_features, in_features]
        # We'll verify shape when loading, but filter by name here
        return True

    def iter_linear_layers(self) -> Iterator[LayerWeights]:
        """Yield linear layers one at a time.

        Loads weight from disk, yields, then allows GC before next.

        Yields:
            LayerWeights for each linear layer found in the model.
        """
        linear_keys = [k for k in self._index if self._is_linear(k)]
        total = len(linear_keys)

        for idx, key in enumerate(sorted(linear_keys)):
            file_path, tensor_key = self._index[key]

            # Load weight tensor
            with safetensors.torch.safe_open(file_path, framework="pt") as f:
                weight = f.get_tensor(tensor_key)

            # Ensure weight is 2D (linear layers should be [out, in])
            if weight.dim() != 2:
                continue

            # Try to load bias if present
            bias_key = key.replace(".weight", ".bias")
            bias = None
            if bias_key in self._index:
                bias_file, bias_tensor_key = self._index[bias_key]
                with safetensors.torch.safe_open(bias_file, framework="pt") as f:
                    bias = f.get_tensor(bias_tensor_key)

            yield LayerWeights(
                name=key,
                weight=weight,
                bias=bias,
                layer_idx=idx,
                total_layers=total,
            )

    def estimate_layer_memory(self) -> dict[str, int]:
        """Estimate memory per layer for RAM planning.

        Returns:
            Dictionary with memory estimates in bytes for different metrics.
        """
        total_params = 0
        max_layer_params = 0
        max_layer_memory = 0

        linear_keys = [k for k in self._index if self._is_linear(k)]

        for key in linear_keys:
            file_path, tensor_key = self._index[key]
            with safetensors.torch.safe_open(file_path, framework="pt") as f:
                # Use get_slice to get shape/dtype without loading full tensor
                slice_obj = f.get_slice(tensor_key)
                shape = slice_obj.get_shape()
                dtype = slice_obj.get_dtype()

                # Calculate params for this tensor
                params = 1
                for dim in shape:
                    params *= dim

                total_params += params

                # Estimate memory (bytes per element based on dtype)
                bytes_per_element = self._dtype_to_bytes(dtype)
                memory = params * bytes_per_element

                # Include bias memory estimate
                bias_key = key.replace(".weight", ".bias")
                if bias_key in self._index:
                    bias_file, bias_tensor_key = self._index[bias_key]
                    with safetensors.torch.safe_open(bias_file, framework="pt") as f2:
                        bias_slice = f2.get_slice(bias_tensor_key)
                        bias_shape = bias_slice.get_shape()
                        bias_params = 1
                        for dim in bias_shape:
                            bias_params *= dim
                        memory += bias_params * bytes_per_element

                max_layer_params = max(max_layer_params, params)
                max_layer_memory = max(max_layer_memory, memory)

        return {
            "total_params": total_params,
            "max_layer_params": max_layer_params,
            "max_layer_memory_bytes": max_layer_memory,
            "num_layers": len(linear_keys),
            "avg_params_per_layer": total_params // max(len(linear_keys), 1),
        }

    def _dtype_to_bytes(self, dtype: str) -> int:
        """Convert dtype string to bytes per element.

        Args:
            dtype: Data type string from safetensors.

        Returns:
            Bytes per element for the given dtype.
        """
        dtype_map = {
            "F64": 8,
            "F32": 4,
            "F16": 2,
            "BF16": 2,
            "I64": 8,
            "I32": 4,
            "I16": 2,
            "I8": 1,
            "U8": 1,
            "BOOL": 1,
            "F8_E4M3": 1,
            "F8_E5M2": 1,
        }
        return dtype_map.get(dtype.upper(), 4)  # Default to F32

    def list_layers(self) -> list[str]:
        """List all available layer names in the model.

        Returns:
            List of tensor keys available in the model.
        """
        return sorted(self._index.keys())
