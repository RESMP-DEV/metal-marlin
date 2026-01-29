"""Loader for EXL3 trellis-quantized models.

The EXL3 format stores each layer in a separate directory with:
- index.json: Metadata about the tensors in the layer
- tensor_*.safetensors: Safetensors files containing the quantized weights

Each quantized weight is stored as four components:
- {name}__indices: int16 trellis indices [tiles_k, tiles_n, 256]
- {name}__scales: float32 per-tile scales [n_groups, N]
- {name}__su: float32 row sign vector [K]
- {name}__sv: float32 column sign vector [N]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


@dataclass
class TrellisWeight:
    """A trellis-quantized weight tensor.
    
    The trellis quantization scheme decomposes a weight matrix W into:
    - indices: Trellis codebook indices for each 16x16 tile
    - scales: Per-tile or per-group scaling factors
    - su: Row sign vector for sign restoration
    - sv: Column sign vector for sign restoration
    
    Attributes:
        indices: Trellis indices [tiles_k, tiles_n, 256] int16
        scales: Per-tile scales [n_groups, N] float32
        su: Row sign vector [K] float32
        sv: Column sign vector [N] float32
        bits: Quantization bit width (2-8)
        original_shape: Original weight shape (K, N)
    """

    indices: torch.Tensor  # [tiles_k, tiles_n, 256] int16
    scales: torch.Tensor  # [n_groups, N] float32
    su: torch.Tensor  # [K] float32
    sv: torch.Tensor  # [N] float32
    bits: int  # 2-8
    original_shape: tuple[int, int]

    @property
    def K(self) -> int:
        """Input dimension."""
        return self.original_shape[0]

    @property
    def N(self) -> int:
        """Output dimension."""
        return self.original_shape[1]


class TrellisModelLoader:
    """Loader for EXL3 trellis-quantized models.
    
    Models are structured with layers in separate directories:
    ```
    model_path/
        layer_0000/index.json, tensor_*.safetensors
        layer_0001/index.json, tensor_*.safetensors
        ...
    ```
    
    Example:
        >>> loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
        >>> print(f"Layers: {loader.get_num_layers()}")
        >>> layer0 = loader.load_layer(0)
        >>> for name, weight in layer0.items():
        ...     print(f"{name}: shape={weight.original_shape}, bits={weight.bits}")
    """

    def __init__(self, model_path: Path | str):
        """Initialize the loader.
        
        Args:
            model_path: Path to the model directory containing layer_*/ subdirectories.
            
        Raises:
            FileNotFoundError: If the model path doesn't exist or has no layers.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Discover available layers
        self._layer_indices: list[int] = []
        for item in self.model_path.iterdir():
            if item.is_dir() and item.name.startswith("layer_"):
                try:
                    layer_idx = int(item.name.split("_")[1])
                    self._layer_indices.append(layer_idx)
                except (IndexError, ValueError):
                    continue

        self._layer_indices.sort()
        if not self._layer_indices:
            raise FileNotFoundError(f"No layer directories found in {model_path}")

        # Cache for layer metadata
        self._layer_metadata: dict[int, dict[str, Any]] = {}

    def _get_layer_path(self, layer_idx: int) -> Path:
        """Get the path to a layer directory."""
        return self.model_path / f"layer_{layer_idx:04d}"

    def _load_layer_metadata(self, layer_idx: int) -> dict[str, Any]:
        """Load and cache the index.json for a layer."""
        if layer_idx not in self._layer_metadata:
            layer_path = self._get_layer_path(layer_idx)
            index_path = layer_path / "index.json"
            if not index_path.exists():
                raise FileNotFoundError(f"Layer index not found: {index_path}")

            with open(index_path) as f:
                self._layer_metadata[layer_idx] = json.load(f)

        return self._layer_metadata[layer_idx]

    def _tensor_name_to_key(self, tensor_name: str) -> str:
        """Convert tensor name (with dots) to safetensors key (with double underscores).
        
        Example: "model.layers.0.mlp.down_proj.weight" -> 
                 "model__layers__0__mlp__down_proj__weight"
        """
        return tensor_name.replace(".", "__")

    def _find_tensor_files(self, layer_idx: int) -> list[Path]:
        """Find all safetensor files for a layer."""
        layer_path = self._get_layer_path(layer_idx)
        return sorted(layer_path.glob("tensor_*.safetensors"))

    def load_layer(self, layer_idx: int) -> dict[str, TrellisWeight]:
        """Load all weights for a single layer.
        
        Args:
            layer_idx: Layer index to load.
            
        Returns:
            Dictionary mapping tensor names to TrellisWeight objects.
            
        Raises:
            FileNotFoundError: If the layer doesn't exist.
            ValueError: If a tensor is missing required components.
        """
        metadata = self._load_layer_metadata(layer_idx)
        tensor_files = self._find_tensor_files(layer_idx)

        if not tensor_files:
            raise FileNotFoundError(f"No tensor files found for layer {layer_idx}")

        # Load all tensors from all files
        all_tensors: dict[str, torch.Tensor] = {}
        for tensor_file in tensor_files:
            file_tensors = load_file(tensor_file)
            all_tensors.update(file_tensors)

        # Parse each tensor from metadata
        weights: dict[str, TrellisWeight] = {}
        tensor_infos = metadata.get("tensors", [])

        for tensor_info in tensor_infos:
            name = tensor_info["name"]
            bits = tensor_info["bits"]
            shape = tuple(tensor_info["shape"])

            base_key = self._tensor_name_to_key(name)

            # Find the components
            indices_key = f"{base_key}__indices"
            scales_key = f"{base_key}__scales"
            su_key = f"{base_key}__su"
            sv_key = f"{base_key}__sv"

            if indices_key not in all_tensors:
                raise ValueError(f"Missing indices for tensor: {name} (key: {indices_key})")
            if scales_key not in all_tensors:
                raise ValueError(f"Missing scales for tensor: {name} (key: {scales_key})")
            if su_key not in all_tensors:
                raise ValueError(f"Missing su for tensor: {name} (key: {su_key})")
            if sv_key not in all_tensors:
                raise ValueError(f"Missing sv for tensor: {name} (key: {sv_key})")

            indices = all_tensors[indices_key]
            scales = all_tensors[scales_key]
            su = all_tensors[su_key]
            sv = all_tensors[sv_key]

            # Validate shapes and dtypes
            if indices.dtype != torch.int16:
                raise ValueError(
                    f"Indices must be int16, got {indices.dtype} for {name}"
                )
            if scales.dtype != torch.float32:
                raise ValueError(
                    f"Scales must be float32, got {scales.dtype} for {name}"
                )
            if su.dtype != torch.float32:
                raise ValueError(f"su must be float32, got {su.dtype} for {name}")
            if sv.dtype != torch.float32:
                raise ValueError(f"sv must be float32, got {sv.dtype} for {name}")

            weights[name] = TrellisWeight(
                indices=indices,
                scales=scales,
                su=su,
                sv=sv,
                bits=bits,
                original_shape=shape,
            )

        return weights

    def get_num_layers(self) -> int:
        """Return the number of layers in the model."""
        return len(self._layer_indices)

    def get_layer_indices(self) -> list[int]:
        """Return the list of available layer indices."""
        return self._layer_indices.copy()

    def get_config(self) -> dict[str, Any]:
        """Return model config (from base model or inferred).
        
        Returns:
            Dictionary with model configuration. If no config.json exists,
            returns a config inferred from layer metadata.
        """
        # Try to load config.json from model root
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        # Infer config from layer metadata
        config: dict[str, Any] = {
            "num_hidden_layers": self.get_num_layers(),
            "quantization": {
                "format": "exl3_trellis",
            },
        }

        # Try to get more info from first layer
        try:
            first_layer_meta = self._load_layer_metadata(self._layer_indices[0])
            tensors = first_layer_meta.get("tensors", [])
            if tensors:
                # Infer hidden size from first tensor shape
                first_shape = tensors[0].get("shape", [0, 0])
                config["hidden_size"] = first_shape[0] if first_shape else None
        except (FileNotFoundError, IndexError):
            pass

        return config

    def load_all_layers(self) -> dict[int, dict[str, TrellisWeight]]:
        """Load all layers into memory.
        
        Warning: This can consume significant memory for large models.
        Consider using load_layer() for individual layers if memory is constrained.
        
        Returns:
            Dictionary mapping layer indices to their weight dictionaries.
        """
        return {idx: self.load_layer(idx) for idx in self._layer_indices}

    def get_layer_tensor_names(self, layer_idx: int) -> list[str]:
        """Get the names of all tensors in a layer without loading them.
        
        Args:
            layer_idx: Layer index to query.
            
        Returns:
            List of tensor names in the layer.
        """
        metadata = self._load_layer_metadata(layer_idx)
        return [t["name"] for t in metadata.get("tensors", [])]

    def get_tensor_info(self, layer_idx: int, tensor_name: str) -> dict[str, Any] | None:
        """Get metadata for a specific tensor.
        
        Args:
            layer_idx: Layer index.
            tensor_name: Name of the tensor.
            
        Returns:
            Tensor metadata dictionary or None if not found.
        """
        metadata = self._load_layer_metadata(layer_idx)
        for tensor_info in metadata.get("tensors", []):
            if tensor_info["name"] == tensor_name:
                return {
                    "name": tensor_info["name"],
                    "bits": tensor_info["bits"],
                    "shape": tuple(tensor_info["shape"]),
                    "mse": tensor_info.get("mse"),
                }
        return None
