"""Loader for Trellis-quantized models.

Supports two storage formats:

**v3 (HuggingFace-style shards)** - Preferred for CUDA-quantized models:
    model_path/
        config.json
        model.safetensors.index.json  # Weight map
        quantization_index.json        # Bits/MSE metadata
        model-00001-of-XXXXX.safetensors
        base_weights.safetensors       # Optional
        tokenizer.json, tokenizer_config.json

**v2 (Layer directories)** - Legacy format:
    model_path/
        layer_0000/
            index.json
            batch_*.safetensors (or tensor_*.safetensors)
        layer_0001/
            ...
        base_weights.safetensors

Each quantized weight is stored as four components:
- {name}__indices: Packed uint8 trellis indices (with header byte for bits)
- {name}__scales: float32 per-group scales [n_groups, N]
- {name}__su: float32 row sign vector [K]
- {name}__sv: float32 column sign vector [N]

Format is auto-detected based on presence of model.safetensors.index.json (v3)
or layer_*/ directories (v2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..mmap_loader import MmapSafetensorsLoader


@dataclass
class TrellisWeight:
    """A trellis-quantized weight tensor.

    The trellis quantization scheme decomposes a weight matrix W into:
    - packed_indices: Packed trellis codebook indices for each 16x16 tile (uint8)
    - scales: Per-tile or per-group scaling factors
    - su: Row sign vector for sign restoration
    - sv: Column sign vector for sign restoration

    Attributes:
        packed_indices: Packed trellis indices [tiles_k, tiles_n, packed_size] uint8
        scales: Per-tile scales [n_groups, N] float32
        su: Row sign vector [K] float32
        sv: Column sign vector [N] float32
        bits: Quantization bit width (2-8)
        original_shape: Original weight shape (K, N)
    """

    packed_indices: torch.Tensor  # [tiles_k, tiles_n, packed_size] uint8
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


def detect_trellis_format(model_path: Path) -> str:
    """Detect trellis model storage format.

    Args:
        model_path: Path to model directory

    Returns:
        "v3_hf_shards": HuggingFace-style sharded safetensors
        "v2_layers": Layer-based directory structure
        "unknown": Unrecognized format
    """
    model_path = Path(model_path)

    # Check for HF-style shards (v3)
    if (model_path / "model.safetensors.index.json").exists():
        return "v3_hf_shards"

    # Check for layer-based format (v2)
    if any(d.name.startswith("layer_") for d in model_path.iterdir() if d.is_dir()):
        return "v2_layers"

    return "unknown"


def _parse_hf_weight_map(index_path: Path) -> dict[str, str]:
    """Parse HuggingFace model.safetensors.index.json.

    Args:
        index_path: Path to model.safetensors.index.json

    Returns:
        Dict mapping tensor names to shard filenames
    """
    with open(index_path) as f:
        index = json.load(f)
    return index.get("weight_map", {})


def _parse_quantization_index(index_path: Path) -> dict[str, dict]:
    """Parse CUDA quantizer's quantization_index.json for bits/MSE metadata.

    Args:
        index_path: Path to quantization_index.json

    Returns:
        Dict mapping tensor names to {"bits": int, "mse": float, "shape": tuple}
    """
    if not index_path.exists():
        return {}

    with open(index_path) as f:
        index = json.load(f)

    # Build lookup from layers list
    tensor_info = {}
    for layer in index.get("layers", []):
        name = layer["name"]
        tensor_info[name] = {
            "bits": layer["bits"],
            "mse": layer.get("mse", 0.0),
            "shape": tuple(layer["shape"]),
        }
    return tensor_info


class TrellisModelLoader:
    """Loader for Trellis-quantized models.

    Models are structured with layers in separate directories:
    ```
    model_path/
        layer_0000/index.json, tensor_*.safetensors
        layer_0001/index.json, tensor_*.safetensors
        ...
    ```

    Supports both formats:
    - trellis_v2: Packed uint8 indices (new, ~5x smaller)
    - legacy: Unpacked int16 indices (backward compatible)

    Example:
        >>> loader = TrellisModelLoader("models/GLM-4.7-Flash-Trellis-3bpw")
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

        # Detect storage format
        self._format: str = detect_trellis_format(self.model_path)

        if self._format == "v3_hf_shards":
            self._init_hf_shards()
        elif self._format == "v2_layers":
            self._init_layer_dirs()
        else:
            raise FileNotFoundError(
                f"Unrecognized model format in {model_path}. "
                "Expected model.safetensors.index.json (v3) or layer_*/ directories (v2)."
            )

    def _init_hf_shards(self) -> None:
        """Initialize loader for HF-style sharded safetensors (v3 format)."""
        # Parse weight map
        index_path = self.model_path / "model.safetensors.index.json"
        self._hf_weight_map = _parse_hf_weight_map(index_path)

        # Parse quantization metadata
        quant_index_path = self.model_path / "quantization_index.json"
        self._quant_info = _parse_quantization_index(quant_index_path)

        # Extract layer indices from tensor names
        layer_set: set[int] = set()
        for tensor_name in self._hf_weight_map.keys():
            # Pattern: model.layers.{N}.* or model__layers__{N}__*
            if "layers" in tensor_name:
                parts = tensor_name.replace("__", ".").split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            layer_set.add(layer_idx)
                        except ValueError:
                            pass

        self._layer_indices = sorted(layer_set)

        if not self._layer_indices:
            raise FileNotFoundError(
                f"No layer tensors found in {self.model_path}. "
                "Weight map doesn't contain model.layers.* tensors."
            )

        # Cache for layer metadata (compatibility with v2 API)
        self._layer_metadata: dict[int, dict[str, Any]] = {}

    def _init_layer_dirs(self) -> None:
        """Initialize loader for layer-based directory structure (v2 format)."""
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
            raise FileNotFoundError(
                f"No layer directories found in {self.model_path}")

        # Cache for layer metadata
        self._layer_metadata: dict[int, dict[str, Any]] = {}

        # v2 doesn't use these
        self._hf_weight_map: dict[str, str] = {}
        self._quant_info: dict[str, dict] = {}

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

    def _get_base_weight_name(self, tensor_name: str) -> str:
        """Extract base weight name from tensor component name.

        Handles both formats:
        - metal_marlin: "weight__indices" -> "weight"
        - exllamav3: "weight.indices" or "weight.scale" -> "weight"
        """
        # Remove component suffixes
        for suffix in [
            "__indices",
            "__scales",
            "__su",
            "__sv",
            ".indices",
            ".scales",
            ".scale",
            ".zero_point",
            ".su",
            ".sv",
            ".row_scale",
            ".col_scale",
            ".qweight",
        ]:
            if tensor_name.endswith(suffix):
                return tensor_name[: -len(suffix)]
        return tensor_name

    def _get_component_name(self, tensor_name: str) -> str:
        """Extract component type from tensor name.

        Returns normalized component name: "indices", "scales", "su", "sv"
        """
        # metal_marlin format
        if tensor_name.endswith("__indices"):
            return "indices"
        if tensor_name.endswith("__scales"):
            return "scales"
        if tensor_name.endswith("__su"):
            return "su"
        if tensor_name.endswith("__sv"):
            return "sv"

        # exllamav3 format
        if ".indices" in tensor_name or ".qweight" in tensor_name:
            return "indices"
        if tensor_name.endswith(".scales") or tensor_name.endswith(".scale"):
            return "scales"
        if tensor_name.endswith(".su") or tensor_name.endswith(".row_scale"):
            return "su"
        if tensor_name.endswith(".sv") or tensor_name.endswith(".col_scale"):
            return "sv"

        # Default: return last part
        parts = tensor_name.replace("__", ".").split(".")
        return parts[-1] if parts else "unknown"

    def _infer_bits(self, indices_tensor: torch.Tensor) -> int:
        """Infer quantization bits from indices tensor shape/dtype.

        For trellis quantization, indices are typically int16 with
        values encoding 4-8 bit quantization.
        """
        # Default to 4 bits if we can't determine
        # In practice, this should come from metadata
        return 4

    def _infer_shape(self, components: dict) -> tuple[int, int]:
        """Infer original weight shape from component tensors.

        Uses su shape for K and sv shape for N, or falls back to
        indices shape computation.
        """
        if "su" in components:
            K = components["su"].shape[0]
        elif "row_scale" in components:
            K = components["row_scale"].shape[0]
        else:
            K = 0

        if "sv" in components:
            N = components["sv"].shape[0]
        elif "col_scale" in components:
            N = components["col_scale"].shape[0]
        else:
            N = 0

        return (K, N)

    def _tensors_to_weights(
        self,
        all_tensors: dict[str, torch.Tensor],
        layer_idx: int,
    ) -> dict[str, TrellisWeight]:
        """Convert loaded tensors to TrellisWeight objects.

        Groups __indices, __scales, __su, __sv components into TrellisWeight.
        """
        weights: dict[str, TrellisWeight] = {}

        # Group tensors by base weight name
        weight_components: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in all_tensors.items():
            # Extract base name and component
            for suffix in ["__indices", "__scales", "__su", "__sv"]:
                if name.endswith(suffix):
                    base_name = name[: -len(suffix)]
                    component = suffix[2:]  # Remove "__"
                    weight_components.setdefault(base_name, {})[
                        component] = tensor
                    break

        # Build TrellisWeight for each base weight
        for base_name, components in weight_components.items():
            if "indices" not in components:
                continue

            indices = components["indices"]
            scales = components.get("scales")
            su = components.get("su")
            sv = components.get("sv")

            if scales is None or su is None or sv is None:
                continue

            # Get metadata from quantization_index.json
            # Convert underscore format to dot format
            orig_name = base_name.replace("__", ".")
            # Only add .weight suffix if not already present
            if not orig_name.endswith(".weight"):
                orig_name = orig_name + ".weight"
            meta = self._quant_info.get(orig_name, {})
            bits = meta.get("bits", 4)  # Default to 4-bit
            shape = meta.get("shape", (su.shape[0], sv.shape[0]))

            # Handle packed format (uint8 with header byte)
            if indices.dtype == torch.uint8:
                # Extract bits from header if present
                if indices.numel() > 0:
                    header_bits = int(indices.flatten()[0].item())
                    if 2 <= header_bits <= 8:
                        bits = header_bits

                # Calculate expected shape and reshape
                K, N = shape
                tiles_k = (K + 15) // 16
                tiles_n = (N + 15) // 16
                packed_bytes_per_tile = (256 * bits + 7) // 8

                # Strip header and reshape
                packed_data = indices.flatten()[1:]
                expected_size = tiles_k * tiles_n * packed_bytes_per_tile
                if packed_data.numel() == expected_size:
                    packed_indices = packed_data.reshape(
                        tiles_k, tiles_n, packed_bytes_per_tile)
                else:
                    # Already reshaped or different format
                    packed_indices = indices
            else:
                packed_indices = indices

            weights[orig_name] = TrellisWeight(
                packed_indices=packed_indices,
                scales=scales,
                su=su,
                sv=sv,
                bits=bits,
                original_shape=tuple(shape),
            )

        return weights

    def _find_tensor_files(self, layer_idx: int) -> list[Path]:
        """Find all safetensor files for a layer.

        Supports both naming conventions:
        - tensor_*.safetensors (from quantize_single_layer.py)
        - batch_*.safetensors (from quantize_layerwise_parallel.py)
        """
        layer_path = self._get_layer_path(layer_idx)
        # Try tensor_* first (single-layer quantizer)
        tensor_files = sorted(layer_path.glob("tensor_*.safetensors"))
        if not tensor_files:
            # Fall back to batch_* (parallel quantizer)
            tensor_files = sorted(layer_path.glob("batch_*.safetensors"))
        return tensor_files

    def _load_layer_hf_shards(self, layer_idx: int) -> dict[str, TrellisWeight]:
        """Load layer from HF-style sharded safetensors (v3 format)."""
        prefix = f"model.layers.{layer_idx}."
        alt_prefix = f"model__layers__{layer_idx}__"

        # Find all tensors for this layer
        layer_tensors: dict[str, str] = {}  # tensor_name -> shard_file
        for name, shard in self._hf_weight_map.items():
            if name.startswith(prefix) or name.startswith(alt_prefix):
                layer_tensors[name] = shard

        if not layer_tensors:
            raise FileNotFoundError(f"No tensors found for layer {layer_idx}")

        # Group by shard file for efficient loading
        shards_to_load: dict[str, list[str]] = {}
        for name, shard in layer_tensors.items():
            shards_to_load.setdefault(shard, []).append(name)

        # Load tensors
        all_tensors: dict[str, torch.Tensor] = {}
        for shard_file, tensor_names in shards_to_load.items():
            shard_path = self.model_path / shard_file
            with MmapSafetensorsLoader(shard_path) as loader:
                for name in tensor_names:
                    if name in loader.keys():
                        all_tensors[name] = loader.get_tensor(name)

        # Convert to TrellisWeight objects
        return self._tensors_to_weights(all_tensors, layer_idx)

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
        if self._format == "v3_hf_shards":
            return self._load_layer_hf_shards(layer_idx)

        metadata = self._load_layer_metadata(layer_idx)
        tensor_files = self._find_tensor_files(layer_idx)

        if not tensor_files:
            raise FileNotFoundError(
                f"No tensor files found for layer {layer_idx}")

        # Load all tensors from all files using mmap
        all_tensors: dict[str, torch.Tensor] = {}
        for tensor_file in tensor_files:
            with MmapSafetensorsLoader(tensor_file) as loader:
                for name in loader.keys():
                    all_tensors[name] = loader.get_tensor(name)

        return self._load_metal_marlin_format(all_tensors, metadata)

    def _load_metal_marlin_format(
        self,
        all_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> dict[str, TrellisWeight]:
        """Load weights in metal_marlin native format.

        Supports both packed (uint8) and unpacked (int16) indices:
        - uint8: trellis_v2 format, ~5x smaller, first byte is bits value
        - int16: legacy format, indices stored directly
        """
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
                raise ValueError(
                    f"Missing indices for tensor: {name} (key: {indices_key})")
            if scales_key not in all_tensors:
                raise ValueError(
                    f"Missing scales for tensor: {name} (key: {scales_key})")
            if su_key not in all_tensors:
                raise ValueError(
                    f"Missing su for tensor: {name} (key: {su_key})")
            if sv_key not in all_tensors:
                raise ValueError(
                    f"Missing sv for tensor: {name} (key: {sv_key})")

            indices_raw = all_tensors[indices_key]
            scales = all_tensors[scales_key]
            su = all_tensors[su_key]
            sv = all_tensors[sv_key]

            # Handle packed vs unpacked indices
            if indices_raw.dtype == torch.uint8:
                # Packed format (trellis_v2): keep as uint8, don't unpack
                # Format: [1 byte header (bits value)] + [packed data]
                # Packed data is stored tile-contiguous without per-tile headers
                K, N = shape
                tiles_k = (K + 15) // 16
                tiles_n = (N + 15) // 16
                # Packed bytes per tile = ceil(256 * bits / 8)
                packed_bytes_per_tile = (256 * bits + 7) // 8
                # Total expected size: 1 header byte + tiles_k * tiles_n * packed_bytes_per_tile
                expected_data_bytes = tiles_k * tiles_n * packed_bytes_per_tile
                expected_total_bytes = 1 + expected_data_bytes

                if indices_raw.numel() != expected_total_bytes:
                    raise ValueError(
                        f"Packed indices size mismatch for {name}: "
                        f"expected {expected_total_bytes} (1 header + {expected_data_bytes} data), "
                        f"got {indices_raw.numel()}"
                    )

                # Strip header byte and reshape to [tiles_K, tiles_N, packed_bytes_per_tile]
                # TrellisWeight convention: K=out_features, N=in_features
                # This is stored as-is; model.py transposes when stacking for MoE kernel
                packed_indices = indices_raw[1:].reshape(
                    tiles_k, tiles_n, packed_bytes_per_tile)
            elif indices_raw.dtype == torch.int16:
                # Legacy format: not supported for memory-efficient loading
                raise ValueError(
                    f"Legacy int16 format not supported. Please use packed uint8 format for {name}"
                )
            else:
                raise ValueError(
                    f"Indices must be uint8 (packed), got {indices_raw.dtype} for {name}"
                )

            # Validate other dtypes
            if scales.dtype != torch.float32:
                raise ValueError(
                    f"Scales must be float32, got {scales.dtype} for {name}")
            if su.dtype != torch.float32:
                raise ValueError(
                    f"su must be float32, got {su.dtype} for {name}")
            if sv.dtype != torch.float32:
                raise ValueError(
                    f"sv must be float32, got {sv.dtype} for {name}")

            weights[name] = TrellisWeight(
                packed_indices=packed_indices,
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
            first_layer_meta = self._load_layer_metadata(
                self._layer_indices[0])
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

    def load_router_weights(self) -> dict[str, torch.Tensor]:
        """Load router weights for MoE layers.

        Router weights are typically stored separately from layer quantized weights
        since they are not quantized (for accuracy in expert selection).

        Uses memory-mapped loading to avoid loading weights into RAM upfront.

        Returns:
            Dictionary mapping router weight names to tensors.
        """
        router_path = self.model_path / "router_weights.safetensors"
        if router_path.exists():
            # Use mmap for lazy loading - weights are paged in by OS on demand
            loader = MmapSafetensorsLoader(router_path, device="cpu")
            weights = {name: loader.get_tensor(name) for name in loader.keys()}
            # Keep loader reference to prevent garbage collection during model load
            weights["_mmap_loader"] = loader  # type: ignore[dict-item]
            return weights

        # Fall back to checking in base weights using mmap
        base_path = self.model_path / "base_weights.safetensors"
        if base_path.exists():
            loader = MmapSafetensorsLoader(base_path, device="cpu")
            # Filter router weights
            router_weights = {}
            for name in loader.keys():
                if "mlp.gate.weight" in name:
                    router_weights[name] = loader.get_tensor(name)
            # Keep loader reference to prevent garbage collection during model load
            router_weights["_mmap_loader"] = loader  # type: ignore[dict-item]
            return router_weights

        raise FileNotFoundError(
            f"Router weights not found in {self.model_path}. "
            "Expected router_weights.safetensors or in base_weights.safetensors"
        )

    def load_base_weights(self, patterns: list[str] | None = None) -> dict[str, torch.Tensor]:
        """Load non-quantized base weights (embeddings, layernorms, etc.).

        Uses memory-mapped loading to avoid loading weights into RAM upfront.

        Args:
            patterns: Optional list of patterns to filter weights.
                     If None, returns all base weights.

        Returns:
            Dictionary mapping weight names to tensors.
        """
        if self._format == "v3_hf_shards":
            return self._load_base_weights_hf_shards(patterns)

        base_path = self.model_path / "base_weights.safetensors"
        if not base_path.exists():
            raise FileNotFoundError(f"Base weights not found: {base_path}")

        # Use mmap for lazy loading - weights are paged in by OS on demand
        loader = MmapSafetensorsLoader(base_path, device="cpu")

        if patterns is None:
            weights = {name: loader.get_tensor(name) for name in loader.keys()}
            # Keep loader reference to prevent garbage collection during model load
            weights["_mmap_loader"] = loader  # type: ignore[dict-item]
            return weights

        filtered = {}
        for name in loader.keys():
            if any(p in name for p in patterns):
                filtered[name] = loader.get_tensor(name)
        # Keep loader reference to prevent garbage collection during model load
        filtered["_mmap_loader"] = loader  # type: ignore[dict-item]
        return filtered

    def _load_base_weights_hf_shards(
        self, patterns: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Load base weights from HF-style shards."""
        # First try dedicated base_weights.safetensors
        base_path = self.model_path / "base_weights.safetensors"
        if base_path.exists():
            loader = MmapSafetensorsLoader(base_path, device="cpu")
            if patterns is None:
                return {name: loader.get_tensor(name) for name in loader.keys()}
            return {
                name: loader.get_tensor(name)
                for name in loader.keys()
                if any(p in name for p in patterns)
            }

        # Fall back to loading from shards
        base_patterns = patterns or [
            "embed_tokens",
            "norm.weight",
            "lm_head",
            "input_layernorm",
            "post_attention_layernorm",
        ]

        results: dict[str, torch.Tensor] = {}
        shards_to_check: set[str] = set()

        for name, shard in self._hf_weight_map.items():
            if any(p in name for p in base_patterns):
                shards_to_check.add(shard)

        for shard_file in shards_to_check:
            shard_path = self.model_path / shard_file
            with MmapSafetensorsLoader(shard_path) as loader:
                for name in loader.keys():
                    if any(p in name for p in base_patterns):
                        results[name] = loader.get_tensor(name)

        return results

    def load_layernorm_weights(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Load layernorm weights for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Dictionary with layernorm weights:
            - input_layernorm.weight
            - post_attention_layernorm.weight
            - self_attn.q_a_layernorm.weight (if present)
            - self_attn.kv_a_layernorm.weight (if present)
        """
        prefix = f"model.layers.{layer_idx}."
        patterns = [
            "input_layernorm",
            "post_attention_layernorm",
            "q_a_layernorm",
            "kv_a_layernorm",
        ]

        try:
            base_weights = self.load_base_weights(patterns)
        except FileNotFoundError:
            return {}

        # Filter to this layer
        layer_weights = {}
        for name, tensor in base_weights.items():
            if name.startswith(prefix):
                # Strip prefix to get relative name
                rel_name = name[len(prefix):]
                layer_weights[rel_name] = tensor

        return layer_weights

    def load_weight(self, layer_idx: int, tensor_name: str) -> TrellisWeight:
        """Load a single weight tensor from a layer.

        Args:
            layer_idx: Layer index to load from.
            tensor_name: Name of the tensor. Can be:
                - Relative name: 'mlp.gate_proj' -> expands to 'model.layers.{layer_idx}.mlp.gate_proj.weight'
                - Full name: 'model.layers.0.mlp.gate_proj.weight'

        Returns:
            TrellisWeight object for the requested tensor.

        Raises:
            FileNotFoundError: If the layer doesn't exist.
            ValueError: If the tensor is not found in the layer.
        """
        layer_weights = self.load_layer(layer_idx)

        # Try the name as-is first
        if tensor_name in layer_weights:
            return layer_weights[tensor_name]

        # Try with .weight suffix if not present
        if not tensor_name.endswith(".weight"):
            full_name = f"{tensor_name}.weight"
            if full_name in layer_weights:
                return layer_weights[full_name]

        # Try expanding relative name to full name
        if not tensor_name.startswith("model.layers."):
            full_name = f"model.layers.{layer_idx}.{tensor_name}.weight"
            if full_name in layer_weights:
                return layer_weights[full_name]

        raise ValueError(
            f"Tensor '{tensor_name}' not found in layer {layer_idx}. "
            f"Available tensors: {list(layer_weights.keys())}"
        )

    def clear_layer_cache(self, layer_idx: int) -> None:
        """Clear cached metadata for a layer to save memory.

        Args:
            layer_idx: Layer index to clear from cache.
        """
        if layer_idx in self._layer_metadata:
            del self._layer_metadata[layer_idx]
