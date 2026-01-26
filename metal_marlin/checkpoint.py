"""Model checkpoint loading utilities for inference on Apple Silicon.

Provides utilities for loading quantized model weights from safetensors files
using PyTorch. Supports sharded checkpoints, weight mapping, and lazy loading
for memory-efficient model initialization.

This module REQUIRES PyTorch to be installed. It uses safetensors for fast,
memory-mapped weight loading which is particularly efficient on systems with
Unified Memory.

Usage:
    # Load a single checkpoint file
    weights = load_checkpoint("model.safetensors")

    # Load sharded checkpoint with index
    weights = load_sharded_checkpoint("model.safetensors.index.json")

    # Use lazy loading for large models
    with LazyCheckpoint("model.safetensors") as ckpt:
        weight = ckpt.get_tensor("model.layers.0.self_attn.q_proj.weight")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._compat import require_torch

if TYPE_CHECKING:
    import torch

# Import safetensors
try:
    from safetensors import safe_open
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file as safetensors_save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def require_safetensors(feature: str = "this operation") -> None:
    """Raise RuntimeError if safetensors is not available.

    Args:
        feature: Description of what requires safetensors for the error message.

    Raises:
        RuntimeError: If safetensors is not installed.
    """
    if not HAS_SAFETENSORS:
        raise RuntimeError(
            f"safetensors is required for {feature}. Install with: pip install safetensors"
        )


@dataclass
class CheckpointMetadata:
    """Metadata about a loaded checkpoint.

    Args:
        format: Checkpoint format ('safetensors', 'pytorch').
        num_shards: Number of shard files (1 for single file).
        total_size_bytes: Total size of all weights in bytes.
        dtype: Primary dtype of weights.
        keys: List of all weight keys.
        metadata: Additional metadata from the checkpoint file.
    """

    format: str
    num_shards: int = 1
    total_size_bytes: int = 0
    dtype: str = "float16"
    keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_parameters(self) -> int:
        """Estimate number of parameters based on size and dtype."""
        bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
        return int(self.total_size_bytes / bytes_per_param.get(self.dtype, 2))


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Load a safetensors checkpoint file.

    Loads all tensors from a safetensors file into a dictionary. For sharded
    checkpoints, use load_sharded_checkpoint() instead.

    Args:
        path: Path to the safetensors file.
        device: Device to load tensors to ('cpu', 'mps', 'cuda').
        dtype: Optional dtype to convert tensors to.

    Returns:
        Dictionary mapping weight names to tensors.

    Raises:
        RuntimeError: If PyTorch or safetensors is not installed.
        FileNotFoundError: If the checkpoint file doesn't exist.
    """
    require_torch("load_checkpoint")
    require_safetensors("load_checkpoint")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    weights = safetensors_load_file(str(path), device=device)

    if dtype is not None:
        weights = {k: v.to(dtype) for k, v in weights.items()}

    return weights


def load_sharded_checkpoint(
    index_path: str | Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    weights_only: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Load a sharded safetensors checkpoint.

    Sharded checkpoints consist of multiple .safetensors files and an index
    JSON file that maps weight names to shard files.

    Args:
        index_path: Path to the index JSON file (e.g., model.safetensors.index.json).
        device: Device to load tensors to.
        dtype: Optional dtype to convert tensors to.
        weights_only: If provided, only load these specific weight names.
            Useful for loading specific layers without loading the entire model.

    Returns:
        Dictionary mapping weight names to tensors.

    Raises:
        RuntimeError: If PyTorch or safetensors is not installed.
        FileNotFoundError: If the index file or any shard doesn't exist.
        ValueError: If the index file format is invalid.
    """
    require_torch("load_sharded_checkpoint")
    require_safetensors("load_sharded_checkpoint")

    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    if "weight_map" not in index:
        raise ValueError(f"Invalid index file format: missing 'weight_map' in {index_path}")

    weight_map: dict[str, str] = index["weight_map"]
    base_dir = index_path.parent

    # Group weights by shard file
    shard_to_weights: dict[str, list[str]] = {}
    for weight_name, shard_name in weight_map.items():
        if weights_only is not None and weight_name not in weights_only:
            continue
        if shard_name not in shard_to_weights:
            shard_to_weights[shard_name] = []
        shard_to_weights[shard_name].append(weight_name)

    # Load weights from each shard
    weights: dict[str, torch.Tensor] = {}
    for shard_name, shard_weights in shard_to_weights.items():
        shard_path = base_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        # Use safe_open for selective loading
        with safe_open(str(shard_path), framework="pt", device=device) as f:
            for weight_name in shard_weights:
                tensor = f.get_tensor(weight_name)
                if dtype is not None:
                    tensor = tensor.to(dtype)
                weights[weight_name] = tensor

    return weights


def get_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    """Get metadata about a checkpoint without loading weights.

    Works with both single files and sharded checkpoints. For sharded
    checkpoints, pass the index file path.

    Args:
        path: Path to safetensors file or index JSON.

    Returns:
        CheckpointMetadata with information about the checkpoint.

    Raises:
        RuntimeError: If safetensors is not installed.
        FileNotFoundError: If the file doesn't exist.
    """
    require_safetensors("get_checkpoint_metadata")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Check if this is an index file
    if path.suffix == ".json" or path.name.endswith(".index.json"):
        return _get_sharded_metadata(path)

    # Single file
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata_dict = dict(f.metadata()) if f.metadata() else {}

        total_size = 0
        dtype_counts: dict[str, int] = {}

        for key in keys:
            # Get tensor info without loading
            tensor = f.get_tensor(key)
            tensor_size = tensor.numel() * tensor.element_size()
            total_size += tensor_size

            dtype_str = str(tensor.dtype).replace("torch.", "")
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

        # Most common dtype
        primary_dtype = (
            max(dtype_counts.keys(), key=lambda d: dtype_counts[d]) if dtype_counts else "float16"
        )

    return CheckpointMetadata(
        format="safetensors",
        num_shards=1,
        total_size_bytes=total_size,
        dtype=primary_dtype,
        keys=keys,
        metadata=metadata_dict,
    )


def _get_sharded_metadata(index_path: Path) -> CheckpointMetadata:
    """Get metadata for a sharded checkpoint."""
    with open(index_path) as f:
        index = json.load(f)

    weight_map: dict[str, str] = index.get("weight_map", {})
    keys = list(weight_map.keys())
    shards = set(weight_map.values())

    # Get total size from index if available
    total_size = index.get("metadata", {}).get("total_size", 0)

    return CheckpointMetadata(
        format="safetensors",
        num_shards=len(shards),
        total_size_bytes=total_size,
        dtype=index.get("metadata", {}).get("dtype", "float16"),
        keys=keys,
        metadata=index.get("metadata", {}),
    )


class LazyCheckpoint:
    """Context manager for lazy/streaming checkpoint loading.

    Provides memory-efficient access to checkpoint weights by loading tensors
    on-demand rather than all at once. Uses memory-mapped I/O when possible.

    This is particularly useful for:
    - Loading specific layers without loading the entire model
    - Iterating over weights for analysis
    - Systems with limited memory

    Args:
        path: Path to safetensors file or index JSON for sharded checkpoints.
        device: Default device for loaded tensors.

    Example:
        with LazyCheckpoint("model.safetensors") as ckpt:
            # Load only what you need
            q_proj = ckpt.get_tensor("model.layers.0.self_attn.q_proj.weight")

            # Iterate over all keys
            for key in ckpt.keys():
                if "embed" in key:
                    tensor = ckpt.get_tensor(key)
    """

    def __init__(self, path: str | Path, device: str = "cpu"):
        require_torch("LazyCheckpoint")
        require_safetensors("LazyCheckpoint")

        self.path = Path(path)
        self.device = device
        self._handles: dict[str, Any] = {}
        self._weight_to_shard: dict[str, str] = {}
        self._is_sharded = False

        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")

    def __enter__(self) -> LazyCheckpoint:
        self._open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._close()

    def _open(self) -> None:
        """Open checkpoint file(s) for reading."""
        if self.path.suffix == ".json" or self.path.name.endswith(".index.json"):
            self._open_sharded()
        else:
            self._open_single()

    def _open_single(self) -> None:
        """Open a single safetensors file."""
        handle = safe_open(str(self.path), framework="pt", device=self.device)
        self._handles["main"] = handle
        self._weight_to_shard = {k: "main" for k in handle.keys()}
        self._is_sharded = False

    def _open_sharded(self) -> None:
        """Open sharded checkpoint files."""
        with open(self.path) as f:
            index = json.load(f)

        weight_map: dict[str, str] = index.get("weight_map", {})
        base_dir = self.path.parent

        # Open each shard file
        shards = set(weight_map.values())
        for shard_name in shards:
            shard_path = base_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            self._handles[shard_name] = safe_open(
                str(shard_path), framework="pt", device=self.device
            )

        self._weight_to_shard = weight_map
        self._is_sharded = True

    def _close(self) -> None:
        """Close all open file handles."""
        self._handles.clear()
        self._weight_to_shard.clear()

    def keys(self) -> list[str]:
        """Get all weight names in the checkpoint."""
        return list(self._weight_to_shard.keys())

    def __contains__(self, key: str) -> bool:
        """Check if a weight name exists in the checkpoint."""
        return key in self._weight_to_shard

    def get_tensor(self, key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Load a single tensor by name.

        Args:
            key: Weight name.
            dtype: Optional dtype to convert to.

        Returns:
            The loaded tensor.

        Raises:
            KeyError: If the weight name doesn't exist.
        """

        if key not in self._weight_to_shard:
            raise KeyError(f"Weight not found: {key}")

        shard_name = self._weight_to_shard[key]
        handle = self._handles[shard_name]
        tensor = handle.get_tensor(key)

        if dtype is not None:
            tensor = tensor.to(dtype)

        return tensor

    def get_tensors(
        self, keys: list[str], dtype: torch.dtype | None = None
    ) -> dict[str, torch.Tensor]:
        """Load multiple tensors by name.

        Args:
            keys: List of weight names.
            dtype: Optional dtype to convert to.

        Returns:
            Dictionary mapping names to tensors.
        """
        return {k: self.get_tensor(k, dtype=dtype) for k in keys}


def save_checkpoint(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save tensors to a safetensors checkpoint.

    Args:
        tensors: Dictionary mapping names to tensors.
        path: Output path.
        metadata: Optional string metadata to include.

    Raises:
        RuntimeError: If PyTorch or safetensors is not installed.
    """
    require_torch("save_checkpoint")
    require_safetensors("save_checkpoint")

    safetensors_save_file(tensors, str(path), metadata=metadata)


def remap_weight_keys(
    weights: dict[str, torch.Tensor],
    mapping: dict[str, str] | None = None,
    prefix_map: dict[str, str] | None = None,
    remove_prefixes: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Remap weight dictionary keys for model compatibility.

    Useful for loading weights from different model implementations
    that use different naming conventions.

    Args:
        weights: Original weight dictionary.
        mapping: Explicit key mapping {old_key: new_key}.
        prefix_map: Prefix replacement {old_prefix: new_prefix}.
        remove_prefixes: List of prefixes to strip from keys.

    Returns:
        New dictionary with remapped keys.

    Example:
        # Remap HuggingFace keys to our format
        weights = remap_weight_keys(
            weights,
            prefix_map={"model.": "transformer."},
            remove_prefixes=["base_model."],
        )
    """
    result: dict[str, torch.Tensor] = {}

    for key, tensor in weights.items():
        new_key = key

        # Apply explicit mapping first
        if mapping and key in mapping:
            new_key = mapping[key]
        else:
            # Remove prefixes
            if remove_prefixes:
                for prefix in remove_prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                        break

            # Apply prefix mapping
            if prefix_map:
                for old_prefix, new_prefix in prefix_map.items():
                    if new_key.startswith(old_prefix):
                        new_key = new_prefix + new_key[len(old_prefix) :]
                        break

        result[new_key] = tensor

    return result


def filter_weights(
    weights: dict[str, torch.Tensor],
    patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Filter weights by key patterns.

    Args:
        weights: Weight dictionary to filter.
        patterns: If provided, only include keys matching these regex patterns.
        exclude_patterns: Exclude keys matching these regex patterns.

    Returns:
        Filtered weight dictionary.

    Example:
        # Get only attention weights
        attn_weights = filter_weights(weights, patterns=[r".*self_attn.*"])

        # Exclude embedding layers
        no_embed = filter_weights(weights, exclude_patterns=[r".*embed.*"])
    """
    result: dict[str, torch.Tensor] = {}

    for key, tensor in weights.items():
        # Check include patterns
        if patterns:
            if not any(re.match(p, key) for p in patterns):
                continue

        # Check exclude patterns
        if exclude_patterns:
            if any(re.match(p, key) for p in exclude_patterns):
                continue

        result[key] = tensor

    return result


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint loading and processing.

    Args:
        device: Target device for loaded tensors.
        dtype: Target dtype for conversion (None to keep original).
        lazy_load: Whether to use lazy loading for large checkpoints.
        weight_patterns: Patterns to filter which weights to load.
        key_mapping: Explicit key remapping.
        prefix_map: Prefix replacement mapping.
    """

    device: str = "cpu"
    dtype: str | None = None
    lazy_load: bool = False
    weight_patterns: list[str] | None = None
    key_mapping: dict[str, str] | None = None
    prefix_map: dict[str, str] | None = None


def load_model_checkpoint(
    path: str | Path,
    config: CheckpointConfig | None = None,
) -> dict[str, torch.Tensor]:
    """High-level checkpoint loading with configuration.

    Combines loading, filtering, and remapping in a single call.

    Args:
        path: Path to checkpoint file or index JSON.
        config: Loading configuration. Uses defaults if not provided.

    Returns:
        Processed weight dictionary.

    Example:
        config = CheckpointConfig(
            device="mps",
            dtype="float16",
            prefix_map={"model.": ""},
        )
        weights = load_model_checkpoint("model.safetensors", config)
    """
    require_torch("load_model_checkpoint")

    import torch

    if config is None:
        config = CheckpointConfig()

    # Determine dtype
    dtype = None
    if config.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(config.dtype)

    path = Path(path)

    # Determine if sharded
    is_sharded = path.suffix == ".json" or path.name.endswith(".index.json")

    # Load weights
    if is_sharded:
        weights = load_sharded_checkpoint(
            path,
            device=config.device,
            dtype=dtype,
            weights_only=None,  # Load all, filter later
        )
    else:
        weights = load_checkpoint(path, device=config.device, dtype=dtype)

    # Filter by patterns
    if config.weight_patterns:
        weights = filter_weights(weights, patterns=config.weight_patterns)

    # Remap keys
    if config.key_mapping or config.prefix_map:
        weights = remap_weight_keys(
            weights,
            mapping=config.key_mapping,
            prefix_map=config.prefix_map,
        )

    return weights
