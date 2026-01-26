"""Mixed FP4/FP8/FP16 model format support.

Enables packing multiple quantization formats within a single model file,
with per-layer format selection stored in safetensors header metadata for
efficient runtime dispatch.

Example usage:

    from metal_marlin.packing import pack_mixed_format_model, load_mixed_format_model
    from metal_marlin.mixed_precision import Precision

    # Define per-layer precision
    precision_map = {
        "model.embed_tokens": (Precision.FP16, 0),  # keep embedding full precision
        "model.layers.0.self_attn.q_proj": (Precision.FP4_E2M1, 64),
        "model.layers.0.mlp.gate_proj": (Precision.FP8_E4M3, 128),
        # ... etc
    }

    # Pack to single file
    header = pack_mixed_format_model(weights, precision_map, Path("model.mixed.safetensors"))

    # Load with format metadata
    weights, header = load_mixed_format_model(Path("model.mixed.safetensors"))
    print(header.layer_formats)  # {"model.embed_tokens": "fp16", ...}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import to_numpy
from ..mixed_precision import Precision

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Metadata key prefix for safetensors header
# Using __metal_marlin__ namespace to avoid collisions
_HEADER_KEY = "__metal_marlin_mixed_format__"


@dataclass
class MixedFormatHeader:
    """Header describing a mixed-format quantized model.

    Stored as JSON in the safetensors metadata header for runtime dispatch.
    The header enables the inference engine to select the correct dequantization
    kernel for each layer without inspecting tensor shapes.

    Attributes:
        layer_formats: Mapping from layer name (without suffix) to format string.
            Format strings: "fp4", "fp8", "int4", "int8", "fp16", "bf16".
        layer_group_sizes: Mapping from layer name to quantization group size.
            Only present for quantized layers (not fp16/bf16).
        total_params: Total parameter count across all layers.
        quantized_params: Number of parameters stored in quantized format.
        average_bits: Weighted average bits per parameter.
        version: Format version for forward compatibility.
    """

    layer_formats: dict[str, str] = field(default_factory=dict)
    layer_group_sizes: dict[str, int] = field(default_factory=dict)
    total_params: int = 0
    quantized_params: int = 0
    average_bits: float = 16.0
    version: int = 1

    def to_json(self) -> str:
        """Serialize header to JSON string for safetensors metadata."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, json_str: str) -> MixedFormatHeader:
        """Deserialize header from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def get_format(self, layer_name: str) -> str | None:
        """Get format for a layer, handling .packed/.scales suffixes.

        Args:
            layer_name: Full tensor name (may include .packed, .scales, etc.)

        Returns:
            Format string or None if layer not found.
        """
        # Strip common suffixes to get base name
        base_name = layer_name
        for suffix in (".packed", ".scales", ".zeros", ".weight", ".bias"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        # Try exact match first, then base name
        if layer_name in self.layer_formats:
            return self.layer_formats[layer_name]
        if base_name in self.layer_formats:
            return self.layer_formats[base_name]
        return None

    def get_group_size(self, layer_name: str) -> int | None:
        """Get group size for a layer."""
        base_name = layer_name
        for suffix in (".packed", ".scales", ".zeros", ".weight", ".bias"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        if layer_name in self.layer_group_sizes:
            return self.layer_group_sizes[layer_name]
        if base_name in self.layer_group_sizes:
            return self.layer_group_sizes[base_name]
        return None

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Mixed Format Model (v{self.version})",
            f"  Total params: {self.total_params / 1e9:.2f}B",
            f"  Quantized:    {self.quantized_params / 1e9:.2f}B "
            f"({100 * self.quantized_params / max(1, self.total_params):.1f}%)",
            f"  Avg bits:     {self.average_bits:.2f}",
        ]

        # Count layers by format
        format_counts: dict[str, int] = {}
        for fmt in self.layer_formats.values():
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        lines.append("  Formats:")
        for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {fmt:6s}: {count} layers")

        return "\n".join(lines)


def _precision_to_format_str(precision: Precision) -> str:
    """Convert Precision enum to format string for header."""
    mapping = {
        Precision.FP16: "fp16",
        Precision.BF16: "bf16",
        Precision.FP8_E4M3: "fp8",
        Precision.FP4_E2M1: "fp4",
        Precision.INT8: "int8",
        Precision.INT4: "int4",
        Precision.INT3: "int3",
        Precision.INT2: "int2",
        Precision.NF3: "nf3",
        Precision.NF2: "nf2",
    }
    return mapping.get(precision, "fp16")


def _format_str_to_precision(fmt: str) -> Precision:
    """Convert format string to Precision enum."""
    mapping = {
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
        "fp8": Precision.FP8_E4M3,
        "fp4": Precision.FP4_E2M1,
        "int8": Precision.INT8,
        "int4": Precision.INT4,
        "int3": Precision.INT3,
        "int2": Precision.INT2,
        "nf3": Precision.NF3,
        "nf2": Precision.NF2,
    }
    return mapping.get(fmt, Precision.FP16)


def _bits_for_precision(precision: Precision) -> int:
    """Return bits per element for a precision level."""
    bits = {
        Precision.FP16: 16,
        Precision.BF16: 16,
        Precision.FP8_E4M3: 8,
        Precision.FP4_E2M1: 4,
        Precision.INT8: 8,
        Precision.INT4: 4,
        Precision.INT3: 3,
        Precision.INT2: 2,
        Precision.NF3: 3,
        Precision.NF2: 2,
    }
    return bits.get(precision, 16)


def _pack_single_layer(
    name: str,
    weight: NDArray[Any],
    precision: Precision,
    group_size: int,
) -> dict[str, NDArray[Any]]:
    """Quantize and pack a single layer according to its precision.

    Args:
        name: Layer name (base name, without .packed suffix)
        weight: Weight tensor as numpy array
        precision: Target precision
        group_size: Quantization group size (ignored for fp16/bf16)

    Returns:
        Dictionary of tensors to save (may include .packed, .scales, etc.)
    """
    # Import quantization functions lazily to avoid circular imports
    from ..mixed_precision import LayerQuantConfig, quantize_tensor

    if precision in (Precision.FP16, Precision.BF16):
        # No quantization, just dtype conversion
        if precision == Precision.FP16:
            return {name: weight.astype(np.float16)}
        else:
            # BF16: store as uint16 with metadata indicating bf16
            # We can't use np.bfloat16 as numpy doesn't support it natively
            # Convert via float32 intermediate
            w32 = weight.astype(np.float32)
            # Truncate mantissa: keep only 7 bits of mantissa (float32 has 23)
            # This is a simplified bf16 conversion
            bf16_bits = (w32.view(np.uint32) >> 16).astype(np.uint16)
            return {name: bf16_bits, f"{name}.__dtype__": np.array([1], dtype=np.uint8)}

    # Create config and quantize
    config = LayerQuantConfig(precision=precision, group_size=group_size)
    result = quantize_tensor(weight, config)

    # Build output tensors with proper naming
    output: dict[str, NDArray[Any]] = {}

    if "packed" in result:
        output[f"{name}.packed"] = result["packed"]
    if "scales" in result:
        output[f"{name}.scales"] = result["scales"]
    if "zeros" in result:
        output[f"{name}.zeros"] = result["zeros"]
    if "data" in result and "packed" not in result:
        # INT8 and similar store in 'data' key
        output[f"{name}.data"] = result["data"]
        if "scales" in result:
            output[f"{name}.scales"] = result["scales"]
        if "zeros" in result:
            output[f"{name}.zeros"] = result["zeros"]

    return output


def pack_mixed_format_model(
    weights: dict[str, NDArray[Any]],
    precision_map: dict[str, tuple[Precision, int]],
    output_path: Path,
) -> MixedFormatHeader:
    """Save model with per-layer precision to safetensors format.

    Each layer is quantized according to its entry in precision_map.
    Layers not in the map are kept at FP16 by default.

    The resulting file contains:
    - Quantized tensors with .packed/.scales/.zeros suffixes
    - FP16/BF16 tensors stored directly
    - JSON metadata in the safetensors header for runtime dispatch

    Args:
        weights: Dictionary mapping layer names to numpy weight arrays.
            Layer names should be the base name (e.g., "model.layers.0.mlp.gate_proj").
        precision_map: Dictionary mapping layer names to (Precision, group_size) tuples.
            Layers not in this map default to FP16 with group_size=0.
        output_path: Output file path. Should end in .safetensors.

    Returns:
        MixedFormatHeader with format metadata.

    Raises:
        ValueError: If weights dictionary is empty.
        ImportError: If safetensors is not installed.

    Example:
        >>> weights = {"layer1": np.random.randn(768, 768).astype(np.float32)}
        >>> precision_map = {"layer1": (Precision.FP4_E2M1, 128)}
        >>> header = pack_mixed_format_model(weights, precision_map, Path("out.safetensors"))
    """
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        raise ImportError(
            "safetensors is required for mixed format models. Install with: pip install safetensors"
        ) from e

    if not weights:
        raise ValueError("weights dictionary cannot be empty")

    # Build output tensors and header
    output_tensors: dict[str, NDArray[Any]] = {}
    header = MixedFormatHeader()

    total_bits = 0

    for name, weight_arr in weights.items():
        # Convert to numpy if needed
        weight = to_numpy(weight_arr)
        param_count = weight.size
        header.total_params += param_count

        # Get precision for this layer
        if name in precision_map:
            precision, group_size = precision_map[name]
        else:
            # Default: keep at fp16
            precision = Precision.FP16
            group_size = 0

        # Track format in header
        format_str = _precision_to_format_str(precision)
        header.layer_formats[name] = format_str

        if group_size > 0:
            header.layer_group_sizes[name] = group_size

        # Track quantized params
        if precision not in (Precision.FP16, Precision.BF16):
            header.quantized_params += param_count

        # Track bits for average calculation
        bits = _bits_for_precision(precision)
        total_bits += param_count * bits

        # Quantize and pack
        packed = _pack_single_layer(name, weight, precision, group_size)
        output_tensors.update(packed)

    # Compute average bits
    if header.total_params > 0:
        header.average_bits = total_bits / header.total_params

    # Serialize header to metadata
    metadata = {_HEADER_KEY: header.to_json()}

    # Save with metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, str(output_path), metadata=metadata)

    return header


def load_mixed_format_model(
    model_path: Path,
) -> tuple[dict[str, NDArray[Any]], MixedFormatHeader]:
    """Load mixed-format model from safetensors file.

    Reads the model file and extracts format metadata from the header.
    Does NOT dequantize the tensors; returns them as stored.

    For inference, use the header.get_format() method to determine
    which dequantization kernel to use for each layer.

    Args:
        model_path: Path to .safetensors file created by pack_mixed_format_model.

    Returns:
        Tuple of (weights dict, MixedFormatHeader).
        The weights dict contains all tensors including .packed/.scales suffixes.

    Raises:
        FileNotFoundError: If model_path doesn't exist.
        ValueError: If file doesn't contain mixed format metadata.
        ImportError: If safetensors is not installed.

    Example:
        >>> weights, header = load_mixed_format_model(Path("model.mixed.safetensors"))
        >>> for name in weights:
        ...     fmt = header.get_format(name)
        ...     print(f"{name}: {fmt}")
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "safetensors is required for mixed format models. Install with: pip install safetensors"
        ) from e

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    weights: dict[str, NDArray[Any]] = {}

    with safe_open(str(model_path), framework="numpy") as f:
        # Extract metadata
        metadata = f.metadata()

        if metadata is None or _HEADER_KEY not in metadata:
            # No mixed format header; create minimal header from tensors
            header = MixedFormatHeader()
            for name in f.keys():
                tensor = f.get_tensor(name)
                weights[name] = tensor
                header.total_params += tensor.size
                # Infer format from tensor name/dtype
                if ".packed" in name:
                    base_name = name.rsplit(".packed", 1)[0]
                    # Could be fp4 or int4; default to fp4
                    header.layer_formats[base_name] = "fp4"
                    header.quantized_params += tensor.size * 8  # 8 values per uint32
                elif ".scales" not in name and ".zeros" not in name:
                    header.layer_formats[name] = "fp16" if tensor.dtype == np.float16 else "fp32"
        else:
            header = MixedFormatHeader.from_json(metadata[_HEADER_KEY])

            # Load all tensors
            for name in f.keys():
                weights[name] = f.get_tensor(name)

    return weights, header


def get_layer_precision(
    header: MixedFormatHeader,
    layer_name: str,
) -> tuple[Precision, int]:
    """Get the precision and group size for a layer.

    Convenience function for inference code to determine how to
    dequantize a layer.

    Args:
        header: MixedFormatHeader from load_mixed_format_model.
        layer_name: Layer name (with or without .packed/.scales suffix).

    Returns:
        Tuple of (Precision enum, group_size). Group size is 0 for fp16/bf16.
    """
    format_str = header.get_format(layer_name)
    if format_str is None:
        return Precision.FP16, 0

    precision = _format_str_to_precision(format_str)
    group_size = header.get_group_size(layer_name) or 0

    return precision, group_size


def merge_mixed_format_models(
    model_paths: list[Path],
    output_path: Path,
) -> MixedFormatHeader:
    """Merge multiple mixed-format model shards into one.

    Useful for combining model shards that were quantized separately.

    Args:
        model_paths: List of paths to model shards.
        output_path: Output path for merged model.

    Returns:
        Combined MixedFormatHeader.
    """
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        raise ImportError("safetensors is required. Install with: pip install safetensors") from e

    combined_weights: dict[str, NDArray[Any]] = {}
    combined_header = MixedFormatHeader()

    for path in model_paths:
        weights, header = load_mixed_format_model(path)
        combined_weights.update(weights)

        # Merge header info
        combined_header.layer_formats.update(header.layer_formats)
        combined_header.layer_group_sizes.update(header.layer_group_sizes)
        combined_header.total_params += header.total_params
        combined_header.quantized_params += header.quantized_params

    # Recompute average bits
    if combined_header.total_params > 0:
        total_bits = sum(
            combined_header.total_params * _bits_for_precision(_format_str_to_precision(fmt))
            for fmt in combined_header.layer_formats.values()
        )
        # This is an approximation; precise calculation would need param counts
        combined_header.average_bits = total_bits / combined_header.total_params

    # Save merged model
    metadata = {_HEADER_KEY: combined_header.to_json()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(combined_weights, str(output_path), metadata=metadata)

    return combined_header
