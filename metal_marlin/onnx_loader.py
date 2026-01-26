"""
ONNX model weight extractor for Marlin quantization.

Extracts weights from ONNX models without running inference. Maps ONNX
initializer names to HuggingFace-style weight names for compatibility
with the rest of the metal_marlin pipeline.

Usage:
    from metal_marlin.onnx_loader import extract_onnx_weights, get_onnx_config

    # Iterate over weights
    for name, tensor, meta in extract_onnx_weights("model.onnx"):
        print(f"{name}: {tensor.shape}")

    # Extract model configuration
    config = get_onnx_config("model.onnx")
    print(f"Hidden size: {config.hidden_size}")

Dependencies:
    pip install onnx  # Optional, only required when using this module
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

from .hf_loader import ModelConfig


def extract_onnx_weights(
    onnx_path: str | Path,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any]]]:
    """
    Load ONNX model and yield (name, tensor, metadata) for each weight.

    Extracts all initializers (weight tensors) from an ONNX model file.
    Converts ONNX node path-style names to HuggingFace-style names for
    compatibility with other loaders in this package.

    Args:
        onnx_path: Path to the ONNX model file.

    Yields:
        Tuples of (hf_name, tensor, metadata) where:
          - hf_name: HuggingFace-style weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
          - tensor: numpy array containing the weight data
          - metadata: dict with at least {"onnx_name": original_name}

    Raises:
        ImportError: If the onnx package is not installed.
        FileNotFoundError: If the ONNX file does not exist.
        onnx.checker.ValidationError: If the ONNX model is malformed.

    Example:
        >>> for name, tensor, meta in extract_onnx_weights("llama.onnx"):
        ...     print(f"{name}: {tensor.shape} (from {meta['onnx_name']})")
        model.embed_tokens.weight: (32000, 4096) (from /model/embed_tokens/Gather)
        model.layers.0.self_attn.q_proj.weight: (4096, 4096) (from /model/layers.0/self_attn/q_proj/MatMul)
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as e:
        raise ImportError(
            "onnx package is required for ONNX model loading. Install with: pip install onnx"
        ) from e

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Load and validate the model
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # Build a mapping of initializer names to their consumers
    # This helps identify weight roles (q_proj, k_proj, etc.)
    consumer_map = _build_consumer_map(model.graph)

    for initializer in model.graph.initializer:
        onnx_name = initializer.name
        tensor = numpy_helper.to_array(initializer)

        # Normalize to HuggingFace-style name
        hf_name = normalize_onnx_name(onnx_name, consumer_map)

        metadata: dict[str, Any] = {
            "onnx_name": onnx_name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }

        # Add tensor type hints based on shape and name patterns
        if tensor.ndim == 2:
            metadata["tensor_type"] = "weight_matrix"
        elif tensor.ndim == 1:
            if "norm" in onnx_name.lower() or "ln" in onnx_name.lower():
                metadata["tensor_type"] = "norm_weight"
            elif "bias" in onnx_name.lower():
                metadata["tensor_type"] = "bias"
            else:
                metadata["tensor_type"] = "1d_tensor"
        elif tensor.ndim == 0:
            metadata["tensor_type"] = "scalar"
        else:
            metadata["tensor_type"] = f"{tensor.ndim}d_tensor"

        yield hf_name, tensor, metadata


def normalize_onnx_name(onnx_name: str, consumer_map: dict[str, list[str]] | None = None) -> str:
    """
    Convert ONNX node path to HuggingFace weight name.

    ONNX models (especially from onnxruntime or torch.onnx.export) use
    path-style names like:
      - "/model/layers.0/self_attn/q_proj/MatMul"
      - "/model/embed_tokens/Gather"
      - "model.layers.0.self_attn.q_proj.weight"

    This function normalizes to HuggingFace convention:
      - "model.layers.0.self_attn.q_proj.weight"
      - "model.embed_tokens.weight"

    Args:
        onnx_name: Original ONNX initializer name.
        consumer_map: Optional mapping of initializer names to consumer ops.
                      Used to better identify tensor roles.

    Returns:
        Normalized HuggingFace-style weight name.

    Examples:
        >>> normalize_onnx_name("/model/layers.0/self_attn/q_proj/MatMul")
        'model.layers.0.self_attn.q_proj.weight'
        >>> normalize_onnx_name("/model/embed_tokens/Gather")
        'model.embed_tokens.weight'
        >>> normalize_onnx_name("model.layers.0.mlp.gate_proj.weight")
        'model.layers.0.mlp.gate_proj.weight'
    """
    name = onnx_name

    # Strip leading slash
    if name.startswith("/"):
        name = name[1:]

    # Replace path separators with dots
    name = name.replace("/", ".")

    # Remove common ONNX op suffixes that indicate the operation type
    op_suffixes = (
        ".MatMul",
        ".Gather",
        ".Gemm",
        ".Add",
        ".Mul",
        ".Cast",
        ".Transpose",
        ".Reshape",
        "_weight",
        "_bias",
    )
    for suffix in op_suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Handle weight/bias naming
    # If name doesn't end with .weight or .bias, and it looks like a weight matrix,
    # add the appropriate suffix
    if not name.endswith((".weight", ".bias", ".scale", ".gamma", ".beta")):
        # Check for bias indicators in the original name
        lower_name = onnx_name.lower()
        if "bias" in lower_name or name.endswith(".b"):
            name = name.rstrip(".b") + ".bias" if name.endswith(".b") else name + ".bias"
        # Check for norm scale/shift
        elif any(pat in lower_name for pat in ("layernorm", "rmsnorm", "ln_")):
            if "_weight" in lower_name or "gamma" in lower_name:
                name = name + ".weight"
            elif "_bias" in lower_name or "beta" in lower_name:
                name = name + ".bias"
            else:
                # Default to .weight for norm layers
                name = name + ".weight"
        else:
            # Default: assume it's a weight
            name = name + ".weight"

    # Clean up any double dots from processing
    while ".." in name:
        name = name.replace("..", ".")

    # Strip leading/trailing dots
    name = name.strip(".")

    return name


def get_onnx_config(onnx_path: str | Path) -> ModelConfig:
    """
    Extract model configuration from ONNX metadata or infer from tensor shapes.

    Attempts to find configuration in the following order:
    1. ONNX model metadata properties (if exported with config)
    2. Inference from weight tensor shapes

    The inferred configuration may be incomplete for complex architectures
    but should cover standard transformer models.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        ModelConfig with detected/inferred parameters.

    Raises:
        ImportError: If the onnx package is not installed.
        ValueError: If no configuration can be extracted or inferred.

    Example:
        >>> config = get_onnx_config("llama-7b.onnx")
        >>> print(f"Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")
        Layers: 32, Hidden: 4096
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError(
            "onnx package is required for ONNX model loading. Install with: pip install onnx"
        ) from e

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))

    # Try to extract config from model metadata
    config_dict = _extract_metadata_config(model)

    # Infer missing values from tensor shapes
    config_dict = _infer_config_from_shapes(model, config_dict)

    if not config_dict:
        raise ValueError(
            f"Could not extract or infer model configuration from {onnx_path}. "
            "The ONNX model may not be a standard transformer architecture."
        )

    return ModelConfig.from_dict(config_dict)


def _build_consumer_map(graph) -> dict[str, list[str]]:
    """
    Build a mapping from initializer names to their consumer operation types.

    This helps identify the role of each weight tensor (e.g., q_proj weight
    feeds into a MatMul for query projection).

    Args:
        graph: ONNX GraphProto

    Returns:
        Dict mapping initializer name -> list of consumer op types
    """
    consumer_map: dict[str, list[str]] = {}

    for node in graph.node:
        for input_name in node.input:
            if input_name not in consumer_map:
                consumer_map[input_name] = []
            consumer_map[input_name].append(node.op_type)

    return consumer_map


def _extract_metadata_config(model) -> dict[str, Any]:
    """
    Extract configuration from ONNX model metadata properties.

    Models exported with torch.onnx.export or optimum sometimes include
    configuration as metadata entries.

    Args:
        model: ONNX ModelProto

    Returns:
        Dict of extracted configuration parameters (may be empty)
    """
    config: dict[str, Any] = {}

    # Check model metadata properties
    for prop in model.metadata_props:
        key = prop.key.lower()
        value = prop.value

        # Try to parse known config keys
        if key in ("hidden_size", "d_model", "n_embd"):
            config["hidden_size"] = int(value)
        elif key in ("num_hidden_layers", "n_layer", "num_layers"):
            config["num_hidden_layers"] = int(value)
        elif key in ("num_attention_heads", "n_head", "num_heads"):
            config["num_attention_heads"] = int(value)
        elif key in ("num_key_value_heads", "num_kv_heads"):
            config["num_key_value_heads"] = int(value)
        elif key in ("intermediate_size", "d_ff", "ffn_dim"):
            config["intermediate_size"] = int(value)
        elif key in ("vocab_size", "n_vocab"):
            config["vocab_size"] = int(value)
        elif key in ("max_position_embeddings", "n_positions", "max_seq_len"):
            config["max_position_embeddings"] = int(value)
        elif key == "model_type":
            config["model_type"] = value

    return config


def _infer_config_from_shapes(model, config: dict[str, Any]) -> dict[str, Any]:
    """
    Infer missing configuration values from tensor shapes.

    Analyzes weight tensor shapes to determine model architecture parameters
    for standard transformer models.

    Args:
        model: ONNX ModelProto
        config: Existing config dict (may be partially filled)

    Returns:
        Updated config dict with inferred values
    """
    from onnx import numpy_helper

    # Collect tensor shapes keyed by normalized name
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for init in model.graph.initializer:
        name = normalize_onnx_name(init.name)
        tensor = numpy_helper.to_array(init)
        tensor_shapes[name] = tensor.shape

    # Infer hidden_size from embedding or attention projection
    if "hidden_size" not in config:
        # Try embedding weight [vocab_size, hidden_size]
        for name, shape in tensor_shapes.items():
            if "embed" in name and "weight" in name and len(shape) == 2:
                config["hidden_size"] = shape[1]
                config.setdefault("vocab_size", shape[0])
                break

        # Try attention Q projection [hidden_size, hidden_size]
        if "hidden_size" not in config:
            for name, shape in tensor_shapes.items():
                if "q_proj" in name and "weight" in name and len(shape) == 2:
                    config["hidden_size"] = shape[0]
                    break

    # Infer num_hidden_layers by counting unique layer indices
    if "num_hidden_layers" not in config:
        layer_indices: set[int] = set()
        layer_pattern = re.compile(r"layers?\.(\d+)\.")
        for name in tensor_shapes:
            match = layer_pattern.search(name)
            if match:
                layer_indices.add(int(match.group(1)))
        if layer_indices:
            config["num_hidden_layers"] = max(layer_indices) + 1

    # Infer num_attention_heads from Q projection shape
    if "num_attention_heads" not in config and "hidden_size" in config:
        hidden_size = config["hidden_size"]
        # Common head dimensions
        for head_dim in (128, 96, 80, 64, 48, 32):
            if hidden_size % head_dim == 0:
                config["num_attention_heads"] = hidden_size // head_dim
                break

    # Default num_key_value_heads to num_attention_heads (standard MHA)
    if "num_key_value_heads" not in config and "num_attention_heads" in config:
        # Check if there are separate k_proj weights that suggest GQA
        kv_heads = None
        for name, shape in tensor_shapes.items():
            if "k_proj" in name and "weight" in name and len(shape) == 2:
                if "hidden_size" in config:
                    # k_proj: [kv_heads * head_dim, hidden_size] or [hidden_size, kv_heads * head_dim]
                    # Determine which dimension is the projection output
                    out_dim = shape[0]
                    if shape[1] == config["hidden_size"]:
                        out_dim = shape[0]
                    elif shape[0] == config["hidden_size"]:
                        out_dim = shape[1]

                    head_dim = config["hidden_size"] // config["num_attention_heads"]
                    if out_dim % head_dim == 0:
                        kv_heads = out_dim // head_dim
                        break

        config["num_key_value_heads"] = kv_heads or config["num_attention_heads"]

    # Infer intermediate_size from MLP gate/up projection
    if "intermediate_size" not in config:
        for name, shape in tensor_shapes.items():
            if any(p in name for p in ("gate_proj", "up_proj", "fc1", "w1")) and "weight" in name:
                if len(shape) == 2:
                    # Shape is [intermediate_size, hidden_size] or [hidden_size, intermediate_size]
                    if "hidden_size" in config:
                        hidden = config["hidden_size"]
                        config["intermediate_size"] = shape[0] if shape[1] == hidden else shape[1]
                    else:
                        config["intermediate_size"] = max(shape)
                    break

    # Provide reasonable defaults for missing values
    if "max_position_embeddings" not in config:
        config.setdefault("max_position_embeddings", 4096)

    if "rope_theta" not in config:
        config.setdefault("rope_theta", 10000.0)

    if "rms_norm_eps" not in config:
        config.setdefault("rms_norm_eps", 1e-6)

    if "model_type" not in config:
        config.setdefault("model_type", "unknown")

    return config


def list_onnx_tensors(onnx_path: str | Path) -> list[dict[str, Any]]:
    """
    List all initializers (weights) in an ONNX model with metadata.

    Provides a quick overview of the model's weight structure without
    loading tensor data into memory.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        List of dicts with keys: onnx_name, hf_name, shape, dtype, size_bytes

    Example:
        >>> for info in list_onnx_tensors("model.onnx"):
        ...     print(f"{info['hf_name']}: {info['shape']}")
    """
    try:
        import onnx
        from onnx import TensorProto
    except ImportError as e:
        raise ImportError(
            "onnx package is required for ONNX model loading. Install with: pip install onnx"
        ) from e

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    consumer_map = _build_consumer_map(model.graph)

    # ONNX dtype mapping
    dtype_map = {
        TensorProto.FLOAT: ("float32", 4),
        TensorProto.FLOAT16: ("float16", 2),
        TensorProto.BFLOAT16: ("bfloat16", 2),
        TensorProto.INT64: ("int64", 8),
        TensorProto.INT32: ("int32", 4),
        TensorProto.INT16: ("int16", 2),
        TensorProto.INT8: ("int8", 1),
        TensorProto.UINT8: ("uint8", 1),
        TensorProto.DOUBLE: ("float64", 8),
    }

    result = []
    for init in model.graph.initializer:
        onnx_name = init.name
        hf_name = normalize_onnx_name(onnx_name, consumer_map)
        shape = list(init.dims)

        dtype_info = dtype_map.get(init.data_type, ("unknown", 0))
        dtype_str = dtype_info[0]
        bytes_per_elem = dtype_info[1]

        n_elements = 1
        for dim in shape:
            n_elements *= dim
        size_bytes = n_elements * bytes_per_elem

        result.append(
            {
                "onnx_name": onnx_name,
                "hf_name": hf_name,
                "shape": shape,
                "dtype": dtype_str,
                "n_elements": n_elements,
                "size_bytes": size_bytes,
            }
        )

    return result
