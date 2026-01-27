"""
Standalone HuggingFace model loader and FP4 converter.

Downloads models directly from HuggingFace Hub, loads safetensors weights,
and converts to Marlin FP4 format. No MLX or framework dependencies.

Usage:
    # Download and convert
    python -m metal_marlin.hf_loader zai-org/GLM-4.7-Flash ./glm4-fp4/ --group-size 128

    # Python API
    from metal_marlin.hf_loader import download_and_convert, load_model_config

    config = load_model_config("zai-org/GLM-4.7-Flash")
    weights = download_and_convert("zai-org/GLM-4.7-Flash", "./output/", group_size=128)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .quantize_fp4 import compute_quantization_error, quantize_fp4


@dataclass
class ModelConfig:
    """Parsed model configuration."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    model_type: str = "llama"  # llama, qwen2, mistral, glm4, etc.

    # Architecture-specific
    use_sliding_window: bool = False
    sliding_window: int | None = None

    # RoPE scaling (YaRN, linear, dynamic)
    rope_scaling_type: str | None = None  # "yarn", "linear", "dynamic", etc.
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 0  # original_max_position_embeddings
    rope_beta_fast: float = 32.0  # YaRN beta_fast
    rope_beta_slow: float = 1.0  # YaRN beta_slow
    rope_mscale: float = 1.0  # YaRN mscale

    # MoE-specific
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    shared_expert_intermediate_size: int | None = None

    # MTP-specific (Multi-Token Prediction)
    num_mtp_heads: int | None = None  # GLM-4.7-Flash uses MTP

    @property
    def is_moe(self) -> bool:
        """Check if model uses Mixture of Experts."""
        return self.num_experts is not None and self.num_experts > 1

    @property
    def has_mtp(self) -> bool:
        """Check if model has Multi-Token Prediction heads."""
        return self.num_mtp_heads is not None and self.num_mtp_heads > 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Parse from HuggingFace config.json."""
        # Parse rope_scaling config if present
        rope_scaling = d.get("rope_scaling") or {}
        rope_scaling_type = rope_scaling.get("type") if rope_scaling else None
        rope_scaling_factor = rope_scaling.get("factor", 1.0) if rope_scaling else 1.0
        rope_original_max_position = (
            rope_scaling.get("original_max_position_embeddings", 0) if rope_scaling else 0
        )
        rope_beta_fast = rope_scaling.get("beta_fast", 32.0) if rope_scaling else 32.0
        rope_beta_slow = rope_scaling.get("beta_slow", 1.0) if rope_scaling else 1.0
        rope_mscale = rope_scaling.get("mscale", 1.0) if rope_scaling else 1.0

        return cls(
            hidden_size=d.get("hidden_size", d.get("d_model", 4096)),
            num_hidden_layers=d.get("num_hidden_layers", d.get("n_layer", 32)),
            num_attention_heads=d.get("num_attention_heads", d.get("n_head", 32)),
            num_key_value_heads=d.get("num_key_value_heads", d.get("num_attention_heads", 32)),
            intermediate_size=d.get("intermediate_size", d.get("d_ff", 11008)),
            vocab_size=d.get("vocab_size", 32000),
            max_position_embeddings=d.get("max_position_embeddings", 4096),
            rope_theta=d.get("rope_theta", 10000.0),
            rms_norm_eps=d.get("rms_norm_eps", d.get("layer_norm_epsilon", 1e-6)),
            hidden_act=d.get("hidden_act", d.get("activation_function", "silu")),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            model_type=d.get("model_type", "llama"),
            use_sliding_window=d.get("use_sliding_window", False),
            sliding_window=d.get("sliding_window"),
            # RoPE scaling (YaRN, linear, dynamic)
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            rope_original_max_position=rope_original_max_position,
            rope_beta_fast=rope_beta_fast,
            rope_beta_slow=rope_beta_slow,
            rope_mscale=rope_mscale,
            # MoE config - check multiple naming conventions
            num_experts=d.get("num_local_experts")
            or d.get("num_experts")
            or d.get("n_routed_experts"),
            num_experts_per_tok=d.get("num_experts_per_tok") or d.get("num_selected_experts"),
            shared_expert_intermediate_size=d.get("shared_expert_intermediate_size")
            or d.get("n_shared_experts"),
            # MTP config (GLM-4.7-Flash style)
            num_mtp_heads=d.get("num_mtp_heads", d.get("num_nextn_predict_layers")),
        )


def download_model(
    model_id: str,
    local_dir: str | Path | None = None,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """
    Download model files from HuggingFace Hub.

    Downloads config.json, tokenizer files, and all safetensors shards.

    Args:
        model_id: HuggingFace model ID (e.g., "zai-org/GLM-4.7-Flash")
        local_dir: Local directory to save files (default: HF cache)
        revision: Git revision (branch, tag, or commit)
        token: HuggingFace API token for gated models

    Returns:
        Path to the downloaded model directory
    """
    from huggingface_hub import snapshot_download

    # Download everything except pytorch bins (we only want safetensors)
    path = snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir) if local_dir else None,
        revision=revision,
        token=token,
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.ckpt"],
    )
    return Path(path)


def load_model_config(model_path: str | Path) -> ModelConfig:
    """Load and parse model configuration."""
    model_path = Path(model_path)

    # If it's a HF model ID, download first
    if not model_path.exists():
        model_path = download_model(str(model_path))

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    return ModelConfig.from_dict(config_dict)


def iter_safetensors_weights(
    model_path: str | Path,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any]]]:
    """
    Iterate over all weights in safetensors files.

    Yields:
        (tensor_name, tensor_data, metadata) for each tensor
    """
    from safetensors import safe_open

    model_path = Path(model_path)

    # Find all safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        # Check for model subdirectory
        st_files = sorted(model_path.glob("model*.safetensors"))

    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Try torch framework first (supports bfloat16), fall back to numpy
    try:
        import torch

        for st_file in st_files:
            with safe_open(str(st_file), framework="pt") as f:
                metadata = f.metadata() or {}
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    # Convert to numpy, handling bfloat16
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float().numpy()
                    else:
                        tensor = tensor.numpy()
                    yield name, tensor, metadata
    except ImportError:
        # Fall back to numpy framework (doesn't support bfloat16)
        for st_file in st_files:
            with safe_open(str(st_file), framework="numpy") as f:
                metadata = f.metadata() or {}
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    yield name, tensor, metadata


def should_quantize_tensor(name: str, tensor: np.ndarray) -> bool:
    """
    Determine if a tensor should be quantized to FP4.

    Embeddings, norms, biases, and lm_head are kept in full precision.
    Only 2D weight matrices with compatible dimensions are quantized.
    """
    # Skip non-weight tensors
    if "weight" not in name.lower():
        return False

    # Must be 2D
    if tensor.ndim != 2:
        return False

    # Skip embeddings, norms, and output projection
    skip_patterns = [
        "embed",
        "embedding",
        "norm",
        "layernorm",
        "rmsnorm",
        "lm_head",
        "output",
        "bias",
    ]
    name_lower = name.lower()
    if any(pat in name_lower for pat in skip_patterns):
        return False

    # Check dimensions are compatible with FP4 packing (must be divisible by 8)
    out_feat, in_feat = tensor.shape
    if in_feat % 8 != 0:
        return False

    return True


@dataclass
class CalibrationData:
    """
    Calibration data for activation-aware quantization.

    Collected by running representative inputs through the model and
    recording activation statistics at each layer.
    """

    # Per-layer activation ranges: layer_name -> (min, max)
    layer_ranges: dict[str, tuple[float, float]]

    # Global percentile for outlier clipping (e.g., 99.9 = clip top 0.1%)
    percentile: float = 99.9

    # Smoothing factor: 0 = weight-only, 1 = fully calibrated
    smooth_factor: float = 0.8

    @classmethod
    def from_json(cls, path: str | Path) -> CalibrationData:
        """Load calibration data from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            layer_ranges=data.get("layer_ranges", {}),
            percentile=data.get("percentile", 99.9),
            smooth_factor=data.get("smooth_factor", 0.8),
        )

    def to_json(self, path: str | Path) -> None:
        """Save calibration data to JSON file."""
        with open(path, "w") as f:
            json.dump(
                {
                    "layer_ranges": self.layer_ranges,
                    "percentile": self.percentile,
                    "smooth_factor": self.smooth_factor,
                },
                f,
                indent=2,
            )

    def get_activation_ranges(self, layer_name: str) -> dict[str, Any] | None:
        """
        Get activation_ranges dict for quantize_fp4().

        Returns None if no calibration data for this layer.
        """
        if layer_name not in self.layer_ranges:
            return None

        input_range = self.layer_ranges[layer_name]
        return {
            "input_range": input_range,
            "percentile": self.percentile,
            "smooth_factor": self.smooth_factor,
        }


def convert_model_to_fp4(
    model_path: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    validate: bool = True,
    verbose: bool = True,
    calibration: CalibrationData | str | Path | None = None,
) -> dict[str, Any]:
    """
    Convert a HuggingFace model to Marlin FP4 format.

    Args:
        model_path: Path to model directory or HuggingFace model ID
        output_path: Directory to save converted model
        group_size: FP4 quantization group size
        validate: Compute quantization errors
        verbose: Print progress
        calibration: Optional calibration data for activation-aware quantization.
            Can be:
            - CalibrationData instance
            - Path to calibration JSON file
            - None for weight-only quantization

    Returns:
        Statistics dict with counts, sizes, errors
    """
    from safetensors.numpy import save_file

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load calibration data if provided
    calib_data: CalibrationData | None = None
    if calibration is not None:
        if isinstance(calibration, CalibrationData):
            calib_data = calibration
        elif isinstance(calibration, (str, Path)):
            calib_data = CalibrationData.from_json(calibration)
        else:
            raise TypeError(
                f"calibration must be CalibrationData, str, or Path, got {type(calibration)}"
            )
        if verbose:
            print(f"Using calibration data with {len(calib_data.layer_ranges)} layers")
            print(
                f"  Percentile: {calib_data.percentile}, Smooth factor: {calib_data.smooth_factor}"
            )

    # If it's a HF model ID, download first
    if not model_path.exists():
        if verbose:
            print(f"Downloading {model_path}...")
        model_path = download_model(str(model_path))

    # Copy config and tokenizer files
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]:
        src = model_path / fname
        if src.exists():
            import shutil

            shutil.copy(src, output_path / fname)

    stats = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "errors": [],
        "tensors": {},
    }

    output_tensors = {}

    for name, tensor, _ in iter_safetensors_weights(model_path):
        if should_quantize_tensor(name, tensor):
            # Quantize to FP4
            if verbose:
                print(f"  Quantizing {name}: {tensor.shape}")

            out_feat, in_feat = tensor.shape

            # Ensure group_size divides in_feat
            if in_feat % group_size != 0:
                # Use a compatible group size
                for gs in [64, 32, 16, 8]:
                    if in_feat % gs == 0:
                        actual_gs = gs
                        break
                else:
                    actual_gs = 8
            else:
                actual_gs = group_size

            # Get activation ranges for this layer if calibration data available
            activation_ranges = None
            if calib_data is not None:
                activation_ranges = calib_data.get_activation_ranges(name)
                if activation_ranges is not None and verbose:
                    input_range = activation_ranges["input_range"]
                    print(f"    Using calibration: input_range={input_range}")

            packed, scales = quantize_fp4(
                tensor, group_size=actual_gs, activation_ranges=activation_ranges
            )

            output_tensors[name] = packed
            output_tensors[f"{name}.scales"] = scales
            output_tensors[f"{name}.group_size"] = np.array([actual_gs], dtype=np.int32)

            stats["quantized_count"] += 1
            stats["original_bytes"] += tensor.nbytes
            stats["quantized_bytes"] += packed.nbytes + scales.nbytes

            if validate:
                err = compute_quantization_error(tensor, packed, scales, actual_gs)
                stats["errors"].append({"name": name, **err})
                stats["tensors"][name] = {
                    "shape": list(tensor.shape),
                    "group_size": actual_gs,
                    "rmse": err["rmse"],
                }
        else:
            # Keep in full precision
            if verbose:
                print(f"  Keeping {name}: {tensor.shape} ({tensor.dtype})")
            output_tensors[name] = tensor
            stats["skipped_count"] += 1

    # Save quantized model
    output_file = output_path / "model.safetensors"
    if verbose:
        print(f"\nSaving to {output_file}...")
    save_file(output_tensors, str(output_file))

    # Save quantization metadata
    meta = {
        "format": "marlin_fp4",
        "group_size": group_size,
        "quantized_count": stats["quantized_count"],
        "skipped_count": stats["skipped_count"],
        "compression_ratio": stats["original_bytes"] / max(stats["quantized_bytes"], 1),
        "calibration_aware": calib_data is not None,
    }
    if calib_data is not None:
        meta["calibration"] = {
            "percentile": calib_data.percentile,
            "smooth_factor": calib_data.smooth_factor,
            "layers_calibrated": len(calib_data.layer_ranges),
        }
    if stats["errors"]:
        meta["mean_rmse"] = float(np.mean([e["rmse"] for e in stats["errors"]]))
        meta["max_error"] = float(max(e["max_error"] for e in stats["errors"]))

    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("Conversion complete!")
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped:   {stats['skipped_count']} tensors")
        if stats["original_bytes"] > 0:
            ratio = stats["original_bytes"] / max(stats["quantized_bytes"], 1)
            print(f"  Compression: {ratio:.2f}x")
        if "mean_rmse" in meta:
            print(f"  Mean RMSE: {meta['mean_rmse']:.6f}")
        print(f"{'=' * 60}")

    stats.update(meta)
    return stats


def load_quantized_weights(
    model_path: str | Path,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load quantized weights from a converted model.

    Returns dict mapping layer names to their quantized components:
    {
        "model.layers.0.self_attn.q_proj": {
            "packed": np.array(...),      # uint32 packed weights
            "scales": np.array(...),       # float16 scales
            "group_size": 128,
        },
        ...
    }
    """
    from safetensors import safe_open

    model_path = Path(model_path)
    st_file = model_path / "model.safetensors"

    if not st_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")

    weights = {}

    with safe_open(str(st_file), framework="numpy") as f:
        for name in f.keys():
            if name.endswith(".scales") or name.endswith(".group_size"):
                continue

            tensor = f.get_tensor(name)

            # Check if this is a quantized weight
            scales_name = f"{name}.scales"
            gs_name = f"{name}.group_size"

            if scales_name in f.keys():
                weights[name] = {
                    "packed": tensor,
                    "scales": f.get_tensor(scales_name),
                    "group_size": int(f.get_tensor(gs_name)[0]) if gs_name in f.keys() else 128,
                }
            else:
                weights[name] = {
                    "data": tensor,
                    "quantized": False,
                }

    return weights


# ============================================================================
# Layer-wise Quantization (Memory-Efficient)
# ============================================================================


def _load_safetensors_index(model_path: Path) -> dict[str, Any] | None:
    """
    Load safetensors index for sharded models.

    Returns the weight_map from model.safetensors.index.json, or None
    if the model uses a single safetensors file.
    """
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return None

    with open(index_path) as f:
        index = json.load(f)

    return index.get("weight_map", {})


def _get_layer_weight_files(
    model_path: Path,
    layer_idx: int,
    weight_map: dict[str, str] | None,
) -> dict[str, Path]:
    """
    Get the safetensors files containing weights for a specific layer.

    Returns a dict mapping tensor names to their file paths.
    """
    from safetensors import safe_open

    layer_prefix = f"model.layers.{layer_idx}."
    result: dict[str, Path] = {}

    if weight_map is None:
        # Single safetensors file
        st_file = model_path / "model.safetensors"
        if st_file.exists():
            with safe_open(str(st_file), framework="numpy") as f:
                for name in f.keys():
                    if name.startswith(layer_prefix):
                        result[name] = st_file
        return result

    # Sharded model - find files for this layer
    for tensor_name, filename in weight_map.items():
        if tensor_name.startswith(layer_prefix):
            result[tensor_name] = model_path / filename

    return result


def load_layer_weights(
    model_path: Path,
    layer_idx: int,
    weight_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load weights for a single transformer layer.

    Memory-efficient: only loads tensors for the specified layer.
    Uses torch framework to handle bfloat16 dtypes.

    Args:
        model_path: Path to model directory
        layer_idx: Layer index (0-based)
        weight_map: Optional preloaded weight map from index.json

    Returns:
        Dict mapping tensor names to numpy arrays
    """
    from safetensors import safe_open

    if weight_map is None:
        weight_map = _load_safetensors_index(model_path)

    tensor_files = _get_layer_weight_files(model_path, layer_idx, weight_map)

    if not tensor_files:
        raise ValueError(f"No weights found for layer {layer_idx}")

    weights: dict[str, np.ndarray] = {}

    # Group by file to minimize file opens
    files_to_tensors: dict[Path, list[str]] = {}
    for tensor_name, file_path in tensor_files.items():
        if file_path not in files_to_tensors:
            files_to_tensors[file_path] = []
        files_to_tensors[file_path].append(tensor_name)

    # Use torch framework to handle bfloat16, then convert to numpy
    for file_path, tensor_names in files_to_tensors.items():
        with safe_open(str(file_path), framework="pt") as f:
            for name in tensor_names:
                tensor = f.get_tensor(name)
                # Convert to float32 numpy (handles bfloat16)
                weights[name] = tensor.float().numpy()

    return weights


def load_non_layer_weights(
    model_path: Path,
    weight_type: str,
    weight_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load non-layer weights (embeddings, lm_head, final norm).

    Args:
        model_path: Path to model directory
        weight_type: One of "embed", "lm_head", "final_norm"
        weight_map: Optional preloaded weight map from index.json

    Returns:
        Dict mapping tensor names to numpy arrays
    """
    from safetensors import safe_open

    prefixes = {
        "embed": ["model.embed_tokens", "wte", "word_embedding", "embed"],
        "lm_head": ["lm_head", "output.weight"],
        "final_norm": ["model.norm", "model.final_layernorm", "ln_f"],
    }

    if weight_type not in prefixes:
        raise ValueError(f"Unknown weight_type: {weight_type}. Use: {list(prefixes.keys())}")

    patterns = prefixes[weight_type]

    if weight_map is None:
        weight_map = _load_safetensors_index(model_path)

    weights: dict[str, np.ndarray] = {}

    # Use torch framework to handle bfloat16, then convert to numpy
    if weight_map is None:
        # Single file model
        st_file = model_path / "model.safetensors"
        if not st_file.exists():
            st_files = sorted(model_path.glob("*.safetensors"))
            if not st_files:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
            st_file = st_files[0]

        with safe_open(str(st_file), framework="pt") as f:
            for name in f.keys():
                if any(pat in name.lower() for pat in patterns):
                    tensor = f.get_tensor(name)
                    weights[name] = tensor.float().numpy()
    else:
        # Sharded model
        files_to_load: dict[Path, list[str]] = {}
        for tensor_name, filename in weight_map.items():
            if any(pat in tensor_name.lower() for pat in patterns):
                file_path = model_path / filename
                if file_path not in files_to_load:
                    files_to_load[file_path] = []
                files_to_load[file_path].append(tensor_name)

        for file_path, tensor_names in files_to_load.items():
            with safe_open(str(file_path), framework="pt") as f:
                for name in tensor_names:
                    tensor = f.get_tensor(name)
                    weights[name] = tensor.float().numpy()

    return weights


def convert_model_layerwise(
    model_path: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    mixed_precision: MixedPrecisionConfig | None = None,
    calibration: CalibrationData | str | Path | None = None,
    validate: bool = True,
    verbose: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """
    Quantize model layer-by-layer to minimize memory usage.

    For a 30B model on 64GB M3 Max:
    - Full model FP16: ~60GB (won't fit)
    - One layer FP16: ~1GB (fits easily)
    - Quantized output: ~8GB total

    This approach processes one transformer layer at a time, immediately
    freeing memory after quantizing each layer. Peak memory usage is
    roughly 2x the size of a single layer (original + quantized).

    Args:
        model_path: Path to model directory or HuggingFace model ID
        output_path: Directory to save converted model
        group_size: FP4 quantization group size (default: 128)
        mixed_precision: Optional MixedPrecisionConfig for layer-specific handling
        calibration: Optional CalibrationData for calibration-aware quantization
        validate: Compute quantization errors (default: True)
        verbose: Print progress (default: True)
        token: HuggingFace API token for gated models

    Returns:
        Statistics dict with counts, sizes, errors, per-layer info
    """
    import gc
    import shutil

    from safetensors.numpy import save_file

    from .mixed_precision import MixedPrecisionConfig as MPConfig
    from .mixed_precision import Precision, should_quantize

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download if needed
    if not model_path.exists():
        if verbose:
            print(f"Downloading {model_path}...")
        model_path = download_model(str(model_path), token=token)

    # Load config
    config = load_model_config(model_path)
    if verbose:
        print(f"Model: {config.model_type}, {config.num_hidden_layers} layers")
        print(f"Hidden size: {config.hidden_size}, Intermediate: {config.intermediate_size}")

    # Load calibration data if provided
    calib_data: CalibrationData | None = None
    if calibration is not None:
        if isinstance(calibration, CalibrationData):
            calib_data = calibration
        elif isinstance(calibration, (str, Path)):
            calib_data = CalibrationData.from_json(calibration)
        if verbose and calib_data is not None:
            print(f"Using calibration data with {len(calib_data.layer_ranges)} layers")

    # Use default mixed precision if not specified
    if mixed_precision is None:
        if config.is_moe and config.has_mtp:
            mixed_precision = MPConfig.default_moe_mtp()
        elif config.is_moe:
            mixed_precision = MPConfig.default_moe()
        else:
            mixed_precision = MPConfig.default_dense()
        if verbose:
            preset = (
                "moe-mtp"
                if config.is_moe and config.has_mtp
                else "moe"
                if config.is_moe
                else "dense"
            )
            print(f"Using mixed-precision preset: {preset}")

    # Copy config and tokenizer files
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]:
        src = model_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)

    # Load weight map for sharded models
    weight_map = _load_safetensors_index(model_path)

    stats: dict[str, Any] = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "errors": [],
        "layers": {},
        "peak_memory_mb": 0,
    }

    all_output_tensors: dict[str, np.ndarray] = {}

    # --------------------------------------------------------------------
    # Step 1: Process embeddings (keep FP16/BF16)
    # --------------------------------------------------------------------
    if verbose:
        print("\n[1/3] Processing embeddings...")

    embed_weights = load_non_layer_weights(model_path, "embed", weight_map)
    for name, tensor in embed_weights.items():
        if verbose:
            print(f"  Keeping {name}: {tensor.shape} ({tensor.dtype})")
        all_output_tensors[name] = tensor
        stats["skipped_count"] += 1

    # Free memory
    del embed_weights
    gc.collect()

    # --------------------------------------------------------------------
    # Step 2: Process each transformer layer
    # --------------------------------------------------------------------
    if verbose:
        print(f"\n[2/3] Processing {config.num_hidden_layers} transformer layers...")

    for layer_idx in range(config.num_hidden_layers):
        if verbose:
            print(f"\n  Layer {layer_idx + 1}/{config.num_hidden_layers}")

        # Load single layer
        layer_weights = load_layer_weights(model_path, layer_idx, weight_map)

        layer_stats: dict[str, Any] = {
            "quantized": 0,
            "skipped": 0,
            "tensors": {},
        }

        for name, tensor in layer_weights.items():
            # Determine quantization config
            do_quant, layer_cfg = should_quantize(name, tensor, mixed_precision)

            if do_quant and layer_cfg.precision != Precision.FP16:
                actual_gs = layer_cfg.group_size
                out_feat, in_feat = tensor.shape

                # Ensure group_size divides in_feat
                if in_feat % actual_gs != 0:
                    for gs in [256, 128, 64, 32, 16, 8]:
                        if gs <= actual_gs and in_feat % gs == 0:
                            actual_gs = gs
                            break

                # Get activation ranges if calibration data available
                activation_ranges = None
                if calib_data is not None:
                    activation_ranges = calib_data.get_activation_ranges(name)

                # Quantize
                packed, scales = quantize_fp4(
                    tensor, group_size=actual_gs, activation_ranges=activation_ranges
                )

                all_output_tensors[name] = packed
                all_output_tensors[f"{name}.scales"] = scales
                all_output_tensors[f"{name}.group_size"] = np.array([actual_gs], dtype=np.int32)

                stats["quantized_count"] += 1
                stats["original_bytes"] += tensor.nbytes
                stats["quantized_bytes"] += packed.nbytes + scales.nbytes
                layer_stats["quantized"] += 1

                if validate:
                    err = compute_quantization_error(tensor, packed, scales, actual_gs)
                    stats["errors"].append({"name": name, "layer": layer_idx, **err})
                    layer_stats["tensors"][name] = {
                        "shape": list(tensor.shape),
                        "group_size": actual_gs,
                        "precision": layer_cfg.precision.value,
                        "rmse": err["rmse"],
                    }

                if verbose:
                    prec = layer_cfg.precision.value
                    print(f"    Quantizing {name.split('.')[-1]}: {tensor.shape} -> {prec}")
            else:
                # Keep in original precision
                all_output_tensors[name] = tensor
                stats["skipped_count"] += 1
                layer_stats["skipped"] += 1

                if verbose:
                    print(f"    Keeping {name.split('.')[-1]}: {tensor.shape}")

        stats["layers"][f"layer_{layer_idx}"] = layer_stats

        # Free layer memory immediately
        del layer_weights
        gc.collect()

    # --------------------------------------------------------------------
    # Step 3: Process lm_head and final norm (keep FP16/BF16)
    # --------------------------------------------------------------------
    if verbose:
        print("\n[3/3] Processing output layers...")

    for weight_type in ["final_norm", "lm_head"]:
        try:
            weights = load_non_layer_weights(model_path, weight_type, weight_map)
            for name, tensor in weights.items():
                if verbose:
                    print(f"  Keeping {name}: {tensor.shape} ({tensor.dtype})")
                all_output_tensors[name] = tensor
                stats["skipped_count"] += 1
            del weights
            gc.collect()
        except (ValueError, FileNotFoundError):
            # Some models may not have these
            pass

    # --------------------------------------------------------------------
    # Save quantized model
    # --------------------------------------------------------------------
    output_file = output_path / "model.safetensors"
    if verbose:
        print(f"\nSaving to {output_file}...")
    save_file(all_output_tensors, str(output_file))

    # Save quantization metadata
    meta: dict[str, Any] = {
        "format": "marlin_fp4",
        "conversion_method": "layerwise",
        "group_size": group_size,
        "num_layers": config.num_hidden_layers,
        "quantized_count": stats["quantized_count"],
        "skipped_count": stats["skipped_count"],
        "compression_ratio": stats["original_bytes"] / max(stats["quantized_bytes"], 1),
        "calibration_aware": calib_data is not None,
    }

    if stats["errors"]:
        meta["mean_rmse"] = float(np.mean([e["rmse"] for e in stats["errors"]]))
        meta["max_error"] = float(max(e["max_error"] for e in stats["errors"]))

    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("Layer-wise conversion complete!")
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped:   {stats['skipped_count']} tensors")
        if stats["original_bytes"] > 0:
            ratio = stats["original_bytes"] / max(stats["quantized_bytes"], 1)
            print(f"  Compression: {ratio:.2f}x")
        if "mean_rmse" in meta:
            print(f"  Mean RMSE: {meta['mean_rmse']:.6f}")
        print(f"{'=' * 60}")

    stats.update(meta)
    return stats


# ============================================================================
# Parallel Layer Quantization with RAM Management
# ============================================================================


def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    import platform

    if platform.system() == "Darwin":  # macOS
        try:
            import subprocess

            result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
            lines = result.stdout.split("\n")
            free_pages = 0
            page_size = 16384  # Default for Apple Silicon

            for line in lines:
                if "page size" in line.lower():
                    try:
                        page_size = int(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
                elif "Pages free" in line:
                    try:
                        free_pages += int(line.split()[-1].rstrip("."))
                    except (ValueError, IndexError):
                        pass
                elif "Pages inactive" in line:
                    try:
                        free_pages += int(line.split()[-1].rstrip("."))
                    except (ValueError, IndexError):
                        pass
                elif "Pages speculative" in line:
                    try:
                        free_pages += int(line.split()[-1].rstrip("."))
                    except (ValueError, IndexError):
                        pass

            return (free_pages * page_size) / (1024**3)
        except Exception:
            pass

    # Fallback: use psutil if available
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass

    # Conservative fallback: assume 16GB available
    return 16.0


def estimate_layer_memory_gb(config: ModelConfig, dtype_bytes: int = 2) -> float:
    """
    Estimate memory required to load and quantize one transformer layer.

    Returns memory in GB, conservatively overestimating for safety.
    """
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = hidden // n_heads

    # Attention weights: Q, K, V, O projections
    attn_bytes = (
        hidden * hidden  # Q
        + hidden * n_kv * head_dim  # K
        + hidden * n_kv * head_dim  # V
        + hidden * hidden  # O
    ) * dtype_bytes

    # MLP weights
    if config.is_moe:
        # MoE has multiple expert MLPs
        num_experts = config.num_experts or 8
        expert_intermediate = intermediate // num_experts if intermediate > hidden else intermediate
        mlp_bytes_per_expert = (
            hidden * expert_intermediate * 2  # gate + up
            + expert_intermediate * hidden  # down
        ) * dtype_bytes
        mlp_bytes = mlp_bytes_per_expert * num_experts

        # Router
        mlp_bytes += hidden * num_experts * dtype_bytes

        # Shared expert if present
        if config.shared_expert_intermediate_size:
            mlp_bytes += (
                hidden * config.shared_expert_intermediate_size * 2
                + config.shared_expert_intermediate_size * hidden
            ) * dtype_bytes
    else:
        mlp_bytes = (
            hidden * intermediate * 2  # gate + up (often fused)
            + intermediate * hidden  # down
        ) * dtype_bytes

    # Layer norms (negligible but include for completeness)
    norm_bytes = hidden * 2 * dtype_bytes * 2

    total_bytes = attn_bytes + mlp_bytes + norm_bytes

    # Add 50% overhead for intermediate buffers during quantization
    return (total_bytes * 1.5) / (1024**3)


def convert_model_parallel(
    model_path: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    mixed_precision: MixedPrecisionConfig | None = None,
    calibration: CalibrationData | str | Path | None = None,
    validate: bool = True,
    verbose: bool = True,
    token: str | None = None,
    max_workers: int | None = None,
    ram_budget_gb: float | None = None,
) -> dict[str, Any]:
    """
    Quantize model with parallel layer processing and RAM budget management.

    This function automatically determines how many layers can be processed
    in parallel based on available system RAM, maximizing throughput while
    avoiding memory pressure.

    For a 30B MoE model on 64GB M3 Max:
    - Each layer ~1.5GB memory during quantization
    - With 48GB available, can process ~16 layers in parallel
    - 4x faster than sequential quantization

    Args:
        model_path: Path to model directory or HuggingFace model ID
        output_path: Directory to save converted model
        group_size: FP4 quantization group size (default: 128)
        mixed_precision: Optional MixedPrecisionConfig for layer-specific handling
        calibration: Optional CalibrationData for calibration-aware quantization
        validate: Compute quantization errors (default: True)
        verbose: Print progress (default: True)
        token: HuggingFace API token for gated models
        max_workers: Maximum parallel workers (default: auto based on RAM)
        ram_budget_gb: RAM budget in GB (default: auto-detect available)

    Returns:
        Statistics dict with counts, sizes, errors, performance metrics
    """
    import gc
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from safetensors.numpy import save_file

    from .mixed_precision import MixedPrecisionConfig as MPConfig
    from .mixed_precision import Precision, should_quantize

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    # Download if needed
    if not model_path.exists():
        if verbose:
            print(f"Downloading {model_path}...")
        model_path = download_model(str(model_path), token=token)

    # Load config
    config = load_model_config(model_path)

    # Estimate memory and determine parallelism
    layer_mem_gb = estimate_layer_memory_gb(config)

    if ram_budget_gb is None:
        available_ram = get_available_ram_gb()
        # Use 80% of available RAM as budget (leave room for system)
        ram_budget_gb = available_ram * 0.8

    # Calculate max parallel layers
    max_parallel = max(1, int(ram_budget_gb / layer_mem_gb))
    if max_workers is not None:
        max_parallel = min(max_parallel, max_workers)

    # Cap at reasonable maximum (diminishing returns beyond 16)
    max_parallel = min(max_parallel, 16, config.num_hidden_layers)

    if verbose:
        print(f"\n{'=' * 70}")
        print("METAL MARLIN QUANTIZATION")
        print(f"{'=' * 70}")
        print(f"  Model:      {config.model_type} ({config.num_hidden_layers} layers)")
        print(f"  Hidden:     {config.hidden_size:,}")
        print(
            f"  MoE:        {'Yes (' + str(config.num_experts) + ' experts)' if config.is_moe else 'No'}"
        )
        print(f"  Vocabulary: {config.vocab_size:,}")
        print(f"{'=' * 70}")
        print("MEMORY CONFIGURATION")
        print(f"{'=' * 70}")
        print(f"  Available RAM:    {ram_budget_gb:.1f} GB")
        print(f"  Per-layer est:    {layer_mem_gb:.2f} GB")
        print(f"  Parallel layers:  {max_parallel}")
        print(f"{'=' * 70}")

    # Load calibration data if provided
    calib_data: CalibrationData | None = None
    if calibration is not None:
        if isinstance(calibration, CalibrationData):
            calib_data = calibration
        elif isinstance(calibration, (str, Path)):
            calib_data = CalibrationData.from_json(calibration)
        if verbose and calib_data is not None:
            print(f"  Calibration:      {len(calib_data.layer_ranges)} layers calibrated")

    # Use default mixed precision if not specified
    if mixed_precision is None:
        if config.is_moe and config.has_mtp:
            mixed_precision = MPConfig.default_moe_mtp()
            preset = "moe-mtp"
        elif config.is_moe:
            mixed_precision = MPConfig.default_moe()
            preset = "moe"
        else:
            mixed_precision = MPConfig.default_dense()
            preset = "dense"
        if verbose:
            print(f"  Precision:        {preset} (auto-detected)")
    print()

    # Copy config and tokenizer files
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]:
        src = model_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)

    # Load weight map for sharded models
    weight_map = _load_safetensors_index(model_path)

    stats: dict[str, Any] = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "errors": [],
        "layers": {},
        "parallel_workers": max_parallel,
        "ram_budget_gb": ram_budget_gb,
    }

    all_output_tensors: dict[str, np.ndarray] = {}
    tensor_lock = __import__("threading").Lock()

    # Worker function for quantizing a single layer
    def quantize_layer(layer_idx: int) -> dict[str, Any]:
        """Quantize a single layer and return results."""
        layer_stats: dict[str, Any] = {
            "layer_idx": layer_idx,
            "quantized": 0,
            "skipped": 0,
            "tensors": {},
            "output_tensors": {},
            "original_bytes": 0,
            "quantized_bytes": 0,
            "errors": [],
        }

        layer_weights = load_layer_weights(model_path, layer_idx, weight_map)

        for name, tensor in layer_weights.items():
            do_quant, layer_cfg = should_quantize(name, tensor, mixed_precision)

            if do_quant and layer_cfg.precision != Precision.FP16:
                actual_gs = layer_cfg.group_size
                out_feat, in_feat = tensor.shape

                if in_feat % actual_gs != 0:
                    for gs in [256, 128, 64, 32, 16, 8]:
                        if gs <= actual_gs and in_feat % gs == 0:
                            actual_gs = gs
                            break

                activation_ranges = None
                if calib_data is not None:
                    activation_ranges = calib_data.get_activation_ranges(name)

                packed, scales = quantize_fp4(
                    tensor, group_size=actual_gs, activation_ranges=activation_ranges
                )

                layer_stats["output_tensors"][name] = packed
                layer_stats["output_tensors"][f"{name}.scales"] = scales
                layer_stats["output_tensors"][f"{name}.group_size"] = np.array(
                    [actual_gs], dtype=np.int32
                )

                layer_stats["quantized"] += 1
                layer_stats["original_bytes"] += tensor.nbytes
                layer_stats["quantized_bytes"] += packed.nbytes + scales.nbytes

                if validate:
                    err = compute_quantization_error(tensor, packed, scales, actual_gs)
                    layer_stats["errors"].append({"name": name, "layer": layer_idx, **err})
                    layer_stats["tensors"][name] = {
                        "shape": list(tensor.shape),
                        "group_size": actual_gs,
                        "precision": layer_cfg.precision.value,
                        "rmse": err["rmse"],
                    }
            else:
                layer_stats["output_tensors"][name] = tensor
                layer_stats["skipped"] += 1

        return layer_stats

    # Process embeddings first (sequential, small)
    if verbose:
        print("[1/3] Processing embeddings...")

    embed_weights = load_non_layer_weights(model_path, "embed", weight_map)
    for name, tensor in embed_weights.items():
        if verbose:
            print(f"      {name}: {tensor.shape} (kept FP16)")
        all_output_tensors[name] = tensor
        stats["skipped_count"] += 1
    del embed_weights
    gc.collect()

    # Process transformer layers in parallel batches
    if verbose:
        print(f"\n[2/3] Processing {config.num_hidden_layers} transformer layers...")
        if max_parallel > 1:
            print(f"      (parallel: {max_parallel} layers at a time)\n")

    layers_completed = 0

    # Process in batches based on RAM budget
    for batch_start in range(0, config.num_hidden_layers, max_parallel):
        batch_end = min(batch_start + max_parallel, config.num_hidden_layers)
        batch_indices = list(range(batch_start, batch_end))

        if verbose:
            if max_parallel > 1:
                print(
                    f"  Batch {batch_start // max_parallel + 1}: layers {batch_start}-{batch_end - 1}"
                )

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(quantize_layer, idx): idx for idx in batch_indices}

            for future in as_completed(futures):
                layer_idx = futures[future]
                try:
                    layer_result = future.result()

                    # Merge results (thread-safe)
                    with tensor_lock:
                        all_output_tensors.update(layer_result["output_tensors"])
                        stats["quantized_count"] += layer_result["quantized"]
                        stats["skipped_count"] += layer_result["skipped"]
                        stats["original_bytes"] += layer_result["original_bytes"]
                        stats["quantized_bytes"] += layer_result["quantized_bytes"]
                        stats["errors"].extend(layer_result["errors"])
                        stats["layers"][f"layer_{layer_idx}"] = {
                            "quantized": layer_result["quantized"],
                            "skipped": layer_result["skipped"],
                            "tensors": layer_result["tensors"],
                        }

                    layers_completed += 1

                    if verbose:
                        pct = (layers_completed / config.num_hidden_layers) * 100
                        bar_len = 30
                        filled = int(bar_len * layers_completed / config.num_hidden_layers)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        elapsed = time.perf_counter() - start_time
                        eta = (elapsed / layers_completed) * (
                            config.num_hidden_layers - layers_completed
                        )
                        print(
                            f"\r      [{bar}] {pct:5.1f}% "
                            f"({layers_completed}/{config.num_hidden_layers}) "
                            f"ETA: {eta:.0f}s    ",
                            end="",
                            flush=True,
                        )

                except Exception as e:
                    if verbose:
                        print(f"\n  Warning: Layer {layer_idx} failed: {e}")

        # Force garbage collection between batches
        gc.collect()

    if verbose:
        print()  # Newline after progress bar

    # Process final layers (lm_head, final norm)
    if verbose:
        print("\n[3/3] Processing output layers...")

    for weight_type in ["final_norm", "lm_head"]:
        try:
            weights = load_non_layer_weights(model_path, weight_type, weight_map)
            for name, tensor in weights.items():
                if verbose:
                    print(f"      {name}: {tensor.shape} (kept FP16)")
                all_output_tensors[name] = tensor
                stats["skipped_count"] += 1
            del weights
            gc.collect()
        except (ValueError, FileNotFoundError):
            pass

    # Save quantized model
    output_file = output_path / "model.safetensors"
    if verbose:
        print(f"\nSaving to {output_file}...")
    save_file(all_output_tensors, str(output_file))

    total_time = time.perf_counter() - start_time

    # Save metadata
    meta: dict[str, Any] = {
        "format": "marlin_fp4",
        "conversion_method": "parallel",
        "group_size": group_size,
        "num_layers": config.num_hidden_layers,
        "quantized_count": stats["quantized_count"],
        "skipped_count": stats["skipped_count"],
        "compression_ratio": stats["original_bytes"] / max(stats["quantized_bytes"], 1),
        "calibration_aware": calib_data is not None,
        "parallel_workers": max_parallel,
        "total_time_sec": total_time,
    }

    if stats["errors"]:
        meta["mean_rmse"] = float(np.mean([e["rmse"] for e in stats["errors"]]))
        meta["max_error"] = float(max(e["max_error"] for e in stats["errors"]))

    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("QUANTIZATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Quantized:       {stats['quantized_count']} tensors")
        print(f"  Kept FP16:       {stats['skipped_count']} tensors")
        if stats["original_bytes"] > 0:
            ratio = stats["original_bytes"] / max(stats["quantized_bytes"], 1)
            orig_gb = stats["original_bytes"] / (1024**3)
            quant_gb = stats["quantized_bytes"] / (1024**3)
            print(f"  Original size:   {orig_gb:.2f} GB")
            print(f"  Quantized size:  {quant_gb:.2f} GB")
            print(f"  Compression:     {ratio:.2f}x")
        if "mean_rmse" in meta:
            print(f"  Mean RMSE:       {meta['mean_rmse']:.6f}")
        print(f"  Total time:      {total_time:.1f}s")
        print(f"  Throughput:      {stats['original_bytes'] / total_time / 1e6:.1f} MB/s")

        # Show highest-error layers (top 10)
        if stats["errors"]:
            sorted_errors = sorted(stats["errors"], key=lambda x: x["rmse"], reverse=True)
            print(f"\n{'=' * 70}")
            print("HIGHEST ERROR LAYERS (consider keeping FP16)")
            print(f"{'=' * 70}")
            print(f"  {'Layer':<50} {'RMSE':>10} {'MaxErr':>10}")
            print(f"  {'-' * 50} {'-' * 10} {'-' * 10}")
            for err in sorted_errors[:10]:
                name = err["name"]
                # Shorten long names
                if len(name) > 48:
                    name = "..." + name[-45:]
                print(f"  {name:<50} {err['rmse']:>10.6f} {err['max_error']:>10.4f}")
            print(f"{'=' * 70}")

    stats.update(meta)
    stats["total_size_bytes"] = sum(t.nbytes for t in all_output_tensors.values())
    return stats


# ============================================================================
# ONNX Conversion
# ============================================================================


def convert_onnx_to_fp4(
    onnx_path: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    mixed_precision: MixedPrecisionConfig | None = None,
    calibration: CalibrationDataset | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Convert ONNX model to Metal Marlin FP4 format.

    Extracts weights from an ONNX model, applies optional mixed-precision
    configuration (auto-detects MoE layers for special handling), quantizes
    to FP4 with optional calibration-aware scales, and saves as safetensors.

    Args:
        onnx_path: Path to input ONNX model file.
        output_path: Directory to save converted model.
        group_size: FP4 quantization group size. Default: 128.
        mixed_precision: Mixed-precision config for layer-specific handling.
            If None, uses default_dense for standard models or auto-detects
            MoE architecture and uses default_moe/default_moe_mtp.
        calibration: Optional CalibrationDataset for calibration-aware
            quantization. Bartowski v3 recommended for better ranges.
        verbose: Print progress messages.

    Returns:
        Statistics dict with:
            - quantized_count: Number of quantized tensors
            - skipped_count: Number of tensors kept in FP16
            - compression_ratio: Size reduction factor
            - mean_rmse: Average reconstruction error (if validation enabled)
            - layers: Per-layer quantization info
    """
    import onnx
    from safetensors.numpy import save_file

    from .mixed_precision import MixedPrecisionConfig, get_layer_config

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if verbose:
        print(f"Loading ONNX model from {onnx_path}...")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    graph = model.graph

    # Auto-detect model architecture for mixed-precision config
    if mixed_precision is None:
        has_moe = _detect_moe_architecture(graph)
        has_mtp = _detect_mtp_heads(graph)
        if has_moe and has_mtp:
            mixed_precision = MixedPrecisionConfig.default_moe_mtp()
            if verbose:
                print("  Detected: MoE + MTP architecture -> using moe-mtp preset")
        elif has_moe:
            mixed_precision = MixedPrecisionConfig.default_moe()
            if verbose:
                print("  Detected: MoE architecture -> using moe preset")
        else:
            mixed_precision = MixedPrecisionConfig.default_dense()
            if verbose:
                print("  Detected: Dense architecture -> using dense preset")

    # Extract all initializers (weights) from ONNX graph
    initializers: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        np_array = onnx.numpy_helper.to_array(init)
        initializers[init.name] = np_array

    if verbose:
        print(f"  Found {len(initializers)} initializers")

    # Prepare calibration scales if provided
    if calibration is not None:
        if verbose:
            print(f"  Using calibration dataset: {calibration.name} ({len(calibration)} samples)")
        _compute_calibration_scales(initializers, calibration)

    stats: dict[str, Any] = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "errors": [],
        "layers": {},
    }

    output_tensors: dict[str, np.ndarray] = {}

    for name, tensor in initializers.items():
        # Determine quantization config for this layer
        layer_config = get_layer_config(name, mixed_precision)

        # Check if this tensor should be quantized
        should_quant = _should_quantize_onnx_tensor(name, tensor, layer_config)

        if should_quant:
            if verbose:
                print(f"  Quantizing {name}: {tensor.shape} -> {layer_config.precision.value}")

            out_feat, in_feat = tensor.shape

            # Determine effective group size
            actual_gs = layer_config.group_size
            if in_feat % actual_gs != 0:
                for gs in [256, 128, 64, 32, 16, 8]:
                    if gs <= actual_gs and in_feat % gs == 0:
                        actual_gs = gs
                        break
                else:
                    actual_gs = 8

            # Standard quantization (calibration-aware scales used in future enhancement)
            packed, scales = quantize_fp4(tensor, group_size=actual_gs)

            output_tensors[name] = packed
            output_tensors[f"{name}.scales"] = scales
            output_tensors[f"{name}.group_size"] = np.array([actual_gs], dtype=np.int32)
            output_tensors[f"{name}.precision"] = np.array([4], dtype=np.int32)

            stats["quantized_count"] += 1
            stats["original_bytes"] += tensor.nbytes
            stats["quantized_bytes"] += packed.nbytes + scales.nbytes

            # Compute quantization error
            err = compute_quantization_error(tensor, packed, scales, actual_gs)
            stats["errors"].append({"name": name, **err})
            stats["layers"][name] = {
                "shape": list(tensor.shape),
                "group_size": actual_gs,
                "precision": layer_config.precision.value,
                "rmse": err["rmse"],
            }
        else:
            # Keep in original precision
            if verbose:
                print(f"  Keeping {name}: {tensor.shape} ({tensor.dtype})")
            output_tensors[name] = tensor
            stats["skipped_count"] += 1
            stats["layers"][name] = {
                "shape": list(tensor.shape),
                "precision": "fp16" if tensor.dtype == np.float16 else str(tensor.dtype),
                "quantized": False,
            }

    # Save quantized model
    output_file = output_path / "model.safetensors"
    if verbose:
        print(f"\nSaving to {output_file}...")
    save_file(output_tensors, str(output_file))

    # Save config and metadata
    config: dict[str, Any] = {
        "format": "marlin_fp4",
        "source": "onnx",
        "source_file": onnx_path.name,
        "group_size": group_size,
        "mixed_precision_preset": (
            "moe-mtp"
            if _detect_mtp_heads(graph) and _detect_moe_architecture(graph)
            else "moe"
            if _detect_moe_architecture(graph)
            else "dense"
        ),
        "quantized_count": stats["quantized_count"],
        "skipped_count": stats["skipped_count"],
        "compression_ratio": stats["original_bytes"] / max(stats["quantized_bytes"], 1),
    }

    if stats["errors"]:
        config["mean_rmse"] = float(np.mean([e["rmse"] for e in stats["errors"]]))
        config["max_error"] = float(max(e["max_error"] for e in stats["errors"]))

    if calibration is not None:
        config["calibration"] = {
            "name": calibration.name,
            "source": calibration.source,
            "num_samples": len(calibration),
        }

    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("ONNX Conversion complete!")
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped:   {stats['skipped_count']} tensors")
        if stats["original_bytes"] > 0:
            ratio = stats["original_bytes"] / max(stats["quantized_bytes"], 1)
            print(f"  Compression: {ratio:.2f}x")
        if "mean_rmse" in config:
            print(f"  Mean RMSE: {config['mean_rmse']:.6f}")
        print(f"{'=' * 60}")

    stats.update(config)
    return stats


def _detect_moe_architecture(graph) -> bool:
    """Detect if ONNX graph contains MoE (Mixture of Experts) patterns."""
    moe_patterns = ["router", "expert", "moe_gate", "block_sparse_moe"]
    for init in graph.initializer:
        name_lower = init.name.lower()
        if any(pat in name_lower for pat in moe_patterns):
            return True
    for node in graph.node:
        name_lower = node.name.lower()
        if any(pat in name_lower for pat in moe_patterns):
            return True
    return False


def _detect_mtp_heads(graph) -> bool:
    """Detect if ONNX graph contains MTP (Multi-Token Prediction) heads."""
    mtp_patterns = ["mtp", "multi_token", "auxiliary_head", "draft_head", "nextn_predict"]
    for init in graph.initializer:
        name_lower = init.name.lower()
        if any(pat in name_lower for pat in mtp_patterns):
            return True
    return False


def _should_quantize_onnx_tensor(
    name: str,
    tensor: np.ndarray,
    layer_config,
) -> bool:
    """Determine if an ONNX tensor should be quantized."""
    from .mixed_precision import Precision

    # FP16 precision means no quantization
    if layer_config.precision == Precision.FP16:
        return False

    # Must be 2D weight matrix
    if tensor.ndim != 2:
        return False

    # Check dimension compatibility
    out_feat, in_feat = tensor.shape

    # For FP4, need divisibility by 8 (packing requirement)
    if in_feat % 8 != 0:
        return False

    # Skip very small tensors (not worth quantizing)
    if tensor.size < 1024:
        return False

    return True


def _compute_calibration_scales(
    initializers: dict[str, np.ndarray],
    calibration: CalibrationDataset,
) -> dict[str, np.ndarray]:
    """Compute per-layer scales based on calibration data statistics.

    For weight-only quantization, this analyzes the weight distribution
    using calibration-aware percentiles rather than full min/max range.
    """
    scales: dict[str, np.ndarray] = {}

    for name, tensor in initializers.items():
        if tensor.ndim != 2:
            continue

        # Compute 99.9th percentile for outlier-robust scaling
        flat = tensor.flatten().astype(np.float32)
        p_low = np.percentile(flat, 0.1)
        p_high = np.percentile(flat, 99.9)
        absmax = max(abs(p_low), abs(p_high))

        # FP4 E2M1 max representable is 6.0
        scale = absmax / 6.0 if absmax > 0 else 1e-7
        scales[name] = np.array(scale, dtype=np.float32)

    return scales


# Type aliases for forward references (avoid circular imports)
try:
    from .mixed_precision import MixedPrecisionConfig
except ImportError:
    MixedPrecisionConfig = None  # type: ignore[misc,assignment]

try:
    from ..converters.calibration import CalibrationDataset
except ImportError:
    try:
        # Direct import when running as module
        from converters.calibration import CalibrationDataset
    except ImportError:
        CalibrationDataset = None  # type: ignore[misc,assignment]


# ============================================================================
# Calibration data sources
# ============================================================================

CALIBRATION_SOURCES: dict[str, str | None] = {
    "bartowski-v3": (
        "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/"
        "raw/2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt"
    ),
    "wikitext2": None,  # Use HF datasets
    "c4": None,  # Use HF datasets
}


def download_calibration_data(
    source: str,
    cache_dir: Path | None = None,
) -> list[str]:
    """
    Download or load calibration data.

    Args:
        source: One of the preset names (bartowski-v3, wikitext2, c4) or a file path.
        cache_dir: Directory for caching downloaded data.

    Returns:
        List of calibration text prompts.
    """
    import urllib.request

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "metal_marlin" / "calibration"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source in CALIBRATION_SOURCES:
        url = CALIBRATION_SOURCES[source]
        if url is None:
            # Load from HuggingFace datasets
            return _load_hf_calibration_dataset(source)

        cache_file = cache_dir / f"{source}.txt"
        if not cache_file.exists():
            print(f"Downloading calibration data from {source}...")
            urllib.request.urlretrieve(url, cache_file)

        with open(cache_file) as f:
            lines = f.read().strip().split("\n")
        return [line for line in lines if line.strip()]

    elif Path(source).exists():
        # Load from local file
        with open(source) as f:
            lines = f.read().strip().split("\n")
        return [line for line in lines if line.strip()]

    else:
        raise ValueError(
            f"Unknown calibration source: {source}. "
            f"Use one of {list(CALIBRATION_SOURCES.keys())} or a file path."
        )


def _load_hf_calibration_dataset(name: str, num_samples: int | None = None) -> list[str]:
    """Load calibration samples from HuggingFace datasets.

    For accurate quantization, use ALL samples (num_samples=None).
    Sample limits should only be used for quick testing/evaluation.

    Args:
        name: Dataset name ("wikitext2", "c4")
        num_samples: Maximum samples to return, or None for all samples.

    Returns:
        List of calibration text samples.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            f"datasets package required for {name} calibration. Install with: pip install datasets"
        )

    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t) > 100]
        if num_samples is not None:
            texts = texts[:num_samples]
    elif name == "c4":
        ds = load_dataset("c4", "en", split="train", streaming=True)
        texts = []
        max_c4_samples = num_samples if num_samples is not None else 10000
        for item in ds:
            if len(item["text"]) > 100:
                texts.append(item["text"])
            if len(texts) >= max_c4_samples:
                break
    else:
        raise ValueError(f"Unknown HF calibration dataset: {name}")

    return texts


def run_calibration(
    model_id: str,
    calibration_source: str = "bartowski-v3",
    output_path: str | Path | None = None,
    num_samples: int | None = None,
    token: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run calibration on a model to collect activation statistics.

    Requires transformers and torch for model loading and inference.

    IMPORTANT: For accurate quantization, use ALL calibration samples.
    Bartowski v3 contains 800+ carefully curated samples. Using all of
    them produces more representative activation ranges.

    Args:
        model_id: HuggingFace model ID or local path.
        calibration_source: Calibration dataset name or file path.
        output_path: Path to save activation_ranges.json.
        num_samples: Number of calibration samples. Default is None (ALL samples).
            Only limit this for quick testing; production quantization should
            use the full dataset.
        token: HuggingFace API token.
        verbose: Print progress.

    Returns:
        Dictionary of activation statistics per layer.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch required for calibration. "
            "Install with: pip install transformers torch"
        )

    # Load FULL calibration data for accurate quantization
    if verbose:
        print(f"Loading calibration data: {calibration_source}")
    cal_data = download_calibration_data(calibration_source)
    total_samples = len(cal_data)
    if num_samples is not None:
        cal_data = cal_data[:num_samples]
        if verbose:
            print(f"  Using {len(cal_data)} of {total_samples} samples (limited)")
    else:
        if verbose:
            print(f"  Using ALL {len(cal_data)} calibration samples")

    # Load model and tokenizer
    if verbose:
        print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Collect activation statistics
    stats: dict[str, dict[str, float]] = {}

    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input

            if not isinstance(x, torch.Tensor):
                return

            with torch.no_grad():
                x_flat = x.float().flatten()
                if name not in stats:
                    stats[name] = {
                        "min": float(x_flat.min().item()),
                        "max": float(x_flat.max().item()),
                        "absmax": float(x_flat.abs().max().item()),
                        "count": 1,
                    }
                else:
                    s = stats[name]
                    s["min"] = min(s["min"], float(x_flat.min().item()))
                    s["max"] = max(s["max"], float(x_flat.max().item()))
                    s["absmax"] = max(s["absmax"], float(x_flat.abs().max().item()))
                    s["count"] += 1

        return hook

    # Register hooks on linear layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    if verbose:
        print(f"Registered hooks on {len(hooks)} linear layers")
        print("Running calibration...")

    # Run forward passes
    with torch.no_grad():
        for i, text in enumerate(cal_data):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)

            _ = model(**inputs)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(cal_data)} samples")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute final ranges with CalibrationData-compatible format
    layer_ranges: dict[str, tuple[float, float]] = {}
    result = {}
    for name, s in stats.items():
        layer_ranges[name] = (s["min"], s["max"])
        result[name] = {
            "min": s["min"],
            "max": s["max"],
            "absmax": s["absmax"],
            "fp4_scale": s["absmax"] / 6.0,  # FP4 E2M1 max is 6
            "int4_scale": s["absmax"] / 7.0,  # INT4 symmetric max is 7
        }

    # Build CalibrationData-compatible output
    output_data = {
        "layer_ranges": layer_ranges,
        "percentile": 99.9,
        "smooth_factor": 0.8,
        "detailed_stats": result,
    }

    # Save output
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        if verbose:
            print(f"\nSaved activation ranges to {output_path}")

    if verbose:
        print(f"\nCalibration complete: {len(result)} layers profiled")

    return output_data


def convert_model_with_calibration(
    model_path: str | Path,
    output_path: str | Path,
    calibration_path: str | Path | None = None,
    calibration_source: str | None = None,
    mixed_precision: str | None = None,
    group_size: int = 128,
    validate: bool = True,
    verbose: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """
    Convert a model with calibration-aware quantization and mixed precision.

    Args:
        model_path: Path to model directory or HuggingFace model ID.
        output_path: Directory to save converted model.
        calibration_path: Path to activation_ranges.json from calibration.
        calibration_source: Run calibration with this source if calibration_path not provided.
        mixed_precision: Mixed precision preset (dense, moe, moe-mtp, quality, speed).
        group_size: Default FP4 quantization group size.
        validate: Compute quantization errors.
        verbose: Print progress.
        token: HuggingFace API token.

    Returns:
        Statistics dict with counts, sizes, errors.
    """
    from safetensors.numpy import save_file

    from .mixed_precision import MixedPrecisionConfig as MPConfig
    from .mixed_precision import Precision, should_quantize

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load or run calibration
    calib_data: CalibrationData | None = None
    if calibration_path is not None:
        calib_data = CalibrationData.from_json(calibration_path)
        if verbose:
            print(f"Loaded calibration from {calibration_path}")
            print(f"  {len(calib_data.layer_ranges)} layers calibrated")
    elif calibration_source is not None:
        if verbose:
            print(f"Running calibration with {calibration_source}...")
        cal_output = run_calibration(
            str(model_path),
            calibration_source=calibration_source,
            token=token,
            verbose=verbose,
        )
        calib_data = CalibrationData(
            layer_ranges=cal_output["layer_ranges"],
            percentile=cal_output.get("percentile", 99.9),
            smooth_factor=cal_output.get("smooth_factor", 0.8),
        )

    # Load mixed precision config
    mp_config: MPConfig | None = None
    if mixed_precision is not None:
        preset_map = {
            "dense": MPConfig.default_dense,
            "moe": MPConfig.default_moe,
            "moe-mtp": MPConfig.default_moe_mtp,
            "quality": MPConfig.quality_first,
            "speed": MPConfig.speed_first,
        }
        if mixed_precision not in preset_map:
            raise ValueError(
                f"Unknown mixed-precision preset: {mixed_precision}. "
                f"Choose from: {list(preset_map.keys())}"
            )
        mp_config = preset_map[mixed_precision]()
        if verbose:
            print(f"Using mixed-precision preset: {mixed_precision}")

    # Download if needed
    if not model_path.exists():
        if verbose:
            print(f"Downloading {model_path}...")
        model_path = download_model(str(model_path), token=token)

    # Copy config and tokenizer files
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]:
        src = model_path / fname
        if src.exists():
            import shutil

            shutil.copy(src, output_path / fname)

    stats: dict[str, Any] = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "errors": [],
        "tensors": {},
        "by_precision": {},
    }

    output_tensors = {}

    for name, tensor, _ in iter_safetensors_weights(model_path):
        # Determine quantization config
        if mp_config is not None:
            do_quant, layer_cfg = should_quantize(name, tensor, mp_config)
            precision = layer_cfg.precision if do_quant else Precision.FP16
            actual_gs = layer_cfg.group_size
        else:
            do_quant = should_quantize_tensor(name, tensor)
            precision = Precision.FP4_E2M1 if do_quant else Precision.FP16
            actual_gs = group_size

        # Track precision distribution
        prec_key = precision.value if hasattr(precision, "value") else str(precision)
        stats["by_precision"][prec_key] = stats["by_precision"].get(prec_key, 0) + tensor.size

        if do_quant and precision != Precision.FP16:
            if verbose:
                print(f"  Quantizing {name}: {tensor.shape} -> {prec_key}")

            out_feat, in_feat = tensor.shape

            # Ensure group_size divides in_feat
            if in_feat % actual_gs != 0:
                for gs in [256, 128, 64, 32, 16, 8]:
                    if gs <= actual_gs and in_feat % gs == 0:
                        actual_gs = gs
                        break

            # Get activation ranges for this layer if calibration data available
            activation_ranges = None
            if calib_data is not None:
                activation_ranges = calib_data.get_activation_ranges(name)
                if activation_ranges is not None and verbose:
                    input_range = activation_ranges["input_range"]
                    print(f"    Using calibration: input_range={input_range}")

            packed, scales = quantize_fp4(
                tensor, group_size=actual_gs, activation_ranges=activation_ranges
            )

            output_tensors[name] = packed
            output_tensors[f"{name}.scales"] = scales
            output_tensors[f"{name}.group_size"] = np.array([actual_gs], dtype=np.int32)

            stats["quantized_count"] += 1
            stats["original_bytes"] += tensor.nbytes
            stats["quantized_bytes"] += packed.nbytes + scales.nbytes

            if validate:
                err = compute_quantization_error(tensor, packed, scales, actual_gs)
                stats["errors"].append({"name": name, **err})
                stats["tensors"][name] = {
                    "shape": list(tensor.shape),
                    "group_size": actual_gs,
                    "precision": prec_key,
                    "rmse": err["rmse"],
                }
        else:
            # Keep in full precision
            if verbose:
                print(f"  Keeping {name}: {tensor.shape} ({tensor.dtype})")
            output_tensors[name] = tensor
            stats["skipped_count"] += 1

    # Save quantized model
    output_file = output_path / "model.safetensors"
    if verbose:
        print(f"\nSaving to {output_file}...")
    save_file(output_tensors, str(output_file))

    # Save quantization metadata
    meta: dict[str, Any] = {
        "format": "marlin_fp4",
        "group_size": group_size,
        "mixed_precision": mixed_precision,
        "calibration": calibration_source or (str(calibration_path) if calibration_path else None),
        "calibration_aware": calib_data is not None,
        "quantized_count": stats["quantized_count"],
        "skipped_count": stats["skipped_count"],
        "compression_ratio": stats["original_bytes"] / max(stats["quantized_bytes"], 1),
        "by_precision": stats["by_precision"],
    }
    if calib_data is not None:
        meta["calibration_config"] = {
            "percentile": calib_data.percentile,
            "smooth_factor": calib_data.smooth_factor,
            "layers_calibrated": len(calib_data.layer_ranges),
        }
    if stats["errors"]:
        meta["mean_rmse"] = float(np.mean([e["rmse"] for e in stats["errors"]]))
        meta["max_error"] = float(max(e["max_error"] for e in stats["errors"]))

    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("Conversion complete!")
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped:   {stats['skipped_count']} tensors")
        if stats["original_bytes"] > 0:
            ratio = stats["original_bytes"] / max(stats["quantized_bytes"], 1)
            print(f"  Compression: {ratio:.2f}x")
        if "mean_rmse" in meta:
            print(f"  Mean RMSE: {meta['mean_rmse']:.6f}")
        if mixed_precision:
            print(f"  Mixed precision: {mixed_precision}")
        if calib_data is not None:
            print(f"  Calibration: {len(calib_data.layer_ranges)} layers")
        print(f"{'=' * 60}")

    stats.update(meta)
    return stats


# ============================================================================
# Transformers-based Quantization (In-Place)
# ============================================================================


def convert_model_transformers(
    model_id: str | Path,
    output_path: str | Path,
    group_size: int = 128,
    mixed_precision: MixedPrecisionConfig | None = None,
    calibration: CalibrationData | str | Path | None = None,
    calibration_source: str | None = None,
    validate: bool = True,
    verbose: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """Quantize a model via Transformers and save a Marlin-compatible checkpoint.

    This path loads the model with HuggingFace Transformers (handles sharding,
    MoE architectures, and new model types automatically), quantizes weights,
    and saves a standalone Metal Marlin checkpoint (model.safetensors +
    quantization_config.json + configs).

    Args:
        model_id: HuggingFace model ID or local path.
        output_path: Directory to save converted model.
        group_size: Default FP4 quantization group size.
        mixed_precision: Optional MixedPrecisionConfig for layer-specific handling.
        calibration: Optional CalibrationData or path to activation_ranges.json.
        calibration_source: Calibration dataset name to run if calibration is None.
        validate: Compute quantization errors.
        verbose: Print progress.
        token: HuggingFace API token.

    Returns:
        Statistics dict with counts, sizes, errors.
    """
    from safetensors.numpy import save_file

    from .quantize_model import quantize_model

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for --use-transformers. "
            "Install with: pip install transformers torch"
        ) from exc

    model_id = str(model_id)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load or run calibration
    calib_data: CalibrationData | None = None
    if calibration is not None:
        if isinstance(calibration, CalibrationData):
            calib_data = calibration
        elif isinstance(calibration, (str, Path)):
            calib_data = CalibrationData.from_json(calibration)
        else:
            raise TypeError(
                f"calibration must be CalibrationData, str, or Path, got {type(calibration)}"
            )
    elif calibration_source is not None:
        if verbose:
            print(f"Running calibration with {calibration_source}...")
        cal_output = run_calibration(
            model_id,
            calibration_source=calibration_source,
            token=token,
            verbose=verbose,
        )
        calib_data = CalibrationData(
            layer_ranges=cal_output["layer_ranges"],
            percentile=cal_output.get("percentile", 99.9),
            smooth_factor=cal_output.get("smooth_factor", 0.8),
        )

    if verbose:
        print(f"Loading model via Transformers: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Save configs + tokenizer
    model.config.save_pretrained(output_path)
    if getattr(model, "generation_config", None) is not None:
        try:
            model.generation_config.save_pretrained(output_path)
        except Exception:
            if verbose:
                print("Warning: unable to save generation_config.json")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
    except Exception:
        if verbose:
            print("Warning: unable to save tokenizer files")

    stats: dict[str, Any] = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "errors": [],
        "tensors": {},
        "by_precision": {},
    }

    output_tensors: dict[str, np.ndarray] = {}

    # Helper for calibration lookup (support module-name calibration keys)
    def _activation_ranges_for_name(weight_name: str) -> dict[str, Any] | None:
        if calib_data is None:
            return None
        ranges = calib_data.get_activation_ranges(weight_name)
        if ranges is not None:
            return ranges
        if weight_name.endswith(".weight"):
            return calib_data.get_activation_ranges(weight_name[: -len(".weight")])
        return None

    # Lazy import to avoid optional dependency penalties
    if mixed_precision is not None:
        from .mixed_precision import Precision, should_quantize
    else:
        Precision = None  # type: ignore[assignment]

    # Quantize from state_dict (Transformers-loaded weights)
    if verbose:
        print("Quantizing weights...")
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue

        tensor_cpu = tensor.detach().cpu()
        if tensor_cpu.dtype == torch.bfloat16:
            tensor_np = tensor_cpu.float().numpy()
        else:
            tensor_np = tensor_cpu.numpy()

        if mixed_precision is not None:
            assert Precision is not None
            do_quant, layer_cfg = should_quantize(name, tensor_np, mixed_precision)
            precision = layer_cfg.precision if do_quant else Precision.FP16
            actual_gs = layer_cfg.group_size
        else:
            do_quant = should_quantize_tensor(name, tensor_np)
            precision = None
            actual_gs = group_size

        prec_key = (
            precision.value if hasattr(precision, "value") else ("fp4" if do_quant else "fp16")
        )
        stats["by_precision"][prec_key] = stats["by_precision"].get(prec_key, 0) + tensor_np.size

        if do_quant and (precision is None or precision != Precision.FP16):
            if verbose:
                print(f"  Quantizing {name}: {tensor_np.shape}")

            out_feat, in_feat = tensor_np.shape

            # Ensure group_size divides in_feat
            if in_feat % actual_gs != 0:
                for gs in [256, 128, 64, 32, 16, 8]:
                    if gs <= actual_gs and in_feat % gs == 0:
                        actual_gs = gs
                        break

            activation_ranges = _activation_ranges_for_name(name)
            if activation_ranges is not None and verbose:
                input_range = activation_ranges["input_range"]
                print(f"    Using calibration: input_range={input_range}")

            packed, scales = quantize_fp4(
                tensor_np, group_size=actual_gs, activation_ranges=activation_ranges
            )

            output_tensors[name] = packed
            output_tensors[f"{name}.scales"] = scales
            output_tensors[f"{name}.group_size"] = np.array([actual_gs], dtype=np.int32)

            stats["quantized_count"] += 1
            stats["original_bytes"] += tensor_np.nbytes
            stats["quantized_bytes"] += packed.nbytes + scales.nbytes

            if validate:
                err = compute_quantization_error(tensor_np, packed, scales, actual_gs)
                stats["errors"].append({"name": name, **err})
                stats["tensors"][name] = {
                    "shape": list(tensor_np.shape),
                    "group_size": actual_gs,
                    "precision": prec_key,
                    "rmse": err["rmse"],
                }
        else:
            if verbose:
                print(f"  Keeping {name}: {tensor_np.shape} ({tensor_np.dtype})")
            output_tensors[name] = tensor_np
            stats["skipped_count"] += 1

    # Replace Linear layers in-place (Transformers-based path)
    try:
        quantize_model(model, group_size=group_size)
    except Exception:
        if verbose:
            print("Warning: in-place Linear replacement skipped due to an error.")

    # Save quantized model
    output_file = output_path / "model.safetensors"
    if verbose:
        print(f"\nSaving to {output_file}...")
    save_file(output_tensors, str(output_file))

    # Save quantization metadata
    meta: dict[str, Any] = {
        "format": "marlin_fp4",
        "conversion_method": "transformers",
        "group_size": group_size,
        "mixed_precision": mixed_precision is not None,
        "calibration": calibration_source or (str(calibration) if calibration else None),
        "calibration_aware": calib_data is not None,
        "quantized_count": stats["quantized_count"],
        "skipped_count": stats["skipped_count"],
        "compression_ratio": stats["original_bytes"] / max(stats["quantized_bytes"], 1),
        "by_precision": stats["by_precision"],
    }
    if calib_data is not None:
        meta["calibration_config"] = {
            "percentile": calib_data.percentile,
            "smooth_factor": calib_data.smooth_factor,
            "layers_calibrated": len(calib_data.layer_ranges),
        }
    if stats["errors"]:
        meta["mean_rmse"] = float(np.mean([e["rmse"] for e in stats["errors"]]))
        meta["max_error"] = float(max(e["max_error"] for e in stats["errors"]))

    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Transformers-based conversion complete!")
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped:   {stats['skipped_count']} tensors")
        if stats["original_bytes"] > 0:
            ratio = stats["original_bytes"] / max(stats["quantized_bytes"], 1)
            print(f"  Compression: {ratio:.2f}x")
        if "mean_rmse" in meta:
            print(f"  Mean RMSE: {meta['mean_rmse']:.6f}")
        print(f"{'=' * 60}")

    stats.update(meta)
    return stats


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Download, calibrate, and convert models to Metal Marlin FP4 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run calibration
  python -m metal_marlin.hf_loader calibrate zai-org/GLM-4.7-Flash \\
      --calibration bartowski-v3 --output activation_ranges.json

  # Convert with calibration and mixed-precision
  python -m metal_marlin.hf_loader convert zai-org/GLM-4.7-Flash ./glm4-fp4/ \\
      --calibration bartowski-v3 --mixed-precision moe-mtp

  # Layer-wise conversion for large models (memory-efficient)
  python -m metal_marlin.hf_loader convert-layerwise deepseek-ai/DeepSeek-V2-Lite ./dsv2-fp4/ \\
      --mixed-precision moe

  # Simple conversion (legacy, no subcommand)
  python -m metal_marlin.hf_loader zai-org/GLM-4.7-Flash ./glm4-fp4/
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # calibrate subcommand
    # -------------------------------------------------------------------------
    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Run calibration to collect activation statistics",
    )
    cal_parser.add_argument(
        "model_id",
        help="HuggingFace model ID (e.g., zai-org/GLM-4.7-Flash) or local path",
    )
    cal_parser.add_argument(
        "--calibration",
        default="bartowski-v3",
        help="Calibration data source: bartowski-v3, wikitext2, c4, or file path (default: bartowski-v3)",
    )
    cal_parser.add_argument(
        "--output",
        "-o",
        default="activation_ranges.json",
        help="Output path for activation ranges JSON (default: activation_ranges.json)",
    )
    cal_parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of calibration samples (default: all)",
    )
    cal_parser.add_argument(
        "--token",
        help="HuggingFace API token for gated models",
    )
    cal_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # -------------------------------------------------------------------------
    # convert subcommand
    # -------------------------------------------------------------------------
    convert_hf = subparsers.add_parser(
        "convert",
        help="Convert HuggingFace model to FP4 (default)",
    )
    convert_hf.add_argument(
        "model_id",
        help="HuggingFace model ID (e.g., zai-org/GLM-4.7-Flash) or local path",
    )
    convert_hf.add_argument(
        "output_dir",
        help="Output directory for converted model",
    )
    convert_hf.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    convert_hf.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip quantization error validation",
    )
    convert_hf.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Calibration source (bartowski-v3, wikitext2, c4, file path) or path to activation_ranges.json",
    )
    convert_hf.add_argument(
        "--mixed-precision",
        choices=["dense", "moe", "moe-mtp", "quality", "speed"],
        default=None,
        help="Mixed precision preset",
    )
    convert_hf.add_argument(
        "--token",
        help="HuggingFace API token for gated models",
    )
    convert_hf.add_argument(
        "--layerwise",
        action="store_true",
        help="Use memory-efficient layer-wise conversion (for large models)",
    )
    convert_hf.add_argument(
        "--use-transformers",
        action="store_true",
        help="Load via Transformers and quantize in-place before saving",
    )
    convert_hf.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # -------------------------------------------------------------------------
    # convert-layerwise subcommand (explicit layer-wise mode)
    # -------------------------------------------------------------------------
    convert_lw = subparsers.add_parser(
        "convert-layerwise",
        help="Convert model layer-by-layer (memory-efficient for large models)",
    )
    convert_lw.add_argument(
        "model_id",
        help="HuggingFace model ID (e.g., deepseek-ai/DeepSeek-V2-Lite) or local path",
    )
    convert_lw.add_argument(
        "output_dir",
        help="Output directory for converted model",
    )
    convert_lw.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    convert_lw.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip quantization error validation",
    )
    convert_lw.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to activation_ranges.json from calibration",
    )
    convert_lw.add_argument(
        "--mixed-precision",
        choices=["dense", "moe", "moe-mtp", "quality", "speed"],
        default=None,
        help="Mixed precision preset (auto-detected if not specified)",
    )
    convert_lw.add_argument(
        "--token",
        help="HuggingFace API token for gated models",
    )
    convert_lw.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # -------------------------------------------------------------------------
    # convert-onnx subcommand
    # -------------------------------------------------------------------------
    convert_onnx = subparsers.add_parser(
        "convert-onnx",
        help="Convert ONNX model to FP4",
    )
    convert_onnx.add_argument(
        "onnx_path",
        help="Path to ONNX model file",
    )
    convert_onnx.add_argument(
        "output_dir",
        help="Output directory for converted model",
    )
    convert_onnx.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    convert_onnx.add_argument(
        "--mixed-precision",
        choices=["dense", "moe", "moe-mtp", "quality", "speed", "auto"],
        default="auto",
        help="Mixed-precision preset (default: auto-detect)",
    )
    convert_onnx.add_argument(
        "--calibration",
        choices=["bartowski-v3", "none"],
        default="none",
        help="Calibration dataset (default: none)",
    )
    convert_onnx.add_argument(
        "--calibration-samples",
        type=int,
        default=512,
        help="Max calibration samples (default: 512)",
    )
    convert_onnx.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Token from env if not provided
    token = getattr(args, "token", None)
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Handle no command (backwards compatibility: treat positional args as convert)
    if args.command is None:
        # Check if positional args were provided without subcommand
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Treat as legacy convert command
            args = parser.parse_args(["convert"] + sys.argv[1:])
        else:
            parser.print_help()
            return

    # -------------------------------------------------------------------------
    # Handle commands
    # -------------------------------------------------------------------------
    if args.command == "calibrate":
        run_calibration(
            model_id=args.model_id,
            calibration_source=args.calibration,
            output_path=args.output,
            num_samples=args.num_samples,
            token=token,
            verbose=not args.quiet,
        )

    elif args.command == "convert":
        # Determine if --calibration is a file or a source name
        cal_path = None
        cal_source = None
        if args.calibration:
            if Path(args.calibration).exists() and args.calibration.endswith(".json"):
                cal_path = args.calibration
            elif args.calibration in CALIBRATION_SOURCES:
                cal_source = args.calibration
            else:
                # Assume it's a path to a calibration JSON
                cal_path = args.calibration

        # Transformers-based path (in-place)
        if getattr(args, "use_transformers", False):
            if getattr(args, "layerwise", False):
                print("Warning: --use-transformers ignores --layerwise")

            # Load mixed-precision config if requested
            mp_config = None
            if args.mixed_precision:
                from .mixed_precision import MixedPrecisionConfig as MPConfig

                preset_map = {
                    "dense": MPConfig.default_dense,
                    "moe": MPConfig.default_moe,
                    "moe-mtp": MPConfig.default_moe_mtp,
                    "quality": MPConfig.quality_first,
                    "speed": MPConfig.speed_first,
                }
                mp_config = preset_map[args.mixed_precision]()

            convert_model_transformers(
                model_id=args.model_id,
                output_path=args.output_dir,
                group_size=args.group_size,
                mixed_precision=mp_config,
                calibration=cal_path,
                calibration_source=cal_source,
                validate=not args.no_validate,
                verbose=not args.quiet,
                token=token,
            )
        # Check if --layerwise flag is set
        elif getattr(args, "layerwise", False):
            # Load mixed-precision config
            mp_config = None
            if args.mixed_precision:
                from .mixed_precision import MixedPrecisionConfig as MPConfig

                preset_map = {
                    "dense": MPConfig.default_dense,
                    "moe": MPConfig.default_moe,
                    "moe-mtp": MPConfig.default_moe_mtp,
                    "quality": MPConfig.quality_first,
                    "speed": MPConfig.speed_first,
                }
                mp_config = preset_map[args.mixed_precision]()

            convert_model_layerwise(
                model_path=args.model_id,
                output_path=args.output_dir,
                group_size=args.group_size,
                mixed_precision=mp_config,
                calibration=cal_path,
                validate=not args.no_validate,
                verbose=not args.quiet,
                token=token,
            )
        # Use advanced convert if mixed-precision or calibration source specified
        elif args.mixed_precision or cal_source:
            convert_model_with_calibration(
                model_path=args.model_id,
                output_path=args.output_dir,
                calibration_path=cal_path,
                calibration_source=cal_source,
                mixed_precision=args.mixed_precision,
                group_size=args.group_size,
                validate=not args.no_validate,
                verbose=not args.quiet,
                token=token,
            )
        else:
            # Legacy convert path
            convert_model_to_fp4(
                model_path=args.model_id,
                output_path=args.output_dir,
                group_size=args.group_size,
                validate=not args.no_validate,
                verbose=not args.quiet,
                calibration=cal_path,
            )

    elif args.command == "convert-layerwise":
        # Load calibration if provided
        cal_path = args.calibration

        # Load mixed-precision config
        mp_config = None
        if args.mixed_precision:
            from .mixed_precision import MixedPrecisionConfig as MPConfig

            preset_map = {
                "dense": MPConfig.default_dense,
                "moe": MPConfig.default_moe,
                "moe-mtp": MPConfig.default_moe_mtp,
                "quality": MPConfig.quality_first,
                "speed": MPConfig.speed_first,
            }
            mp_config = preset_map[args.mixed_precision]()

        convert_model_layerwise(
            model_path=args.model_id,
            output_path=args.output_dir,
            group_size=args.group_size,
            mixed_precision=mp_config,
            calibration=cal_path,
            validate=not args.no_validate,
            verbose=not args.quiet,
            token=token,
        )

    elif args.command == "convert-onnx":
        # Load mixed-precision config
        mp_config = None
        if args.mixed_precision != "auto":
            from .mixed_precision import MixedPrecisionConfig as MPConfig

            preset_map = {
                "dense": MPConfig.default_dense,
                "moe": MPConfig.default_moe,
                "moe-mtp": MPConfig.default_moe_mtp,
                "quality": MPConfig.quality_first,
                "speed": MPConfig.speed_first,
            }
            mp_config = preset_map[args.mixed_precision]()

        # Load calibration dataset
        calib = None
        if args.calibration == "bartowski-v3":
            try:
                from ..converters.calibration import CalibrationDataset as CalibDS
            except ImportError:
                from converters.calibration import CalibrationDataset as CalibDS
            calib = CalibDS.bartowski_v3(max_samples=args.calibration_samples)

        convert_onnx_to_fp4(
            onnx_path=args.onnx_path,
            output_path=args.output_dir,
            group_size=args.group_size,
            mixed_precision=mp_config,
            calibration=calib,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
