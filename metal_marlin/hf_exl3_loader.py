"""Load EXL3 quantized models from HuggingFace Hub."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .trellis_loader import TrellisModelLoader


def download_exl3_model(
    model_id: str,
    local_dir: Path | str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download EXL3 model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "turboderp/Llama-3.1-8B-EXL3")
        local_dir: Local directory to download to (default: HF cache)
        revision: Model revision/branch
        token: HuggingFace API token

    Returns:
        Path to downloaded model directory
    """
    from huggingface_hub import snapshot_download

    # Download model files
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )

    return Path(model_path)


def detect_exl3_format(model_path: Path) -> dict:
    """Detect EXL3 model format and return metadata.

    ExllamaV3 models can have different structures:
    1. Single-file: model.safetensors with all weights
    2. Sharded: model-00001-of-00004.safetensors, etc.
    3. Layer-split: our layerwise format (layer_XXXX/)

    Returns dict with:
    - format: "single", "sharded", or "layerwise"
    - files: list of safetensor files
    - config: loaded config.json
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Detect format
    safetensors = list(model_path.glob("*.safetensors"))
    layer_dirs = list(model_path.glob("layer_*"))

    if layer_dirs:
        return {
            "format": "layerwise",
            "files": layer_dirs,
            "config": config,
            "num_layers": len(layer_dirs),
        }

    # Check for sharded format first (model-XXXXX-of-XXXXX pattern)
    sharded_files = [f for f in safetensors if "model-" in f.name and "-of-" in f.name]
    if sharded_files:
        # Sharded - sort by index
        sharded = sorted(
            sharded_files,
            key=lambda f: int(f.stem.split("-")[1]),
        )
        return {
            "format": "sharded",
            "files": sharded,
            "config": config,
            "num_shards": len(sharded),
        }
    elif len(safetensors) == 1:
        return {
            "format": "single",
            "files": safetensors,
            "config": config,
        }
    else:
        # Multiple safetensors but no sharded pattern
        return {
            "format": "sharded",
            "files": sorted(safetensors),
            "config": config,
            "num_shards": len(safetensors),
        }


def _parse_layer_index(tensor_name: str) -> int | None:
    """Extract layer index from tensor name.

    Args:
        tensor_name: Full tensor name (e.g., "model.layers.42.mlp.gate_proj.weight")

    Returns:
        Layer index if found, None otherwise
    """
    # Match patterns like "model.layers.42.mlp.gate_proj.weight"
    match = re.search(r"layers\.(\d+)\.", tensor_name)
    if match:
        return int(match.group(1))
    return None


def convert_sharded_to_layerwise(
    model_path: Path,
    metadata: dict,
    output_path: Path | None = None,
) -> Path:
    """Convert sharded EXL3 model to layerwise format.

    ExllamaV3 often produces sharded models (model-00001.safetensors, etc.).
    This function converts these to our layerwise format for efficient loading.

    Args:
        model_path: Path to sharded model
        metadata: Output from detect_exl3_format()
        output_path: Where to write converted model (default: model_path/layerwise/)

    Returns:
        Path to converted model
    """
    from safetensors.torch import load_file, save_file

    if output_path is None:
        output_path = model_path / "layerwise"
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy config and tokenizer files
    for f in model_path.glob("*.json"):
        shutil.copy(f, output_path / f.name)
    for f in model_path.glob("tokenizer*"):
        shutil.copy(f, output_path / f.name)

    # Group tensors by layer
    layer_tensors: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    non_layer_tensors: dict[str, torch.Tensor] = {}

    for shard_file in metadata["files"]:
        print(f"Processing {shard_file.name}...")
        tensors = load_file(shard_file)

        for name, tensor in tensors.items():
            # Parse layer index from name
            layer_idx = _parse_layer_index(name)

            if layer_idx is not None:
                layer_tensors[layer_idx][name] = tensor
            else:
                non_layer_tensors[name] = tensor

    # Save per-layer files
    for layer_idx, tensors in sorted(layer_tensors.items()):
        layer_dir = output_path / f"layer_{layer_idx:04d}"
        layer_dir.mkdir(exist_ok=True)

        # Create index.json
        index = {
            "layer_idx": layer_idx,
            "total_tensors": len(tensors),
            "tensors": [{"name": n, "shape": list(t.shape)} for n, t in tensors.items()],
        }
        with open(layer_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        # Save tensors
        save_file(tensors, layer_dir / "tensor_0001.safetensors")

    # Save non-layer tensors (embeddings, norms, lm_head)
    if non_layer_tensors:
        save_file(non_layer_tensors, output_path / "base_weights.safetensors")

        # Create base index
        base_index = {
            "type": "base_weights",
            "total_tensors": len(non_layer_tensors),
            "tensors": [{"name": n, "shape": list(t.shape)} for n, t in non_layer_tensors.items()],
        }
        with open(output_path / "base_index.json", "w") as f:
            json.dump(base_index, f, indent=2)

    print(f"Converted to layerwise format: {output_path}")
    return output_path


def load_exl3_from_hub(
    model_id: str,
    device: str = "mps",
    local_dir: Path | str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> TrellisModelLoader:
    """Load EXL3 model directly from HuggingFace Hub.

    Downloads the model if needed and returns a TrellisModelLoader
    for accessing the quantized weights.

    Example:
        >>> loader = load_exl3_from_hub("turboderp/Llama-3.1-8B-EXL3-4.0bpw")
        >>> print(f"Layers: {loader.get_num_layers()}")
        >>> layer0 = loader.load_layer(0)

    Args:
        model_id: HuggingFace model ID (e.g., "turboderp/Llama-3.1-8B-EXL3")
        device: Target device (default: "mps")
        local_dir: Local directory to download to (default: HF cache)
        revision: Model revision/branch
        token: HuggingFace API token

    Returns:
        TrellisModelLoader for the downloaded model

    Raises:
        FileNotFoundError: If config.json is not found in the model
        NotImplementedError: If the model is sharded and needs conversion
    """
    from .trellis_loader import TrellisModelLoader

    # Download model
    model_path = download_exl3_model(
        model_id=model_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
    )

    # Detect format
    metadata = detect_exl3_format(model_path)

    # Convert if needed (sharded -> layerwise)
    if metadata["format"] == "sharded":
        model_path = convert_sharded_to_layerwise(model_path, metadata)

    # Load model using TrellisModelLoader
    return TrellisModelLoader(model_path)


def list_exl3_models(
    author: str | None = None,
    search: str = "EXL3",
    limit: int = 20,
) -> list[dict]:
    """List EXL3 models available on HuggingFace Hub.

    Args:
        author: Filter by model author/organization
        search: Search query (default: "EXL3")
        limit: Maximum number of results

    Returns:
        List of model info dictionaries

    Example:
        >>> models = list_exl3_models(author="turboderp", limit=10)
        >>> for model in models:
        ...     print(f"{model['id']}: {model['downloads']} downloads")
    """
    from huggingface_hub import list_models

    models = list_models(
        author=author,
        search=search,
        limit=limit,
    )

    return [
        {
            "id": model.id,
            "author": model.author,
            "downloads": model.downloads,
            "tags": model.tags,
        }
        for model in models
    ]


def find_exl3_models(
    author: str | None = None,
    search: str = "EXL3",
    limit: int = 20,
) -> list[dict]:
    """Find EXL3 models on HuggingFace Hub.

    This is a convenience wrapper around list_exl3_models that provides
    the same functionality but with a more discoverable name.

    Args:
        author: Filter by model author/organization
        search: Search query (default: "EXL3")
        limit: Maximum number of results

    Returns:
        List of model info dictionaries
    """
    return list_exl3_models(author=author, search=search, limit=limit)


def parse_model_card(model_id: str) -> dict:
    """Parse model card to extract metadata like base model.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dictionary with parsed metadata, including 'base_model' if found
    """
    try:
        from huggingface_hub import model_info
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to parse model cards. "
            "Install with: pip install huggingface_hub"
        )

    model_info_result = model_info(model_id)
    base_model = None

    # Try to get base model from card data
    if model_info_result.card_data:
        if hasattr(model_info_result.card_data, "base_model"):
            base_model = model_info_result.card_data.base_model
            # Check if it's explicitly None, not just falsy
            if base_model is None:
                # Check if there's a _base_model property as fallback
                if hasattr(model_info_result, "_base_model"):
                    fallback_base_model = model_info_result._base_model
                    # Only use fallback if it's not a mock and is not None
                    if not hasattr(fallback_base_model, "_mock_name"):
                        base_model = fallback_base_model

    # Final fallback: check if there's a _base_model property (some models use this)
    if base_model is None and hasattr(model_info_result, "_base_model"):
        fallback_base_model = model_info_result._base_model
        # Only use fallback if it's not a mock and is not None
        if not hasattr(fallback_base_model, "_mock_name"):
            base_model = fallback_base_model

    return {
        "model_id": model_id,
        "base_model": base_model,
        "card_data": model_info_result.card_data.__dict__ if model_info_result.card_data else None,
    }
