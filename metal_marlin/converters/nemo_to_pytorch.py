"""Convert NeMo checkpoints to standard PyTorch format.

NeMo models use a custom checkpoint format that wraps standard PyTorch
state_dict with additional metadata. This converter extracts:

- Encoder weights (Conformer layers)
- Decoder weights (TDT joint network)
- Preprocessor config (mel spectrogram parameters)

The converter handles both ASR and TTS models from NeMo's speech collection.

Example:
    from metal_marlin.converters.nemo_to_pytorch import convert_nemo_to_pytorch

    # Convert .nemo to .pt
    convert_nemo_to_pytorch(
        nemo_path=Path("model.nemo"),
        output_path=Path("model.pt")
    )

    # Convert and extract config only
    convert_nemo_to_pytorch(
        nemo_path=Path("model.nemo"),
        output_path=Path("model.pt"),
        extract_config_only=True,
        config_output=Path("config.json")
    )
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


def _extract_nemo_archive(nemo_path: Path, extract_dir: Path) -> dict[str, Any]:
    """Extract .nemo archive and return configuration.

    NeMo checkpoints are tar.gz archives containing:
    - model_weights.pt: PyTorch state_dict
    - model_config.yaml: Model configuration

    Args:
        nemo_path: Path to .nemo file
        extract_dir: Directory to extract contents

    Returns:
        Configuration dictionary from model_config.yaml
    """
    with tarfile.open(nemo_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    # Load model configuration
    config_path = extract_dir / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("model_config.yaml not found in NeMo archive")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def _load_nemo_state_dict(extract_dir: Path) -> dict[str, torch.Tensor]:
    """Load PyTorch state_dict from extracted NeMo archive.

    Args:
        extract_dir: Directory containing extracted NeMo files

    Returns:
        Raw state_dict from NeMo checkpoint
    """
    weights_path = extract_dir / "model_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError("model_weights.pt not found in NeMo archive")

    return torch.load(weights_path, map_location="cpu")


def _map_encoder_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map NeMo Conformer encoder weights to standard format.

    NeMo uses prefixes like 'encoder.' that may conflict with
    standard PyTorch loading. This function removes unnecessary prefixes.

    Args:
        state_dict: Raw state_dict from NeMo

    Returns:
        Mapped state_dict with standardized encoder weights
    """
    mapped = {}

    for key, tensor in state_dict.items():
        # Remove NeMo-specific prefixes
        if key.startswith("encoder."):
            new_key = key[8:]  # Remove 'encoder.' prefix
        elif key.startswith("decoder."):
            new_key = key[8:]  # Remove 'decoder.' prefix
        elif key.startswith("preprocessor."):
            # Skip preprocessor weights in main state_dict
            continue
        else:
            new_key = key

        mapped[new_key] = tensor

    return mapped


def _extract_preprocessor_config(nemo_config: dict[str, Any]) -> dict[str, Any]:
    """Extract mel spectrogram preprocessor configuration.

    Args:
        nemo_config: Full NeMo model configuration

    Returns:
        Preprocessor configuration with mel spectrogram parameters
    """
    preprocessor_cfg = nemo_config.get("preprocessor", {})

    # Extract key mel spectrogram parameters
    preprocessor_config = {
        "sample_rate": preprocessor_cfg.get("sample_rate", 16000),
        "n_fft": preprocessor_cfg.get("n_fft", 512),
        "hop_length": preprocessor_cfg.get("hop_length", 160),
        "win_length": preprocessor_cfg.get("win_length", 400),
        "n_mels": preprocessor_cfg.get("features", 80),
        "mel_fmin": preprocessor_cfg.get("mel_fmin", 0.0),
        "mel_fmax": preprocessor_cfg.get("mel_fmax", None),
        "normalize": preprocessor_cfg.get("normalize", "per_feature"),
        "pad_to": preprocessor_cfg.get("pad_to", 16),
    }

    return preprocessor_config


def _save_safetensors(state_dict: dict[str, torch.Tensor], output_path: Path) -> None:
    """Save state_dict in safetensors format if available.

    Args:
        state_dict: State_dict to save
        output_path: Output path (will be modified to .safetensors)
    """
    try:
        from safetensors.torch import save_file

        safetensors_path = output_path.with_suffix(".safetensors")
        save_file(state_dict, safetensors_path)
        print(f"Saved safetensors: {safetensors_path}")
    except ImportError:
        # Fallback to PyTorch format
        torch.save(state_dict, output_path)
        print(f"Saved PyTorch: {output_path}")


def convert_nemo_to_pytorch(
    nemo_path: Path,
    output_path: Path,
    *,
    extract_config_only: bool = False,
    config_output: Path | None = None,
    use_safetensors: bool = True,
) -> dict[str, Any]:
    """Convert NeMo checkpoint to standard PyTorch format.

    Args:
        nemo_path: Path to NeMo .nemo checkpoint file
        output_path: Output path for converted weights (.pt or .safetensors)
        extract_config_only: If True, only extract configuration
        config_output: Output path for extracted configuration (JSON)
        use_safetensors: If True, save in safetensors format when available

    Returns:
        Dictionary containing extracted configuration and metadata

    Raises:
        FileNotFoundError: If required files are not found in archive
        ValueError: If nemo_path doesn't exist or has wrong extension
    """
    if not nemo_path.exists():
        raise FileNotFoundError(f"NeMo file not found: {nemo_path}")

    if nemo_path.suffix != ".nemo":
        raise ValueError(f"Expected .nemo file, got: {nemo_path.suffix}")

    # Create temporary extraction directory
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_dir = Path(temp_dir)

        # Extract archive and load configuration
        print(f"Extracting {nemo_path}...")
        nemo_config = _extract_nemo_archive(nemo_path, extract_dir)

        # Extract preprocessor configuration
        preprocessor_config = _extract_preprocessor_config(nemo_config)

        if extract_config_only:
            # Save only configuration
            config_out = config_output or output_path.with_suffix(".json")

            config_data = {
                "nemo_config": nemo_config,
                "preprocessor_config": preprocessor_config,
                "metadata": {
                    "source_nemo": str(nemo_path),
                    "model_type": nemo_config.get("target", "unknown"),
                },
            }

            with open(config_out, "w") as f:
                json.dump(config_data, f, indent=2)

            print(f"Extracted configuration: {config_out}")
            return config_data

        # Load and map weights
        print("Loading model weights...")
        state_dict = _load_nemo_state_dict(extract_dir)

        print("Mapping weight names...")
        mapped_state_dict = _map_encoder_weights(state_dict)

        # Save converted weights
        print("Converting to PyTorch format...")
        if use_safetensors:
            _save_safetensors(mapped_state_dict, output_path)
        else:
            torch.save(mapped_state_dict, output_path)
            print(f"Saved PyTorch: {output_path}")

        # Save configuration alongside weights
        config_out = output_path.with_suffix(".json")
        config_data = {
            "preprocessor_config": preprocessor_config,
            "model_config": {
                "encoder": nemo_config.get("encoder", {}),
                "decoder": nemo_config.get("decoder", {}),
                "target": nemo_config.get("target", "unknown"),
            },
            "metadata": {
                "source_nemo": str(nemo_path),
                "model_type": nemo_config.get("target", "unknown"),
                "num_parameters": sum(p.numel() for p in mapped_state_dict.values()),
                "converted_keys": len(mapped_state_dict),
            },
        }

        with open(config_out, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"Saved configuration: {config_out}")

        return config_data


def main() -> None:
    """Command-line interface for NeMo to PyTorch conversion."""
    parser = argparse.ArgumentParser(
        description="Convert NeMo checkpoints to standard PyTorch format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert .nemo to .pt
  python nemo_to_pytorch.py model.nemo --output model.pt
  
  # Convert to safetensors format
  python nemo_to_pytorch.py model.nemo --output model.pt --safetensors
  
  # Extract only configuration
  python nemo_to_pytorch.py model.nemo --extract-config --output config.json
        """,
    )

    parser.add_argument(
        "nemo_path",
        type=Path,
        help="Path to NeMo .nemo checkpoint file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for converted weights (.pt) or configuration (.json)",
    )

    parser.add_argument(
        "--extract-config",
        action="store_true",
        help="Extract only configuration (no weights)",
    )

    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Save in PyTorch format instead of safetensors",
    )

    args = parser.parse_args()

    try:
        result = convert_nemo_to_pytorch(
            nemo_path=args.nemo_path,
            output_path=args.output,
            extract_config_only=args.extract_config,
            use_safetensors=not args.no_safetensors,
        )

        if args.extract_config:
            print(f"\n✓ Configuration extracted from {args.nemo_path}")
            print(f"  Model type: {result['metadata']['model_type']}")
            print(f"  Output: {args.output}")
        else:
            print(f"\n✓ Converted {args.nemo_path}")
            print(f"  Model type: {result['metadata']['model_type']}")
            print(f"  Parameters: {result['metadata']['num_parameters']:,}")
            print(f"  Keys: {result['metadata']['converted_keys']}")
            print(f"  Output: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
