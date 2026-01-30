#!/usr/bin/env python3
"""Convert nvidia/parakeet-tdt-0.6b-v3 HuggingFace weights to Metal Marlin format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def download_parakeet_v3(output_dir: Path) -> Path:
    """Download Parakeet-TDT 0.6B v3 from HuggingFace.

    Args:
        output_dir: Directory to save downloaded files

    Returns:
        Path to downloaded model directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = "nvidia/parakeet-tdt-0.6b-v3"

    print(f"Downloading {model_id}...")
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir / "parakeet-tdt-0.6b-v3",
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {model_path}")
    return Path(model_path)


def convert_weights(nemo_path: Path, output_path: Path) -> None:
    """Convert NeMo checkpoint to our ParakeetTDT format.

    The Parakeet model uses NeMo format with specific weight naming.

    Args:
        nemo_path: Path to downloaded NeMo model
        output_path: Path to save converted checkpoint
    """
    import nemo.collections.asr as nemo_asr

    print(f"Loading NeMo model from {nemo_path}...")

    # Load NeMo model
    # Note: This requires nemo_toolkit[asr] to be installed
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        str(nemo_path / "parakeet-tdt-0.6b-v3.nemo")
    )

    # Extract state dict
    state_dict = model.state_dict()

    # Map NeMo keys to our format
    # NeMo uses:
    #   encoder.pre_encode.conv.0.conv -> encoder.subsampling.conv.0
    #   encoder.layers.X.self_attn -> encoder.layers.X.self_attn
    #   decoder.prediction.embed -> decoder.predictor.embedding
    #   joint.joint_net -> decoder.joint.joint

    converted_state = {}
    key_mapping = {
        "encoder.pre_encode": "encoder.subsampling",
        "decoder.prediction": "decoder.predictor",
        "joint.joint_net": "decoder.joint",
    }

    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in key_mapping.items():
            if k.startswith(old_prefix):
                new_key = k.replace(old_prefix, new_prefix)
                break
        converted_state[new_key] = v

    # Create config
    config = {
        "conformer": {
            "num_layers": 17,
            "hidden_size": 512,
            "num_attention_heads": 8,
            "ffn_intermediate_size": 2048,
            "conv_kernel_size": 31,
            "dropout": 0.1,
            "n_mels": 80,
            "sample_rate": 16000,
            "subsampling_factor": 4,
        },
        "tdt": {
            "vocab_size": 1024,
            "predictor_hidden_size": 320,
            "predictor_num_layers": 2,
            "encoder_hidden_size": 512,
            "joint_hidden_size": 512,
            "blank_id": 0,
            "max_duration": 100,
        },
    }

    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(converted_state, output_path / "model.pt")

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved converted model to {output_path}")
    print(f"  Weights: {output_path / 'model.pt'}")
    print(f"  Config: {output_path / 'config.json'}")


def main():
    parser = argparse.ArgumentParser(description="Convert Parakeet-TDT v3 weights")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("models/downloads"),
        help="Directory to download HF model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/parakeet-tdt-0.6b-v3"),
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, assume model is already downloaded",
    )
    args = parser.parse_args()

    if not args.skip_download:
        nemo_path = download_parakeet_v3(args.download_dir)
    else:
        nemo_path = args.download_dir / "parakeet-tdt-0.6b-v3"

    convert_weights(nemo_path, args.output_dir)


if __name__ == "__main__":
    main()
