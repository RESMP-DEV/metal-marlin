"""Quantize a model using AWQ (Activation-aware Weight Quantization).

This script quantizes all linear layers in a model using AWQ algorithm.
AWQ provides better accuracy than GPTQ for many LLMs while maintaining
fast inference speed.

Usage:
    uv run python scripts/quantize_awq.py \
        --model-path /path/to/model \
        --output-path /path/to/output \
        --activations-path /path/to/activations.npz \
        --group-size 128 \
        --salient-ratio 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantize a model using AWQ (Activation-aware Weight Quantization)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to source model (safetensors file or HuggingFace directory)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--activations-path",
        type=str,
        required=True,
        help="Path to calibration activation statistics (.npz file)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--salient-ratio",
        type=float,
        default=0.01,
        help="Fraction of salient weights to protect (default: 0.01)",
    )
    parser.add_argument(
        "--activation-method",
        type=str,
        default="rms",
        choices=["mean", "max", "rms"],
        help="Method for computing channel importance (default: rms)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Print progress (default: True)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run calibration statistics generation without quantizing",
    )

    return parser.parse_args()


def generate_calibration_activations(
    model_path: str,
    num_samples: int = 100,
    seq_len: int = 128,
) -> dict[str, np.ndarray]:
    """Generate dummy calibration activation statistics.

    In production, this should be replaced with real activation statistics
    collected from running the model on representative data.

    Args:
        model_path: Path to model
        num_samples: Number of calibration samples
        seq_len: Sequence length

    Returns:
        Dictionary mapping layer names to activation tensors
    """
    from safetensors import safe_open

    activations_dict = {}

    # Find safetensors files
    model_path = Path(model_path)
    if model_path.is_file():
        safetensors_files = [model_path]
    else:
        safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    for sf_path in safetensors_files:
        with safe_open(str(sf_path), framework="numpy") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)

                # Only generate activations for weight matrices
                if "weight" in name.lower() and tensor.ndim == 2:
                    in_features = tensor.shape[0]

                    # Generate dummy activations
                    # In production, run the model on real calibration data
                    activations = np.random.randn(num_samples, seq_len, in_features).astype(
                        np.float32
                    )

                    activations_dict[name] = activations

    return activations_dict


def load_calibration_activations(activations_path: str) -> dict[str, np.ndarray]:
    """Load calibration activation statistics from file.

    Args:
        activations_path: Path to .npz file containing activation statistics

    Returns:
        Dictionary mapping layer names to activation tensors
    """
    activations = np.load(activations_path, allow_pickle=True)

    # Convert to dict (npz loads as NpzFile)
    activations_dict = {key: activations[key] for key in activations.files}

    return activations_dict


def save_calibration_activations(activations_dict: dict[str, np.ndarray], output_path: str):
    """Save calibration activation statistics to file.

    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        output_path: Path to save .npz file
    """
    np.savez_compressed(output_path, **activations_dict)
    print(f"  Saved activation statistics to {output_path}")


def main():
    """Main quantization pipeline."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("AWQ Quantization Pipeline")
    print("=" * 70 + "\n")

    # Check if activation statistics exist
    activations_path = Path(args.activations_path)

    if not activations_path.exists():
        print(f"Activation statistics not found at {activations_path}")
        print("Generating dummy activation statistics...")

        activations_dict = generate_calibration_activations(args.model_path)
        save_calibration_activations(activations_dict, str(activations_path))

        print("\n" + "!" * 70)
        print("WARNING: Using dummy activation statistics!")
        print("For production use, collect real activation statistics from")
        print("running the model on representative calibration data.")
        print("!" * 70 + "\n")
    else:
        print(f"Loading activation statistics from {activations_path}")
        activations_dict = load_calibration_activations(str(activations_path))

    if args.dry_run:
        print("Dry run mode: activation statistics generated, skipping quantization")
        return

    # Quantize model
    from metal_marlin import awq_quantize_model

    print("\nQuantizing model with AWQ:")
    print(f"  Model path: {args.model_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  Group size: {args.group_size}")
    print(f"  Salient ratio: {args.salient_ratio}")
    print(f"  Activation method: {args.activation_method}")
    print("\n" + "-" * 70 + "\n")

    stats = awq_quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        activations_path=args.activations_path,
        group_size=args.group_size,
        salient_ratio=args.salient_ratio,
        activation_method=args.activation_method,
        verbose=args.verbose,
    )

    print("\n" + "-" * 70)
    print("Quantization complete!")
    print("-" * 70 + "\n")

    print("Quantization statistics:")
    print(f"  Layers quantized: {stats['quantized_count']}")
    print(f"  Layers skipped: {stats['skipped_count']}")
    print(f"  Original size: {stats['original_bytes'] / (1024**3):.2f} GB")
    print(f"  Quantized size: {stats['quantized_bytes'] / (1024**3):.2f} GB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Quantization type: {stats['quant_type']}")
    print(f"  Group size: {stats['group_size']}")
    print(f"  Salient ratio: {stats['salient_ratio']}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
