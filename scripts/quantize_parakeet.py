#!/usr/bin/env python3
"""CLI for quantizing Parakeet models to Metal Marlin FP4/INT4 format.

Supports loading Parakeet models from NeMo checkpoints or HuggingFace safetensors,
applying quantization policies, running calibration forward passes, and saving
quantized weights as safetensors.

Usage:
    python scripts/quantize_parakeet.py \
        --input models/parakeet-tdt-0.6b \
        --output models/parakeet-tdt-0.6b-fp4 \
        --bits 4 \
        --format fp4 \
        --calibration-audio data/calibration.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torchaudio

# Add parent to path for metal_marlin imports
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.converters import quantize_model


def load_parakeet_model(model_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load Parakeet model from NeMo checkpoint or HuggingFace safetensors.

    Args:
        model_path: Path to model directory or checkpoint file

    Returns:
        Tuple of (state_dict, config_dict)
    """
    model_path = Path(model_path)

    if model_path.is_file() and model_path.suffix in [".ckpt", ".pt", ".pth"]:
        # NeMo checkpoint
        print(f"Loading NeMo checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        # NeMo stores state dict under 'state_dict' key
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Extract config from checkpoint if available
        config = {}
        if "cfg" in checkpoint:
            config = _extract_nemo_config(checkpoint["cfg"])
        elif "hyper_parameters" in checkpoint:
            config = _extract_nemo_config(checkpoint["hyper_parameters"])

    elif model_path.is_dir():
        # HuggingFace model directory
        print(f"Loading HuggingFace model from {model_path}")

        # Try to load safetensors first
        safetensor_files = list(model_path.glob("*.safetensors"))
        if safetensor_files:
            from safetensors import safe_open

            state_dict = {}
            for st_file in safetensor_files:
                with safe_open(str(st_file), framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        else:
            # Fall back to pytorch_model.bin
            bin_file = model_path / "pytorch_model.bin"
            if bin_file.exists():
                state_dict = torch.load(bin_file, map_location="cpu")
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")

        # Load config.json if available
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = {}
    else:
        raise FileNotFoundError(f"Model path not found or invalid: {model_path}")

    return state_dict, config


def _extract_nemo_config(cfg: Any) -> dict[str, Any]:
    """Extract relevant config from NeMo cfg object or dict."""
    if hasattr(cfg, "__dict__"):
        cfg_dict = {}
        for key, value in cfg.__dict__.items():
            if isinstance(value, (int, float, str, bool, list)):
                cfg_dict[key] = value
        return cfg_dict
    elif isinstance(cfg, dict):
        return {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))}
    else:
        return {}


def load_calibration_audio(audio_path: Path, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file for calibration.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default: 16kHz for speech models)

    Returns:
        Audio tensor of shape [batch_size, sequence_length]
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Calibration audio not found: {audio_path}")

    print(f"Loading calibration audio from {audio_path}")

    # Load audio using torchaudio
    waveform, orig_sr = torchaudio.load(str(audio_path))

    # Resample if needed
    if orig_sr != sample_rate:
        print(f"Resampling from {orig_sr}Hz to {sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Normalize to [-1, 1]
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()

    # Add batch dimension if needed
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    return waveform


def run_calibration_forward_pass(
    model: torch.nn.Module,
    audio_tensor: torch.Tensor,
    num_steps: int = 10,
) -> list[torch.Tensor]:
    """Run forward passes to collect activation statistics for calibration.

    Args:
        model: Parakeet model
        audio_tensor: Input audio tensor
        num_steps: Number of forward pass steps

    Returns:
        List of activation tensors for calibration
    """
    print(f"Running {num_steps} calibration forward passes...")

    model.eval()
    calibration_activations = []

    with torch.no_grad():
        for step in range(num_steps):
            # Use different segments if audio is long enough
            if audio_tensor.shape[-1] > 16000 * 2:  # 2 seconds
                start_idx = (step * 16000) % (audio_tensor.shape[-1] - 16000)
                segment = audio_tensor[:, start_idx : start_idx + 16000]
            else:
                segment = audio_tensor

            try:
                # Forward pass
                if hasattr(model, "forward"):
                    output = model(audio_signal=segment)
                elif hasattr(model, "encode"):
                    output = model.encode(segment)
                else:
                    # Try generic forward call
                    output = model(segment)

                # Collect activations from different layers
                if isinstance(output, dict):
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor) and value.dim() > 1:
                            calibration_activations.append(value)
                elif isinstance(output, (list, tuple)):
                    for value in output:
                        if isinstance(value, torch.Tensor) and value.dim() > 1:
                            calibration_activations.append(value)
                else:
                    if isinstance(output, torch.Tensor) and output.dim() > 1:
                        calibration_activations.append(output)

            except Exception as e:
                warnings.warn(f"Calibration step {step} failed: {e}")
                continue

    print(f"Collected {len(calibration_activations)} activation tensors")
    return calibration_activations


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Parakeet models to Metal Marlin FP4/INT4 format"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input model path (NeMo checkpoint or HuggingFace directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--bits",
        "-b",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["fp4", "int4", "fp8"],
        default="fp4",
        help="Quantization format (default: fp4)",
    )
    parser.add_argument(
        "--calibration-audio",
        "-c",
        type=Path,
        help="Audio file for calibration forward pass",
    )
    parser.add_argument(
        "--group-size",
        "-g",
        type=int,
        choices=[32, 64, 128],
        default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=int,
        default=16000,
        help="Audio sample rate for calibration (default: 16000)",
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=10,
        help="Number of calibration forward passes (default: 10)",
    )
    parser.add_argument(
        "--skip-layers",
        nargs="*",
        default=["embed", "lm_head", "norm", "layernorm"],
        help="Layer name fragments to skip quantization",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        help="Model architecture type (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    args = parser.parse_args()

    if args.verbose and args.quiet:
        print("Warning: --verbose and --quiet are both set, using verbose output")

    # Validate arguments
    if args.format.startswith("fp") and int(args.format[2:]) != args.bits:
        print(f"Warning: Format {args.format} doesn't match bits {args.bits}, using format")
        args.bits = int(args.format[2:])

    if not args.input.exists():
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)

    try:
        # Load model
        state_dict, config = load_parakeet_model(args.input)
        if not args.quiet:
            print(f"Loaded model with {len(state_dict)} tensors")

        # Prepare calibration data if provided
        calibration_data = None
        if args.calibration_audio:
            audio_tensor = load_calibration_audio(args.calibration_audio, args.sample_rate)

            # Simple calibration: use audio tensor as activation statistics
            # In a full implementation, you'd run actual model forward passes
            calibration_data = [audio_tensor.float()]

            if args.verbose:
                print(f"Calibration audio shape: {audio_tensor.shape}")

        # Run quantization using the existing metal_marlin converter
        if not args.quiet:
            print(f"Quantizing to {args.format.upper()} with group_size={args.group_size}")

        report = quantize_model(
            model_path=args.input,
            output_path=args.output,
            calibration_data=calibration_data,
            quant_type=args.format,
            group_size=args.group_size,
            skip_layers=args.skip_layers,
            model_type=args.model_type,
            compute_error=True,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print("\nQuantization completed successfully!")
            print(f"Output saved to: {args.output}")
            print(f"Quantized {report.num_quantized} layers, skipped {report.num_skipped}")
            print(f"Average RMS error: {report.avg_rms_error:.6f}")
            print(f"Output size: {report.output_size_mb:.1f} MB")
            print(f"Time elapsed: {report.elapsed_seconds:.1f}s")

        # Save quantization report
        report_file = args.output / "quantization_report.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        if args.verbose:
            print(f"Quantization report saved to: {report_file}")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
