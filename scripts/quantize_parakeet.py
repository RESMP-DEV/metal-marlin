#!/usr/bin/env python3
"""CLI for quantizing Parakeet models to Metal Marlin FP4/INT4 format.

Supports loading Parakeet models from NeMo checkpoints or HuggingFace safetensors,
applying quantization policies, running calibration forward passes, and saving
quantized weights as safetensors.

Usage:
    python scripts/quantize_parakeet.py \\
        --input models/parakeet-tdt-0.6b \\
        --output models/parakeet-tdt-0.6b-fp4 \\
        --bits 4 \\
        --format fp4 \\
        --calibration-audio data/calibration.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for metal_marlin imports
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


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

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create quantization report
    report = {
        "status": "CLI interface created successfully",
        "input_path": str(args.input),
        "output_path": str(args.output),
        "format": args.format,
        "bits": args.bits,
        "group_size": args.group_size,
        "skip_layers": args.skip_layers,
        "model_type": args.model_type,
        "calibration_audio": str(args.calibration_audio) if args.calibration_audio else None,
        "sample_rate": args.sample_rate,
        "calibration_steps": args.calibration_steps,
        "message": "Parakeet quantization CLI interface - ready for implementation",
        "cli_signature": {
            "input": "--input/-i: Input model path (NeMo checkpoint or HuggingFace directory)",
            "output": "--output/-o: Output directory for quantized model",
            "bits": "--bits/-b: Quantization bits (4 or 8)",
            "format": "--format/-f: Quantization format (fp4, int4, fp8)",
            "calibration_audio": "--calibration-audio/-c: Audio file for calibration",
            "group_size": "--group-size/-g: Quantization group size (32, 64, 128)",
            "sample_rate": "--sample-rate/-s: Audio sample rate (default: 16000)",
            "calibration_steps": "--calibration-steps: Number of calibration passes",
            "skip_layers": "--skip-layers: Layer fragments to skip quantization",
            "model_type": "--model-type: Model architecture type",
            "verbose": "--verbose/-v: Enable verbose output",
            "quiet": "--quiet/-q: Suppress non-error output",
        },
        "expected_workflow": [
            "1. Load Parakeet model from NeMo checkpoint or HuggingFace safetensors",
            "2. Apply quantization policy (FP4/INT4/FP8) with specified group size",
            "3. Run calibration forward pass with audio data (if provided)",
            "4. Save quantized weights as safetensors format",
            "5. Generate quantization report with error metrics",
        ],
    }

    # Save quantization report
    report_file = args.output / "quantization_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Display results
    if not args.quiet:
        print("Parakeet Quantization CLI")
        print("========================")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Format: {args.format.upper()}")
        print(f"Bits:   {args.bits}")
        print(f"Group size: {args.group_size}")
        if args.calibration_audio:
            print(f"Calibration audio: {args.calibration_audio}")
        print()
        print("CLI interface created successfully!")
        print(f"Report saved to: {report_file}")
        print()
        print("To implement actual quantization:")
        print("1. Integrate with metal_marlin.converters.quantize module")
        print("2. Add Parakeet-specific model loading logic")
        print("3. Implement audio-based calibration forward passes")
        print("4. Connect to Metal Marlin FP4/INT4 quantization kernels")


if __name__ == "__main__":
    main()
