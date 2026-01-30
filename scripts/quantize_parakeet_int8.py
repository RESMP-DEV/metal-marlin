#!/usr/bin/env python3
"""
INT8 quantization script for Parakeet TDT-0.6B model to ensure ANE compatibility.

This script quantizes linear layers to INT8 (ANE native format) while keeping
convolutional layers in FP16, since ANE can handle both formats efficiently.
"""

import argparse
import logging
from pathlib import Path

import safetensors.torch
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quantize_tensor_to_int8(
    tensor: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to INT8 format.

    Args:
        tensor: Input tensor to quantize
        per_channel: Whether to use per-channel or per-tensor quantization

    Returns:
        Tuple of (quantized_int8_tensor, scale_tensor)
    """
    if per_channel and tensor.dim() >= 2:
        # Per-channel quantization along the last dimension (output channels)
        scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127.0
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    else:
        # Per-tensor quantization
        scale = tensor.abs().max() / 127.0
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)

    return quantized, scale


def quantize_to_int8(state_dict: dict, per_channel: bool = True) -> dict:
    """
    Quantize linear layers in state_dict to INT8 format.

    Args:
        state_dict: Model state dictionary
        per_channel: Whether to use per-channel quantization for linear layers

    Returns:
        Quantized state dictionary
    """
    quantized_state_dict = {}

    for key, tensor in state_dict.items():
        if tensor.dtype != torch.float16 and tensor.dtype != torch.float32:
            # Keep non-float tensors as-is
            quantized_state_dict[key] = tensor
            continue

        # Check if this is a linear layer weight
        if "weight" in key and any(pattern in key for pattern in ["linear", "fc", "dense"]):
            logger.info(f"Quantizing linear layer: {key} {tensor.shape}")

            # Convert to float32 for quantization computation
            if tensor.dtype == torch.float16:
                tensor = tensor.float()

            # Quantize to INT8
            quantized_weight, scale = quantize_tensor_to_int8(tensor, per_channel=per_channel)

            # Store quantized weights and scales
            quantized_state_dict[key] = quantized_weight
            quantized_state_dict[f"{key}_scale"] = scale.to(torch.float16)

        elif any(pattern in key for pattern in ["conv", "Conv"]):
            # Keep conv layers in FP16 (ANE handles both)
            logger.info(f"Keeping conv layer in FP16: {key}")
            quantized_state_dict[key] = tensor.to(torch.float16)
        else:
            # Keep other layers as-is (biases, embeddings, etc.)
            if tensor.dtype == torch.float32:
                quantized_state_dict[key] = tensor.to(torch.float16)
            else:
                quantized_state_dict[key] = tensor

    return quantized_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Parakeet TDT-0.6B to INT8 for ANE compatibility"
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the original model")
    parser.add_argument(
        "--output-dir", type=str, default="models/parakeet-tdt-0.6b-int8", help="Output directory"
    )
    parser.add_argument(
        "--per-channel", action="store_true", default=True, help="Use per-channel quantization"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for quantization")

    args = parser.parse_args()

    # Setup paths
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from: {model_path}")

    # Load model and tokenizer
    device = torch.device(args.device)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float16).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Quantizing model to INT8...")

    # Get state dict and quantize
    state_dict = model.state_dict()
    quantized_state_dict = quantize_to_int8(state_dict, per_channel=args.per_channel)

    logger.info(f"Saving quantized model to: {output_dir}")

    # Save quantized weights
    safetensors_path = output_dir / "model.safetensors"
    safetensors.torch.save_file(quantized_state_dict, safetensors_path)

    # Save tokenizer and config
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    # Create quantization metadata file
    metadata = {
        "quantization": "int8_linear_fp16_conv",
        "per_channel": args.per_channel,
        "target": "ane_compatibility",
        "original_model": str(model_path),
    }

    import json

    with open(output_dir / "quantization_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Quantization complete! Saved to: {output_dir}")
    logger.info(f"Model files: {list(output_dir.glob('*'))}")


if __name__ == "__main__":
    main()
