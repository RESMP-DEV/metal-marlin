from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from metal_marlin._quantized_weights import _apply_quantized_weights
from metal_marlin.layer_replacement import replace_linear_layers
from metal_marlin.mmfp4_loader import MMFP4ModelLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_prequantized_mmfp4_model(
    model_path: str,
    device: str = "mps",
    bits: int = 4,
) -> tuple[Any, Any]:
    """Load an MMFP4 quantized model with all optimizations enabled.

    This function loads the model architecture WITHOUT weights, replaces
    linear layers with FP4 quantized versions, then loads the packed
    quantized weights from safetensors.

    Args:
        model_path: Path to the model directory
        device: Device to load model on ("mps", "cuda", "cpu")
        bits: Quantization bits (default 4 for FP4)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading MMFP4 model from {model_path}...")
    quantized_path = Path(model_path)

    # Step 1: Load config only (no weights)
    print("  Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Step 2: Create model architecture with random weights (not BF16 pretrained)
    print("  Creating model architecture...")
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Step 3: Replace linear layers with quantized versions (empty, no RTN)
    print("  Replacing linear layers with FP4 quantized versions...")
    replace_linear_layers(model, bits=bits, prequantized=True)

    # Step 4: Load quantized weights from safetensors
    if (quantized_path / "model.safetensors").exists() or (
        quantized_path / "model.safetensors.index.json"
    ).exists():
        print("  Loading quantized weights...")
        loader = MMFP4ModelLoader(quantized_path)
        loaded = _apply_quantized_weights(model, loader, device)
        print(f"  Loaded {loaded} quantized weight tensors")
    else:
        raise ValueError(
            f"No quantized weights found at {quantized_path}. "
            "Expected model.safetensors or model.safetensors.index.json"
        )

    # Move model to device
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    model.eval()
    return model, tokenizer
