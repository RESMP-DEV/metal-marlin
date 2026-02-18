#!/usr/bin/env python3
"""
Create draft model weights for MMFP4 speculative decoding.

This script initializes a MMFP4MTPHead (Multi-Token Prediction Head)
with standard dimensions and saves the weights to a checkpoint file.
These weights serve as a starting point for fine-tuning or testing
speculative decoding.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path to import metal_marlin modules
# REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# sys.path.append(str(REPO_ROOT))

# Prefer local import to avoid double-loading extension modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from metal_marlin.layers.mmfp4_mtp_head import MMFP4MTPHead
except ImportError:
    # Fallback to repo root if needed
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.append(str(REPO_ROOT))
    from contrib.metal_marlin.metal_marlin.layers.mmfp4_mtp_head import MMFP4MTPHead

def main():
    print("Creating draft model weights for MMFP4...")
    
    # Configuration matches typical MMFP4 models (e.g. Llama-3-8B-Instruct)
    config = {
        "hidden_size": 4096,
        "vocab_size": 128256,  # Llama 3 vocab size
        "num_predictions": 4,  # Speculation depth
        "group_size": 128,     # Quantization group size
    }
    
    print(f"Configuration: {config}")
    
    # Initialize model
    model = MMFP4MTPHead(
        hidden_size=config["hidden_size"],
        vocab_size=config["vocab_size"],
        num_predictions=config["num_predictions"],
        group_size=config["group_size"],
    )
    
    # Initialize weights (standard initialization)
    print("Initializing weights...")
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "mmfp4_draft_weights.pt"
    
    # Save checkpoint
    print(f"Saving weights to {output_path}...")
    torch.save({
        "config": config,
        "state_dict": model.state_dict(),
    }, output_path)
    
    # Verify file creation
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully created draft model weights ({size_mb:.2f} MB)")
        
        # Verify loading
        print("Verifying load...")
        checkpoint = torch.load(output_path, map_location="cpu")
        loaded_model = MMFP4MTPHead(**checkpoint["config"])
        loaded_model.load_state_dict(checkpoint["state_dict"])
        print("✅ Weights loaded successfully")
        
    else:
        print("❌ Failed to create weight file")
        sys.exit(1)

if __name__ == "__main__":
    main()
