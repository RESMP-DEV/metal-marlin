"""Loader for quantized Parakeet ASR models.

This module provides functionality to load quantized Parakeet models
configured with Conformer encoder and TDT (Transducer Dynamic Temperature) decoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .conformer_config import ConformerConfig
from .tdt_config import TDTConfig


class ParakeetTDT(torch.nn.Module):
    """Parakeet ASR model with Conformer encoder and TDT decoder.

    This is a simplified version for loading quantized models. The full
    model architecture should be defined elsewhere.
    """

    def __init__(self, conformer_cfg: ConformerConfig, tdt_cfg: TDTConfig):
        super().__init__()
        self.conformer_cfg = conformer_cfg
        self.tdt_cfg = tdt_cfg

        # TODO: Implement actual model architecture
        # This is a placeholder for the model structure
        self.encoder = torch.nn.Linear(conformer_cfg.hidden_size, conformer_cfg.hidden_size)
        self.decoder = torch.nn.Linear(tdt_cfg.encoder_hidden_size, tdt_cfg.vocab_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Placeholder implementation
        encoded = self.encoder(audio_features)
        decoded = self.decoder(encoded)
        return decoded


def load_quantized_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    """Load quantized state dictionary from model path.

    Args:
        model_path: Path to the model directory containing quantized weights

    Returns:
        Dictionary mapping parameter names to quantized tensors

    Raises:
        FileNotFoundError: If model files are not found
    """
    model_path = Path(model_path)

    # Load quantized weights from safetensors
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        weights_path = model_path / "model.safetensors"  # Try exact name

    if not weights_path.exists():
        raise FileNotFoundError(f"No model.safetensors found in {model_path}")

    state_dict = load_file(str(weights_path))

    # Load configuration if available
    config_path = model_path / "quantization_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        # Config could be used for loading parameters
        _ = config  # Use config if needed

    return state_dict


class ParakeetQuantizedLoader:
    """Loader for quantized Parakeet ASR models."""

    def __init__(self, model_path: Path):
        """Initialize the loader with model path.

        Args:
            model_path: Path to the quantized model directory
        """
        self.model_path = Path(model_path)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load model configuration.

        Returns:
            Configuration dictionary with encoder and decoder settings
        """
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            # Try alternative config names
            for name in ["model_config.json", "parakeet_config.json"]:
                config_path = self.model_path / name
                if config_path.exists():
                    break
            else:
                # Return default configuration if no config found
                return {
                    "encoder": {
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
                    "decoder": {
                        "vocab_size": 1024,
                        "predictor_hidden_size": 320,
                        "predictor_num_layers": 2,
                        "encoder_hidden_size": 512,
                        "joint_hidden_size": 512,
                        "blank_id": 0,
                        "max_duration": 100,
                    },
                }

        return json.loads(config_path.read_text())

    def load(self, device: str = "mps") -> ParakeetTDT:
        """Load quantized model ready for inference.

        Args:
            device: Device to load the model on ('mps', 'cpu', 'cuda')

        Returns:
            Loaded ParakeetTDT model with quantized weights
        """
        # Create model configurations
        conformer_cfg = ConformerConfig(**self.config["encoder"])
        tdt_cfg = TDTConfig(**self.config["decoder"])

        # Initialize model
        model = ParakeetTDT(conformer_cfg, tdt_cfg)

        # Load quantized weights
        state_dict = load_quantized_state_dict(self.model_path)
        model.load_state_dict(state_dict, strict=False)

        # Replace linear layers with quantized versions
        # Conv layers remain FP16
        # TODO: Implement layer replacement logic

        # Move model to device
        model = model.to(device)

        # Set model to evaluation mode
        model.eval()

        return model
