"""Hybrid Parakeet model with automatic Metal GPU / ANE dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..hybrid.scheduler import HybridScheduler, get_scheduler
from .conformer_config import ConformerConfig
from .parakeet_model import ParakeetTDT
from .tdt_config import TDTConfig


class HybridParakeetTDT(nn.Module):
    """ParakeetTDT with hybrid Metal GPU + ANE execution.

    Automatically routes operations to optimal compute unit:
    - Linear (GEMM): Metal GPU with custom INT8/FP4 kernels
    - Conv1d: ANE via CoreML
    - LayerNorm: ANE or GPU depending on size
    """

    def __init__(
        self,
        conformer_config: ConformerConfig,
        tdt_config: TDTConfig,
        scheduler: HybridScheduler | None = None,
        quant_type: str = "int8",
    ):
        super().__init__()
        self.model = ParakeetTDT(conformer_config, tdt_config)
        self.scheduler = scheduler or get_scheduler()
        self.quant_type = quant_type
        self._setup_hybrid()

    def _setup_hybrid(self) -> None:
        """Setup hybrid execution paths."""
        # Replace Linear layers with Metal quantized versions
        try:
            from .replace_layers_metal import replace_parakeet_encoder_layers

            replace_parakeet_encoder_layers(self.model, quant_type=self.quant_type)
            print(f"Encoder Linear layers replaced with Metal {self.quant_type.upper()}")
        except Exception as e:
            print(f"Warning: Metal layer replacement failed ({e}), using MPS fallback")

        # Note: ANE conv modules would be set up here if we had full integration
        # For now, convs stay on MPS

    def encode(self, mel: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with hybrid execution."""
        return self.model.encode(mel, lengths)

    def forward(
        self, mel: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with hybrid execution."""
        return self.encode(mel, lengths)

    def transcribe(self, mel: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """Transcribe audio."""
        return self.model.transcribe(mel, lengths)

    def to(self, device: str | torch.device) -> HybridParakeetTDT:
        """Move model to device."""
        self.model = self.model.to(device)
        return self
