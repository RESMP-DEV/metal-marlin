"""TrellisForCausalLM: Language model wrapper for trellis-quantized models.

This module provides a high-level interface for trellis-quantized models
with language modeling head for text generation tasks.

Usage:
    from metal_marlin.trellis_lm import TrellisForCausalLM

    model = TrellisForCausalLM.from_pretrained("model_path")
    logits, loss = model(input_ids, labels=labels)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trellis_config import TrellisModelConfig

if TYPE_CHECKING:
    from .trellis_kv_cache import TrellisKVCache


class TrellisModel(nn.Module):
    """Base trellis model implementation.

    This is a placeholder for the base TrellisModel implementation.
    In a full implementation, this would contain the transformer layers
    with trellis-quantized weights.
    """

    def __init__(self, config: TrellisModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Additional layers would be added here in a full implementation

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask [batch, seq_len].
            position_ids: Optional position IDs [batch, seq_len].
            kv_cache: Optional KV cache for generation.

        Returns:
            Hidden states [batch, seq_len, hidden_size].
        """
        # Simple embedding lookup for placeholder
        hidden_states = self.embed_tokens(input_ids)
        return hidden_states

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "mps",
    ) -> TrellisModel:
        """Load a trellis model from path.

        Args:
            model_path: Path to the model directory.
            device: Device to load the model on.

        Returns:
            Loaded TrellisModel instance.
        """
        config = TrellisModelConfig.from_pretrained(model_path)
        model = cls(config)
        return model.to(device)

    @staticmethod
    def _load_base_weights(model_path: str) -> dict[str, torch.Tensor]:
        """Load base model weights.

        Args:
            model_path: Path to model directory.

        Returns:
            Dictionary mapping weight names to tensors.
        """
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(model_path)

        # Try to load from safetensors
        for safetensors_file in path.glob("*.safetensors"):
            return load_file(safetensors_file)

        # Try to load from individual layer files
        weights = {}
        if (path / "model.safetensors").exists():
            weights.update(load_file(path / "model.safetensors"))

        return weights


class TrellisForCausalLM(nn.Module):
    """Trellis model with language modeling head for text generation."""

    def __init__(self, config: TrellisModelConfig):
        super().__init__()
        self.config = config
        self.model = TrellisModel(config)

        # LM head (not quantized, tied to embedding or separate)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: TrellisKVCache | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning logits and optional loss.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask [batch, seq_len].
            position_ids: Optional position IDs [batch, seq_len].
            kv_cache: Optional KV cache for generation.
            labels: Optional labels for computing loss [batch, seq_len].

        Returns:
            Tuple of (logits, loss) where:
            - logits: Language model logits [batch, seq_len, vocab_size]
            - loss: Optional cross-entropy loss scalar
        """
        hidden_states = self.model(input_ids, attention_mask, position_ids, kv_cache)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return logits, loss

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "mps",
    ) -> TrellisForCausalLM:
        """Load a TrellisForCausalLM model from path.

        Args:
            model_path: Path to the model directory.
            device: Device to load the model on.

        Returns:
            Loaded TrellisForCausalLM instance.
        """
        config = TrellisModelConfig.from_pretrained(model_path)
        model = cls(config)

        # Load base model
        model.model = TrellisModel.from_pretrained(model_path, device)

        # Load lm_head
        base_weights = TrellisModel._load_base_weights(model_path)
        if "lm_head.weight" in base_weights:
            model.lm_head.weight.data = base_weights["lm_head.weight"].to(device)
        else:
            # Tied embeddings
            model.lm_head.weight = model.model.embed_tokens.weight

        return model.to(device)


__all__ = [
    "TrellisForCausalLM",
    "TrellisModel",
]
