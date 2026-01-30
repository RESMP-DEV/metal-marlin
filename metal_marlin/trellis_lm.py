"""TrellisForCausalLM: Language model wrapper for trellis-quantized models.

This module provides a high-level interface for trellis-quantized models
with language modeling head for text generation tasks.

Usage:
    from metal_marlin.trellis_lm import TrellisForCausalLM

    model = TrellisForCausalLM.from_pretrained("model_path")
    logits = model(input_ids)

    # Generate text
    generated = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
"""

from __future__ import annotations

# Re-export from trellis_model for backwards compatibility
from .trellis_model import TrellisForCausalLM, TrellisModel

__all__ = [
    "TrellisForCausalLM",
    "TrellisModel",
]
