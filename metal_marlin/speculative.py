"""Speculative decoding: fast autoregressive inference with draft-then-verify.

This module provides a high-level interface to the speculative decoding engine.
For advanced usage (adaptive depths, EAGLE, n-gram), use the classes in
`metal_marlin.speculative`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._compat import require_torch, torch
from .speculative.draft import SmallModelDraft
from .speculative.engine import (
    GenerationStats,
    SpeculativeConfig,
    SpeculativeEngine,
)

if TYPE_CHECKING:
    from .kv_cache import KVCache


# Alias for backward compatibility
SpeculativeStats = GenerationStats


def generate_speculative(
    target_model: Any,
    draft_model: Any,
    input_ids: torch.Tensor,
    config: SpeculativeConfig | None = None,
    target_cache: KVCache | None = None,
    draft_cache: Any | None = None,
    streamer: Callable[[int], None] | None = None,
) -> tuple[torch.Tensor, SpeculativeStats]:
    """Generate tokens using speculative decoding with draft-then-verify.

    The draft model proposes K tokens (default 4) autoregressively.
    The target model verifies all K proposals in a single forward pass.
    Accepted tokens skip target decode steps; rejected tokens are resampled
    from the target's residual distribution.

    Args:
        target_model: Main (expensive) language model to sample from.
        draft_model: Small (cheap) draft model for proposing tokens.
        input_ids: Prompt token IDs [1, seq_len].
        config: Speculative decoding configuration.
        target_cache: Optional pre-created target KV cache.
        draft_cache: Optional pre-created draft KV cache (managed internally).
        streamer: Optional callback invoked with each generated token.

    Returns:
        Tuple of:
            output_ids: Full sequence [1, total_len] (prompt + generated)
            stats: Generation statistics (acceptance rate, speedup metrics)
    """
    require_torch()

    if config is None:
        config = SpeculativeConfig(draft_type="small_model")

    # Ensure draft type matches usage
    if config.draft_type != "small_model":
        config.draft_type = "small_model"

    device = input_ids.device

    # Wrap the draft model logic
    # Note: SpeculativeEngine handles the draft cache internally via the DraftModel wrapper
    draft_worker = SmallModelDraft(
        draft_model,
        max_speculative=config.num_speculative_tokens,
        device=device
    )

    engine = SpeculativeEngine(
        target_model=target_model,
        draft_model=draft_worker,
        config=config,
        device=device,
    )

    # Run generation
    # We use a large max_tokens limit since the loop is typically EOS-controlled
    output_ids = engine.generate_all(
        input_ids,
        max_tokens=4096,
        target_cache=target_cache,
        streamer=streamer
    )

    return output_ids, engine.stats


__all__ = [
    "SpeculativeConfig",
    "SpeculativeStats",
    "generate_speculative",
]
