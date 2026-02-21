"""GLM 4.7 Flash Multi-Token Prediction head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GLMMTPHead(nn.Module):
    """MTP head that predicts N future tokens in parallel."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # One head per future token.
        self.heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_tokens)
            ]
        )

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            hidden: [batch, seq, hidden_size] - last hidden state
        Returns:
            List of [batch, seq, vocab_size] logits for each future token
        """
        return [head(hidden) for head in self.heads]

    @classmethod
    def from_model(cls, model: Any) -> "GLMMTPHead | None":
        """Extract MTP head from a loaded model if available."""
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None) or getattr(
            model, "hidden_size", None
        )
        vocab_size = getattr(config, "vocab_size", None) or getattr(
            model, "vocab_size", None
        )
        if hidden_size is None or vocab_size is None:
            return None

        num_tokens_hint = (
            getattr(config, "num_nextn_predict_layers", None)
            or getattr(config, "num_mtp_heads", None)
            or 0
        )

        candidates: list[nn.Module] = []

        def _collect(obj: Any) -> None:
            if obj is None:
                return

            if isinstance(obj, nn.Linear):
                candidates.append(obj)
                return

            if isinstance(obj, nn.ModuleList | list | tuple):
                for item in obj:
                    _collect(item)
                return

            if not isinstance(obj, nn.Module):
                return

            # Common MTP containers.
            if hasattr(obj, "heads"):
                _collect(getattr(obj, "heads"))
            if hasattr(obj, "head"):
                _collect(getattr(obj, "head"))

        # Common top-level paths across model variants.
        for root in (model, getattr(model, "model", None)):
            if root is None:
                continue
            for name in (
                "mtp_head",
                "multi_token_head",
                "draft_head",
                "auxiliary_head",
                "nextn_predict_layers",
                "nextn_layers",
                "shared_head",
            ):
                _collect(getattr(root, name, None))

        # GLM 4.7-style auxiliary head may be attached to trailing layers.
        layers = getattr(getattr(model, "model", None), "layers", None)
        if isinstance(layers, nn.ModuleList | list | tuple):
            for layer in reversed(layers):
                shared_head = getattr(layer, "shared_head", None)
                if shared_head is not None:
                    _collect(shared_head)
                if len(candidates) >= 1:
                    # GLM 4.7 Flash currently exposes one next-token head.
                    break

        # Deduplicate while preserving order.
        deduped: list[nn.Module] = []
        seen_ids: set[int] = set()
        for module in candidates:
            module_id = id(module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            deduped.append(module)

        weight_sources: list[torch.Tensor] = []
        for module in deduped:
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight) or weight.ndim != 2 or weight.is_meta:
                continue

            if tuple(weight.shape) == (vocab_size, hidden_size):
                weight_sources.append(weight.detach())
                continue
            if tuple(weight.shape) == (hidden_size, vocab_size):
                weight_sources.append(weight.detach().T)

        if not weight_sources:
            return None

        if isinstance(num_tokens_hint, int) and num_tokens_hint > 0:
            num_tokens = min(num_tokens_hint, len(weight_sources))
        else:
            num_tokens = len(weight_sources)

        if num_tokens <= 0:
            return None

        head = cls(hidden_size, vocab_size, num_tokens=num_tokens)
        with torch.no_grad():
            for idx in range(num_tokens):
                source = weight_sources[idx]
                target = head.heads[idx].weight
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        return head


__all__ = ["GLMMTPHead"]
