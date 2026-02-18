"""Perplexity evaluation for served models.

Perplexity measures how well a language model predicts the next token.
Lower perplexity = better prediction (model is less "surprised").

PPL = exp(cross_entropy_loss) = exp(-1/N * sum(log P(x_i | x_{<i})))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..inference.pipeline_v2 import TransformersMarlinPipeline


@dataclass
class PerplexityResult:
    """Result of perplexity evaluation."""

    perplexity: float
    tokens: int
    loss: float
    # Per-token perplexities for analysis
    token_ppls: list[float] | None = None
    # Optional chunk-based (e.g., per-100-tokens)
    chunk_ppls: list[float] | None = None
    chunk_size: int = 100


@torch.no_grad()
def compute_perplexity(
    pipeline: TransformersMarlinPipeline,
    text: str,
    chunk_size: int = 100,
    include_token_ppls: bool = False,
) -> PerplexityResult:
    """Compute perplexity of text under the model.

    Args:
        pipeline: The model pipeline
        text: Text to evaluate
        chunk_size: Chunk size for computing rolling perplexity
        include_token_ppls: If True, compute per-token perplexities

    Returns:
        PerplexityResult with overall and optional chunk-level perplexities
    """
    # Tokenize
    inputs = pipeline.tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].to(pipeline.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(pipeline.device)

    # Forward pass
    outputs = pipeline.model(
        input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # Shift for next-token prediction
    # Logits[i] predicts token[i+1]
    shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab]
    shift_labels = input_ids[:, 1:].contiguous()  # [1, seq_len-1]

    # Compute per-token cross-entropy loss
    # Flatten for cross_entropy
    vocab_size = shift_logits.size(-1)
    shift_logits_flat = shift_logits.view(-1, vocab_size)  # [seq_len-1, vocab]
    shift_labels_flat = shift_labels.view(-1)  # [seq_len-1]

    # Per-token losses (no reduction)
    token_losses = F.cross_entropy(
        shift_logits_flat,
        shift_labels_flat,
        reduction="none",
    )  # [seq_len-1]

    # Overall perplexity
    num_tokens = token_losses.numel()
    avg_loss = token_losses.mean().item()
    overall_ppl = float(torch.exp(torch.tensor(avg_loss)).item())

    # Per-token perplexities (optional, costly for long sequences)
    token_ppls = None
    if include_token_ppls:
        token_ppls = [float(torch.exp(loss).item()) for loss in token_losses]

    # Chunk-based perplexities
    chunk_ppls = None
    if chunk_size > 0 and num_tokens >= chunk_size:
        chunk_ppls = []
        for i in range(0, num_tokens, chunk_size):
            chunk_loss = token_losses[i: i + chunk_size].mean().item()
            chunk_ppl = float(torch.exp(torch.tensor(chunk_loss)).item())
            chunk_ppls.append(chunk_ppl)

    return PerplexityResult(
        perplexity=overall_ppl,
        tokens=num_tokens,
        loss=avg_loss,
        token_ppls=token_ppls,
        chunk_ppls=chunk_ppls,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def compute_perplexity_from_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> float:
    """Compute perplexity from pre-tokenized input IDs.

    Args:
        model: The language model
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask

    Returns:
        Perplexity as a float
    """
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    logits = outputs.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute loss
    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="mean",
    )

    return float(torch.exp(loss).item())


class PerplexityTracker:
    """Tracks running perplexity statistics during serving.

    Useful for monitoring model quality over time, detecting drift,
    or comparing model versions.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._ppls: list[tuple[float, int, str]] = []  # (ppl, tokens, source)

    def add(self, ppl: float, tokens: int, source: str = "unknown"):
        """Add a perplexity observation."""
        self._ppls.append((ppl, tokens, source))
        if len(self._ppls) > self.window_size:
            self._ppls = self._ppls[-self.window_size:]

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated perplexity statistics."""
        if not self._ppls:
            return {
                "mean_perplexity": None,
                "median_perplexity": None,
                "min_perplexity": None,
                "max_perplexity": None,
                "sample_count": 0,
                "total_tokens": 0,
            }

        ppls = [p[0] for p in self._ppls]
        total_tokens = sum(p[1] for p in self._ppls)

        sorted_ppls = sorted(ppls)
        median_idx = len(sorted_ppls) // 2
        median = (
            (sorted_ppls[median_idx - 1] + sorted_ppls[median_idx]) / 2
            if len(sorted_ppls) % 2 == 0
            else sorted_ppls[median_idx]
        )

        return {
            "mean_perplexity": round(sum(ppls) / len(ppls), 2),
            "median_perplexity": round(median, 2),
            "min_perplexity": round(min(ppls), 2),
            "max_perplexity": round(max(ppls), 2),
            "sample_count": len(self._ppls),
            "total_tokens": total_tokens,
            "recent": self._ppls[-10:] if self._ppls else [],
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        stats = self.get_stats()
        lines = []

        if stats["mean_perplexity"] is not None:
            lines.extend(
                [
                    "# HELP metal_marlin_perplexity_mean Mean perplexity over recent samples",
                    "# TYPE metal_marlin_perplexity_mean gauge",
                    f"metal_marlin_perplexity_mean {stats['mean_perplexity']}",
                    "",
                    "# HELP metal_marlin_perplexity_median Median perplexity over recent samples",
                    "# TYPE metal_marlin_perplexity_median gauge",
                    f"metal_marlin_perplexity_median {stats['median_perplexity']}",
                    "",
                    "# HELP metal_marlin_perplexity_min Minimum perplexity observed",
                    "# TYPE metal_marlin_perplexity_min gauge",
                    f"metal_marlin_perplexity_min {stats['min_perplexity']}",
                    "",
                    "# HELP metal_marlin_perplexity_max Maximum perplexity observed",
                    "# TYPE metal_marlin_perplexity_max gauge",
                    f"metal_marlin_perplexity_max {stats['max_perplexity']}",
                    "",
                    "# HELP metal_marlin_perplexity_samples Number of samples in window",
                    "# TYPE metal_marlin_perplexity_samples gauge",
                    f"metal_marlin_perplexity_samples {stats['sample_count']}",
                    "",
                    "# HELP metal_marlin_perplexity_tokens_total Total tokens evaluated",
                    "# TYPE metal_marlin_perplexity_tokens_total counter",
                    f"metal_marlin_perplexity_tokens_total {stats['total_tokens']}",
                ]
            )

        return "\n".join(lines)
