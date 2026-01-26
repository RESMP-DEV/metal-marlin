"""Speculative decoding verification via rejection sampling.

Implements the verification algorithm from Leviathan et al.,
"Fast Inference from Transformers via Speculative Decoding" (ICML 2023).

The key insight: given draft model q(x) and target model p(x), we can verify
K speculated tokens in a single target forward pass. Accepted tokens provably
follow the target distribution exactly, with no approximation.

Algorithm for each position i:
  1. Draw r ~ Uniform(0, 1)
  2. If r < p(x_i) / q(x_i): accept token x_i
  3. Else: reject x_i, sample replacement from norm(max(0, p - q))
  4. Stop accepting at first rejection

After all accepted tokens, sample one bonus token from the target model's
distribution at position (num_accepted + 1). This gives us at least 1 and
at most K+1 tokens per verification step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class VerifyResult:
    """Result of speculative verification.

    Attributes:
        accepted_tokens: [batch, num_spec] token IDs; valid up to num_accepted per row.
        num_accepted: [batch] count of accepted tokens per sequence.
        next_token: [batch] the token sampled after rejection (or bonus token
            if all accepted).
    """

    accepted_tokens: Tensor
    num_accepted: Tensor
    next_token: Tensor


def verify_speculative(
    draft_tokens: Tensor,
    draft_probs: Tensor,
    target_logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> VerifyResult:
    """Verify draft tokens using rejection sampling.

    Args:
        draft_tokens: [batch, num_spec] proposed token IDs from draft model.
        draft_probs: [batch, num_spec, vocab] probability distributions from
            draft model.
        target_logits: [batch, num_spec+1, vocab] raw logits from target model.
            Position i contains logits conditioned on prefix + draft_tokens[:i].
            Position num_spec is the "bonus" position after all draft tokens.
        temperature: Sampling temperature applied to target logits.
            Values <= 0 are treated as greedy (argmax).
        top_p: Nucleus sampling threshold for rejection/bonus sampling.
            1.0 means no filtering.

    Returns:
        VerifyResult with accepted tokens, acceptance counts, and next token.
    """
    batch_size, num_spec = draft_tokens.shape
    device = draft_tokens.device

    # Convert target logits to probabilities
    if temperature <= 0:
        # Greedy: one-hot on argmax
        target_probs = _greedy_probs(target_logits)
    else:
        target_probs = torch.softmax(target_logits / temperature, dim=-1)

    # Per-position acceptance loop (K is small, typically 4-8)
    accepted_list: list[Tensor] = []
    num_accepted = torch.zeros(batch_size, dtype=torch.int32, device=device)
    still_accepting = torch.ones(batch_size, dtype=torch.bool, device=device)
    rejection_pos = torch.full((batch_size,), num_spec, dtype=torch.int32, device=device)

    for i in range(num_spec):
        draft_token = draft_tokens[:, i]  # [batch]
        token_idx = draft_token.reshape(-1, 1)  # [batch, 1]

        # Gather q(x_i) and p(x_i)
        p_draft = torch.gather(draft_probs[:, i, :], dim=1, index=token_idx).squeeze(-1)
        p_target = torch.gather(target_probs[:, i, :], dim=1, index=token_idx).squeeze(-1)

        # Accept if r < min(1, p/q)
        r = torch.rand(batch_size, device=device)
        ratio = p_target / torch.clamp(p_draft, min=1e-10)
        accept = (r < ratio) & still_accepting

        # Record this position's accepted token (0 for rejected)
        accepted_list.append(torch.where(accept, draft_token, torch.zeros_like(draft_token)))
        num_accepted = torch.where(accept, num_accepted + 1, num_accepted)

        # Track first rejection position
        newly_rejected = still_accepting & ~accept
        rejection_pos = torch.where(
            newly_rejected,
            torch.tensor(i, dtype=torch.int32, device=device),
            rejection_pos,
        )
        still_accepting = still_accepting & accept

    accepted_tokens = torch.stack(accepted_list, dim=1)  # [batch, num_spec]

    # Sample next token based on whether all were accepted or not
    next_token = _sample_next_token(
        still_accepting,
        rejection_pos,
        num_spec,
        target_logits,
        target_probs,
        draft_probs,
        temperature,
        top_p,
        batch_size,
        device,
    )

    return VerifyResult(
        accepted_tokens=accepted_tokens,
        num_accepted=num_accepted,
        next_token=next_token,
    )


def _greedy_probs(logits: Tensor) -> Tensor:
    """Convert logits to one-hot probability distributions (greedy).

    Args:
        logits: [batch, seq, vocab] raw logits.

    Returns:
        [batch, seq, vocab] with 1.0 at argmax positions, 0 elsewhere.
    """
    batch, seq, vocab = logits.shape
    argmax_ids = logits.argmax(dim=-1)  # [batch, seq]

    # Build one-hot via scatter
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(2, argmax_ids.unsqueeze(-1), 1.0)
    return one_hot


def _sample_next_token(
    all_accepted: Tensor,
    rejection_pos: Tensor,
    num_spec: int,
    target_logits: Tensor,
    target_probs: Tensor,
    draft_probs: Tensor,
    temperature: float,
    top_p: float,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Sample the next token after verification.

    For fully-accepted sequences: sample from target at the bonus position.
    For rejected sequences: sample from the residual distribution
    norm(max(0, p_target - p_draft)) at the rejection position.
    """
    next_token = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Bonus token for fully-accepted sequences
    if all_accepted.any():
        bonus_logits = target_logits[:, num_spec, :]
        bonus_token = _sample(bonus_logits, temperature, top_p, device)
        next_token = torch.where(all_accepted, bonus_token, next_token)

    # Residual sampling for rejected sequences
    rejected_mask = ~all_accepted
    if rejected_mask.any():
        next_token = _sample_residual_batched(
            rejected_mask,
            rejection_pos,
            target_probs,
            draft_probs,
            target_logits,
            temperature,
            top_p,
            batch_size,
            next_token,
            device,
        )

    return next_token


def _sample_residual_batched(
    rejected_mask: Tensor,
    rejection_pos: Tensor,
    target_probs: Tensor,
    draft_probs: Tensor,
    target_logits: Tensor,
    temperature: float,
    top_p: float,
    batch_size: int,
    next_token: Tensor,
    device: torch.device,
) -> Tensor:
    """Sample from residual distribution for rejected batch elements.

    The residual distribution is norm(max(0, p_target - p_draft)), which
    represents the "bonus" probability mass the target assigns beyond
    what the draft model predicted. Sampling from this ensures the
    combined accept/reject procedure produces exactly the target distribution.
    """
    # Gather target and draft probs at each element's rejection position.
    # rejection_pos is [batch], we need probs at those specific seq positions.
    for b in range(batch_size):
        if not rejected_mask[b].item():
            continue
        pos = int(rejection_pos[b].item())
        p_t = target_probs[b, pos, :]  # [vocab]
        p_d = draft_probs[b, pos, :]  # [vocab]

        # Residual distribution: max(0, p_target - p_draft)
        residual = torch.clamp(p_t - p_d, min=0.0)
        residual_sum = residual.sum()

        if residual_sum.item() < 1e-10:
            # Degenerate case: target and draft agree perfectly.
            # Fall back to sampling from target directly.
            token = _sample(target_logits[b : b + 1, pos, :], temperature, top_p, device)
        else:
            # Sample from normalized residual
            log_residual = torch.log(residual / residual_sum + 1e-10)
            token = torch.multinomial(
                torch.softmax(log_residual, dim=-1).unsqueeze(0), num_samples=1
            )

        token_val = int(token.reshape(()).item())
        next_token[b] = token_val

    return next_token


def _sample(logits: Tensor, temperature: float, top_p: float, device: torch.device) -> Tensor:
    """Sample from logits with temperature and nucleus (top-p) sampling.

    Args:
        logits: [batch, vocab] or [vocab] raw logits.
        temperature: Temperature for softmax. <= 0 means greedy.
        top_p: Nucleus sampling threshold. 1.0 means no filtering.
        device: Device for output tensor.

    Returns:
        [batch] sampled token indices.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if temperature <= 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        probs = _apply_top_p(probs, top_p)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _apply_top_p(probs: Tensor, p: float) -> Tensor:
    """Apply nucleus (top-p) filtering to a probability distribution.

    Zeros out tokens outside the smallest set whose cumulative probability
    exceeds p, then renormalizes.

    Args:
        probs: [batch, vocab] normalized probability distribution.
        p: Cumulative probability threshold.

    Returns:
        [batch, vocab] filtered and renormalized probabilities.
    """
    # Sort descending by probability
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

    # Cumulative sum in sorted order
    cumsum = sorted_probs.cumsum(dim=-1)

    # Keep tokens where cumulative probability hasn't exceeded p yet.
    # Always keep at least the top-1 token.
    mask = torch.cat(
        [
            torch.ones(*probs.shape[:-1], 1, dtype=torch.bool, device=probs.device),
            cumsum[..., :-1] < p,
        ],
        dim=-1,
    )

    # Zero out tokens outside nucleus in sorted space
    filtered_sorted = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))

    # Unsort: scatter back to original positions
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)

    # Renormalize
    total = filtered.sum(dim=-1, keepdim=True)
    return filtered / torch.clamp(total, min=1e-10)
