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

import mlx.core as mx


@dataclass
class VerifyResult:
    """Result of speculative verification.

    Attributes:
        accepted_tokens: [batch, num_spec] token IDs; valid up to num_accepted per row.
        num_accepted: [batch] count of accepted tokens per sequence.
        next_token: [batch] the token sampled after rejection (or bonus token
            if all accepted).
    """

    accepted_tokens: mx.array
    num_accepted: mx.array
    next_token: mx.array


def verify_speculative(
    draft_tokens: mx.array,
    draft_probs: mx.array,
    target_logits: mx.array,
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

    # Convert target logits to probabilities
    if temperature <= 0:
        # Greedy: one-hot on argmax
        target_probs = _greedy_probs(target_logits)
    else:
        target_probs = mx.softmax(target_logits / temperature, axis=-1)

    # Per-position acceptance loop (K is small, typically 4-8)
    accepted_list: list[mx.array] = []
    num_accepted = mx.zeros(batch_size, dtype=mx.int32)
    still_accepting = mx.ones(batch_size, dtype=mx.bool_)
    rejection_pos = mx.full((batch_size,), num_spec, dtype=mx.int32)

    for i in range(num_spec):
        draft_token = draft_tokens[:, i]  # [batch]
        token_idx = draft_token.reshape(-1, 1)  # [batch, 1]

        # Gather q(x_i) and p(x_i)
        p_draft = mx.take_along_axis(
            draft_probs[:, i, :], token_idx, axis=1
        ).squeeze(-1)
        p_target = mx.take_along_axis(
            target_probs[:, i, :], token_idx, axis=1
        ).squeeze(-1)

        # Accept if r < min(1, p/q)
        r = mx.random.uniform(shape=(batch_size,))
        ratio = p_target / mx.maximum(p_draft, mx.array(1e-10))
        accept = (r < ratio) & still_accepting

        # Record this position's accepted token (0 for rejected)
        accepted_list.append(mx.where(accept, draft_token, mx.zeros_like(draft_token)))
        num_accepted = mx.where(accept, num_accepted + 1, num_accepted)

        # Track first rejection position
        newly_rejected = still_accepting & ~accept
        rejection_pos = mx.where(newly_rejected, mx.array(i, dtype=mx.int32), rejection_pos)
        still_accepting = still_accepting & accept

    accepted_tokens = mx.stack(accepted_list, axis=1)  # [batch, num_spec]

    # Sample next token based on whether all were accepted or not
    next_token = _sample_next_token(
        still_accepting, rejection_pos, num_spec,
        target_logits, target_probs, draft_probs,
        temperature, top_p, batch_size,
    )

    return VerifyResult(
        accepted_tokens=accepted_tokens,
        num_accepted=num_accepted,
        next_token=next_token,
    )


def _greedy_probs(logits: mx.array) -> mx.array:
    """Convert logits to one-hot probability distributions (greedy).

    Args:
        logits: [batch, seq, vocab] raw logits.

    Returns:
        [batch, seq, vocab] with 1.0 at argmax positions, 0 elsewhere.
    """
    batch, seq, vocab = logits.shape
    argmax_ids = mx.argmax(logits, axis=-1)  # [batch, seq]

    # Build one-hot via comparison with vocab range
    vocab_range = mx.arange(vocab).reshape(1, 1, vocab)  # [1, 1, vocab]
    one_hot = (argmax_ids.reshape(batch, seq, 1) == vocab_range).astype(mx.float32)
    return one_hot


def _sample_next_token(
    all_accepted: mx.array,
    rejection_pos: mx.array,
    num_spec: int,
    target_logits: mx.array,
    target_probs: mx.array,
    draft_probs: mx.array,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> mx.array:
    """Sample the next token after verification.

    For fully-accepted sequences: sample from target at the bonus position.
    For rejected sequences: sample from the residual distribution
    norm(max(0, p_target - p_draft)) at the rejection position.
    """
    next_token = mx.zeros(batch_size, dtype=mx.int32)

    # Bonus token for fully-accepted sequences
    any_all_accepted = mx.any(all_accepted)
    mx.eval(any_all_accepted)
    if any_all_accepted.item():
        bonus_logits = target_logits[:, num_spec, :]
        bonus_token = _sample(bonus_logits, temperature, top_p)
        next_token = mx.where(all_accepted, bonus_token, next_token)

    # Residual sampling for rejected sequences
    rejected_mask = ~all_accepted
    any_rejected = mx.any(rejected_mask)
    mx.eval(any_rejected)
    if any_rejected.item():
        next_token = _sample_residual_batched(
            rejected_mask, rejection_pos, target_probs, draft_probs,
            target_logits, temperature, top_p, batch_size, next_token,
        )

    return next_token


def _sample_residual_batched(
    rejected_mask: mx.array,
    rejection_pos: mx.array,
    target_probs: mx.array,
    draft_probs: mx.array,
    target_logits: mx.array,
    temperature: float,
    top_p: float,
    batch_size: int,
    next_token: mx.array,
) -> mx.array:
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
        p_d = draft_probs[b, pos, :]   # [vocab]

        # Residual distribution: max(0, p_target - p_draft)
        residual = mx.maximum(p_t - p_d, mx.array(0.0))
        residual_sum = mx.sum(residual)
        mx.eval(residual_sum)

        if residual_sum.item() < 1e-10:
            # Degenerate case: target and draft agree perfectly.
            # Fall back to sampling from target directly.
            token = _sample(target_logits[b:b+1, pos, :], temperature, top_p)
        else:
            # Sample from normalized residual
            log_residual = mx.log(residual / residual_sum + 1e-10)
            token = mx.random.categorical(log_residual[None, :])

        mx.eval(token)
        token_val = int(token.reshape(()).item())
        # Update this batch element
        batch_mask = mx.arange(batch_size) == b
        next_token = mx.where(
            batch_mask,
            mx.full((batch_size,), token_val, dtype=next_token.dtype),
            next_token,
        )

    return next_token


def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Sample from logits with temperature and nucleus (top-p) sampling.

    Args:
        logits: [batch, vocab] or [vocab] raw logits.
        temperature: Temperature for softmax. <= 0 means greedy.
        top_p: Nucleus sampling threshold. 1.0 means no filtering.

    Returns:
        [batch] sampled token indices.
    """
    if logits.ndim == 1:
        logits = logits[None, :]

    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    scaled_logits = logits / temperature
    probs = mx.softmax(scaled_logits, axis=-1)

    if top_p < 1.0:
        probs = _apply_top_p(probs, top_p)

    return mx.random.categorical(mx.log(probs + 1e-10))


def _apply_top_p(probs: mx.array, p: float) -> mx.array:
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
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Cumulative sum in sorted order
    cumsum = mx.cumsum(sorted_probs, axis=-1)

    # Keep tokens where cumulative probability hasn't exceeded p yet.
    # Always keep at least the top-1 token.
    mask = mx.concatenate(
        [mx.ones((*probs.shape[:-1], 1), dtype=mx.bool_),
         cumsum[..., :-1] < p],
        axis=-1,
    )

    # Zero out tokens outside nucleus in sorted space
    filtered_sorted = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))

    # Unsort: argsort of the sort indices gives the inverse permutation
    inv_indices = mx.argsort(sorted_indices, axis=-1)
    filtered = mx.take_along_axis(filtered_sorted, inv_indices, axis=-1)

    # Renormalize
    total = mx.sum(filtered, axis=-1, keepdims=True)
    return filtered / mx.maximum(total, mx.array(1e-10))
