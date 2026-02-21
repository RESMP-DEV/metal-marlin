"""Speculative decoding pipeline for MMFP4 models.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..layers.mmfp4_mtp_head import MMFP4MTPHead as GLMMTPHead
else:
    GLMMTPHead = Any


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample a single token from logits for each batch element."""
    if logits.dim() == 3 and logits.shape[1] == 1:
        logits = logits[:, 0, :]
    if logits.dim() != 2:
        raise ValueError(f"Expected logits with shape [batch, vocab], got {tuple(logits.shape)}")

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / max(float(temperature), 1e-8)
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _split_draft_logits(
    draft_logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    num_draft: int,
) -> list[torch.Tensor]:
    """Normalize MTP outputs into per-step [batch, vocab] logits."""
    if isinstance(draft_logits, torch.Tensor):
        if draft_logits.dim() == 3:
            steps = min(num_draft, draft_logits.shape[1])
            return [draft_logits[:, i, :] for i in range(steps)]
        if draft_logits.dim() == 4 and draft_logits.shape[2] == 1:
            steps = min(num_draft, draft_logits.shape[1])
            return [draft_logits[:, i, 0, :] for i in range(steps)]
        raise ValueError(
            f"Unsupported tensor draft logits shape: {tuple(draft_logits.shape)}"
        )

    logits_steps: list[torch.Tensor] = []
    for step_logits in list(draft_logits)[:num_draft]:
        if step_logits.dim() == 3 and step_logits.shape[1] == 1:
            logits_steps.append(step_logits[:, 0, :])
        elif step_logits.dim() == 2:
            logits_steps.append(step_logits)
        else:
            raise ValueError(
                f"Unsupported per-step draft logits shape: {tuple(step_logits.shape)}"
            )
    return logits_steps


def speculative_decode(
    model,
    mtp_head: GLMMTPHead,
    input_ids: torch.Tensor,
    num_draft: int = 4,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """
    Speculative decoding with MTP head.

    Returns:
        (new_tokens, num_accepted) - generated tokens and acceptance count
    """
    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids shape [batch, seq], got {tuple(input_ids.shape)}")
    if input_ids.shape[0] != 1:
        raise NotImplementedError("speculative_decode currently supports batch size 1")
    if num_draft <= 0:
        return input_ids.new_empty((input_ids.shape[0], 0)), 0

    # 1. Get hidden state from model
    with torch.no_grad():
        hidden = model.get_hidden(input_ids[:, -1:])  # Last token

    # 2. Draft N tokens using MTP head
    with torch.no_grad():
        raw_draft_logits = mtp_head(hidden[:, -1:, :])
    draft_logits = _split_draft_logits(raw_draft_logits, num_draft)
    if not draft_logits:
        return input_ids.new_empty((input_ids.shape[0], 0)), 0

    draft_tokens = []
    for logits in draft_logits:
        token = _sample(logits, temperature)
        draft_tokens.append(token)

    # 3. Verify with main model (single forward)
    draft_seq = torch.cat(draft_tokens, dim=1)
    full_input = torch.cat([input_ids, draft_seq], dim=1)
    with torch.no_grad():
        full_output = model(full_input)
    full_logits = full_output.logits if hasattr(full_output, "logits") else full_output
    if full_logits.dim() != 3:
        raise ValueError(f"Expected full logits [batch, seq, vocab], got {tuple(full_logits.shape)}")

    # 4. Verify tokens (compare with actual model predictions)
    accepted = 0
    for i, draft_token in enumerate(draft_tokens):
        target_pos = input_ids.shape[1] + i
        if target_pos >= full_logits.shape[1]:
            break
        target_logit = full_logits[:, target_pos, :]
        target_token = _sample(target_logit, temperature)

        if torch.equal(target_token, draft_token):
            accepted += 1
        else:
            # Replace with correct token and stop
            draft_tokens[i] = target_token
            break

    # Return accepted tokens (including first corrected one)
    new_tokens = torch.cat(draft_tokens[: accepted + 1], dim=1)
    return new_tokens, accepted


class MMFP4SpeculativeDecoder:
    """Implements speculative decoding with a verification kernel."""

    def __init__(self, model):
        self.model = model
        self.verify_kernel = None # Placeholder for the Metal kernel

    def _load_kernels(self):
        """Load and compile the Metal verification kernel."""
        # TODO: Implement kernel loading
        pass

    def verify(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[int, bool]:
        """Verify draft tokens against target model logits."""
        
        # Python implementation from MMFP4Pipeline._draft_verify
        if torch.isnan(target_logits).any():
            return 0, False

        if draft_tokens.numel() == 0:
            return 0, False

        batch_size, seq_len = draft_tokens.shape
        if batch_size != 1:
            raise NotImplementedError("Batch size > 1 not supported for draft verification")
        if seq_len == 0:
            return 0, False

        if temperature > 0 and temperature != 1.0:
            target_logits = target_logits / temperature

        target_probs = torch.softmax(target_logits, dim=-1)

        if temperature <= 0:
            target_tokens = torch.argmax(target_probs, dim=-1)
            matches = (draft_tokens == target_tokens).squeeze(0)
        else:
            draft_token_probs = torch.gather(
                target_probs, -1, draft_tokens.unsqueeze(-1)
            ).squeeze(-1).squeeze(0)
            random_thresholds = torch.rand_like(draft_token_probs)
            matches = draft_token_probs > random_thresholds

        cumulative_matches = matches.to(torch.int).cumprod(dim=0)
        num_accepted = int(cumulative_matches.sum())

        all_accepted = num_accepted == seq_len
        return num_accepted, all_accepted

__all__ = ["MMFP4SpeculativeDecoder", "speculative_decode"]
