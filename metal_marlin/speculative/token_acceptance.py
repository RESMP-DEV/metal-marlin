"""Draft token acceptance utilities for speculative decoding.

This module provides utilities for tracking and managing accepted tokens
during the speculative decoding process. It handles:

1. Token acceptance tracking: Counting accepted vs rejected tokens
2. Acceptance statistics: Per-step and cumulative acceptance metrics
3. Sequence assembly: Building output sequences from accepted tokens
4. Draft synchronization: Preparing accepted tokens for draft model updates

The token acceptance logic works with the verification results from
verify.py to manage the flow of accepted tokens through the generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class AcceptanceResult:
    """Result of a token acceptance step.

    Attributes:
        accepted_tokens: [batch, num_accepted] accepted token IDs.
        num_accepted: [batch] count of accepted tokens per sequence.
        next_token: [batch] the token sampled after rejection (or bonus token).
        total_new_tokens: [batch] total new tokens (accepted + next_token).
        accepted_mask: [batch, num_spec] boolean mask of accepted positions.
        rejected_positions: [batch] first rejected position per sequence (num_spec if all accepted).
    """

    accepted_tokens: Tensor
    num_accepted: Tensor
    next_token: Tensor
    total_new_tokens: Tensor
    accepted_mask: Tensor
    rejected_positions: Tensor


@dataclass
class AcceptanceStats:
    """Statistics for token acceptance tracking.

    Attributes:
        total_accepted: Total number of accepted draft tokens.
        total_rejected: Total number of rejected draft tokens.
        total_bonus_tokens: Total number of bonus tokens sampled.
        total_proposed: Total number of draft tokens proposed.
        step_count: Number of acceptance steps performed.
        acceptance_history: Per-step acceptance rates.
    """

    total_accepted: int = 0
    total_rejected: int = 0
    total_bonus_tokens: int = 0
    total_proposed: int = 0
    step_count: int = 0
    acceptance_history: list[float] = field(default_factory=list)

    @property
    def overall_acceptance_rate(self) -> float:
        """Fraction of all proposed tokens that were accepted."""
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def average_acceptance_per_step(self) -> float:
        """Average number of tokens accepted per step."""
        if self.step_count == 0:
            return 0.0
        return self.total_accepted / self.step_count

    def get_recent_acceptance_rate(self, window: int = 10) -> float:
        """Acceptance rate over the most recent steps."""
        if not self.acceptance_history:
            return 0.0
        recent = self.acceptance_history[-window:]
        return sum(recent) / len(recent)

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_bonus_tokens = 0
        self.total_proposed = 0
        self.step_count = 0
        self.acceptance_history.clear()


class TokenAcceptanceTracker:
    """Tracks and manages token acceptance during speculative decoding.

    This class provides utilities for:
    1. Computing acceptance results from verification output
    2. Assembling sequences from accepted tokens
    3. Tracking acceptance statistics over time
    4. Managing draft model synchronization

    Usage:
        tracker = TokenAcceptanceTracker()

        # After verification
        result = tracker.compute_acceptance(
            draft_tokens, verify_result, num_speculative
        )

        # Update statistics
        tracker.update_stats(result)

        # Assemble output sequence
        output_ids = tracker.assemble_sequence(
            input_ids, result, include_next=True
        )
    """

    def __init__(self, track_history: bool = True, max_history: int = 1000):
        """Initialize the token acceptance tracker.

        Args:
            track_history: Whether to track per-step acceptance history.
            max_history: Maximum number of steps to keep in history.
        """
        self.track_history = track_history
        self.max_history = max_history
        self.stats = AcceptanceStats()

    def compute_acceptance(
        self,
        draft_tokens: Tensor,
        num_accepted: Tensor,
        next_token: Tensor,
        num_speculative: int,
    ) -> AcceptanceResult:
        """Compute acceptance result from verification output.

        Args:
            draft_tokens: [batch, num_spec] draft tokens proposed.
            num_accepted: [batch] count of accepted tokens per sequence.
            next_token: [batch] the token sampled after rejection.
            num_speculative: Number of speculative tokens proposed.

        Returns:
            AcceptanceResult with full acceptance information.
        """
        batch_size = draft_tokens.shape[0]
        device = draft_tokens.device

        # Create accepted mask
        positions = torch.arange(num_speculative, device=device).unsqueeze(0).expand(batch_size, -1)
        accepted_mask = positions < num_accepted.unsqueeze(-1)

        # Extract accepted tokens (zero-padded beyond num_accepted)
        max_accepted = int(num_accepted.max().item())
        if max_accepted > 0:
            accepted_tokens = torch.where(
                accepted_mask,
                draft_tokens,
                torch.zeros_like(draft_tokens),
            )[:, :max_accepted]
        else:
            accepted_tokens = torch.zeros(batch_size, 0, dtype=draft_tokens.dtype, device=device)

        # Find first rejected position
        rejected_positions = torch.where(
            num_accepted < num_speculative,
            num_accepted,
            torch.full_like(num_accepted, num_speculative),
        )

        # Total new tokens = accepted + next_token
        total_new_tokens = num_accepted + 1

        return AcceptanceResult(
            accepted_tokens=accepted_tokens,
            num_accepted=num_accepted,
            next_token=next_token,
            total_new_tokens=total_new_tokens,
            accepted_mask=accepted_mask,
            rejected_positions=rejected_positions,
        )

    def assemble_sequence(
        self,
        input_ids: Tensor,
        acceptance: AcceptanceResult,
        include_next: bool = True,
    ) -> Tensor:
        """Assemble output sequence from input and accepted tokens.

        Args:
            input_ids: [batch, seq_len] input token IDs.
            acceptance: AcceptanceResult from compute_acceptance.
            include_next: Whether to include the next_token in output.

        Returns:
            [batch, seq_len + num_accepted + (1 if include_next else 0)] output sequence.
        """
        batch_size = input_ids.shape[0]

        # Start with input tokens
        output_parts: list[Tensor] = [input_ids]

        # Add accepted tokens for each batch element
        for b in range(batch_size):
            n_acc = int(acceptance.num_accepted[b].item())
            if n_acc > 0:
                accepted = acceptance.accepted_tokens[b, :n_acc]
                output_parts.append(accepted.unsqueeze(0))

        # Add next token if requested
        if include_next:
            output_parts.append(acceptance.next_token.unsqueeze(-1))

        # Concatenate along sequence dimension
        if len(output_parts) == 1:
            return input_ids

        # Handle per-batch assembly
        result_parts: list[Tensor] = []
        for b in range(batch_size):
            parts: list[Tensor] = [input_ids[b:b + 1]]

            n_acc = int(acceptance.num_accepted[b].item())
            if n_acc > 0:
                accepted = acceptance.accepted_tokens[b, :n_acc]
                parts.append(accepted.unsqueeze(0))

            if include_next:
                parts.append(acceptance.next_token[b:b + 1].unsqueeze(-1))

            result_parts.append(torch.cat(parts, dim=1))

        # Pad to same length and stack
        max_len = max(r.shape[1] for r in result_parts)
        padded_results = []
        for r in result_parts:
            if r.shape[1] < max_len:
                padding = torch.zeros(
                    r.shape[0], max_len - r.shape[1], dtype=r.dtype, device=r.device
                )
                r = torch.cat([r, padding], dim=1)
            padded_results.append(r)

        return torch.cat(padded_results, dim=0)

    def update_stats(self, acceptance: AcceptanceResult, num_speculative: int) -> None:
        """Update acceptance statistics with a new result.

        Args:
            acceptance: AcceptanceResult from compute_acceptance.
            num_speculative: Number of speculative tokens proposed.
        """
        batch_size = acceptance.num_accepted.shape[0]

        total_accepted = int(acceptance.num_accepted.sum().item())
        total_proposed = batch_size * num_speculative
        total_rejected = total_proposed - total_accepted

        self.stats.total_accepted += total_accepted
        self.stats.total_rejected += total_rejected
        self.stats.total_bonus_tokens += batch_size
        self.stats.total_proposed += total_proposed
        self.stats.step_count += 1

        if self.track_history:
            step_acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0
            self.stats.acceptance_history.append(step_acceptance_rate)

            # Trim history if needed
            if len(self.stats.acceptance_history) > self.max_history:
                self.stats.acceptance_history = self.stats.acceptance_history[-self.max_history :]

    def get_draft_sync_tokens(self, acceptance: AcceptanceResult) -> list[Tensor]:
        """Get tokens to feed into draft model for synchronization.

        After tokens are accepted, the draft model needs to be updated
to reflect the true sequence state. This method returns the tokens
        that should be fed into the draft model.

        Args:
            acceptance: AcceptanceResult from compute_acceptance.

        Returns:
            List of [1, num_tokens] tensors per batch element.
        """
        sync_tokens: list[Tensor] = []
        batch_size = acceptance.num_accepted.shape[0]

        for b in range(batch_size):
            n_acc = int(acceptance.num_accepted[b].item())
            if n_acc > 0:
                tokens = acceptance.accepted_tokens[b, :n_acc].unsqueeze(0)
                sync_tokens.append(tokens)
            else:
                # No accepted tokens - return empty tensor
                sync_tokens.append(
                    torch.zeros(1, 0, dtype=acceptance.accepted_tokens.dtype, device=acceptance.accepted_tokens.device)
                )

        return sync_tokens

    def compute_adaptive_speculation_length(
        self,
        current_length: int,
        min_length: int = 1,
        max_length: int = 8,
        target_acceptance_rate: float = 0.6,
        adjustment_factor: float = 0.2,
    ) -> int:
        """Compute adjusted speculation length based on recent acceptance.

        Uses recent acceptance rate to adjust the number of speculative tokens.
        High acceptance -> increase speculation. Low acceptance -> decrease.

        Args:
            current_length: Current speculation length.
            min_length: Minimum speculation length.
            max_length: Maximum speculation length.
            target_acceptance_rate: Target acceptance rate to maintain.
            adjustment_factor: How aggressively to adjust (0-1).

        Returns:
            New speculation length.
        """
        if not self.track_history or len(self.stats.acceptance_history) < 3:
            return current_length

        recent_rate = self.stats.get_recent_acceptance_rate(window=5)

        # Compare to target
        if recent_rate > target_acceptance_rate * 1.1:
            # High acceptance - increase speculation
            adjustment = max(1, int(current_length * adjustment_factor))
            new_length = min(max_length, current_length + adjustment)
        elif recent_rate < target_acceptance_rate * 0.9:
            # Low acceptance - decrease speculation
            adjustment = max(1, int(current_length * adjustment_factor))
            new_length = max(min_length, current_length - adjustment)
        else:
            # Within target range - keep current
            new_length = current_length

        return new_length

    def compute_speculation_length_step(
        self,
        current_length: int,
        min_length: int,
        max_length: int,
        decrease_threshold: float,
        increase_threshold: float,
        window: int = 10,
    ) -> int:
        """Compute new speculation length using step adjustments (+/- 1).

        Args:
            current_length: Current speculation length.
            min_length: Minimum speculation length.
            max_length: Maximum speculation length.
            decrease_threshold: Decrease length if rate < this.
            increase_threshold: Increase length if rate > this.
            window: Number of recent steps to average.

        Returns:
            New speculation length.
        """
        if not self.track_history or len(self.stats.acceptance_history) < window:
            return current_length

        recent_rate = self.stats.get_recent_acceptance_rate(window=window)

        if recent_rate < decrease_threshold:
            return max(min_length, current_length - 1)
        elif recent_rate > increase_threshold:
            return min(max_length, current_length + 1)

        return current_length

    def reset(self) -> None:
        """Reset the tracker state."""
        self.stats.reset()

    def get_stats(self) -> AcceptanceStats:
        """Get current acceptance statistics."""
        return self.stats


def compute_acceptance_probabilities(
    draft_probs: Tensor,
    target_probs: Tensor,
    draft_tokens: Tensor,
) -> Tensor:
    """Compute acceptance probabilities for draft tokens.

    The acceptance probability for each token is min(1, p_target / p_draft).
    This is the probability that the token would be accepted in rejection sampling.

    Args:
        draft_probs: [batch, num_spec, vocab] draft probability distributions.
        target_probs: [batch, num_spec, vocab] target probability distributions.
        draft_tokens: [batch, num_spec] draft token IDs.

    Returns:
        [batch, num_spec] acceptance probabilities.
    """
    batch_size, num_spec, vocab = draft_probs.shape
    device = draft_probs.device

    # Gather probabilities for the specific draft tokens
    token_indices = draft_tokens.unsqueeze(-1)  # [batch, num_spec, 1]

    p_draft = torch.gather(draft_probs, dim=-1, index=token_indices).squeeze(-1)
    # p_draft: [batch, num_spec]

    p_target = torch.gather(target_probs, dim=-1, index=token_indices).squeeze(-1)
    # p_target: [batch, num_spec]

    # Acceptance probability = min(1, p_target / p_draft)
    ratio = p_target / torch.clamp(p_draft, min=1e-10)
    acceptance_probs = torch.clamp(ratio, max=1.0)

    return acceptance_probs


def estimate_optimal_speculation_length(
    acceptance_probs: Tensor,
    draft_cost_ratio: float = 0.1,
    overhead: float = 0.05,
) -> int:
    """Estimate optimal speculation length given expected acceptance probabilities.

    The optimal length balances the cost of drafting vs the benefit of acceptance.
    More accepted tokens -> longer speculation is better. Lower acceptance -> shorter.

    Args:
        acceptance_probs: [batch, num_spec] expected acceptance probabilities.
        draft_cost_ratio: Cost of one draft forward vs target forward (e.g., 0.1 = 10%).
        overhead: Fixed overhead per verification step.

    Returns:
        Estimated optimal speculation length.
    """
    # Average across batch
    avg_probs = acceptance_probs.float().mean(dim=0)  # [num_spec]

    # Compute expected tokens per step for different speculation lengths
    best_length = 1
    best_efficiency = 0.0

    for k in range(1, len(avg_probs) + 1):
        # Expected accepted tokens with k speculative tokens
        # E[accepted] = sum_{i=0}^{k-1} prod_{j=0}^{i} p_j
        expected_accepted = 0.0
        cumulative_prob = 1.0

        for i in range(k):
            cumulative_prob *= float(avg_probs[i].item())
            expected_accepted += cumulative_prob

        # Cost = target_cost + k * draft_cost_ratio * target_cost + overhead
        # Efficiency = expected_accepted / cost (normalized by target_cost = 1)
        cost = 1.0 + k * draft_cost_ratio + overhead
        efficiency = expected_accepted / cost

        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_length = k

    return best_length


def create_acceptance_report(tracker: TokenAcceptanceTracker) -> dict:
    """Create a detailed report of acceptance statistics.

    Args:
        tracker: TokenAcceptanceTracker with accumulated statistics.

    Returns:
        Dictionary with acceptance metrics.
    """
    stats = tracker.stats

    report = {
        "total_accepted": stats.total_accepted,
        "total_rejected": stats.total_rejected,
        "total_proposed": stats.total_proposed,
        "total_bonus_tokens": stats.total_bonus_tokens,
        "step_count": stats.step_count,
        "overall_acceptance_rate": stats.overall_acceptance_rate,
        "average_acceptance_per_step": stats.average_acceptance_per_step,
    }

    if stats.acceptance_history:
        recent = stats.get_recent_acceptance_rate(window=10)
        report["recent_acceptance_rate"] = recent

        # Compute acceptance rate trend
        if len(stats.acceptance_history) >= 20:
            early = sum(stats.acceptance_history[:10]) / 10
            late = sum(stats.acceptance_history[-10:]) / 10
            report["acceptance_trend"] = "improving" if late > early * 1.1 else "declining" if late < early * 0.9 else "stable"

    return report
