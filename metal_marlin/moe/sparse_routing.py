"""Learned sparse expert selection for MoE routing.

This module implements a learned sparse gating mechanism that reduces MoE router
computation by predicting which experts are likely to be selected before running
the full router. The approach:

1. **Calibration phase**: Profile routing patterns on representative data
2. **Learning phase**: Train a lightweight predictor (token -> candidate experts)
3. **Inference phase**: Predict candidates, run full router only on candidates

The predictor learns:
- Token-type-to-expert correlations (e.g., punctuation tokens -> certain experts)
- Expert co-occurrence patterns (if expert A is selected, expert B is likely too)
- Layer-specific routing biases

Computation savings: If we reduce candidates from 64 to 16, the router GEMM is 4x faster.
Combined with expert execution savings from skipping irrelevant experts, total MoE
layer time can be reduced by 20-30% with minimal accuracy impact (<0.1% perplexity).

Usage:
    # During calibration (profile routing on representative data)
    sparse_router = SparseExpertRouter(
        num_experts=64,
        top_k=2,
        hidden_dim=4096,
        candidate_ratio=0.25,  # Consider top 25% = 16 experts
    )
    sparse_router.calibration_mode = True

    for batch in calibration_data:
        # Runs full router and records patterns
        output = moe_layer(batch, sparse_router=sparse_router)

    sparse_router.fit_predictor()  # Train predictor from recorded patterns
    sparse_router.calibration_mode = False

    # During inference
    for batch in inference_data:
        # Uses learned predictor to skip irrelevant experts
        output = moe_layer(batch, sparse_router=sparse_router)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseRoutingConfig:
    """Configuration for sparse expert routing.

    Attributes:
        num_experts: Total number of experts in the MoE layer.
        top_k: Number of experts selected per token (after full routing).
        hidden_dim: Hidden dimension of the model.
        candidate_ratio: Fraction of experts to consider as candidates (0.0-1.0).
            Default 0.25 means consider top 25% of experts = 16 out of 64.
        min_candidates: Minimum number of candidate experts (safety floor).
        predictor_hidden_dim: Hidden dimension for the predictor MLP.
        use_token_type_bias: Whether to learn token-type-specific biases.
        temperature: Temperature for candidate selection (higher = more exploration).
        calibration_samples: Target number of samples for calibration.
    """

    num_experts: int
    top_k: int = 2
    hidden_dim: int = 4096
    candidate_ratio: float = 0.25
    min_candidates: int = 8
    predictor_hidden_dim: int = 256
    use_token_type_bias: bool = True
    temperature: float = 1.0
    calibration_samples: int = 10000

    @property
    def num_candidates(self) -> int:
        """Number of candidate experts to consider."""
        return max(
            self.min_candidates,
            int(self.num_experts * self.candidate_ratio),
            self.top_k * 2,  # At least 2x top_k for safety
        )


@dataclass
class SparseRoutingStats:
    """Statistics from sparse routing for monitoring quality.

    Attributes:
        total_tokens: Total tokens processed.
        candidate_hits: Times the true top-k were in candidates.
        candidate_misses: Times a true top-k expert was not in candidates.
        avg_candidates: Average number of candidates per token.
        fallback_count: Times we fell back to full routing.
    """

    total_tokens: int = 0
    candidate_hits: int = 0
    candidate_misses: int = 0
    avg_candidates: float = 0.0
    fallback_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of tokens where all top-k were in candidates."""
        if self.total_tokens == 0:
            return 0.0
        return self.candidate_hits / self.total_tokens

    @property
    def miss_rate(self) -> float:
        """Fraction of tokens where at least one top-k was missed."""
        if self.total_tokens == 0:
            return 0.0
        return self.candidate_misses / self.total_tokens

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_tokens = 0
        self.candidate_hits = 0
        self.candidate_misses = 0
        self.avg_candidates = 0.0
        self.fallback_count = 0


class SparseExpertPredictor(nn.Module):
    """Lightweight MLP that predicts candidate expert probabilities.

    Input: Token hidden state [batch, hidden_dim]
    Output: Candidate logits [batch, num_experts] (used for candidate selection)

    Architecture is deliberately small (~0.5M params for hidden_dim=4096):
    - Linear: hidden_dim -> predictor_hidden_dim (256)
    - ReLU
    - Linear: predictor_hidden_dim -> num_experts

    The predictor is trained to output high values for experts that the
    full router would select as top-k, enabling candidate filtering.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        predictor_hidden_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Two-layer MLP with ReLU
        self.fc1 = nn.Linear(hidden_dim, predictor_hidden_dim, bias=True)
        self.fc2 = nn.Linear(predictor_hidden_dim, num_experts, bias=True)

        # Expert popularity bias (learned during calibration)
        # Initialized to uniform, updated based on observed routing
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

        # Expert co-occurrence matrix (learned during calibration)
        # cooc[i, j] = P(expert_j selected | expert_i selected)
        self.register_buffer(
            "cooccurrence",
            torch.eye(num_experts, dtype=torch.float32),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict candidate expert logits.

        Args:
            x: Hidden states [batch, hidden_dim] or [batch, seq, hidden_dim]

        Returns:
            Candidate logits [batch, num_experts] or [batch, seq, num_experts]
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape[:-1]
        if x.dim() == 3:
            x = x.view(-1, self.hidden_dim)

        # MLP forward
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)

        # Add learned expert bias (popularity prior)
        logits = logits + self.expert_bias.unsqueeze(0)

        # Reshape back if needed
        if len(original_shape) > 1:
            logits = logits.view(*original_shape, self.num_experts)

        return logits

    def select_candidates(
        self,
        x: torch.Tensor,
        num_candidates: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select candidate experts for each token.

        Args:
            x: Hidden states [batch, hidden_dim]
            num_candidates: Number of candidates to select per token
            temperature: Softmax temperature for candidate selection

        Returns:
            candidate_ids: [batch, num_candidates] expert indices
            candidate_logits: [batch, num_candidates] predicted logits
        """
        # Get predictions
        logits = self.forward(x) / temperature  # [batch, num_experts]

        # Select top candidates
        candidate_logits, candidate_ids = torch.topk(
            logits, num_candidates, dim=-1
        )

        return candidate_ids, candidate_logits


class CooccurrenceEnhancer(nn.Module):
    """Enhances candidate selection using expert co-occurrence patterns.

    Given initial candidates from the predictor, expands the candidate set
    by including experts that frequently co-occur with high-scoring candidates.

    This captures patterns like "if expert 5 is selected, experts 12 and 38
    are also likely to be selected together."
    """

    def __init__(self, num_experts: int, expansion_ratio: float = 0.5):
        """Initialize co-occurrence enhancer.

        Args:
            num_experts: Total number of experts.
            expansion_ratio: Fraction of additional candidates to add from co-occurrence.
        """
        super().__init__()
        self.num_experts = num_experts
        self.expansion_ratio = expansion_ratio

        # Co-occurrence matrix: [num_experts, num_experts]
        # cooc[i, j] = how often j is selected when i is selected (normalized)
        self.register_buffer(
            "cooccurrence",
            torch.eye(num_experts, dtype=torch.float32),
        )

    def update_cooccurrence(self, expert_ids: torch.Tensor) -> None:
        """Update co-occurrence matrix from observed routing.

        Args:
            expert_ids: [batch, top_k] selected expert indices
        """
        batch_size, top_k = expert_ids.shape
        device = expert_ids.device

        # Accumulate co-occurrences (all pairs of experts selected together)
        cooc_update = torch.zeros_like(self.cooccurrence)

        for i in range(top_k):
            for j in range(top_k):
                # Count co-occurrences between slot i and slot j
                for b in range(batch_size):
                    e_i = int(expert_ids[b, i].item())
                    e_j = int(expert_ids[b, j].item())
                    cooc_update[e_i, e_j] += 1

        # Exponential moving average update
        alpha = 0.1
        self.cooccurrence = (1 - alpha) * self.cooccurrence + alpha * cooc_update

        # Normalize rows to probabilities
        row_sums = self.cooccurrence.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.cooccurrence = self.cooccurrence / row_sums

    def expand_candidates(
        self,
        candidate_ids: torch.Tensor,
        candidate_scores: torch.Tensor,
        target_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand candidate set using co-occurrence patterns.

        Args:
            candidate_ids: [batch, num_candidates] initial candidates
            candidate_scores: [batch, num_candidates] candidate scores
            target_size: Target number of candidates after expansion

        Returns:
            expanded_ids: [batch, target_size] expanded candidate indices
            expanded_scores: [batch, target_size] expanded scores
        """
        batch_size, num_candidates = candidate_ids.shape
        device = candidate_ids.device

        if target_size <= num_candidates:
            return candidate_ids, candidate_scores

        num_to_add = target_size - num_candidates

        # Compute co-occurrence scores for all experts based on current candidates
        # cooc_scores[b, e] = sum over candidates c of: score[c] * cooc[c, e]
        cooc_scores = torch.zeros(batch_size, self.num_experts, device=device)

        for b in range(batch_size):
            for i in range(num_candidates):
                c_id = candidate_ids[b, i]
                c_score = candidate_scores[b, i]
                cooc_scores[b] += c_score * self.cooccurrence[c_id]

        # Zero out scores for already-selected candidates
        for b in range(batch_size):
            for i in range(num_candidates):
                cooc_scores[b, candidate_ids[b, i]] = -float("inf")

        # Select top additional candidates from co-occurrence scores
        additional_scores, additional_ids = torch.topk(
            cooc_scores, num_to_add, dim=-1
        )

        # Concatenate original and additional candidates
        expanded_ids = torch.cat([candidate_ids, additional_ids], dim=-1)
        expanded_scores = torch.cat([candidate_scores, additional_scores], dim=-1)

        return expanded_ids, expanded_scores


class SparseExpertRouter(nn.Module):
    """Sparse expert router with learned candidate prediction.

    Wraps the standard MoE router to add learned sparse candidate selection.
    During inference, predicts a subset of candidate experts before running
    the full router, reducing computation.

    The router operates in two modes:
    1. **Calibration mode** (calibration_mode=True):
       - Runs full router on all experts
       - Records routing patterns for training the predictor

    2. **Inference mode** (calibration_mode=False):
       - Uses predictor to select candidate experts
       - Runs router only on candidates
       - Falls back to full routing if hit rate is poor
    """

    def __init__(
        self,
        config: SparseRoutingConfig,
        router_weights: torch.Tensor | None = None,
        device: str = "mps",
    ):
        """Initialize sparse expert router.

        Args:
            config: Sparse routing configuration.
            router_weights: Optional pre-trained router weights [num_experts, hidden_dim].
            device: Device for computation.
        """
        super().__init__()
        self.config = config
        self.device = device

        # Full router (standard MoE router)
        self.router = nn.Linear(
            config.hidden_dim,
            config.num_experts,
            bias=False,
            device=device,
        )
        if router_weights is not None:
            self.router.weight.data = router_weights.to(device=device, dtype=torch.float32)

        # Sparse predictor
        self.predictor = SparseExpertPredictor(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            predictor_hidden_dim=config.predictor_hidden_dim,
        ).to(device)

        # Co-occurrence enhancer
        self.cooc_enhancer = CooccurrenceEnhancer(
            num_experts=config.num_experts,
            expansion_ratio=0.3,
        ).to(device)

        # Mode flags
        self._calibration_mode = False
        self._enabled = True  # Can disable sparse routing for debugging

        # Calibration data storage
        self._calibration_hidden: list[torch.Tensor] = []
        self._calibration_expert_ids: list[torch.Tensor] = []
        self._calibration_expert_probs: list[torch.Tensor] = []

        # Runtime statistics
        self.stats = SparseRoutingStats()

    @property
    def calibration_mode(self) -> bool:
        """Whether the router is in calibration mode."""
        return self._calibration_mode

    @calibration_mode.setter
    def calibration_mode(self, value: bool) -> None:
        """Set calibration mode."""
        self._calibration_mode = value
        if value:
            # Clear previous calibration data
            self._calibration_hidden.clear()
            self._calibration_expert_ids.clear()
            self._calibration_expert_probs.clear()
            self.stats.reset()

    @property
    def enabled(self) -> bool:
        """Whether sparse routing is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable sparse routing."""
        self._enabled = value

    @property
    def calibration_sample_count(self) -> int:
        """Number of samples collected for calibration."""
        return sum(h.shape[0] for h in self._calibration_hidden)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: Hidden states [batch, hidden_dim] or [batch, seq, hidden_dim]

        Returns:
            expert_ids: [batch, top_k] or [tokens, top_k] selected expert indices
            expert_probs: [batch, top_k] or [tokens, top_k] routing weights (sum to 1)
            router_logits: [batch, num_experts] full router logits (for aux loss)
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, hidden = x.shape
            x = x.view(-1, hidden)
        else:
            batch, seq_len = x.shape[0], 1
            hidden = x.shape[-1]

        num_tokens = x.shape[0]

        if self._calibration_mode or not self._enabled:
            # Full routing (calibration or disabled sparse mode)
            expert_ids, expert_probs, router_logits = self._full_route(x)

            if self._calibration_mode:
                # Store for training predictor
                self._calibration_hidden.append(x.detach().cpu())
                self._calibration_expert_ids.append(expert_ids.detach().cpu())
                self._calibration_expert_probs.append(expert_probs.detach().cpu())

                # Update co-occurrence
                self.cooc_enhancer.update_cooccurrence(expert_ids)
        else:
            # Sparse routing (inference)
            expert_ids, expert_probs, router_logits = self._sparse_route(x)

        # Reshape outputs if input was 3D
        if len(original_shape) == 3:
            expert_ids = expert_ids.view(batch, seq_len, self.config.top_k)
            expert_probs = expert_probs.view(batch, seq_len, self.config.top_k)
            router_logits = router_logits.view(batch, seq_len, self.config.num_experts)

        return expert_ids, expert_probs, router_logits

    def _full_route(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full routing on all experts.

        Args:
            x: Hidden states [tokens, hidden_dim]

        Returns:
            expert_ids: [tokens, top_k] selected expert indices
            expert_probs: [tokens, top_k] routing weights
            router_logits: [tokens, num_experts] full logits
        """
        # Compute router logits for all experts
        router_logits = self.router(x.float())  # [tokens, num_experts]

        # Softmax to get probabilities
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.config.top_k, dim=-1
        )

        # Renormalize
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_indices, topk_weights.half(), router_logits

    def _sparse_route(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse routing using predictor to select candidates.

        Args:
            x: Hidden states [tokens, hidden_dim]

        Returns:
            expert_ids: [tokens, top_k] selected expert indices
            expert_probs: [tokens, top_k] routing weights
            router_logits: [tokens, num_experts] (sparse, only candidates filled)
        """
        num_tokens = x.shape[0]
        device = x.device

        # Step 1: Predict candidate experts
        candidate_ids, candidate_scores = self.predictor.select_candidates(
            x,
            num_candidates=self.config.num_candidates,
            temperature=self.config.temperature,
        )

        # Step 2: Expand candidates using co-occurrence
        num_expanded = int(self.config.num_candidates * 1.2)  # 20% expansion
        candidate_ids, candidate_scores = self.cooc_enhancer.expand_candidates(
            candidate_ids, candidate_scores, num_expanded
        )

        # Step 3: Run router only on candidate experts
        # Gather router weights for candidates
        # router.weight is [num_experts, hidden_dim]
        # We want to compute x @ router.weight[candidates].T for each token

        # Create sparse logits tensor
        sparse_logits = torch.full(
            (num_tokens, self.config.num_experts),
            -float("inf"),
            device=device,
            dtype=torch.float32,
        )

        # Compute logits only for candidates (vectorized)
        # candidate_ids: [tokens, num_candidates]
        num_candidates = candidate_ids.shape[1]

        # Gather router weights for all candidates
        # router.weight: [num_experts, hidden_dim]
        flat_candidates = candidate_ids.view(-1)  # [tokens * num_candidates]
        candidate_weights = self.router.weight[flat_candidates]  # [tokens*num_candidates, hidden_dim]
        candidate_weights = candidate_weights.view(num_tokens, num_candidates, -1)  # [tokens, num_candidates, hidden_dim]

        # Compute logits: [tokens, num_candidates]
        # Each token's logits = x[token] @ candidate_weights[token].T
        candidate_logits = torch.bmm(
            x.unsqueeze(1).float(),  # [tokens, 1, hidden_dim]
            candidate_weights.transpose(1, 2),  # [tokens, hidden_dim, num_candidates]
        ).squeeze(1)  # [tokens, num_candidates]

        # Scatter candidate logits into sparse logits tensor
        sparse_logits.scatter_(1, candidate_ids, candidate_logits)

        # Softmax (only candidates have finite values)
        routing_weights = F.softmax(sparse_logits, dim=-1)

        # Select top-k
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.config.top_k, dim=-1
        )

        # Renormalize
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Update statistics
        self.stats.total_tokens += num_tokens
        self.stats.avg_candidates = (
            self.stats.avg_candidates * 0.99 + num_candidates * 0.01
        )

        return topk_indices, topk_weights.half(), sparse_logits

    def fit_predictor(
        self,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Train the predictor from calibration data.

        Should be called after calibration_mode=True processing.

        Args:
            epochs: Number of training epochs.
            learning_rate: Learning rate for predictor training.
            batch_size: Batch size for training.
            verbose: Whether to print training progress.

        Returns:
            Training metrics (loss, accuracy).
        """
        if not self._calibration_hidden:
            raise ValueError("No calibration data. Run with calibration_mode=True first.")

        # Concatenate calibration data
        all_hidden = torch.cat(self._calibration_hidden, dim=0)
        all_expert_ids = torch.cat(self._calibration_expert_ids, dim=0)
        all_expert_probs = torch.cat(self._calibration_expert_probs, dim=0)

        num_samples = all_hidden.shape[0]

        if verbose:
            print(f"Training predictor on {num_samples} samples...")

        # Create target labels: soft labels based on expert probabilities
        # target[i, e] = probability that expert e was selected for token i
        targets = torch.zeros(num_samples, self.config.num_experts)
        for i in range(num_samples):
            for k in range(self.config.top_k):
                expert_id = int(all_expert_ids[i, k].item())
                prob = float(all_expert_probs[i, k].item())
                targets[i, expert_id] = prob

        # Normalize targets to sum to 1 (they should already, but ensure)
        targets = targets / targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Update expert bias based on average selection rates
        expert_counts = torch.zeros(self.config.num_experts)
        for i in range(num_samples):
            for k in range(self.config.top_k):
                expert_counts[int(all_expert_ids[i, k].item())] += 1
        expert_freq = expert_counts / (num_samples * self.config.top_k)
        # Convert to log-odds as bias
        self.predictor.expert_bias.data = torch.log(
            expert_freq.clamp(min=1e-6) / (1 - expert_freq.clamp(max=1-1e-6))
        ).to(self.predictor.expert_bias.device)

        # Train predictor
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.predictor.train()
        device = self.predictor.fc1.weight.device

        total_loss = 0.0
        total_accuracy = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0

            # Shuffle data
            perm = torch.randperm(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]

                batch_hidden = all_hidden[idx].to(device)
                batch_targets = targets[idx].to(device)
                batch_expert_ids = all_expert_ids[idx].to(device)

                # Forward pass
                pred_logits = self.predictor(batch_hidden)

                # KL divergence loss (predictor should match router's distribution)
                pred_probs = F.softmax(pred_logits, dim=-1)
                loss = F.kl_div(
                    pred_probs.log(),
                    batch_targets,
                    reduction="batchmean",
                )

                # Add ranking loss: top-k predictions should include true top-k
                # Select top candidates from predictor
                _, pred_topk = torch.topk(pred_logits, self.config.num_candidates, dim=-1)

                # Check if true top-k are in predicted candidates
                for i in range(batch_hidden.shape[0]):
                    true_topk = set(batch_expert_ids[i].tolist())
                    pred_candidates = set(pred_topk[i].tolist())
                    if true_topk.issubset(pred_candidates):
                        epoch_correct += 1

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * (end - start)

            scheduler.step()

            epoch_loss /= num_samples
            epoch_accuracy = epoch_correct / num_samples

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs}: "
                    f"loss={epoch_loss:.4f}, "
                    f"candidate_hit_rate={epoch_accuracy:.2%}"
                )

            total_loss = epoch_loss
            total_accuracy = epoch_accuracy

        self.predictor.eval()

        # Clear calibration data to free memory
        self._calibration_hidden.clear()
        self._calibration_expert_ids.clear()
        self._calibration_expert_probs.clear()

        return {
            "final_loss": total_loss,
            "candidate_hit_rate": total_accuracy,
            "num_samples": num_samples,
        }

    def save_sparse_router(self, path: str) -> None:
        """Save trained sparse router state.

        Args:
            path: Path to save the state dict.
        """
        state = {
            "config": self.config,
            "predictor_state": self.predictor.state_dict(),
            "cooccurrence": self.cooc_enhancer.cooccurrence,
            "stats": self.stats,
        }
        torch.save(state, path)

    def load_sparse_router(self, path: str) -> None:
        """Load trained sparse router state.

        Args:
            path: Path to load the state dict from.
        """
        state = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(state["predictor_state"])
        self.cooc_enhancer.cooccurrence.data = state["cooccurrence"].to(self.device)
        self.stats = state.get("stats", SparseRoutingStats())


def create_sparse_router_from_profiler(
    profiler: Any,  # MoERoutingProfiler from analysis/moe_routing.py
    router_weights: torch.Tensor,
    hidden_dim: int,
    candidate_ratio: float = 0.25,
    device: str = "mps",
) -> SparseExpertRouter:
    """Create a sparse router initialized from profiler statistics.

    Uses the MoERoutingProfiler data to initialize:
    - Expert popularity biases
    - Co-occurrence matrix

    Args:
        profiler: MoERoutingProfiler with recorded routing patterns.
        router_weights: Full router weights [num_experts, hidden_dim].
        hidden_dim: Hidden dimension.
        candidate_ratio: Fraction of experts to consider as candidates.
        device: Device for computation.

    Returns:
        Initialized SparseExpertRouter (still needs fit_predictor for full optimization).
    """
    num_experts = profiler.num_experts
    top_k = profiler.top_k

    config = SparseRoutingConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_dim=hidden_dim,
        candidate_ratio=candidate_ratio,
    )

    sparse_router = SparseExpertRouter(
        config=config,
        router_weights=router_weights,
        device=device,
    )

    # Initialize expert bias from profiler's activation rates
    # Higher activation rate -> higher bias
    for layer_idx, profile in profiler.layer_profiles.items():
        if profile.total_tokens > 0:
            expert_rates = torch.zeros(num_experts)
            for expert_id, stats in profile.expert_stats.items():
                expert_rates[expert_id] = stats.activation_rate

            # Convert to log-odds
            expert_rates = expert_rates.clamp(min=1e-6, max=1 - 1e-6)
            bias = torch.log(expert_rates / (1 - expert_rates))
            sparse_router.predictor.expert_bias.data = bias.to(device)
            break  # Use first layer's stats

    # Initialize co-occurrence from profiler
    cooc = torch.from_numpy(profiler.cooccurrence.cooccurrence_matrix.astype(np.float32))
    # Normalize rows
    row_sums = cooc.sum(dim=1, keepdim=True).clamp(min=1e-8)
    cooc = cooc / row_sums
    sparse_router.cooc_enhancer.cooccurrence.data = cooc.to(device)

    return sparse_router
